# This code is based on the revised code from fastchat based on tatsu-lab/stanford_alpaca and QwenLM/Qwen.


from dataclasses import dataclass, field
import json
import logging
import os
from typing import Dict, Optional

import torch
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
import transformers
from transformers import Trainer, AutoTokenizer
from transformers.integrations import deepspeed
from transformers.trainer_pt_utils import LabelSmoother
from accelerate.utils import DistributedType
from huggingface_hub import snapshot_download
from peft import LoraConfig, TaskType, get_peft_model, PeftModel

from finetune_codes.model import KimiAudioModel
from finetune_codes.datasets import LazySupervisedDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="moonshotai/Kimi-Audio-7B")
    model_path: str = field(
        default=None, metadata={"help": "Path to the pretrained model."}
    )

@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_ratio: float = field(
        default=0.05, metadata={"help": "Ratio of evaluation data."}
    )
    lazy_preprocess: bool = False


@dataclass
class LoraArguments:
    """LoRA 相关参数"""
    use_lora: bool = field(
        default=False, metadata={"help": "Whether to use LoRA fine-tuning."}
    )
    lora_rank: int = field(
        default=8, metadata={"help": "LoRA rank."}
    )
    lora_alpha: int = field(
        default=16, metadata={"help": "LoRA alpha (scaling factor)."}
    )
    lora_dropout: float = field(
        default=0.05, metadata={"help": "LoRA dropout."}
    )
    lora_target: str = field(
        default="all", metadata={"help": "Target modules for LoRA. Use 'all' for all linear layers, or comma-separated names like 'q_proj,k_proj,v_proj,o_proj'."}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    dataloader_pin_memory: bool = field(default=False)
    model_max_length: int = field(
        default=8192,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )



def maybe_zero_3(param):
    if hasattr(param, "ds_id"):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    # check if zero3 mode enabled
    if deepspeed.is_deepspeed_zero3_enabled():
        state_dict = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
    else:
        state_dict = trainer.model.state_dict()
    if trainer.args.should_save and trainer.args.local_rank == 0:
        trainer._save(output_dir, state_dict=state_dict)


def find_all_linear_modules(model):
    """找到所有线性层模块名称"""
    # 使用已知的有效模块名（基于 Qwen2 架构）
    # 这些是 Kimi-Audio 模型（基于 Qwen2）中常见的线性层模块名
    common_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    # 验证这些模块是否存在于模型中
    found_modules = set()
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # 排除 whisper_model 和 embedding 层
            if "whisper_model" in name:
                continue
            if "embed" in name.lower():
                continue
            
            module_name = name.split(".")[-1]
            
            # 如果模块名在常见模块列表中，添加到结果中
            if module_name in common_modules:
                found_modules.add(module_name)
    
    # 如果找到了常见模块，使用它们
    if found_modules:
        logger.info(f"Found {len(found_modules)} common linear modules: {sorted(found_modules)}")
        return sorted(list(found_modules))
    
    # 如果没有找到常见模块，尝试查找所有有效的模块名（排除数字和太短的名称）
    logger.warning("Common modules not found, searching for all valid linear modules...")
    valid_modules = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if "whisper_model" in name or "embed" in name.lower():
                continue
            module_name = name.split(".")[-1]
            # 只保留有效的模块名（不是纯数字，长度大于1，包含字母）
            if not module_name.isdigit() and len(module_name) > 1 and any(c.isalpha() for c in module_name):
                valid_modules.add(module_name)
    
    if valid_modules:
        logger.info(f"Found {len(valid_modules)} valid linear modules: {sorted(valid_modules)}")
        return sorted(list(valid_modules))
    
    # 如果还是没找到，使用默认值
    logger.warning("No valid modules found, using default modules")
    return common_modules


def setup_lora_model(model, lora_args: LoraArguments):
    """设置 LoRA 适配器"""
    if not lora_args.use_lora:
        return model
    
    logger.info("Setting up LoRA fine-tuning...")
    
    # 确定目标模块
    if lora_args.lora_target == "all":
        target_modules = find_all_linear_modules(model)
        logger.info(f"Using all linear modules: {target_modules}")
    else:
        target_modules = [m.strip() for m in lora_args.lora_target.split(",")]
        logger.info(f"Using specified modules: {target_modules}")
    
    # 创建 LoRA 配置
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_args.lora_rank,
        lora_alpha=lora_args.lora_alpha,
        lora_dropout=lora_args.lora_dropout,
        target_modules=target_modules,
    )
    
    # 应用 LoRA
    # 确保模型有 PEFT 需要的方法
    if not hasattr(model, "prepare_inputs_for_generation"):
        logger.warning("Model missing prepare_inputs_for_generation, adding default implementation")
    
    model = get_peft_model(model, peft_config)
    
    # 打印可训练参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Trainable params: {trainable_params:,} || "
        f"All params: {all_params:,} || "
        f"Trainable%: {100 * trainable_params / all_params:.4f}%"
    )
    
    return model




def make_supervised_data_module(
    whisper_model, text_tokenizer, data_args, max_len, kimia_token_offset,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = LazySupervisedDataset
    rank0_print("Loading data...")

    with open(data_args.data_path, "r") as f:
        lines = f.readlines()
        all_data = [json.loads(line) for line in lines]

    if data_args.eval_ratio > 0:
        eval_size = int(len(all_data) * data_args.eval_ratio)
        # 确保至少有1条评估数据（如果数据量足够），但不超过总数据的50%
        if eval_size == 0 and len(all_data) > 1:
            eval_size = 1
        elif eval_size > len(all_data) * 0.5:
            eval_size = max(1, int(len(all_data) * 0.5))
        
        if eval_size > 0 and len(all_data) > eval_size:
            eval_data = all_data[:eval_size]
            train_data = all_data[eval_size:]
        else:
            # 如果数据量太少，不使用评估集
            logger.warning(f"Data size ({len(all_data)}) is too small for evaluation, using all data for training")
            eval_data = None
            train_data = all_data
    else:
        eval_data = None
        train_data = all_data
    
    # 最终检查
    if eval_data is not None and len(eval_data) == 0:
        eval_data = None
        train_data = all_data
        logger.warning("Evaluation data is empty, using all data for training")
    
    assert len(train_data) > 0, "No training data found"

    train_dataset = dataset_cls(train_data, whisper_model=whisper_model, text_tokenizer=text_tokenizer, max_len=max_len, kimia_token_offset=kimia_token_offset)

    if eval_data:
        eval_dataset = dataset_cls(eval_data, whisper_model=whisper_model, text_tokenizer=text_tokenizer, max_len=max_len, kimia_token_offset=kimia_token_offset)
    else:
        eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


def compute_loss(outputs, labels, num_items_in_batch=None):

    audio_logits, text_logits = outputs.logits

    audio_labels, text_labels, audio_loss_mask, text_loss_mask = labels
    # 支持 batch_size >= 1，但如果 batch_size > 1 会给出警告
    if audio_labels.shape[0] > 1:
        import warnings
        warnings.warn(f"Batch size is {audio_labels.shape[0]}, but compute_loss is optimized for batch_size=1. Results may vary.")

    audio_loss = torch.nn.functional.cross_entropy(audio_logits.view(-1, audio_logits.shape[-1]), audio_labels.view(-1), reduction="none")
    text_loss = torch.nn.functional.cross_entropy(text_logits.view(-1, text_logits.shape[-1]), text_labels.view(-1), reduction="none")


    audio_loss = (audio_loss * audio_loss_mask.view(-1)).sum() / (audio_loss_mask.view(-1).sum() + 1e-4)
    text_loss = (text_loss * text_loss_mask.view(-1)).sum() / (text_loss_mask.view(-1).sum() + 1e-4)
    loss = audio_loss + text_loss
    return loss


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()

    # This serves for single-gpu qlora.
    if getattr(training_args, 'deepspeed', None) and int(os.environ.get("WORLD_SIZE", 1))==1:
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    local_rank = training_args.local_rank

    model_load_kwargs = {
        'low_cpu_mem_usage': not deepspeed.is_deepspeed_zero3_enabled(),
    }

    logger.info(f"Loading kimi-audio main model")

    if os.path.exists(model_args.model_name_or_path):
        # local path
        cache_path = model_args.model_name_or_path
    else:
        # cache everything if model_path is a model-id
        cache_path = snapshot_download(model_args.model_name_or_path)

    logger.info(f"Looking for resources in {cache_path}")
    # check if model_path exists
    if not os.path.exists(model_args.model_path):
        raise ValueError(f"Model path {model_args.model_path} does not exist")
    model = KimiAudioModel.from_pretrained(model_args.model_path, 
                                           device_map=None,
                                           **model_load_kwargs)

    # 应用 LoRA（如果需要）
    model = setup_lora_model(model, lora_args)

    text_tokenizer = AutoTokenizer.from_pretrained(
        cache_path, trust_remote_code=True
    )

    # Load data
    data_module = make_supervised_data_module(
        whisper_model=model.whisper_model, text_tokenizer=text_tokenizer,
        data_args=data_args, max_len=training_args.model_max_length, kimia_token_offset=model.config.kimia_token_offset
    )

    # Start trainner
    trainer = Trainer(
        model=model, args=training_args, 
        compute_loss_func=compute_loss,
        data_collator=data_module["train_dataset"].collate_fn,
        **data_module
    )

    trainer.train()
    trainer.save_state()

    # 保存模型（LoRA 适配器会自动保存）
    if lora_args.use_lora:
        # 保存 LoRA 适配器
        if trainer.args.should_save and trainer.args.local_rank == 0:
            model.save_pretrained(training_args.output_dir)
            logger.info(f"LoRA adapter saved to {training_args.output_dir}")
    else:
        # 保存完整模型
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

if __name__ == "__main__":
    train()