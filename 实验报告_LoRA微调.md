# Kimi-Audio LoRA 微调实验报告

## 实验日期
2025年11月13日

## 实验目标
在 Kimi-Audio 项目中实现 LoRA (Low-Rank Adaptation) 微调功能，实现参数高效的模型微调。

## 实验环境
- 项目路径：`/mnt/workspace/hyq/code/come_in/asr/xjz-assignment/kimi-audio/Kimi-Audio`
- 基础模型：`moonshotai/Kimi-Audio-7B`
- 数据集：`finetune_codes/demo_data/audio_understanding/data_with_semantic_codes.jsonl` (8个样本)

## 实验过程

### 1. 学习 LLaMA-Factory 中的 LoRA 实现

首先参考了 LLaMA-Factory 项目中的 LoRA 实现方式：

**关键文件：**
- `LLaMA-Factory/examples/train_lora/llama3_lora_sft.yaml` - LoRA 配置示例
- `LLaMA-Factory/src/llamafactory/hparams/finetuning_args.py` - LoRA 参数定义
- `LLaMA-Factory/src/llamafactory/model/adapter.py` - LoRA 适配器实现

**关键参数：**
- `lora_rank`: LoRA 的秩（通常为 4, 8, 16, 32）
- `lora_alpha`: LoRA 的缩放因子（通常为 rank 的 2 倍）
- `lora_dropout`: LoRA 的 dropout 率
- `lora_target`: 应用 LoRA 的目标模块

### 2. 在 Kimi-Audio 中实现 LoRA 微调

#### 2.1 添加依赖

在 `requirements.txt` 中添加：
```
peft>=0.10.0
```

#### 2.2 修改 `finetune.py`

**主要修改：**

1. **导入 PEFT 库**
```python
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
```

2. **添加 LoRA 参数类**
```python
@dataclass
class LoraArguments:
    use_lora: bool = field(default=False, metadata={"help": "Whether to use LoRA"})
    lora_rank: int = field(default=8, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=16, metadata={"help": "LoRA alpha"})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout"})
    lora_target: str = field(default="all", metadata={"help": "LoRA target modules"})
```

3. **实现 `find_all_linear_modules` 函数**
   - 自动识别模型中的线性层
   - 优先选择常见的注意力层（q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj）
   - 实现三层回退机制确保找到有效的模块

4. **实现 `setup_lora_model` 函数**
   - 创建 `LoraConfig`
   - 使用 `get_peft_model` 应用 LoRA
   - 打印可训练参数统计

5. **修改训练流程**
   - 在加载模型后调用 `setup_lora_model`
   - 调整模型保存逻辑，只保存 LoRA 适配器

#### 2.3 修改 `finetune_codes/model.py`

**主要修改：**

1. **添加 PEFT 必需的方法**
   - `prepare_inputs_for_generation`: 为生成准备输入
   - `_reorder_cache`: 支持 beam search

2. **修复 whisper_model 的类型问题**
   - 在 `__init__` 中将 `speech_encoder` 转换为 bf16
   - 确保所有参数和 buffer 都是 bf16，以支持 Flash Attention
   - 在 `forward` 中使用 bf16 autocast

### 3. 遇到的问题及解决方案

#### 问题 1: LoRA 模块名称错误
**错误信息：**
```
ValueError: Target module MoonshotDecoderLayer(...) is not supported.
```

**原因：** `find_all_linear_modules` 返回了模块索引而不是模块名称。

**解决方案：** 
- 优化函数逻辑，过滤掉纯数字的模块名
- 优先选择包含字母的模块名
- 实现三层回退机制

#### 问题 2: 评估数据为空
**错误信息：**
```
AssertionError: No evaluation data found
```

**原因：** 数据集太小（8个样本），`eval_ratio=0.05` 导致评估集为空。

**解决方案：** 
- 修改 `make_supervised_data_module` 函数
- 如果 `eval_size` 为 0 但总数据 > 1，设置 `eval_size = 1`
- 如果评估集仍为空，发出警告并使用所有数据训练

#### 问题 3: Batch size 断言
**错误信息：**
```
assert len(batch) == 1, "micro batch size is 1 for demo"
```

**原因：** 代码假设 batch size 为 1，但实际可能更大。

**解决方案：** 
- 将断言改为警告
- 在 `collate_fn` 中明确处理 batch size > 1 的情况

#### 问题 4: Whisper 模型类型不匹配
**错误信息：**
```
RuntimeError: Input type (c10::BFloat16) and bias type (float) should be the same
```

**原因：** `whisper_model` 的某些层（如 conv1d）的 bias 是 float32，但输入是 bf16。

**解决方案：** 
- 在 `__init__` 中将 `speech_encoder` 的所有参数转换为 bf16
- 在 `forward` 中使用 bf16 autocast

#### 问题 5: Flash Attention 类型错误
**错误信息：**
```
FlashAttention only support fp16 and bf16 data type
```

**原因：** `speech_encoder` 的参数是 float32，但 Flash Attention 需要 fp16/bf16。

**解决方案：** 
- 在 `__init__` 中将整个 `speech_encoder` 转换为 bf16
- 确保所有参数、buffer 和输入都是 bf16
- 使用 bf16 autocast 确保计算过程中的类型一致性

### 4. 最终配置

**LoRA 参数：**
- `lora_rank`: 8
- `lora_alpha`: 16
- `lora_dropout`: 0.05
- `lora_target`: all (应用到所有线性层)

**训练参数：**
- `num_train_epochs`: 3
- `per_device_train_batch_size`: 1
- `gradient_accumulation_steps`: 8
- `learning_rate`: 1e-4
- `bf16`: True

**目标模块：**
- gate_proj
- down_proj
- k_proj
- o_proj
- v_proj
- q_proj
- up_proj

### 5. 训练结果

**训练统计：**
- 训练时间：20.93 秒
- 训练步数：3 步（8个样本，batch_size=1，gradient_accumulation_steps=8）
- 最终损失：13.93
- 可训练参数：26,476,544 (0.2539% of total)
- 总参数：10,429,779,968

**输出文件：**
- `output/lora_finetuned/adapter_config.json` - LoRA 配置
- `output/lora_finetuned/adapter_model.safetensors` - LoRA 权重
- `output/lora_finetuned/checkpoint-3/` - 检查点
- `output/lora_finetuned/trainer_state.json` - 训练状态

**监控工具：**
- SwanLab: https://swanlab.cn/@ligengze/Kimi-Audio
- WandB: https://wandb.ai/ligengh/huggingface

## 关键代码修改

### finetune.py 关键修改

1. **LoRA 参数定义**
```python
@dataclass
class LoraArguments:
    use_lora: bool = field(default=False)
    lora_rank: int = field(default=8)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.05)
    lora_target: str = field(default="all")
```

2. **查找线性模块**
```python
def find_all_linear_modules(model):
    # 优先选择常见的注意力层
    common_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 
                      'gate_proj', 'up_proj', 'down_proj']
    # 实现三层回退机制
    ...
```

3. **设置 LoRA 模型**
```python
def setup_lora_model(model, lora_args: LoraArguments):
    target_modules = find_all_linear_modules(model)
    lora_config = LoraConfig(
        r=lora_args.lora_rank,
        lora_alpha=lora_args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    return model
```

### model.py 关键修改

1. **初始化时转换类型**
```python
def __init__(self, config):
    super().__init__(config)
    self.whisper_model = WhisperEncoder(...)
    # 将 speech_encoder 转换为 bf16
    speech_encoder = self.whisper_model.speech_encoder
    speech_encoder = speech_encoder.to(torch.bfloat16)
    # 确保所有参数都是 bf16
    for name, param in speech_encoder.named_parameters():
        if param.dtype.is_floating_point:
            param.data = param.data.to(torch.bfloat16)
```

2. **Forward 方法中使用 bf16**
```python
def forward(...):
    # 将 mel 转换为 bf16
    mel_input = mel.unsqueeze(0).to(torch.bfloat16)
    # 使用 bf16 autocast
    with torch.amp.autocast(device_type='cuda', enabled=True, dtype=torch.bfloat16):
        whisper_feats = speech_encoder(mel_input, return_dict=True).last_hidden_state
    ...
```

## 使用方法

### 训练命令

```bash
cd /mnt/workspace/hyq/code/come_in/asr/xjz-assignment/kimi-audio/Kimi-Audio

python finetune.py \
    --model_name_or_path moonshotai/Kimi-Audio-7B \
    --model_path output/pretrained_hf \
    --data_path finetune_codes/demo_data/audio_understanding/data_with_semantic_codes.jsonl \
    --output_dir output/lora_finetuned \
    --use_lora \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target all \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 \
    --bf16 \
    --logging_steps 10 \
    --save_steps 500 \
    --save_total_limit 3 \
    --overwrite_output_dir
```

### 加载训练好的模型

```python
from peft import PeftModel
from finetune_codes.model import KimiAudioModel

# 加载基础模型
base_model = KimiAudioModel.from_pretrained("moonshotai/Kimi-Audio-7B")

# 加载 LoRA 适配器
model = PeftModel.from_pretrained(base_model, "output/lora_finetuned")
```

## 经验总结

1. **类型一致性很重要**：在使用 Flash Attention 时，必须确保所有参数和输入都是 fp16 或 bf16。

2. **小数据集处理**：对于小数据集，需要特别处理评估集的划分，避免空评估集。

3. **模块查找策略**：实现多层回退机制，确保能找到有效的线性模块。

4. **参数高效微调**：LoRA 只训练 0.25% 的参数，大大降低了训练成本。

5. **监控工具**：使用 SwanLab 和 WandB 可以方便地查看训练过程和 loss 曲线。

## 后续优化建议

1. **增加数据集大小**：当前只有 8 个样本，建议增加更多训练数据。

2. **调整 logging_steps**：对于小数据集，建议将 `logging_steps` 设置为 1，以便看到每一步的 loss。

3. **尝试不同的 LoRA rank**：可以尝试 rank=4, 16, 32 等不同值，找到最佳平衡点。

4. **添加评估指标**：除了 loss，可以添加更多评估指标（如准确率、BLEU 等）。

5. **超参数调优**：可以尝试不同的学习率、dropout 等超参数。

## 参考资料

- [PEFT 文档](https://huggingface.co/docs/peft)
- [LoRA 论文](https://arxiv.org/abs/2106.09685)
- LLaMA-Factory 项目实现

---

**实验完成时间：** 2025年11月13日 17:07
**实验状态：** ✅ 成功完成

