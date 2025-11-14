#!/usr/bin/env python3
"""
合并 LoRA 适配器到基础模型，并导出为推理格式
"""
import os
import argparse
import shutil
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM
from huggingface_hub import snapshot_download

from finetune_codes.model import KimiAudioModel
from finetune_codes.modeling_kimia import MoonshotKimiaForCausalLM

def merge_and_export_lora_model(
    base_model_path: str,
    lora_adapter_path: str,
    output_dir: str,
    model_name_or_path: str = "moonshotai/Kimi-Audio-7B"
):
    """
    合并 LoRA 适配器到基础模型并导出
    
    Args:
        base_model_path: 基础模型路径（output/pretrained_hf）
        lora_adapter_path: LoRA 适配器路径（output/lora_finetuned）
        output_dir: 输出目录（合并后的模型）
        model_name_or_path: 模型名称或路径（用于加载配置）
    """
    print(f"="*80)
    print("合并 LoRA 适配器到基础模型")
    print(f"="*80)
    
    # 1. 加载基础模型
    print(f"\n[1/4] 加载基础模型: {base_model_path}")
    if not os.path.exists(base_model_path):
        raise ValueError(f"基础模型路径不存在: {base_model_path}")
    
    base_model = KimiAudioModel.from_pretrained(
        base_model_path,
        device_map=None,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    print("✓ 基础模型加载完成")
    
    # 2. 加载 LoRA 适配器
    print(f"\n[2/4] 加载 LoRA 适配器: {lora_adapter_path}")
    if not os.path.exists(lora_adapter_path):
        raise ValueError(f"LoRA 适配器路径不存在: {lora_adapter_path}")
    
    # 使用 PEFT 加载适配器
    model = PeftModel.from_pretrained(base_model, lora_adapter_path)
    print("✓ LoRA 适配器加载完成")
    
    # 3. 合并适配器到基础模型
    print(f"\n[3/4] 合并 LoRA 适配器到基础模型...")
    model = model.merge_and_unload()  # 合并并卸载适配器
    print("✓ LoRA 适配器已合并到基础模型")
    
    # 4. 导出为推理格式
    print(f"\n[4/4] 导出模型到: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存主模型（不包含 whisper_model）
    print("  保存主模型...")
    audio_model = MoonshotKimiaForCausalLM(model.config)
    audio_model_state_dict = {
        k: v for k, v in model.state_dict().items() 
        if not k.startswith("whisper_model")
    }
    audio_model.load_state_dict(audio_model_state_dict)
    audio_model.save_pretrained(output_dir)
    print("  ✓ 主模型已保存")
    
    # 复制配置文件
    print("  复制配置文件...")
    config_src = "finetune_codes/configuration_moonshot_kimia.py"
    config_dst = os.path.join(output_dir, "configuration_moonshot_kimia.py")
    if os.path.exists(config_src):
        shutil.copyfile(config_src, config_dst)
        print("  ✓ 配置文件已复制")
    
    modeling_src = "finetune_codes/modeling_kimia.py"
    modeling_dst = os.path.join(output_dir, "modeling_moonshot_kimia.py")
    if os.path.exists(modeling_src):
        shutil.copyfile(modeling_src, modeling_dst)
        print("  ✓ 模型文件已复制")
    
    # 处理 whisper_model（如果需要）
    print("  处理 whisper_model...")
    from kimia_infer.models.tokenizer.whisper_Lv3.whisper import WhisperModel
    
    whisper_model = WhisperModel.from_pretrained("openai/whisper-large-v3")
    kimiaudio_whisper_encoder_state_dict = {
        k.replace("speech_encoder.", "encoder."): v 
        for k, v in model.whisper_model.state_dict().items() 
        if k.startswith("speech_encoder")
    }
    
    missing_keys, unexpected_keys = whisper_model.load_state_dict(
        kimiaudio_whisper_encoder_state_dict, strict=False
    )
    assert len(unexpected_keys) == 0, f"Unexpected keys: {unexpected_keys}"
    
    for k in missing_keys:
        assert k.startswith("decoder"), f"Missing keys: {k}"
    
    whisper_output_dir = os.path.join(output_dir, "whisper-large-v3")
    os.makedirs(whisper_output_dir, exist_ok=True)
    whisper_model.save_pretrained(whisper_output_dir)
    print("  ✓ whisper_model 已保存")
    
    print(f"\n{'='*80}")
    print(f"✓ 模型合并和导出完成！")
    print(f"  输出目录: {output_dir}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="合并 LoRA 适配器并导出为推理模型")
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="output/pretrained_hf",
        help="基础模型路径"
    )
    parser.add_argument(
        "--lora_adapter_path",
        type=str,
        default="output/lora_finetuned",
        help="LoRA 适配器路径"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/lora_merged_for_inference",
        help="输出目录（合并后的推理模型）"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="moonshotai/Kimi-Audio-7B",
        help="模型名称或路径（用于加载配置）"
    )
    
    args = parser.parse_args()
    
    merge_and_export_lora_model(
        base_model_path=args.base_model_path,
        lora_adapter_path=args.lora_adapter_path,
        output_dir=args.output_dir,
        model_name_or_path=args.model_name_or_path
    )

