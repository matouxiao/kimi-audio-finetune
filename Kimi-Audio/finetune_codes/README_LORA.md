# LoRA 微调使用指南

本文档介绍如何在 Kimi-Audio 项目中使用 LoRA 进行模型微调。

## 安装依赖

首先确保安装了 `peft` 库：

```bash
pip install peft>=0.10.0
```

或者从 requirements.txt 安装所有依赖：

```bash
pip install -r requirements.txt
```

## LoRA 参数说明

### 主要参数

- `--use_lora`: 是否使用 LoRA 微调（默认: False）
- `--lora_rank`: LoRA 的秩，控制适配器的大小（默认: 8）
- `--lora_alpha`: LoRA 的缩放系数，通常设置为 rank 的 2 倍（默认: 16）
- `--lora_dropout`: LoRA 层的 dropout 率（默认: 0.05）
- `--lora_target`: 目标模块，可以是：
  - `"all"`: 应用到所有线性层（推荐）
  - 逗号分隔的模块名，如 `"q_proj,k_proj,v_proj,o_proj"`

## 使用方法

### 方法 1: 使用脚本

```bash
bash finetune_codes/finetune_lora.sh
```

### 方法 2: 直接使用命令行

```bash
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
    --model_max_length 8192
```

## 参数调优建议

### LoRA Rank
- **较小值 (4-8)**: 参数更少，训练更快，但可能表达能力有限
- **中等值 (16-32)**: 平衡性能和效率
- **较大值 (64+)**: 更强的表达能力，但参数更多

### LoRA Alpha
- 通常设置为 `lora_rank * 2`
- 可以尝试 `lora_rank` 的 1-4 倍来调整适配器的影响

### LoRA Target
- **"all"**: 推荐用于大多数场景，自动应用到所有线性层
- **指定模块**: 如果只想微调注意力层，可以使用 `"q_proj,k_proj,v_proj,o_proj"`

## 输出文件

使用 LoRA 微调后，输出目录将包含：

- `adapter_config.json`: LoRA 配置
- `adapter_model.bin` 或 `adapter_model.safetensors`: LoRA 权重
- 其他训练相关文件（checkpoints, logs 等）

## 加载 LoRA 适配器进行推理

```python
from peft import PeftModel
from finetune_codes.model import KimiAudioModel

# 加载基础模型
base_model = KimiAudioModel.from_pretrained("output/pretrained_hf")

# 加载 LoRA 适配器
model = PeftModel.from_pretrained(base_model, "output/lora_finetuned")

# 如果需要合并适配器到基础模型（可选）
# model = model.merge_and_unload()
```

## 与全参数微调的对比

| 特性 | LoRA 微调 | 全参数微调 |
|------|-----------|------------|
| 参数量 | 仅适配器权重（通常 < 1%） | 全部模型参数 |
| 显存占用 | 低 | 高 |
| 训练速度 | 快 | 慢 |
| 模型大小 | 小（仅保存适配器） | 大（保存完整模型） |
| 灵活性 | 可以切换不同适配器 | 需要完整模型 |

## 注意事项

1. **Whisper 模型**: LoRA 不会应用到 `whisper_model` 部分，该部分保持冻结
2. **Embedding 层**: 默认不会应用到 embedding 层，如需微调 embedding，需要修改 `find_all_linear_modules` 函数
3. **DeepSpeed**: 支持 DeepSpeed ZeRO-2/3，但 ZeRO-3 下保存适配器可能需要特殊处理
4. **多 GPU**: 支持多 GPU 训练，使用标准的 HuggingFace Trainer 分布式训练

## 故障排除

### 问题 1: 找不到线性模块
如果指定了 `lora_target` 但报错找不到模块，可以先运行查看所有可用模块：

```python
from finetune_codes.model import KimiAudioModel
model = KimiAudioModel.from_pretrained("output/pretrained_hf")
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        print(name)
```

### 问题 2: 显存不足
- 减小 `lora_rank`
- 减小 `per_device_train_batch_size`
- 增加 `gradient_accumulation_steps`
- 使用 DeepSpeed ZeRO

### 问题 3: 训练效果不佳
- 尝试增加 `lora_rank`
- 调整 `lora_alpha`
- 检查数据质量和数量
- 调整学习率

## 参考

- [PEFT 文档](https://huggingface.co/docs/peft)
- [LoRA 论文](https://arxiv.org/abs/2106.09685)
- LLaMA-Factory 实现参考: `/mnt/workspace/hyq/code/come_in/asr/xjz-assignment/LLaMA-Factory`

