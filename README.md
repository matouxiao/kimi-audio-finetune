# Kimi-Audio 微调实验总结

## 实验概述

本实验使用自定义的中文语音数据对 Kimi-Audio-7B 模型进行微调，实现中文语音转文字（ASR）任务。实验过程中遇到了多个技术挑战，并逐一解决。

## 实验数据

- **训练数据**：9条中文语音音频文件（.wav格式）
- **数据位置**：`finetune_codes/demo_data/audio_understanding/audio/`
- **标注文件**：`split_lingyin_audio_test.csv`（包含音频ID和对应转录文本）
- **测试数据**：10条测试音频，标准答案存储在 `split_lingyin_audio_answer.csv`

## 实验流程

### 1. 数据准备

#### 1.1 创建数据准备脚本

创建了 `prepare_data.py` 脚本，功能：
- 读取CSV文件，提取音频ID和转录文本
- 在audio文件夹中匹配对应的音频文件
- 生成符合微调格式要求的JSONL文件

**脚本位置**：`finetune_codes/demo_data/audio_understanding/prepare_data.py`

**使用方法**：
```bash
cd /mnt/workspace/hyq/code/come_in/asr/xjz-assignment/kimi-audio/Kimi-Audio
python finetune_codes/demo_data/audio_understanding/prepare_data.py
```

**生成文件**：`data.jsonl`（包含9条训练数据）

#### 1.2 数据格式

每条数据格式：
```json
{
    "task_type": "understanding",
    "conversation": [
        {
            "role": "user",
            "message_type": "text",
            "content": "请将语音内容转录为文字。"
        },
        {
            "role": "user",
            "message_type": "audio",
            "content": "/path/to/audio.wav"
        },
        {
            "role": "assistant",
            "message_type": "text",
            "content": "转录文本"
        }
    ]
}
```

### 2. 提取语义Token

对JSONL数据进行预处理，提取音频的语义token：

```bash
CUDA_VISIBLE_DEVICES=0 python -m finetune_codes.extract_semantic_codes \
    --input_file "finetune_codes/demo_data/audio_understanding/data.jsonl" \
    --output_file "finetune_codes/demo_data/audio_understanding/data_with_semantic_codes.jsonl"
```

**输出**：`data_with_semantic_codes.jsonl`（每条数据的音频消息中添加了`audio_tokens`字段）

### 3. 下载预训练模型

```bash
CUDA_VISIBLE_DEVICES=0 python -m finetune_codes.model \
    --model_name "moonshotai/Kimi-Audio-7B" \
    --output_dir "output/pretrained_hf"
```

**注意**：模型较大，下载需要一定时间，需要访问HuggingFace。

### 4. 模型微调

#### 4.1 遇到的问题及解决方案

##### 问题1：参数名称错误
**错误信息**：
```
ValueError: Some specified arguments are not used by the HfArgumentParser: ['--evaluation_strategy', 'no']
```

**原因**：新版本的transformers库将参数名从 `evaluation_strategy` 改为 `eval_strategy`

**解决方案**：修改 `finetune_codes/finetune_ds.sh` 第99行：
```bash
# 修改前
--evaluation_strategy "no" \

# 修改后
--eval_strategy "no" \
```

##### 问题2：Checkpoint未保存
**现象**：微调完成后找不到checkpoint文件

**原因分析**：
- 数据量小：只有9条数据
- batch_size=1，gradient_accumulation_steps=1
- 每个epoch只有9步，5个epoch共45步
- 但 `save_steps=1000`，所以从未触发保存

**解决方案**：修改 `finetune_codes/finetune_ds.sh`：
```bash
--save_steps 10  # 从1000改为10，确保每10步保存一次
--save_total_limit 5  # 只保留最新5个checkpoint
```

##### 问题3：多GPU训练卡住
**现象**：使用8个GPU时出现NCCL通信hang警告，训练卡住

**原因**：
- 数据量太小（9条），每个GPU只能分到1条数据
- 通信开销远大于计算，导致通信卡住

**解决方案**：改用单GPU训练
```bash
export CUDA_VISIBLE_DEVICES=1  # 只使用GPU 1
bash finetune_codes/finetune_ds.sh ...
```

##### 问题4：CUDA Out of Memory (OOM)
**错误信息**：
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 3.73 GiB.
```

**原因**：
- 使用DeepSpeed Zero3时，在单GPU上内存开销过大
- 模型参数量大（11.31B参数），显存占用高

**解决方案**：修改DeepSpeed配置，启用CPU Offload
- 修改 `finetune_codes/ds_config_zero3.json`：
  - `offload_optimizer`: `"device": "none"` → `"device": "cpu"`
  - `offload_param`: `"device": "none"` → `"device": "cpu"`
  - 减小内存相关参数：`stage3_max_live_parameters` 从 `1e9` 改为 `1e8`

**效果**：将优化器状态和部分模型参数offload到CPU，显著降低GPU显存占用

##### 问题5：训练速度变慢
**现象**：使用CPU offload后，每步训练和保存checkpoint时间变长

**原因**：CPU-GPU数据传输需要额外时间

**处理**：这是正常的权衡，为了在单GPU上运行大模型必须接受的速度损失

#### 4.2 超参数调整

针对小数据集（9条数据）优化了超参数：

| 参数 | 原始值 | 调整后 | 原因 |
|------|--------|--------|------|
| `eval_ratio` | 0.05 | 0.0 | 数据太少，不分割验证集 |
| `num_train_epochs` | 5 | 10 | 增加训练轮数 |
| `gradient_accumulation_steps` | 1 | 4 | 模拟更大batch size，训练更稳定 |
| `learning_rate` | 1e-5 | 3e-6 | 降低学习率，减少过拟合风险 |
| `weight_decay` | 0.1 | 0.01 | 降低正则化强度 |
| `warmup_ratio` | 0.01 | 0.1 | 增加warmup步数，训练更稳定 |
| `save_steps` | 1000 | 5 | 确保小数据集能保存checkpoint |

**最终训练命令**：
```bash
export CUDA_VISIBLE_DEVICES=1
bash finetune_codes/finetune_ds.sh \
    --model_path "output/pretrained_hf" \
    --data "finetune_codes/demo_data/audio_understanding/data_with_semantic_codes.jsonl"
```

**训练结果**：
- 训练轮数：10 epochs
- 总步数：30步（9条数据，每步包含4个累积梯度）
- Checkpoint保存位置：`output/kimiaudio_ckpts/checkpoint-5, checkpoint-10, ..., checkpoint-30`

### 5. 模型导出

将训练好的checkpoint转换为推理格式：

```bash
CUDA_VISIBLE_DEVICES=1 python -m finetune_codes.model \
    --model_name "moonshotai/Kimi-Audio-7B" \
    --action "export_model" \
    --input_dir "output/kimiaudio_ckpts/checkpoint-30" \
    --output_dir "output/finetuned_hf_for_inference"
```

**输出**：`output/finetuned_hf_for_inference/` 目录，包含可用于推理的模型文件

### 6. 模型测试与评估

#### 6.1 创建测试脚本

创建了 `test_model.py` 脚本，功能：
- 加载微调后的模型
- 对测试音频进行推理
- 与标准答案对比
- 计算性能指标（CER、相似度等）

**脚本位置**：`finetune_codes/demo_data/audio_understanding/test_model.py`

**使用方法**：
```bash
cd /mnt/workspace/hyq/code/come_in/asr/xjz-assignment/kimi-audio/Kimi-Audio
CUDA_VISIBLE_DEVICES=1 python finetune_codes/demo_data/audio_understanding/test_model.py
```

#### 6.2 评估指标

- **完全匹配率**：完全相同的转录结果占比
- **CER (字符错误率)**：基于编辑距离计算的字符级错误率
- **相似度分数**：字符级相似度（1 - CER）

## 主要修改的文件

1. **finetune_codes/finetune_ds.sh**
   - 修改参数名：`evaluation_strategy` → `eval_strategy`
   - 调整超参数以适应小数据集
   - `save_steps` 从1000改为5

2. **finetune_codes/ds_config_zero3.json**
   - 启用CPU offload：`offload_optimizer` 和 `offload_param` 改为 `"cpu"`
   - 减小内存相关参数

3. **finetune_codes/demo_data/audio_understanding/prepare_data.py**（新创建）
   - 数据准备脚本，将CSV转换为JSONL格式

4. **finetune_codes/demo_data/audio_understanding/test_model.py**（新创建）
   - 测试脚本，用于评估微调后的模型

## 经验总结

### 小数据集微调的注意事项

1. **保存策略**：`save_steps` 应该根据数据量设置，确保训练过程中能保存checkpoint
2. **GPU选择**：小数据集不适合多GPU分布式训练，应使用单GPU
3. **内存管理**：大模型在单GPU上训练需要使用CPU offload，但会降低速度
4. **超参数调整**：小数据集需要更小的学习率和更多的训练轮数，避免过拟合

### DeepSpeed配置要点

- **Zero3 + CPU Offload**：适合在单GPU上训练大模型
- **参数调优**：根据显存大小调整 `stage3_max_live_parameters` 等参数
- **性能权衡**：CPU offload会增加训练时间，但能避免OOM

### 常见问题排查

1. **Checkpoint未保存**：检查 `save_steps` 是否合理
2. **NCCL通信问题**：小数据集改用单GPU
3. **OOM错误**：启用CPU offload或减小 `model_max_length`
4. **导入错误**：确保Python路径正确，项目根目录在sys.path中

## 文件结构

```
kimi-audio/Kimi-Audio/
├── finetune_codes/
│   ├── demo_data/
│   │   └── audio_understanding/
│   │       ├── audio/                    # 训练音频文件
│   │       ├── test_audio/               # 测试音频文件
│   │       ├── prepare_data.py           # 数据准备脚本（新建）
│   │       ├── test_model.py             # 测试评估脚本（新建）
│   │       ├── data.jsonl                # 训练数据（生成）
│   │       ├── data_with_semantic_codes.jsonl  # 预处理后的数据（生成）
│   │       ├── split_lingyin_audio_test.csv    # 训练数据标注
│   │       └── split_lingyin_audio_answer.csv   # 测试标准答案
│   ├── finetune_ds.sh                    # 微调脚本（已修改）
│   └── ds_config_zero3.json              # DeepSpeed配置（已修改）
└── output/
    ├── pretrained_hf/                    # 预训练模型
    ├── kimiaudio_ckpts/                  # 训练checkpoint
    └── finetuned_hf_for_inference/       # 导出后的推理模型
```

## 后续改进方向

1. **增加训练数据**：当前只有9条数据，建议增加到数百条或更多
2. **数据增强**：对小数据集可以使用数据增强技术
3. **评估指标**：可以添加更详细的评估指标，如WER、BLEU等
4. **模型优化**：尝试不同的超参数组合，找到最优配置

## 参考资源

- [Kimi-Audio官方文档](https://github.com/MoonshotAI/Kimi-Audio)
- [DeepSpeed文档](https://www.deepspeed.ai/)
- [Transformers文档](https://huggingface.co/docs/transformers)

---

**实验日期**：2025年10月31日  
**实验环境**：Linux, CUDA, 单GPU (PPU-ZW810E, 95GB显存)  
**模型版本**：moonshotai/Kimi-Audio-7B

