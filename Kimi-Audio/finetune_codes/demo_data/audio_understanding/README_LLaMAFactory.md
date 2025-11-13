# 使用 LLaMAFactory 微调 Kimi-Audio 数据集说明

## 数据转换

### 1. 转换脚本

已创建转换脚本 `convert_to_llamafactory.py`，用于将 Kimi-Audio 格式转换为 LLaMAFactory 的 sharegpt 格式。

### 2. 使用方法

```bash
cd /mnt/workspace/hyq/code/come_in/asr/xjz-assignment/kimi-audio/Kimi-Audio/finetune_codes/demo_data/audio_understanding

# 转换数据
python convert_to_llamafactory.py \
    --input_file data.jsonl \
    --output_file llamafactory_data.json
```

### 3. 数据集配置

转换后的数据已复制到 LLaMAFactory 的 data 目录：
- 文件路径: `llama/LLaMA-Factory/data/kimi_audio_data.json`
- 数据集名称: `kimi_audio_data`

已在 `llama/LLaMA-Factory/data/dataset_info.json` 中注册该数据集。

## 数据格式说明

### Kimi-Audio 格式（输入）
```json
{
  "task_type": "understanding",
  "conversation": [
    {"role": "user", "message_type": "text", "content": "请将语音内容转录为文字。"},
    {"role": "user", "message_type": "audio", "content": "/path/to/audio.wav"},
    {"role": "assistant", "message_type": "text", "content": "转录文本"}
  ]
}
```

### LLaMAFactory sharegpt 格式（输出）
```json
{
  "messages": [
    {"role": "user", "content": "请将语音内容转录为文字。<audio>"},
    {"role": "assistant", "content": "转录文本"}
  ],
  "audios": ["/path/to/audio.wav"]
}
```

## 在 LLaMAFactory 中使用

### 1. 训练配置

在训练配置文件中指定数据集：

```yaml
dataset: kimi_audio_data
template: qwen2_audio  # 或使用支持音频的其他模板
```

### 2. 注意事项

1. **模板选择**: 
   - 如果使用 Kimi-Audio 模型，可能需要自定义模板（参考 `template.py` 中的 `kimi_audio` 模板）
   - 如果使用其他支持音频的模型，可以使用 `qwen2_audio` 等模板

2. **音频路径**: 
   - 确保音频文件路径在训练时可访问
   - 转换脚本已使用绝对路径

3. **音频 token**: 
   - LLaMAFactory 使用 `<audio>` token
   - Kimi-Audio 使用 `<|im_media_begin|>` token
   - 如果直接使用 Kimi-Audio 模型，可能需要自定义模板来匹配特殊 token

## 文件位置

- 转换脚本: `convert_to_llamafactory.py`
- 原始数据: `data.jsonl` (Kimi-Audio 格式)
- 转换后数据: `llamafactory_data.json` (LLaMAFactory 格式)
- LLaMAFactory 数据目录: `llama/LLaMA-Factory/data/kimi_audio_data.json`

