#!/usr/bin/env python3
"""
测试合并后的 LoRA 模型
"""
import os
from kimia_infer.api.kimia import KimiAudio

def main():
    # 模型路径
    model_path = "output/lora_merged_for_inference"
    
    if not os.path.exists(model_path):
        print(f"错误: 模型路径不存在: {model_path}")
        print("请先运行 merge_lora_adapter.py 合并模型")
        return
    
    print(f"加载模型: {model_path}")
    model = KimiAudio(model_path=model_path, load_detokenizer=False)
    print("模型加载完成！\n")
    
    # 设置采样参数
    sampling_params = {
        "audio_temperature": 0.8,
        "audio_top_k": 10,
        "text_temperature": 0.0,
        "text_top_k": 5,
        "audio_repetition_penalty": 1.0,
        "audio_repetition_window_size": 64,
        "text_repetition_penalty": 1.0,
        "text_repetition_window_size": 16,
    }
    
    # 测试 ASR 任务
    print("="*80)
    print("测试 ASR 任务")
    print("="*80)
    
    # 如果有测试音频，使用它；否则使用示例音频
    test_audio = "test_audios/asr_example.wav"
    if not os.path.exists(test_audio):
        print(f"警告: 测试音频不存在: {test_audio}")
        print("请准备测试音频文件")
        return
    
    messages = [
        {"role": "user", "message_type": "text", "content": "请将语音内容转录为文字。"},
        {
            "role": "user",
            "message_type": "audio",
            "content": test_audio,
        },
    ]
    
    try:
        wav, text = model.generate(messages, **sampling_params, output_type="text")
        print(f"\n输入音频: {test_audio}")
        print(f"转录结果: {text}\n")
    except Exception as e:
        print(f"错误: {str(e)}")

if __name__ == "__main__":
    main()

