#!/usr/bin/env python3
"""
测试微调后的模型，对测试音频进行推理并评估结果
"""
import os
import csv
import sys

# 添加项目根目录到Python路径，以便导入kimia_infer
script_dir = os.path.dirname(os.path.abspath(__file__))
# 从 audio_understanding -> demo_data -> finetune_codes -> Kimi-Audio (项目根目录)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from kimia_infer.api.kimia import KimiAudio

# 设置路径（script_dir已在上面定义）
csv_path = os.path.join(script_dir, "split_lingyin_audio_answer.csv")
test_audio_dir = os.path.join(script_dir, "test_audio")
model_path = os.path.join(project_root, "output/finetuned_hf_for_inference")

def main():
    # 检查模型路径是否存在
    if not os.path.exists(model_path):
        print(f"错误: 模型路径不存在: {model_path}")
        print(f"请先运行模型导出命令将checkpoint转换为推理格式")
        sys.exit(1)
    
    # 加载模型
    print(f"正在加载模型: {model_path}")
    model = KimiAudio(model_path=model_path, load_detokenizer=False)
    print("模型加载完成！")
    
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
    
    # 读取CSV文件
    results = []
    total = 0
    correct = 0
    
    print(f"\n开始测试，读取CSV文件: {csv_path}")
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            audio_id = row['id'].strip()
            ground_truth = row['no_point_text'].strip()
            
            # 构建音频文件路径
            audio_file = os.path.join(test_audio_dir, f"{audio_id}.wav")
            
            if not os.path.exists(audio_file):
                print(f"警告: 音频文件不存在，跳过: {audio_file}")
                continue
            
            print(f"\n[{idx}/{10}] 处理音频: {audio_id}")
            
            # 准备输入
            messages = [
                {"role": "user", "message_type": "text", "content": "请将语音内容转录为文字。"},
                {
                    "role": "user",
                    "message_type": "audio",
                    "content": os.path.abspath(audio_file),
                },
            ]
            
            try:
                # 生成转录文本
                wav, predicted_text = model.generate(messages, **sampling_params, output_type="text")
                predicted_text = predicted_text.strip()
                
                # 记录结果
                results.append({
                    "id": audio_id,
                    "ground_truth": ground_truth,
                    "predicted": predicted_text,
                    "match": (ground_truth == predicted_text)
                })
                
                total += 1
                if ground_truth == predicted_text:
                    correct += 1
                    status = "✓ 匹配"
                else:
                    status = "✗ 不匹配"
                
                print(f"标准答案: {ground_truth}")
                print(f"模型输出: {predicted_text}")
                print(f"结果: {status}")
                
            except Exception as e:
                print(f"错误: 处理音频 {audio_id} 时出错: {str(e)}")
                results.append({
                    "id": audio_id,
                    "ground_truth": ground_truth,
                    "predicted": f"ERROR: {str(e)}",
                    "match": False
                })
                total += 1
    
    # 输出总结
    print("\n" + "="*80)
    print("测试总结")
    print("="*80)
    print(f"总测试数: {total}")
    print(f"完全匹配: {correct}")
    print(f"匹配率: {correct/total*100:.2f}%" if total > 0 else "N/A")
    print("\n详细结果:")
    print("-"*80)
    for i, result in enumerate(results, start=1):
        status = "✓" if result["match"] else "✗"
        print(f"\n[{i}] {status} ID: {result['id']}")
        print(f"    标准答案: {result['ground_truth']}")
        print(f"    模型输出: {result['predicted']}")
    
    # 保存结果到文件
    output_file = os.path.join(script_dir, "test_results.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("测试结果\n")
        f.write("="*80 + "\n")
        f.write(f"总测试数: {total}\n")
        f.write(f"完全匹配: {correct}\n")
        f.write(f"匹配率: {correct/total*100:.2f}%\n" if total > 0 else "N/A\n")
        f.write("\n详细结果:\n")
        f.write("-"*80 + "\n")
        for i, result in enumerate(results, start=1):
            status = "✓" if result["match"] else "✗"
            f.write(f"\n[{i}] {status} ID: {result['id']}\n")
            f.write(f"    标准答案: {result['ground_truth']}\n")
            f.write(f"    模型输出: {result['predicted']}\n")
    
    print(f"\n结果已保存到: {output_file}")

if __name__ == "__main__":
    main()

