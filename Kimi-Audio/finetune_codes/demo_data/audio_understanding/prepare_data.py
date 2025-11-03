import os
import csv
import json
import argparse

def prepare_data_from_csv(csv_path: str, audio_dir: str, output_path: str):
    """
    从CSV文件和音频文件夹生成微调用的JSONL数据
    
    Args:
        csv_path: CSV文件路径 (split_lingyin_audio_test.csv)
        audio_dir: 音频文件所在目录
        output_path: 输出的JSONL文件路径
    """
    question = "请将语音内容转录为文字。"  # 中文提示词
    
    # 确保音频目录存在
    if not os.path.exists(audio_dir):
        raise ValueError(f"音频目录不存在: {audio_dir}")
    
    # 读取CSV文件
    data_list = []
    skipped_count = 0
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            audio_id = row['id'].strip()  # 例如: 1052f191ac834ac7947334fff7ea36ee_1756295348611500000_10
            transcript = row['no_point_text'].strip()  # 转录文本
            
            # 构建音频文件路径（假设文件名就是id.wav）
            audio_path = os.path.join(audio_dir, f"{audio_id}.wav")
            
            # 检查音频文件是否存在
            if not os.path.exists(audio_path):
                print(f"Warning: 音频文件不存在，跳过: {audio_path} (id: {audio_id})")
                skipped_count += 1
                continue
            
            # 使用绝对路径，确保微调时可以找到文件
            audio_path_abs = os.path.abspath(audio_path)
            
            # 构建JSON格式的数据
            data_item = {
                "task_type": "understanding",
                "conversation": [
                    {
                        "role": "user",
                        "message_type": "text",
                        "content": question
                    },
                    {
                        "role": "user",
                        "message_type": "audio",
                        "content": audio_path_abs
                    },
                    {
                        "role": "assistant",
                        "message_type": "text",
                        "content": transcript
                    }
                ]
            }
            data_list.append(data_item)
    
    # 写入JSONL文件
    with open(output_path, 'w', encoding='utf-8') as f:
        for data_item in data_list:
            f.write(json.dumps(data_item, ensure_ascii=False) + '\n')
    
    print(f"成功处理 {len(data_list)} 条数据，跳过 {skipped_count} 条，保存到 {output_path}")

if __name__ == "__main__":
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    parser = argparse.ArgumentParser(description="从CSV文件准备微调数据")
    parser.add_argument("--csv_path", type=str, 
                        default=os.path.join(script_dir, "split_lingyin_audio_test.csv"),
                        help="CSV文件路径，默认为当前脚本目录下的split_lingyin_audio_test.csv")
    parser.add_argument("--audio_dir", type=str, 
                        default=os.path.join(script_dir, "audio"),
                        help="音频文件目录，默认为当前脚本目录下的audio文件夹")
    parser.add_argument("--output_path", type=str,
                        default=os.path.join(script_dir, "data.jsonl"),
                        help="输出的JSONL文件路径，默认为当前脚本目录下的data.jsonl")
    args = parser.parse_args()
    
    # 转换为绝对路径
    csv_path = os.path.abspath(args.csv_path)
    audio_dir = os.path.abspath(args.audio_dir)
    output_path = os.path.abspath(args.output_path)
    
    # 检查CSV文件是否存在
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV文件不存在: {csv_path}")
    
    print(f"CSV文件路径: {csv_path}")
    print(f"音频目录: {audio_dir}")
    print(f"输出文件: {output_path}")
    print("-" * 50)
    
    prepare_data_from_csv(csv_path, audio_dir, output_path)

