#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将 Kimi-Audio 格式的数据转换为 LLaMAFactory 的 sharegpt 格式

使用方法:
    python convert_to_llamafactory.py \
        --input_file data.jsonl \
        --output_file llamafactory_data.json
"""

import os
import json
import argparse
from typing import List, Dict, Any


def convert_kimi_to_llamafactory(input_file: str, output_file: str):
    """
    将 Kimi-Audio 格式转换为 LLaMAFactory sharegpt 格式
    
    Args:
        input_file: 输入的 JSONL 文件路径（Kimi-Audio 格式）
        output_file: 输出的 JSON 文件路径（LLaMAFactory sharegpt 格式）
    """
    data_list = []
    skipped_count = 0
    audio_not_found_count = 0
    
    # 读取 JSONL 文件
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: 第 {line_num} 行 JSON 解析失败: {e}")
                skipped_count += 1
                continue
            
            # 检查必需字段
            if "conversation" not in item:
                print(f"Warning: 第 {line_num} 行缺少 'conversation' 字段，跳过")
                skipped_count += 1
                continue
            
            conversation = item["conversation"]
            if not isinstance(conversation, list) or len(conversation) == 0:
                print(f"Warning: 第 {line_num} 行 conversation 为空，跳过")
                skipped_count += 1
                continue
            
            # 提取消息和音频路径
            messages = []
            audio_paths = []
            text_prompt = None
            
            for msg in conversation:
                role = msg.get("role", "")
                message_type = msg.get("message_type", "")
                content = msg.get("content", "")
                
                if message_type == "text":
                    if role == "user":
                        # 保存用户文本提示
                        text_prompt = content
                    elif role == "assistant":
                        # 助手回复
                        messages.append({
                            "role": "assistant",
                            "content": content
                        })
                elif message_type == "audio":
                    if role == "user":
                        # 音频文件路径
                        audio_path = content
                        # 检查文件是否存在
                        if not os.path.exists(audio_path):
                            print(f"Warning: 第 {line_num} 行音频文件不存在: {audio_path}")
                            audio_not_found_count += 1
                            # 仍然添加，让用户自己处理
                        
                        audio_paths.append(audio_path)
            
            # 构建用户消息：文本提示 + <audio> token
            if text_prompt is None:
                text_prompt = "请将语音内容转录为文字。"
            
            # 为每个音频添加 <audio> token
            user_content = text_prompt
            for _ in audio_paths:
                user_content += "<audio>"
            
            messages.insert(0, {
                "role": "user",
                "content": user_content
            })
            
            # 检查是否有助手回复
            if len(messages) < 2:
                print(f"Warning: 第 {line_num} 行缺少助手回复，跳过")
                skipped_count += 1
                continue
            
            # 检查音频数量是否匹配
            if len(audio_paths) == 0:
                print(f"Warning: 第 {line_num} 行没有音频文件，跳过")
                skipped_count += 1
                continue
            
            # 构建 LLaMAFactory 格式的数据项
            llamafactory_item = {
                "messages": messages,
                "audios": audio_paths
            }
            
            data_list.append(llamafactory_item)
    
    # 写入 JSON 文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, ensure_ascii=False, indent=2)
    
    print(f"\n转换完成！")
    print(f"成功转换: {len(data_list)} 条数据")
    print(f"跳过: {skipped_count} 条数据")
    print(f"音频文件不存在: {audio_not_found_count} 个")
    print(f"输出文件: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将 Kimi-Audio 格式转换为 LLaMAFactory sharegpt 格式")
    parser.add_argument(
        "--input_file",
        type=str,
        default="data.jsonl",
        help="输入的 JSONL 文件路径（Kimi-Audio 格式），默认为 data.jsonl"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="llamafactory_data.json",
        help="输出的 JSON 文件路径（LLaMAFactory sharegpt 格式），默认为 llamafactory_data.json"
    )
    
    args = parser.parse_args()
    
    # 转换为绝对路径
    input_file = os.path.abspath(args.input_file)
    output_file = os.path.abspath(args.output_file)
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"输入文件不存在: {input_file}")
    
    # 创建输出目录（如果不存在）
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    print("-" * 50)
    
    convert_kimi_to_llamafactory(input_file, output_file)

