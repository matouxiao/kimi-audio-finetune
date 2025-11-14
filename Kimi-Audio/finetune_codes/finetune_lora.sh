#!/bin/bash

# LoRA 微调脚本示例
# 使用 LoRA 进行 Kimi-Audio 模型微调
# 支持多GPU训练

export CUDA_DEVICE_MAX_CONNECTIONS=1

# 自动检测GPU数量
GPUS_PER_NODE=$(python -c 'import torch; print(torch.cuda.device_count())')
echo "Detected ${GPUS_PER_NODE} GPU(s)"

# 模型路径
MODEL_NAME_OR_PATH="moonshotai/Kimi-Audio-7B"
MODEL_PATH="output/pretrained_hf"

# 数据路径
DATA_PATH="finetune_codes/demo_data/audio_understanding/data_10k_with_semantic_codes.jsonl"

# 输出路径
OUTPUT_DIR="output/lora_finetuned"

# LoRA 参数
USE_LORA=true
LORA_RANK=8
LORA_ALPHA=16
LORA_DROPOUT=0.05
LORA_TARGET="all"  # 或指定模块，如 "q_proj,k_proj,v_proj,o_proj"

# 训练参数
NUM_EPOCHS=5
PER_DEVICE_TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=8
LEARNING_RATE=5e-5
MODEL_MAX_LENGTH=8192
EVAL_RATIO=0.05

# 运行训练（支持多GPU）
# 如果只有1个GPU，直接使用python；多个GPU使用torchrun
if [ ${GPUS_PER_NODE} -eq 1 ]; then
    echo "Using single GPU training"
    python finetune.py \
        --model_name_or_path ${MODEL_NAME_OR_PATH} \
        --model_path ${MODEL_PATH} \
        --data_path ${DATA_PATH} \
        --output_dir ${OUTPUT_DIR} \
        --use_lora \
        --lora_rank ${LORA_RANK} \
        --lora_alpha ${LORA_ALPHA} \
        --lora_dropout ${LORA_DROPOUT} \
        --lora_target ${LORA_TARGET} \
        --num_train_epochs ${NUM_EPOCHS} \
        --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
        --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
        --learning_rate ${LEARNING_RATE} \
        --model_max_length ${MODEL_MAX_LENGTH} \
        --eval_ratio ${EVAL_RATIO} \
        --bf16 \
        --logging_steps 50 \
        --save_steps 1000 \
        --save_total_limit 5 \
        --overwrite_output_dir
else
    echo "Using multi-GPU training with ${GPUS_PER_NODE} GPUs"
    torchrun --nproc_per_node=${GPUS_PER_NODE} --master_port=29500 finetune.py \
        --model_name_or_path ${MODEL_NAME_OR_PATH} \
        --model_path ${MODEL_PATH} \
        --data_path ${DATA_PATH} \
        --output_dir ${OUTPUT_DIR} \
        --use_lora \
        --lora_rank ${LORA_RANK} \
        --lora_alpha ${LORA_ALPHA} \
        --lora_dropout ${LORA_DROPOUT} \
        --lora_target ${LORA_TARGET} \
        --num_train_epochs ${NUM_EPOCHS} \
        --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
        --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
        --learning_rate ${LEARNING_RATE} \
        --model_max_length ${MODEL_MAX_LENGTH} \
        --eval_ratio ${EVAL_RATIO} \
        --bf16 \
        --logging_steps 50 \
        --save_steps 1000 \
        --save_total_limit 5 \
        --overwrite_output_dir
fi

echo "LoRA fine-tuning completed! Adapter saved to ${OUTPUT_DIR}"

