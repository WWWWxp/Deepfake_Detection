#!/bin/bash
export NCCL_IB_DISABLE=0  # 单机不使用 IB
export TORCH_DYNAMO_CACHE_SIZE_LIMIT=16

NNODES=1
NODE_RANK=0
MASTER_ADDR="127.0.0.1"
MASTER_PORT=29501

# 每个 GPU 进程使用 8 线程更合理（避免 CPU 线程爆炸）
CUDA_LAUNCH_BLOCKING=7 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 OMP_NUM_THREADS=125 torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --nproc_per_node=7 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train.py ./exports_wav-norm_augment ./data/train.txt data
