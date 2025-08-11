#!/usr/bin/env bash
# Example: infer.sh best_checkpoint.pt

set -e
MODEL_PATH="$1"

if [ -z "${MODEL_PATH}" ]; then
  echo "Usage: ./infer.sh <model_ckpt.pt>"
  exit 1
fi

# 可根据需要修改
TEST_TXT=./data/test.txt
AUDIO_ROOT=./data
BATCH=512                 # 总 batch，DataParallel 会自动切分到各 GPU
THRESHOLD=0.95
OUT_CSV=./WWWxp-result.csv

# 指定使用哪几张卡
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6

python inference.py \
    --wav_file   "${TEST_TXT}" \
    --audio_root "${AUDIO_ROOT}" \
    --model_path "${MODEL_PATH}" \
    --batch_size "${BATCH}" \
    --threshold  "${THRESHOLD}" \
    --output_file "${OUT_CSV}"
