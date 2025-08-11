#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-GPU inference for Deepfake detection (Wav2Vec2_AASIST).
Uses torch.nn.DataParallel – simply export CUDA_VISIBLE_DEVICES
to control how many GPUs are used.
"""
import os, csv, argparse, warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader

from dataset import from_test_list          # 自己的数据集工具
from params  import params                  # 配置
from model   import Wav2Vec2_AASIST         # 模型
warnings.filterwarnings("ignore")


# ---------- utils ----------
def compute_eer(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return (fpr[idx] + fnr[idx]) / 2


# ---------- inference wrapper ----------
class Inference:
    def __init__(self, device, model_path, model):
        self.device     = device
        self.model_path = model_path
        self.model      = model.to(device).eval()

    # 兼容 DataParallel / 单卡
    def _load_state_dict(self, state):
        target = self.model.module if hasattr(self.model, "module") else self.model
        target.load_state_dict(state["model"], strict=True)

    def restore(self):
        ckpt = torch.load(self.model_path, map_location=self.device)
        self._load_state_dict(ckpt)

    @torch.no_grad()
    def predict(self, waveforms):
        logits = self.model(waveforms)             # (B, 2)
        probs  = F.softmax(logits, dim=1)[:, 0]    # bonafide 概率
        print(probs[:4])
        return probs.cpu().numpy().ravel()


# ---------- main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav_file",   default="./data/test.txt")
    parser.add_argument("--audio_root", default="./data")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_file",default="./result.csv")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--threshold",  type=float, default=0.95)
    args = parser.parse_args()

    # ---------- device & model ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_model = Wav2Vec2_AASIST(layers=params.layers).eval()
    if torch.cuda.device_count() > 1 and torch.cuda.is_available():
        print(f"[Info] Detected {torch.cuda.device_count()} GPUs ➜ DataParallel enabled")
        base_model = torch.nn.DataParallel(base_model)

    infer = Inference(device, args.model_path, base_model)
    infer.restore()
    print(f"[Info] Model restored from {args.model_path}")

    # ---------- dataset ----------
    loader = from_test_list(args.wav_file, args.audio_root, params,
                            batch_size=args.batch_size)

    all_probs, all_utt = [], []

    # 保存 utt-path 列表，用于最终写 csv
    utt_path_list = []
    with open(args.wav_file, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        first  = next(reader)
        if first[0].lower() in ("utt", "utt_id"):
            rows = reader
        else:
            utt_path_list.append((first[0], first[1]))
            rows = reader
        for r in rows:
            utt_path_list.append((r[0], r[1]))

    # ---------- inference loop ----------
    for batch in tqdm(loader, desc="Inferencing"):
        if batch is None:              # collate 可能返回 None
            continue
        wav = batch["waveforms"].to(device)        # (B, N)
        ids = batch["utt_ids"]
        probs = infer.predict(wav)                 # (B,)
        all_probs.extend(probs)
        all_utt.extend(ids)

    # ---------- threshold → binary label ----------
    y_prob = np.array(all_probs)
    y_bin  = (y_prob >= args.threshold).astype(int)    # 1 Bonafide / 0 Spoof

    # ---------- save csv ----------
    with open(args.output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["utt", "path", "label"])
        for (utt, path), lbl in zip(utt_path_list, y_bin):
            writer.writerow([utt, path, "Bonafide" if lbl else "Spoof"])

    print(f"[Done] Results saved to {args.output_file}")
    print(f"Total: {len(all_utt)}  |  Bonafide: {y_bin.sum()}  |  Spoof: {len(y_bin)-y_bin.sum()}")


if __name__ == "__main__":
    main()
