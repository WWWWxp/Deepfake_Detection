# -*- coding: utf-8 -*-
import os
from pathlib import Path
from typing import Dict, List, Optional
import random
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchaudio
import torchaudio.functional as AF
from torchaudio import transforms as T
from multiprocessing import cpu_count
from params import params
warnings.filterwarnings("ignore")


# ---------- Audio Augmentation Utils ----------
def add_noise_at_snr(clean_audio: torch.Tensor, noise_audio: torch.Tensor, snr_db: float) -> torch.Tensor:
    """
    在指定SNR下添加噪声
    
    Args:
        clean_audio: 干净音频 (1, N)
        noise_audio: 噪声音频 (1, N)
        snr_db: 信噪比(dB)
    
    Returns:
        加噪后的音频
    """
    # 计算信号和噪声的RMS
    clean_rms = torch.sqrt(torch.mean(clean_audio ** 2))
    noise_rms = torch.sqrt(torch.mean(noise_audio ** 2))
    
    # 根据SNR计算噪声缩放因子
    snr_linear = 10 ** (snr_db / 20)
    noise_scale = clean_rms / (noise_rms * snr_linear + 1e-8)
    
    # 添加缩放后的噪声
    noisy_audio = clean_audio + noise_scale * noise_audio
    return noisy_audio


def adjust_length(audio: torch.Tensor, target_length: int) -> torch.Tensor:
    """
    调整音频长度：截取或循环填充
    
    Args:
        audio: 输入音频 (1, N)
        target_length: 目标长度
    
    Returns:
        调整后的音频 (1, target_length)
    """
    current_length = audio.size(1)
    
    if current_length > target_length:
        # 随机截取
        start = random.randint(0, current_length - target_length)
        return audio[:, start:start + target_length]
    elif current_length < target_length:
        # 循环填充
        repeat_times = (target_length + current_length - 1) // current_length
        repeated = audio.repeat(1, repeat_times)
        return repeated[:, :target_length]
    else:
        return audio


def load_augmentation_data_from_params(params):
    """
    从params配置中加载增强数据（RIR、噪声等）
    
    Args:
        params: 参数对象，包含各类增强数据的base path
    
    Returns:
        包含增强数据文件列表的字典
    """
    import glob
    import os
    
    augment_data = {
        'rir_files': [],
        'noise_files': [],
        'music_files': [],
        'speech_files': [],
        'env_files': []
    }
    
    def find_wav_files(base_path):
        """递归查找base_path下的所有wav文件"""
        if not base_path or not os.path.exists(base_path):
            return []
        
        wav_files = []
        # 使用递归glob查找所有wav文件
        wav_files.extend(glob.glob(os.path.join(base_path, "**/*.wav"), recursive=True))
        wav_files.extend(glob.glob(os.path.join(base_path, "**/*.WAV"), recursive=True))
        return wav_files
    
    try:
        # 加载RIR文件
        if hasattr(params, 'rir_base_path') and params.rir_base_path:
            rir_files = find_wav_files(params.rir_base_path)
            augment_data['rir_files'] = rir_files
            print(f"从 {params.rir_base_path} 加载了 {len(rir_files)} 个RIR文件")
        
        # 加载MUSAN噪声文件
        if hasattr(params, 'musan_noise_base_path') and params.musan_noise_base_path:
            noise_files = find_wav_files(params.musan_noise_base_path)
            augment_data['noise_files'] = noise_files
            print(f"从 {params.musan_noise_base_path} 加载了 {len(noise_files)} 个噪声文件")
        
        # 加载MUSAN音乐文件
        if hasattr(params, 'musan_music_base_path') and params.musan_music_base_path:
            music_files = find_wav_files(params.musan_music_base_path)
            augment_data['music_files'] = music_files
            print(f"从 {params.musan_music_base_path} 加载了 {len(music_files)} 个音乐文件")
        
        # 加载MUSAN语音文件
        if hasattr(params, 'musan_speech_base_path') and params.musan_speech_base_path:
            speech_files = find_wav_files(params.musan_speech_base_path)
            augment_data['speech_files'] = speech_files
            print(f"从 {params.musan_speech_base_path} 加载了 {len(speech_files)} 个语音文件")
        
        # 加载环境声文件
        if hasattr(params, 'env_base_path') and params.env_base_path:
            env_files = find_wav_files(params.env_base_path)
            augment_data['env_files'] = env_files
            print(f"从 {params.env_base_path} 加载了 {len(env_files)} 个环境声文件")
            
    except Exception as e:
        print(f"[警告] 加载增强数据时出错: {e}")
    
    total_files = sum(len(files) for files in augment_data.values())
    if total_files > 0:
        print(f"总共加载了 {total_files} 个增强文件")
    else:
        print("未加载任何增强文件，将跳过音频增强")
    
    return augment_data


def apply_reverb(waveform: torch.Tensor, rir_files: List[str], target_sr: int) -> torch.Tensor:
    """
    应用混响增强
    
    Args:
        waveform: 输入音频 (1, N)
        rir_files: RIR文件列表
        target_sr: 目标采样率
    
    Returns:
        混响后的音频
    """
    if not rir_files:
        return waveform
    
    try:
        rir_file = random.choice(rir_files)
        rir_waveform, rir_sr = torchaudio.load(rir_file)
        
        # 重采样RIR到目标采样率
        if rir_sr != target_sr:
            resampler = T.Resample(rir_sr, target_sr)
            rir_waveform = resampler(rir_waveform)
        
        # 转为单声道
        if rir_waveform.size(0) > 1:
            rir_waveform = rir_waveform.mean(dim=0, keepdim=True)
        
        # 应用卷积混响
        original_length = waveform.size(1)
        reverb_audio = AF.fftconvolve(waveform, rir_waveform, mode='full')
        
        # 裁剪到原始长度
        reverb_audio = reverb_audio[:, :original_length]
        
        return reverb_audio
    except Exception as e:
        print(f"[混响增强错误] {e}")
        return waveform


def apply_additive_noise(waveform: torch.Tensor, noise_files: List[str], 
                        target_sr: int, snr_range: tuple = (5, 20)) -> torch.Tensor:
    """
    应用加性噪声增强
    
    Args:
        waveform: 输入音频 (1, N)
        noise_files: 噪声文件列表
        target_sr: 目标采样率
        snr_range: SNR范围(dB)
    
    Returns:
        加噪后的音频
    """
    if not noise_files:
        return waveform
    
    try:
        noise_file = random.choice(noise_files)
        noise_waveform, noise_sr = torchaudio.load(noise_file)
        
        # 重采样噪声到目标采样率
        if noise_sr != target_sr:
            resampler = T.Resample(noise_sr, target_sr)
            noise_waveform = resampler(noise_waveform)
        
        # 转为单声道
        if noise_waveform.size(0) > 1:
            noise_waveform = noise_waveform.mean(dim=0, keepdim=True)
        
        # 调整噪声长度
        noise_waveform = adjust_length(noise_waveform, waveform.size(1))
        
        # 随机SNR
        snr_db = random.uniform(*snr_range)
        
        # 添加噪声
        noisy_audio = add_noise_at_snr(waveform, noise_waveform, snr_db)
        
        return noisy_audio
    except Exception as e:
        print(f"[噪声增强错误] {e}")
        return waveform


# ---------- util: down-sample to target_sr ----------
def normalize_audio(waveform: torch.Tensor, method: str = "rms") -> torch.Tensor:
    """
    音频能量归一化，适配wav2vec2模型
    
    Args:
        waveform: 音频波形 (1, N)
        method: 归一化方法 ("rms", "peak", "zscore")
    
    Returns:
        归一化后的音频波形
    """
    if method == "rms":
        # RMS归一化 - 推荐用于wav2vec2
        rms = torch.sqrt(torch.mean(waveform ** 2))
        if rms > 1e-8:  # 避免除零
            waveform = waveform / (rms + 1e-8)
            # 限制幅度范围到[-1, 1]
            waveform = torch.clamp(waveform, -1.0, 1.0)
    elif method == "peak":
        # 峰值归一化
        peak = torch.max(torch.abs(waveform))
        if peak > 1e-8:
            waveform = waveform / (peak + 1e-8)
    elif method == "zscore":
        # Z-score标准化
        mean = torch.mean(waveform)
        std = torch.std(waveform)
        if std > 1e-8:
            waveform = (waveform - mean) / (std + 1e-8)
    
    return waveform


def load_wav_resample(path: str, target_sr: int, normalize: bool = True) -> torch.Tensor:
    """加载并重采样音频文件"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"音频文件不存在: {path}")
    
    waveform, sr = torchaudio.load(path)           # (C, N)
    if sr != target_sr:
        resampler = T.Resample(sr, target_sr)
        waveform = resampler(waveform)
    # 若多声道，取平均 → 单声道
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # 音频归一化 - 对wav2vec2很重要
    # if normalize:
    #     waveform = normalize_audio(waveform, method="rms")
    
    return waveform                                 # (1, N')


# ---------- Dataset ----------
class DeepFakeWavDataset(Dataset):
    """
    读取 wav（自动重采样到 16 kHz），返回:
        {"utt_id": str,
         "waveform": Tensor [1, N],
         "label": int}   # Bonafide→0, Spoof→1
    """
    def __init__(self, list_file: str, audio_root: str, params,
                 max_duration: float = 4, normalize_audio: bool = True) -> None:
        super().__init__()
        self.params = params
        self.target_sr = params.target_sr
        self.audio_root = Path(audio_root)
        self.max_duration = max_duration
        self.normalize_audio = normalize_audio
        
        # 从params中读取增强配置
        self.augment_prob = getattr(params, 'augment_prob', 0.2)
        self.snr_range = getattr(params, 'snr_range', (5, 20))
        
        # self.num_samples = int(target_sr * max_duration)  # 4秒 = 64000 samples at 16kHz
        self.num_samples = 64600
        
        # 从params加载增强数据
        self.augment_data = load_augmentation_data_from_params(params)

        # 验证文件存在
        if not os.path.exists(list_file):
            raise FileNotFoundError(f"数据列表文件不存在: {list_file}")
        if not os.path.exists(audio_root):
            raise FileNotFoundError(f"音频根目录不存在: {audio_root}")

        self.data_entries = []
        self._load_data_entries(list_file)
        
        if len(self.data_entries) == 0:
            raise ValueError(f"没有找到有效的数据条目在: {list_file}")
        
        random.shuffle(self.data_entries)
        print(f"成功加载 {len(self.data_entries)} 个音频条目")
        print(f"标签分布: Bonafide={sum(1 for e in self.data_entries if e['label']==0)}, "
              f"Spoof={sum(1 for e in self.data_entries if e['label']==1)}")

    def _load_data_entries(self, list_file: str) -> None:
        """加载数据条目"""
        valid_labels = {"bonafide", "spoof"}
        skipped_count = 0
        
        with open(list_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # 跳过空行
                    continue
                    
                parts = line.split(",")
                if len(parts) < 3:
                    print(f"[警告] 第{line_num}行格式错误，跳过: {line}")
                    skipped_count += 1
                    continue
                
                utt_id, fname, label_str = parts[:3]
                label_str_lower = label_str.lower()
                
                # 验证标签
                if label_str_lower not in valid_labels:
                    print(f"[警告] 第{line_num}行标签无效 '{label_str}'，跳过: {line}")
                    skipped_count += 1
                    continue
                
                # 构建完整路径
                full_path = self.audio_root / fname
                
                # 验证文件是否存在
                if not full_path.exists():
                    print(f"[警告] 第{line_num}行音频文件不存在，跳过: {full_path}")
                    skipped_count += 1
                    continue
                
                label = 0 if label_str_lower == "bonafide" else 1
                self.data_entries.append({
                    "utt_id": utt_id,
                    "path": str(full_path),
                    "label": label
                })
        
        if skipped_count > 0:
            print(f"跳过了 {skipped_count} 个无效条目")

    # ---------- pytorch Dataset API ----------
    def __len__(self):
        return len(self.data_entries)

    def __getitem__(self, idx: int) -> Optional[Dict]:
        entry = self.data_entries[idx]
        try:
            wav = load_wav_resample(entry["path"], self.target_sr, self.normalize_audio)  # (1, N)
            wav_len = wav.size(1)
            
            # 处理音频长度
            if wav_len > self.num_samples:
                # 随机截取一段
                start = random.randint(0, wav_len - self.num_samples)
                wav = wav[:, start:start + self.num_samples]
            elif wav_len < self.num_samples:
                # 重复音频直到达到目标长度
                repeat_times = (self.num_samples + wav_len - 1) // wav_len
                wav = wav.repeat(1, repeat_times)
                wav = wav[:, :self.num_samples]
            # else: wav_len == self.num_samples, 无需处理
            
            # 音频增强（训练时应用）
            if random.random() < self.augment_prob:
                wav = self._apply_augmentation(wav)
            
            return {
                "utt_id": entry["utt_id"],
                "waveform": wav,
                "label": entry["label"]
            }
        except Exception as e:
            print(f"[数据集错误] {entry['path']} - {e}")
            # 返回一个默认样本而不是None，避免batch处理问题
            return self._get_default_sample(entry["utt_id"], entry["label"])

    def _apply_augmentation(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        应用音频增强
        
        Args:
            waveform: 输入音频 (1, N)
        
        Returns:
            增强后的音频
        """
        # 随机选择增强类型
        aug_types = []
        if self.augment_data['rir_files']:
            aug_types.append("reverb")
        if (self.augment_data['noise_files'] or 
            self.augment_data['music_files'] or 
            self.augment_data['speech_files']):
            aug_types.append("noise")
        if self.augment_data['env_files']:
            aug_types.append("env")
        
        if not aug_types:
            return waveform
        
        aug_type = random.choice(aug_types)
        
        try:
            if aug_type == "reverb":
                # 应用混响
                waveform = apply_reverb(waveform, self.augment_data['rir_files'], self.target_sr)
                
            elif aug_type == "noise":
                # 随机选择噪声类型
                noise_categories = []
                if self.augment_data['noise_files']:
                    noise_categories.append('noise_files')
                if self.augment_data['music_files']:
                    noise_categories.append('music_files')
                if self.augment_data['speech_files']:
                    noise_categories.append('speech_files')
                
                if noise_categories:
                    category = random.choice(noise_categories)
                    noise_files = self.augment_data[category]
                    waveform = apply_additive_noise(waveform, noise_files, 
                                                  self.target_sr, self.snr_range)
                    
            elif aug_type == "env":
                # 应用环境声增强
                waveform = apply_additive_noise(waveform, self.augment_data['env_files'], 
                                              self.target_sr, self.snr_range)
            
            # 防止溢出，限制幅度
            peak = torch.max(torch.abs(waveform))
            if peak > 1.0:
                waveform = waveform / peak
                
        except Exception as e:
            print(f"[增强错误] {e}")
            # 如果增强失败，返回原始音频
            pass
        
        return waveform

    def _get_default_sample(self, utt_id: str, label: int) -> Dict:
        """返回默认样本（静音）"""
        default_wav = torch.zeros(1, self.num_samples)
        return {
            "utt_id": utt_id,
            "waveform": default_wav,
            "label": label
        }

    # ---------- collate_fn ----------
    def collate(self, batch: List[Dict]) -> Dict:
        batch = [b for b in batch if b is not None]
        if not batch:
            return None

        utt_ids = [b["utt_id"] for b in batch]
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        waveforms = [b["waveform"].squeeze(0) for b in batch]   # list[(N,)]

        # 所有音频现在都是相同长度，直接堆叠
        waveforms_tensor = torch.stack(waveforms)  # (B, N)
        lengths = torch.full((len(batch),), self.num_samples, dtype=torch.long)

        return {
            "utt_ids": utt_ids,
            "waveforms": waveforms_tensor,
            "labels": labels,
            "lengths": lengths,
        }


# ---------- factory for DataLoader ----------
def from_train_list(list_file: str, audio_root: str, params,
                    is_distributed=False, max_duration: float = 4, 
                    normalize_audio: bool = True):
    """
    创建训练数据加载器
    
    Args:
        list_file: 数据列表文件路径 (train.txt / test.txt)
        audio_root: 音频文件根目录
        params: 包含所有配置参数的对象（包括增强配置）
        is_distributed: 是否使用分布式训练
        max_duration: 音频最大时长（秒）
        normalize_audio: 是否进行音频归一化（推荐用于wav2vec2）
    """
    ds = DeepFakeWavDataset(
        list_file,
        audio_root,
        params,
        max_duration=max_duration,
        normalize_audio=normalize_audio
    )

    return DataLoader(
        ds,
        batch_size=params.batch_size,
        shuffle=not is_distributed,
        sampler=DistributedSampler(ds) if is_distributed else None,
        collate_fn=ds.collate,
        pin_memory=True,
        persistent_workers=True,
        num_workers=min(8, cpu_count()),  # 减少worker数量
        drop_last=True
    )


# ---------- Test Dataset for inference ----------
class TestDataset(Dataset):
    """
    测试数据集，处理只有两列（utt_id, path）的测试文件
    返回格式与训练数据集相同，但label为占位符
    """
    def __init__(self, list_file: str, audio_root: str, params,
                 max_duration: float = 4, normalize_audio: bool = True) -> None:
        super().__init__()
        self.params = params
        self.target_sr = params.target_sr
        self.audio_root = Path(audio_root)
        self.max_duration = max_duration
        self.normalize_audio = normalize_audio
        self.num_samples = 64600

        # 验证文件存在
        if not os.path.exists(list_file):
            raise FileNotFoundError(f"测试文件不存在: {list_file}")
        if not os.path.exists(audio_root):
            raise FileNotFoundError(f"音频根目录不存在: {audio_root}")

        self.data_entries = []
        self._load_data_entries(list_file)
        
        if len(self.data_entries) == 0:
            raise ValueError(f"没有找到有效的测试条目在: {list_file}")
        
        print(f"成功加载 {len(self.data_entries)} 个测试音频条目")

    def _load_data_entries(self, list_file: str) -> None:
        """加载测试数据条目（只有两列：utt_id, path）"""
        skipped_count = 0
        
        with open(list_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # 跳过空行
                    continue
                    
                parts = line.split(",")
                if len(parts) < 2:
                    print(f"[警告] 第{line_num}行格式错误，跳过: {line}")
                    skipped_count += 1
                    continue
                
                utt_id, fname = parts[:2]
                
                # 构建完整路径
                full_path = self.audio_root / fname
                
                # 验证文件是否存在
                if not full_path.exists():
                    print(f"[警告] 第{line_num}行音频文件不存在，跳过: {full_path}")
                    skipped_count += 1
                    continue
                
                # 测试数据没有标签，设为0（占位符）
                self.data_entries.append({
                    "utt_id": utt_id,
                    "path": str(full_path),
                    "label": 0  # 占位符，实际不会使用
                })
        
        if skipped_count > 0:
            print(f"跳过了 {skipped_count} 个无效条目")

    def __len__(self):
        return len(self.data_entries)

    def __getitem__(self, idx: int) -> Optional[Dict]:
        entry = self.data_entries[idx]
        try:
            wav = load_wav_resample(entry["path"], self.target_sr, self.normalize_audio)  # (1, N)
            wav_len = wav.size(1)
            
            # 处理音频长度
            if wav_len > self.num_samples:
                #start = torch.randint(0, wav_len-self.num_samples, (1,)).item()
                #wav   = wav[:, start:start+self.num_samples]
                wav = wav[:, :self.num_samples]
            elif wav_len < self.num_samples:
                # 重复音频直到达到目标长度
                repeat_times = (self.num_samples + wav_len - 1) // wav_len
                wav = wav.repeat(1, repeat_times)
                wav = wav[:, :self.num_samples]
            else: wav_len == self.num_samples
            
            return {
                "utt_id": entry["utt_id"],
                "waveform": wav,
                "label": entry["label"]  # 占位符
            }
        except Exception as e:
            print(f"[测试数据集错误] {entry['path']} - {e}")
            # 返回一个默认样本而不是None，避免batch处理问题
            return self._get_default_sample(entry["utt_id"], entry["label"])

    def _get_default_sample(self, utt_id: str, label: int) -> Dict:
        """返回默认样本（静音）"""
        default_wav = torch.zeros(1, self.num_samples)
        return {
            "utt_id": utt_id,
            "waveform": default_wav,
            "label": label
        }

    def collate(self, batch: List[Dict]) -> Dict:
        batch = [b for b in batch if b is not None]
        if not batch:
            return None

        utt_ids = [b["utt_id"] for b in batch]
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        waveforms = [b["waveform"].squeeze(0) for b in batch]   # list[(N,)]

        # 所有音频现在都是相同长度，直接堆叠
        waveforms_tensor = torch.stack(waveforms)  # (B, N)
        lengths = torch.full((len(batch),), self.num_samples, dtype=torch.long)

        return {
            "utt_ids": utt_ids,
            "waveforms": waveforms_tensor,
            "labels": labels,
            "lengths": lengths,
        }


def from_test_list(test_file: str, audio_root: str, params,
                   batch_size: int = 16, normalize_audio: bool = True):
    """
    创建测试数据加载器
    
    Args:
        test_file: 测试文件路径 (utt_id,path format)
        audio_root: 音频文件根目录
        params: 包含所有配置参数的对象
        batch_size: 批次大小
        normalize_audio: 是否进行音频归一化（推荐用于wav2vec2）
    """
    ds = TestDataset(
        test_file,
        audio_root,
        params,
        max_duration=4,
        normalize_audio=normalize_audio
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,  # 测试时不打乱
        collate_fn=ds.collate,
        pin_memory=True,
        persistent_workers=True,
        num_workers=min(8, cpu_count()),
        drop_last=False  # 测试时保留所有样本
    )


# ---------- quick test ----------
if __name__ == "__main__":
    class _P:                      # mock params
        batch_size = 4
        target_sr  = 16000
        # 增强配置
        augment_prob = 0.0         # 测试时禁用增强
        snr_range = (5, 20)
        rir_base_path = None
        musan_noise_base_path = None
        musan_music_base_path = None
        musan_speech_base_path = None
        env_base_path = None
        
    params = params
    # train dataset
    train_txt = "./data/train.txt"
    train_root = "./data/"  # 假设音频文件在当前目录

    try:
        loader = from_train_list(train_txt, train_root, params)
        
        for batch in loader:
            print(f"波形形状: {batch['waveforms'].shape}")  # (B, N)
            print(f"标签: {batch['labels']}")           # tensor([...])
            print(f"长度: {batch['lengths']}")          # tensor([...])
            print(f"样本ID: {batch['utt_ids']}")       # list[...]
            break
    except Exception as e:
        print(f"测试失败: {e}")


    # test dataset
    test_txt = "./data/test.txt"
    test_root = "./data/"
    try:
        loader = from_test_list(test_txt, test_root, params)

        for batch in loader:
            print(f"波形形状: {batch['waveforms'].shape}")  # (B, N)
            print(f"标签: {batch['labels']}")           # tensor([...])
            print(f"长度: {batch['lengths']}")          # tensor([...])
            print(f"样本ID: {batch['utt_ids']}")       # list[...]
            break
    except Exception as e:
        print(f"测试失败: {e}")



# ---------- 音频增强使用示例 ----------
"""
使用音频增强的示例：

# 1. 在params.py中配置增强数据路径
params = AttrDict(
    # ... 其他参数 ...
    
    # Audio Augmentation params
    augment_prob=0.3,           # 30%概率应用增强
    snr_range=(5, 20),          # SNR范围5-20dB
    
    # Augmentation data base paths
    rir_base_path='/path/to/rir',                    # RIR混响文件基础路径
    musan_noise_base_path='/path/to/musan/noise',    # MUSAN噪声基础路径
    musan_music_base_path='/path/to/musan/music',    # MUSAN音乐基础路径
    musan_speech_base_path='/path/to/musan/speech',  # MUSAN语音基础路径
    env_base_path='/path/to/esc50',                  # ESC-50环境声基础路径
)

# 2. 创建带增强的训练数据加载器（自动从params读取增强配置）
train_loader = from_train_list(
    list_file="./data/train.txt",
    audio_root="./data/",
    params=params
)

# 3. 如果没有增强数据，在params中设置augment_prob=0.0即可禁用增强
params.augment_prob = 0.0
"""