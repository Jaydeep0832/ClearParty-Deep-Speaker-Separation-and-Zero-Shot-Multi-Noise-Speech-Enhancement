"""
dataset.py — NOIZEUS Speech Enhancement Dataset
Supports:
  1. Single-Noise Dynamic Synthesis (Training)
  2. Single-Noise Benchmark (Validation / Testing)
  3. Compound Multi-Noise Stress Testing (2 or 3 simultaneous noises)
"""
import os
import glob
import random
import torch
import soundfile as sf
import numpy as np
from torch.utils.data import Dataset
from src.config import Config


class NOIZEUSEnhancementDataset(Dataset):
    def __init__(self, data_root=None, split='train', num_samples_per_epoch=1000, multi_noise_count=2):
        """
        Args:
            data_root: Path to extracted NOIZEUS folder (contains clean/ and isolated_noise/)
            split: 'train' (single noise), 'val' (single noise), 'test_single', or 'test_multi' (2+ noises)
            num_samples_per_epoch: Virtual dataset length for training permutations
            multi_noise_count: Number of simultaneous noises to mix in 'test_multi' mode (e.g. 2 or 3)
        """
        self.root = data_root or Config.DATA_DIR
        self.split = split
        self.num_samples = num_samples_per_epoch
        self.multi_noise_count = multi_noise_count
        self.chunk_samples = Config.CHUNK_SAMPLES

        clean_dir = os.path.join(self.root, 'clean')
        all_clean_files = sorted(glob.glob(os.path.join(clean_dir, '*.wav')))

        if len(all_clean_files) == 0:
            raise FileNotFoundError(f"No clean files found in {clean_dir}")

        # Partition 30 IEEE sentences:
        # Train: sp01 - sp24 (24 sentences, 80%)
        # Test / Val: sp25 - sp30 (6 sentences, 20% held-out)
        if split == 'train':
            self.clean_files = all_clean_files[:24]
        else:
            self.clean_files = all_clean_files[24:]

        # Index all isolated noise files
        noise_dir = os.path.join(self.root, 'isolated_noise')
        self.noise_files = sorted(glob.glob(os.path.join(noise_dir, '*', '*.wav')))
        
        # Group noises by noise type (airport, babble, car, etc.)
        self.noises_by_type = {}
        for nf in self.noise_files:
            # folder is e.g. "car_0dB" -> noise_type is "car"
            folder_name = os.path.basename(os.path.dirname(nf))
            ntype = folder_name.rsplit('_', 1)[0]
            if ntype not in self.noises_by_type:
                self.noises_by_type[ntype] = []
            self.noises_by_type[ntype].append(nf)

        self.noise_types = sorted(list(self.noises_by_type.keys()))

    def __len__(self):
        if self.split == 'train':
            return self.num_samples
        elif self.split in ['val', 'test_single']:
            return len(self.clean_files) * len(self.noise_types)  # 6 sentences * 8 noises = 48 pairs
        else:  # 'test_multi'
            return len(self.clean_files) * 10  # 60 compound test mixtures

    def _load_audio(self, path):
        audio, _ = sf.read(path, dtype='float32')
        return audio

    def _scale_noise_to_snr(self, clean, noise, target_snr_db):
        p_clean = np.mean(clean ** 2) + 1e-8
        p_noise = np.mean(noise ** 2) + 1e-8
        desired_p_noise = p_clean / (10 ** (target_snr_db / 10.0))
        scale = np.sqrt(desired_p_noise / p_noise)
        return noise * scale

    def _pad_or_chunk(self, audio, length):
        if len(audio) >= length:
            if self.split == 'train':
                start = random.randint(0, len(audio) - length)
            else:
                start = (len(audio) - length) // 2
            return audio[start:start + length]
        else:
            pad = length - len(audio)
            return np.pad(audio, (0, pad))

    def _get_noise_clip(self, noise_path, target_len):
        noise = self._load_audio(noise_path)
        if len(noise) < target_len:
            repeats = int(np.ceil(target_len / len(noise)))
            noise = np.tile(noise, repeats)
        start = random.randint(0, len(noise) - target_len)
        return noise[start:start + target_len]

    def __getitem__(self, idx):
        # 1. Pick Clean Sentence
        if self.split == 'train':
            clean_path = random.choice(self.clean_files)
        else:
            clean_idx = idx % len(self.clean_files)
            clean_path = self.clean_files[clean_idx]

        clean = self._load_audio(clean_path)
        target_len = self.chunk_samples
        clean = self._pad_or_chunk(clean, target_len)

        # 2. Noise Mixing
        if self.split in ['train', 'val', 'test_single']:
            # ─── SINGLE NOISE MIXING ───
            if self.split == 'train':
                snr = random.uniform(Config.TRAIN_SNR_MIN, Config.TRAIN_SNR_MAX)
                ntype = random.choice(self.noise_types)
            else:
                noise_idx = (idx // len(self.clean_files)) % len(self.noise_types)
                ntype = self.noise_types[noise_idx]
                snr = 5.0  # Fixed standard evaluation SNR

            noise_path = random.choice(self.noises_by_type[ntype])
            noise = self._get_noise_clip(noise_path, target_len)
            noise_scaled = self._scale_noise_to_snr(clean, noise, snr)
            mixture = clean + noise_scaled
            noise_info = ntype

        else:
            # ─── COMPOUND MULTI-NOISE STRESS TESTING ───
            # Pick 2 or 3 distinct simultaneous noises!
            selected_types = random.sample(self.noise_types, k=min(self.multi_noise_count, len(self.noise_types)))
            combined_noise = np.zeros_like(clean)

            # Each noise contributes at an individual SNR (e.g. 5 dB, 10 dB)
            for ntype in selected_types:
                noise_path = random.choice(self.noises_by_type[ntype])
                clip = self._get_noise_clip(noise_path, target_len)
                snr_i = random.choice([5.0, 10.0])
                scaled_clip = self._scale_noise_to_snr(clean, clip, snr_i)
                combined_noise += scaled_clip

            mixture = clean + combined_noise
            noise_info = " + ".join(selected_types)

        # 3. Peak Normalization to 0.90 (Prevents digital clipping)
        max_val = np.max(np.abs(mixture)) + 1e-8
        mixture = (mixture / max_val) * 0.90
        clean = (clean / max_val) * 0.90

        mix_tensor = torch.from_numpy(mixture).float()
        clean_tensor = torch.from_numpy(clean).float()

        if self.split == 'test_multi':
            return mix_tensor, clean_tensor, noise_info
        return mix_tensor, clean_tensor
