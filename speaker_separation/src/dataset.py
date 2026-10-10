"""
dataset.py — Libri2Mix Dataset with max_samples support
"""
import os
import random
import torch
import soundfile as sf
import numpy as np
from torch.utils.data import Dataset
from config import Config


class Libri2MixDataset(Dataset):
    def __init__(self, split='train-100', max_samples=None):
        self.split = split
        self.chunk_samples = Config.CHUNK_SAMPLES

        if split == 'train-100':
            split_dir = '/content/Libri2Mix/wav8k/min/train-100'
        elif split == 'dev':
            split_dir = '/content/drive/MyDrive/cocktail_party_project/data/libri2mix/Libri2Mix/wav8k/min/dev'
        elif split == 'test':
            split_dir = '/content/drive/MyDrive/cocktail_party_project/data/libri2mix/Libri2Mix/wav8k/min/test'
        else:
            raise ValueError(f"Unknown split: {split}")

        self.mix_dir = os.path.join(split_dir, 'mix_clean')
        self.s1_dir  = os.path.join(split_dir, 's1')
        self.s2_dir  = os.path.join(split_dir, 's2')

        all_files = sorted([f for f in os.listdir(self.mix_dir) if f.endswith('.wav')])

        # Shorten if requested (e.g. 300 for fast validation)
        if max_samples is not None and max_samples < len(all_files):
            # Take evenly spaced samples across the set for an unbiased representation
            step = len(all_files) // max_samples
            self.filenames = all_files[::step][:max_samples]
        else:
            self.filenames = all_files

        print(f"  📂 {split.upper()}: {len(self.filenames)} pairs from {split_dir}")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]

        mixture, _ = sf.read(os.path.join(self.mix_dir, fname), dtype='float32')
        s1, _      = sf.read(os.path.join(self.s1_dir, fname), dtype='float32')
        s2, _      = sf.read(os.path.join(self.s2_dir, fname), dtype='float32')

        min_len = min(len(mixture), len(s1), len(s2))
        mixture, s1, s2 = mixture[:min_len], s1[:min_len], s2[:min_len]

        mixture, s1, s2 = self._chunk(mixture, s1, s2)

        max_val = np.max(np.abs(mixture)) + 1e-8
        mixture /= max_val
        s1 /= max_val
        s2 /= max_val

        mixture = torch.from_numpy(mixture).float()
        sources = torch.stack([
            torch.from_numpy(s1).float(),
            torch.from_numpy(s2).float()
        ])
        return mixture, sources

    def _chunk(self, mixture, s1, s2):
        length = len(mixture)
        if length >= self.chunk_samples:
            if 'train' in self.split:
                start = random.randint(0, length - self.chunk_samples)
            else:
                start = (length - self.chunk_samples) // 2
            end = start + self.chunk_samples
            return mixture[start:end], s1[start:end], s2[start:end]
        else:
            pad = self.chunk_samples - length
            return np.pad(mixture, (0, pad)), np.pad(s1, (0, pad)), np.pad(s2, (0, pad))


print("✅ dataset.py updated with max_samples support")
