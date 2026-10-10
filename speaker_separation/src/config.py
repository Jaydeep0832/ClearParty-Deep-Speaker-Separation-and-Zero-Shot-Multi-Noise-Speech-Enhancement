"""
config.py — High-Efficiency GPU Configuration
"""
import torch
import os

class Config:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4             # 4 parallel workers for fast audio streaming

    CHECKPOINT_DIR = '/content/drive/MyDrive/cocktail_party_project/checkpoints_v2'

    # Audio
    SAMPLE_RATE = 8000
    CHUNK_DURATION = 4.0
    CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)  # 32,000 samples
    NUM_SPEAKERS = 2

    # Architecture (Paper settings)
    ENC_DIM = 256               # N
    ENC_KERNEL = 20             # L
    ENC_STRIDE = 10             # L/2
    BN_DIM = 256                # B
    HIDDEN_DIM = 512            # H
    SC_DIM = 256                # Sc
    CONV_KERNEL = 3             # P
    NUM_BLOCKS = 8              # X
    NUM_REPEATS = 3             # R
    NUM_SPEAKERS_OUT = 2        # C
    MASK_ACT = 'relu'

    # GPU-Optimized Training
    BATCH_SIZE = 12             # 3x larger batch to fill GPU VRAM
    USE_AMP = True              # FP16 Automatic Mixed Precision (Tensor Cores)
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 0
    NUM_EPOCHS = 100
    GRAD_CLIP = 5.0
    SCHEDULER_PATIENCE = 3

    PRINT_EVERY = 100           # Print every 100 batches

print("✅ config.py updated: Batch Size = 12 | Workers = 4 | FP16 AMP = Enabled")
