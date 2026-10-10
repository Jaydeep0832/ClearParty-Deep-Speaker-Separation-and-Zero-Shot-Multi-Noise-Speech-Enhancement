"""
config.py — Configuration for Speech Enhancement (Denoising) on NOIZEUS
"""
import torch
import os

class Config:
    # Hardware & Precision
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 2
    USE_AMP = True                      # FP16 Automatic Mixed Precision

    # Storage Paths
    DATA_DIR = "data/noizeus"
    CHECKPOINT_DIR = "checkpoints_denoising"

    # Audio Specifications
    SAMPLE_RATE = 8000
    CHUNK_DURATION = 3.0                # 3.0 seconds chunks
    CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)  # 24,000 samples
    NUM_OUTPUT_CHANNELS = 1             # C=1 (Target: Single Clean Speech)

    # Conv-TasNet Denoising Architecture
    ENC_DIM = 256                       # N: Filter bank dimensions
    ENC_KERNEL = 20                     # L: Kernel size (2.5 ms @ 8 kHz)
    ENC_STRIDE = 10                     # S: Stride (50% overlap)
    BN_DIM = 256                        # B: Bottleneck dimension
    HIDDEN_DIM = 512                    # H: Depthwise convolution channels
    SC_DIM = 256                        # Sc: Skip-connection channels
    CONV_KERNEL = 3                     # P: Dilated kernel size
    NUM_BLOCKS = 8                      # X: 8 dilated convolution blocks per repeat
    NUM_REPEATS = 3                     # R: 3 repeats (Total: 24 blocks)
    MASK_ACT = 'sigmoid'                # Sigmoid mask for energy attenuation [0, 1]

    # Training & Optimization
    BATCH_SIZE = 16                     # Batch size for T4 GPU
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5
    NUM_EPOCHS = 50
    GRAD_CLIP = 5.0
    SCHEDULER_PATIENCE = 3
    SCHEDULER_FACTOR = 0.5
    PRINT_EVERY = 20

    # Noise Simulation Ranges
    TRAIN_SNR_MIN = 0.0                 # dB
    TRAIN_SNR_MAX = 15.0                # dB
