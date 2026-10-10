"""
train.py — GPU-Accelerated Training Pipeline for Speech Denoising
"""
import os
import time
import argparse
import torch
from torch.utils.data import DataLoader
from src.config import Config
from src.dataset import NOIZEUSEnhancementDataset
from src.model import ConvTasNetDenoiser
from src.utils import denoising_loss


def train(args):
    print("=" * 65)
    print("🚀 Conv-TasNet Speech Enhancement Training Pipeline")
    print(f"   Device: {Config.DEVICE} | Precision: {'FP16 (AMP)' if Config.USE_AMP else 'FP32'}")
    print(f"   Target: Single Clean Speech Extraction (C=1)")
    print("=" * 65)

    # 1. Dataset & DataLoaders
    train_ds = NOIZEUSEnhancementDataset(data_root=args.data_dir, split='train', num_samples_per_epoch=args.samples_per_epoch)
    val_ds   = NOIZEUSEnhancementDataset(data_root=args.data_dir, split='val')

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=Config.NUM_WORKERS, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=Config.NUM_WORKERS, pin_memory=True
    )

    # 2. Model, Optimizer, Scheduler
    model = ConvTasNetDenoiser().to(Config.DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=Config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=Config.SCHEDULER_PATIENCE, factor=Config.SCHEDULER_FACTOR
    )
    scaler = torch.amp.GradScaler('cuda', enabled=Config.USE_AMP)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_ckpt_path = os.path.join(args.checkpoint_dir, 'denoiser_best.pth')

    best_val_sisdr = float('-inf')

    print(f"Batches per Epoch: {len(train_loader)} | Batch Size: {args.batch_size}")
    print("=" * 65)

    # 3. Training Loop
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_sisdr, n = 0, 0

        for batch_idx, (mix, clean) in enumerate(train_loader):
            mix = mix.to(Config.DEVICE, non_blocking=True)
            clean = clean.to(Config.DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=Config.USE_AMP):
                est = model(mix)
                loss, sisdr_val = denoising_loss(est, clean, l1_weight=0.1)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()

            train_sisdr += sisdr_val
            n += 1

            if (batch_idx + 1) % Config.PRINT_EVERY == 0:
                print(f"  Epoch {epoch:03d} | Batch {batch_idx+1}/{len(train_loader)} | SI-SDR: {train_sisdr/n:+.2f} dB")

        train_sisdr /= n

        # 4. Validation on Held-Out Sentences
        model.eval()
        val_sisdr, vn = 0, 0
        with torch.no_grad():
            for mix, clean in val_loader:
                mix = mix.to(Config.DEVICE, non_blocking=True)
                clean = clean.to(Config.DEVICE, non_blocking=True)
                with torch.amp.autocast('cuda', enabled=Config.USE_AMP):
                    est = model(mix)
                    _, sisdr_val = denoising_loss(est, clean, l1_weight=0.1)
                val_sisdr += sisdr_val
                vn += 1
        val_sisdr /= vn

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch:03d}/{args.epochs}] ({elapsed:.0f}s) | Train: {train_sisdr:+.2f} dB | Val: {val_sisdr:+.2f} dB | LR: {lr:.1e}")

        scheduler.step(val_sisdr)

        if val_sisdr > best_val_sisdr:
            best_val_sisdr = val_sisdr
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_sisdr': val_sisdr,
                'train_sisdr': train_sisdr,
            }, best_ckpt_path)
            print(f"  💾 Best model saved! (Val SI-SDR: {val_sisdr:+.2f} dB)")

    print("\n" + "=" * 65)
    print(f"🏁 Training Complete! Best Validation SI-SDR: {best_val_sisdr:+.2f} dB")
    print(f"   Checkpoint: {best_ckpt_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Conv-TasNet Denoiser on NOIZEUS")
    parser.add_argument('--data_dir', type=str, default=Config.DATA_DIR, help="Path to extracted NOIZEUS folder")
    parser.add_argument('--checkpoint_dir', type=str, default=Config.CHECKPOINT_DIR, help="Checkpoint directory")
    parser.add_argument('--epochs', type=int, default=Config.NUM_EPOCHS, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=Config.BATCH_SIZE, help="Batch size")
    parser.add_argument('--lr', type=float, default=Config.LEARNING_RATE, help="Learning rate")
    parser.add_argument('--samples_per_epoch', type=int, default=1000, help="Number of dynamic training samples per epoch")
    args = parser.parse_args()
    train(args)
