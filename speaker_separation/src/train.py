import os, time, torch
from torch.utils.data import DataLoader
from config import Config
from dataset import Libri2MixDataset
from model import ConvTasNet
from utils import pit_loss


def train():
    print(f"Device: {Config.DEVICE} ({torch.cuda.get_device_name(0)})")
    print(f"CUDA Capability: {torch.cuda.get_device_capability(0)}")
    print("=" * 60)

    print("Loading datasets...")
    train_ds = Libri2MixDataset(split='train-100')
    val_ds   = Libri2MixDataset(split='dev', max_samples=300)

    train_loader = DataLoader(
        train_ds, batch_size=Config.BATCH_SIZE,
        shuffle=True, num_workers=Config.NUM_WORKERS,
        pin_memory=True, drop_last=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=Config.BATCH_SIZE,
        shuffle=False, num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )

    model = ConvTasNet().to(Config.DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: Conv-TasNet | Parameters: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=Config.SCHEDULER_PATIENCE, factor=0.5)

    # Mixed Precision Scaler for T4 Tensor Cores
    scaler = torch.amp.GradScaler('cuda', enabled=Config.USE_AMP)

    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    ckpt_path = os.path.join(Config.CHECKPOINT_DIR, 'tasnet_v2_best.pth')
    start_epoch = 1
    best_val = float('-inf')

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=Config.DEVICE)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_val = ckpt['val_sisdr']
        print(f"  ✅ Resumed from Epoch {ckpt['epoch']} (Best: {best_val:+.2f} dB)")

    print("=" * 60)
    print(f"  Train: {len(train_ds)} pairs | Val: {len(val_ds)} pairs")
    print(f"  Batches/epoch: {len(train_loader)} | Batch size: {Config.BATCH_SIZE}")
    print(f"  Mixed Precision (AMP): {Config.USE_AMP}")
    print("=" * 60)

    for epoch in range(start_epoch, Config.NUM_EPOCHS + 1):
        t0 = time.time()

        # TRAIN
        model.train()
        train_loss, train_sisdr, n = 0, 0, 0
        for batch_idx, (mix, src) in enumerate(train_loader):
            mix = mix.to(Config.DEVICE, non_blocking=True)
            src = src.to(Config.DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # Accelerated forward pass with AMP
            with torch.amp.autocast('cuda', enabled=Config.USE_AMP):
                est = model(mix)
                loss, sisdr = pit_loss(est, src)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            train_sisdr += sisdr
            n += 1

            if (batch_idx + 1) % Config.PRINT_EVERY == 0:
                print(f"  Epoch {epoch:03d} | Batch {batch_idx+1}/{len(train_loader)} | "
                      f"SI-SDR: {train_sisdr/n:+.2f} dB")

        train_sisdr /= n

        # VALIDATION (300 samples)
        model.eval()
        val_sisdr, vn = 0, 0
        with torch.no_grad():
            for mix, src in val_loader:
                mix = mix.to(Config.DEVICE, non_blocking=True)
                src = src.to(Config.DEVICE, non_blocking=True)
                with torch.amp.autocast('cuda', enabled=Config.USE_AMP):
                    est = model(mix)
                    _, sisdr = pit_loss(est, src)
                val_sisdr += sisdr
                vn += 1
        val_sisdr /= vn

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch:03d}/{Config.NUM_EPOCHS}] ({elapsed:.0f}s) | "
              f"Train: {train_sisdr:+.2f} dB | Val: {val_sisdr:+.2f} dB | LR: {lr:.1e}")

        scheduler.step(val_sisdr)

        if val_sisdr > best_val:
            best_val = val_sisdr
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_sisdr': val_sisdr,
                'train_sisdr': train_sisdr,
            }, ckpt_path)
            print(f"  💾 Best model saved! (Val: {val_sisdr:+.2f} dB)")

        if epoch % 5 == 0:
            periodic_path = os.path.join(Config.CHECKPOINT_DIR, f'tasnet_v2_epoch{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_sisdr': val_sisdr,
            }, periodic_path)
            print(f"  📌 Checkpoint saved (Epoch {epoch})")

    print(f"\n{'='*60}")
    print(f"🏁 Training complete! Best Val SI-SDR: {best_val:+.2f} dB")
    print(f"   Checkpoint: {ckpt_path}")


if __name__ == '__main__':
    train()
