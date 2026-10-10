import os, torch, numpy as np, soundfile as sf
from config import Config
from model import ConvTasNet
from utils import si_sdr

def evaluate_full_audio():
    model = ConvTasNet().to(Config.DEVICE)
    ckpt_path = os.path.join(Config.CHECKPOINT_DIR, 'tasnet_v2_best.pth')
    ckpt = torch.load(ckpt_path, map_location=Config.DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"✅ Loaded: Epoch {ckpt['epoch']}, Val SI-SDR: {ckpt['val_sisdr']:+.2f} dB")

    test_dir = '/content/drive/MyDrive/cocktail_party_project/data/libri2mix/Libri2Mix/wav8k/min/test'
    mix_dir = os.path.join(test_dir, 'mix_clean')
    filenames = sorted([f for f in os.listdir(mix_dir) if f.endswith('.wav')])

    all_improvements = []
    with torch.no_grad():
        for idx, fname in enumerate(filenames):
            mix, _ = sf.read(os.path.join(test_dir, 'mix_clean', fname), dtype='float32')
            s1, _  = sf.read(os.path.join(test_dir, 's1', fname), dtype='float32')
            s2, _  = sf.read(os.path.join(test_dir, 's2', fname), dtype='float32')

            # Use full audio length, no chunking!
            min_len = min(len(mix), len(s1), len(s2))
            mix, s1, s2 = mix[:min_len], s1[:min_len], s2[:min_len]

            mix_t = torch.from_numpy(mix).float().unsqueeze(0).to(Config.DEVICE)
            s1_t = torch.from_numpy(s1).float().unsqueeze(0).to(Config.DEVICE)
            s2_t = torch.from_numpy(s2).float().unsqueeze(0).to(Config.DEVICE)

            with torch.amp.autocast('cuda', enabled=Config.USE_AMP):
                est = model(mix_t)

            p1 = (si_sdr(est[:,0], s1_t).item() + si_sdr(est[:,1], s2_t).item()) / 2
            p2 = (si_sdr(est[:,0], s2_t).item() + si_sdr(est[:,1], s1_t).item()) / 2
            best = max(p1, p2)

            mb = (si_sdr(mix_t, s1_t).item() + si_sdr(mix_t, s2_t).item()) / 2
            all_improvements.append(best - mb)

            if (idx + 1) % 100 == 0:
                print(f"  Sample {idx+1}/{len(filenames)} | Avg SI-SDRi: {np.mean(all_improvements):+.2f} dB")

    print(f"\n📊 FINAL FULL-AUDIO TEST RESULTS: {np.mean(all_improvements):+.2f} dB")

if __name__ == '__main__':
    evaluate_full_audio()
