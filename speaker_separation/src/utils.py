import torch
import itertools

def si_sdr(estimated, target, eps=1e-5):  # Changed from 1e-8 for FP16 stability
    estimated = estimated - estimated.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    dot = torch.sum(estimated * target, dim=-1, keepdim=True)
    s_energy = torch.sum(target ** 2, dim=-1, keepdim=True) + eps
    proj = dot * target / s_energy
    noise = estimated - proj
    return 10 * torch.log10(
        torch.sum(proj ** 2, dim=-1) / (torch.sum(noise ** 2, dim=-1) + eps) + eps)

def pit_loss(est_sources, target_sources):
    B, C, T = est_sources.shape
    perms = list(itertools.permutations(range(C)))
    perm_si_sdrs = []
    for perm in perms:
        s = sum(si_sdr(est_sources[:, c], target_sources[:, p]) for c, p in enumerate(perm)) / C
        perm_si_sdrs.append(s)
    perm_si_sdrs = torch.stack(perm_si_sdrs, dim=0)
    best_si_sdr, _ = perm_si_sdrs.max(dim=0)
    return -best_si_sdr.mean(), best_si_sdr.mean().item()
