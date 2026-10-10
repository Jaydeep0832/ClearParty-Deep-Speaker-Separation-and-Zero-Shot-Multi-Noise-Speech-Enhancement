import torch
import torch.nn as nn
from config import Config


class Encoder(nn.Module):
    def __init__(self, N, L, stride):
        super().__init__()
        self.conv = nn.Conv1d(1, N, kernel_size=L, stride=stride, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x.unsqueeze(1)))


class Decoder(nn.Module):
    def __init__(self, N, L, stride):
        super().__init__()
        self.deconv = nn.ConvTranspose1d(N, 1, kernel_size=L, stride=stride, bias=False)

    def forward(self, x):
        return self.deconv(x).squeeze(1)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, hidden_ch, kernel, dilation, skip_ch):
        super().__init__()
        self.prelu1 = nn.PReLU()
        self.norm1 = nn.GroupNorm(1, in_ch)
        self.dconv = nn.Conv1d(in_ch, in_ch, kernel, dilation=dilation,
                               padding=(kernel - 1) * dilation // 2,
                               groups=in_ch, bias=False)
        self.prelu2 = nn.PReLU()
        self.norm2 = nn.GroupNorm(1, in_ch)
        self.skip_conv = nn.Conv1d(in_ch, skip_ch, 1, bias=False)
        self.res_conv = nn.Conv1d(in_ch, hidden_ch, 1, bias=False)

    def forward(self, x):
        y = self.prelu1(self.norm1(x))
        y = self.dconv(y)
        y = self.prelu2(self.norm2(y))
        skip = self.skip_conv(y)
        res = self.res_conv(y) + x
        return res, skip


class TemporalConvNet(nn.Module):
    def __init__(self, N, B, H, Sc, P, X, R, C, mask_act):
        super().__init__()
        self.C, self.N = C, N
        self.layer_norm = nn.GroupNorm(1, N)
        self.bottleneck = nn.Conv1d(N, B, 1, bias=False)

        self.blocks = nn.ModuleList()
        for r in range(R):
            for x in range(X):
                self.blocks.append(DepthwiseSeparableConv(B, B, P, 2**x, Sc))

        self.output_conv = nn.Conv1d(Sc, N * C, 1, bias=False)
        self.mask_act = nn.ReLU() if mask_act == 'relu' else nn.Sigmoid()

    def forward(self, x):
        B_size, _, K = x.shape
        y = self.bottleneck(self.layer_norm(x))
        skip_sum = 0
        for block in self.blocks:
            y, skip = block(y)
            skip_sum = skip_sum + skip
        masks = self.output_conv(skip_sum)
        masks = masks.view(B_size, self.C, self.N, K)
        return self.mask_act(masks)


class ConvTasNet(nn.Module):
    def __init__(self):
        super().__init__()
        N, L, S = Config.ENC_DIM, Config.ENC_KERNEL, Config.ENC_STRIDE
        self.encoder = Encoder(N, L, S)
        self.separator = TemporalConvNet(
            N, Config.BN_DIM, Config.HIDDEN_DIM, Config.SC_DIM,
            Config.CONV_KERNEL, Config.NUM_BLOCKS, Config.NUM_REPEATS,
            Config.NUM_SPEAKERS_OUT, Config.MASK_ACT)
        self.decoder = Decoder(N, L, S)

    def forward(self, mixture):
        enc = self.encoder(mixture)
        masks = self.separator(enc)
        est = []
        for c in range(Config.NUM_SPEAKERS_OUT):
            est.append(self.decoder(enc * masks[:, c]))
        est = torch.stack(est, dim=1)
        return est[..., :mixture.shape[-1]]


print("✅ model.py created")
