import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _sinusoidal_embedding(t, dim):
    """Sinusoidal timestep embedding. t: (B,) int64 → (B, dim) float."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / max(half - 1, 1)
    )
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
class UNet(nn.Module):
    def __init__(self, in_channel, out_channel, T=1, bilinear=False):
        super(UNet, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.bilinear = bilinear

        self.inc = DoubleConv(in_channel, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        bottleneck_ch = 1024 // factor
        self.down4 = Down(512, bottleneck_ch)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, out_channel)

        # Multi-scale timestep conditioning: inject at bottleneck + every decoder stage.
        # Shallow decoder stages (up3/up4) recover boundary details, so they also need
        # t-conditioning to behave differently across diffusion steps.
        self.T = T
        if T > 1:
            _emb_dim = 64
            self._emb_dim = _emb_dim
            # Shared MLP trunk: sin/cos emb → hidden
            self.time_mlp = nn.Sequential(
                nn.Linear(_emb_dim, _emb_dim * 4),
                nn.SiLU(),
            )
            _h = _emb_dim * 4
            # Per-scale projections (additive bias, no scale needed for single-channel input)
            self.t_proj_bot  = nn.Linear(_h, bottleneck_ch)
            self.t_proj_up1  = nn.Linear(_h, 512 // factor)
            self.t_proj_up2  = nn.Linear(_h, 256 // factor)
            self.t_proj_up3  = nn.Linear(_h, 128 // factor)
            self.t_proj_up4  = nn.Linear(_h, 64)

    def forward(self, x, t=None):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        if t is not None and self.T > 1:
            t_h = self.time_mlp(_sinusoidal_embedding(t, self._emb_dim))  # (B, 256)
            _add = lambda feat, proj: feat + proj(t_h).unsqueeze(-1).unsqueeze(-1)
            x5 = _add(x5, self.t_proj_bot)
            d1 = _add(self.up1(x5, x4), self.t_proj_up1)
            d2 = _add(self.up2(d1, x3), self.t_proj_up2)
            d3 = _add(self.up3(d2, x2), self.t_proj_up3)
            d4 = _add(self.up4(d3, x1), self.t_proj_up4)
        else:
            d1 = self.up1(x5, x4)
            d2 = self.up2(d1, x3)
            d3 = self.up3(d2, x2)
            d4 = self.up4(d3, x1)

        return self.outc(d4)

    