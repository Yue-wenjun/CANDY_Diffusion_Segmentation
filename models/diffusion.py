import torch
import torch.nn as nn
import torch.nn.functional as F
from models.candy import CANDY
from models.unet import UNet
from models.segformer_b0 import SegFormerB0
from models.mobilevit_small import MobileViTSmall

_DECODERS = {
    "unet": UNet,
    "segformer_b0": SegFormerB0,
    "mobilevit_small": MobileViTSmall,
}


class DiffusionModel(nn.Module):
    def __init__(
        self,
        batch_size,
        in_channel,
        hidden_channel,
        out_channel,
        input_size,
        hidden_size,
        T,
        num_classes,
        decoder_type="unet",
        **kwargs,          # absorb extra config keys (lr, epochs, test_data_dir, …)
    ):
        super(DiffusionModel, self).__init__()
        self.in_channel = in_channel
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.batch_size = batch_size
        self.T = T

        # T independent CANDYs: each step learns a different forward transformation (~315K each, cheap)
        self.candies = nn.ModuleList([
            CANDY(batch_size, in_channel, hidden_channel, out_channel, input_size, hidden_size)
            for _ in range(T)
        ])

        # Decoder: shared UNet (with timestep emb) or a standalone baseline (SegFormer / MobileViT)
        if decoder_type not in _DECODERS:
            raise ValueError(f"decoder_type '{decoder_type}' not in {list(_DECODERS)}")
        if decoder_type == "unet":
            self.unet = UNet(in_channel, out_channel, T=T)
        else:
            self.unet = _DECODERS[decoder_type](in_channel, out_channel)

        # T fusion convs stay per-step (tiny: 18 params each)
        self.fusion_convs = nn.ModuleList([
            nn.Conv2d(in_channel * 2, in_channel, kernel_size=3, padding=1)
            for _ in range(T)
        ])

        self.seg_head = nn.Conv2d(out_channel, num_classes, kernel_size=1)

    def forward(self, x, graph_schedule=None):
        device = x.device
        B = x.shape[0]
        origin = torch.zeros(self.T, B, self.in_channel, self.hidden_size, self.input_size, device=device)
        feat = x

        # ====== Forward diffusion process ======
        for t in range(self.T):
            origin[t] = feat
            feat = self.candies[t](feat)

        # ====== Graph schedule ======
        if graph_schedule is None:
            graph_schedule = torch.linspace(0.7, 0.2, self.T, device=device)

        # ====== Reverse diffusion process ======
        t_idx = torch.zeros(B, dtype=torch.long, device=device)
        for t in reversed(range(self.T)):
            graph_factor = graph_schedule[t]
            t_idx.fill_(t)

            fused = self.fusion_convs[t](torch.cat([feat, origin[t]], dim=1))
            reverse_input = (1 - graph_factor) * fused + graph_factor * origin[t]
            feat = self.unet(reverse_input, t_idx)

        # ====== Segmentation head ======
        return self.seg_head(feat)   # [B, num_classes, H, W] logits
