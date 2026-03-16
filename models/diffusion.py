import torch
import torch.nn as nn
import torch.nn.functional as F
from models.candy import CANDY
from models.unet import UNet


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
    ):
        super(DiffusionModel, self).__init__()
        self.in_channel = in_channel
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.batch_size = batch_size
        self.T = T  # Number of diffusion steps

        # ====== Forward modules ======
        self.candies = nn.ModuleList(
            [
                CANDY(
                    batch_size,
                    in_channel,
                    hidden_channel,
                    out_channel,
                    input_size,
                    hidden_size,
                )
                for _ in range(T)
            ]
        )

        self.unets = nn.ModuleList([UNet(in_channel, out_channel) for _ in range(T)])

        # ====== New: Fusion convs for reverse diffusion ======
        self.fusion_convs = nn.ModuleList(
            [
                nn.Conv2d(in_channel * 2, in_channel, kernel_size=3, padding=1)
                for _ in range(T)
            ]
        )

        self.seg_head = nn.Conv2d(out_channel, num_classes, kernel_size=1)

    def forward(self, x, graph_schedule=None):
        device = x.device
        origin = torch.zeros(
            self.T, self.batch_size, self.in_channel, self.hidden_size, self.input_size
        ).to(device)
        input = x

        # ====== Forward diffusion process ======
        for t in range(self.T):
            output = self.candies[t](input)
            origin[t] = input
            input = output
            output = self.candies[t](input)

        # ====== Graph schedule ======
        if graph_schedule is None:
            graph_schedule = torch.linspace(0.7, 0.2, self.T).to(device)

        # ====== Reverse diffusion process ======
        for t in reversed(range(self.T)):
            graph_factor = graph_schedule[t]

            # Step 1: concatenate noisy and original feature maps
            fusion_input = torch.cat([input, origin[t]], dim=1)

            # Step 2: fuse via learnable conv
            fused_feature = self.fusion_convs[t](fusion_input)

            # Step 3: apply graph_factor modulation
            reverse_input = (1 - graph_factor) * fused_feature + graph_factor * origin[t]

            # Step 4: U-Net reconstruction
            output = self.unets[t](reverse_input)
            input = output  # feedback for next step

        # ====== Segmentation head ======
        output_seg = self.seg_head(output)
        output_seg = torch.sum(output_seg, dim=1, keepdim=True)

        return output_seg
