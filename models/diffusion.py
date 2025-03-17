# models.py
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

        # Create a list of CANDY modules, one for each time step
        self.candies = nn.ModuleList(
            [
                CANDY(batch_size, in_channel, hidden_channel, out_channel, input_size, hidden_size,)
                for _ in range(T)
            ]
        )

        # Create a list of UNet modules, one for each time step
        self.unets = nn.ModuleList([UNet(in_channel, out_channel) for _ in range(T)])

        self.seg_head = nn.Conv2d(out_channel, num_classes, kernel_size=1)

    def forward(self, x, graph_schedule=None):
        device = x.device
        origin = torch.zeros(
            self.T, self.batch_size, self.in_channel, self.hidden_size, self.input_size
        ).to(device)
        input = x

        # Forward diffusion process (adding noise in each step)
        for t in range(self.T):
            # Pass through the t-th CANDY module (feature processing)
            output = self.candies[t](input)
            origin[t] = input
            input = output

        if graph_schedule is None:
            graph_schedule = torch.linspace(0.7, 0.2, self.T).to(device)

        # Reverse diffusion process (denoising)
        for t in reversed(range(self.T)):
            graph_factor = graph_schedule[t]
            reverse_input = (1 - graph_factor) * input + graph_factor * origin[t]

            # Pass through the t-th UNet module for reconstruction
            output = self.unets[t](reverse_input)
            input = output  # Update input for the next step

        output_seg = self.seg_head(output)
        output_seg = torch.sum(output_seg, dim=1, keepdim=True)

        return output_seg
