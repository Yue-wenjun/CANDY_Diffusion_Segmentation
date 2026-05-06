import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomActivation(nn.Module):
    def forward(self, x):
        return (torch.abs(x + 1) - torch.abs(x - 1)) / 2


class CANDY(nn.Module):
    def __init__(self, batch_size, in_channel, hidden_channel, out_channel, input_size, hidden_size):
        super(CANDY, self).__init__()
        self.batch_size   = batch_size    # kept for API compat; forward uses x.shape[0]
        self.input_size   = input_size
        self.hidden_size  = hidden_size
        self.in_channel   = in_channel
        self.hidden_channel = hidden_channel
        self.out_channel  = out_channel

        # fc0 kept for checkpoint compatibility (not used in forward)
        self.fc0 = nn.Linear(batch_size * in_channel, hidden_channel)

        self.p_mask = nn.Parameter(torch.randn(hidden_size, input_size))

        self.p_output_layer = nn.Sequential(
            CustomActivation(),
            nn.Linear(input_size, input_size),
            CustomActivation(),
        )

        # Wp is stored as strictly lower-triangular (diagonal = 0).
        # The actual diagonal comes from softplus(Wp_diag), which is always positive.
        self.Wp      = nn.Parameter(torch.tril(torch.randn(hidden_size, hidden_size), diagonal=-1))
        self.Wp_diag = nn.Parameter(torch.zeros(hidden_size))   # softplus → positive diagonal

        self.z_output_layer = nn.Sequential(
            CustomActivation(),
            nn.Linear(input_size, input_size),
            CustomActivation(),
        )
        self.Wzp = nn.Parameter(torch.randn(hidden_size, hidden_size))

        self.fc1 = nn.Sequential(
            nn.Linear(hidden_channel, out_channel),
            nn.ReLU(inplace=True),
        )

    def split_input(self, x):
        # p_mask: [H, S] — broadcasts over [B, C, H, S]
        p_mask = CustomActivation()(self.p_mask)
        return x * p_mask

    def forward(self, x):
        B = x.shape[0]
        # Reshape to 4-D so matmul can broadcast over the batch & channel dims
        x     = x.view(B, self.hidden_channel, self.hidden_size, self.input_size)  # [B, C, H, S]
        p_set = self.split_input(x)                                                 # [B, C, H, S]

        # Lower-triangular Wp with positive diagonal — no in-place Parameter mutation
        # torch.tril(..., diagonal=-1) zeroes the diagonal in the computation graph,
        # so upper-tri + diagonal gradients are naturally zeroed without no_grad tricks.
        Wp = torch.tril(self.Wp, diagonal=-1) + torch.diag(F.softplus(self.Wp_diag))

        # [H, H] @ [B, C, H, S] — matmul broadcasts over B and C → [B, C, H, S]
        p_output = self.p_output_layer(torch.matmul(Wp, p_set))          # [B, C, H, S]
        z_output = self.z_output_layer(torch.matmul(self.Wzp, p_output)) # [B, C, H, S]

        combined = p_output + z_output             # [B, C, H, S]
        combined = combined.permute(0, 2, 3, 1)    # [B, H, S, C]
        output   = self.fc1(combined)              # [B, H, S, out_channel]
        output   = output.permute(0, 3, 1, 2)      # [B, out_channel, H, S]
        return output
