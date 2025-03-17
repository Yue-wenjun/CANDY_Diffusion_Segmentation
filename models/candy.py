import torch
import torch.nn as nn

class CustomActivation(nn.Module):
    def __init__(self):
        super(CustomActivation, self).__init__()

    def forward(self, x):
        return (torch.abs(x + 1) - torch.abs(x - 1)) / 2

class CANDY(nn.Module):
    def __init__(self, batch_size, in_channel, hidden_channel, out_channel, input_size, hidden_size):
        super(CANDY, self).__init__()
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.in_channel = in_channel
        self.hidden_channel = hidden_channel
        self.out_channel = out_channel

        self.fc0 = nn.Linear(batch_size*in_channel, hidden_channel)
        self.p_mask = nn.Parameter(torch.randn(hidden_size, input_size), requires_grad=True)
        self.p_output_layer = nn.Sequential(
            CustomActivation(),
            nn.Linear(input_size, input_size),
            CustomActivation()
        )
        self.Wp = nn.Parameter(torch.tril(torch.randn(hidden_size, hidden_size)), requires_grad=True)
        self.Wp.data.diagonal().fill_(1)
        self.Wp_diag = nn.Parameter(torch.ones(hidden_size), requires_grad=True)
        self.z_output_layer = nn.Sequential(
            CustomActivation(),
            nn.Linear(input_size, input_size),
            CustomActivation()
        )
        self.Wzp = nn.Parameter(torch.randn(hidden_size, hidden_size), requires_grad=True)
        self.fc1 = nn.Sequential(
            nn.Linear(hidden_channel, out_channel),
            nn.ReLU(True))

    def split_input(self, x):
        p_mask = CustomActivation()(self.p_mask).to(x.device)
        p_set = x * p_mask
        return p_set

    def forward(self, x):
        # x = x.view(self.batch_size * self.in_channel, self.hidden_size, self.input_size)
        x = x.view(self.batch_size * self.hidden_channel, self.hidden_size, self.input_size)
        p_set = self.split_input(x)

        with torch.no_grad():
            self.Wp.data = torch.tril(self.Wp.data)
            self.Wp.data.diagonal().clamp_(min=0, max=1)
        Wp = self.Wp + torch.diag(self.Wp_diag)
        Wzp = self.Wzp

        p_output = torch.Tensor(self.batch_size * self.hidden_channel, self.hidden_size, self.input_size).to(x.device)
        z_output = torch.Tensor(self.batch_size * self.hidden_channel, self.hidden_size, self.input_size).to(x.device)
        for i in range(self.hidden_channel):
            temp_p_output = torch.mm(Wp, p_set[i])
            temp_p_output = self.p_output_layer(temp_p_output)
            temp_z_output = torch.mm(Wzp, temp_p_output)
            temp_z_output = self.z_output_layer(temp_z_output)
            p_output[i] = temp_p_output.clone()
            z_output[i] = temp_z_output.clone()

        combined_output = p_output + z_output
        output = combined_output.view(self.batch_size, self.out_channel, self.hidden_size, self.input_size)
        return output
