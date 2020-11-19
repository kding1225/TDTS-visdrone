import torch
import torch.nn as nn


class ChannelNorm(nn.Module):

    def __init__(self, in_channels):
        super(ChannelNorm, self).__init__()

        self.weight = nn.Parameter(torch.ones(1, in_channels), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(1, in_channels), requires_grad=True)

    def forward(self, x):
        is_4d_tensor = x.ndim == 4

        if is_4d_tensor:
            n, c, h, w = x.shape
            x = x.permute(0, 2, 3, 1).reshape(-1, c)

        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        x = (x-mean) / (var + 1e-5).sqrt() *self.weight + self.bias

        if is_4d_tensor:
            x = x.view(n, h, w, c).permute(0, 3, 1, 2).contiguous()

        return x


class NoopLayer(nn.Module):
    def __init__(self):
        super(NoopLayer, self).__init__()
    def forward(self, x):
        return x