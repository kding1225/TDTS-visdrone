from torch import nn


class NoopLayer(nn.Module):
    def __init__(self):
        super(NoopLayer, self).__init__()

    def forward(self, x):
        return x