import torch.nn as nn
from torch.nn import functional as F

class ResBN_Block(nn.Module):

    def __init__(self, input_size, output_size, hidden_size, scale_factor=1):
        super().__init__()

        self.scale_factor = scale_factor

        self.layer1 = nn.Linear(input_size, hidden_size)
        self.bn1    = nn.BatchNorm1d(hidden_size)
        self.layer2 = nn.Linear(input_size, output_size)
        self.bn2    = nn.BatchNorm1d(hidden_size)

        if input_size != output_size:
            self.skip_layer = nn.Linear(input_size, output_size, bias=False)
        else:
            self.skip_layer = nn.Identity()

    def forward(self, x):

        identity = self.skip_layer(x)

        o = F.relu(self.bn1(self.layer1(x)))
        residual = self.bn2(self.layer2(o))

        return F.relu(identity + residual * self.scale_factor)

class Res_Block(nn.Module):

    def __init__(self, input_size, output_size, hidden_size, scale_factor=1):
        super().__init__()

        self.scale_factor = scale_factor

        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(input_size, output_size)

        if input_size != output_size:
            self.skip_layer = nn.Linear(input_size, output_size, bias=False)
        else:
            self.skip_layer = nn.Identity()

    def forward(self, x):

        identity = self.skip_layer(x)

        o = F.relu(self.layer1(x))
        residual = self.layer2(o)

        return F.relu(identity + residual * self.scale_factor)

