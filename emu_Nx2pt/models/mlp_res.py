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

class MLP_Res(nn.Module):

    def __init__(self, input_size, output_size, hidden_size, Nblocks, is_batchNorm=True, scale_factor=1):
        super().__init__()

        self.Nblocks = Nblocks

        self.in_layer = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU())

        self.mid_layers = nn.ModuleDict()
        if is_batchNorm:
            for i in range(Nblocks):
                self.mid_layers[f"block_{i}"] = ResBN_Block(hidden_size, hidden_size, hidden_size, scale_factor)
        else:
            for i in range(Nblocks):
                self.mid_layers[f"block_{i}"] = Res_Block(hidden_size, hidden_size, hidden_size, scale_factor)
        
        self.out_layer = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):

        x = self.in_layer(x)

        for i in range(self.Nblocks):
            x = self.mid_layers[f"block_{i}"](x)
        
        y = self.out_layer(x)

        return y