import torch.nn as nn


class MLP(nn.Module):
    
    def __init__(self, Npco, Nout, Nblocks, Nhidden):
        super().__init__()
        self.Nblocks = Nblocks

        self.in_layer   = nn.Sequential(nn.Linear(Npco, Nhidden), nn.ReLU())

        self.mid_layers = nn.ModuleDict()
        for i in range(Nblocks):
            self.mid_layers[f"block_{i}"] = nn.Sequential(nn.Linear(Nhidden, Nhidden), nn.ReLU())

        self.out_layer  = nn.Linear(Nhidden, Nout)
        
    
    def forward(self, x):
        x = self.in_layer(x)
        
        for i in range(self.Nblocks):
            x = self.mid_layers[f"block_{i}"](x)
        
        x = self.out_layer(x)
        
        return x
