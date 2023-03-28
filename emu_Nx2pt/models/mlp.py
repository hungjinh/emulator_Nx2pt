import torch.nn as nn


class MLP(nn.Module):
    
    def __init__(self, input_size, output_size, Nblocks, hidden_size):
        super().__init__()
        self.Nblocks = Nblocks

        self.in_layer   = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU())

        self.mid_layers = nn.ModuleDict()
        for i in range(Nblocks):
            self.mid_layers[f"block_{i}"] = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU())

        self.out_layer  = nn.Linear(hidden_size, output_size)
        
    
    def forward(self, x):
        x = self.in_layer(x)
        
        for i in range(self.Nblocks):
            x = self.mid_layers[f"block_{i}"](x)
        
        x = self.out_layer(x)
        
        return x
