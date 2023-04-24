import torch.nn as nn
from emu_Nx2pt.models.blocks import ResBN_Block, Res_Block, MultiHeadAttention, Reshape

class MLP(nn.Module):
    
    def __init__(self, input_size, output_size, hidden_size, Nblocks, is_batchNorm=True):
        super().__init__()
        self.Nblocks = Nblocks

        self.in_layer   = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU())

        self.mid_layers = nn.ModuleDict()

        if is_batchNorm:
            for i in range(Nblocks):
                self.mid_layers[f"block_{i}"] = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.BatchNorm1d(hidden_size), nn.ReLU())
        else:
            for i in range(Nblocks):
                self.mid_layers[f"block_{i}"] = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU())

        self.out_layer  = nn.Linear(hidden_size, output_size)
        
    
    def forward(self, x):
        x = self.in_layer(x)
        
        for i in range(self.Nblocks):
            x = self.mid_layers[f"block_{i}"](x)
        
        y = self.out_layer(x)
        
        return y

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

class AttentionBasedMLP(nn.Module):

    def __init__(self, input_size, output_size, hidden_size, d_embed):
        super().__init__()

        Nseq = hidden_size // d_embed
        
        self.model = nn.Sequential(
                        nn.Linear(input_size, hidden_size), nn.ReLU(),
                        nn.Linear(hidden_size, hidden_size), nn.ReLU(),
                        nn.Reshap((Nseq, d_embed)),
                        MultiHeadAttention(d_embed=d_embed, n_heads=2),
                        MultiHeadAttention(d_embed=d_embed, n_heads=2),
                        nn.Reshap((hidden_size, )),
                        nn.Relu(),
                        nn.Linear(hidden_size, output_size)
                        )

    
    def forward(self, x):
        
        return self.model(x)
