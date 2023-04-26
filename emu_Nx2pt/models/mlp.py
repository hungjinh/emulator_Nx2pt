import torch
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

    def __init__(self, input_size, output_size, hidden_size, Nblocks, is_batchNorm=False, scale_factor=1):
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

class ParallelMicroNets(nn.Module):
    
    def __init__(self, input_size, encode_size, hidden_size, output_sizes, Nblocks, scale_factor=1):
        '''
            Args:
                output_sizes: 
                    e.g. output_sizes = [2425, 707] -> The final dataT length would be 2425+707=3132
        '''
        
        super().__init__()
        
        self.par_layer = nn.Sequential(nn.Linear(input_size, encode_size), nn.ReLU())

        self.Nnets = len(output_sizes)

        self.microNets = nn.ModuleDict()

        for i in range(self.Nnets):
            
            if Nblocks > 1:
                self.microNets[f'net_{i}'] = nn.ModuleList([Res_Block(
                    encode_size, hidden_size, hidden_size, scale_factor)] + [Res_Block(hidden_size, hidden_size, hidden_size, scale_factor) for _ in range(Nblocks-1)])
            else:
                self.microNets[f'net_{i}'] = nn.ModuleList([Res_Block(encode_size, hidden_size, hidden_size, scale_factor)])
            
            self.microNets[f'net_{i}'].append(nn.Linear(hidden_size, output_sizes[i]))
        
    
    def forward(self, par):
        
        par_encode = self.par_layer(par)
        
        result = []

        for i in range(self.Nnets):
            
            x = par_encode
            for layer in self.microNets[f'net_{i}']:
                x = layer(x)
            result.append(x)
        
        if len(result[0].shape) == 1:
            return torch.cat(result, dim=0)
        else:
            return torch.cat(result, dim=1)


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
