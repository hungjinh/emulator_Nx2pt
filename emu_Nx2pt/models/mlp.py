import torch
import torch.nn as nn
from emu_Nx2pt.models.blocks import ResBN_Block, Res_Block
from emu_Nx2pt.models.blocks import TransformerEncoderBlock, Reshape

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

    def __init__(self, input_size, output_size, hidden_sizes, Nblocks, Nseq, num_heads=2, mlp_ratio=4, scale_factor=1):
        '''
        Args:
            hidden_sizes: hidden_sizes of the ResNet block and the TransformerEncoder block
                e.g. hidden_sizes = [512, 1024]
            Nblocks: number of blocks for the ResNet and the TransformerEncoder
                e.g. Nblocks = [2, 1]
        '''
        super().__init__()
        
        self.Nseq = Nseq
        self.hidden_Res, self.hidden_Trans = hidden_sizes
        self.embed_dim = self.hidden_Trans // Nseq
        
        self.Nblocks_Res, self.Nblocks_TE = Nblocks

        self.in_layer = nn.Sequential(nn.Linear(input_size, self.hidden_Res), nn.ReLU())

        self.res_layers = nn.ModuleDict()
        for i in range(self.Nblocks_Res):
            self.res_layers[f"block_{i}"] = Res_Block(
                self.hidden_Res, self.hidden_Res, self.hidden_Res, scale_factor)
        
        self.mid_layer = nn.Sequential(nn.Linear(self.hidden_Res, self.hidden_Trans), nn.ReLU())
        
        self.trans_layers = nn.ModuleDict()
        for i in range(self.Nblocks_TE):
            self.trans_layers[f"block_{i}"] = TransformerEncoderBlock(self.embed_dim, num_heads, mlp_ratio)
        
        self.out_layer = nn.Linear(self.hidden_Trans, output_size)

        # self.model = nn.Sequential(
        #                 nn.Linear(input_size, hidden_size), nn.ReLU(),
        #                 nn.Linear(hidden_size, hidden_size), nn.ReLU(),
        #                 Reshape((Nseq, embed_dim)),
        #                 TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio),
        #                 Reshape((hidden_size, )),
        #                 nn.Linear(hidden_size, output_size)
        #                 )
    
    def forward(self, x):

        x = self.in_layer(x)

        for i in range(self.Nblocks_Res):
            x = self.res_layers[f"block_{i}"](x)
        
        x = self.mid_layer(x)
        
        x = x.view(x.size(0), self.Nseq, self.embed_dim)

        for i in range(self.Nblocks_TE):
            x = self.trans_layers[f"block_{i}"](x)
        
        x = x.view(x.size(0), self.hidden_Trans)

        y = self.out_layer(x)

        return y
