import math
import torch
import torch.nn as nn
from torch.nn import functional as F

class ResBN_Block(nn.Module):

    def __init__(self, input_size, output_size, hidden_size, scale_factor=1):
        super().__init__()

        self.scale_factor = scale_factor

        self.layer1 = nn.Linear(input_size, hidden_size)
        self.bn1    = nn.BatchNorm1d(hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
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
        self.layer2 = nn.Linear(hidden_size, output_size)

        if input_size != output_size:
            self.skip_layer = nn.Linear(input_size, output_size, bias=False)
        else:
            self.skip_layer = nn.Identity()

    def forward(self, x):

        identity = self.skip_layer(x)

        o = F.relu(self.layer1(x))
        residual = self.layer2(o)

        return F.relu(identity + residual * self.scale_factor)


class DotProductAttention(nn.Module):
    '''
    Args:
        p_dropout: probability of an element to be zeroed. Default: 0., i.e. no dropout
    '''

    def __init__(self, dropout_p=0.):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, queries, keys, values):
        d = queries.shape[-1]

        scores = (queries @ keys.transpose(-2, -1))/math.sqrt(d)
        self.attn_weights = nn.functional.softmax(scores, dim=-1)

        return self.dropout(self.attn_weights) @ values


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_p=0., bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout_p)

        self.Wq = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.Wk = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.Wv = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.Wo = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(self, queries, keys, values, need_weights=False, average_attn_weights=False):
        queries = self.transpose_qkv(self.Wq(queries))
        keys = self.transpose_qkv(self.Wq(keys))
        values = self.transpose_qkv(self.Wq(values))

        output = self.attention(queries, keys, values) # (batch_size*num_heads, n_seq, embed_dim/num_heads)
        output_concat = self.transpose_output(output)  # (batch_size, n_seq, embed_dim)

        if need_weights:
            ori_attn_weights = self.attention.attn_weights
            ori_attn_weights = ori_attn_weights.reshape(-1, self.num_heads, ori_attn_weights.shape[1], ori_attn_weights.shape[2])
            if average_attn_weights:
                return self.Wo(output_concat), ori_attn_weights.mean(dim=1)
            else:
                return self.Wo(output_concat), ori_attn_weights
        else:
            return self.Wo(output_concat)

    def transpose_qkv(self, X):
        '''Transposition for parallel computation of multiple attention heads.
            input  X.shape = (batch_size, n_seq, embed_dim) 
            output X.shape = (batch_size*num_heads, n_seq, embed_dim/num_heads)
        '''
        X = X.reshape(X.shape[0], X.shape[1], self.num_heads, -1)
        X = X.permute(0, 2, 1, 3)
        return X.reshape(-1, X.shape[2], X.shape[3])

    def transpose_output(self, X):
        '''Reverse the operation of transpose_qkv.'''
        X = X.reshape(-1, self.num_heads, X.shape[1], X.shape[2])
        X = X.permute(0, 2, 1, 3)
        return X.reshape(X.shape[0], X.shape[1], -1)


class AddNorm(nn.Module):
    def __init__(self, norm_shape, dropout_p=0.):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
        self.ln = nn.LayerNorm(norm_shape)
    
    def forward(self, X, Y):
        return self.ln( self.dropout(Y) + X )


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=1, dropout_p=0., bias=True):
        super().__init__()

        self.mha = MultiHeadAttention(embed_dim, num_heads, dropout_p, bias)

        self.addnorm1 = AddNorm(embed_dim, dropout_p)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, mlp_ratio * embed_dim),
            nn.GELU(),
            nn.Linear(mlp_ratio * embed_dim, embed_dim)
        )

        self.addnorm2 = AddNorm(embed_dim, dropout_p)

    def forward(self, X):
        Y = self.addnorm1(X, self.mha(X, X, X))
        return self.addnorm2(Y, self.ffn(Y))
