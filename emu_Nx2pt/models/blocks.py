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

class MultiHeadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim  # embedded dimension for each token in a sequence
        self.num_heads = num_heads

        assert embed_dim % num_heads == 0, f"Can't divide dimension {embed_dim} into {num_heads} heads"

        d_head = int(embed_dim / num_heads)

        self.Wq = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.num_heads)])
        self.Wk = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.num_heads)])
        self.Wv = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.num_heads)])
        self.d_head = d_head
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, sequences):
        # Sequences has shape (N, seq_length, embed_dim), where embed_dim = token dimension
        result = []
        for sequence in sequences:
            seq_result = []
            for head in range(self.num_heads):
                Wq = self.Wq[head]
                Wk = self.Wk[head]
                Wv = self.Wv[head]

                seq = sequence[:, head * self.d_head: (head + 1) * self.d_head]
                q, k, v = Wq(seq), Wk(seq), Wv(seq)

                attention = self.softmax(q @ k.T / (self.d_head ** 0.5))
                seq_result.append(attention @ v)
            result.append(torch.hstack(seq_result))
        return torch.cat([torch.unsqueeze(r, dim=0) for r in result])


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.mhsa = MultiHeadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_ratio * embed_dim),
            nn.GELU(),
            nn.Linear(mlp_ratio * embed_dim, embed_dim)
        )
    
    def forward(self, x):
        out = x + self.mhsa(self.norm1(x))
        out = out + self.mlp(self.norm2(out))
        return out


class Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.size(0), *self.shape)
