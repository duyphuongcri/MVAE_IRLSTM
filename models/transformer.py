from functools import partial
from typing import Callable, List

import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version,
        # can set manually to be compat with prev weights
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x) \
            .reshape(B, N, 3, self.num_heads, C // self.num_heads) \
            .permute(2, 0, 3, 1, 4)
        
        q, k, v = qkv[0],  qkv[1],  qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class MLP(nn.Module):
    def __init__(self, hidden_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size*4)
        self.fc2 = nn.Linear(hidden_size*4, hidden_size)

        self.act_fn = nn.GELU()
        self.dropout = nn.Dropout(0.0)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
    
class TransformerLayer(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(TransformerLayer, self).__init__()

        self.layernorm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.mhsa = Attention(dim=hidden_size, num_heads=num_heads)#multihead self-attention
        self.mlp = MLP(hidden_size)

    def forward(self, x):
        """
        x shape: (batchsize, n_patches, hidden_size)
        """
        x = self.mhsa(self.layernorm(x)) +x

        x = self.mlp(self.layernorm(x)) + x

        return x  