import torch
import torch.nn as nn
import torch.functional as F

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

import config

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size : int = (16,16), emb_size : int = 768):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.emb_size = emb_size

        self.embedding = nn.Conv2d(in_channels=3, out_channels=self.emb_size, kernel_size=self.patch_size, stride=self.patch_size)
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)

    def forward(self, input):
        # input : Batch X Channels X Width X Height
        b, c, w, h = input.shape
        # Patch num(N) = (Height * Width) / (Patch Size^2)
        N = (h * w) // (self.patch_size[0] ** 2)

        embedded_data = self.embedding(input)
        flatted_data = self.flatten(embedded_data)
        transposed_data = torch.transpose(flatted_data, dim0=1, dim1=2)

        if torch.cuda.is_available():
            cls_token = nn.Parameter(torch.randn((b, 1, self.emb_size))).cuda()
            position_emb = nn.Parameter(torch.randn(N + 1, self.emb_size)).cuda()
        else:
            cls_token = nn.Parameter(torch.randn((b, 1, self.emb_size)))
            position_emb = nn.Parameter(torch.randn(N + 1, self.emb_size))

        embedded_data = torch.cat([cls_token, transposed_data], dim=1)
        embedded_data += position_emb
        # embedded_data : Batch, Patch num(N) + 1, Embedding Size
        return embedded_data

class MultiHeadAttention(nn.Module):
    def __init__(self, head_num : int = 8, emb_size : int = 768):
        super(MultiHeadAttention, self).__init__()
        self.head_num = head_num
        self.emb_size = emb_size

        self.q_layer = nn.Linear(emb_size, emb_size)
        self.k_layer = nn.Linear(emb_size, emb_size)
        self.v_layer = nn.Linear(emb_size, emb_size)

        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, input):
        # input : Batch, Patch num(N) + 1, Embedding Size
        b, n, e = input.shape

        queries = self.q_layer(input)
        keys = self.k_layer(input)
        values = self.v_layer(input)

        queries = torch.reshape(queries, [b, self.head_num, n, int(e / self.head_num)])
        keys = torch.reshape(keys, [b, self.head_num, n, int(e / self.head_num)])
        values = torch.reshape(values, [b, self.head_num, n, int(e / self.head_num)])

        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        scailing_factor = self.emb_size ** (1/2)
        attention_score = torch.softmax(energy, dim=-1) / scailing_factor

        out = torch.einsum('bhal, bhlv -> bhav ', attention_score, values)

        b, h, n, d = out.shape
        out = torch.reshape(out, [b, n, int(h * d)])
        out = self.projection(out)
        # out : Batch, Patch num(N) + 1, Embedding Size
        return out

class FeedForwardBlock(nn.Module):
    def __init__(self, emb_size : int = 768, expansion : int = 4, drop_probability : float = 0.):
        super(FeedForwardBlock, self).__init__()
        self.linear_layer1 = nn.Linear(emb_size, expansion * emb_size)
        self.GeLU = nn.GELU()
        self.Dropout = nn.Dropout(drop_probability)
        self.linear_layer2 = nn.Linear(expansion * emb_size, emb_size)

    def forward(self, input):
        out = self.linear_layer1(input)
        out = self.GeLU(out)
        out = self.Dropout(out)
        out = self.linear_layer2(out)

        return out

class TransformerEncoderBlock(nn.Module):
    def __init__(self, emb_size : int = 768):
        super(TransformerEncoderBlock, self).__init__()
        # self.embedding = PatchEmbedding(patch_size=(16, 16), emb_size=emb_size)

        self.norm1 = nn.LayerNorm(emb_size)
        self.MHA = MultiHeadAttention(emb_size=emb_size)
        self.norm2 = nn.LayerNorm(emb_size)
        self.MLP = FeedForwardBlock(emb_size=emb_size)

    def forward(self, input):
        residual = input
        out = self.norm1(residual)
        out = self.MHA(out)
        residual = residual + out

        out = self.norm2(out)
        out = self.MLP(out)
        residual = residual + out

        return residual

class TransformerEncoder(nn.Sequential):
    def __init__(self, depth : int = 12, **kwargs):
        super(TransformerEncoder, self).__init__(
            PatchEmbedding(patch_size=(16, 16), emb_size=768),
            *[TransformerEncoderBlock(**kwargs) for _ in range(depth)],
            ClassificationHead()
        )

class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size : int = 768, classes : int = 10):
        super(ClassificationHead, self).__init__(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, classes)
        )

if __name__ == '__main__':
    x = torch.randn(8, 3, 32, 32)
    Net = TransformerEncoder()
    embedded = Net(x)
    print(embedded.shape)