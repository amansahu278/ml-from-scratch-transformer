import torch.nn as nn
from attention import MultiHeadAttention
from positional_encoding import PositionalEncoding
from utils import LayerNorm, PositionWiseFeedForward


class EncoderLayer(nn.Module):
    def __init__(self, input_dim, d_model, n_heads, num_layers=6):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_layers = num_layers

        self.mha = MultiHeadAttention(d_model, n_heads)
        self.layer_norm1 = LayerNorm(d_model)

        self.pffn = PositionWiseFeedForward(input_dim=d_model, output_dim=d_model)
        self.layer_norm2 = LayerNorm(d_model)

    def forward(self, x):
        x = self.mha(x, x, x) + x
        x = self.layer_norm1(x)
        x = self.pffn(x) + x
        x = self.layer_norm2(x)
        return x


class Encoder(nn.Module):
    def __init__(self, input_dim, d_model, n_heads, num_layers=6):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_layers = num_layers

        self.positional_encoding = PositionalEncoding(d_model)
        self.embedding = nn.Embedding(input_dim, d_model)

        encoder_layers = nn.ModuleList(
            [EncoderLayer(input_dim, d_model, n_heads) for _ in range(num_layers)]
        )
        self.sequence = nn.Sequential(*encoder_layers)

    def forward(self, x):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.sequence(x)
        return x
