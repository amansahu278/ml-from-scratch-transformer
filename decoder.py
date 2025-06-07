import torch
import torch.nn as nn
from attention import MultiHeadAttention
from encoder import Encoder
from positional_encoding import PositionalEncoding
from utils import LayerNorm, PositionWiseFeedForward
from torchinfo import summary


class DecoderLayer(nn.Module):
    def __init__(self, input_dim, d_model, n_heads, num_layers=6):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_layers = num_layers

        self.mha1 = MultiHeadAttention(d_model, n_heads)
        self.layer_norm1 = LayerNorm(d_model)

        self.mha2 = MultiHeadAttention(d_model, n_heads)
        self.layer_norm2 = LayerNorm(d_model)

        self.pffn = PositionWiseFeedForward(input_dim=d_model, output_dim=d_model)
        self.layer_norm3 = LayerNorm(d_model)

    def forward(self, x, encoder_output):
        seq_len = x.size(1)
        causal_mask = torch.tril(torch.ones((seq_len, seq_len)), device=x.device)

        # Masked multi-head attention
        x = self.mha1(x, x, x, causal_mask) + x
        x = self.layer_norm1(x)

        x = self.mha2(q=x, k=encoder_output, v=encoder_output) + x
        x = self.layer_norm2(x)

        x = self.pffn(x) + x
        x = self.layer_norm3(x)
        return x


class Decoder(nn.Module):
    def __init__(self, input_dim, d_model, n_heads, num_layers=6):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_layers = num_layers

        self.positional_encoding = PositionalEncoding(d_model)
        self.embedding = nn.Embedding(input_dim, d_model)

        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(input_dim, d_model, n_heads) for _ in range(num_layers)]
        )

    def forward(self, x, encoder_output):
        x = self.embedding(x)
        x = self.positional_encoding(x)

        for layer in self.decoder_layers:
            x = layer(x, encoder_output)

        return x


if __name__ == "__main__":
    # Example usage
    input_dim = 1000  # Vocabulary size
    d_model = 512  # Embedding dimension
    n_heads = 8  # Number of attention heads
    num_layers = 6  # Number of encoder layers

    sample_input = torch.randint(
        0, input_dim, (32, 10)
    )  # Batch size of 32, sequence length of 10

    encoder = Encoder(input_dim, d_model, n_heads, num_layers)
    decoder = Decoder(input_dim, d_model, n_heads, num_layers)
    enc_output = encoder(sample_input)
    output = decoder(sample_input, enc_output)
    print(f"Output shape: {output.shape}")  # Should be (32, 10, d_model)
    # Should be (32, 10, d_model)
    summary(
        decoder,
        input_data=[sample_input, enc_output],
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
    )
