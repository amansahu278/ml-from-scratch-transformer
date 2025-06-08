import torch
import torch.nn as nn
from utils import Linear


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        # Temperature is the scaling factor for the dot product
        # As per the original paper, it is the sqrt of the dimension of the key vector
        self.temperature = temperature

    def forward(self, q, k, v, mask=None):
        # Temperature = sqrt(k.size(-1))
        # (B, seq_len, d_k)
        attn = (q @ k.transpose(-2, -1)) / self.temperature
        if mask is not None:
            # Replace all positions where mask is 0 with -inf
            attn = attn.masked_fill(mask == 0, float("-inf"))

        # We apply softmax over the last dimension to get the attention score of
        # each token over all positions in the sequence
        # attn = attn.softmax(dim=-1)
        attn = attn.softmax(dim=-1)  # (B, n_heads, seq_len, seq_len)
        attn = attn @ v
        return attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.linear_q = Linear(d_model, d_model)
        self.linear_k = Linear(d_model, d_model)
        self.linear_v = Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention(temperature=self.d_k**0.5)
        # This is the output linear layer that combines the outputs of the different heads
        # This is how the relations between the heads are learned
        self.linear_out = Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None, return_kv=False, return_attn_probs=False):
        q = self.linear_q(q)  # (B, seq_len, d_model)
        # (B, seq_len, d_model) -> (B, seq_len, n_heads, d_k)
        q = q.view(q.size(0), q.size(1), self.n_heads, self.d_k)
        # Attention is on (B, seq_len, d_k)
        q = q.transpose(1, 2)  # (B, n_heads, seq_len, d_k)
        k = (
            self.linear_k(k)
            .view(k.size(0), k.size(1), self.n_heads, self.d_k)
            .transpose(1, 2)
        )
        v = (
            self.linear_v(v)
            .view(v.size(0), v.size(1), self.n_heads, self.d_k)
            .transpose(1, 2)
        )

        attn = self.attention(q, k, v, mask)

        # Concatenate
        attn = attn.transpose(1, 2).contiguous()
        attn = attn.view(attn.size(0), attn.size(1), -1)

        # Output linear layer
        attn = self.linear_out(attn)  # (B, seq_len, d_model)
        if return_kv:
            return attn, (k, v)
        return attn


if __name__ == "__main__":
    d_model = 512
    n_heads = 8
    batch_size = 2
    seq_len = 10

    multi_head_attention = MultiHeadAttention(d_model, n_heads)

    q = torch.randn(batch_size, seq_len, d_model)
    k = torch.randn(batch_size, seq_len, d_model)
    v = torch.randn(batch_size, seq_len, d_model)

    output = multi_head_attention(q, k, v)
    print(f"Output shape: {output.shape}")
    assert output.shape == (
        batch_size,
        seq_len,
        d_model,
    ), f"Expected output shape {(batch_size, seq_len, d_model)}, got {output.shape}"
