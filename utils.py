import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Linear, self).__init__()

        # nn.Parameter makes it trainable
        self.w = nn.Parameter(torch.empty(input_dim, output_dim))
        self.b = nn.Parameter(torch.empty(output_dim))

        # Initialization (can be other ones too)
        nn.init.xavier_uniform_(self.w)
        nn.init.zeros_(self.b)

    def forward(self, x):
        return x @ self.w + self.b


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, bias=False, affine=False, eps=1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps

        self.affine = affine
        # When afffine, we shift the normalized output by a scale and bias,
        # this allows the model to be flexible in learning
        if affine:
            if bias == False:
                raise ValueError("If affine is True, bias must also be True")
            self.gamma = nn.Parameter(torch.ones(normalized_shape))
            self.beta = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.gamma = None
            self.beta = None

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)

        # Variance can be biased or unbiased, biased if /n, unbiased if /n-1
        # This is not related to the bias in the layer norm
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        y = (x - mean) / (torch.sqrt(var + self.eps))

        if self.affine:
            y = y * self.gamma + self.beta

        return y


class PositionWiseFeedForward(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=2048, output_dim=512):
        super().__init__()
        self.linear1 = Linear(input_dim, hidden_dim)
        self.linear2 = Linear(hidden_dim, output_dim)

    def forward(self, x):
        # X (batch_size, seq_len, d_model)
        # It is position-wise, since we apply the FFN to each position in seq_len

        # print(f"Input shape: {x.shape}")
        x = self.linear1(x)
        # print(f"After first linear layer shape: {x.shape}")
        # F is recommended to be used
        x = F.relu(x)
        x = self.linear2(x)
        # print(f"After second linear layer shape: {x.shape}")

        return x


if __name__ == "__main__":

    seq_len = 10
    d_model = 512

    x = torch.randn(2, seq_len, d_model)

    pffn = PositionWiseFeedForward(
        input_dim=d_model, hidden_dim=2048, output_dim=d_model
    )
    output = pffn(x)
    assert output.shape == (
        2,
        seq_len,
        d_model,
    ), f"Expected output shape {(2, d_model)}, got {output.shape}"
