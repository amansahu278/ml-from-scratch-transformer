import torch.nn as nn
import torch
import math


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=3000):
        super().__init__()
        
        # The encodings are the same dimension as the model, so taht they can be summed
        self.pos_enc = torch.zeros(max_len, d_model)

        # They use log and exp since it is quicker and more stable
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(1000.0) / d_model))
        pos = torch.arange(0, max_len).unsqueeze(1)  # (seq_len, 1)
        
        self.pos_enc[:, ::2] = torch.sin(pos * div_term)
        self.pos_enc[:, 1::2] = torch.cos(pos * div_term)

        self.pos_enc = self.pos_enc.unsqueeze(0)  # Add batch dimension (1, max_len, d_model)
        
    def forward(self, x):
        print(self.pos_enc[:, :x.size(1), :])
        seq_len = x.size(1)
        return x + self.pos_enc[:, :seq_len, :]

if __name__ == "__main__":
    d_model = 512
    max_len = 100
    pos_enc = PositionalEncoding(d_model, max_len)
    x = torch.randn(1, max_len, d_model)  # (batch_size, seq_len, d_model)
    output = pos_enc(x)
    print(output.shape)  # Should be (1, max_len, d_model)
    print(output)