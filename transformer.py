import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder import Encoder
from decoder import Decoder
from utils import Linear


class Transformer(nn.Module):
    def __init__(self, input_dim, d_model, n_heads, num_layers=6):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_layers = num_layers

        self.encoder = Encoder(input_dim, d_model, n_heads, num_layers)
        self.decoder = Decoder(input_dim, d_model, n_heads, num_layers)

        self.output_proj = Linear(d_model, input_dim)
        # Done to reduce parameters
        self.output_proj.weight = self.decoder.emebedding.weight
        # The output project and the decoder embeddings will operate over the same vocabulary
        # Hence for representational symmetry

    def forward(self, src, tgt, return_dict=False):

        encoder_output = self.encoder(src)
        decoder_output = self.decoder(tgt, encoder_output, return_kv=return_dict)
        if return_dict:
            decoder_output, kv_list = decoder_output

        out = self.output_proj(decoder_output)
        # Logits are passed to loss function, not probabilities, so removing softmax here
        # out = F.softmax(out, dim=-1)

        # I remember having to use past_key_values when implementing
        # custom KV cache management, so i thought i'd implement it here
        if return_dict:
            return {"logits": out, "past_key_values": kv_list}
        return out

    def generate(self, src, sos_token_id=1, max_len=50):
        self.eval()

        encoder_output = self.encoder(src)
        batch_size = src.size(0)
        device = src.device

        tgt = torch.full((batch_size, 1), sos_token_id, device=device, dtype=torch.long) # Embedding requires long tensor

        for _ in range(max_len):
            # We can also call forward here,
            # But for more customization, say kv_caching etc, it is better to call the decoder instead
            decoder_output = self.decoder(tgt, encoder_output)
            logits_for_next_token = self.output_proj(decoder_output[:, -1, :])
            next_token = torch.argmax(logits_for_next_token, dim=-1, keepdim=True) # The vocab token with the highest value

            tgt = torch.cat([tgt, next_token], dim=1)        
        
        return tgt
