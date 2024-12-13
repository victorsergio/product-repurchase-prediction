import torch.nn as nn
import torch, math
import time
from positional_encoder import PositionalEncoder

"""
The architecture is based on the paper “Attention Is All You Need”.
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017.
"""


class Transformer(nn.Module):

    def __init__(self, model_config):
        super(Transformer, self).__init__()

        n_encoder_layers = model_config["n_encoder_layers"]
        input_size = model_config["input_size"]
        dim_val = model_config["dim_val"]
        dropout_encoder = model_config["dropout_encoder"]
        dropout_pos_enc = model_config["dropout_pos_enc"]
        n_heads = model_config["n_heads"]
        num_predicted_features = model_config["num_predicted_features"]
        max_seq_len = model_config["max_seq_len"]

        self.encoder_input_layer = nn.Linear(in_features=input_size, out_features=dim_val)
        self.positional_encoding_layer = PositionalEncoder(d_model=dim_val, dropout=dropout_pos_enc,
                                                           max_seq_len=max_seq_len, batch_first=True)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=dim_val, nhead=n_heads, dropout=dropout_encoder,
                                                        batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=n_encoder_layers)

        self.linear_mapping = nn.Linear(in_features=dim_val, out_features=num_predicted_features)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.linear_mapping.bias.data.zero_()
        self.linear_mapping.weight.data.uniform_(-initrange, initrange)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, device):
        src = self.encoder_input_layer(src)
        src = self.positional_encoding_layer(src)

        mask = self._generate_square_subsequent_mask(src.size(1)).to(device)

        output = self.encoder(src, mask)
        output = self.linear_mapping(output[:, -1, :])
        output = output.unsqueeze(1)

        return output
