import torch.nn as nn
import torch
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        empty_positions = torch.zeros(max_len, d_model)
        
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # 10 000 was used in the paper
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Calculates sin for even positions i in d_model and cos for odd
        empty_positions[:, 0::2] = torch.sin(position * div_term)
        empty_positions[:, 1::2] = torch.cos(position * div_term)

        pos_encoding = empty_positions.unsqueeze(0)
        self.register_buffer('pos_encoding', pos_encoding)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]