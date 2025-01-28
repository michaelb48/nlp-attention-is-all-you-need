import torch.nn as nn
from Sublayer import MHAttentionSublayer, FeedForwardSublayer

class EncoderLayer(nn.Module):
    def __init__(self,
                 d_model: int,
                 d_ff_inner: int,
                 t_enc_heads: int,
                 d_query_key_head: int,
                 d_value_head: int,
                 t_dropout: float,
                 t_dot_product: bool
                 ):
        super().__init__()

        self.attention_sublayer = MHAttentionSublayer(d_model=d_model,
                                                      t_heads=t_enc_heads,
                                                      d_query_key_head=d_query_key_head,
                                                      d_value_head=d_value_head,
                                                      t_dropout=t_dropout,
                                                      t_dot_product=t_dot_product)

        self.feed_forward_sublayer = FeedForwardSublayer(d_model=d_model,
                                                         d_ff_inner=d_ff_inner,
                                                         t_dropout=t_dropout)
        
    def forward(self, seq, mask):
        return self.feed_forward_sublayer(self.attention_sublayer(query=seq, key=seq, value=seq, seq_mask=mask))


class Encoder(nn.Module):
    def __init__(self,
                 d_model: int,
                 d_ff_inner: int,
                 t_enc_layer_num: int,
                 t_enc_heads: int,
                 d_query_key_head: int,
                 d_value_head: int,
                 t_dropout: float,
                 t_dot_product: bool
                 ):
        super().__init__()

        layer_list = [EncoderLayer(d_model=d_model,
                                   d_ff_inner=d_ff_inner,
                                   t_enc_heads=t_enc_heads,
                                   d_query_key_head=d_query_key_head,
                                   d_value_head=d_value_head,
                                   t_dropout=t_dropout,
                                   t_dot_product=t_dot_product) for _ in range(t_enc_layer_num)]
        self.encoder_layer_stack = nn.ModuleList(layer_list)

        self.normalization = nn.LayerNorm(d_model)

    def forward(self, seq, mask):
        layer_out = seq

        # run the sequence through every layer of the encoder with the mask
        for layer in self.encoder_layer_stack:
            layer_out = layer(layer_out, mask)

        # return the encoder output as d_model tensor
        return self.normalization(layer_out)
