import torch.nn as nn
from Sublayer import MHAttentionSublayer, FeedForwardSublayer

class DecoderLayer(nn.Module):
    def __init__(self,
                 d_model: int,
                 d_ff_inner: int,
                 t_dec_heads: int,
                 d_query_key_head: int,
                 d_value_head: int,
                 t_dropout: float
                 ):
        super().__init__()

        self.masked_attention_sublayer = MHAttentionSublayer(d_model=d_model,
                                                             t_heads=t_dec_heads,
                                                             d_query_key_head=d_query_key_head,
                                                             d_value_head=d_value_head,
                                                             t_dropout=t_dropout)

        self.attention_sublayer = MHAttentionSublayer(d_model=d_model,
                                                      t_heads=t_dec_heads,
                                                      d_query_key_head=d_query_key_head,
                                                      d_value_head=d_value_head,
                                                      t_dropout=t_dropout)

        self.feed_forward_sublayer = FeedForwardSublayer(d_model=d_model,
                                                         d_ff_inner=d_ff_inner,
                                                         t_dropout=t_dropout)

    def forward(self, seq, mask, encoder_output, encoder_mask, t_dot_product):
        # calculate masked attended values
        masked_attn_values = self.masked_attention_sublayer(query=seq, key=seq, value=seq, seq_mask=mask,
                                                            t_dot_product=t_dot_product)

        # calculate attention with encoder output
        attn_values = self.attention_sublayer(query=masked_attn_values, key=encoder_output, value=encoder_output,
                                              seq_mask=encoder_mask, t_dot_product=t_dot_product)

        # linear projection with the same matrix identically for all words
        return self.feed_forward_sublayer(value=attn_values)


class Decoder(nn.Module):
    def __init__(self,
                 d_model: int,
                 d_ff_inner: int,
                 t_dec_layer_num: int,
                 t_dec_heads: int,
                 d_query_key_head: int,
                 d_value_head: int,
                 t_dropout: float
                 ):
        super().__init__()

        layer_list = [DecoderLayer(d_model=d_model,
                                   d_ff_inner=d_ff_inner,
                                   t_dec_heads=t_dec_heads,
                                   d_query_key_head=d_query_key_head,
                                   d_value_head=d_value_head,
                                   t_dropout=t_dropout) for _ in range(t_dec_layer_num)]
        self.decoder_layer_stack = nn.ModuleList(layer_list)

    def forward(self, encoder_output, encoder_mask, seq, mask, t_dot_product):
        layer_out = seq

        # run the sequence through every layer of the encoder with the mask
        for layer in self.decoder_layer_stack:
            layer_out = layer(layer_out, mask, encoder_output, encoder_mask, t_dot_product)

        # return the encoder output as d_model tensor
        return layer_out
