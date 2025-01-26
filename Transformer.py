import torch.nn as nn
import torch
from PositionalEncoding import PositionalEncoding
from Decoder import Decoder
from Encoder import Encoder
import math

class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self,
            n_vocab_len: int,
            i_vocab_padding: int,
            d_model: int = 512,
            d_dec_ff_inner: int = 2048,
            t_dec_heads: int = 8,
            t_dec_layer_num: int = 6,
            d_enc_ff_inner: int = 2048,
            t_enc_heads: int = 8,
            t_enc_layer_num: int = 6,
            d_query_key_head: int = 64,
            d_value_head: int = 64,
            t_dropout: float = 0.1,
            device: str = 'cpu'
    ):

        super().__init__()

        self.device = device

        self.d_model = d_model
        self.padding = i_vocab_padding

        # define embedding layer and positional encoding for encoder and decoder; according to the paper the embedding layer weights are shared
        self.vocab_embedding = nn.Embedding(num_embeddings=n_vocab_len, embedding_dim=d_model,
                                            padding_idx=i_vocab_padding)
        self.positional_encoding = PositionalEncoding(d_model=self.d_model)

        # define decoder and encoder
        self.encoder = Encoder(
            d_model=d_model,
            d_ff_inner=d_enc_ff_inner,
            t_enc_layer_num=t_enc_layer_num,
            t_enc_heads=t_enc_heads,
            d_query_key_head=d_query_key_head,
            d_value_head=d_value_head,
            t_dropout=t_dropout
        )

        self.decoder = Decoder(
            d_model=d_model,
            d_ff_inner=d_dec_ff_inner,
            t_dec_layer_num=t_dec_layer_num,
            t_dec_heads=t_dec_heads,
            d_query_key_head=d_query_key_head,
            d_value_head=d_value_head,
            t_dropout=t_dropout
        )

        # define the linear layer for the projection before softmax; this layer shares the weights with the embedding layers
        self.linear_to_vocab = nn.Linear(d_model, n_vocab_len, bias=False)

        # according to the paper the weights were scaled in the embedding
        self.scaling = math.sqrt(self.d_model)

        # dropout after the sum of embeddings and positional encoding
        self.dropout = nn.Dropout(t_dropout)

        # init parameters
        for p in self.parameters():
            if len(p.size()) > 1:
                nn.init.xavier_normal_(p)

    def get_attention_mask(self, padded_seq: torch.Tensor, masked_attention:bool = False) -> torch.Tensor:
        # use broadcast to create mask
        mask = (padded_seq != self.padding).unsqueeze(-2).to(self.device)
        if masked_attention:
            seq_len = padded_seq.size()[-1]
            decoder_mask = torch.ones(1, seq_len, seq_len).to(self.device)
            upper_triangle = torch.triu(decoder_mask,diagonal=1)
            lower_triangle = 1 - upper_triangle
            mask = lower_triangle.bool() & mask
        return mask

    def forward(self, input_seq_padded, target_seq_padded, t_dot_product: bool = True):
        # create attention masks for relevant tokens
        src_seq_mask = self.get_attention_mask(input_seq_padded)
        target_seq_mask = self.get_attention_mask(target_seq_padded,masked_attention=True)

        # get embeddings and scale with scaling value
        src_seq_embedding = self.vocab_embedding(input_seq_padded) * (self.scaling)
        target_seq_embedding= self.vocab_embedding(target_seq_padded) * (self.scaling)

        # add positional embeddings
        src_seq_embedding_position_encoded = self.positional_encoding(src_seq_embedding)
        target_seq_embedding_position_encoded = self.positional_encoding(target_seq_embedding)

        # apply dropout
        src_seq_embedding_position_encoded = self.dropout(src_seq_embedding_position_encoded)
        target_seq_embedding_position_encoded = self.dropout(target_seq_embedding_position_encoded)

        # compute encoder and decoder output
        enc_output = self.encoder(src_seq_embedding_position_encoded, src_seq_mask, t_dot_product)
        dec_output = self.decoder(enc_output, src_seq_mask, target_seq_embedding_position_encoded, target_seq_mask, t_dot_product)

        # compute linear projection back to vocab
        return self.linear_to_vocab(dec_output)