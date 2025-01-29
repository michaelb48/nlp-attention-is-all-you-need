import torch.nn as nn
import math
import torch

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    # this code draws heavily from https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html#torch.nn.functional.scaled_dot_product_attention
    def forward(self, query, key, value, attn_mask=None, scale=None):
        scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale

        # dim: batch x head x len x d_k
        attn = torch.matmul(query, key.transpose(-2, -1)) * scale_factor

        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask == 0, float('-inf'))

        return torch.matmul(torch.softmax(attn, dim=-1), value)

class MHAttention(nn.Module):
    def __init__(self,
                 d_model: int,
                 t_heads: int,
                 d_query_key_head: int,
                 d_value_head: int,
                 t_dot_product: bool = True
                 ):
        super().__init__()

        self.d_model = d_model
        self.t_heads = t_heads

        # store dimensions used for the separate linear projections of each attention head
        self.d_k = d_query_key_head
        self.d_v = d_value_head

        # create unique weight matrix for each attention head and type of input; bias is set to false as we only want linear projections per head
        self.key_proj = nn.Linear(self.d_model, self.t_heads * self.d_k, bias=False)
        self.query_proj = nn.Linear(self.d_model, self.t_heads * self.d_k, bias=False)
        self.value_proj = nn.Linear(self.d_model, self.t_heads * self.d_v, bias=False)

        # use our scaled-dot-product-attention
        if t_dot_product:
            self.scaled_dot_product_attention = ScaledDotProductAttention()
        else:
            self.scaled_dot_product_attention = # Alibi class goes here

        # use another linear layer to project the concatenation after attention of each head was computed
        self.concat_proj = nn.Linear(self.t_heads * self.d_v, d_model, bias=False)

    def forward(self, query, key, value, seq_mask):
        # get different view of the matrices to represent separate attention heads; use torch.view to create copy to not change residual
        # used the second answer in this https://stackoverflow.com/questions/42479902/what-does-view-do-in-pytorch
        batch_size = query.size(0)
        queries_len = query.size(1)
        keys_len = key.size(1)
        values_len = value.size(1)

         # mask is only passed for the sequence; the size of mask is batch x se_len x d_model; in order to perform broadcast we need one more dimension for every head
        seq_mask = seq_mask.unsqueeze(1)  # dimension is one because 0 is batch

        # Now we need to transpose the heads like in the lecture:b x n x h x d/h -> b x h x n x d/h
        queries = self.query_proj(query).view(batch_size, queries_len, self.t_heads, self.d_k).transpose(1,2)
        keys = self.key_proj(key).view(batch_size, keys_len, self.t_heads, self.d_k).transpose(1,2)
        values = self.value_proj(value).view(batch_size, values_len, self.t_heads, self.d_v).transpose(1,2)

        # pass to the scaled dot product attention and transpose back
        #attention_values = F.scaled_dot_product_attention(queries, keys, values, seq_mask).transpose(1,2)
        attention_values = self.scaled_dot_product_attention(queries, keys, values, seq_mask).transpose(1,2)

        # get original view of the matrix to represent the concatenation of attention heads and return the concatenation through a linear layer
        return self.concat_proj(attention_values.contiguous().view(batch_size, queries_len, -1))


class MHAttentionSublayer(nn.Module):
    """Implements the MHAttention sublayer of a transformer."""

    def __init__(self,
                 d_model: int,
                 t_heads: int,
                 d_query_key_head: int,
                 d_value_head: int,
                 t_dropout: float,
                t_dot_product: bool
                ):
        super().__init__()
        self.multi_headed_attention = MHAttention(d_model=d_model,
                                                  t_heads=t_heads,
                                                  d_query_key_head=d_query_key_head,
                                                  d_value_head=d_value_head,
                                                 t_dot_product=t_dot_product)

        self.dropout = nn.Dropout(t_dropout)

        self.normalization = nn.LayerNorm(d_model)

    def forward(self, query, key, value, seq_mask):
        # store the original tensor for the residual connection; as none of the forward operations happen inplace we can simply assign
        residual = query

        # let the input token pass forward through the operation; apply dropout before adding residual
        op_result = residual + self.dropout(self.multi_headed_attention(query=query, key=key, value=value, seq_mask=seq_mask))

        # return the batch normalized result
        return self.normalization(op_result)


class FeedForwardUnit(nn.Module):
    '''Feedforward sublayer used in the encoder and decoder. According to the paper each of the layers in the encoder and decoder contains a fully connected feed-forward network, which is applied to each position separately and identically.
        This feedforward layer consists of a linear transformation to the inner dimension followed by a ReLU activation. We finish the operation by applying a second linear transformation to get back to d_model. The formula
        in the paper includes additive bias which is why we will not modify it.'''

    def __init__(self, d_model: int, d_ff_inner: int):
        super().__init__()

        # First linear transformation to transform the input token from the model dimension to the inner layer dimension
        self.d_model_to_d_inner = nn.Linear(d_model, d_ff_inner)

        # ReLU activation function
        self.relu = nn.ReLU()

        # Second linear transformation to get back to the model dimensions
        self.d_inner_to_d_model = nn.Linear(d_ff_inner, d_model)

        self.feed_forward_stack = nn.Sequential(
            self.d_model_to_d_inner,
            self.relu,
            self.d_inner_to_d_model
        )

    def forward(self, x):
        return self.feed_forward_stack(x)


class FeedForwardSublayer(nn.Module):
    """Implements a sublayer of a transformer."""

    def __init__(self,
                 d_model: int,
                 d_ff_inner: int,
                 t_dropout: float):
        super().__init__()
        self.linear_proj = FeedForwardUnit(d_model=d_model, d_ff_inner=d_ff_inner)

        self.dropout = nn.Dropout(t_dropout)

        self.normalization = nn.LayerNorm(d_model)

    def forward(self, value):
        # store the original tensor for the residual connection; as none of the forward operations happen inplace we can simply assign
        residual = value

        # let the input token pass forward through the operation before applying dropout before adding residual
        op_result = residual + self.dropout(self.linear_proj(value))

        # return the batch normalized result
        return self.normalization(op_result)
