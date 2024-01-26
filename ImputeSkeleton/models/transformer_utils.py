import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_

from omegaconf import DictConfig
from einops import rearrange


def _get_attn_block(attn_type: str, d_model: int, num_heads: int):
    if attn_type == 'standard':
        return MultiHeadSelfAttention(d_model, num_heads)
    else:
        raise ValueError('Unknown attention type {}'.format(attn_type))


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == "elu":
        return F.elu
    elif activation == "lrelu":
        return F.leaky_relu
    elif activation == "tanh":
        return F.tanh
    raise ValueError("Given activation ({}) not supported".format(activation))


class FixedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000, device='cpu'):
        super(FixedPositionalEncoding, self).__init__()
        pos = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1)
        pe = pos / torch.pow(10000, torch.arange(0, d_model, 2, dtype=torch.float32) / d_model)
        self.pe = torch.zeros((1, max_len, d_model)).to(device)
        self.pe[:, :, 0::2] = torch.sin(pe)
        self.pe[:, :, 1::2] = torch.cos(pe)

    def forward(self, x):
        return x + self.pe[:, :x.shape[1], :]


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1024):
        super().__init__()
        self.positional_embedding = nn.Embedding(max_len, d_model)
        self.register_buffer('positions', torch.arange(max_len))

    def forward(self, x):
        return x + self.positional_embedding(self.positions)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, activation: str):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.act = _get_activation_fn(activation)

    def forward(self, x):
        return self.linear2(self.act(self.linear1(x)))


class Normalization(nn.Module):
    def __init__(self, method, d_model):
        super().__init__()
        assert method in ["layer", "batch", "none"]
        if method == "layer":
            self.norm = nn.LayerNorm(d_model)
        elif method == "batch":
            self.norm = nn.BatchNorm1d(d_model)
        elif method == "none":
            self.norm = lambda x: x
        self.method = method

    def forward(self, x):
        # x.shape = [batch, seq, features]
        if self.method == "batch":
            return self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        return self.norm(x)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super(MultiHeadSelfAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        attn_out, attn_weights = self.attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        return attn_out


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, dim_ff: int, num_heads: int,
                 activation: str = 'relu',
                 norm_first: bool = True,
                 norm: str = 'layer',
                 attn_type: str = 'standard'):
        super(EncoderLayer, self).__init__()
        self.d_model = d_model
        self.norm_first = norm_first

        self.norm1 = Normalization(method=norm, d_model=d_model)
        self.norm2 = Normalization(method=norm, d_model=d_model)
        self.self_attention = _get_attn_block(attn_type=attn_type, d_model=d_model, num_heads=num_heads)
        self.ffn = FeedForward(d_model, dim_ff, activation)

    def forward(self, x, key_padding_mask=None):
        if self.norm_first:
            x = x + self.self_attention(self.norm1(x), key_padding_mask=key_padding_mask)
            x = x + self.ffn(self.norm2(x))
        else:
            x = self.norm1(x + self.self_attention(x, key_padding_mask=key_padding_mask))
            x = self.norm2(x + self.ffn(x))

        return x


class CustomizedEncoderLayer(nn.Module):  # PreNorm
    def __init__(self, d_model: int, d_ff: int, num_heads: int, max_seq_len: int, n_keypoints: int, dropout: float,
                 activation: str = 'relu', norm: str = 'layer'):
        super(CustomizedEncoderLayer, self).__init__()
        self.norm1 = Normalization(method=norm, d_model=d_model)
        self.norm2 = Normalization(method=norm, d_model=d_model)
        self.norm3 = Normalization(method=norm, d_model=d_model)

        self.dropout = nn.Dropout(dropout)
        self.global_attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)

        self.d_model = d_model
        self.n_keypoints = n_keypoints
        self.max_seq_len = max_seq_len

        self.mha_t = nn.MultiheadAttention(d_model, num_heads, batch_first=True)  # for local attention

        self.ffn = FeedForward(d_model, d_ff, activation)

    def forward(self, x, key_padding_mask=None):
        # local attention (from "Long-Range Transformers for Dynamic Spatiotemporal Forecasting" paper)
        y = self.norm1(x)
        y = rearrange(y, "batch (len kp) dim -> (batch kp) len dim", kp=self.n_keypoints)
        attn_t, _ = self.mha_t(y, y, y, key_padding_mask)  # [kp * batch, len, d_model], weights have shape [kp * batch, len, len]
        attn_t = rearrange(attn_t, "(batch kp) len dim -> batch (len kp) dim", kp=self.n_keypoints)
        x = x + self.dropout(attn_t)

        y = self.norm2(x)
        attn_g, _ = self.global_attention(y, y, y)
        x = x + self.dropout(attn_g)

        x = x + self.dropout(self.ffn(self.norm3(x)))
        return x



class VanillaTransformer(nn.Module):  # is left here to compare with PyTorch implementation
    def __init__(self, input_size: int, output_size: int, max_seq_len: int, cfg: DictConfig, device: str = 'cpu'):
        super(VanillaTransformer, self).__init__()
        self.device = device
        self.d_model = cfg.network.d_model

        self.pos_encoder = FixedPositionalEncoding(d_model=self.d_model, device=device, max_len=max_seq_len).to(device)

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model,
                                                   nhead=cfg.network.num_heads,
                                                   dim_feedforward=cfg.network.dim_ff,
                                                   dropout=cfg.network.dropout,
                                                   activation=cfg.network.activation,
                                                   norm_first=cfg.network.norm_first,
                                                   batch_first=True
                                                   )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, cfg.network.num_layers).to(device)

        self.in_proj = nn.Linear(input_size, self.d_model, bias=True).to(device)
        self.out_linear = nn.Linear(self.d_model, output_size, bias=True).to(device)

        self.init_weights()

    def init_weights(self):
        for param in self.transformer_encoder.parameters():
            if param.dim() > 1:
                xavier_uniform_(param)

    def forward(self, input_tensor):
        out = self.pos_encoder(input_tensor)
        out = self.in_proj(out)
        out = self.transformer_encoder(out)
        out = self.linear(out)
        return out
