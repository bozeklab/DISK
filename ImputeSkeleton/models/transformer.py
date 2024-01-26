import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_
from omegaconf import DictConfig
from ImputeSkeleton.models.transformer_utils import EncoderLayer, Normalization, LearnablePositionalEncoding, \
    FixedPositionalEncoding
from ImputeSkeleton.models.distribution_head import NormalOutput


class InputEncoding(nn.Module):
    def __init__(self, input_size: int, d_model: int, n_keypoints: int, max_seq_len: int):
        super(InputEncoding, self).__init__()
        self.d_model = d_model
        self.n_keypoints = n_keypoints
        self.max_seq_len = max_seq_len
        self.input_size = input_size


class InputEncodingMixed(InputEncoding):
    def __init__(self, input_size: int, d_model: int, n_keypoints: int, max_seq_len: int, device: str, enc_type: str):
        super(InputEncodingMixed, self).__init__(input_size, d_model, n_keypoints, max_seq_len)
        self.in_proj = nn.Linear(input_size * n_keypoints, self.d_model)
        if enc_type == 'learnable':
            self.pos_encoder = LearnablePositionalEncoding(d_model=self.d_model, max_len=max_seq_len)
        else:
            self.pos_encoder = FixedPositionalEncoding(d_model=self.d_model, max_len=max_seq_len, device=device)

    def forward(self, x, missing_mask):
        x = x.view(-1, self.max_seq_len, self.n_keypoints * self.input_size)
        x = self.in_proj(x)
        x = self.pos_encoder(x)
        return x


class InputEncodingIndependent(InputEncoding):
    def __init__(self, input_size: int, d_model: int, n_keypoints: int, max_seq_len: int, device: str):
        super(InputEncodingIndependent, self).__init__(input_size, d_model, n_keypoints, max_seq_len)

        self.in_proj = nn.Linear(input_size, self.d_model)

        self.time_emb = nn.Embedding(num_embeddings=max_seq_len, embedding_dim=self.d_model)
        self.keypoints_emb = nn.Embedding(num_embeddings=n_keypoints, embedding_dim=self.d_model)
        self.given_emb = nn.Embedding(num_embeddings=2, embedding_dim=self.d_model)

        self.register_buffer('time_pos', torch.arange(self.max_seq_len))
        self.register_buffer('keypoints_pos', torch.arange(self.n_keypoints))

    def forward(self, x, missing_mask):
        x = x.view(-1, self.max_seq_len * self.n_keypoints, self.input_size)
        x = self.in_proj(x)
        t = self.time_emb(self.time_pos.repeat_interleave(self.n_keypoints))
        k = self.keypoints_emb(self.keypoints_pos.repeat(self.max_seq_len))
        missing_mask = missing_mask.view(missing_mask.shape[0], self.max_seq_len * self.n_keypoints)
        x = x + t + k + self.given_emb(missing_mask)
        return x


class TransformerModel(nn.Module):
    def __init__(self, input_size: int, output_size: int, max_seq_len: int, n_keypoints: int, cfg: DictConfig,
                 mu_sigma: bool=False, device: str = 'cpu'):
        super(TransformerModel, self).__init__()
        self.device = device
        self.d_model = cfg.network.d_model
        self.num_heads = cfg.network.num_heads
        self.num_layers = cfg.network.num_layers
        self.norm_first = cfg.network.norm_first

        if cfg.network.input_type == 'mixed':
            self.proj_input = InputEncodingMixed(input_size, self.d_model, n_keypoints, max_seq_len, self.device,
                                                 enc_type=cfg.network.encoding).to(device)
        else:
            self.proj_input = InputEncodingIndependent(input_size, self.d_model, n_keypoints, max_seq_len, device).to(device)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model=self.d_model,
                                                          dim_ff=cfg.network.dim_ff,
                                                          num_heads=self.num_heads,
                                                          activation=cfg.network.activation,
                                                          attn_type=cfg.network.attn_type,
                                                          norm_first=cfg.network.norm_first,
                                                          norm=cfg.network.norm) for _ in range(self.num_layers)]).to(device)

        if self.norm_first:
            self.final_norm = Normalization(method=cfg.network.norm, d_model=self.d_model).to(device)

        self.mu_sigma = mu_sigma
        if self.mu_sigma:
            self.distribution_output = NormalOutput(dim=output_size)
            self.parameter_projection = self.distribution_output.get_parameter_projection(self.d_model).to(device)
        else:
            if cfg.network.input_type == 'mixed':
                self.out_linear = nn.Linear(self.d_model, output_size * n_keypoints, bias=True).to(device)
            else:
                self.out_linear = nn.Linear(self.d_model, output_size, bias=True).to(device)

        self.init_weights()

    def init_weights(self):
        for layer in self.encoder_layers:
            for param in layer.parameters():
                if param.dim() > 1:
                    xavier_uniform_(param)

    def forward(self, x, missing_mask, **kwargs):  # can be generalized to one model
        if 'key_padding_mask' in kwargs:
            # when sequences don't have the same length
            key_padding_mask = kwargs['key_padding_mask']
        else:
            key_padding_mask = None

        x = self.proj_input(x, missing_mask)

        for i in range(self.num_layers):
            x = self.encoder_layers[i](x, key_padding_mask=key_padding_mask)

        if self.norm_first:
            x = self.final_norm(x)

        if self.mu_sigma:
            x = self.parameter_projection(x)
        else:
            x = self.out_linear(x)
        return x
