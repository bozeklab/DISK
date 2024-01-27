import os

import torch
import torch.nn as nn
from omegaconf import DictConfig
from DISK.models.distribution_head import NormalOutput


# network module only set encoder to be bidirectional
class BiGRU(nn.Module):
    def __init__(self, input_size: int, output_size: int, cfg: DictConfig, mu_sigma: bool=False, device: str = 'cpu'):
        super(BiGRU, self).__init__()
        self.hidden_size = cfg.network.size_layer
        self.device = device
        self.num_layers = cfg.network.num_layers

        self.gru = nn.GRU(input_size, self.hidden_size, num_layers=cfg.network.num_layers,
                          bidirectional=True, batch_first=True).to(device)
        self.dropout = nn.Dropout(cfg.network.dropout)
        self.linear = nn.Linear(self.hidden_size, output_size, bias=True).to(device)

        self.mu_sigma = mu_sigma
        if self.mu_sigma:
            self.distribution_output = NormalOutput(dim=output_size)
            self.parameter_projection = self.distribution_output.get_parameter_projection(self.hidden_size).to(device)

        self.init_weights()

    def init_weights(self):
        # initialize weights
        for param in list(self.gru.parameters()):
            if param.dim() == 2:
                nn.init.xavier_uniform_(param)

    def forward(self, input_tensor, **kwargs):
        self.gru.flatten_parameters()
        if 'key_padding_mask' in kwargs:
            lengths = kwargs['key_padding_mask']
            in_ = nn.utils.rnn.pack_padded_sequence(input_tensor, lengths, batch_first=True)
        else:
            in_ = input_tensor
        previous_hidden = self.init_hidden(input_tensor).to(self.device)
        enout_tmp, hidden_tmp = self.gru(in_, previous_hidden)
        rnn_out = (enout_tmp[:, :, :self.hidden_size] +
                   enout_tmp[:, :, self.hidden_size:])
        if self.mu_sigma:
            out = self.parameter_projection(rnn_out)  # outputs.last_hidden_state
        else:
            out = self.linear(self.dropout(rnn_out))
        return out, hidden_tmp

    def init_hidden(self, input_tensor):
        return torch.randn((2 * self.num_layers, input_tensor.shape[0], self.hidden_size))
