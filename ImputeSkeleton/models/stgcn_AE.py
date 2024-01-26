import torch
import torch.nn as nn

from ImputeSkeleton.models.graph import Graph
from ImputeSkeleton.models.st_gcn import st_gcn, rst_gcn


class STGCN_Model(nn.Module):
    r"""Spatial temporal graph convolutional networks.

    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_args (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """

    def __init__(self, in_channels, num_layers, graph_args,
                 edge_importance_weighting, temporal_kernel_size=9,
                 dim_start=68, out_channels=3, **kwargs):
        super().__init__()

        # load graph
        self.graph = Graph(**graph_args)
        enc_A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('enc_A', enc_A)

        # build networks
        spatial_kernel_size = self.enc_A.size(0)
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * self.enc_A.size(1))
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        module_list = []
        dim_in = int(in_channels)
        dim_out = int(dim_start)
        self.out_channels = out_channels
        for i in range(num_layers):
            if i == 0:
                module_list.append(st_gcn(in_channels, dim_out, kernel_size, 1, residual=False, **kwargs0))
            else:
                module_list.append(st_gcn(dim_in, dim_out, kernel_size, 1, **kwargs))
            dim_in = dim_out
            dim_out *= 2

        self.enc_networks = nn.ModuleList(module_list)
        ## originally
        # self.enc_networks = nn.ModuleList((
        #     st_gcn(in_channels, 64, kernel_size, 1, residual=False, **kwargs0),
        #     st_gcn(64, 128, kernel_size, 1, **kwargs),
        #     st_gcn(128, 256, kernel_size, 1, **kwargs)
        # ))

        # add reconstruction branch
        self.rec_graph = Graph(**graph_args)
        rec_A = torch.tensor(self.rec_graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('rec_A', rec_A)
        spatial_kernel_size = self.rec_A.size(0)
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        module_list = []
        dim_out = int(dim_in / 2)
        for i in range(num_layers):
            if i == 0:
                module_list.append(rst_gcn(dim_in, dim_out, kernel_size, 1, residual=False, **kwargs))
            else:
                module_list.append(rst_gcn(dim_in, dim_out, kernel_size, 1, residual=True, **kwargs))
            dim_in = dim_out
            dim_out = int(dim_out / 2)
        # In the following line, I replaced C by 3 here because we want the output to be in 3D where the input has a 4th dimension which is the mask
        module_list.append(rst_gcn(dim_in, self.out_channels, kernel_size, 1, residual=True, **kwargs))
        self.rec_networks = nn.ModuleList(module_list)
        ## originally
        # self.rec_networks = nn.ModuleList((
        #     rst_gcn(256, 128, kernel_size, 1, residual=False, **kwargs),
        #     rst_gcn(128, 64, kernel_size, 1, residual=True, **kwargs),
        #     rst_gcn(64, 32, kernel_size, 1, residual=True, **kwargs),
        #     rst_gcn(32, in_channels, kernel_size, 1, residual=True, **kwargs)
        # ))

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.enc_edge_imp = nn.ParameterList(
                [nn.Parameter(torch.ones(self.enc_A.size())) for _ in self.enc_networks])
            self.rec_edge_imp = nn.ParameterList(
                [nn.Parameter(torch.ones(self.rec_A.size())) for _ in self.rec_networks])
        else:
            self.enc_edge_imp = [1] * len(self.enc_networks)
            self.rec_edge_imp = [1] * len(self.rec_networks)

        self.data_bn = nn.BatchNorm1d(in_channels * self.enc_A.size(1))

    def forward(self, x, **kwargs):

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # encode
        for gcn, importance in zip(self.enc_networks, self.enc_edge_imp):
            x, _ = gcn(x, self.enc_A * importance)  # x ~ (N * M, C', T, V)

        # add a branch for reconstruction
        x2 = x
        for gcn, importance in zip(self.rec_networks, self.rec_edge_imp):
            x2, _ = gcn(x2, self.rec_A * importance)                     # (N * M, C', T, V)
        # I replaced C by 3 here because we want the output to be in 3D where the input has a 4th dimension which is the mask
        x2 = x2.view(N, M, self.out_channels, T, V)                                       # (N, M, C, T, V)
        x2 = x2.permute(0, 2, 3, 4, 1)                                   # (N, C, T, V, M)

        return x2
