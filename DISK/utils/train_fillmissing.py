import numpy as np
import torch
import logging

from DISK.models.TCN import TemporalConvNet
from DISK.models.stgcn_AE import STGCN_Model
from DISK.models.sts_gcn import STS_GCN
from DISK.models.GRU_models import BiGRU
from DISK.models.transformer import TransformerModel


def _rmse(data, de_out, mask_holes_tensor, n_missing_per_sample):
    rmse = torch.sum(torch.sqrt(((de_out - data) ** 2) * mask_holes_tensor)[:, 1:, :], dim=(1, 2, 3))
    rmse = torch.masked_select(rmse, n_missing_per_sample > 0) / \
           torch.masked_select(n_missing_per_sample, n_missing_per_sample > 0)

    if torch.any(torch.isnan(rmse)):
        raise ValueError('[ERROR][TRAIN_FILMISSING][_rmse function] !!!! nan in RMSE !!!!')

    return rmse


def _loss(data, de_out, out_distribution, uncertainty_estimate, mask_holes_tensor, criterion_seq, cfg):
    """
    Computes the loss given data and network output.

    :params data:
    """

    # if training mask_loss
    if cfg.training.loss.get('mask'):
        # copy mask with holes
        mask_loss = mask_holes_tensor[:, 1:, :]
    else:
        # all ones, no mask
        mask_loss = torch.ones_like(mask_holes_tensor[:, 1:, :])
    n_missing_per_sample = torch.sum(mask_loss[..., -1], dim=(1, 2))
    if not torch.all(n_missing_per_sample != 0):
        logging.info(f'n_missing_per_sample: {n_missing_per_sample}')
        raise ValueError('[ERROR][TRAIN_FILLMISSING][_loss function] at least one sample has no missing value. '
                         'It is usually caused by a problem in the gap making (see transforms code and proba_missing files).')

    # if mu_sigma
    if cfg.training.get('mu_sigma'):
        # NLL loss
        loss = - out_distribution.log_prob(data)
        if cfg.training.beta_mu_sigma > 0:
            ### from paper "On the Pitfalls of Heteroscedastic Uncertainty Estimation with Probabilistic Neural Networks"
            loss *= torch.mean(uncertainty_estimate.detach(), dim=-1) ** cfg.training.beta_mu_sigma
        # sum over time and keypoints
        loss = torch.sum(loss[:, 1:] * mask_loss[..., 0], dim=(1, 2)) / n_missing_per_sample
        # mean over batch
        loss_original = torch.mean(loss)
    else:
        # normal loss
        loss = criterion_seq(de_out[:, 1:, :], data[:, 1:, :])
        # sum over time, keypoint, 3D dimensions
        loss = torch.sum(loss * mask_loss, dim=(1, 2, 3)) / n_missing_per_sample
        loss_original = torch.mean(loss)

    # clamping the loss because sometimes one sample give an absurd big loss (in DF3D??)
    loss = torch.clamp(loss, -20, 20)  # at the sample level

    # multiply by loss factor
    # mean per batch
    loss = torch.mean(torch.masked_select(loss, n_missing_per_sample > 0))
    if cfg.training.loss.get('factor'):
       loss *= cfg.training.loss.factor

    return loss, loss_original


def apply_model(model, input_tensor_with_holes, mask_holes, n_dim, cfg, device,
                data_full=None, criterion_seq=None, **kwargs):

    uncertainty_estimate = None
    out_distribution = None

    if cfg.network.type in ['GRU', 'BiGRU']:
        input_tensor_with_holes = input_tensor_with_holes.view(
            (input_tensor_with_holes.shape[0], input_tensor_with_holes.shape[1], -1))
        de_out, _ = model(input_tensor_with_holes, **kwargs)

        if cfg.training.mu_sigma:
            de_out, uncertainty_estimate = de_out
            uncertainty_estimate = uncertainty_estimate.view(
                (input_tensor_with_holes.shape[0], input_tensor_with_holes.shape[1], -1, n_dim))
            de_out = de_out.view(
                (input_tensor_with_holes.shape[0], input_tensor_with_holes.shape[1], -1, n_dim))
            out_distribution = model.distribution_output.distribution((de_out, uncertainty_estimate))

        else:
            de_out = de_out.view(
                    (input_tensor_with_holes.shape[0], input_tensor_with_holes.shape[1], -1, n_dim))

        de_out = de_out.view(
                (input_tensor_with_holes.shape[0], input_tensor_with_holes.shape[1], -1, n_dim))

    elif cfg.network.type == 'ST_GCN':
        inputs = torch.unsqueeze(torch.moveaxis(input_tensor_with_holes, -1, 1), -1)
        de_out = model(inputs, **kwargs)
        de_out = torch.squeeze(torch.moveaxis(de_out, 1, 3), -1)

    elif cfg.network.type == 'TCN':
        inputs = torch.moveaxis(input_tensor_with_holes.view(
            (input_tensor_with_holes.shape[0], input_tensor_with_holes.shape[1], -1)), 1, 2)
        de_out = model(inputs, **kwargs)
        de_out = torch.moveaxis(de_out, 1, 2)
        de_out = de_out.view(
                (input_tensor_with_holes.shape[0], input_tensor_with_holes.shape[1], -1, n_dim))

    elif cfg.network.type == 'STS_GCN':
        de_out = model(torch.moveaxis(input_tensor_with_holes, 3, 1), **kwargs)
        de_out = torch.moveaxis(de_out, 3, 2)\
            .reshape(input_tensor_with_holes.shape[0], input_tensor_with_holes.shape[1], -1, n_dim)

    elif cfg.network.type == 'transformer':
        de_out = model(input_tensor_with_holes, mask_holes, **kwargs)
        if cfg.training.mu_sigma:
            de_out, uncertainty_estimate = de_out
            uncertainty_estimate = uncertainty_estimate.view(
                (input_tensor_with_holes.shape[0], input_tensor_with_holes.shape[1], input_tensor_with_holes.shape[2], -1))
            de_out = de_out.view(
                (input_tensor_with_holes.shape[0], input_tensor_with_holes.shape[1], input_tensor_with_holes.shape[2], -1))
            out_distribution = model.distribution_output.distribution((de_out, uncertainty_estimate))
        else:
            de_out = de_out.view(
                    (input_tensor_with_holes.shape[0], input_tensor_with_holes.shape[1], input_tensor_with_holes.shape[2], -1))
    else:
        raise ValueError(f'[TRAIN_FILLMISSING][APPLY_MODEL function] model type {cfg.network.type} not understood.')

    if data_full is None or criterion_seq is None:
        return de_out, uncertainty_estimate
    else:
        mask_holes_tensor = torch.repeat_interleave(mask_holes, n_dim, dim=-1).reshape(data_full.shape)
        n_missing_per_sample = torch.sum(mask_holes, dim=(1, 2))

        rmse = _rmse(data_full, de_out, mask_holes_tensor, n_missing_per_sample)

        loss, loss_original = _loss(data_full, de_out, out_distribution, uncertainty_estimate, mask_holes_tensor, criterion_seq, cfg)

        return de_out, uncertainty_estimate, loss, loss_original, rmse


def feed_forward_list(data_with_holes, mask_holes, n_dim, models, cfgs, device,
                      data_full=None, criterion_seq=None):
    de_out = []
    uncertainty_out = []
    loss_out = []
    rmse_out = []
    for m, cfg in zip(models, cfgs):
        if data_full is None or criterion_seq is None:
            out, uncertainty_estimate = feed_forward(data_with_holes, mask_holes, n_dim, m, cfg, device,
                      data_full=None, criterion_seq=None)
            de_out.append(out)
            if uncertainty_estimate:
                uncertainty_out.append(uncertainty_estimate)
        else:
            out, uncertainty_estimate, loss, _, list_rmse = feed_forward(data_with_holes, mask_holes, n_dim, m, cfg, device,
                                                   data_full=data_full, criterion_seq=criterion_seq)
            de_out.append(out)
            uncertainty_out.append(uncertainty_estimate)
            loss_out.append(loss.item())
            rmse_out.append(torch.mean(list_rmse).item())

    if data_full is None or criterion_seq is None:
        return de_out, uncertainty_out
    else:
        return de_out, uncertainty_out, np.array(loss_out), np.array(rmse_out)


def feed_forward(data_with_holes, mask_holes, n_dim, model, cfg, device,
                 data_full=None, criterion_seq=None, **kwargs):
    if cfg.feed_data.mask:
        input_tensor_with_holes = torch.cat([data_with_holes[..., :n_dim], torch.unsqueeze(mask_holes, dim=-1)], dim=3)
    else:
        input_tensor_with_holes = data_with_holes[..., :3].detach().clone().type(torch.float32)
    input_tensor_with_holes[:, 1:, :] = input_tensor_with_holes[:, :-1, :].clone()
    if data_full is None or mask_holes is None or criterion_seq is None:
        de_out, uncertainty_estimate = apply_model(model, input_tensor_with_holes, mask_holes, n_dim, cfg, device,
                             data_full=None, criterion_seq=None)

        return de_out, uncertainty_estimate
    else:
        de_out, uncertainty_estimate, loss, loss_original, list_rmse = apply_model(model, input_tensor_with_holes, mask_holes, n_dim, cfg, device,
                                                             data_full=data_full, criterion_seq=criterion_seq, **kwargs)

        return de_out, uncertainty_estimate, loss, loss_original, list_rmse


def compute_loss(model, data_loader, n_dim, criterion_seq, cfg, device):
    """
    Compute average loss for all the data in the loader (iterates on the loader)

    :args model: torch model object
    :args data_loader: torch data loader
    :args criterion_seq: torch loss function
    :args device: device on which the model is loaded ('cpu' or 'cuda' e.g.)

    :return: detached tensor, average loss
    """
    total_loss = 0
    total_rmse = 0
    loss_original = 0
    for ind, data_dict in enumerate(data_loader):
        data_with_holes = data_dict['X'].to(device)
        data_full = data_dict['x_supp'].to(device)
        mask_holes = data_dict['mask_holes'].to(device)

        max_len = (data_full.shape[0])
        data_full = data_full[:max_len]
        data_with_holes = data_with_holes[:max_len]
        mask_holes = mask_holes[:max_len]

        _, _, tl, lo, lr = feed_forward(data_with_holes, mask_holes, n_dim, model, cfg, device, data_full=data_full,
                                     criterion_seq=criterion_seq)
        total_loss += tl.item()
        total_rmse += torch.mean(lr).item()
        loss_original += lo.item()

    ave_loss = total_loss / (ind + 1)
    ave_rmse = total_rmse / (ind + 1)
    loss_original = loss_original / (ind + 1)

    return ave_loss, ave_rmse, loss_original


def construct_NN_model(cfg, dataset_constants, skeleton_file, device):
    """
    :args n_dim: 2 or 3, for 2D or 3D
    """
    dim = dataset_constants.DIVIDER + int(cfg.feed_data.mask)
    output_size = dataset_constants.DIVIDER * dataset_constants.N_KEYPOINTS

    if cfg.network.type in ['GRU', 'BiGRU']:
        input_size = dim * dataset_constants.N_KEYPOINTS
        model = BiGRU(input_size, output_size, cfg=cfg, mu_sigma=cfg.training.mu_sigma, device=device)

    elif cfg.network.type == 'ST_GCN':
        if skeleton_file is None:
            raise ValueError('You need to provide a valid skeleton file when using ST_GCN architecture.')
        # ST GCN
        graph_args = {'file': skeleton_file,
                      'strategy': 'uniform',
                      'max_hop': 1,
                      'dilation': 1}
        edge_importance_weighting = True
        model = STGCN_Model(dim, cfg.network.num_layers, graph_args, edge_importance_weighting,
                            dim_start=cfg.network.size_layer, out_channels=dataset_constants.DIVIDER).to(device)
    elif cfg.network.type == 'STS_GCN':
        model = STS_GCN(input_channels=dim, # input_dim
                        output_channels=dataset_constants.DIVIDER,
                         input_time_frame=dataset_constants.SEQ_LENGTH,
                         output_time_frame=dataset_constants.SEQ_LENGTH,
                         st_gcnn_dropout=0,
                         joints_to_consider=dataset_constants.N_KEYPOINTS,
                         n_gcn_layers=cfg.network.en_num_layers,
                         n_txcnn_layers=cfg.network.de_num_layers,
                         txc_kernel_size=(cfg.network.kernel_size, cfg.network.kernel_size),
                         txc_dropout=0.,
                         size_layer=cfg.network.size_layer).to(device)

    elif cfg.network.type == 'TCN':
        model = TemporalConvNet(dim * dataset_constants.N_KEYPOINTS,
                                [cfg.network.size_layer for _ in range(cfg.network.num_layers - 1)] + [output_size],
                                kernel_size=cfg.network.kernel_size, dropout=cfg.network.dropout).to(device)

    elif cfg.network.type == 'transformer':
        model = TransformerModel(dim, dataset_constants.DIVIDER, max_seq_len=dataset_constants.SEQ_LENGTH,
                                 n_keypoints=dataset_constants.N_KEYPOINTS, cfg=cfg, mu_sigma=cfg.training.mu_sigma,
                                 device=device)
    else:
        raise NotImplementedError(f'The only supported models are GRU, BiGRU, ST_GCN, STS_GCN or TCN. '
                                  f'Given: {cfg.network.type}')

    return model
