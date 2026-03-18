import os, sys
from glob import glob
from pathlib import Path
import json

import tqdm
from scipy.stats import pearsonr
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import yaml

from DISK.utils.dataset_utils import load_datasets
from DISK.utils.utils import read_constant_file, plot_save, compute_interp, find_holes, load_checkpoint
from DISK.utils.transforms import init_transforms, reconstruct_before_normalization
from DISK.utils.train_fillmissing import construct_NN_model, feed_forward_list
from DISK.utils.coordinates_utils import plot_sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def test(project_path: str,
         output_dir: str,
         dataset_path: str,
         dataset_name:str,
         skeleton_graph,
         model_checkpoints: list,
         batch_size,
         n_cpus: int,
         loss_type,
         loss_mask,
         loss_factor,
         proba_file,
         proba_length_file,
         indep_keypoints,
         add_missing_pad,
         viewinvariant,
         normalize,
         normalizecube,
         swap,
         add_missing,
         test_original_coordinates,
         test_threshold_pck,
         n_repeat,
         total_n_plots,
         plot2d_only_holes,
         plot3d_size,
         plot3d_azim,
         logger,
         suffix='',
         stride=None,
         verbose=0) -> (list, list):

    logger.debug(f'{project_path}')

    dataset_constants = read_constant_file(os.path.join(dataset_path, f'constants.py'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info("Device: {}".format(device))

    paths_to_models = []
    model_configs = []
    model_names = []
    for cf in model_checkpoints:
        config_file = os.path.join(cf, 'config', 'config_train.yaml')
        if os.path.exists(config_file):
            with open(config_file, 'r') as file:
                cfg_model = yaml.safe_load(file)
            try:
                model_path = glob(os.path.join(cf, 'model_epoch*'))[0] # model_epoch to not take the model from the lastepoch
            except IndexError:
                raise Exception(f'No model checkpoint found at path {cf}')
            logger.info(f'Found model at path {cf}')
            paths_to_models.append(model_path)
            model_configs.append(cfg_model)
            model_names.append(os.path.basename(cf))
        else:
            for path in Path(os.path.join(project_path, cf)).rglob('model_epoch*'):
                logger.info(f'Found model at path {str(path)}')
                paths_to_models.append(str(path))
                config_file = os.path.join(os.path.dirname(path), 'config', 'config_train.yaml')
                with open(config_file, 'r') as file:
                    cfg_model = yaml.safe_load(file)
                model_configs.append(cfg_model)
                model_names.append(os.path.basename(path))

    n_models = len(paths_to_models)
    logger.info(f'Number of compared models: {n_models}')
    if n_models == 0:
        sys.exit('No files found.')

    logger.debug(f'Full path to 1st model: {paths_to_models[0]}')

    assert len(model_configs) == n_models

    logger.info('Loading prediction model...')
    # load model
    models = []
    full_name = ''
    for imodel, model_cfg in enumerate(model_configs):
        models.append(construct_NN_model(model_cfg['network'], dataset_constants.KEYPOINTS, dataset_constants.DIVIDER,
                                         dataset_constants.SEQ_LENGTH,
                                         skeleton_graph,
                                         device))

        logger.info(f'Network {full_name} constructed')

    for path, model in zip(paths_to_models, models):
        load_checkpoint(model, None, path, device, logger)
        model.eval()

    """ DATA """
    transforms = init_transforms(dataset_constants.KEYPOINTS,
                                 dataset_constants.DIVIDER,
                                 dataset_constants.SEQ_LENGTH,
                                 output_dir,
                                 logger,
                                 add_missing_pad,
                                 viewinvariant,
                                 normalize,
                                 normalizecube,
                                 swap,
                                 proba_file,
                                 proba_length_file,
                                 indep_keypoints,
                                 add_missing,
                                 verbose)

    logger.info('Loading datasets...')
    if stride is None:
        stride = dataset_constants.STRIDE
    train_dataset, val_dataset, test_dataset = load_datasets(
            dataset_path=dataset_path,
            transform=transforms,
            outputdir=output_dir,
            skeleton_graph=skeleton_graph,
            dataset_type='full_length',
            suffix='_w-0-nans',
            root_path=project_path,
            label_type=None,  # don't care, not using
            verbose=verbose,
            keypoints=dataset_constants.KEYPOINTS,
            divider=dataset_constants.DIVIDER,
            stride=stride,
            length_sample=dataset_constants.SEQ_LENGTH,
            freq=dataset_constants.FREQ,
            logger=logger
        )

    if test_original_coordinates:
        pck_final_threshold = train_dataset.kwargs['max_dist_bw_keypoints'] * test_threshold_pck
    else:
        # when normalized coordinates, approximation of the PCK score from the furthest away points could be (1, 1, 1) and (-1, -1, -1)
        # divider should be 2 in 2D and 3 in 3D
        pck_final_threshold = 2 * np.sqrt(dataset_constants.DIVIDER) * test_threshold_pck

    pck_name = f'PCK@{test_threshold_pck}'

    if n_cpus == 0:
        persistent_workers = False
    else:
        persistent_workers = True
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=n_cpus, persistent_workers=persistent_workers)

    if loss_type == 'l1':
        criterion_seq = nn.L1Loss(reduction='none')
    elif loss_type == 'l2':
        criterion_seq = nn.MSELoss(reduction='none')
    else:
        raise NotImplementedError(f'[ERROR][MAIN_FILLMISSING] Loss type should be "l1" or '
                                  f'"l2". '
                                  f'Given: {loss_type}')

    visualize_val_outputdir = os.path.join(output_dir, 'visualize_prediction_val')
    if not os.path.isdir(visualize_val_outputdir):
        os.mkdir(visualize_val_outputdir)

    mean_RMSE = []
    for i_repeat in range(n_repeat):
        suffix = suffix + f'_repeat-{i_repeat}'
        """RMSE computation"""
        total_rmse = {'id_sample': [], 'id_hole': [], 'keypoint': [], 'method': [], 'method_param': [], 'RMSE': [],
                      'MPJPE': [], pck_name: [], 'mean_uncertainty': [], 'length_hole': [], 'swap_kp_id': [],
                      'swap_length': [], 'average_dist_bw_swap_kp': []}

        id_sample = 0
        n_plots = 0
        """Visualization 3D, one timepoint each"""

        with (torch.no_grad()):
            logger.info(f'Starting evaluation...')

            for ind, data_dict in tqdm.tqdm(enumerate(test_loader), desc='Testing trained model -- iterating on data',
                                            total=len(
                    test_loader)):
                """Compute the prediction from networks"""

                data_with_holes = data_dict['X'].to(device)  # shape (timepoints, n_keypoints, 2 or 3 or 4)
                data_full = data_dict['x_supp'].to(device)
                mask_holes = data_dict['mask_holes'].to(device)
                # if swap_bool:
                #     data_swapped_np = data_dict['x_swap'].detach().cpu().numpy()
                    #if 'x_swap' in data_dict \
                    #                  else np.zeros((_cfg.evaluate.batch_size, data_dict['X'].shape[1], dataset_constants.N_KEYPOINTS, dataset_constants.DIVIDER)) * np.nan
                assert not torch.any(torch.isnan(data_with_holes))
                assert not torch.any(torch.isnan(data_full))

                de_outs, uncertainty_estimates, _, _ = feed_forward_list(data_with_holes, mask_holes,
                                                                         dataset_constants.DIVIDER, models,
                                                                         loss_mask, loss_factor,
                                                                         [m['network'] for m in model_configs],
                                                                          data_full=data_full,
                                                                         criterion_seq=criterion_seq,
                                                                         logger=logger)

                full_data_np = data_full.detach().cpu().clone().numpy()
                data_with_holes_np = data_with_holes.detach().cpu().numpy()
                mask_holes_np = mask_holes.detach().cpu().numpy().astype(bool)
                data_with_holes_np[mask_holes_np] = np.nan
                """Linear interpolation"""

                ### put everything we need in numpy
                indices_sample = data_dict['index'].detach().cpu().numpy()

                reshaped_mask_holes = np.repeat(mask_holes_np, dataset_constants.DIVIDER, axis=-1).reshape(full_data_np.shape)
                # gives the total number of missing values in a sample (can be from multiple keypoints):
                n_missing = np.sum(mask_holes_np, axis=(1, 2))  ## (batch,)

                x_outputs_np = [out.detach().cpu().numpy() for out in de_outs]
                # List(number of models) of tensors of size (batch, time, keypoints, 3D) if mu_sigma GRU or transformer model
                uncertainty_estimates_np = [unc if unc is None else unc.detach().cpu().numpy() for unc in
                                            uncertainty_estimates]

                swap_samples, swap_times, swap_keypoints = np.where(~np.isclose(data_with_holes_np[..., 0], full_data_np[..., 0],
                                                                                atol=1.e-6, equal_nan=True) * ~mask_holes_np)

                if test_original_coordinates:
                    full_data_np = reconstruct_before_normalization(full_data_np, data_dict, transforms)
                    data_with_holes_np = reconstruct_before_normalization(data_with_holes_np, data_dict, transforms)

                    max_uncertainty_margin_orig = [unc if unc is None else
                                                   reconstruct_before_normalization(out + unc, data_dict, transforms)
                                                   for out, unc in zip(x_outputs_np, uncertainty_estimates_np)]

                    x_outputs_np = [reconstruct_before_normalization(out, data_dict, transforms)
                               for out in x_outputs_np]

                    uncertainty_estimates_np = [y if y is None else y - out for out, y in zip(x_outputs_np,
                                                                                           max_uncertainty_margin_orig)]


                uncertainty = [unc if unc is None else np.sum(np.sqrt((unc ** 2) * reshaped_mask_holes), axis=3)
                               for unc in uncertainty_estimates_np]  # sum on the XYZ dimension, output shape (batch, time, keypoint)

                # de_out : model output, pytorch tensor of shape (batch, time, keypoints, n_dim)
                euclidean_distance = [np.sqrt(np.sum(((out - full_data_np) ** 2) * reshaped_mask_holes, axis=3))
                                      for out in x_outputs_np]  # sum on the XYZ dimension, output shape (batch, time, keypoint)
                pck = [euc <= pck_final_threshold for euc in euclidean_distance]
                rmse = [np.sum(((out - full_data_np) ** 2) * reshaped_mask_holes, axis=3)
                                      for out in x_outputs_np]  # sum on the XYZ dimension, output shape (batch, time, keypoint)

                if np.min(add_missing_pad) > 0:
                    linear_interp_data = compute_interp(data_with_holes_np, mask_holes_np, dataset_constants.KEYPOINTS,
                                                        dataset_constants.DIVIDER)
                    rmse_linear_interp = np.sum(((linear_interp_data - full_data_np) ** 2) * reshaped_mask_holes,
                                                axis=3)  # sum on the XYZ dimension, output shape (batch, time, keypoint)
                    euclidean_distance_linear_interp = np.sqrt(np.sum(((linear_interp_data - full_data_np) ** 2) * reshaped_mask_holes,
                                                axis=3))  # sum on the XYZ dimension, output shape (batch, time, keypoint)
                    pck_linear_interpolation = euclidean_distance_linear_interp <= pck_final_threshold

                for i_sample_in_batch in range(data_with_holes_np.shape[0]):
                    # if swap_bool:
                    if i_sample_in_batch in swap_samples:
                        swap_length = np.max(swap_times[swap_samples == i_sample_in_batch]) - np.min(swap_times[swap_samples == i_sample_in_batch]) + 1

                        # Euclidean distance between keypoints that are swapped during the swap

                        swap_dist = np.mean(np.sqrt(np.sum((data_with_holes_np[i_sample_in_batch, swap_times[swap_samples == i_sample_in_batch]][:,
                                                            swap_keypoints[swap_samples == i_sample_in_batch]] - full_data_np[i_sample_in_batch, swap_times[swap_samples == i_sample_in_batch]][:,
                                                                              swap_keypoints[swap_samples == i_sample_in_batch]]) ** 2, axis=-1)))
                    else:
                        swap_length = 0
                        swap_dist = 0

                    ## gives the length of a hole, one keypoint at a time, a sample can have multiple holes one after the other:
                    id_hole = 0
                    out = find_holes(mask_holes_np[i_sample_in_batch], dataset_constants.KEYPOINTS, indep=False)
                    for o in out:  # (start, length, keypoint_name)
                        slice_ = tuple([i_sample_in_batch, slice(o[0], o[0] + o[1], 1), [dataset_constants.KEYPOINTS.index(kp) for kp in o[2].split(' ')]])
                        for i_model in range(n_models):
                            mean_euclidean = np.mean(euclidean_distance[i_model][slice_])
                            mean_rmse = np.sqrt(np.mean(rmse[i_model][slice_]))
                            mean_pck = np.sum(pck[i_model][slice_] * mask_holes_np[slice_])/ np.sum(mask_holes_np[slice_])
                            total_rmse['id_sample'].append(id_sample)
                            total_rmse['id_hole'].append(id_hole)
                            total_rmse['keypoint'].append(o[2])
                            total_rmse['method'].append(model_configs[i_model]['network']['type'])
                            total_rmse['method_param'].append(model_names[i_model])
                            total_rmse['RMSE'].append(mean_rmse)
                            total_rmse['MPJPE'].append(mean_euclidean)
                            total_rmse[pck_name].append(mean_pck)
                            total_rmse['mean_uncertainty'].append(np.nan)
                            total_rmse['length_hole'].append(o[1])
                            total_rmse['swap_kp_id'].append(tuple(np.unique(swap_keypoints[swap_samples == i_sample_in_batch])))
                            total_rmse['swap_length'].append(swap_length)
                            total_rmse['average_dist_bw_swap_kp'].append(swap_dist)

                        if np.min(add_missing_pad) > 0:
                            mean_rmse_linear = np.sqrt(np.mean(rmse_linear_interp[slice_]))
                            mean_euclidean_linear = np.mean(euclidean_distance_linear_interp[slice_])
                            mean_pck_linear = np.sum(pck_linear_interpolation[slice_] * mask_holes_np[slice_])\
                                              / np.sum(mask_holes_np[slice_])
                            total_rmse['id_sample'].append(id_sample)
                            total_rmse['id_hole'].append(id_hole)
                            total_rmse['keypoint'].append(o[2])
                            total_rmse['method'].append('linear_interp')
                            total_rmse['method_param'].append('linear_interp')
                            total_rmse['RMSE'].append(mean_rmse_linear)
                            total_rmse['MPJPE'].append(mean_euclidean_linear)
                            total_rmse[pck_name].append(mean_pck_linear)
                            total_rmse['mean_uncertainty'].append(np.nan)
                            total_rmse['length_hole'].append(o[1])
                            total_rmse['swap_kp_id'].append(tuple(np.unique(swap_keypoints[swap_samples == i_sample_in_batch])))
                            total_rmse['swap_length'].append(swap_length)
                            total_rmse['average_dist_bw_swap_kp'].append(swap_dist)
                        id_hole += 1

                    ## the sample as a whole, not hole by hole
                    if np.min(add_missing_pad) > 0:
                        total_rmse['id_sample'].append(id_sample)
                        total_rmse['id_hole'].append(-1)
                        total_rmse['keypoint'].append('all')
                        total_rmse['method'].append('linear_interp')
                        total_rmse['method_param'].append('linear_interp')
                        total_rmse['RMSE'].append(np.sqrt(np.sum(rmse_linear_interp[i_sample_in_batch]) / n_missing[i_sample_in_batch]))
                        total_rmse['MPJPE'].append(np.sum(euclidean_distance_linear_interp[i_sample_in_batch]) / n_missing[i_sample_in_batch])
                        total_rmse[pck_name].append(
                            np.sum(pck_linear_interpolation[i_sample_in_batch] * mask_holes_np[i_sample_in_batch]) /
                            n_missing[i_sample_in_batch])
                        total_rmse['mean_uncertainty'].append(np.nan)
                        total_rmse['length_hole'].append(n_missing[i_sample_in_batch])
                        total_rmse['swap_kp_id'].append(tuple(np.unique(swap_keypoints[swap_samples == i_sample_in_batch])))
                        total_rmse['swap_length'].append(swap_length)
                        total_rmse['average_dist_bw_swap_kp'].append(swap_dist)
                    for i_model in range(n_models):
                        if model_configs[i_model]['network']['mu_sigma']:
                            mean_uncertainty_model = np.sum(uncertainty[i_model][i_sample_in_batch]) / n_missing[i_sample_in_batch]
                        else:
                            mean_uncertainty_model = np.nan
                        total_rmse['id_sample'].append(id_sample)
                        total_rmse['id_hole'].append(-1)
                        total_rmse['keypoint'].append('all')
                        total_rmse['method'].append(model_configs[i_model]['network']['type'])
                        total_rmse['method_param'].append(model_names[i_model])
                        total_rmse['RMSE'].append(np.sqrt(np.sum(rmse[i_model][i_sample_in_batch]) / n_missing[i_sample_in_batch]))
                        total_rmse['MPJPE'].append(np.sum(euclidean_distance[i_model][i_sample_in_batch]) / n_missing[i_sample_in_batch])
                        total_rmse[pck_name].append(
                            np.sum(pck[i_model][i_sample_in_batch] * mask_holes_np[i_sample_in_batch]) / n_missing[
                                i_sample_in_batch])
                        total_rmse['mean_uncertainty'].append(mean_uncertainty_model)
                        total_rmse['length_hole'].append(n_missing[i_sample_in_batch])
                        total_rmse['swap_kp_id'].append(tuple(np.unique(swap_keypoints[swap_samples == i_sample_in_batch])))
                        total_rmse['swap_length'].append(swap_length)
                        total_rmse['average_dist_bw_swap_kp'].append(swap_dist)
                    id_sample += 1

                """VISUALIZATION, only first batch"""
                if n_plots < total_n_plots:
                    logger.info(f'Plotting sample: {n_plots} / {total_n_plots}')
                    potential_indices = np.where(n_missing > 0)[0]
                    np.random.seed(0)
                    indices = np.random.choice(potential_indices,  #full_data_np.shape[0],
                                              min(len(potential_indices), total_n_plots),
                                              replace=False)
                    uncertainty_str = []
                    for i_model in range(n_models):
                        if uncertainty[i_model] is None:
                            uncertainty_str.append(["None"] * len(rmse[i_model]))
                        else:
                            uncertainty_str.append([f'{np.sum(uncertainty[i_model][i]) / n_missing[i] :.2f}' for
                                                    i in indices])

                    for i in indices:
                        if skeleton_graph is not None:
                            for i_model, xo in enumerate(x_outputs_np):
                                save_path = os.path.join(
                                    visualize_val_outputdir,
                                    f'traj3D_{indices_sample[i][0]}{model_names[i_model]}{suffix}'
                                )
                                plot_sequence(full_data_np[i, :], xo[i, :], mask_holes_np[i, :], skeleton_graph, nplots=15,
                                              save_path=save_path,
                                              size=plot3d_size, azim=plot3d_azim,
                                              normalized_coordinates=(not test_original_coordinates))

                        title = f'{"RMSE"} | {"MPJPE"} | estimated error  \n'
                        title += '\n'.join(
                            [f'{model_names[i_model]}: '
                             f'{np.sqrt(np.sum(rmse[i_model][i]) / n_missing[i]):.2f} |'
                             f' {np.sum(euclidean_distance[i_model][i]) / n_missing[i]:.2f} | '
                             f'{uncertainty_str[i_model][i]} '
                             for i_model in range(n_models)])

                        if np.min(add_missing_pad) > 0:
                            title += (f'\n{"linear"}: '
                                      f'{np.sqrt(np.sum(rmse_linear_interp[i]) / n_missing[i]):.2f} | '
                                      f'{np.sum(euclidean_distance_linear_interp[i]) / n_missing[i]:.2f}')
                        def make_xyz_plot():
                            fig, axes = plt.subplots(dataset_constants.N_KEYPOINTS, dataset_constants.DIVIDER,
                                                     figsize=(max(dataset_constants.SEQ_LENGTH // 10,
                                                                  dataset_constants.DIVIDER * 7),
                                                              dataset_constants.NUM_FEATURES),
                                                     sharex='all', sharey='col')
                            fig.suptitle(title, size=30)
                            axes = axes.flatten()
                            t_vect = np.arange(0, dataset_constants.SEQ_LENGTH) / dataset_constants.FREQ

                            for j in range(dataset_constants.N_KEYPOINTS):
                                if plot2d_only_holes:
                                    t_mask = (mask_holes_np[i, :, j] == 1)
                                    t_mask_holes = (mask_holes_np[i, :, j] == 1)
                                else:
                                    t_mask = np.ones_like(mask_holes_np[i, :, j]).astype(bool)
                                    t_mask_holes = (mask_holes_np[i, :, j] == 1)
                                for i_dim in range(dataset_constants.DIVIDER):
                                    if swap_length > 0 and j in swap_keypoints[swap_samples == i]:
                                        axes[dataset_constants.DIVIDER * j + i_dim].plot(t_vect, data_with_holes_np[
                                            i, :, j, i_dim], '+--', color='grey', label='swap')
                                    axes[dataset_constants.DIVIDER * j + i_dim].plot(t_vect, full_data_np[i, :, j, i_dim], 'o-')
                                    if np.sum(t_mask) > 0:
                                        for i_model, xo in enumerate(x_outputs_np):
                                            plot_ = axes[dataset_constants.DIVIDER * j + i_dim].plot(t_vect[t_mask], xo[i, :, j, i_dim][t_mask], 'o',
                                                             label=model_names[i_model], )
                                            if model_configs[i_model]['network']['mu_sigma']:
                                                # 3 * std otherwise 1/ we do not see anything,
                                                # 2/ because the underlying distribution is supposed to be Gaussian
                                                axes[dataset_constants.DIVIDER * j + i_dim]\
                                                    .fill_between(t_vect[t_mask], xo[i, :, j, i_dim][t_mask]
                                                                          - 3 * uncertainty_estimates_np[i_model][i, :, j, i_dim][t_mask],
                                                                          xo[i, :, j, i_dim][t_mask]
                                                                          + 3 * uncertainty_estimates_np[i_model][i, :, j, i_dim][t_mask],
                                                                          color=plot_[0].get_color(), alpha=0.2)
                                            assert not np.any(np.isnan(xo))

                                    out = find_holes(np.array(t_mask_holes).reshape(dataset_constants.SEQ_LENGTH, 1).astype(int), ['0'], indep=True)
                                    if np.min(add_missing_pad) > 0:
                                        for o in out:
                                            axes[dataset_constants.DIVIDER * j + i_dim].plot(t_vect[o[0] - 1: o[0] + o[1] + 1],
                                                                                     linear_interp_data[i, :, j, i_dim][o[0] - 1: o[0] + o[1] + 1], 'r-',
                                                     label='linear interp 1D')

                                    if not test_original_coordinates:
                                        axes[dataset_constants.DIVIDER * j + i_dim].set_ylim(-1.2, 1.2)

                                if np.any(t_mask_holes) or swap_length > 0 and j in swap_keypoints[swap_samples == i]:
                                    axes[dataset_constants.DIVIDER * j].legend()

                                axes[dataset_constants.DIVIDER * j].set_ylabel(dataset_constants.KEYPOINTS[j])
                            axes[0].set_title('X')
                            axes[1].set_title('Y')
                            if dataset_constants.DIVIDER >= 3:
                                axes[2].set_title('Z')

                            return

                        plot_save(make_xyz_plot,
                                  title=f'RMSE_reconstruction_xyz_{indices_sample[i][0]}{suffix}',
                                  only_png=False,
                                  outputdir=visualize_val_outputdir)

                        n_plots += 1

                    logger.info(f'Done with sample plot {n_plots}')

                for _ in range(3):
                    torch.cuda.empty_cache()
                    gc.collect()

        logger.info(f'Finished with iterating the dataset')
        total_rmse = pd.DataFrame.from_dict(total_rmse)
        total_rmse = total_rmse.reset_index().convert_dtypes()
        logger.info(f'n lines in result df: {total_rmse.shape[0]}')
        logger.info(f"RMSE per sample averaged: \n"
                     f"{total_rmse[(total_rmse['keypoint'] == 'all')].groupby(['method_param'])[[pck_name, 'RMSE', 'MPJPE']].agg('mean')}")
        tmp = total_rmse[(total_rmse['keypoint'] == 'all')].groupby(['method', 'method_param'])[[pck_name, 'RMSE', 'MPJPE']].agg('mean').reset_index()
        tmp['repeat'] = i_repeat
        tmp['dataset'] = dataset_name
        mean_RMSE.append(tmp)

        plt.close('all')

        def barplot_RMSE_keypoint():
            mask = (total_rmse['keypoint'] != 'all')
            n_keypoints = len(total_rmse['keypoint'].unique())

            if n_keypoints > 40:
                total_rmse['n_keypoints'] = total_rmse.loc[:, 'keypoint'].apply(lambda s: len(s.split(' ')))
                total_rmse['simplified_keypoint'] = total_rmse.apply(lambda a:
                                                        str(a['keypoint']) if a['n_keypoints'] == 1
            else f'{a["n_keypoints"]:02d}_keypoints', axis=1)
                n_keypoints = len(total_rmse['simplified_keypoint'].unique())
                keypoints = total_rmse['simplified_keypoint'].unique()
                keypoints = np.delete(keypoints, keypoints == 'all')
                order_keypoints = np.sort(keypoints)
                sns.catplot(data=total_rmse.loc[mask, :], kind='bar', y='simplified_keypoint',
                            hue='method_param', x=metric, height=max(5, n_keypoints // 8),
                            order=order_keypoints,)
                plt.tight_layout()
            else:
                keypoints = total_rmse['keypoint'].unique()
                keypoints = np.delete(keypoints, keypoints == 'all')
                order_keypoints = np.sort(keypoints)
                sns.catplot(data=total_rmse.loc[mask, :], kind='bar', y='keypoint',
                            hue='method_param', x=metric, height=max(5, n_keypoints // 8),
                            raw_order=order_keypoints,)
                plt.tight_layout()


        for metric in [pck_name, 'RMSE', 'MPJPE']:
            plot_save(barplot_RMSE_keypoint,
                      title=f'barplot_comparison_{metric}{suffix}', only_png=False,
                      outputdir=output_dir)
            plt.close('all')

        def lineplot_length():
            mask = (total_rmse['keypoint'] != 'all')
            total_rmse['length_hole'] = total_rmse.loc[:, 'length_hole'].astype('float')
            sns.lineplot(data=total_rmse.loc[mask, :], x='length_hole', y=metric,
                         hue='method_param')
            plt.tight_layout()

        for metric in [pck_name, 'RMSE', 'MPJPE']:
            plot_save(lineplot_length,
                  title=f'comparison_length_hole_kp_vs_{metric}{suffix}', only_png=False,
                  outputdir=output_dir)
        plt.close('all')


        def lineplot_all_length():
            mask = (total_rmse['keypoint'] != 'all')
            total_rmse.loc[:, 'length_hole'] = total_rmse.loc[:, 'length_hole'].astype('float')
            sns.lineplot(data=total_rmse.loc[mask, :], x='length_hole', y=metric,
                         hue='method_param')
            plt.tight_layout()

        for metric in [pck_name, 'RMSE', 'MPJPE']:
            plot_save(lineplot_all_length,
                      title=f'comparison_length_hole_all_vs_{metric}{suffix}', only_png=False,
                      outputdir=output_dir)
        plt.close('all')

        total_rmse.to_csv(os.path.join(output_dir, f'total_metrics{suffix}.csv'), index=False)

        thresholding_df = pd.DataFrame(columns=['th', 'RMSE', 'RMSE_std', 'MPJPE', 'MPJPE_std',
                                                pck_name, f'{pck_name}_std', 'count', 'method'])
        pcoeff_per_model = []
        for i_model in range(n_models):
            if uncertainty_estimates[i_model] is not None:
                # pivot_df only for one method
                mask = (total_rmse['keypoint'] == 'all') * (total_rmse['method_param'] == model_names[i_model])
                pcoeff, ppval = pearsonr(total_rmse.loc[mask, 'RMSE'].values, total_rmse.loc[mask, 'mean_uncertainty'])
                pcoeff_per_model.append((pcoeff, ppval))
                logger.info(f'Model {model_names[i_model]}: PEARSONR COEFF w RMSE {pcoeff}, PVAL {ppval}')

                def corr_plot():
                    total_rmse['mean_uncertainty'] = total_rmse['mean_uncertainty'].astype(float)
                    total_rmse['RMSE'] = total_rmse['RMSE'].astype(float)
                    sns.histplot(data=total_rmse.loc[mask, :], x=metric, y='mean_uncertainty')
                    sns.kdeplot(data=total_rmse.loc[mask, :], x=metric, y='mean_uncertainty')
                    plt.plot([0, total_rmse[metric].max()], [0, total_rmse[metric].max()], 'r--')
                    plt.title(f'Pearson coeff: {pcoeff:.3f}')

                metric = 'RMSE'
                plot_save(corr_plot,
                          title=f'corrplot-model-{metric}-{model_names[i_model]}{suffix}', only_png=False,
                          outputdir=output_dir)
                plt.close('all')

                th_vals = np.unique(total_rmse.loc[mask, 'mean_uncertainty'])
                th_vals = th_vals[::max(1, len(th_vals) // 10)]
                for th in th_vals:
                    filtered_id_samples = total_rmse.loc[(total_rmse[metric] <= th) *
                        (total_rmse['keypoint'] == 'all') * (total_rmse['method_param'] == model_names[i_model]),
                        'id_sample'].values
                    if len(filtered_id_samples) == 0:
                        continue
                    vals_RMSE = total_rmse[(total_rmse['keypoint'] == 'all') *
                                      (total_rmse['method_param'] == model_names[i_model]) *
                                      (total_rmse['id_sample'].isin(filtered_id_samples))]['RMSE'].agg(['mean', 'std', 'count'])
                    vals_MPJPE = total_rmse[(total_rmse['keypoint'] == 'all') *
                                           (total_rmse['method_param'] == model_names[i_model]) *
                                           (total_rmse['id_sample'].isin(filtered_id_samples))]['MPJPE'].agg(
                        ['mean', 'std', 'count'])
                    vals_pck = total_rmse[(total_rmse['keypoint'] == 'all') *
                                           (total_rmse['method_param'] == model_names[i_model]) *
                                           (total_rmse['id_sample'].isin(filtered_id_samples))][pck_name].agg(
                        ['mean', 'std', 'count'])
                    ## add values in thresholding_df which holds the results for all uncertainty methods
                    thresholding_df.loc[thresholding_df.shape[0], :] = [th, vals_RMSE['mean'], vals_RMSE['std'],
                                                                        vals_MPJPE['mean'], vals_MPJPE['std'],
                                                                        vals_pck['mean'], vals_pck['std'],
                                                                        vals_RMSE['count'], model_names[i_model]]

            else:
                pcoeff_per_model.append((None, None))

        if np.any([unc is not None for unc in uncertainty_estimates]):
            def plot_thresholding():
                err_sup_PCK = [-1 for _ in range(n_models)]
                fig, ax1 = plt.subplots(1, 1)
                for i_model in range(n_models):
                    if not model_configs[i_model]['network']['mu_sigma']:
                        continue
                    m = model_names[i_model]
                    count = thresholding_df.loc[thresholding_df['method'] == m, 'count'].astype(int)
                    metric = thresholding_df.loc[thresholding_df['method'] == m, metric_name].astype(float)
                    metric_std = thresholding_df.loc[thresholding_df['method'] == m, f'{metric_name}_std'].astype(float)
                    if 'PCK' in metric_name:
                        metric_sup_index = np.where(metric > 0.8)
                        if len(metric_sup_index[0]) > 0:
                            err_sup_PCK[i_model] = th_vals[metric_sup_index[0][-1]]

                    pl = ax1.plot(count, metric, '+-', label=m)
                    ax1.fill_between(x=count, y1=metric - metric_std, y2=metric + metric_std, label=m,
                                     color=pl[0].get_color(),
                                     alpha=0.5)
                ax1.legend()
                ax1.set_ylabel(f'Mean {metric_name}')
                ax1.set_xlabel('Remaining samples')

                return err_sup_PCK

            for metric_name in [pck_name, 'RMSE', 'MPJPE']:
                if metric_name == pck_name:
                    err_sup_PCK = plot_save(plot_thresholding,
                                          title=f'thresholding_curve_{metric_name}{suffix}', only_png=False,
                                          outputdir=output_dir)
                else:
                    plot_save(plot_thresholding,
                              title=f'thresholding_curve_{metric_name}{suffix}', only_png=False,
                              outputdir=output_dir)
                plt.close('all')

        else:
            err_sup_PCK = [None] * n_models
    pd.concat(mean_RMSE).to_csv(os.path.join(output_dir, f'mean_metrics{suffix}.csv'), index=False)

    return pcoeff_per_model, err_sup_PCK