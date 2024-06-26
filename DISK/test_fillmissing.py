import os, sys
from glob import glob
from pathlib import Path
import logging
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
import hydra
from omegaconf import DictConfig, OmegaConf

from DISK.utils.dataset_utils import load_datasets
from DISK.utils.utils import read_constant_file, plot_save, compute_interp, find_holes, load_checkpoint
from DISK.utils.transforms import init_transforms, reconstruct_before_normalization
from DISK.utils.train_fillmissing import construct_NN_model, feed_forward_list
from DISK.utils.coordinates_utils import plot_sequence
from DISK.models.graph import Graph

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@hydra.main(version_base=None, config_path="conf", config_name="conf_test")
def evaluate(_cfg: DictConfig) -> None:
    outputdir = os.getcwd()
    basedir = hydra.utils.get_original_cwd()
    logging.info(f'[BASEDIR] {basedir}')
    logging.info(f'[OUTPUT DIR] {outputdir}')
    """ LOGGING AND PATHS """

    logging.info(f'{_cfg}')

    dataset_constants = read_constant_file(os.path.join(basedir, 'datasets', _cfg.dataset.name, f'constants.py'))
    if _cfg.dataset.skeleton_file is not None:
        skeleton_file_path = os.path.join(basedir, 'datasets', _cfg.dataset.skeleton_file)
        skeleton_graph = Graph(file=skeleton_file_path)
        if not os.path.exists(skeleton_file_path):
            raise ValueError(f'no skeleton file found in', skeleton_file_path)
    else:
        skeleton_graph = None
        skeleton_file_path = None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info("Device: {}".format(device))

    paths_to_models = []
    model_configs = []
    for cf in _cfg.evaluate.checkpoints:
        config_file = os.path.join(cf, '.hydra', 'config.yaml')
        if os.path.exists(config_file):
            cfg_model = OmegaConf.load(config_file)
            logging.info(f'Found model at path {cf}')
            model_path = glob(os.path.join(cf, 'model_epoch*'))[0] # model_epoch to not take the model from the lastepoch
            paths_to_models.append(model_path)
            if not cfg_model.training.get('mu_sigma'):
                cfg_model.training['mu_sigma'] = False
            model_configs.append(cfg_model)
        else:
            for path in Path(os.path.join(basedir, cf)).rglob('model_epoch*'):
                logging.info(f'Found model at path {str(path)}')
                paths_to_models.append(str(path))
                config_file = os.path.join(os.path.dirname(path), '.hydra', 'config.yaml')
                cfg_model = OmegaConf.load(config_file)
                if not cfg_model.training.get('mu_sigma'):
                    cfg_model.training['mu_sigma'] = False
                model_configs.append(cfg_model)

    n_models = len(paths_to_models)
    logging.info(f'Number of compared models: {n_models}')
    if n_models == 0:
        sys.exit('No files found.')

    logging.debug(f'Full path to 1st model: {paths_to_models[0]}')

    assert len(model_configs) == n_models

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info("Device: {}".format(device))

    logging.info('Loading prediction model...')
    # load model
    models = []
    model_name = []
    full_name = ''
    for imodel, model_cfg in enumerate(model_configs):
        models.append(construct_NN_model(model_cfg, dataset_constants, skeleton_file_path, device))
        for ini, name_item in enumerate(_cfg.evaluate.name_items):
            val = model_cfg[name_item[0]]
            for item in name_item[1:]:
                val = val[item]
            if ini == 0:
                full_name = f'{name_item[-1]}-{val}'
            else:
                full_name += f'_{name_item[-1]}-{val}'
        if not _cfg.evaluate.merge:
            full_name += f'_{imodel}'
        model_name.append(full_name)
        logging.info(f'Network {full_name} constructed')

    for path, model in zip(paths_to_models, models):
        load_checkpoint(model, None, path, device)
        model.eval()

    """ DATA """
    transforms, _ = init_transforms(_cfg, dataset_constants.KEYPOINTS, dataset_constants.DIVIDER,
                                 dataset_constants.SEQ_LENGTH, basedir, outputdir)

    logger.info('Loading datasets...')
    train_dataset, val_dataset, test_dataset = load_datasets(dataset_name=_cfg.dataset.name,
                                                             dataset_constants=dataset_constants,
                                                             transform=transforms,
                                                             dataset_type='full_length',
                                                             suffix='_w-0-nans',
                                                             root_path=basedir,
                                                             outputdir=outputdir,
                                                             label_type=None,  # don't care, not using
                                                             verbose=_cfg.feed_data.verbose,
                                                             keypoints_bool=True,
                                                             skeleton_file=skeleton_file_path,
                                                             stride=_cfg.dataset.stride,
                                                             length_sample=dataset_constants.SEQ_LENGTH,
                                                             freq=dataset_constants.FREQ)

    test_loader = DataLoader(test_dataset, batch_size=_cfg.evaluate.batch_size, shuffle=False,
                             num_workers=_cfg.evaluate.n_cpus, persistent_workers=True)

    criterion_seq = nn.L1Loss(reduction='none')

    visualize_val_outputdir = os.path.join(outputdir, 'visualize_prediction_val')
    if not os.path.isdir(visualize_val_outputdir):
        os.mkdir(visualize_val_outputdir)

    mean_RMSE = []
    for i_repeat in range(_cfg.evaluate.n_repeat):
        suffix = _cfg.evaluate.suffix + f'_repeat-{i_repeat}'
        """RMSE computation"""
        total_rmse = pd.DataFrame(columns=['id_sample', 'id_hole', 'keypoint', 'method', 'method_param', 'RMSE',
                                           'type_RMSE', 'length_hole'])
        id_sample = 0
        n_plots = 0
        """Visualization 3D, one timepoint each"""

        with torch.no_grad():
            logging.info(f'Starting evaluation...')

            for ind, data_dict in tqdm.tqdm(enumerate(test_loader), desc='Iterating on batch', total=len(test_loader)):
                """Compute the prediction from networks"""

                data_with_holes = data_dict['X'].to(device)  # shape (timepoints, n_keypoints, 2 or 3 or 4)
                data_full = data_dict['x_supp'].to(device)
                mask_holes = data_dict['mask_holes'].to(device)
                assert not torch.any(torch.isnan(data_with_holes))
                assert not torch.any(torch.isnan(data_full))

                de_outs, uncertainty_estimates, _, _ = feed_forward_list(data_with_holes, mask_holes, dataset_constants.DIVIDER, models,
                                                  model_configs, device, data_full=data_full, criterion_seq=criterion_seq)

                full_data_np = data_full.detach().cpu().clone().numpy()
                data_with_holes_np = data_with_holes.detach().cpu().numpy()

                if _cfg.evaluate.original_coordinates:
                    full_data_np = reconstruct_before_normalization(full_data_np, data_dict, transforms)
                    data_with_holes_np = reconstruct_before_normalization(data_with_holes_np, data_dict, transforms)

                """Linear interpolation"""
                mask_holes_np = mask_holes.detach().cpu().numpy()

                ### put everything we need in numpy
                indices_sample = data_dict['index'].detach().cpu().numpy()

                reshaped_mask_holes = np.repeat(mask_holes_np, dataset_constants.DIVIDER, axis=-1).reshape(full_data_np.shape)
                # gives the total number of missing values in a sample (can be from multiple keypoints):
                n_missing = np.sum(mask_holes_np, axis=(1, 2))  ## (batch,)

                x_outputs_np = [out.detach().cpu().numpy() for out in de_outs]
                if _cfg.evaluate.original_coordinates:
                    x_outputs_np = [reconstruct_before_normalization(out, data_dict, transforms)
                               for out in x_outputs_np]

                # List(number of models) of tensors of size (batch, time, keypoints, 3D) if mu_sigma GRU or transformer model
                ## TODO: need to scale this in case of original coordinates!!
                uncertainty_estimates_np = [unc if unc is None else unc.detach().cpu().numpy() for unc in uncertainty_estimates]
                uncertainty = [unc if unc is None else np.sum(np.sqrt((unc ** 2) * reshaped_mask_holes), axis=3)
                               for unc in uncertainty_estimates_np]  # sum on the keypoint on dimension, shape (batch, time, keypoint)

                rmse = [np.sum(np.sqrt(((out - full_data_np) ** 2) * reshaped_mask_holes), axis=3)
                               for out in x_outputs_np]  # sum on the keypoint on dimension, shape (batch, time, keypoint)

                if np.min(_cfg.feed_data.transforms.add_missing.pad) > 0:
                    linear_interp_data = compute_interp(data_with_holes_np, mask_holes_np, dataset_constants.KEYPOINTS,
                                                        dataset_constants.DIVIDER)
                    rmse_linear_interp = np.sum(np.sqrt(((linear_interp_data - full_data_np) ** 2) * reshaped_mask_holes),
                                                axis=3)  # sum on the keypoint on dimension, shape (batch, time, keypoint)

                coverage = [[]] * n_models
                bandexcess = [[]] * n_models

                for i_model in range(n_models):
                    if model_configs[i_model].training.mu_sigma:
                        factor = 2
                        in_ = np.sum((full_data_np <= x_outputs_np[i_model] + uncertainty_estimates_np[i_model] * factor) *
                                     (full_data_np >= x_outputs_np[i_model] - uncertainty_estimates_np[i_model] * factor) *
                                     reshaped_mask_holes,
                                     axis=(1, 2, 3))
                        out_ = np.sum(((full_data_np > x_outputs_np[i_model] + uncertainty_estimates_np[i_model] * factor) +
                                       (full_data_np < x_outputs_np[i_model] - uncertainty_estimates_np[i_model] * factor)) *
                                      reshaped_mask_holes,
                                      axis=(1, 2, 3))
                        coverage[i_model] = in_ / (in_ + out_)

                        be = np.sum(
                            np.abs(np.abs(x_outputs_np[i_model] - full_data_np) - uncertainty_estimates_np[i_model] * factor) * reshaped_mask_holes,
                            axis=(1, 2, 3))
                        bandexcess[i_model] = be[n_missing > 0] / be[n_missing > 0]


                for i in range(data_with_holes_np.shape[0]):
                    ## gives the length of a hole, one keypoint at a time, a sample can have multiple holes one after the other:
                    id_hole = 0
                    out = find_holes(mask_holes_np[i], dataset_constants.KEYPOINTS, indep=False)
                    for o in out:  # (start, keypoint_name, length)
                        for j in range(n_models):
                            total_rmse.loc[total_rmse.shape[0], :] = [id_sample, id_hole, o[2],
                                                                      model_configs[j].network.type, model_name[j],
                                                                      np.mean(rmse[j][i, o[0]: o[0] + o[1], [dataset_constants.KEYPOINTS.index(kp) for kp in o[2].split(' ')]]), '3D',
                                                                      o[1]]
                            if model_configs[j].training.mu_sigma:
                                total_rmse.loc[total_rmse.shape[0], :] = [id_sample, id_hole, o[2], model_configs[0].network.type,
                                                                          model_name[j],
                                                                          np.mean(uncertainty[j][i, o[0]: o[0] + o[1],  [dataset_constants.KEYPOINTS.index(kp) for kp in o[2].split(' ')]]), 'mean_uncertainty',
                                                                          o[1]]
                                total_rmse.loc[total_rmse.shape[0], :] = [id_sample, id_hole, o[2], model_configs[0].network.type,
                                                                          model_name[j],
                                                                          np.max(uncertainty[j][i, o[0]: o[0] + o[1],  [dataset_constants.KEYPOINTS.index(kp) for kp in o[2].split(' ')]]), 'max_uncertainty',
                                                                          o[1]]
                                total_rmse.loc[total_rmse.shape[0], :] = [id_sample, id_hole, o[2], model_configs[0].network.type,
                                                                          model_name[j],
                                                                          bandexcess[j][i], 'bandexcess_2sigma',
                                                                          o[1]]
                                total_rmse.loc[total_rmse.shape[0], :] = [id_sample, id_hole, o[2], model_configs[0].network.type,
                                                                          model_name[j],
                                                                          coverage[j][i], 'coverage_2sigma',
                                                                          o[1]]
                        if np.min(_cfg.feed_data.transforms.add_missing.pad) > 0:

                            total_rmse.loc[total_rmse.shape[0], :] = [id_sample, id_hole, o[2],
                                                                      'linear_interp', 'linear_interp',
                                                                      np.mean(rmse_linear_interp[i, o[0]: o[0] + o[1],  [dataset_constants.KEYPOINTS.index(kp) for kp in o[2].split(' ')]]),
                                                                      '3D',
                                                                      o[1]]
                        id_hole += 1

                    ## the sample as a whole, not hole by hole
                    if np.min(_cfg.feed_data.transforms.add_missing.pad) > 0:
                        total_rmse.loc[total_rmse.shape[0], :] = [id_sample, -1, 'all',
                                                                  'linear_interp', 'linear_interp',
                                                                  np.sum(rmse_linear_interp[i]) / n_missing[i],
                                                                  '3D',
                                                                  n_missing[i]]
                    for j in range(n_models):
                        total_rmse.loc[total_rmse.shape[0], :] = [id_sample, -1, 'all',
                                                                  model_configs[j].network.type, model_name[j],
                                                                  np.sum(rmse[j][i]) / n_missing[i], '3D',
                                                                  n_missing[i]]
                        if model_configs[j].training.mu_sigma:
                            total_rmse.loc[total_rmse.shape[0], :] = [id_sample, -1, 'all', model_configs[0].network.type,
                                                                      model_name[j],
                                                                      np.sum(uncertainty[j][i]) / n_missing[i],
                                                                      'mean_uncertainty',
                                                                      o[1]]

                    id_sample += 1

                """VISUALIZATION, only first batch"""
                if n_plots < _cfg.evaluate.n_plots:
                    logging.info(f'Starting sample plots')
                    potential_indices = np.where(n_missing > 0)[0]
                    np.random.seed(0)
                    for i in np.random.choice(potential_indices,  #full_data_np.shape[0],
                                              min(len(potential_indices), _cfg.evaluate.n_plots),
                                              replace=False):
                        if skeleton_graph is not None:
                            for i_model, xo in enumerate(x_outputs_np):
                                plot_sequence(full_data_np[i, 1:], xo[i, 1:], mask_holes_np[i, 1:], skeleton_graph, nplots=15,
                                              save_path=os.path.join(visualize_val_outputdir,
                                                                     f'traj3D_{indices_sample[i][0]}{model_name[i_model]}{suffix}'),
                                              size=_cfg.evaluate.size, azim=_cfg.evaluate.azim,
                                              normalized_coordinates=(not _cfg.evaluate.original_coordinates))

                        title = f'RMSE '
                        title += ' -  '.join(
                            [f'{i_model}: {np.mean(rmse[i_model][i]):.3f}' for i_model in range(n_models)])
                        if np.min(_cfg.feed_data.transforms.add_missing.pad) > 0:
                            title += f'; linear:{np.mean(rmse_linear_interp[i]):.3f}'

                        def make_xyz_plot():
                            fig, axes = plt.subplots(dataset_constants.N_KEYPOINTS, dataset_constants.DIVIDER,
                                                     figsize=(max(dataset_constants.SEQ_LENGTH // 10,
                                                                  dataset_constants.DIVIDER * 7),
                                                              dataset_constants.NUM_FEATURES),
                                                     sharex='all', sharey='col')
                            axes = axes.flatten()
                            t_vect = np.arange(1, dataset_constants.SEQ_LENGTH) / dataset_constants.FREQ

                            for j in range(dataset_constants.N_KEYPOINTS):
                                if _cfg.evaluate.only_holes:
                                    t_mask = (mask_holes_np[i, 1:, j] == 1)
                                else:
                                    t_mask = np.ones_like(mask_holes_np[i, 1:, j]).astype(bool)
                                for i_dim in range(dataset_constants.DIVIDER):
                                    axes[dataset_constants.DIVIDER * j + i_dim].plot(t_vect, full_data_np[i, 1:, j, i_dim], 'o-')
                                    if np.sum(t_mask) > 0:
                                        for i_model, xo in enumerate(x_outputs_np):
                                            plot_ = axes[dataset_constants.DIVIDER * j + i_dim].plot(t_vect[t_mask], xo[i, 1:, j, i_dim][t_mask], 'o',
                                                             label=model_name[i_model], )
                                            if model_configs[i_model].training.mu_sigma:
                                                # 3 * std otherwise 1/ we do not see anything,
                                                # 2/ because the underlying distribution is supposed to be Gaussian
                                                axes[dataset_constants.DIVIDER * j + i_dim]\
                                                    .fill_between(t_vect[t_mask], xo[i, 1:, j, i_dim][t_mask]
                                                                          - 3 * uncertainty_estimates_np[i_model][i, 1:, j, i_dim][t_mask],
                                                                          xo[i, 1:, j, i_dim][t_mask]
                                                                          + 3 * uncertainty_estimates_np[i_model][i, 1:, j, i_dim][t_mask],
                                                                          color=plot_[0].get_color(), alpha=0.2)
                                            assert not np.any(np.isnan(xo))

                                    out = find_holes(np.array(t_mask).reshape(dataset_constants.SEQ_LENGTH - 1, 1).astype(int), ['0'], indep=True)
                                    if np.min(_cfg.feed_data.transforms.add_missing.pad) > 0:
                                        for o in out:
                                            axes[dataset_constants.DIVIDER * j + i_dim].plot(t_vect[o[0]:o[0]+o[1]],
                                                                                     linear_interp_data[i, 1:, j, i_dim][o[0]:o[0]+o[1]], 'r-',
                                                     label='linear interp 1D')

                                    if not _cfg.evaluate.original_coordinates:
                                        axes[dataset_constants.DIVIDER * j + i_dim].set_ylim(-1.2, 1.2)

                                if np.any(t_mask):
                                    axes[dataset_constants.DIVIDER * j].legend()
                                    axes[dataset_constants.DIVIDER * j + 1].set_title(title)

                                axes[dataset_constants.DIVIDER * j].set_ylabel(dataset_constants.KEYPOINTS[j])
                            axes[0].set_title('X')
                            axes[1].set_title('Y')
                            if dataset_constants.DIVIDER >= 3:
                                axes[2].set_title('Z')

                            return

                        plot_save(make_xyz_plot,
                                  title=f'reconstruction_xyz_{indices_sample[i][0]}{suffix}',
                                  only_png=False,
                                  outputdir=visualize_val_outputdir)

                        n_plots += 1

                    logging.info('Done with sample plots')

                for _ in range(3):
                    torch.cuda.empty_cache()
                    gc.collect()

        logging.info(f'Finished with iterating the dataset')
        total_rmse = total_rmse.reset_index().convert_dtypes()
        logging.info(f'n lines in result df: {total_rmse.shape[0]}')
        logging.info(f"RMSE per sample averaged: \n"
                     f"{total_rmse[(total_rmse['type_RMSE'] == '3D') * (total_rmse['keypoint'] == 'all')].groupby('method_param')['RMSE'].agg('mean')}")
        tmp = total_rmse[(total_rmse['type_RMSE'] == '3D') * (total_rmse['keypoint'] == 'all')].groupby(['method', 'method_param'])['RMSE'].agg('mean').reset_index()
        tmp['repeat'] = i_repeat
        tmp['dataset'] = _cfg.dataset.name
        mean_RMSE.append(tmp)

        plt.close('all')
        mask = (total_rmse['type_RMSE'] == '3D')

        def barplot_RMSE_keypoint():
            if len(_cfg.evaluate.merge_sets_file) > 0:
                with open(os.path.join(basedir, _cfg.evaluate.merge_sets_file)) as f:
                    sets2merge = json.load(f)
                for v in total_rmse.loc[total_rmse['keypoint'] != 'all', 'keypoint'].unique():
                    vv = ''.join([str(dataset_constants.KEYPOINTS.index(xx)) for xx in v.split(' ')])
                    if not vv in sets2merge:
                        sets2merge[vv] = vv
                total_rmse.loc[total_rmse['keypoint'] != 'all', 'sets'] = total_rmse.loc[total_rmse['keypoint'] != 'all', 'keypoint'].apply(lambda x: sets2merge[''.join([str(dataset_constants.KEYPOINTS.index(xx)) for xx in x.split(' ')])])
                sns.catplot(data=total_rmse.loc[mask * (total_rmse['keypoint'] != 'all'), :], kind='bar', y='sets',
                            hue='method_param', x='RMSE', orient='h')
            else:
                sns.catplot(data=total_rmse.loc[mask * (total_rmse['keypoint'] != 'all'), :], kind='bar', x='keypoint',
                            hue='method_param', y='RMSE')
            plt.tight_layout()

        plot_save(barplot_RMSE_keypoint,
                  title=f'barplot_comparison_RMSE{suffix}', only_png=False,
                  outputdir=outputdir)
        plt.close('all')

        def lineplot_length():
            total_rmse['length_hole'] = total_rmse.loc[:, 'length_hole'].astype('float')
            sns.lineplot(data=total_rmse.loc[mask * (total_rmse['keypoint'] != 'all'), :], x='length_hole', y='RMSE',
                         hue='method_param')
            plt.tight_layout()

        plot_save(lineplot_length,
                  title=f'comparison_length_hole_kp_vs_RMSE{suffix}', only_png=False,
                  outputdir=outputdir)
        plt.close('all')


        def lineplot_all_length():
            total_rmse.loc[:, 'length_hole'] = total_rmse.loc[:, 'length_hole'].astype('float')
            sns.lineplot(data=total_rmse.loc[mask * (total_rmse['keypoint'] == 'all'), :], x='length_hole', y='RMSE',
                         hue='method_param')
            plt.tight_layout()

        plot_save(lineplot_all_length,
                  title=f'comparison_length_hole_all_vs_RMSE{suffix}', only_png=False,
                  outputdir=outputdir)
        plt.close('all')

        total_rmse.to_csv(os.path.join(outputdir, f'total_RMSE{suffix}.csv'), index=False)

        thresholding_df = pd.DataFrame(columns=['th', 'RMSE', 'RMSE_std', 'count', 'method'])
        for i_model in range(n_models):
            if uncertainty_estimates[i_model] is not None:
                # pivot_df only for one method
                pivot_df = pd.pivot(
                    total_rmse.loc[(total_rmse['keypoint'] == 'all') * (total_rmse['method_param'] == model_name[i_model]), :],
                    values='RMSE', index='id_sample', columns='type_RMSE')
                pivot_df['mean_uncertainty'] = pivot_df['mean_uncertainty'].astype(float)
                pivot_df['3D'] = pivot_df['3D'].astype(float)
                pcoeff, ppval = pearsonr(pivot_df['3D'].values, pivot_df['mean_uncertainty'].values)
                logging.info(f'Model {model_name[i_model]}: PEARSONR COEFF {pcoeff}, PVAL {ppval}')

                def corr_plot():
                    sns.histplot(data=pivot_df, x='3D', y='mean_uncertainty')
                    sns.kdeplot(data=pivot_df, x='3D', y='mean_uncertainty')
                    plt.plot([0, pivot_df['3D'].max()], [0, pivot_df['3D'].max()], 'r--')
                    plt.title(f'Pearson coeff: {pcoeff:.3f}')

                plot_save(corr_plot,
                          title=f'corrplot-model-{model_name[i_model]}{suffix}', only_png=False,
                          outputdir=outputdir)
                plt.close('all')

                th_vals = np.unique(pivot_df['mean_uncertainty'])[10:]
                th_vals = th_vals[::len(th_vals) // 10]
                for th in th_vals:
                    filtered_id_samples = total_rmse.loc[
                        (total_rmse['type_RMSE'] == 'mean_uncertainty') * (total_rmse['RMSE'] <= th) * (total_rmse['keypoint'] == 'all') * (total_rmse['method_param'] == model_name[i_model]), 'id_sample'].values
                    if len(filtered_id_samples) == 0:
                        continue
                    vals = total_rmse[(total_rmse['type_RMSE'] == '3D') * (total_rmse['keypoint'] == 'all') * (total_rmse['method_param'] == model_name[i_model]) * (total_rmse['id_sample'].isin(filtered_id_samples))]['RMSE'].agg(['mean', 'std', 'count'])
                    ## add values in thresholding_df which holds the results for all uncertainty methods
                    thresholding_df.loc[thresholding_df.shape[0], :] = [th, vals['mean'], vals['std'], vals['count'], model_name[i_model]]

        if np.any([unc is not None for unc in uncertainty_estimates]):
            def plot_thresholding():
                fig, ax1 = plt.subplots(1, 1)
                for i_model in range(n_models):
                    if not model_configs[i_model].training.mu_sigma:
                        continue
                    m = model_name[i_model]
                    count = thresholding_df.loc[thresholding_df['method'] == m, 'count'].astype(int)
                    rmse = thresholding_df.loc[thresholding_df['method'] == m, 'RMSE'].astype(float)
                    rmse_std = thresholding_df.loc[thresholding_df['method'] == m, 'RMSE_std'].astype(float)
                    pl = ax1.plot(count, rmse,'+-', label=m)
                    ax1.fill_between(x=count, y1=rmse - rmse_std, y2=rmse + rmse_std, label=m, color=pl[0].get_color(), alpha=0.5)
                ax1.legend()
                ax1.set_ylabel('Mean RMSE')
                ax1.set_xlabel('Remaining samples')

            plot_save(plot_thresholding,
                      title=f'thresholding_curve{suffix}', only_png=False,
                      outputdir=outputdir)
            plt.close('all')

    pd.concat(mean_RMSE).to_csv(os.path.join(outputdir, f'mean_RMSE{_cfg.evaluate.suffix}.csv'), index=False)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format=f'[%(levelname)s][%(asctime)s] %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S',
                        )
    logger = logging.getLogger(__name__)

    evaluate()
