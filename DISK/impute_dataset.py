import os
import shutil
from glob import glob
from pathlib import Path
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat, loadmat
import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import pickle
import h5py
from shutil import rmtree

from DISK.utils.dataset_utils import load_datasets
from DISK.utils.logger_setup import logger
from DISK.utils.utils import read_constant_file, load_checkpoint
from DISK.utils.transforms import init_transforms, reconstruct_before_normalization
from DISK.utils.train_fillmissing import construct_NN_model, feed_forward
from DISK.test_fillmissing import plot_save
from DISK.create_dataset import chop_coordinates_in_timeseries
from DISK.utils.coordinates_utils import plot_sequence
from DISK.models.graph import Graph

import torch
from torch.utils.data import DataLoader


def save_data_original_format(data, time, file, dataset_constants, cfg_dataset, new_folder):
    """
    :args data: numpy array of 2 dimensions (timepoints, keypoints * 2D or 3D)
    :args time: numpy array with timepoints
    :args file: path to original file (Str)
    :args dataset_constants: dataset constants (dict)
    :args new_folder: path to new folder to save the imputed data in original format (str)

    :return: None
    """
    new_file = os.path.join(new_folder, os.path.basename(file))

    if cfg_dataset['sequential']:
        if os.path.exists(new_file):
            # reopen the new_file because it is to be complete multiple times
            file = new_file

    data = data[time != -1]
    time = time[time != -1] # time is original time / subsampling_freq
    time_int = np.array(np.round(time * dataset_constants.FREQ, 0), dtype=np.uint64)

    if dataset_constants.ORIG_FREQ > dataset_constants.FREQ:
        time_orig = np.unique(np.linspace(time[0] * dataset_constants.ORIG_FREQ / dataset_constants.FREQ,
                                time[-1] * dataset_constants.ORIG_FREQ / dataset_constants.FREQ,
                                int(len(time) * dataset_constants.ORIG_FREQ / dataset_constants.FREQ)).astype(int))

        data_orig = np.vstack([np.interp(time_orig, time, d) for d in data.T]).T
        time = time_orig
        data = data_orig

    data = data[:len(time)].reshape((time.shape[0], len(dataset_constants.KEYPOINTS), -1))

    if dataset_constants.FILE_TYPE == 'mat_dannce':
        # for Rat7M dataset
        # mat['mocap'][0][0].dtype.fields.keys = keypoints
        mat = loadmat(file)

        logger.debug(f'Changing file {os.path.basename(file)} from {int(time[0])} to {int(time[-1])}')

        orig_data = np.array(list(mat['mocap'][0][0]))
        orig_data[:, time.astype(int)] = np.moveaxis(data, 0, 1)
        mat['mocap'] = ((orig_data,),)

        savemat(new_file, mat)

    elif dataset_constants.FILE_TYPE == 'mat_qualisys':
        mat = loadmat(file)
        exp_name = [m for m in mat.keys() if m[:2] != '__'][0]  ## TOCHANGE
        # for in house mouse data, QUALISYS software
        mat[exp_name][0, 0]['Trajectories'][0, 0]['Labeled']['Data'][0, 0] = np.moveaxis(data, 0, 2)
        mat[exp_name][0, 0]['Trajectories'][0, 0]['Labeled']['Labels'][0, 0][0] = dataset_constants.KEYPOINTS
        savemat(new_file, mat)

    elif dataset_constants.FILE_TYPE == 'simple_csv':
        ## for fish data from Liam
        # columns frame_index, keypoint_x, kp_y, kp_z

        columns = []
        for k in dataset_constants.KEYPOINTS:
            for ii in range(dataset_constants.DIVIDER):
                columns.append(k + ['_x', '_y', '_z'][ii])

        df = pd.read_csv(file)

        logger.debug(
            f'BEFORE -- nb of nans in data: {np.sum(np.isnan(data))}; '
            f'nb of nans in df: {df.loc[time_int, columns].isna().sum().sum()}')

        to_replace = data.reshape((data.shape[0], -1))
        if np.any(np.isnan(to_replace)):
            to_replace[np.isnan(to_replace)] = df.loc[time_int, columns].values[np.isnan(to_replace)]

        logger.debug(f'modifying {np.sum(~np.isclose(to_replace,  df.loc[time_int, columns].values))} values between '
                  f'indices {np.min(time_int)} and {np.max(time_int)}')

        df.loc[time_int, columns] = to_replace

        logger.debug(
            f'AFTER -- nb of nans in data: {np.sum(np.isnan(data))}; '
            f'nb of nans in df: {df.loc[time_int, columns].isna().sum().sum()}')

        df.to_csv(new_file, index=False)

    elif 'dlc' in dataset_constants.FILE_TYPE:
        # the dlc_h5 format is quite similar as dlc csv, the "table" is corresponding to the values of the csv
        # the idea is to do the manipulation on a pandas df format
        # the df is a multi-index df with 3 levels when 1 animal, and 4 levels when multianimal

        if dataset_constants.FILE_TYPE == 'dlc_h5':
            content = h5py.File(file)
            extracted_content = np.vstack([c[1] for c in content['df_with_missing']['table'][:]])
            values_block = content['df_with_missing']['table'].attrs['values_block_0_kind']
            multi_index = pickle.loads(values_block)

            index = pd.MultiIndex.from_tuples(multi_index)
            df = pd.DataFrame(columns=index, data=extracted_content)

            if len(multi_index[0]) > 3:
                # multianimal
                df.loc[:, ('scorer', 'individuals', 'bodyparts', 'coords')] = np.arange(len(df))
            else:
                # 1 animal
                df.loc[:, ('scorer', 'bodyparts', 'coords')] = np.arange(len(df))

        elif dataset_constants.FILE_TYPE == 'dlc_csv':
            ## for csv from DeepLabCut
            df = pd.read_csv(file, header=[0, 1, 2])

        if 'individuals' in df.columns.levels[1]:
            if dataset_constants.FILE_TYPE == 'dlc_csv':
                df = pd.read_csv(file, header=[0, 1, 2, 3])
            header = [c for c in df.columns.levels[0] if c != 'scorer'][0]

            # multianimal
            individuals = [ind for ind in df.columns.levels[1] if ind != 'individuals']
            individuals.sort()
            keypoints = [bp for bp in df.columns.levels[2] if bp != 'bodyparts']
            keypoints.sort()
            coordinates = [c for c in df.columns.levels[3] if c != 'likelihood' and c != 'coords']

            # WIP: how to replace the likelihood where we have changed the values
            columns = []
            likelihood_columns = []
            for ind in individuals:
                for k in keypoints:
                    likelihood_columns.append((header, ind, k, 'likelihood'))
                    for c in coordinates:
                        columns.append((header, ind, k, c))
                        # df.loc[df.loc[:, (header, ind, k, 'likelihood')] <= dataset_constants.DLC_LIKELIHOOD_THRESHOLD, (header, ind, k, c)] = np.nan
                    # df.loc[df.loc[:, (header, ind, k, 'likelihood')] <= dataset_constants.DLC_LIKELIHOOD_THRESHOLD, (header, ind, k, 'likelihood')] = np.nan
            # make sure the time mask and the number of values we want to modify are the same

            if not np.sum(df[('scorer', 'individuals', 'bodyparts', 'coords')].isin(time_int)) == data.shape[0]:
                raise ValueError('[save_data_original_format][dlc_csv] shape incompatibility')

            logger.debug(f'BEFORE -- nb of nans in data: {np.sum(np.isnan(data))}; nb of nans in df: {df[columns].isna().sum().sum()}')

            to_replace = np.array(data.reshape((data.shape[0], -1)))
            to_replace[np.isnan(to_replace)] = df.loc[df[('scorer', 'individuals', 'bodyparts', 'coords')].isin(time_int), columns].values[np.isnan(to_replace)]
            df.loc[df[('scorer', 'individuals', 'bodyparts', 'coords')].isin(time_int), columns] = to_replace

            # for now replace likelihood with -1 to mark the positions where we modified the coordinate values
            logger.debug(f'AFTER -- nb of nans in data: {np.sum(np.isnan(data))}; nb of nans in df: {df[columns].isna().sum().sum()}')
            logger.debug(f'modifying values between indices {np.min(time_int)} and {np.max(time_int)}')
        else:
            # single animal
            header = [c for c in df.columns.levels[0] if c != 'scorer'][0]
            keypoints = [bp for bp in df.columns.levels[1] if bp != 'bodyparts']
            keypoints.sort()
            coordinates = [c for c in df.columns.levels[2] if c != 'likelihood' and c != 'coords']

            # how to replace the likelihood where we have changed the values
            columns = []
            likelihood_columns = []
            for k in keypoints:
                likelihood_columns.append((header, k, 'likelihood'))
                for c in coordinates:
                    columns.append((header, k, c))
                    # df.loc[df.loc[:, (header, k, 'likelihood')] <= dataset_constants.DLC_LIKELIHOOD_THRESHOLD, (header, k, c)] = np.nan
                # df.loc[df.loc[:, (header, k, 'likelihood')] <= dataset_constants.DLC_LIKELIHOOD_THRESHOLD, (header, k, 'likelihood')] = np.nan
            assert np.sum(df[('scorer', 'bodyparts', 'coords')].isin(time_int)) == data.shape[0]

            logger.debug(f'BEFORE -- nb of nans in data: {np.sum(np.isnan(data))}; nb of nans in df: {df[columns].isna().sum().sum()}')

            to_replace = np.array(data.reshape((data.shape[0], -1)))
            to_replace[np.isnan(to_replace)] = df.loc[df[('scorer', 'bodyparts', 'coords')].isin(time_int), columns].values[np.isnan(to_replace)]
            df.loc[df[('scorer', 'bodyparts', 'coords')].isin(time_int), columns] = to_replace

            logger.debug(f'AFTER -- nb of nans in data: {np.sum(np.isnan(to_replace))}; nb of nans in df: {df[columns].isna().sum().sum()}')
            logger.debug(f'modifying values between indices {np.min(time_int)} and {np.max(time_int)}')

        if dataset_constants.FILE_TYPE == 'dlc_csv':
            # save to csv
            df.to_csv(new_file, index=False)

        elif dataset_constants.FILE_TYPE == 'dlc_h5':
            attrs_dict = dict(content['df_with_missing']['table'].attrs)
            i_table = content['df_with_missing']['_i_table']
            content.close()
            with h5py.File(new_file, 'w') as openedf:
                dataset = openedf.create_dataset('df_with_missing/table',
                                                 data=np.array([(int(i_), c) for i_, c in zip(df.values[:, -1], df.values[:, :-1])],
                                                               dtype=[('index', '<i8'), ('values_block_0', '<f8', (df.shape[1] -1,))]))
                for k, v in attrs_dict.items():
                    dataset.attrs[k] = v
                openedf.create_group('df_with_missing/_i_table', i_table)

    elif dataset_constants.FILE_TYPE == 'npy':
        ## for human MoCap files
        # plain npy, no keypoints name, expected shape (n_samples, n_keypoints, n_dim)

        orig_data = np.load(file)
        to_save = np.array(orig_data)
        to_data = np.array(data)
        to_data[~np.isnan(orig_data[time_int])] = to_save[time_int][~np.isnan(orig_data[time_int])]
        to_save[time_int] = to_data
        np.save(new_file, to_save)

        logger.debug(
            f'modifying {np.sum(~np.isclose(to_save, orig_data))} values between indices {np.min(time_int)} '
            f'and {np.max(time_int)}, file: {os.path.basename(new_file)}')

    elif dataset_constants.FILE_TYPE == 'df3d_pkl':
        ## for DeepFly data
        pkl_content = {'points3d': data, 'keypoints': dataset_constants.KEYPOINTS}
        with open(new_file, 'rb') as openedf:
            pickle.dump(pkl_content, openedf, protocol=pickle.HIGHEST_PROTOCOL)
        """ from DeepFly3D paper
        38 landmarks per animal: (i) five on each limb – the thorax-coxa, coxa-femur, femur-tibia, and tibia-tarsus 
        joints as well as the pretarsus, (ii) six on the abdomen - three on each side, and (iii) one on each antenna
         - for measuring head rotations.
         see image on github too
        """

    elif dataset_constants.FILE_TYPE == 'sleap_h5':
        ## compatibility with SLEAP analysis h5 files
        if dataset_constants.KEYPOINTS[0].startswith('animal'):
            # several animals
            keypoints_per_animal = ['_'.join(k.split('_')[1:]) for k in dataset_constants.KEYPOINTS if k.startswith('animal0')]
            data = np.moveaxis(data.reshape(data.shape[0], -1, len(keypoints_per_animal), data.shape[2]), 1, 3)
        else:
            # one animal
            data = data[..., np.newaxis]
            keypoints_per_animal = dataset_constants.KEYPOINTS

        with h5py.File(new_file, 'w') as openedf:
            openedf['tracks'] = data.T
            openedf["node_names"] = keypoints_per_animal

    else:
        raise ValueError(f'File format not understood {file}')

    return


@hydra.main(version_base=None, config_path="conf", config_name="conf_impute")
def evaluate(_cfg: DictConfig) -> None:
    outputdir = os.getcwd()
    basedir = hydra.utils.get_original_cwd()
    logger.info(f'[BASEDIR] {basedir}')
    logger.info(f'[OUTPUT DIR] {outputdir}')
    """ LOGGING AND PATHS """

    logger.info(f'{_cfg}')

    dataset_path = os.path.join(basedir, 'datasets', _cfg.dataset.name)
    constant_file_path = os.path.join(dataset_path, f'constants.py')
    if not os.path.exists(constant_file_path):
        raise ValueError(f'no constant file found at {constant_file_path}')
    dataset_constants = read_constant_file(constant_file_path)
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

    dataset_config_file = os.path.join(dataset_path, '.hydra', 'config_create_dataset.yaml')
    cfg_dataset = OmegaConf.load(dataset_config_file)

    config_file = os.path.join(_cfg.evaluate.checkpoint, '.hydra', 'config.yaml')
    cfg_model = None
    if os.path.exists(config_file):
        cfg_model = OmegaConf.load(config_file)
        logger.info(f'Found model at path {_cfg.evaluate.checkpoint}')
        model_path = glob(os.path.join(_cfg.evaluate.checkpoint, 'model_epoch*'))[0]
    else:
        for path in Path(os.path.join(basedir, _cfg.evaluate.checkpoint)).rglob('model_epoch*'):
            logger.info(f'Found model at path {str(path)}')
            config_file = os.path.join(os.path.dirname(path), '.hydra', 'config.yaml')
            cfg_model = OmegaConf.load(config_file)
            model_path = path
    if cfg_model is None:
        raise ValueError(f'no model found at path {_cfg.evaluate.checkpoint}')
    logger.debug(f'Full path to model: {model_path}')

    """ DATA """
    logger.info('Loading prediction model...')
    # load model
    model_name = ''
    model = construct_NN_model(cfg_model, dataset_constants, _cfg.dataset.skeleton_file, device)
    for ini, name_item in enumerate(_cfg.evaluate.name_items):
        val = cfg_model[name_item[0]]
        for item in name_item[1:]:
            val = val[item]
        if ini == 0:
            model_name = f'{name_item[-1]}-{val}'
        else:
            model_name += f'_{name_item[-1]}-{val}'
    logger.info(f'Network {model_name} constructed')

    load_checkpoint(model, None, model_path, device)

    visualize_val_outputdir = os.path.join(outputdir, 'visualize_prediction')
    if not os.path.isdir(visualize_val_outputdir):
        os.mkdir(visualize_val_outputdir)

    """RMSE computation"""
    """Visualization 3D, one timepoint each"""
    data_subpath = os.path.join(dataset_path, 'original_data_format')
    if os.path.exists(data_subpath):
        rmtree(data_subpath)
    os.mkdir(data_subpath)

    n_plots = 0

    transforms, proba_n_missing = init_transforms(cfg_model, dataset_constants.KEYPOINTS, dataset_constants.DIVIDER,
                                 dataset_constants.SEQ_LENGTH, basedir, outputdir, add_missing=False)
    
    if proba_n_missing is None or np.max(np.where(proba_n_missing > 0)[0]) > 1:
        all_segments = True
    else:
        all_segments = False

    # return full length dataset for imputation
    train_dataset, val_dataset, test_dataset = load_datasets(dataset_name=_cfg.dataset.name,
                                                             dataset_constants=dataset_constants,
                                                             transform=transforms,
                                                             dataset_type='impute',
                                                             suffix='_w-all-nans',
                                                             root_path=basedir,
                                                             outputdir=outputdir,
                                                             label_type='all',  # don't care, not using
                                                             verbose=_cfg.feed_data.verbose,
                                                             padding=_cfg.feed_data.pad,
                                                             keypoints_bool=True,
                                                             skeleton_file=skeleton_file_path,
                                                             stride=dataset_constants.STRIDE,
                                                             length_sample=dataset_constants.SEQ_LENGTH,
                                                             freq=dataset_constants.FREQ,
                                                             all_segments=all_segments)

    """LOOPING ON DATA"""
    with torch.no_grad():
        for subset, dataset in {'test': test_dataset, 'val': val_dataset, 'train': train_dataset}.items():
            data_loader = DataLoader(dataset, batch_size=_cfg.feed_data.batch_size, shuffle=False)
            # num_workers = _cfg.evaluate.n_cpus, persistent_workers = True)

            with torch.no_grad():

                for ind, data_dict in tqdm(enumerate(data_loader), desc=f'Running DISK on the {subset} dataset',
                                                total=len(data_loader)):
                    """Compute the prediction from networks"""

                    transformed_data = data_dict['X'].to(device)
                    mask_holes = data_dict['mask_holes'].to(device)
                    lengths = data_dict['length_seq'].to(device)
                    assert not torch.any(torch.isnan(transformed_data))

                    de_out = feed_forward(transformed_data, mask_holes,  # 1 for missing, 0 for non-missing
                                          dataset_constants.DIVIDER, model, cfg_model,
                                          key_padding_mask=lengths)
                    # References for key_padding_mask for transformer
                    # https://pytorch.org/docs/stable/_modules/torch/nn/modules/activation.html#MultiheadAttention
                    # https://stackoverflow.com/questions/62629644/what-the-difference-between-att-mask-and-key-padding-mask-in-multiheadattnetion
                    # and for GRU
                    # https://www.kaggle.com/code/kaushal2896/packed-padding-masking-with-attention-rnn-gru
                    # https://gist.github.com/HarshTrivedi/f4e7293e941b17d19058f6fb90ab0fec

                    transformed_data_np = transformed_data.detach().cpu().numpy()
                    untransformed_data_np = reconstruct_before_normalization(transformed_data_np, data_dict, transforms)

                    x_output_np = de_out[0].detach().cpu().numpy()
                    x_output_np = reconstruct_before_normalization(x_output_np, data_dict, transforms)

                    mask_holes_np = mask_holes.detach().cpu().numpy()

                    if de_out[1] is not None:
                        # for proba models
                        reshaped_mask_holes = np.repeat(mask_holes_np, dataset_constants.DIVIDER, axis=-1)\
                                              .reshape(x_output_np.shape)
                        uncertainty = np.sum(
                        np.sqrt((de_out[1].detach().cpu().numpy() ** 2) * reshaped_mask_holes),
                        axis=3)  # sum on the keypoint on dimension, shape (batch, time, keypoint)
                    else:
                        uncertainty = None

                    dataset.update_dataset(data_dict['index'], x_output_np, uncertainty,
                                                                threshold=_cfg.evaluate.threshold_error_score)

                    """VISUALIZATION, only first batch"""
                    if _cfg.evaluate.n_plots > 0 and n_plots <= _cfg.evaluate.n_plots:

                        mean_ = np.nanmean(untransformed_data_np, axis=(1, 2))
                        max_ = np.nanmax(untransformed_data_np, axis=(1, 2))
                        min_ = np.nanmin(untransformed_data_np, axis=(1, 2))

                        for i in np.random.choice(untransformed_data_np.shape[0],
                                                  min(untransformed_data_np.shape[0], _cfg.evaluate.n_plots),
                                                  replace=False):
                            if skeleton_graph is not None:
                                plot_sequence(transformed_data_np[i, 1:], x_output_np[i, 1:], mask_holes_np[i, 1:].astype('int'), skeleton_graph,
                                              nplots=15,
                                              save_path=os.path.join(visualize_val_outputdir,
                                                                     f'traj3D_{data_dict["indices_file"][i]}-{data_dict["indices_pos"][i]}{_cfg.evaluate.suffix}'),
                                              size=2, normalized_coordinates=False)

                            def make_xyz_plot():
                                fig, axes = plt.subplots(dataset_constants.N_KEYPOINTS, dataset_constants.DIVIDER,
                                                         figsize=(max(dataset_constants.SEQ_LENGTH // 3, 16),
                                                                  dataset_constants.NUM_FEATURES))
                                axes = axes.flatten()
                                t_vect = np.arange(0, dataset_constants.SEQ_LENGTH) / dataset_constants.FREQ

                                for j in range(dataset_constants.N_KEYPOINTS):
                                    if _cfg.evaluate.only_holes:
                                        t_mask = (mask_holes[i, :, j] == 1).detach().cpu().numpy()
                                    else:
                                        t_mask = np.ones_like(mask_holes[i, :, j].detach().cpu().numpy()).astype(bool)

                                    for i_dim in range(dataset_constants.DIVIDER):
                                        d = untransformed_data_np[i, :, j, i_dim]
                                        axes[dataset_constants.DIVIDER * j + i_dim].plot(t_vect[:lengths[i]][~t_mask[:lengths[i]]],
                                                                                         d[:lengths[i]][~t_mask[:lengths[i]]],
                                                                                         'o-',
                                                                                         label='reconstruct after norm')

                                        axes[dataset_constants.DIVIDER * j + i_dim].plot(t_vect[1:lengths[i]][t_mask[1:lengths[i]]],
                                                                                 x_output_np[i, 1:lengths[i], j, i_dim][t_mask[1:lengths[i]]],
                                                                                 'o', label=model_name)

                                        assert not np.any(np.isnan(x_output_np))

                                        axes[dataset_constants.DIVIDER * j + i_dim].set_ylim(min(mean_[i, i_dim] - 20, min_[i, i_dim] - 5),
                                                                                             max(mean_[i, i_dim] + 20, max_[i, i_dim] + 5))

                                    if np.any(t_mask):
                                        axes[dataset_constants.DIVIDER * j].legend()

                                    axes[dataset_constants.DIVIDER * j].set_ylabel(dataset_constants.KEYPOINTS[j])

                                axes[0].set_title('X')
                                axes[1].set_title('Y')
                                if dataset_constants.DIVIDER >= 3:
                                    axes[2].set_title('Z')

                                return

                            if _cfg.evaluate.save:
                                plot_save(make_xyz_plot,
                                          title=f'reconstruction_xyz_{data_dict["indices_file"][i]}-{data_dict["indices_pos"][i]}{_cfg.evaluate.suffix}',
                                          only_png=False,
                                          outputdir=visualize_val_outputdir)
                            else:
                                make_xyz_plot()
                                plt.show()
                            n_plots += 1

            logger.info(f'{subset}, dataset_path = {dataset_path}')

            if _cfg.evaluate.save_dataset:
                if dataset.y is None:
                    np.savez(os.path.join(dataset_path, f'{subset}_fulllength_dataset_imputed.npz'),
                             X=dataset.X, time=dataset.time)

                else:
                    np.savez(os.path.join(dataset_path, f'{subset}_fulllength_dataset_imputed.npz'),
                             X=dataset.X, y=dataset.y, time=dataset.time)

                if dataset.files is not None:
                    for i_f, f in enumerate(dataset.files):
                        save_data_original_format(dataset.X[i_f], dataset.time[i_f],
                                                  os.path.join(basedir, _cfg.evaluate.path_to_original_files, f),
                                                  dataset_constants, cfg_dataset, data_subpath)

                # saving new chunked dataset
                new_dataset = []
                new_lengths = []
                new_y = []
                for i_recording in range(dataset.X.shape[0]):
                    mask_t = dataset.time[i_recording] > -1
                    logger.debug(f'LINE 412 in IMPUTE_DATASET - shape: {mask_t.shape} {dataset.X[i_recording].shape} {dataset.X[i_recording][mask_t].shape}')
                    x = dataset.X[i_recording]
                    x = x.reshape(mask_t.shape[0], len(dataset_constants.KEYPOINTS), -1) # should be of shape (timepoints, keypoints, 2 or 3)
                    data, len_, t_ = chop_coordinates_in_timeseries(dataset.time[i_recording][mask_t],
                                                                    x[mask_t],
                                                                    stride=dataset_constants.STRIDE,
                                                                    length=dataset_constants.SEQ_LENGTH)

                    if len(data) > 0:
                        new_dataset.extend(data)
                        new_lengths.extend(len_)
                        if dataset.y is not None:
                            new_y.extend(
                                np.hstack([np.tile(dataset.y[i_recording], len(len_)).reshape(
                                    (len(len_), dataset.y.shape[-1])), np.expand_dims(t_ / dataset_constants.FREQ, 1)]))

                logger.debug(f'New dataset {subset} has shape {np.stack(new_dataset, axis=0).shape}.')
                if dataset.y is None:
                    np.savez(os.path.join(dataset_path, f'{subset}_dataset_imputed.npz'), X=np.stack(new_dataset, axis=0),
                             lengths=np.stack(new_lengths))
                else:
                    np.savez(os.path.join(dataset_path, f'{subset}_dataset_imputed.npz'), X=np.stack(new_dataset, axis=0),
                             y=np.stack(new_y), lengths=np.stack(new_lengths))


if __name__ == '__main__':

    evaluate()
