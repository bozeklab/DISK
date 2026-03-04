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
from DISK.utils.utils import read_constant_file, load_checkpoint
from DISK.utils.transforms import init_transforms, reconstruct_before_normalization
from DISK.utils.train_fillmissing import construct_NN_model, feed_forward
from DISK.test_fillmissing import plot_save
from DISK.create_dataset import chop_coordinates_in_timeseries
from DISK.utils.coordinates_utils import plot_sequence
from DISK.models.graph import Graph

import torch
from torch.utils.data import DataLoader


def save_data_original_format(data, time, file, file_type, keypoints, orig_freq, subsampling_freq, data_divider,
                              new_folder, logger):
    """
    :args data: numpy array of 2 dimensions (timepoints, keypoints * 2D or 3D)
    :args time: numpy array with timepoints
    :args file: path to original file (Str)
    :args dataset_constants: dataset constants (dict)
    :args new_folder: path to new folder to save the imputed data in original format (str)

    :return: None
    """
    new_file = os.path.join(new_folder, os.path.basename(file))

    if os.path.exists(new_file):
        # reopen the new_file because it is to be complete multiple times
        file = new_file

    data = data[time != -1]
    time = time[time != -1] # time is original time / subsampling_freq
    time_int = np.array(np.round(time * subsampling_freq, 0), dtype=np.uint64)

    if orig_freq > subsampling_freq:
        time_orig = np.unique(np.linspace(time[0] * orig_freq / subsampling_freq,
                                time[-1] * orig_freq / subsampling_freq,
                                int(len(time) * orig_freq / subsampling_freq)).astype(int))

        data_orig = np.vstack([np.interp(time_orig, time, d) for d in data.T]).T
        time = time_orig
        data = data_orig

    data = data[:len(time)].reshape((time.shape[0], len(keypoints), -1))

    if file_type == 'mat_dannce':
        # for Rat7M dataset
        # mat['mocap'][0][0].dtype.fields.keys = keypoints
        mat = loadmat(file)

        logger.debug(f'Changing file {os.path.basename(file)} from {int(time[0])} to {int(time[-1])}')

        orig_data = np.array(list(mat['mocap'][0][0]))
        orig_data[:, time.astype(int)] = np.moveaxis(data, 0, 1)
        mat['mocap'] = ((orig_data,),)

        savemat(new_file, mat)

    elif file_type == 'mat_qualisys':
        mat = loadmat(file)
        exp_name = [m for m in mat.keys() if m[:2] != '__'][0]  ## TOCHANGE
        # for in house mouse data, QUALISYS software
        mat[exp_name][0, 0]['Trajectories'][0, 0]['Labeled']['Data'][0, 0] = np.moveaxis(data, 0, 2)
        mat[exp_name][0, 0]['Trajectories'][0, 0]['Labeled']['Labels'][0, 0][0] = keypoints
        savemat(new_file, mat)

    elif file_type == 'simple_csv':
        ## for fish data from Liam
        # columns frame_index, keypoint_x, kp_y, kp_z

        columns = []
        for k in keypoints:
            for ii in range(data_divider):
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

    elif 'dlc' in file_type:
        # the dlc_h5 format is quite similar as dlc csv, the "table" is corresponding to the values of the csv
        # the idea is to do the manipulation on a pandas df format
        # the df is a multi-index df with 3 levels when 1 animal, and 4 levels when multianimal

        if file_type == 'dlc_h5':
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

        elif file_type == 'dlc_csv':
            ## for csv from DeepLabCut
            df = pd.read_csv(file, header=[0, 1, 2])

        if 'individuals' in df.columns.levels[1]:
            if file_type == 'dlc_csv':
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

        if file_type == 'dlc_csv':
            # save to csv
            df.to_csv(new_file, index=False)

        elif file_type == 'dlc_h5':
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

    elif file_type == 'npy':
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

    elif file_type == 'df3d_pkl':
        ## for DeepFly data
        pkl_content = {'points3d': data, 'keypoints': keypoints}
        with open(new_file, 'rb') as openedf:
            pickle.dump(pkl_content, openedf, protocol=pickle.HIGHEST_PROTOCOL)
        """ from DeepFly3D paper
        38 landmarks per animal: (i) five on each limb – the thorax-coxa, coxa-femur, femur-tibia, and tibia-tarsus 
        joints as well as the pretarsus, (ii) six on the abdomen - three on each side, and (iii) one on each antenna
         - for measuring head rotations.
         see image on github too
        """

    elif file_type == 'sleap_h5':
        ## compatibility with SLEAP analysis h5 files
        if keypoints[0].startswith('animal'):
            # several animals
            keypoints_per_animal = ['_'.join(k.split('_')[1:]) for k in keypoints if k.startswith('animal0')]
            data = np.moveaxis(data.reshape(data.shape[0], -1, len(keypoints_per_animal), data.shape[2]), 1, 3)
        else:
            # one animal
            data = data[..., np.newaxis]
            keypoints_per_animal = keypoints

        with h5py.File(new_file, 'w') as openedf:
            openedf['tracks'] = data.T
            openedf["node_names"] = keypoints_per_animal

    else:
        raise ValueError(f'File format not understood {file}')

    return


def impute(project_dir, impute_dir, plot_dir, file_type, dataset_path, skeleton_graph, checkpoint, batch_size,
           threshold_error_score, total_n_plots, plot_only_holes, missing_pad, verbose=0, logger=None) -> None:


    constant_file_path = os.path.join(dataset_path, f'constants.py')
    if not os.path.exists(constant_file_path):
        raise ValueError(f'no constant file found at {constant_file_path}')
    dataset_constants = read_constant_file(constant_file_path)
    n_keypoints = len(dataset_constants.KEYPOINTS)
    stride = dataset_constants.STRIDE
    keypoints = dataset_constants.KEYPOINTS
    orig_freq = dataset_constants.ORIG_FREQ
    subsampling_freq = dataset_constants.FREQ
    seq_length = dataset_constants.SEQ_LENGTH
    data_divider = dataset_constants.DIVIDER

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info("Device: {}".format(device))

    # dataset_config_file = os.path.join(dataset_path, 'config', 'config_prepare_data.yaml')
    # cfg_dataset = OmegaConf.load(dataset_config_file)

    config_file = os.path.join(checkpoint, 'config', 'config_train.yaml')
    cfg_model = None
    if os.path.exists(config_file):
        cfg_model = OmegaConf.load(config_file)
        logger.info(f'Found model at path {checkpoint}')
        model_path = glob(os.path.join(checkpoint, 'model_epoch*'))[0]
        model_name = os.path.basename(model_path)
    else:
        for path in Path(checkpoint).rglob('model_epoch*'):
            logger.info(f'Found model at path {str(path)}')
            config_file = os.path.join(os.path.dirname(path), 'config', 'config_train.yaml')
            cfg_model = OmegaConf.load(config_file)
            model_path = path
            model_name = os.path.basename(model_path)
    if cfg_model is None:
        raise ValueError(f'no model found at path {checkpoint}')
    logger.debug(f'Full path to model: {model_path}')

    """ DATA """
    logger.info('Loading prediction model...')
    # load model
    model_name = ''
    model = construct_NN_model(cfg_model.network, keypoints, data_divider, seq_length, skeleton_graph, device)

    logger.info(f'Network {model_name} constructed')

    load_checkpoint(model, None, model_path, device, logger)

    """RMSE computation"""
    """Visualization 3D, one timepoint each"""

    n_plots = 0

    transforms = init_transforms(
                                keypoints,
                                data_divider,
                                seq_length,
                                impute_dir,
                                logger,
                                add_missing=False,
                                viewinvariant=cfg_model.transforms.viewinvariant,
                                normalize=cfg_model.transforms.normalize,
                                normalizecube=cfg_model.transforms.normalizecube,
                                swap=0
                                        )


    # return full length dataset for imputation
    train_dataset, val_dataset, test_dataset = load_datasets(
        dataset_path=dataset_path,
         transform=transforms,
         dataset_type='impute',
         suffix='_w-all-nans',
         root_path=project_dir,
         outputdir=impute_dir,
        keypoints=keypoints,
         label_type='all',  # don't care, not using
         verbose=verbose,
         padding=missing_pad,
         skeleton_graph=skeleton_graph,
         seq_length=seq_length,
         stride=stride,
         freq=subsampling_freq,
        divider=data_divider,
    logger=logger)

    """LOOPING ON DATA"""
    with torch.no_grad():
        for subset, dataset in {'test': test_dataset, 'val': val_dataset, 'train': train_dataset}.items():
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
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
                                          data_divider, model,
                                          True, 1,
                                          cfg_model.network,
                                          key_padding_mask=lengths, logger=logger)
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
                        reshaped_mask_holes = np.repeat(mask_holes_np, data_divider, axis=-1)\
                                              .reshape(x_output_np.shape)
                        uncertainty = np.sum(
                        np.sqrt((de_out[1].detach().cpu().numpy() ** 2) * reshaped_mask_holes),
                        axis=3)  # sum on the keypoint on dimension, shape (batch, time, keypoint)
                    else:
                        uncertainty = None

                    dataset.update_dataset(data_dict['index'], x_output_np, uncertainty,
                                                                threshold=threshold_error_score)

                    """VISUALIZATION, only first batch"""
                    if total_n_plots > 0 and n_plots <= total_n_plots:

                        mean_ = np.nanmean(untransformed_data_np, axis=(1, 2))
                        max_ = np.nanmax(untransformed_data_np, axis=(1, 2))
                        min_ = np.nanmin(untransformed_data_np, axis=(1, 2))

                        for i in np.random.choice(untransformed_data_np.shape[0],
                                                  min(untransformed_data_np.shape[0], total_n_plots),
                                                  replace=False):
                            if skeleton_graph is not None:
                                plot_sequence(transformed_data_np[i, 1:], x_output_np[i, 1:], mask_holes_np[i, 1:].astype('int'), skeleton_graph,
                                              nplots=15,
                                              save_path=os.path.join(plot_dir,
                                                                     f'traj3D_{data_dict["indices_file"][i]}-{data_dict["indices_pos"][i]}'),
                                              size=2, normalized_coordinates=False)

                            def make_xyz_plot():
                                fig, axes = plt.subplots(n_keypoints, data_divider,
                                                         figsize=(max(seq_length // 3, 16),
                                                                  n_keypoints * data_divider))
                                axes = axes.flatten()
                                t_vect = np.arange(0, seq_length) / subsampling_freq

                                for j in range(n_keypoints):
                                    if plot_only_holes:
                                        t_mask = (mask_holes[i, :, j] == 1).detach().cpu().numpy()
                                    else:
                                        t_mask = np.ones_like(mask_holes[i, :, j].detach().cpu().numpy()).astype(bool)

                                    for i_dim in range(data_divider):
                                        d = untransformed_data_np[i, :, j, i_dim]
                                        axes[data_divider * j + i_dim].plot(t_vect[:lengths[i]][~t_mask[:lengths[i]]],
                                                                                         d[:lengths[i]][~t_mask[:lengths[i]]],
                                                                                         'o-',
                                                                                         label='reconstruct after norm')

                                        axes[data_divider * j + i_dim].plot(t_vect[1:lengths[i]][t_mask[1:lengths[i]]],
                                                                                 x_output_np[i, 1:lengths[i], j, i_dim][t_mask[1:lengths[i]]],
                                                                                 'o', label=model_name)

                                        assert not np.any(np.isnan(x_output_np))

                                        axes[data_divider * j + i_dim].set_ylim(min(mean_[i, i_dim] - 20, min_[i,
                                        i_dim] - 5),
                                                                                             max(mean_[i, i_dim] + 20, max_[i, i_dim] + 5))

                                    if np.any(t_mask):
                                        axes[data_divider * j].legend()

                                    axes[data_divider * j].set_ylabel(keypoints[j])

                                axes[0].set_title('X')
                                axes[1].set_title('Y')
                                if data_divider >= 3:
                                    axes[2].set_title('Z')

                                return

                            plot_save(make_xyz_plot,
                                          title=f'reconstruction_xyz_{data_dict["indices_file"][i]}-{data_dict["indices_pos"][i]}',
                                          only_png=True,
                                          outputdir=plot_dir)

                            n_plots += 1

            logger.info(f'{subset}, dataset_path = {dataset_path}')

            if dataset.y is None:
                np.savez(os.path.join(dataset_path, f'{subset}_fulllength_dataset_imputed.npz'),
                         X=dataset.X, time=dataset.time)

            else:
                np.savez(os.path.join(dataset_path, f'{subset}_fulllength_dataset_imputed.npz'),
                         X=dataset.X, y=dataset.y, time=dataset.time)

            if dataset.files is not None:
                for i_f, f in enumerate(dataset.files):
                    save_data_original_format(dataset.X[i_f], dataset.time[i_f],
                                              f, file_type,
                                              keypoints, orig_freq, subsampling_freq,
                                              data_divider,
                                              impute_dir, logger)


            # saving new chunked dataset
            new_dataset = []
            new_lengths = []
            new_y = []
            for i_recording in range(dataset.X.shape[0]):
                mask_t = dataset.time[i_recording] > -1
                logger.debug(f'LINE 412 in IMPUTE_DATASET - shape: {mask_t.shape} {dataset.X[i_recording].shape} {dataset.X[i_recording][mask_t].shape}')
                x = dataset.X[i_recording]
                x = x.reshape(mask_t.shape[0], len(keypoints), -1) # should be of shape (timepoints, keypoints, 2 or 3)
                data, len_, t_ = chop_coordinates_in_timeseries(dataset.time[i_recording][mask_t],
                                                                x[mask_t],
                                                                stride=stride,
                                                                length=seq_length)

                if len(data) > 0:
                    new_dataset.extend(data)
                    new_lengths.extend(len_)
                    if dataset.y is not None:
                        new_y.extend(
                            np.hstack([np.tile(dataset.y[i_recording], len(len_)).reshape(
                                (len(len_), dataset.y.shape[-1])), np.expand_dims(t_ / subsampling_freq, 1)]))

            logger.debug(f'New dataset {subset} has shape {np.stack(new_dataset, axis=0).shape}.')
            if dataset.y is None:
                np.savez(os.path.join(dataset_path, f'{subset}_dataset_imputed.npz'), X=np.stack(new_dataset, axis=0),
                         lengths=np.stack(new_lengths))
            else:
                np.savez(os.path.join(dataset_path, f'{subset}_dataset_imputed.npz'), X=np.stack(new_dataset, axis=0),
                         y=np.stack(new_y), lengths=np.stack(new_lengths))

