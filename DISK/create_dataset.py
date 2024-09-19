import os
import tqdm
import shutil
import numpy as np
import scipy.io
import pandas as pd
import pickle
import h5py

import logging
import hydra
from omegaconf import DictConfig


def chop_coordinates_in_timeseries(time_vect: np.array,
                                   coordinates: np.array,
                                   stride: int = 1,
                                   length: int = 1,):
    """

    :param time_vect: 1D numpy array
    :param coordinates: 3D numpy array (timepoints, keypoints, 3)
    Should be in kw_cfg:
    :param length: in timepoints
    :param stride: in timepoints
    :param 
    :return:
    """

    breakpoints = np.where(np.diff(list(time_vect)) > 1)[0]
    breakpoints = np.insert(breakpoints, 0, 0)  # add first point = index 0
    breakpoints = np.insert(breakpoints, len(breakpoints),
                            len(time_vect))  # add last point = index len of the vector
    good_segments = np.where(np.diff(breakpoints) >= length)[0]  # is the segment longer than our lower bound
    dataset = []
    lengths = []
    times = []
    if len(good_segments) == 0:
        logging.debug('No long enough segments.')
    for index_good_segment in good_segments:
        data = coordinates[breakpoints[index_good_segment] + 1: breakpoints[index_good_segment + 1]]

        i = 0
        while len(data) - i * stride > length:
            subdata = data[int(i * stride): int(i * stride) + length, ...]
            times.append(time_vect[breakpoints[index_good_segment] + 1 + int(i * stride)])
            lengths.append(length)
            dataset.append(subdata.reshape(length, -1))
            i += 1
    dataset = np.array(dataset)
    lengths = np.array(lengths)
    times = np.array(times)
    return dataset, lengths, times


def find_hole_nan(mask):
    # data shape (time, columns)
    out = []
    for ic in range(mask.shape[1]):
        start = 0
        while start < mask.shape[0]:
            if not np.any(np.isnan(mask[start:, ic])):
                break
            index_start_nan = np.where(np.isnan(mask[start:, ic]))
            start += index_start_nan[0][0]
            if np.any(~(np.isnan(mask[start:, ic]))):
                length_nan = np.where(~(np.isnan(mask[start:, ic])))[0][0]
            else:
                # the nans go until the end of the vector
                length_nan = mask.shape[0] - start
            out.append((start, length_nan, ic))
            start = start + length_nan
    return out  # returns a list of tuples (start, length_nan, keypoint_name)


def open_and_extract_data(f, file_type, dlc_likelihood_threshold):
    """
    :args f: (str) path to data file

    Supported formats:
       - .mat from QUALISYS software
       - .csv format with 3 columns per keypoints with names {kp}_x, {kp}_y, {kp}_z

    If keypoints not found in file, then they will be named '0', '1', '2', ...

    :return data: numpy array of shape (timesteps, n_keypoints, 3)
    :return keypoints: list of keypoint names as strings
    """
    if file_type == 'mat_dannce':
        mat = scipy.io.loadmat(f)
        # for Rat7M dataset
        data = np.moveaxis(np.array(list(mat['mocap'][0][0])), 1, 0)
        keypoints = list(mat['mocap'][0][0].dtype.fields.keys())

    elif file_type == 'mat_qualisys':
        # for in house mouse data, QUALISYS software
        mat = scipy.io.loadmat(f)
        exp_name = [m for m in mat.keys() if m[:2] != '__'][0]  ## TOCHANGE
        data = np.moveaxis(mat[exp_name][0, 0]['Trajectories'][0, 0]['Labeled']['Data'][0, 0],
                           2, 0)
        keypoints = [label[0].replace('coordinate', 'coord') for label in
                     mat[exp_name][0, 0]['Trajectories'][0, 0]['Labeled']['Labels'][0, 0][0]]

        # very important
        # make sure the keypoints are always in the same order even if not saved so in the original files
        new_order = np.argsort(keypoints)
        keypoints = [keypoints[n] for n in new_order]
        data = data[:, new_order, :]

    elif file_type == 'simple_csv':
        ## for fish data from Liam
        df = pd.read_csv(f)  # columns time, keypoint_x, kp_y, kp_z
        # sort the keypoints with np.unique
        keypoints = list(np.unique([c.rstrip('_xyz') for c in df.columns if c.endswith('_x') or c.endswith('_y') or c.endswith('_z')]))
        columns = []
        for k in keypoints:
            columns.extend([k + '_x', k + '_y', k + '_z'])
        # get the columns corresponding to sorted keypoints so the data can be stacked
        data = df.loc[:, columns].values.reshape((df.shape[0], len(keypoints), -1))

    elif file_type == 'npy':
        ## for human MoCap files
        data = np.array(np.load(f))
        logging.info(f'[WARNING][CREATE_DATASET][OPEN_AND_EXTRACT_DATA function][NPY INPUT FILES] keypoints cannot be loaded from input files. '
                     f'Expected behavior: the columns correspond to the keypoints and are in fixed order')
        # WARNING - here no information about keypoint, so we expect that the columns match for every file
        keypoints = [f'{i:02d}' for i in range(data.shape[1])]

    elif file_type == 'df3d_pkl':
        ## for DeepFly data
        with open(f, 'rb') as openedf:
            pkl_content = pickle.load(openedf)
        data = pkl_content['points3d']
        logging.info(f'[WARNING][CREATE_DATASET][OPEN_AND_EXTRACT_DATA function][PKL INPUT FILES] keypoints cannot be loaded from input files. '
                     f'Expected behavior: the columns correspond to the keypoints and are in fixed order')
        keypoints = [f'{i:02d}' for i in range(data.shape[1])]
        """ from DeepFly3D paper
        38 landmarks per animal: (i) five on each limb – the thorax-coxa, coxa-femur, femur-tibia, and tibia-tarsus 
        joints as well as the pretarsus, (ii) six on the abdomen - three on each side, and (iii) one on each antenna
         - for measuring head rotations.
         see image on github too
        """

    elif file_type == 'dlc_csv':
        ## for csv from DeepLabCut
        df = pd.read_csv(f, header=[1, 2])
        keypoints = [bp for bp in df.columns.levels[0] if bp != 'bodyparts']
        keypoints.sort()
        coordinates = [c for c in df.columns.levels[1] if c != 'likelihood' and c != 'coords']
        likelihood_columns = []
        for k in keypoints:
            likelihood_columns.append((k, 'likelihood'))

        columns = []
        for k in keypoints:
            for c in coordinates:
                df.loc[df.loc[:, (k, 'likelihood')] <= dlc_likelihood_threshold, (k, c)] = np.nan
                columns.append((k, c))
        data = df.loc[:, columns].values.reshape((df.shape[0], len(keypoints), -1))

    elif file_type == 'dlc_h5':
        content = h5py.File(f)
        extracted_content = np.vstack([c[1] for c in content['df_with_missing']['table'][:]])
        mask_columns_likelihood = np.all((extracted_content <= 1) * (extracted_content >= 0), axis=0)
        likelihood_columns = extracted_content[:, mask_columns_likelihood] <= dlc_likelihood_threshold
        coordinates_columns = extracted_content[:, ~mask_columns_likelihood]
        n_dim = coordinates_columns.shape[1] / likelihood_columns.shape[1]
        assert int(n_dim) == n_dim
        n_dim = int(n_dim)
        for i_dim in range(n_dim):
            coordinates_columns[:, i_dim::n_dim][likelihood_columns] = np.nan
        data = coordinates_columns.reshape((coordinates_columns.shape[0], -1, n_dim))
        keypoints = [f'{i:02d}' for i in range(data.shape[1])]

    elif file_type == 'sleap_h5':
        ## compatibility with SLEAP analysis h5 files
        with h5py.File(f, 'r') as openedf:
            data = openedf['tracks'][:].T
            keypoints = [n.decode() for n in openedf["node_names"][:]]

        if data.shape[3] > 1:
            # multi-animal scenario
            new_keypoints = []
            for animal_id in range(data.shape[3]):
                new_keypoints.extend([f'animal{animal_id}_{k}' for k in keypoints])
            keypoints = new_keypoints
            data = np.moveaxis(data, 3, 1).reshape(data.shape[0], -1, data.shape[2])
        else:
            # one animal, remove the last axis
            data = data[..., 0]

        # very important
        # make sure the keypoints are always in the same order even if not saved so in the original files
        new_order = np.argsort(keypoints)
        keypoints = [keypoints[n] for n in new_order]
        data = data[:, new_order]

    else:
        raise ValueError(f'File format not understood {f}, should be one of the following: mat_dannce, '
                         f'mat_qualisys,simple_csv, dlc_csv, npy, df3d_pkl, sleap_h5')

    # we replace the spaces by underscore because when dealing with set of keypoints we separate them by spaces (fake and orginal holes)
    keypoints = [k.replace(' ', '_') for k in keypoints]
    return data, keypoints


@hydra.main(version_base=None, config_path="conf", config_name="conf_create_dataset")
def create_dataset(_cfg: DictConfig) -> None:
    basedir = hydra.utils.get_original_cwd()
    logging.info(f'[BASEDIR] {basedir}')
    """ LOGGING AND PATHS """

    logging.info(f'{_cfg}')
    outputdir = os.path.join(basedir, 'datasets', _cfg.dataset_name)

    constant_file_path = os.path.join(outputdir, 'constants.py')

    if not os.path.isdir(outputdir):
        os.mkdir(outputdir)

    #################################################################################################
    ### OPEN FILES AND PROCESS DATA
    #################################################################################################
    p_train_val_test = np.array([0, 0.7, 0.85, 1])
    nan_modalities = [0, 1, np.inf]
    nan_modalities_names = ['0', '1', 'all']
    partitions = ['train', 'val', 'test']

    dataset = {}
    data_lengths = {}

    fulllength_data = {}
    fulllength_time = {}
    fulllength_maxlength = {}
    fulllength_original_files = {}

    for n in nan_modalities_names:
        for t in partitions:
            dataset[(n, t)] = []
            data_lengths[(n, t)] = []
            fulllength_data[(n, t)] = []
            fulllength_original_files[(n, t)] = []
            fulllength_time[(n, t)] = []
            fulllength_maxlength[(n, t)] = []


    # and we want to count the number of effective files
    for i_file, f in tqdm.tqdm(enumerate(_cfg.input_files)):

        # one partition for one file, even if some will be missing because of too many nans
        # TODO: test it with low number of files plot
        if i_file % 10 == 1:
            partition = 'val'
        elif i_file % 10 == 2:
            partition = 'test'
        else:
            partition = 'train'

        data, keypoints = open_and_extract_data(f, _cfg.file_type, _cfg.dlc_likelihood_threshold)

        # shape (keypoints, coordinates + residual, timepoints)
        data = data[_cfg.discard_beginning * _cfg.original_freq: _cfg.discard_end * _cfg.original_freq, :, :3]

        if _cfg.drop_keypoints is not None:
            try:
                indices = []
                for k in _cfg.drop_keypoints:
                    if type(k) == str:
                        if k in keypoints:
                            indices.append(keypoints.index(k))
                    else:
                        if f'{k:02d}' in keypoints:
                            indices.append(keypoints.index(f'{k:02d}'))
            except ValueError:
                logging.error(f'keypoints to drop {_cfg.drop_keypoints} not found in {f} with keypoints {keypoints}')
                raise ValueError
            other_indices = np.array([i for i in range(len(keypoints)) if not i in indices])
            data = data[:, other_indices]
            logging.debug(f'After removing keypoints, data shape {data.shape} from file {f}')
            keypoints = [keypoints[i] for i in other_indices]

        if i_file == 0:
            logging.info(f'Found keypoints:  {keypoints}')
            old_keypoints = list(keypoints)
        else:
            if len(set(old_keypoints).symmetric_difference(set(keypoints))) > 0:
                print(f'[WARNING][CREATE_DATASET] Mismatch between keypoints: \n'
                                 f'Found {old_keypoints} in file {_cfg.input_files[0]} \n'
                                 f'and {keypoints} in file {f}\n')
                continue

        # step 1. interpolation
        if _cfg.fill_gap > 0:
            flattened_data = data.reshape(data.shape[0], -1)
            out = find_hole_nan(flattened_data)
            for start, length, index_column in out:
                if length > _cfg.fill_gap:
                    # gap too long, skip
                    continue
                if start == 0 or start + length >= data.shape[0]:
                    # no interpolation can be made, skip
                    continue
                interp = np.linspace(flattened_data[start - 1, index_column],
                                     flattened_data[start + length, index_column], length)
                flattened_data[start: start + length, index_column] = interp
            data = flattened_data.reshape(data.shape)

        # step 2. subsampling
        if _cfg.subsampling_freq < _cfg.original_freq:
            # the reshape creates an additional dimenesion to be averaged by the nanmean with axis = 1
            data = np.nanmean(
                data[:int(len(data) / (_cfg.original_freq / _cfg.subsampling_freq)) * int(_cfg.original_freq / _cfg.subsampling_freq)] \
                    .reshape((-1, int(_cfg.original_freq / _cfg.subsampling_freq), data.shape[1], data.shape[2])), axis=1)

        # until now we have eventually filled the gaps with linear interpolation and resampled
        # but we haven't removed the lines with nan, so we can assume the corresponding time vector is simply this:
        time_vect = np.arange(data.shape[0])

        # step 3. remove remaining nans
        nb_nans_per_timestep = np.sum(np.isnan(data[..., 0]), axis=1)

        # logging.info(f'Timepoint with at least one missing: {np.sum(nb_nans_per_timestep > 0)} / {len(data)}')

        for nb_allowed_nans, nan_name in zip(nan_modalities, nan_modalities_names):

            # if in subset_columns there are some keypoints then one keypoint's coordinates are spread in 3 columns
            mask_rows = nb_nans_per_timestep <= nb_allowed_nans
            new_data = data[mask_rows]
            new_time_vect = time_vect[mask_rows]

            if _cfg.sequential:
                total_len = len(new_data)
                indices_ttv = (total_len * np.array(p_train_val_test)).astype('int')

                for i_partition, partition in enumerate(partitions):
                    # chopped_data has shape (n_samples, times, keypoints * 3D)
                    chopped_data, len_, times = chop_coordinates_in_timeseries(new_time_vect[indices_ttv[i_partition]: indices_ttv[i_partition + 1]],
                                                                               new_data[indices_ttv[i_partition]: indices_ttv[i_partition + 1]],
                                                                               length=_cfg.length,
                                                                               stride=_cfg.stride)

                    # NB: times gives the beginning of the sample in the raw indices
                    if len(chopped_data) > 0:
                        dataset[(nan_name, partition)].extend(chopped_data)
                        data_lengths[(nan_name, partition)].extend(len_)
                    crop_len = indices_ttv[i_partition + 1] - indices_ttv[i_partition]
                    if crop_len > 0:
                        fulllength_data[(nan_name, partition)].append(new_data[indices_ttv[i_partition]: indices_ttv[i_partition + 1]].reshape(crop_len, -1))
                        fulllength_time[(nan_name, partition)].append(new_time_vect[indices_ttv[i_partition]: indices_ttv[i_partition + 1]] / _cfg.subsampling_freq)
                        fulllength_maxlength[(nan_name, partition)].append(crop_len)
                        fulllength_original_files[(nan_name, partition)].append(os.path.basename(f))

                    i_file += 1
            else:
                # chopped_data has shape (n_samples, times, keypoints * 3D)
                chopped_data, len_, times = chop_coordinates_in_timeseries(new_time_vect,
                                                                           new_data,
                                                                           length=_cfg.length,
                                                                           stride=_cfg.stride)


                # NB: times gives the beginning of the sample in the raw indices
                if len(chopped_data) == 0:
                    logging.info(f'[WARNING] file {i_file} has not long enough segments for {nan_name}')
                    continue
                dataset[(nan_name, partition)].extend(chopped_data)
                data_lengths[(nan_name, partition)].extend(len_)

                fulllength_data[(nan_name, partition)].append(new_data.reshape(new_data.shape[0], -1))
                fulllength_time[(nan_name, partition)].append(new_time_vect / _cfg.subsampling_freq)
                fulllength_maxlength[(nan_name, partition)].append(new_data.shape[0])
                fulllength_original_files[(nan_name, partition)].append(os.path.basename(f))

    ####################################################################################################
    ###### END FOR LOOP ON THE FILES ######
    ####################################################################################################

    for nb_allowed_nans, nan_name in zip(nan_modalities, nan_modalities_names):

        for i_ttv, partition in enumerate(partitions):
            if len(dataset[(nan_name, partition)]) == 0:
                raise ValueError(f'no data for {partition} with {nan_name} NaNs, probably due to not long enough '
                                 f'segments')

            subdata = np.stack(dataset[(nan_name, partition)], axis=0)
            sublengths = np.array(data_lengths[(nan_name, partition)])

            sub_fulllength_data = np.ones(
                (len(fulllength_maxlength[(nan_name, partition)]), max(fulllength_maxlength[(nan_name, partition)]), subdata.shape[-1])) * (-1)
            for i_length, length in enumerate(fulllength_maxlength[(nan_name, partition)]):
                sub_fulllength_data[i_length, :length] = fulllength_data[(nan_name, partition)][i_length]
            # need to pad the fulllength_time with -1
            sub_fulllength_time = np.ones((len(fulllength_maxlength[(nan_name, partition)]), max(fulllength_maxlength[(nan_name, partition)]))) * (-1)
            for i_length, length in enumerate(fulllength_maxlength[(nan_name, partition)]):
                sub_fulllength_time[i_length, :length] = fulllength_time[(nan_name, partition)][i_length]

            logging.info(f'In {partition} with {nan_name} NaNs, Shape: {subdata.shape}')
            outputfile = os.path.join(outputdir, f'{partition}_dataset_w-{nan_name}-nans')
            print(f'saving in {outputfile}...')
            np.savez(outputfile, X=subdata, lengths=sublengths)

            outputfile = os.path.join(outputdir, f'{partition}_fulllength_dataset_w-{nan_name}-nans')
            print(f'saving in {outputfile}...')
            np.savez(outputfile, X=sub_fulllength_data, time=sub_fulllength_time,
                     files=np.array(fulllength_original_files[(nan_name, partition)]))

            ### WRITE THE CONSTANTS.PY FILE CORRESPONDING TO THIS NEW DATASET
            if i_ttv == 0:
                with open(constant_file_path, 'w') as opened_file:
                    txt = f"NUM_FEATURES = {subdata.shape[2]}\n"
                    txt += f"KEYPOINTS = {keypoints}\n"
                    # DIVIDER= 2 for 2D, 3 for 3D, sometimes additional dimension for a confidence score or an error
                    # score for the detection
                    txt += f"DIVIDER = {data.shape[-1]}\n"
                    txt += f"ORIG_FREQ = {_cfg.original_freq}\n"
                    txt += f"FREQ = {_cfg.subsampling_freq}\n"
                    txt += f"SEQ_LENGTH = {subdata.shape[1]}\n"
                    txt += f"STRIDE = {_cfg.stride}\n"
                    txt += f"W_RESIDUALS = False\n"  # for compatibility reasons (see dataset classes)
                    txt += f"FILE_TYPE = '{_cfg.file_type}'\n"
                    txt += f"DLC_LIKELIHOOD_THRESHOLD = {_cfg.dlc_likelihood_threshold}"
                    opened_file.write(txt)


    create_skeleton_file = input('Would you like to create a skeleton file [y/n]? \n'
          '(If this is the first time creating a dataset for a specific data, then type y. \n'
          'If a skeleton file has already been generated for this type of data (animal + recording type), then type n. ')

    possible_colors = ['orange', 'gold', 'grey', 'cornflowerblue', 'turquoise', 'hotpink', 'purple', 'blue', 'seagreen',
                       'darkolivegreen', ]

    if create_skeleton_file.lower() == 'y': ## answer is yes, create a skeleton file
        print('The keypoints are:')
        [print(f'{"":>11}{i} - {keypoints[i]}') for i in range(len(keypoints))]
        print('Please indicate the links between keypoints (if possible in groups of links,\n'
              'e.g. a leg, or the spine - groups of links will be displayed in the same color. ')
        neighbor_links = []
        link_colors = []
        i = 0
        while True:
            groups_of_neighbors = input("Indicate the first group using the keypoints' indices and "
                                        "follow the convention (0, 2), (0, 6), (2, 4) or (0, 2) \n"
                                        "for only one link in a group. "
                                        "Just press <Enter> if no more links. ")
            if groups_of_neighbors == '':
                break
            group = eval(groups_of_neighbors)
            neighbor_links.append(group)
            link_colors.append(possible_colors[i % len(possible_colors)])
            i += 1

        center = None
        while center is None:
            center_index = input("Indicate which keypoint index is closer to the center of mass of the animal. "
                                 "Please pick only one index. Should be an integer. ")
            try:
                center = int(center_index)
            except NameError:
                print('Wrong input')


        ## Now right the skeleton file
        skeleton_file_path = os.path.join(outputdir, 'skeleton.py')
        with open(skeleton_file_path, 'w') as opened_file:
            txt = f"num_keypoints = {len(keypoints)}\n"
            txt += f"keypoints = {keypoints}\n"
            # DIVIDER= 2 for 2D, 3 for 3D, sometimes additional dimension for a confidence score or an error
            # score for the detection
            txt += f"center = {center}\n"
            txt += f"original_directory = '{outputdir}'\n"
            txt += f"neighbor_links = {neighbor_links}\n"
            txt += f"link_colors = {link_colors}\n"

            opened_file.write(txt)

    # we copy the config file, because it might be overwritten by the create_proba_missing config files
    # and we need the original one for imputation later
    shutil.copy(os.path.join('.hydra', 'config.yaml'), os.path.join('.hydra', 'config_create_dataset.yaml'))

    print('______ End of create_dataset ______')


if __name__ == '__main__':

    logging.basicConfig(level=logging.DEBUG,
                        format=f'[%(levelname)s][%(asctime)s] %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S',
                        )
    logger = logging.getLogger(__name__)

    create_dataset()
