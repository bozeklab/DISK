import os
import sys
import tqdm
import logging
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm

from DISK.utils.dataset_utils import load_datasets
from DISK.utils.utils import read_constant_file, plot_save, find_holes

if __name__ == '__main__':
    ###############################################################################################
    ### Explore the proportion of missing keypoints and their distribution of a dataset via plots
    ###############################################################################################

    dataset_name = 'Fish_v3_60stride60'
    stride = 60
    seq_length = 60 ## for averaging
    suffix_dataset = '_w-all-nans'
    basedir = '/home/france/Mounted_dir/'

    save_bool = True
    outputdir = os.path.join(basedir, 'results_behavior/datasets', dataset_name)
    if not os.path.exists(outputdir):
        print(f'[ERROR] dataset folder {outputdir} not found')
        raise ValueError

    dataset_constants = read_constant_file(os.path.join(os.path.join(basedir, 'results_behavior/datasets', dataset_name,
                                                                     f'constants.py')))

    """ LOGGING """
    logging.basicConfig(level=logging.INFO,
                        format=f'[%(levelname)s][%(asctime)s] %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("numpy").setLevel(logging.WARNING)
    sns.set_style("white")
    transforms = None
    train_dataset, val_dataset, test_dataset = load_datasets(dataset_name=dataset_name,
                                                             dataset_constants=dataset_constants,
                                                             transform=transforms,
                                                             dataset_type='full_length',
                                                             root_path=os.path.join(os.path.join(basedir, 'results_behavior')),
                                                             suffix=suffix_dataset,
                                                             verbose=0)

    original_divider = dataset_constants.DIVIDER + 1 if dataset_constants.W_RESIDUALS else dataset_constants.DIVIDER
    ## counting the nans = keypoint missing per timepoints
    ## (first dimension is the files, second is the timesteps inside each file)
    X = np.sum(np.isnan(train_dataset.X[..., ::original_divider]), axis=2)
    max_columns = max(1000, int(np.sqrt(X.shape[0] * X.shape[1])))

    percentage_missing = np.mean([np.sum(v) / np.sum(train_dataset.time[i] != -1) for i, v in enumerate(X > 0)]) * 100
    print(f'Percentage of timepoints with at least one missing keypoint: {percentage_missing:.02f} %')

    ## same thing but with
    max_columns = max(1000, int(np.sqrt(X.shape[0] * X.shape[1] / seq_length)))
    if X.shape[1] >= max_columns * seq_length:
        ## just a reshape
        X_new = []
        for x in X:
            subx = np.nanmean(x[:int(x.shape[0] / seq_length) * seq_length].reshape(-1, seq_length), axis=1)
            sup_ = int(np.ceil(subx.shape[0] / max_columns))
            inf_ = int(np.floor(subx.shape[0] / max_columns))
            x_new = np.zeros((sup_ + 1, max_columns))
            x_new[:inf_] = subx[:inf_ * max_columns].reshape(-1, max_columns)
            x_new[-1, :subx.shape[0] - inf_ * max_columns] = subx[inf_ * max_columns:].flatten()
            X_new.append(x_new)
        X_new = np.vstack(X_new)
    else:
        X_new = []
        for x in X:
            subx = np.nanmean(x[:int(x.shape[0] / seq_length) * seq_length].reshape(-1, seq_length), axis=1)
            X_new.append(subx)
        X_new = np.vstack(X_new)

    def mat_number_missing():
        sns.set_style("white")
        cmap = cm.get_cmap('Greys_r', dataset_constants.N_KEYPOINTS)
        cmap.set_under(color='black')  # set no missing to black (this line + vmin=0.5)
        plt.figure(figsize=(15, 9))
        plt.imshow(X_new, cmap=cmap, vmin=0.5)
        plt.colorbar()
        plt.tight_layout()

    plot_save(mat_number_missing, save_bool, title=f'averaged_mat_num_missing_{dataset_name}{suffix_dataset}',
              only_png=False,
              outputdir=outputdir)

    max_columns = max(1000, int(np.sqrt(X.shape[0] * X.shape[1])))

    lines_separating_files = np.cumsum([int(np.ceil(x.shape[0] / max_columns)) + 1 for x in X])
    if X.shape[1] >= max_columns:
        ## just a reshape
        X_new = []
        for x in X:
            sup_ = int(np.ceil(x.shape[0] / max_columns))
            inf_ = int(np.floor(x.shape[0] / max_columns))
            x_new = np.zeros((sup_ + 1, max_columns))
            x_new[:inf_] = x[:inf_ * max_columns].reshape(-1, max_columns)
            x_new[-1, :x.shape[0] - inf_ * max_columns] = x[inf_ * max_columns:].flatten()
            X_new.append(x_new)
        X_new = np.vstack(X_new)
    else:
        X_new = X


    plot_save(mat_number_missing, save_bool, title=f'mat_num_missing_{dataset_name}{suffix_dataset}',
              only_png=False,
              outputdir=outputdir)


    sys.exit(0)

    ## counts the missing parts independently for each keypoint
    n_missing_df = []
    for x in X:
        for i in range(1, dataset_constants.N_KEYPOINTS + 1):
            out = find_holes(x.reshape(x.shape[0], 1), ['all'], target_val=i)
            tmp = pd.DataFrame(columns=['n missing keypoints', 'length'],
                               data=np.vstack([[i] * len(out), [o[1] for o in out]]).T)
            n_missing_df.append(tmp)
    n_missing_df = pd.concat(n_missing_df).reset_index()

    def hist_n_missing():
        sns.histplot(data=n_missing_df, x='n missing keypoints', y='length', discrete=True)
        m_ = max(n_missing_df.loc[:, 'length']) * 1.1
        plt.ylim(0, m_)
        plt.xlim(0, dataset_constants.N_KEYPOINTS + 1)
        for v in n_missing_df.groupby('n missing keypoints').count().reset_index().iterrows():
            plt.text(v[1]['n missing keypoints'] - 0.1,
                     max(n_missing_df.loc[n_missing_df['n missing keypoints'] == v[1]['n missing keypoints'], 'length']) + 0.05 * m_,
                     int(v[1]['length']))

    plot_save(hist_n_missing, save_bool, title=f'hist_length_per_n_missing_keypoint_{dataset_name}{suffix_dataset}',
              only_png=False,
              outputdir=outputdir)

    df = pd.DataFrame(columns=['index_sample', 'length', 'keypoint'])
    i_data = 0
    for x in tqdm.tqdm(train_dataset.X, desc="Collecting single missing values"):
        mask_holes = np.isnan(x)
        if np.any(mask_holes):
            out = find_holes(mask_holes, dataset_constants.KEYPOINTS, target_val=True)
            original = True
            for (start, length_nan, keypoint_name) in out:
                df.loc[df.shape[0], :] = [i_data, int(length_nan), keypoint_name]
            out = find_holes(np.sum(mask_holes, axis=1).reshape(mask_holes.shape[0], 1), ['non_missing'], target_val=0)
            for (start, length_nan, name) in out:
                df.loc[df.shape[0], :] = [i_data, int(length_nan), name]
            i_data += 1

    print(df)
    df = df.convert_dtypes()

    if df.shape[0] > 1000:
        sub_df = df.sample(n=1000)
    else:
        sub_df = df

    def hist_length_per_keypoint():
        keypoints = sub_df.loc[(sub_df['keypoint'] != 'non_missing'), 'keypoint'].unique()
        fig, axes = plt.subplots(int(np.ceil((len(keypoints) + 1) / 2)), 2, figsize=(10, 10))
        axes = axes.flatten()
        bins = np.arange(0, np.percentile(df.loc[df['keypoint'].isin(keypoints), 'length'], 99)) - 0.5
        for ikp, kp in enumerate(keypoints):
            axes[ikp].hist(sub_df.loc[(sub_df['keypoint'] == kp) & (sub_df['keypoint'] != 'non_missing'), 'length'], bins=bins,
                           density=True)
            axes[ikp].set_title(kp)
        axes[ikp + 1].hist(sub_df.loc[(sub_df['keypoint'].isin(keypoints)), 'length'], bins=bins,
                       density=True)
        axes[ikp + 1].set_title('all')
        plt.tight_layout()


    plot_save(hist_length_per_keypoint, save_bool, title=f'hist_length_per_keypoint_{dataset_name}{suffix_dataset}',
              only_png=False,
              outputdir=outputdir)


    def count_vs_keypoint():
        pivot_df = sub_df[sub_df['keypoint'] != 'non_missing'].groupby(['keypoint'])['index_sample'].agg('count').reset_index().rename(
            {'index_sample': 'count'}, axis=1)
        pivot_df.loc[:, 'count'] /= pivot_df.loc[:, 'count'].sum()
        plt.figure()
        sns.barplot(data=pivot_df, y='count', x='keypoint')
        plt.tight_layout()

    plot_save(count_vs_keypoint, save_bool, title=f'count_vs_keypoint_{dataset_name}{suffix_dataset}', only_png=False,
              outputdir=outputdir)


    ## look at the interaction between missing keypoints
    file_id, time_id, kp_id = np.where(np.isnan(train_dataset.X[..., ::3]))
    set_kp = []
    kp_per_time = []
    for f in np.unique(file_id):
        real_length_file = int(np.sum(train_dataset.time[f] != -1))
        kp_per_time.append([''] * real_length_file)
        for t in tqdm.tqdm(np.unique(time_id[time_id <= real_length_file]),
                           desc=f'Looking for sets of missing keypoints in file #{f}'):
            tmp = ''.join([str(v) for v in np.sort(kp_id[(file_id == f) * (time_id == t)])])
            set_kp.append(tmp)
            kp_per_time[-1][t] = tmp
    sets, counts = np.unique(set_kp, return_counts=True)

    df_sets = []
    for f in np.unique(file_id):
        for s in sets:
            out = find_holes(np.array(kp_per_time[f]).reshape(-1, 1) == s, [s], True)
            tmp_df = pd.DataFrame(columns=['start', 'length', 'set_kp'], data=np.array(out))
            tmp_df.loc[:, 'index_sample'] = f
            df_sets.append(tmp_df)
    df_sets = pd.concat(df_sets)
    df_sets.loc[:, 'start'] = df_sets['start'].astype(int)
    df_sets.loc[:, 'length'] = df_sets['length'].astype(int)

    ## count only starts
    count_starts = df_sets.groupby('set_kp')['length'].count()
    ## count all missing times
    total_length_per_set = df_sets.groupby('set_kp')['length'].sum()

    def count_vs_setkp():
        plt.figure()
        plt.bar(x=np.arange(len(sets)), height=counts)
        plt.xticks(np.arange(len(sets)), sets, rotation=90)
        plt.tight_layout()

    plot_save(count_vs_setkp, save_bool, title=f'total_count_vs_keypoint_{dataset_name}{suffix_dataset}', only_png=False,
              outputdir=outputdir)

    def starting_count_vs_setkp():
        plt.figure()
        plt.bar(x=np.arange(len(sets)), height=count_starts.values)
        plt.xticks(np.arange(len(sets)), count_starts.index, rotation=90)
        plt.tight_layout()

    plot_save(starting_count_vs_setkp, save_bool, title=f'starting_site_count_vs_keypoint_{dataset_name}{suffix_dataset}', only_png=False,
              outputdir=outputdir)

    def hist_length_per_setkp():
        keypoints = df_sets.loc[:, 'set_kp'].unique()
        fig, axes = plt.subplots(int(np.ceil((len(keypoints) + 1) / 4)), 4, figsize=(10, 10))
        axes = axes.flatten()
        bins = np.arange(0, np.percentile(df_sets.loc[df_sets['set_kp'].isin(set_kp), 'length'], 99)) - 0.5
        for ikp, kp in enumerate(keypoints):
            axes[ikp].hist(df_sets.loc[(df_sets['set_kp'] == kp), 'length'], bins=bins,
                           density=True)
            axes[ikp].set_title(kp)
        axes[ikp + 1].hist(df_sets.loc[(df_sets['set_kp'].isin(keypoints)), 'length'], bins=bins,
                       density=True)
        axes[ikp + 1].set_title('all')

    plot_save(hist_length_per_setkp, save_bool, title=f'hist_length_per_keypoint_{dataset_name}{suffix_dataset}',
              only_png=False,
              outputdir=outputdir)
    print('ouf')