import os, sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from glob import glob
import logging
import seaborn as sns
import torch
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig

from DISK.utils.dataset_utils import load_datasets
from DISK.utils.utils import read_constant_file, plot_save, find_holes
from DISK.utils.transforms import AddMissing_LengthProba


def create_uniform_proba(min_len, max_len, keypoints):
    assert 0 < min_len < max_len
    lengths = np.arange(min_len, max_len).astype('int')
    proba = np.ones(lengths.shape[0], dtype='float') / len(lengths)
    dfs = []
    for k in keypoints + ['non_missing']:
        dfs.append(pd.DataFrame(columns=['original', 'keypoint', 'length', 'proba'],
                          data=np.vstack([[True] * len(lengths), [k] * len(lengths), lengths, proba]).T))
    df_proba_init = pd.DataFrame(columns=['keypoint', 'proba'],
                                 data=np.vstack([keypoints + ['non_missing'], [1 / len(keypoints)] * len(keypoints) + [0]]).T)
    return pd.concat(dfs).reset_index().drop('index', axis=1), df_proba_init


@hydra.main(version_base=None, config_path="conf", config_name="conf_proba_missing_files")
def create_proba_missing_files(_cfg: DictConfig) -> None:
    """Check if the artificial missing coordinates match the original coordinates"""
    basedir = hydra.utils.get_original_cwd()
    logging.info(f'[BASEDIR] {basedir}')
    """ LOGGING AND PATHS """

    logging.info(f'{_cfg}')
    outputdir = os.path.join(basedir, 'datasets', _cfg.dataset_name,)
    if not os.path.exists(outputdir):
        os.mkdir(outputdir)

    constant_file_path = os.path.join(outputdir, 'constants.py')
    if not os.path.exists(constant_file_path):
        raise ValueError(f'no constant file found in', constant_file_path)
    dataset_constants = read_constant_file(constant_file_path)

    suffix = f'_set_keypoints' if not _cfg.indep_keypoints else ''
    if _cfg.indep_keypoints:
        if _cfg.merge_keypoints:
            logging.info(f'merge_keypoints = True is not a valid option when indep_keypoints = True. '
                         f'merge_keypoints would be considered False')
            suffix += f'_merged'
    no_original_missing = False

    for initial in [True, False]:

        if not initial:
            length_proba_df = pd.read_csv(os.path.join(outputdir, f'proba_missing_length{suffix}.csv'))
            init_proba = pd.read_csv(os.path.join(outputdir, f'proba_missing{suffix}.csv'))
            # n_proba = pd.read_csv('/home/france/Mounted_dir/results_behavior/datasets/proba_n_missing_1_6.txt',
            #                       header=None)

            addmissing_transform = AddMissing_LengthProba(length_proba_df, dataset_constants.KEYPOINTS, init_proba,
                                                          proba_n_missing=None,
                                                          divider=dataset_constants.DIVIDER,
                                                          indep_keypoints=_cfg.indep_keypoints,
                                                          pad=(0, 0), verbose=0, proba=1)
            transform = [addmissing_transform]
        else:
            transform = None

        train_dataset, val_dataset, test_dataset = load_datasets(dataset_name=_cfg.dataset_name,
                                                                 suffix='_w-all-nans',
                                                                 dataset_constants=dataset_constants,
                                                                 transform=transform,
                                                                 dataset_type='supervised',
                                                                 root_path=basedir,
                                                                 outputdir=outputdir,
                                                                 skeleton_file=_cfg.skeleton_file,
                                                                 verbose=0)

        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

        df = pd.DataFrame(columns=['index_sample', 'length', 'keypoint', 'original'])
        i_data = 0

        for data_dict in tqdm(train_loader):
            logging.debug(f'{df.shape}')
            mask_holes = data_dict['mask_holes']
            mask_original = data_dict['original_mask']
            if torch.any(mask_original == 0):
                '''original hole'''
                out = find_holes(mask_holes[0], dataset_constants.KEYPOINTS, indep=_cfg.indep_keypoints)
                original = True
                for (_, length_nan, keypoint_name) in out:
                    df.loc[df.shape[0], :] = [i_data, int(length_nan), keypoint_name, original]
                out = find_holes(torch.sum(mask_holes[0], dim=1).view(mask_holes.shape[1], 1),
                                 ['non_missing'], target_val=0)
                for (_, length_nan, name) in out:
                    df.loc[df.shape[0], :] = [i_data, int(length_nan), name, original]
                i_data += 1
            elif not initial and torch.all(mask_original == 1) and torch.any(mask_holes == 1):
                original = False

                out = find_holes(mask_holes[0], dataset_constants.KEYPOINTS, indep=_cfg.indep_keypoints)
                for (_, length_nan, keypoint_name) in out:
                    df.loc[df.shape[0], :] = [i_data, int(length_nan), keypoint_name, original]
                out = find_holes(torch.sum(mask_holes[0], dim=1).view(mask_holes.shape[1], 1),
                                 ['non_missing'], target_val=0)
                for (_, length_nan, name) in out:
                    df.loc[df.shape[0], :] = [i_data, int(length_nan), name, original]
                i_data += 1

        logging.info(f'Done with the loop(s)')
        df = df.convert_dtypes()

        if initial:
            if len(df[df['original']]) == 0:
                no_original_missing = True
                ## no missing datapoints in the original files
                logging.info(f'No Missing keypoints in the original files. Create uniform missing proba.')
                proba_df, df_init_proba = create_uniform_proba(1, dataset_constants.SEQ_LENGTH - 1,
                                                               dataset_constants.KEYPOINTS)
                suffix = f'_uniform{suffix}'
            else:
                tmp = df.loc[df['original'], ['keypoint', 'original']].groupby('keypoint').count()
                set_keypoints = np.unique(df.loc[df['keypoint'] != 'non_missing', 'keypoint'])
                print(tmp, tmp.loc[set_keypoints, 'original'])
                init_proba = tmp.loc[set_keypoints, 'original'].values.astype('float')
                init_proba /= np.sum(init_proba)
                set_keypoints = np.unique(df.loc[df['keypoint'] != 'non_missing', 'keypoint'])#df.loc[:, 'keypoint'])
                init_proba = np.append(init_proba, 0)
                print(set_keypoints, init_proba.shape)
                df_init_proba = pd.DataFrame(columns=['keypoint', 'proba'], data=np.vstack([set_keypoints, init_proba]).T)
            df_init_proba.to_csv(os.path.join(outputdir, f'proba_missing{suffix}.csv'), index=False)

        if (not _cfg.indep_keypoints) and _cfg.merge_keypoints:
            """
            The idea here is to merge some keypoints in order to estimate better the missing probability.
            We use "merged_set" as a temporary variable to pool the corresponding keypoints, 
            estimate the corresponding probabilities, and attribute them to the original set of keypoints
            We replace the set_kp by their equivalent with full names of keypoints
            For the plots, we use the 'keypoint' column from df, so we substitute the 'set_kp' by the merged ones.
            
            """
            df.loc[:, 'merged_set'] = df['keypoint'].apply(lambda x: f'len_{len(x.split(" "))}' if len(x.split(" ")) > 1 else x)

            if initial:
                ## count only starts
                # count_starts = df.loc[(df['keypoint'] != 'non_missing') * df['original']].groupby('merged_set')['length'].count()
                # count_starts_merged = count_starts.reset_index()
                # count_starts_merged = count_starts_merged.rename({'length': 'proba'}, axis=1)
                # ## to reassign the probabilities to the original set_kp, we create a new df with the right number of rows
                # df_init_proba = pd.DataFrame(columns=['keypoint', 'merged_set'],
                #                              data=df.loc[(df['keypoint'] != 'non_missing') * (df['original']), ['keypoint', 'merged_set']].drop_duplicates())
                # df_init_proba = pd.merge(df_init_proba, count_starts_merged, how='left', on=['merged_set'])
                # count_starts_merged.loc[:, 'proba'] /= np.sum(count_starts_merged['proba'])
                # df_init_proba[['keypoint', 'proba']].to_csv(os.path.join(outputdir, f'proba_missing{suffix}.csv'), index=False)

                # keep the non_missing for the length proba
                total_count_keypoint = df.loc[df['original'], :].groupby(['merged_set'])['index_sample'].agg(
                    'count').reset_index().rename({'index_sample': 'total'}, axis=1)
                count_per_length_keypoint = df.loc[df['original'], :].groupby(['length', 'merged_set'])[
                    'index_sample'].agg('count').reset_index().rename({'index_sample': 'count'}, axis=1)
                proba_df = pd.DataFrame(columns=['keypoint', 'merged_set'],
                                        data=df.loc[df['original'], ['keypoint', 'merged_set']].drop_duplicates())
                proba_df = pd.merge(proba_df, pd.merge(count_per_length_keypoint, total_count_keypoint, on=['merged_set'],
                                                       how='left'), on=['merged_set'], how='left')
                proba_df.loc[:, 'proba'] = proba_df['count'] / proba_df['total']
                proba_df[['keypoint', 'length', 'proba']].sort_values(['keypoint', 'length'])\
                    .to_csv(os.path.join(outputdir, f'proba_missing_length{suffix}.csv'), index=False)

            df.loc[:, 'keypoint'] = df['merged_set']

        elif (not _cfg.merge_keypoints) and initial:
            # the counts here are the counts of starting points
            if not no_original_missing:
                logging.info('Computing the output proba files')
                total_count_keypoint = df.loc[df['original'], :].groupby(['keypoint'])['index_sample'].agg('count').reset_index().rename({'index_sample': 'total'}, axis=1)
                count_per_length_keypoint = df.loc[df['original'], :].groupby(['length', 'keypoint'])['index_sample'].agg('count').reset_index().rename({'index_sample': 'count'}, axis=1)
                proba_df = pd.merge(count_per_length_keypoint, total_count_keypoint, on=['keypoint'], how='left')
                proba_df.loc[:, 'proba'] = proba_df['count'] / proba_df['total']
                for kp in dataset_constants.KEYPOINTS:
                    if kp not in proba_df['keypoint'].unique():
                        ## length, keypoint, count, total, proba
                        proba_df.loc[proba_df.shape[0], :] = [0, kp, 1, 1, 1]
            proba_df[['keypoint', 'length', 'proba']].sort_values(['keypoint', 'length'])\
                .to_csv(os.path.join(outputdir, f'proba_missing_length{suffix}.csv'), index=False)

        def hist_length_original_vs_fake():
            plt.figure()
            bins = np.arange(0, dataset_constants.SEQ_LENGTH + 2) - 0.5
            if no_original_missing:
                plt.bar((bins[1:] + bins[:-1]) / 2, [1 / (len(bins) - 1)] * (len(bins) - 1), alpha=.5, label='original',
                        color='cornflowerblue')
            else:
                plt.hist(df.loc[df['original'] * (df['keypoint'] != 'non_missing'), 'length'], bins=bins, alpha=.5,
                         label='original', density=True, color='cornflowerblue')
            if not initial:
                plt.hist(df.loc[~df['original'] * (df['keypoint'] != 'non_missing'), 'length'], bins=bins, alpha=.5,
                         label='fake', density=True, color='seagreen')
            plt.legend()

        plot_save(hist_length_original_vs_fake, ~initial,
                  title=f'hist_length_original_vs_fake{suffix}',
                  only_png=False, outputdir=outputdir)

        def hist_length_per_keypoint():
            keypoints = df.loc[df['keypoint'] != 'non_missing', 'keypoint'].unique()
            n_rows = int(np.ceil(np.sqrt(len(keypoints))))
            n_cols = int(np.ceil(len(keypoints) / n_rows))
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
            axes = axes.flatten()
            bins = np.arange(0, dataset_constants.SEQ_LENGTH + 2) - 0.5
            for ikp, kp in enumerate(keypoints):
                if not no_original_missing:
                    distrib = df.loc[(df['original']) & (df['keypoint'] == kp), 'length'].values
                    weights = np.ones_like(distrib) / len(distrib)
                    axes[ikp].hist(distrib, bins=bins, histtype='step', weights=weights,
                                   label='original', linewidth=2)
                else:
                    axes[ikp].plot((bins[1:] + bins[:-1]) / 2, [1 / (len(bins) - 1)] * (len(bins) - 1),
                                   label='original', linewidth=2)
                if not initial:
                    distrib = df.loc[(~df['original']) & (df['keypoint'] == kp), 'length'].values
                    weights = np.ones_like(distrib) / len(distrib)
                    axes[ikp].hist(distrib, bins=bins, histtype='step', weights=weights,
                                   label='fake', linewidth=2)
                axes[ikp].set_title(kp)
                if ikp == 0:
                    axes[ikp].legend()
            plt.tight_layout()

        if not (no_original_missing and initial):
            plot_save(hist_length_per_keypoint, ~initial, title=f'hist_length_per_keypoint{suffix}', only_png=True,
                          outputdir=outputdir)

        def count_vs_keypoint():
            keypoints = df['keypoint'].unique()
            pivot_df = df.sample(min(df.shape[0] - 1, 5000), replace=False).groupby(['keypoint', 'original'])['index_sample'].agg('count').reset_index().rename({'index_sample': 'count'}, axis=1)
            if not no_original_missing:
                pivot_df.loc[pivot_df['original'], 'count'] /= pivot_df.loc[pivot_df['original'], 'count'].sum()
            pivot_df.loc[~pivot_df['original'], 'count'] /= pivot_df.loc[~pivot_df['original'], 'count'].sum()
            print(pivot_df['count'])
            sns.catplot(data=pivot_df, x='count', hue='original', y='keypoint', orient='h', alpha=0.9,
                        height=max(5, len(keypoints) // 10))
            if no_original_missing:
                plt.axvline(x=1 / len(keypoints), label='original')

        if not (no_original_missing and initial):
            plot_save(count_vs_keypoint, ~initial, title=f'count_vs_keypoint{suffix}', only_png=False,
                      outputdir=outputdir)

        logging.info(f'Done with initial = {initial}')


if __name__ == '__main__':
    """ LOGGING """
    logging.basicConfig(level=logging.DEBUG,
                        format=f'[%(levelname)s][%(asctime)s] %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("numpy").setLevel(logging.WARNING)

    create_proba_missing_files()