import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, to_rgba
import argparse
import os
from tqdm import tqdm
import pandas as pd
from umap import UMAP
from glob import glob
from omegaconf import OmegaConf
import logging
import seaborn as sns
from scipy.stats import mode
import plotly.express as px
from sklearn import preprocessing
import gc
from scipy.spatial.transform import Rotation
from matplotlib import gridspec
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

from DISK.utils.utils import read_constant_file, load_checkpoint
from DISK.utils.dataset_utils import load_datasets
from DISK.utils.transforms import init_transforms
from DISK.utils.train_fillmissing import construct_NN_model
from DISK.models.graph import Graph

import torch
from torch.utils.data import DataLoader


def statistics_human(input_tensor, dataset_constants, device):
    coordinates = input_tensor[:, :, :, :dataset_constants.DIVIDER]

    barycenter = torch.mean(coordinates[:, :, np.array([0, 12, 16]), :], dim=2)

    movement = torch.mean(torch.abs(torch.diff(coordinates, dim=1)), dim=(1, 2, 3))
    speed_z = torch.mean(torch.diff(barycenter[..., 2], dim=1), dim=1)
    speed_xy = torch.mean(torch.diff(barycenter[..., :2], dim=1), dim=(1, 2))
    upside_down = torch.mean(torch.mean(coordinates[:, :, np.array([16, 17, 18, 19, 12, 13, 14, 15]), 2], dim=1)
                             - torch.mean(coordinates[:, :, np.array([2, 3, 4, 5, 6, 7, 8, 9, 10]), 2]), dim=1)
    ## average height
    average_height = torch.mean(coordinates[:, :, :, 2], dim=(1, 2))
    ## back length
    mean_hips = torch.mean(coordinates[:, :, np.array([0, 12, 16]), :], dim=2)
    mean_shoulders = torch.mean(coordinates[:, :, np.array([2, 3, 4, 8]), :], dim=2)
    back_length = torch.sqrt(torch.sum((mean_hips - mean_shoulders) ** 2, dim=-1))  # of shape (batch, time)
    back_length = torch.mean(back_length, dim=1)  # shape (batch,)
    ## relative mobility of shoulders

    ### distance barycenter-shoulder
    dist_barycenter_shoulders = torch.sqrt(
        torch.sum((barycenter - mean_shoulders) ** 2, dim=-1))  # of shape (batch, time)
    dist_barycenter_shoulders = torch.mean(dist_barycenter_shoulders, dim=1)
    ### relative height of shoulder wrt the back
    height_shoulders = torch.mean(mean_shoulders[:, :, 2] - barycenter[:, :, 2],
                                  dim=-1)  ## look only at z coordinates
    ### angle of shoulders wrt the back
    mean_hips_coords = torch.mean(coordinates[:, :, np.array([0, 1, 12, 16]), :], dim=2)
    barycenter_back_vect = (mean_hips - mean_hips_coords)[:, :, :2]
    barycenter_shoulder_vect = (mean_shoulders - mean_hips_coords)[:, :, :2]
    # atan2(v2.y,v2.x) - atan2(v1.y,v1.x)
    angleXY_shoulders = torch.mean(
        torch.atan2(barycenter_shoulder_vect[:, :, 0], barycenter_shoulder_vect[:, :, 1]) - torch.atan2(
            barycenter_back_vect[:, :, 0], barycenter_back_vect[:, :, 1]), dim=1)
    angleXY_shoulders_base = torch.mean(
        torch.atan2(barycenter_shoulder_vect[:, :, 0], barycenter_shoulder_vect[:, :, 1]) - torch.atan2(
            torch.zeros_like(barycenter_back_vect[:, :, 0]), torch.ones_like(barycenter_back_vect[:, :, 0])).to(
            device), dim=1)

    # distance between knees
    dist_bw_knees = torch.mean(
        torch.sqrt(torch.sum((coordinates[:, :, 13, :] - coordinates[:, :, 17, :]) ** 2, dim=-1)), dim=1)
    # distance between shoulders and knees
    mean_knees = torch.mean(coordinates[:, :, np.array([13, 17]), :], dim=2)
    dist_knees_shoulders = torch.mean(torch.sqrt(torch.sum((mean_shoulders - mean_knees) ** 2, dim=-1)), dim=1)

    N = coordinates.shape[1]
    input_fft = coordinates.reshape(coordinates.shape[0], coordinates.shape[1], -1)
    coordinates_fft, _ = torch.max(torch.abs(2.0/N * torch.fft.fft(input_fft, dim=1)[:, 5:N//2]), dim=1)
    mask_low_fft = torch.all(coordinates_fft < 0.1, dim=1)
    mask_high_fft = torch.any(coordinates_fft > 2, dim=1)
    logging.info(f'[STATISTICS] #low_fft {np.sum(mask_low_fft)} #high_fft {np.sum(mask_high_fft)}')

    periodicity_cat = mask_low_fft.type(torch.float) * -1 + mask_high_fft.type(torch.float) * 1

    return (movement, upside_down, speed_xy, speed_z, average_height, back_length, dist_barycenter_shoulders,
            height_shoulders, angleXY_shoulders, dist_bw_knees, dist_knees_shoulders, angleXY_shoulders_base,
            periodicity_cat)

def statistics_MABe(input_tensor, dataset_constants, device):
    """
    Careful true keypoints are the following, not the ones
    KEYPOINTS = ['animal0_snout', 'animal0_leftear', 'animal0_rightear', 'animal0_neck', 'animal0_left', 'animal0_right', 'animal0_tail',
                 'animal1_snout', 'animal1_leftear', 'animal1_rightear', 'animal1_neck', 'animal1_left', 'animal1_right', 'animal1_tail']
    """
    coordinates = input_tensor[:, :, :, :dataset_constants.DIVIDER]

    barycenter = torch.mean(coordinates[:, :, :, :], dim=2)

    movement = torch.mean(torch.abs(torch.diff(coordinates, dim=1)), dim=(1, 2, 3))
    movement_mouse1 = torch.mean(torch.abs(torch.diff(coordinates[:, :, :7], dim=1)), dim=(1, 2, 3))
    movement_mouse2 = torch.mean(torch.abs(torch.diff(coordinates[:, :, 7:], dim=1)), dim=(1, 2, 3))
    movement_mouse1_mouse2 = torch.mean(torch.abs(
        torch.diff(coordinates[:, :, :7], dim=1) - torch.diff(coordinates[:, :, 7:],
                                                                               dim=1)), dim=(1, 2, 3))
    speed_xy = torch.mean(torch.diff(barycenter[..., :2], dim=1), dim=(1, 2))
    ## average height
    dist_bw_mice = torch.mean(torch.sum((torch.mean(coordinates[:, :, 3:7, :], dim=2) - torch.mean(
        coordinates[:, :, 11:, :], dim=2)) ** 2, dim=2), dim=1)
    angle_base = torch.mean(torch.atan2(barycenter[:, :, 0], barycenter[:, :, 1]) - torch.atan2(
        torch.zeros_like(barycenter[:, :, 0]), torch.ones_like(barycenter[:, :, 0])).to(device), dim=1)
    angle_2mice = torch.mean(
        torch.atan2(coordinates[:, :, 6, 0] - coordinates[:, :, 0, 0],
                    coordinates[:, :, 6, 1] - coordinates[:, :, 0, 1]) - torch.atan2(
            coordinates[:, :, 13, 0] - coordinates[:, :, 7, 0], coordinates[:, :, 13, 1] - coordinates[:, :, 7, 1]),
        dim=1)
    angle_mouse1 = torch.mean(
        torch.atan2(coordinates[:, :, 6, 0] - coordinates[:, :, 0, 0],
                    coordinates[:, :, 6, 1] - coordinates[:, :, 0, 1]) - torch.atan2(
        torch.zeros_like(barycenter[:, :, 0]), torch.ones_like(barycenter[:, :, 0])).to(device),
        dim=1)
    angle_mouse2 = torch.mean(
        torch.atan2(coordinates[:, :, 13, 0] - coordinates[:, :, 7, 0],
                    coordinates[:, :, 13, 1] - coordinates[:, :, 7, 1]) - torch.atan2(
        torch.zeros_like(barycenter[:, :, 0]), torch.ones_like(barycenter[:, :, 0])).to(device),
        dim=1)


    N = coordinates.shape[1]
    input_fft = coordinates.reshape(coordinates.shape[0], coordinates.shape[1], -1)
    coordinates_fft, _ = torch.max(torch.abs(2.0/N * torch.fft.fft(input_fft, dim=1)[:, 5:N//2]), dim=1)
    mask_low_fft = torch.all(coordinates_fft < 0.1, dim=1)
    mask_high_fft = torch.any(coordinates_fft > 2, dim=1)
    logging.info(f'[STATISTICS] #low_fft {np.sum(mask_low_fft)} #high_fft {np.sum(mask_high_fft)}')

    periodicity_cat = mask_low_fft.type(torch.float) * -1 + mask_high_fft.type(torch.float) * 1

    return (movement, movement_mouse1, movement_mouse2, movement_mouse1_mouse2, speed_xy,
            dist_bw_mice, angle_base, angle_2mice, angle_mouse1, angle_mouse2, periodicity_cat)

def extract_hidden(model, data_loader, dataset_constants, model_cfg, device,
                   compute_statistics=False, original_coordinates=False):
    label_ = []
    index_file = []
    index_pos = []
    hidden_array_ = []
    if len(dataset_constants.KEYPOINTS) == 14:
        statistics = {'movement': [],
                      'movement_mouse1': [],
                      'movement_mouse2': [],
                      'movement_mouse1-mouse2': [],
                      'speed_xy': [],
                      'dist_bw_mice': [],
                      'angle_2mice': [],
                      'angle_base': [],
                      'angle_mouse1': [],
                      'angle_mouse2': [],
                      'periodicity_cat': []
                     }
    elif len(dataset_constants.KEYPOINTS) == 20:
        statistics = {'movement': [],
                      'upside_down': [],
                      'speed_xy': [],
                      'speed_z': [],
                      'average_height': [],
                      'back_length': [],
                      'dist_barycenter_shoulders': [],
                      'height_shoulders': [],
                      'angleXY_shoulders': [],
                      'dist_bw_knees': [],
                      'dist_knees_shoulders': [],
                      'angle_back_base': [],
                      'periodicity_cat': []
                      }
    else:
        raise NotImplementedError

    for ith, data_dict in tqdm(enumerate(data_loader), total=len(data_loader), ascii=True, desc='Extract hidden'):
        if ith >= 2000:
            break ## for test purposes, quicker TO REMOVE !!!!!
        input_tensor = data_dict['x_supp'].to(device)
        index_file.append(data_dict['indices_file'].numpy())
        index_pos.append(data_dict['indices_pos'].numpy())
        if 'label' in data_dict.keys():
            labels = data_dict['label']
            label_.append(torch.squeeze(labels, 1))
        if torch.sum(torch.isnan(input_tensor)) != 0:
            print('stop')

        data_with_holes = data_dict['X'].to(device)
        mask_holes = data_dict['mask_holes'].to(device)
        input_tensor_with_holes = torch.cat([data_with_holes[..., :dataset_constants.DIVIDER], torch.unsqueeze(mask_holes, dim=-1)], dim=3)
        input_tensor_with_holes[:, 1:, :] = input_tensor_with_holes[:, :-1, :].clone()
        de_out = model.proj_input(input_tensor_with_holes, mask_holes)

        for i in range(model.num_layers):
            de_out = model.encoder_layers[i](de_out)

        if compute_statistics:  # TODO: implement with mask
            if len(dataset_constants.KEYPOINTS) == 14:
                (movement, movement_mouse1, movement_mouse2, movement_mouse1_mouse2, speed_xy,
                 dist_bw_mice, angle_base, angle_2mice, angle_mouse1, angle_mouse2,
                 periodicity_cat) = statistics_MABe(input_tensor, dataset_constants, device)

                statistics['movement'].extend(movement.detach().cpu().numpy())
                statistics['movement_mouse1'].extend(movement_mouse1.detach().cpu().numpy())
                statistics['movement_mouse2'].extend(movement_mouse2.detach().cpu().numpy())
                statistics['movement_mouse1-mouse2'].extend(movement_mouse1_mouse2.detach().cpu().numpy())
                statistics['speed_xy'].extend(speed_xy.detach().cpu().numpy())
                statistics['dist_bw_mice'].extend(dist_bw_mice.detach().cpu().numpy())
                statistics['angle_2mice'].extend(angle_2mice.detach().cpu().numpy())
                statistics['angle_base'].extend(angle_base.detach().cpu().numpy())
                statistics['angle_mouse1'].extend(angle_mouse1.detach().cpu().numpy())
                statistics['angle_mouse2'].extend(angle_mouse2.detach().cpu().numpy())
                statistics['periodicity_cat'].extend(periodicity_cat.detach().cpu().numpy())

            elif len(dataset_constants.KEYPOINTS) == 20:
                (movement, upside_down, speed_xy, speed_z, average_height, back_length, dist_barycenter_shoulders,
                 height_shoulders, angleXY_shoulders, dist_bw_knees, dist_knees_shoulders, angleXY_shoulders_base,
                 periodicity_cat) = statistics_human(input_tensor, dataset_constants, device)

                statistics['movement'].extend(movement.detach().cpu().numpy())
                statistics['upside_down'].extend(upside_down.detach().cpu().numpy())
                statistics['speed_xy'].extend(speed_xy.detach().cpu().numpy())
                statistics['speed_z'].extend(speed_z.detach().cpu().numpy())
                statistics['average_height'].extend(average_height.detach().cpu().numpy())
                statistics['back_length'].extend(back_length.detach().cpu().numpy())
                statistics['dist_barycenter_shoulders'].extend(dist_barycenter_shoulders.detach().cpu().numpy())
                statistics['height_shoulders'].extend(height_shoulders.detach().cpu().numpy())
                statistics['angleXY_shoulders'].extend(angleXY_shoulders.detach().cpu().numpy())
                statistics['dist_bw_knees'].extend(dist_bw_knees.detach().cpu().numpy())
                statistics['dist_knees_shoulders'].extend(dist_knees_shoulders.detach().cpu().numpy())
                statistics['angle_back_base'].extend(angleXY_shoulders_base.detach().cpu().numpy())
                statistics['periodicity_cat'].extend(periodicity_cat.detach().cpu().numpy())

        hidden_array_.append(de_out.view(de_out.shape[0], de_out.shape[1] * de_out.shape[2]).detach().cpu().numpy())

    if len(label_) > 0:
        label_ = np.vstack(label_)
        print(label_.shape)
    else:
        label_ = np.array([])
    hidden_array_ = np.vstack(hidden_array_)
    index_pos = np.concatenate(index_pos)
    index_file = np.concatenate(index_file)
    if compute_statistics:
        return hidden_array_, label_, index_file, index_pos, statistics
    else:
        return hidden_array_, label_, index_file, index_pos, None


def prepare_big_matrix(df_gp, train_or_test_id, dataset, outputdir='', treatment_first=True, length_line=100, cluster_algo=''):
    big_matrix = []
    percent_matrix = []
    if treatment_first:
        col_first = 'treatment_detail'
        col_second = 'mouse_id'
    else:
        col_first = 'mouse_id'
        col_second = 'treatment_detail'
    first_labels = []
    second_labels = []
    n_second_per_first = [0, ]
    train_or_test = []
    n_clusters = int(np.max(df_gp['cluster'])) + 1
    for tot in train_or_test_id:
        for first_id in np.unique(df_gp[col_first]):
            i = 0
            for second_id in np.unique(df_gp.loc[df_gp[col_first] == first_id, col_second]):
                df_gp_mask = (df_gp[col_second] == second_id) & (df_gp[col_first] == first_id) & (
                            df_gp['train_or_test'] == tot)
                line_values = np.zeros(length_line) * np.nan
                percent_line_values = np.zeros(n_clusters)
                offset = 0
                for cluster_id in np.sort(df_gp.loc[df_gp_mask, 'cluster'].unique()):
                    p = df_gp.loc[df_gp_mask & (df_gp['cluster'] == cluster_id), 'percent'].values[0]
                    rounded_p = int(np.round(p * length_line, 0))
                    line_values[offset: offset + rounded_p] = cluster_id
                    percent_line_values[int(cluster_id)] = p
                    offset += rounded_p
                if offset > 0:
                    # else it means no data have been found, we don't want to append a line without info
                    big_matrix.append(line_values)
                    percent_matrix.append(percent_line_values)
                    second_labels.append(second_id)
                    i += 1

            n_second_per_first.append(i)
            train_or_test.append(tot)

            first_labels.append(first_id)

    big_matrix = np.vstack(big_matrix)
    percent_matrix = np.vstack(percent_matrix)
    n_second_per_first = np.cumsum(n_second_per_first)
    train_or_test = np.array(train_or_test)

    plot_big_matrix(big_matrix, n_second_per_first, first_labels, col_first, train_or_test,
                    title='Cluster proportions', cluster_algo=cluster_algo, dataset=dataset, k=n_clusters,
                    outputdir=outputdir)

    return big_matrix, percent_matrix, n_second_per_first, np.array(first_labels), train_or_test

def plot_big_matrix(big_matrix, n_second_per_first, first_labels, col_first, train_or_test_id,
                    title='Cluster vs time', cluster_algo='kmeans', k=10, dataset='', outputdir=''):
    fig, ax = plt.subplots(1, 1, figsize=(15, 9))
    map_ = ax.imshow(big_matrix, cmap=ListedColormap(sns.color_palette("hls", k).as_hex()))

    for i in range(len(n_second_per_first) - 1):
        ax.axhline(y=n_second_per_first[i + 1] + 0.5, c='k')
    if np.all(train_or_test_id == 'train'): ## only one class
        ax.set_title(title + 'train')
        plot_name = 'only_train'
    elif np.all(train_or_test_id == 'eval'): ## only one class
        ax.set_title(title + 'test')
        plot_name = 'only_test'
    else: ## two classes
        ax.set_title(title + 'train and test')
        train_vs_test_i = np.where(np.diff(train_or_test_id != 'train'))[0][0]
        ax.axhline(y=n_second_per_first[train_vs_test_i] + 0.5, c='w')
        plot_name = 'train_and_test'

    ## Y AXIS LABELS, FIRST LABELS
    y_pos, y_label = [], []
    for i in range(len(n_second_per_first) - 1):
        y_pos.append((n_second_per_first[i] + n_second_per_first[i + 1]) / 2)
        y_label.append(first_labels[i])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_label)
    ax.set_ylabel(col_first)
    ax.set_ylim([len(big_matrix), -1])

    plt.colorbar(map_)
    plt.tight_layout()
    plt.savefig(
        os.path.join(outputdir,
                     f'{dataset}_{cluster_algo}-{k}_{"-".join(title.split(" "))}_by_{col_first}_first_{plot_name}_latent.svg'))


def plot_umaps(df, all_columns, outputdir, dataset_name, suffix):

    for label_name in all_columns:
        logging.info(f'drawing umap with colors = {label_name}')

        fig = px.scatter(df, x='umap_x', y='umap_y', color=label_name,
                         hover_data=all_columns + ['train_or_test'])

        fig.write_html(os.path.join(outputdir, f'{dataset_name}_normed_umap_colors-{label_name}_latent{suffix}.html'))

    logging.info(f'drawing umap overview')
    ncols = int(np.sqrt(len(all_columns)))
    nrows = int(np.ceil(len(all_columns) / ncols))
    fig, ax = plt.subplots(nrows, ncols, figsize=(20, 15), sharex='all', sharey='all')
    ax = ax.flatten()
    for i, label_name in enumerate(all_columns):
        if df[label_name].dtype in [float, np.float32, np.float64, int, np.int32, np.int64]:
            ax[i].set_title(
                f'{label_name}: min {np.min(df[label_name]):.2f} - max {np.max(df[label_name]):.2f}')
            ax[i].scatter(df['umap_x'], df['umap_y'], c=df[label_name], s=1, cmap='jet')
            ax[i].set_title(label_name)
        else:
            ids, names = pd.factorize(df[label_name])
            ax[i].scatter(df['umap_x'], df['umap_y'], c=ids, s=1)
            if len(names) < 5:
                ax[i].set_title(f'{label_name}: {" ".join(names)}')
            else:
                ax[i].set_title(f'{label_name}')

    plt.tight_layout()
    plt.savefig(os.path.join(outputdir, f'{dataset_name}_normed_umap_overview_latent{suffix}.png'))


def get_cmap(matrix, cmap_str='jet'):
    num_ = np.max(matrix)
    unique_ids = np.unique(matrix)
    unique_ids = unique_ids[unique_ids > 0]
    cmap_internal = plt.get_cmap(cmap_str)

    colors = cmap_internal([float(i) / len(unique_ids) for i in range(len(unique_ids))])
    background = "black"
    all_colors = [background if not j in unique_ids else colors[i] for i, j in enumerate(range(num_))]
    cmap_internal = ListedColormap(all_colors)
    return cmap_internal, all_colors


def plot_cluster_expression(df, scalar_columns, all_colors):
    if len(scalar_columns) == 0:
        return
    heatmap_clusters_index = np.unique(df['cluster'])
    heatmap_vectors = []
    col_colors = []
    acc_idx = 0
    acc_sizes = []
    for cl_idx in heatmap_clusters_index:
        cl_vecs = df.loc[df['cluster'] == cl_idx, scalar_columns][::10]
        acc_sizes.append(cl_vecs.shape[0])
        col = all_colors[cl_idx - 2]
        heatmap_vectors.append(cl_vecs)
        col_colors.append([to_rgba(col)] * cl_vecs.shape[0])
        acc_idx = np.sum(acc_sizes)
    heatmap_vectors = np.vstack(heatmap_vectors)
    col_colors = np.vstack(col_colors)
    gene_exp_heatmap = heatmap_vectors.T
    gene_exp_heatmap = preprocessing.scale(gene_exp_heatmap, axis=1)
    g = sns.clustermap(gene_exp_heatmap, figsize=[12, 9], yticklabels=scalar_columns,
                       cmap='bwr', row_cluster=True, col_cluster=False,
                       col_colors=col_colors, xticklabels=1000, vmin=-2.5, vmax=2.5)
    g.cax.set_visible(False)
    g.ax_heatmap.tick_params(labelright=False, labelleft=True, right=False)


def apply_kmeans(k, hi_train, hi_eval, df, proj_train, proj_eval, metadata_columns,
                 outputfile=''):
    kmeans = KMeans(n_clusters=k, n_init=10).fit(hi_train)

    kmeans_clustering_train = kmeans.predict(hi_train)
    kmeans_clustering_eval = kmeans.predict(hi_eval)

    # UMAP
    fig = plt.figure(figsize=(15, 9))
    plt.scatter(proj_train[:, 0], proj_train[:, 1], c=kmeans_clustering_train, cmap='Set2')
    plt.scatter(proj_eval[:, 0], proj_eval[:, 1], c=kmeans_clustering_eval, cmap='Set2', marker='v')
    plt.colorbar()
    plt.tight_layout()

    plt.savefig(outputfile)
    plt.close()

    # Build dataframe with cluster information and metadata info
    df.loc[:, 'cluster'] = np.concatenate([kmeans_clustering_train, kmeans_clustering_eval])

    # Build a dataframe with percentage of each cluster per mouse x experiment
    df_gp = df.groupby(metadata_columns + ['cluster', 'train_or_test'])['cluster'].agg('count').rename('count').reset_index()
    if len(metadata_columns) > 0:
        df_count_per_GT = df.groupby(metadata_columns).agg('count').rename({'cluster': 'count'}, axis=1).reset_index()
    
        def norm(x):
            mask = (df_count_per_GT[metadata_columns[0]] == x[metadata_columns[0]])
            for c in metadata_columns[1:]:
                mask = mask & (df_count_per_GT[c] == x[c])
            return x['count'] / df_count_per_GT.loc[mask, 'count'].values[0]
    
        df_gp.loc[:, 'percent'] = df_gp.apply(norm, axis=1)
    else:
        df_gp.loc[:, 'percent'] = df_gp['count'] / df_gp.shape[0]

    return df, df_gp, kmeans.cluster_centers_


def plot_sequential(coordinates, skeleton_graph, keypoints, nplots, save_path, size=40,
                  normalized_coordinates=False, azim=60):
    """
    Plot sequence as 3D poses, using skeleton information
    """

    min_ = np.nanmin(coordinates, axis=(0, 1))
    max_ = np.nanmax(coordinates, axis=(0, 1))
    n_dim = len(min_)

    plt.ioff()
    gs = gridspec.GridSpec(1, nplots, width_ratios=[1] * nplots,
                           wspace=0.0, hspace=0.0, top=1, bottom=0, left=0, right=1)
    plt.figure(figsize=(nplots * size, size), facecolor=(1, 1, 1))

    if n_dim > 2: # Human Mocap data because MABe are only 2D
        index_rhip = 12 #keypoints.index('leg1_0')
        index_lhip = 16 #keypoints.index('leg1_1')
        coordinates = np.dstack([coordinates[..., -1], coordinates[..., 1], coordinates[..., 0]])
        rhip_coord = coordinates[len(coordinates) // 2, index_rhip]
        lhip_coord = coordinates[len(coordinates) // 2, index_lhip]
        norm_ref = np.linalg.norm(rhip_coord - lhip_coord)
        vect_ref = (rhip_coord - lhip_coord) / norm_ref

    for idx_time in range(nplots):
        matrix_gt = coordinates[int(idx_time / nplots * len(coordinates))]
        if n_dim > 2:
            # Update 3D poses
            norm_ = np.linalg.norm(matrix_gt[index_rhip] - matrix_gt[index_lhip])
            matrix_gt = (matrix_gt - matrix_gt[index_lhip])
            old_vect = matrix_gt[index_rhip] / norm_
            v = np.cross(old_vect, vect_ref)
            vx = np.array([[0, - v[2], v[1]],
                           [v[2], 0, - v[0]],
                           [- v[1], v[0], 0]])
            c = np.dot(old_vect, vect_ref)
            s = np.linalg.norm(v)
            R = np.identity(3) + vx + vx**2 * (1 - c) / (s ** 2 + 1e-9)
            matrix_gt = Rotation.from_matrix(R).apply(matrix_gt)
            ax = plt.subplot(gs[idx_time], projection='3d')
            ax.view_init(elev=15., azim=azim)
        else:
            matrix_gt = coordinates[int(idx_time / nplots * len(coordinates))]
            ax = plt.subplot(gs[idx_time])

        try:
            ax.set_aspect('equal')
        except NotImplementedError:
            ax.set_aspect('auto')

        if n_dim > 2:
            ax.set_xlim3d([min_[0], max_[0]])
            ax.set_ylim3d([min_[1], max_[1]])
            ax.set_zlim3d([min_[2], max_[2]])
        else:
            ax.set_xlim([min_[0], max_[0]])
            ax.set_ylim([min_[1], max_[1]])

        for nl, nlcolor in zip(skeleton_graph.neighbor_link, skeleton_graph.neighbor_link_color):
            if n_dim > 2:
                ax.plot([matrix_gt[nl[0], 0], matrix_gt[nl[1], 0]],
                        [matrix_gt[nl[0], 1], matrix_gt[nl[1], 1]],
                        [matrix_gt[nl[0], 2], matrix_gt[nl[1], 2]],
                        'o-', color=nlcolor, lw=4)
            else:
                ax.plot([matrix_gt[nl[0], 0], matrix_gt[nl[1], 0]],
                        [matrix_gt[nl[0], 1], matrix_gt[nl[1], 1]],
                        'o-', color=nlcolor, lw=4)

        ax.set_axis_off()
        plt.subplots_adjust(wspace=0, hspace=0, left=0, top=1, bottom=0, right=1)

    plt.savefig(save_path + '.svg')
    plt.close()


if __name__ == '__main__':

    ###################################################################################################################
    ### Only works with Human Mocap and MABe dataset
    ### And DISK-transformer models
    ###################################################################################################################

    p = argparse.ArgumentParser()
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--checkpoint_folder", type=str, required=True)
    p.add_argument("--stride", type=float, required=True, default='in seconds')
    p.add_argument("--suffix", type=str, default='', help='string suffix added to the save files')
    p.add_argument("--dataset_path", type=str, default='', help='absolute path where to find datasets')
    p.add_argument("--k", type=int, default=10, help='number of k-means clusters')
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format=f'[%(levelname)s][%(asctime)s] %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("numpy").setLevel(logging.WARNING)

    config_file = os.path.join(args.checkpoint_folder, '.hydra', 'config.yaml')
    model_cfg = OmegaConf.load(config_file)
    model_path = glob(os.path.join(args.checkpoint_folder, 'model_epoch*'))[0]  # model_epoch to not take the model from the lastepoch

    dataset_constants = read_constant_file(os.path.join(args.dataset_path, 'datasets', model_cfg.dataset.name, 'constants.py'))

    if model_cfg.dataset.skeleton_file is not None:
        skeleton_file_path = os.path.join(args.dataset_path, 'datasets', model_cfg.dataset.skeleton_file)
        if not os.path.exists(skeleton_file_path):
            raise ValueError(f'no skeleton file found in', skeleton_file_path)
        skeleton_graph = Graph(file=skeleton_file_path)
    else:
        skeleton_graph = None
        skeleton_file_path = None

    """ DATA """
    transforms, _ = init_transforms(model_cfg, dataset_constants.KEYPOINTS, dataset_constants.DIVIDER,
                                 dataset_constants.SEQ_LENGTH, args.dataset_path, args.checkpoint_folder)

    logging.info('Loading datasets...')
    train_dataset, val_dataset, test_dataset = load_datasets(dataset_name=model_cfg.dataset.name,
                                                             dataset_constants=dataset_constants,
                                                             transform=transforms,
                                                             dataset_type='full_length',
                                                             stride=args.stride,
                                                             suffix='_w-0-nans',
                                                             root_path=args.dataset_path,
                                                             length_sample=dataset_constants.SEQ_LENGTH,
                                                             freq=dataset_constants.FREQ,
                                                             outputdir=args.checkpoint_folder,
                                                             skeleton_file=None,
                                                             label_type='all',  # don't care, not using
                                                             verbose=model_cfg.feed_data.verbose)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info("Device: {}".format(device))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    logging.info('Loading transformer model...')
    # load model
    model = construct_NN_model(model_cfg, dataset_constants, skeleton_file_path, device)
    logging.info(f'Network constructed')

    load_checkpoint(model, None, model_path, device)

    logging.info('Extract hidden representation...')
    ### DIRECT KNN ON SEQ2SEQ LATENT SPACE
    hi_train, label_train, index_file_train, index_pos_train, statistics_train = extract_hidden(model, train_loader, dataset_constants, model_cfg,
                                           device, compute_statistics=True)
    time_train = train_dataset.possible_times
    logging.info('Done with train hidden representation...')

    hi_eval, label_eval, index_file_eval, index_pos_eval, statistics_eval = extract_hidden(model, val_loader, dataset_constants, model_cfg,
                                         device, compute_statistics=True)
    time_eval = val_dataset.possible_times
    logging.info('Done with val hidden representation...')


    logging.info(f'hidden eval vectors {hi_eval.shape}')
    logging.info(f'hidden train vectors {hi_train.shape}')


    ##############################################################################################
    ### Plot umap with different coloring
    #############################################################################################
    try:
        metadata_columns = dataset_constants.METADATA
    except AttributeError:
        metadata_columns = []
    scalar_columns = list(statistics_train.keys()) if statistics_train is not None else []

    # Create dataframe with metdata
    df = pd.DataFrame()

    df.loc[:, 'train_or_test'] = np.concatenate([['train'] * hi_train.shape[0], ['eval'] * hi_eval.shape[0]])
    df.loc[df['train_or_test'] == 'train', 'index_file'] = index_file_train
    df.loc[df['train_or_test'] == 'eval', 'index_file'] = index_file_eval
    df.loc[df['train_or_test'] == 'train', 'index_pos'] = index_pos_train
    df.loc[df['train_or_test'] == 'eval', 'index_pos'] = index_pos_eval
    for imc, mc in enumerate(metadata_columns):
        df.loc[df['train_or_test'] == 'train', mc] = label_train[:, imc]
        df.loc[df['train_or_test'] == 'eval', mc] = label_eval[:, imc]
    df.loc[:, 'time'] = np.concatenate([time_train[:len(label_train)], time_eval])

    if 'Mocap' in model_cfg.dataset.name and 'action' in df.columns:
        reverse_dict_label = {0: 'Walk', 1: 'Wash', 2: 'Run', 3: 'Jump', 4: 'Animal Behavior', 5: 'Dance',
                              6: 'Step', 7: 'Climb', 8: 'unknown'}
        df.loc[:, 'action_str'] = df['action'].apply(lambda x: reverse_dict_label[int(x)])
        metadata_columns += ['action_str']
    if 'MAB' in model_cfg.dataset.name and 'action' in df.columns:
        reverse_dict_label = {0: 'attack', 1: 'investigation', 2: 'mount', 3: 'other'}
        df.loc[:, 'action_str'] = df['action'].apply(lambda x: reverse_dict_label[int(x)])
        metadata_columns += ['action_str']
    all_columns = metadata_columns + scalar_columns
    logging.info(f'columns: {all_columns}')

    # get the representation per time
    bin_edges = np.arange(int(np.ceil(df['time'].max())) + 2) - 0.5
    bin_centers = bin_edges[:-1] + 0.5
    df.loc[:, 'time_bin'] = pd.cut(df['time'], bins=bin_edges, labels=bin_centers)

    if statistics_train is not None:
        for key in statistics_train.keys():
            df.loc[:, key] = statistics_train[key] + statistics_eval[key]

    logging.info('Computing the umap projection')

    for _ in range(3):
        gc.collect()

    myumap = UMAP(low_memory=True, densmap=False)
    if len(hi_train) > 5000:
        vect2project = hi_train[np.random.choice(np.arange(hi_train.shape[0], dtype=int), 5000, replace=False)]
    else:
        vect2project = np.array(hi_train)
    myumap.fit(vect2project)
    logging.info('Finished projecting')

    for _ in range(3):
        gc.collect()

    proj_train = myumap.transform(hi_train)
    logging.info('Finished projecting on the train')
    proj_eval = myumap.transform(hi_eval)
    logging.info('Finished projecting on the eval')
    df.loc[df['train_or_test'] == 'train', ['umap_x', 'umap_y']] = proj_train
    df.loc[df['train_or_test'] == 'eval', ['umap_x', 'umap_y']] = proj_eval

    logging.info('Apply k-means...')
    df, df_percent, cluster_centers = apply_kmeans(args.k, hi_train, hi_eval, df, proj_train, proj_eval, metadata_columns,
                                                   outputfile=os.path.join(args.checkpoint_folder,
                                                   f'{model_cfg.dataset.name}_normed_train_umap_colors-kmeans_latent_1.png'))

    plot_umaps(df, all_columns + ['cluster'], args.checkpoint_folder, model_cfg.dataset.name, args.suffix)

    logging.info('Saving data in csv and npy')
    df.to_csv(os.path.join(args.checkpoint_folder, f'{model_cfg.dataset.name}.csv'),
              index=False)
    columns = [c for c in df.columns if 'latent' not in c]
    df[columns].to_csv(os.path.join(args.checkpoint_folder, f'{model_cfg.dataset.name}_metadata.csv'),
              index=False)
    np.save(os.path.join(args.checkpoint_folder, f'{model_cfg.dataset.name}_latent_train'), hi_train)
    # np.save(os.path.join(args.checkpoint_folder, f'{model_cfg.dataset.name}_latent_eval'), hi_eval)
    np.save(os.path.join(args.checkpoint_folder, f'{model_cfg.dataset.name}_cluster_centers'), cluster_centers)

    ##############################################################################################
    ### With a fix value of kmeans, compute umap and "signature" plot
    ##############################################################################################

    logging.debug(f"plot_cluster_expression")
    cmap_internal, all_colors = get_cmap(df['cluster'].values, "prism")
    plot_cluster_expression(df, scalar_columns, all_colors)

    ## Find cluster representatives
    for lbl, center in zip(np.unique(df['cluster']), cluster_centers):
        train_or_eval = df.loc[df['cluster'] == lbl, 'train_or_test'].values[0]
        if train_or_eval == 'train':
            train_df = df.loc[df['train_or_test'] == train_or_eval]
            mask = (train_df['cluster'] == lbl).values
            vect = hi_train[mask]
            dist_to_center = cdist(vect, [center])
        indices = np.argsort(dist_to_center.flatten())[:10]
        original_indices = df.loc[df['cluster'] == lbl].index[indices]
        train_or_test = df.loc[df['cluster'] == lbl, 'train_or_test'].values[indices]
        coordinates = []
        labels = []
        for ind, tt in zip(original_indices, train_or_test):
            if tt == 'train':
                data_dict = train_dataset.__getitem__(ind)
            else:
                data_dict = val_dataset.__getitem__(ind - len(hi_train))
            coordinates.append(data_dict['x_supp'][..., :dataset_constants.DIVIDER].detach().numpy())
            # print(data_dict['label'].detach().numpy()[0, dataset_constants.METADATA.index('action')])
            labels.append(reverse_dict_label[data_dict['label'].detach().numpy()[0, dataset_constants.METADATA.index('action')]])
        for i in range(3):
            # print(coordinates[i].shape)
            plot_sequential(coordinates[i], skeleton_graph, dataset_constants.KEYPOINTS, 20,
                            os.path.join(args.checkpoint_folder, f'cluster-{lbl}_representatives-{i}_traj3D_{labels[i]}'), size=20, azim=45)
