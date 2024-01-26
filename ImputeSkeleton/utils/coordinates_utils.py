import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import os
import plotly.graph_objects as go
from collections.abc import Iterable


########################################################################################
### The idea here is to regroup all the functions that takes a matrix of coordinates
### i.e (n_keypoints, 3D)
########################################################################################

def create_skeleton_plot(matrix, skeleton_graph, color='darkblue', name=''):
    """
    :param matrix: (n_keypoints, 3D)
    :param color: str, matplotlib color
    :param name: str, label for the plotted skeleton
    :param drop_ankles: bool, if True expects coordinatey correspondings to ankles on 8th and 9th row, if False, does not draw ankles.
                        default: False

    :return: the 2 go.Scatter3D plots
    """
    neighbor_links = skeleton_graph.neighbor_link
    if matrix.shape[1] > 2:
        output = []
        for nl in neighbor_links:
            output.append(go.Scatter3d(x=[matrix[nl[0], 0], matrix[nl[1], 0]],
                                       y=[matrix[nl[0], 1], matrix[nl[1], 1]],
                                       z=[matrix[nl[0], 2], matrix[nl[1], 2]],
                                       line=dict(color=color, width=2),
                                       marker=dict(size=1, color=color),
                                       name=f'{nl[0]} - {nl[1]}'))
    else:
        output = []
        for nl in neighbor_links:
            output.append(go.Scatter(x=[matrix[nl[0], 0], matrix[nl[1], 0]],
                                     y=[matrix[nl[0], 1], matrix[nl[1], 1]],
                                     line=dict(color=color, width=2),
                                     marker=dict(size=1, color=color),
                                     name=f'{nl[0]} - {nl[1]}'))

    return output


def compare2skeletons_plot(mat0, mat1, skeleton_graph,
                           color0='darkblue', color1='darkgreen',
                           name0='', name1=''):
    """

    :param matrix: (n_keypoints, 3D)
    :param color: str, matplotlib color
    :param name: str, label for the plotted skeleton

    :return: the 2 go.Scatter3D plots
    """
    assert len(mat0) == len(mat1)
    output = []
    output.extend(create_skeleton_plot(mat0, skeleton_graph, color=color0, name=name0))
    output.extend(create_skeleton_plot(mat1, skeleton_graph, color=color1, name=name1))
    if mat0.shape[1] > 2:
        for joint in range(len(mat0)):
            b = go.Scatter3d(x=[mat0[joint, 0], mat1[joint, 0]],
                             y=[mat0[joint, 1], mat1[joint, 1]],
                             z=[mat0[joint, 2], mat1[joint, 2]],
                             line=dict(color='red', width=2),
                             marker=dict(size=0.1, color='red'),
                             name=None,
                             opacity=0.4)
            output.append(b)
    else:
        for joint in range(len(mat0)):
            b = go.Scatter(x=[mat0[joint, 0], mat1[joint, 0]],
                             y=[mat0[joint, 1], mat1[joint, 1]],
                             line=dict(color='red', width=2),
                             marker=dict(size=0.1, color='red'),
                             name=None,
                             opacity=0.4)
            output.append(b)
    return output


def create_seq_plot(matrices, skeleton_graph, cmap_name, name='', max_n_display=8, color_middle=f'rgba(255,0,0,1)'):
    output = []
    subsample = len(matrices) // max_n_display
    num_colors = max_n_display + 1  # first color is white, we cannot differentiate between cmaps so leave it out
    mycm = plt.get_cmap(cmap_name)
    middle_index = len(matrices[::subsample]) // 2
    for i, m in enumerate(matrices[::subsample]):
        r, g, b, a = mycm(1. * (i + 1) / num_colors)
        if i == middle_index:
            color4plotly = color_middle
        else:
            color4plotly = f'rgba({r * 255:.0f},{g * 255:.0f},{b * 255:.0f},{a:.1f})'
        output.extend(create_skeleton_plot(m, skeleton_graph, color=color4plotly, name=f'{name} t = {i * subsample}'))

    return output


def compare2seq_plot(mat0, res0, mat1, res1, skeleton_graph,
                     cmap_name0, cmap_name1,
                     name0='', name1='',
                     max_n_display=8,
                     color_middle0=f'rgba(255,0,0,1)', color_middle1=f'rgba(255,0,0,1)'):
    assert len(mat0) == len(mat1)
    output = []
    subsample = len(mat0) // max_n_display
    num_colors = max_n_display + 1
    mycm0 = plt.get_cmap(cmap_name0)
    mycm1 = plt.get_cmap(cmap_name1)
    middle_index = len(mat0[::subsample]) // 2
    for i in range(0, len(mat0), subsample):
        r0, g0, b0, a0 = mycm0(1. * (i + 1) / num_colors)
        r1, g1, b1, a1 = mycm1(1. * (i + 1) / num_colors)
        if i == middle_index:
            color4plotly0 = color_middle0
            color4plotly1 = color_middle1
        else:
            color4plotly0 = f'rgba({r0 * 255:.0f},{g0 * 255:.0f},{b0 * 255:.0f},{a0:.1f})'
            color4plotly1 = f'rgba({r1 * 255:.0f},{g1 * 255:.0f},{b1 * 255:.0f},{a1:.1f})'
        output.extend(compare2skeletons_plot(mat0[i], mat1[i], skeleton_graph,
                                             color0=color4plotly0, color1=color4plotly1,
                                             name0=f'{name0} t = {i}', name1=f'{name1} t = {i}'))
    return output



from mpl_toolkits.mplot3d.axes3d import get_test_data
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

def plot_sequence(coordinates, output, mask_holes, skeleton_graph, nplots, save_path, size=40,
                  normalized_coordinates=False, azim=60):
    """
    Plot sequence as 3D poses, using skeleton information
    """

    min_ = np.minimum(np.nanmin(coordinates, axis=(0, 1)), np.nanmin(output, axis=(0, 1)))
    max_ = np.maximum(np.nanmax(coordinates, axis=(0, 1)), np.nanmax(output, axis=(0, 1)))
    n_dim = len(min_)

    plt.ioff()
    gs = gridspec.GridSpec(1, nplots, width_ratios=[1] * nplots,
                           wspace=0.0, hspace=0.0, top=1, bottom=0, left=0, right=1)
    plt.figure(figsize=(nplots * size, size), facecolor=(1, 1, 1))

    for idx_time in range(nplots):
        # Update 3D poses
        matrix_gt = coordinates[int(idx_time / nplots * len(coordinates))]
        matrix_out = output[int(idx_time / nplots * len(coordinates))]
        mask = mask_holes[int(idx_time / nplots * len(coordinates))]

        if n_dim > 2:
            ax = plt.subplot(gs[idx_time], projection='3d')
            ax.view_init(elev=15., azim=azim)
        else:
            ax = plt.subplot(gs[idx_time])

        try:
            ax.set_aspect('equal')
        except NotImplementedError:
            ax.set_aspect('auto')
        # if normalized_coordinates:

        if n_dim > 2:
            ax.set_xlim3d([min_[0], max_[0]])
            ax.set_ylim3d([min_[1], max_[1]])
            ax.set_zlim3d([min_[2], max_[2]])
        else:
            ax.set_xlim([min_[0], max_[0]])
            ax.set_ylim([min_[1], max_[1]])

        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        if n_dim > 2:
            ax.zaxis.set_ticklabels([])

        for line in ax.xaxis.get_ticklines():
            line.set_visible(False)
        for line in ax.yaxis.get_ticklines():
            line.set_visible(False)
        if n_dim > 2:
            for line in ax.zaxis.get_ticklines():
                line.set_visible(False)

        for nl, nlcolor in zip(skeleton_graph.neighbor_link, skeleton_graph.neighbor_link_color):
            if n_dim > 2:
                ax.plot([matrix_gt[nl[0], 0], matrix_gt[nl[1], 0]],
                        [matrix_gt[nl[0], 1], matrix_gt[nl[1], 1]],
                        [matrix_gt[nl[0], 2], matrix_gt[nl[1], 2]],
                        'o-', color=nlcolor)
                if mask[nl[0]] == 1 and mask[nl[1]] == 1:  ## both missing
                    ax.plot([matrix_out[nl[0], 0], matrix_out[nl[1], 0]],
                            [matrix_out[nl[0], 1], matrix_out[nl[1], 1]],
                            [matrix_out[nl[0], 2], matrix_out[nl[1], 2]],
                            'x-', color='crimson', ms=10)
                elif mask[nl[0]] == 1: ## only nl[0] missing
                    if np.any(np.isnan(matrix_gt[nl[0], 0])):
                        ax.plot([matrix_out[nl[0], 0], matrix_gt[nl[1], 0]],
                                [matrix_out[nl[0], 1], matrix_gt[nl[1], 1]],
                                [matrix_out[nl[0], 2], matrix_gt[nl[1], 2]],
                                'o-', color=nlcolor)
                    ax.plot([matrix_out[nl[0], 0], ],
                            [matrix_out[nl[0], 1], ],
                            [matrix_out[nl[0], 2], ],
                            'x', color='crimson', ms=10)
                elif mask[nl[1]] == 1:
                    if np.any(np.isnan(matrix_gt[nl[1], 0])):
                        ax.plot([matrix_gt[nl[0], 0], matrix_out[nl[1], 0]],
                                [matrix_gt[nl[0], 1], matrix_out[nl[1], 1]],
                                [matrix_gt[nl[0], 2], matrix_out[nl[1], 2]],
                                'o-', color=nlcolor)
                    ax.plot([matrix_out[nl[1], 0], ],
                            [matrix_out[nl[1], 1], ],
                            [matrix_out[nl[1], 2], ],
                            'x', color='crimson', ms=10)
            else:
                ax.plot([matrix_gt[nl[0], 0], matrix_gt[nl[1], 0]],
                        [matrix_gt[nl[0], 1], matrix_gt[nl[1], 1]],
                        'o-', color=nlcolor)
                if mask[nl[0]] == 1 and mask[nl[1]] == 1:  ## both missing
                    ax.plot([matrix_out[nl[0], 0], matrix_out[nl[1], 0]],
                            [matrix_out[nl[0], 1], matrix_out[nl[1], 1]],
                            'x-', color='crimson', ms=10)
                elif mask[nl[0]] == 1:  ## only nl[0] missing
                    if np.any(np.isnan(matrix_gt[nl[0], 0])):
                        ax.plot([matrix_out[nl[0], 0], matrix_gt[nl[1], 0]],
                                [matrix_out[nl[0], 1], matrix_gt[nl[1], 1]],
                                'o-', color=nlcolor)
                    ax.plot([matrix_out[nl[0], 0], ],
                            [matrix_out[nl[0], 1], ],
                            'x', color='crimson', ms=10)
                elif mask[nl[1]] == 1:
                    if np.any(np.isnan(matrix_gt[nl[1], 0])):
                        ax.plot([matrix_gt[nl[0], 0], matrix_out[nl[1], 0]],
                                [matrix_gt[nl[0], 1], matrix_out[nl[1], 1]],
                                'o-', color=nlcolor)
                    ax.plot([matrix_out[nl[1], 0], ],
                            [matrix_out[nl[1], 1], ],
                            'x', color='crimson', ms=10)

        ax.set_axis_off()
        plt.subplots_adjust(wspace=0, hspace=0, left=0, top=1, bottom=0, right=1)

    plt.savefig(save_path + '.svg')
    plt.savefig(save_path + '.png')

    plt.close()


def f2m(vect, divider=3, n_keypoints=10):
    """flat2matrix_kexpoints"""
    if len(vect.shape) == 1:
        matrix = vect.reshape(n_keypoints, divider)
    elif len(vect.shape) == 2:
        matrix = vect.reshape(-1, n_keypoints, divider)
    elif len(vect.shape) == divider:
        matrix = vect.reshape((vect.shape[0], vect.shape[1], n_keypoints, divider))
    else:
        raise NotImplementedError(f'vect has to be of len 1, 2, or 3')
    return matrix


def m2f(matrix):
    """matrix2flat_keypoints"""
    if len(matrix.shape) == 3:
        return matrix.reshape(matrix.shape[0], matrix.shape[1] * matrix.shape[2])
    elif len(matrix.shape) <= 2:
        return matrix.flatten()
    elif len(matrix.shape) > 3:
        raise NotImplementedError(f'vect has to be of len 1, 2, or 3')


def compute_barycenter(points):
    """
    It is better if the head components, knees and ankles are removed
    :args points: numpy array (keypoints, 3D). Should be hip, coord and back.
              so we approximate the plane of the mouse back
    :returns: barycenter of given points (3D)
    """
    if len(points) > 0:
        return np.nanmean(points, axis=0)
    else:
        return np.nan


def compute_svd(points):
    """


    :args points: numpy array (keypoints, 3D). Should be hip, coord and back. 
                  so we approximate the plane of the mouse back 
    :returns: barycenter coordinates (numpy array of 3 elements)
              transition matrix (3, 3), 
              both can be in transform_points 
    """
    points = points[~np.any(np.isnan(points), axis=1)]
    if len(points) > 0:
        svd = np.linalg.svd((points - np.mean(points, axis=0)).T)
        barycenter = compute_barycenter(points)

        return barycenter, svd[0]

    else:
        return np.nan, np.nan


def transform_svd_points(points, barycenter, A):
    """
    Convert the coordinates in the new base formed by the barycenter and the transition matrix
    """
    return np.dot(A, (points - barycenter).T).T