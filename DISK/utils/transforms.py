import numpy as np
import plotly.graph_objects as go
import os
import pandas as pd
import torch
import logging

from DISK.utils.coordinates_utils import create_skeleton_plot, compute_svd


def init_transforms(_cfg, keypoints, divider, length_input_seq, basedir, outputdir, add_missing=True):
    transforms = []

    if 'add_missing' in _cfg.feed_data.transforms.keys():
        length_proba_df = pd.read_csv(os.path.join(basedir, 'datasets', _cfg.feed_data.transforms.add_missing.files[1]))
        if 'length' not in length_proba_df.columns:
            raise ValueError(f'No "length" column in file {_cfg.feed_data.transforms.add_missing.files[1]}.')
        length_proba_df['length'] = length_proba_df['length'].astype('int')
        length_proba_df['keypoint'] = length_proba_df['keypoint'].astype('str')
        if _cfg.feed_data.transforms.add_missing.files[0].endswith('.txt'):
            init_proba = np.loadtxt(os.path.join(basedir, 'datasets', _cfg.feed_data.transforms.add_missing.files[0]))
            init_proba_df = pd.DataFrame(columns=('keypoint',), data=keypoints)
            init_proba_df.loc[:, 'proba'] = init_proba
        elif _cfg.feed_data.transforms.add_missing.files[0].endswith('.csv'):
            init_proba_df = pd.read_csv(os.path.join(basedir, 'datasets', _cfg.feed_data.transforms.add_missing.files[0]),
                                        dtype={'keypoint': str})
        else:
            raise ValueError('[init_transforms] First missing file should be a txt file or a csv file with a valid '
                             'extension')

        logging.info(f'INIT TRANSFORMS, {init_proba_df["keypoint"].unique()} {length_proba_df["keypoint"].unique()}')
        indep_keypoints = False if 'set_keypoint' in _cfg.feed_data.transforms.add_missing.files[1] else True

        if len(_cfg.feed_data.transforms.add_missing.files) > 2:
            proba_n_missing = np.loadtxt(
                os.path.join(basedir, 'datasets', _cfg.feed_data.transforms.add_missing.files[2]))
        else:
            proba_n_missing = None

        if add_missing:
            addmissing_transform = AddMissing_LengthProba(length_proba_df, keypoints, init_proba_df, divider=divider,
                                                          proba_n_missing=proba_n_missing,
                                                          indep_keypoints=indep_keypoints,
                                                          pad=_cfg.feed_data.transforms.add_missing.pad,
                                                          verbose=0, proba=1, outputdir=outputdir)
            transforms.append(addmissing_transform)

    if _cfg.feed_data.transforms.viewinvariant:
        transforms.append(ViewInvariant(proba=1, divider=divider, verbose=0, index_frame=int(length_input_seq / 2),
                                        outputdir=outputdir))
    if _cfg.feed_data.transforms.normalize:
        transforms.append(Normalize(proba=1, divider=divider, verbose=0, outputdir=outputdir))
    if _cfg.feed_data.transforms.normalizecube:
        transforms.append(NormalizeCube(proba=1, divider=divider, verbose=0, outputdir=outputdir))

    return transforms, proba_n_missing


class Transform(object):
    """
    The transforms are applied to input tensor of shape (timepoints, n_keypoints * 3D in format xyzxyz...)
    The same exact transform (same parameters) needs to be applied to the input tensor
    because it corresponds to one sequence, one movement
    """

    def __init__(self, proba, divider, verbose=0, outputdir='', **kwargs):
        self.proba = proba
        self.verbose = verbose
        self.outputdir = outputdir
        self.divider = divider

    @staticmethod
    def apply_transform(x, *args):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        ### NB: I added this option for x_supp where the transform is calculated on x and applied on x and x_supp
        raise NotImplementedError

    def untransform(self, x, *args, **kwargs):
        raise NotImplementedError


class ViewInvariant(Transform):
    """
    See: Xia, H., & Gao, X. (2021). Multi-scale mixed dense graph convolution network for skeleton-based action
         recognition. IEEE Access, 9, 36475–36484. https://doi.org/10.1109/ACCESS.2020.3049029

    This only applies a rotation in the X-Y plane. No norm transformation is carried.

    IMPORTANT DIFFERENCE BETWEEN ON THE GROUND AND CLIMBING SEQUENCES:
    - for the climbing sequences, the 3rd SVD vector is chosen to align (the one perpendicular to the back) instead
    of the 1st one
    - indeed, when the mouse is standing, the main vector of the SVD, the one along the back has only a small component
    on the XY plane, which makes the projection quite noisy and unstable.
    - the output transformed sequences should have the mouse main axis aligned with the y=0 axis and face the increasing
    x numbers - to continue to check
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.index_frame = kwargs['index_frame']

    def __str__(self):
        return 'ViewInvariant'

    def compute_transform(self, x):
        """Compute the transform"""
        # indices_for_barycenter = np.arange(0, 6, dtype=int) # back, coord, & hips # changed 2022-07-07
        # x is of shape (timepoints, n_keypoints , 3)
        if x.shape[0] <= self.index_frame:
            idx = x.shape[0] - 1
        else:
            idx = self.index_frame
        points = x[idx, :, :]
        mask_na_points = np.any(np.isnan(points), axis=1)
        if np.all(mask_na_points):
            idces = np.where(np.sum(np.isnan(x), axis=(1, 2)) == 0)[0]
            idx = idces[np.argmin(np.abs(idces - self.index_frame))]
            points = x[idx, :, :]
            mask_na_points = np.any(np.isnan(points), axis=1)
        points = points[~mask_na_points]

        barycenter, A = compute_svd(points)

        max_component_in_A = np.argmax(np.abs(A), axis=0)
        # choosing the vector which has the max component with our first_axis:
        if max_component_in_A[0] == 2:
            # the mouse is standing or climbing
            index_vect = 2
        else:
            # the mouse is on the ground, the main axis is the vector we need
            index_vect = 0
        vectx = A[:, index_vect]
        first_axis = np.array([1, 0, 0])
        # the angle with cos is the absolute measure of theta without a direction, only the tan calculus is giving the direction
        angle = np.arctan2(first_axis[1], first_axis[0]) - np.arctan2(vectx[1], vectx[0])
        return barycenter, A, index_vect, angle

    @staticmethod
    def apply_transform(x, barycenter, angle):
        if x is None:
            return None
        x_norm = np.copy(x) - barycenter
        x_prime = np.copy(x) - barycenter

        # modify the x coordinates
        x_prime[:, :, 0] = x_norm[:, :, 0] * np.cos(angle) - x_norm[:, :, 1] * np.sin(angle)
        # modify the y coordinates
        x_prime[:, :, 1] = x_norm[:, :, 0] * np.sin(angle) + x_norm[:, :, 1] * np.cos(angle)

        x_tmp = np.copy(x) - barycenter
        x_tmp = torch.from_numpy(x_tmp)
        x_prime_torch = torch.from_numpy(x_prime)
        angle_torch = torch.Tensor([angle]).type(torch.float)
        x_tmp[:, :, 0] = torch.cos(angle_torch) * x_prime_torch[:, :, 0] + torch.sin(angle_torch) * x_prime_torch[:, :, 1]
        x_tmp[:, :, 1] = - torch.sin(angle_torch) * x_prime_torch[:, :, 0] + torch.cos(angle_torch) * x_prime_torch[:, :, 1]

        return x_prime

    def make_visualization(self, x, barycenter, A, index_vectx, x_prime, **kwargs):
        a = create_skeleton_plot(x[self.index_frame], kwargs['skeleton_graph'], color='darkblue', name='Original')
        names = ['chosen' if index_vectx == i else 'not' for i in range(3)]
        scale = (kwargs['max_sample'] - kwargs['min_sample'])/2
        if self.divider > 2:
            a.append(go.Scatter3d(x=[barycenter[0], barycenter[0] + scale[0] * A[0, 0]],
                                    y=[barycenter[1], barycenter[1] + scale[0] * A[1, 0]],
                                    z=[barycenter[2], barycenter[2] + scale[0] * A[2, 0]],
                                    line=dict(color='blue', width=2),
                                    name=names[0]))
            a.append(go.Scatter3d(x=[barycenter[0], barycenter[0] + scale[1] * A[0, 1]],
                                    y=[barycenter[1], barycenter[1] + scale[1] * A[1, 1]],
                                    z=[barycenter[2], barycenter[2] + scale[1] * A[2, 1]],
                                    line=dict(color='orange', width=2),
                                    name=names[1]))
            a.append(go.Scatter3d(x=[barycenter[0], barycenter[0] + scale[2] * A[0, 2]],
                                    y=[barycenter[1], barycenter[1] + scale[2] * A[1, 2]],
                                    z=[barycenter[2], barycenter[2] + scale[2] * A[2, 2]],
                                    line=dict(color='green', width=2),
                                    name=names[2]))
            layout = go.Layout(scene=dict(aspectmode='cube',
                                          xaxis=dict(
                                             backgroundcolor="rgba(0, 0, 0,0)",
                                             gridcolor="white",
                                             showbackground=True,
                                             zerolinecolor="white",),
                                          yaxis=dict(
                                            backgroundcolor="rgba(0, 0, 0,0)",
                                            gridcolor="white",
                                            showbackground=True,
                                            zerolinecolor="white"),
                                          zaxis=dict(
                                            backgroundcolor="rgba(0, 0, 0,0)",
                                            gridcolor="white",
                                            showbackground=True,
                                            zerolinecolor="white",),))
        else:
            a.append(go.Scatter(x=[barycenter[0], barycenter[0] + scale[0] * A[0, 0]],
                                    y=[barycenter[1], barycenter[1] + scale[0] * A[1, 0]],
                                    line=dict(color='blue', width=2),
                                    name=names[0]))
            a.append(go.Scatter(x=[barycenter[0], barycenter[0] + scale[1] * A[0, 1]],
                                    y=[barycenter[1], barycenter[1] + scale[1] * A[1, 1]],
                                    line=dict(color='orange', width=2),
                                    name=names[1]))
            a.append(go.Scatter(x=[barycenter[0], barycenter[0] + scale[2] * A[0, 2]],
                                    y=[barycenter[1], barycenter[1] + scale[2] * A[1, 2]],
                                    line=dict(color='green', width=2),
                                    name=names[2]))
            layout = go.Layout(scene=dict(aspectmode='cube',
                                          xaxis=dict(
                                             backgroundcolor="rgba(0, 0, 0,0)",
                                             gridcolor="white",
                                             showbackground=True,
                                             zerolinecolor="white",),
                                          yaxis=dict(
                                            backgroundcolor="rgba(0, 0, 0,0)",
                                            gridcolor="white",
                                            showbackground=True,
                                            zerolinecolor="white"),))

        fig = go.Figure(data=a, layout=layout)
        fig.write_html(os.path.join(self.outputdir, f"data_before_view_invariant_{kwargs['index']}.html"))

        a1 = create_skeleton_plot(x_prime[self.index_frame, :].reshape(-1, self.divider), kwargs['skeleton_graph'],
                                  color='darkblue', name=f'Transformed index {index_vectx}')
        if self.divider > 2:
            a1.append(go.Scatter3d(x=[0, scale[0]], y=[0, 0],  z=[0, 0],  line=dict(color='blue', width=2)))
            a1.append(go.Scatter3d(x=[0, 0],  y=[0, scale[1]], z=[0, 0],  line=dict(color='orange', width=2)))
            a1.append(go.Scatter3d(x=[0, 0],  y=[0, 0],  z=[0, scale[2]], line=dict(color='green', width=2)))
        else:
            a1.append(go.Scatter(x=[0, scale[0]], y=[0, 0],line=dict(color='blue', width=2)))
            a1.append(go.Scatter(x=[0, 0],  y=[0, scale[1]], line=dict(color='orange', width=2)))
            a1.append(go.Scatter(x=[0, 0],  y=[0, 0], line=dict(color='green', width=2)))
        fig = go.Figure(data=a1, layout=layout)
        fig.write_html(os.path.join(self.outputdir, f"data_after_view_invariant_{kwargs['index']}.html"))

        fig = go.Figure(data=a[:-3], layout=layout)
        fig.write_html(os.path.join(self.outputdir, f"data_before_view_invariant_no_basis_{kwargs['index']}.html"))


    def __call__(self, x, *args, x_supp=None, **kwargs):

        barycenter, A, index_vect, angle = self.compute_transform(x)

        """ Apply the transform """
        # x and x_prime is with holes if any
        x_prime = self.apply_transform(x, barycenter, angle)
        # x_supp_prime has no holes if any, if not, not used
        x_supp_prime = self.apply_transform(x_supp, barycenter, angle)

        if (kwargs['verbose_sample'] or self.verbose == 2) and kwargs['skeleton_graph'] is not None:
            ### visualization for debugging puposes
            self.make_visualization(x, barycenter, A, index_vect, x_prime, **kwargs)

        kwargs['VI_barycenter'] = barycenter
        kwargs['VI_angle'] = angle
        ## update min and max sample in the kwargs dict after ViewInvariant
        max_ = np.nanmax(x_prime, axis=(0, 1))  # should be of shape 3 (for the x, y, and z axes)
        min_ = np.nanmin(x_prime, axis=(0, 1))  # same
        kwargs['min_sample'] = min_
        kwargs['max_sample'] = max_

        if np.all(np.isnan(x_prime)):
            print('[ViewInvariant] all nan in x_prime')

        return x_prime, x_supp_prime, kwargs

    def untransform(self, x, *args, **kwargs):
        ## then un-View Invariant

        if torch.is_tensor(kwargs['VI_barycenter']):
            angle = kwargs['VI_angle'].detach().cpu().numpy()
            barycenter = kwargs['VI_barycenter'].detach().cpu().numpy()
        else:
            angle = kwargs['VI_angle']
            barycenter = kwargs['VI_barycenter']

        if len(x.shape) == 3:  # time, keypoints, 3D or 2D

            x_norm = np.array(x)
            x_prime = np.array(x)
            barycenter = np.tile(np.tile(barycenter, x.shape[1]), x.shape[0]).reshape(x.shape)
            x_prime[:, :, 0] = np.cos(angle) * x_norm[:, :, 0] + np.sin(angle) * x_norm[:, :, 1]
            x_prime[:, :, 1] = - np.sin(angle) * x_norm[:, :, 0] + np.cos(angle) * x_norm[:, :, 1]
            x_prime += barycenter

            x_prime[x == 0] = 0

            return x_prime

        elif len(x.shape) == 4:  # batch, time, keypoints, 2D or 3D

            x_norm = np.array(x)
            x_prime = np.array(x)

            angle_tiled = np.tile(np.tile(angle,  x.shape[2]), x.shape[1]).reshape(x.shape[:-1])
            barycenter = np.tile(np.tile(kwargs['VI_barycenter'], x.shape[2]), x.shape[1]).reshape(x.shape)
            #print(barycenter[0, 0, 0, :], barycenter[0, 0, :2, 0], barycenter[0, :2, 0, 0])

            x_prime[:, :, :, 0] = np.cos(angle_tiled) * x_norm[:, :, :, 0] + np.sin(angle_tiled) * x_norm[:, :, :, 1]
            x_prime[:, :, :, 1] = - np.sin(angle_tiled) * x_norm[:, :, :, 0] + np.cos(angle_tiled) * x_norm[:, :, :, 1]
            x_prime += barycenter

            x_prime[x == 0] = 0

            return x_prime

        else:
            raise ValueError


class NormalizeCube(Transform):
    """
    See: https://github.com/shlizee/Predict-Cluster/blob/master/ucla_demo.ipynb
    normalize_ucla_data function

    Normalization per sample, outputs between -1 and 1
    for the 3 axes.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __str__(self):
        return 'Normalize_Cube'

    def __call__(self, x, *args, x_supp=None, **kwargs):
        """Compute the transform"""
        # x of shape (time points, keypoints,  3)
        max_ = np.nanmax(x, axis=(0, 1))  # should be of shape 3 (for the x, y, and z axes)
        min_ = np.nanmin(x, axis=(0, 1))  # same
        amplitude = np.max(max_ - min_)  # same for every axis
        kwargs['min_sample'] = min_
        kwargs['max_sample'] = max_
        if np.any(np.isnan(min_)) or np.any(np.isnan(max_)):
            print(f'[Problem in NormalizeCube] {min_}, {max_}, {x}')

        """Apply the transform"""
        x_prime = 2 * (x - ((max_ + min_) / 2)) / amplitude  # normalizes between -1 and 1
        if x_supp is None:
            x_supp_prime = None
        else:
            x_supp_prime = 2 * (x_supp - ((max_ + min_) / 2)) / amplitude  # normalizes between -1 and 1

        return x_prime, x_supp_prime, kwargs

    def untransform(self, x, *args, **kwargs):
        if torch.is_tensor(kwargs['min_sample']):
            min_sample = kwargs['min_sample'].detach().cpu().numpy()
            max_sample = kwargs['max_sample'].detach().cpu().numpy()
        else:
            min_sample = kwargs['min_sample']
            max_sample = kwargs['max_sample']

        if len(x.shape) == 3:  # time, keypoints, 3D or 2D

            min_tiled = np.tile(np.tile(min_sample, x.shape[1]), x.shape[0]).reshape(x.shape)#[np.newaxis], (x.shape[0], 1, 1))
            max_tiled = np.tile(np.tile(max_sample, x.shape[1]), x.shape[0]).reshape(x.shape)

            amplitude = np.max(max_sample - min_sample) # scalar

        elif len(x.shape) == 4:
            # x shape: batch, time, keypoints, 2D or 3D
            # min_sample shape: batch, 2D or 3D
            min_tiled = np.array([np.tile(np.tile(m,  x.shape[2]), x.shape[1]) for m in min_sample]).reshape(x.shape)
            max_tiled = np.array([np.tile(np.tile(m,  x.shape[2]), x.shape[1]) for m in max_sample]).reshape(x.shape)
            # max_tiled = np.tile(np.tile(max_sample,  x.shape[2]), x.shape[1]).reshape(x.shape)
            #print('min_Tiled', min_tiled[0, 0, :2, 0], min_tiled[0, 0, 0, :])

            amplitude = np.repeat(np.repeat(np.repeat(np.max(max_sample - min_sample, axis=1), x.shape[3]),  x.shape[2]), x.shape[1]).reshape(x.shape)
            #print('amplitude', amplitude[0, 0, 0, :], amplitude[0, 0, :2, 0])
        else:
            raise ValueError

        # same for every axis
        reconstructed = amplitude / 2 * x + (max_tiled + min_tiled) / 2

        return reconstructed

class Normalize(Transform):
    """
    See: https://github.com/shlizee/Predict-Cluster/blob/master/ucla_demo.ipynb
    normalize_ucla_data function

    Normalization per sample, outputs between -1 and 1
    for the 3 axes.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __str__(self):
        return 'Normalize'

    def __call__(self, x, *args, x_supp=None, **kwargs):
        """Compute the transform"""
        # x of shape (time points, keypoints,  3)
        max_ = np.nanmax(x, axis=(0, 1))  # should be of shape 3 (for the x, y, and z axes)
        min_ = np.nanmin(x, axis=(0, 1))  # same
        kwargs['min_sample'] = min_
        kwargs['max_sample'] = max_
        if np.any(np.isnan(min_)) or np.any(np.isnan(max_)):
            print(f'[Problem in Normalize] {min_}, {max_}, {x}')

        """Apply the transform"""
        x_prime = 2 * (x - min_) / (max_ - min_) - 1  # normalizes between -1 and 1
        if x_supp is None:
            x_supp_prime = None
        else:
            x_supp_prime = 2 * (x_supp - min_) / (max_ - min_) - 1  # normalizes between -1 and 1

        return x_prime, x_supp_prime, kwargs

    def untransform(self, x, *args, **kwargs):
        if torch.is_tensor(kwargs['min_sample']):
            min_sample = kwargs['min_sample'].detach().cpu().numpy()
            max_sample = kwargs['max_sample'].detach().cpu().numpy()
        else:
            min_sample = kwargs['min_sample']
            max_sample = kwargs['max_sample']

        if len(x.shape) == 3:  # time, keypoints, 3D or 2D

            min_tiled = np.tile(np.tile(min_sample, x.shape[1]), x.shape[0]).reshape(x.shape)
            max_tiled = np.tile(np.tile(max_sample, x.shape[1]), x.shape[0]).reshape(x.shape)

        elif len(x.shape) == 4:  # batch, time, keypoints, 2D or 3D
            min_tiled = np.tile(np.tile(min_sample, x.shape[2]), x.shape[1]).reshape(x.shape)
            max_tiled = np.tile(np.tile(max_sample, x.shape[2]), x.shape[1]).reshape(x.shape)
        else:
            raise ValueError

        reconstructed = min_tiled + (max_tiled - min_tiled) * (1 + x) / 2

        return reconstructed



class AddMissing_LengthProba(Transform):
    ### FR - 2022-06-07:
    ### should be first transform, it is not checked automatically though
    ### this is implemented as a transform because
    ### - it needs to be applied first before other normlization to  not leak data through the normalization
    ### - it needs to be applied differently for each sample at each epoch

    def __init__(self, length_proba_df, list_keypoints, init_proba_df, indep_keypoints=True, proba_n_missing=None, pad=(0, 0),
                 **kwargs):
        self.length_proba_df = length_proba_df
        self.init_proba_df = init_proba_df
        self.list_keypoints = list_keypoints
        self.pad_before = max(int(pad[0]), 0)  # if 0, means we can alter all time points, if > 0 gives the number of frames untouched at the beginning and end
        self.pad_after = max(int(pad[1]), 0)  # if 0, means we can alter all time points, if > 0 gives the number of frames untouched at the beginning and end
        self.proba_n_missing = proba_n_missing
        self.indep_keypoints = indep_keypoints
        self.cumsum_proba_n_missing = np.cumsum(self.proba_n_missing)

        super().__init__(**kwargs)

    def __call__(self, x, *args, verbose_sample=False, **kwargs):
        # x of shape (time points, keypoints, 3 or 4)
        x_with_holes = np.array(x)

        if np.max(np.sum(np.any(np.isnan(x), axis=2), axis=1)) > 0:
            if self.verbose == 2 or verbose_sample:
                print('[AddMissing Transform] There is already a missing keypoint in the sequence. Not adding more')
        else:
            # missing value place holder
            missing_values_placeholder = np.nan
            # while not np.any(np.sum(np.isnan(x_with_holes), axis=(1, 2))):
            # for now only one hole per sample
            if self.proba_n_missing is None:
                n_missing = 1
            else:
                n_missing = np.where(np.random.rand() <= self.cumsum_proba_n_missing)[0][0] + 1

            if not self.indep_keypoints:
                ## in the proba file there should be probability of missing sets of keypoints
                ## so we don't draw missing keypoint one by one but directly a set
                ## this should be activated only when more than 1 keypoint is missing at a time

                buffer = int(self.pad_before)
                while buffer < x.shape[0] - self.pad_after:
                    ## choose id of missing keypoints
                    rd_kp = np.random.choice(a=self.init_proba_df['keypoint'],
                                             size=1,
                                             p=self.init_proba_df['proba'].values,
                                             replace=False)[0]

                    ## choose length for the keypoint set
                    length_df = self.length_proba_df.loc[self.length_proba_df['keypoint'] == rd_kp, :].sample(n=1, weights='proba')
                    length_input = length_df['length'].values[0]
                    ## verify it's not too long
                    length_input = min(length_input, x.shape[0] - buffer - self.pad_after)

                    ## chosen first index indep per keypoint
                    inter_lengths = np.fmin(self.length_proba_df.loc[self.length_proba_df['keypoint'] == 'non_missing', :].sample(n=1, weights='proba')['length'].values[0],
                                            x.shape[0] - self.pad_after - length_input - buffer)

                    start_missing = buffer + inter_lengths
                    end_missing = start_missing + length_input
                    buffer = end_missing

                    for missing_kp_index in rd_kp.split(' '):
                        x_with_holes[start_missing: end_missing, self.list_keypoints.index(missing_kp_index), :] = missing_values_placeholder

            else:
                # all the keypoints are considered independent

                buffer = self.pad_before
                while buffer < x.shape[0] - self.pad_after:

                    ## choose id of missing keypoints
                    rd_kp = np.random.choice(a=self.init_proba_df['keypoint'],
                                             size=n_missing,
                                             p=self.init_proba_df['proba'].values,
                                             replace=False)  # shape: n_missing

                    ## choose length per keypoint
                    length_df = self.length_proba_df.groupby('keypoint').sample(n=1, weights='proba')
                    length_input = np.random.choice(length_df.loc[length_df['keypoint'].isin(rd_kp), 'length'].values, 1)[0]
                    ## verify it's not too long
                    lengths = np.fmin(length_input, x.shape[0] - buffer - self.pad_after)  # shape: n_missing

                    ## chosen first index indep per keypoint
                    inter_lengths = np.fmin(length_df.loc[length_df['keypoint'] == 'non_missing', 'length'].values,
                                            x.shape[0] - self.pad_after - buffer - lengths)[0]

                    start_missing = buffer + inter_lengths
                    end_missing = start_missing + lengths
                    buffer = int(end_missing)

                    index_rd_kp = self.list_keypoints.index(rd_kp)
                    x_with_holes[start_missing: end_missing, index_rd_kp, :] = missing_values_placeholder

            if self.verbose == 2 or verbose_sample:
                print("nb of missing kp:", np.sum(np.sum(np.any(np.isnan(x_with_holes), axis=2), axis=0) > 0))
            v = np.sum(np.isnan(x_with_holes[..., 0]))
            if v == 0:
                print("nb of missing values:", v)

        return x_with_holes


def transform_x(x, transformations, **kwargs):
    '''

    :param x: can have nan in the places where coordinates is missing
    :param transformations:
    :param kwargs:
    :return:
    '''
    x_supp = None
    if isinstance(transformations[0], AddMissing_LengthProba):
        x_supp = np.copy(x)  # the supp sample is the one without holes, but other reflection, normalization, ...
        # will be computed on x and applied both on x_supp and x
        x = transformations[0](x, **kwargs)  # the main sample is the one with holes
        # in the case, where no hole is added, x is original x, and x_supp is None
        for t in transformations[1:]:
            x, x_supp, kwargs = t(x, x_supp=x_supp, **kwargs)
    else:
        for t in transformations:
            if 'x_supp' in kwargs:
                print(t)
            x, x_supp, kwargs = t(x, **kwargs)
    return x, x_supp, kwargs


def reconstruct_before_normalization(data, data_dict, transforms):

    for transform in transforms[::-1]:
        if isinstance(transform, AddMissing_LengthProba):
            continue
        data = transform.untransform(data, **data_dict)
    return data
