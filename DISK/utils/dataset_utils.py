import time

import torch
from torch.utils import data
import numpy as np
from scipy.spatial.distance import pdist
import os
import importlib.util
import logging
from tqdm import tqdm

from DISK.utils.coordinates_utils import f2m, create_skeleton_plot, create_seq_plot
from DISK.utils.transforms import transform_x
from DISK.utils.utils import find_holes
from DISK.models.graph import Graph


def load_datasets(dataset_name, dataset_constants, suffix='', dataset_type='supervised', root_path='', **kwargs):
    """
    Folder structure: all pt files in the same subfolder in datasets

    X: (batch size, time, channels)
    y: (batch_size) with integer class
    lens: (batch_size) gives the input sequence length, the rest is filled with 0

    3 datasets: train, validation and test

    :param dataset_name: subfolder name
    :return: 3 torch datasets (train_dataset, val_dataset, test_dataset)
    """

    data_path = os.path.join(root_path, 'datasets', dataset_name)
    if dataset_type == 'supervised':
        train_dataset = SupervisedDataset(os.path.join(data_path, f'train_dataset{suffix}.npz'), dataset_constants, **kwargs)
        try:
            test_dataset = SupervisedDataset(os.path.join(data_path, f'test_dataset{suffix}.npz'), dataset_constants, **kwargs)
        except FileNotFoundError:
            test_dataset = None
        val_dataset = SupervisedDataset(os.path.join(data_path, f'val_dataset{suffix}.npz'),  dataset_constants, **kwargs)
    elif dataset_type == 'full_length':
        train_dataset = FullLengthDataset(os.path.join(data_path, f'train_fulllength_dataset{suffix}.npz'), dataset_constants, **kwargs)
        test_dataset = FullLengthDataset(os.path.join(data_path, f'test_fulllength_dataset{suffix}.npz'), dataset_constants, **kwargs)
        val_dataset = FullLengthDataset(os.path.join(data_path, f'val_fulllength_dataset{suffix}.npz'), dataset_constants, **kwargs)
    elif dataset_type == 'impute':
        train_dataset = ImputeDataset(os.path.join(data_path, f'train_fulllength_dataset{suffix}.npz'), dataset_constants, **kwargs)
        test_dataset = ImputeDataset(os.path.join(data_path, f'test_fulllength_dataset{suffix}.npz'), dataset_constants, **kwargs)
        val_dataset = ImputeDataset(os.path.join(data_path, f'val_fulllength_dataset{suffix}.npz'), dataset_constants, **kwargs)
    else:
        raise ValueError(f'[load_datasets function] argument dataset_type = {dataset_type} is not recognized. '
                         f'Authorized values are "supervised" or "full_length"')

    return train_dataset, val_dataset, test_dataset

class ParentDataset(data.Dataset):
    def __init__(self,
                 file: str,
                 dataset_constants: object,
                 transform: list,
                 outputdir: str,
                 skeleton_file: str,
                 *args,
                 label_type: str=None,
                 verbose: int = 0,
                 **kwargs
                 ):
        with np.load(file, allow_pickle=True) as data:
            self.data_dict = {key: data[key] for key in data.files}
        if 'y' in self.data_dict.keys():
            self.y = self.data_dict['y']
        else:
            self.y = None

        if 'ground_truth' in self.data_dict.keys():
            optipose_outlier = -4668
            self.X_gt = self.data_dict['ground_truth']
            self.X = self.data_dict['X']
            self.X[self.X == optipose_outlier] = np.nan
            self.X_gt[self.X_gt == optipose_outlier] = np.nan

            logging.info(f'[DATASET LOADER] {np.isnan(self.X[..., 0]).shape} {np.all(np.isnan(self.X[..., 0]), axis=2).shape} '
                         f'{np.any(np.all(np.isnan(self.X[..., 0]), axis=2), axis=1).shape} {np.any(np.all(np.isnan(self.X[..., 0]), axis=2), axis=1)}')

            mask_all_nans = np.any(np.all(np.isnan(self.X[..., 0]), axis=2), axis=1)
            mask_0_nans = np.any(np.all(~np.isnan(self.X[..., 0]), axis=2), axis=1)
            mask_nan_first_position = np.all(~np.isnan(self.X[:, 0, ..., 0]), axis=-1)
            logging.info(f'[MASKING] {np.sum(np.isnan(self.X[:, 0, ..., 0]))}')

            # self.X[~mask_nan_first_position, 0] = self.X_gt[~mask_nan_first_position, 0]
            mask_nan_first_position = np.all(~np.isnan(self.X[:, 0, ..., 0]), axis=-1)
            logging.info(f'[MASKING] {np.sum(np.isnan(self.X[:, 0, ..., 0]))}')
            logging.info(f'[AFTER MASKING] {self.X.shape} {self.X_gt.shape}')
            self.X = self.X[(~mask_all_nans) * (~mask_0_nans)]
            self.X_gt = self.X_gt[(~mask_all_nans) * (~mask_0_nans)]
            logging.info(f'[AFTER MASKING] {self.X.shape} {self.X_gt.shape}')

        else:
            self.X = self.data_dict['X']  # shape (batch, max_len, features)
            self.X_gt = None

        if 'files' in self.data_dict.keys():
            self.files = self.data_dict['files']
        else:
            self.files = None
        self.mask = self.get_mask()

        self.n_keypoints = dataset_constants.N_KEYPOINTS
        self.divider = dataset_constants.DIVIDER
        self.seq_length = dataset_constants.SEQ_LENGTH
        self.original_divider = self.divider + 1 if dataset_constants.W_RESIDUALS else self.divider
        if skeleton_file is not None:
            self.skeleton_graph = Graph(file=skeleton_file, strategy='uniform', max_hop=1, dilation=1)
        else:
            self.skeleton_graph = None

        if transform is None:
            print('[WARNING] NO TRANSFORM will be applied to the data. Is it really the behavior you are expecting??? '
                  'If not, check that you are passing a transform (without an s) list to the constructor.')
        if 'transforms' in kwargs:
            raise Warning('You are giving transforms (with an s) keyword argument. Expected: transform')
        self.transform = transform

        self.outputdir = outputdir
        self.label_type = label_type
        self.verbose = verbose

    def _get_sample(self, index: int):
        print('calling get_sample from ParentDataset')
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def get_mask(self):
        logging.info(f'[GET_MASK] {self.X.shape} {np.sum(np.isnan(self.X[:, 0]))}')
        mask = ~np.isnan(self.X)  # False when no coordinates available
        return mask

    def __getitem__(self, index: int, **kwargs):
        """

        :param index:
        :param kwargs:
        :return: 3 tensors (data, mask, seq_len)
        """
        sample = self._get_sample(index)
        timepoints = sample['x'].shape[0]

        x_coordinates = sample['x'].reshape((timepoints, self.n_keypoints, self.original_divider))[..., :self.divider]
        m = np.all(sample['m'].reshape((timepoints, self.n_keypoints, self.original_divider)), axis=-1)

        # Copy x for debug purpose and verify the transformation
        x_prev = np.array(x_coordinates)
        verbose_sample = self.verbose == 2 or (self.verbose == 1 and np.random.rand() < 1e-3)

        self.kwargs.update({'min_sample': self.min_per_sample[index],
                            'max_sample': self.max_per_sample[index],
                            'index': index,
                            'verbose_sample': verbose_sample,
                            'fillvalue_coordinates': 0,
                            'skeleton_graph': self.skeleton_graph})
        ## allowed transforms: rotation, translation, reflection, small gaussian noise on positions
        x_supp = sample['x_gt'] # x_gt or None
        if self.transform is not None and len(self.transform) > 0:
            # x has nans here
            x_coordinates, x_supp, self.kwargs = transform_x(x_coordinates, self.transform, x_supp=x_supp, **self.kwargs)

        if verbose_sample and self.skeleton_graph is not None:
            ### DEBUG
            import plotly.graph_objects as go

            plot_original = create_seq_plot(x_prev, self.skeleton_graph,
                                            max_n_display=6,
                                            cmap_name='Greens', name='Original')

            layout = go.Layout(scene=dict(aspectmode='cube'),  plot_bgcolor='rgb(255, 255, 255)',)
            fig = go.Figure(data=plot_original, layout=layout)
            fig.write_html(os.path.join(self.outputdir, f"data_original_{index}.html"))
            plot_x1 = create_seq_plot(x_coordinates, self.skeleton_graph,
                                      cmap_name='Blues', name='Transform',
                                      max_n_display=6)
            fig = go.Figure(data=plot_x1, layout=layout)
            fig.write_html(os.path.join(self.outputdir, f"data_transform_{index}.html"))

        """Fill the holes with the given placeholder"""  # transform the nans to 0s for the NN
        mask_holes = np.isnan(x_coordinates)[..., 0]  # True if hole, False else

        x_coordinates[mask_holes] = self.kwargs['fillvalue_coordinates']
        # here change nans of x_coordinates into 0 in final_x
        final_x = np.zeros((timepoints, self.n_keypoints, self.divider))
        final_x[..., :3] = x_coordinates

        # Typecasting
        final_x = torch.from_numpy(final_x).type(torch.float)
        m = torch.from_numpy(m).type(torch.int)
        new_m = torch.from_numpy(mask_holes).type(torch.int)

        z = torch.from_numpy(sample['z']).type(torch.long)
        torch_index = torch.from_numpy(np.array([index])).type(torch.long)
        torch_min = torch.from_numpy(self.kwargs['min_sample']).type(torch.float)
        torch_max = torch.from_numpy(self.kwargs['max_sample']).type(torch.float)

        output = dict(X=final_x,  # shape (timepoints, n_keypoints, 2 to 4)
                      original_mask=m,  # shape (timepoints, n_keypoints)
                      mask_holes=new_m,  # shape (timepoints, n_keypoints)
                      length_seq=z,
                      index=torch_index,
                      min_sample=torch_min,
                      max_sample=torch_max)

        if 'VI_angle' in self.kwargs:
            torch_angle = torch.from_numpy(np.array([self.kwargs['VI_angle']])).type(torch.float)
            torch_barycenter = torch.from_numpy(self.kwargs['VI_barycenter']).type(torch.float)
            output['VI_angle'] = torch_angle
            output['VI_barycenter'] = torch_barycenter

        if not x_supp is None:
            x_supp = torch.from_numpy(x_supp).type(torch.float)
            # normally sequence without additional holes but after the other transforms
            output['x_supp'] = x_supp
            # can be none or the original sample without holes
            ## FR: maybe not the best, but haven't found any better yet

        if 'i_file' in sample.keys():
            output['indices_file'] = sample['i_file']
            output['indices_pos'] = sample['i_pos']

        if self.label_type is not None and 'y' in sample and sample['y'] is not None:
            output['label'] = torch.from_numpy(sample['y']).type(torch.float)

        return output


class SupervisedDataset(ParentDataset):
    def __init__(self,
                 file: str,
                 dataset_constants: object,
                 transform: list = None,
                 skeleton_file: str = None,
                 outputdir: str = '',
                 verbose: int = 0,
                 **kwargs):
        super(SupervisedDataset, self).__init__(file, dataset_constants, transform, outputdir, skeleton_file, verbose)
        self.len_seq = self.data_dict['lengths']

        # compute an estimation of the average distance between keypoints of one pose, so when adding the gaussian noise
        # we add it proportionally
        if self.X_gt is not None:
            subsample = self.X_gt[np.random.choice(self.__len__(), min(1000, self.__len__()), replace=False)].reshape(
                (-1, self.n_keypoints, self.original_divider))
            max_dist_bw_keypoints = np.max(list(map(pdist, subsample)))
            self.max_dataset = np.max(self.X_gt.max(axis=(0, 1)).reshape((-1, self.original_divider)),
                                      axis=0)  # should be of shape divider (for the x, y, and z axes)
            self.min_dataset = np.max(self.X_gt.min(axis=(0, 1)).reshape((-1, self.original_divider)), axis=0)  # same
            self.min_per_sample = np.nanmin(
                self.X_gt.reshape(self.X_gt.shape[0], self.X_gt.shape[1], self.n_keypoints, self.original_divider), axis=(1, 2))
            self.max_per_sample = np.nanmax(
                self.X_gt.reshape(self.X_gt.shape[0], self.X_gt.shape[1], self.n_keypoints, self.original_divider), axis=(1, 2))
        else:
            subsample = self.X[np.random.choice(self.__len__(), min(1000, self.__len__()), replace=False)].reshape(
                (-1, self.n_keypoints, self.original_divider))
            max_dist_bw_keypoints = np.max(list(map(pdist, subsample)))
            self.max_dataset = np.max(self.X.max(axis=(0, 1)).reshape((-1, self.original_divider)),
                                      axis=0)  # should be of shape divider (for the x, y, and z axes)
            self.min_dataset = np.max(self.X.min(axis=(0, 1)).reshape((-1, self.original_divider)), axis=0)  # same
            self.min_per_sample = np.nanmin(
                self.X.reshape(self.X.shape[0], self.X.shape[1], self.n_keypoints, self.original_divider), axis=(1, 2))
            self.max_per_sample = np.nanmax(
                self.X.reshape(self.X.shape[0], self.X.shape[1], self.n_keypoints, self.original_divider), axis=(1, 2))

        self.kwargs = dict(min_coordinates_dataset=self.min_dataset,
                           max_coordinates_dataset=self.max_dataset,
                           max_dist_bw_keypoints=max_dist_bw_keypoints)
        self.kwargs.update(kwargs)

    def __len__(self):
        return self.X.shape[0]

    def _get_sample(self, index):
        m = self.mask[index]
        if self.y is not None:
            y = self.y[index: index + 1, :]
        else:
            y = None
        if self.X_gt is not None:
            x = self.X[index]
            x_gt = self.X_gt[index]
        else:
            x = self.X[index]
            x_gt = None
        z = self.len_seq[index: index + 1]
        sample = {'x': x,
                  'm': m,
                  'y': y,
                  'x_gt': x_gt,
                  'z': z}
        return sample


class FullLengthDataset(ParentDataset):

    def __init__(self,
                 file: str,
                 dataset_constants: object,
                 transform: list = None,
                 skeleton_file: str = None,
                 freq: int = 60,
                 length_sample: int = 120,  # in number of frames
                 stride: int = 2,  # in number of frames, int
                 outputdir='',
                 verbose=1,
                 **kwargs):

        super(FullLengthDataset, self).__init__(file, dataset_constants, transform, outputdir, skeleton_file,
                                                verbose, **kwargs)
        self.time = self.data_dict['time']  # shape (batch, max_len time)

        self.freq = freq
        self.stride = stride
        self.length_sample = length_sample

        self.kwargs = kwargs
        self.possible_indices = self.get_possible_indices()

        self.possible_times = []
        for index in self.possible_indices:
            self.possible_times.append(self.time[index[0], index[1]])

        self.min_per_sample = [np.nanmin(
            self.X[pi[0], pi[1]: pi[1] + self.length_sample].reshape(-1, self.n_keypoints, self.original_divider),
            axis=(0, 1))[..., :self.divider] for pi in self.possible_indices]
        self.max_per_sample = [np.nanmax(
            self.X[pi[0], pi[1]: pi[1] + self.length_sample].reshape(-1, self.n_keypoints, self.original_divider),
            axis=(0, 1))[..., :self.divider] for pi in self.possible_indices]

        subsample = self.X[self.time > -1]
        subsample = subsample[np.random.choice(len(subsample), min(10000, len(subsample)), replace=False)]\
            .reshape((-1, self.n_keypoints, self.original_divider))
        max_dist_bw_keypoints = np.max(list(map(pdist, subsample)))

        self.kwargs = dict(max_dist_bw_keypoints=max_dist_bw_keypoints)
        self.kwargs.update(kwargs)

    def get_possible_indices(self):
        logging.debug('calling get_possible_indices from FullLengthDataset')
        possible_indices = []
        for i_file, file_time in enumerate(self.time):
            file_time = file_time[file_time > -1]
            breakpoints = np.where(np.diff(file_time) > 1 / self.freq + 1e-9)[0]
            if len(breakpoints) == 0 or (len(breakpoints) > 0 and breakpoints[0] != 0):
                breakpoints = np.insert(breakpoints, 0, 0)  # add first point = index 0
            if -1 in file_time:
                end_point = np.where(file_time == -1)[0][0]
            else:
                # it is the max sequence
                end_point = len(file_time)
            breakpoints = np.insert(breakpoints, len(breakpoints), end_point)  # add first point = index 0
            # is the segment longer than our lower bound
            good_segments = np.where(np.diff(breakpoints) >= self.length_sample)[0]

            if len(good_segments) == 0:
                print('No long enough segments.')

            for index_good_segment in good_segments:
                good_times = np.arange(breakpoints[index_good_segment] + 1,
                                       breakpoints[index_good_segment + 1] - self.length_sample, self.stride,
                                       dtype=int)
                possible_indices.extend(np.vstack([[i_file] * len(good_times), good_times]).T)
        return np.array(possible_indices)  # should be (n, 2)

    def __len__(self):
        return len(self.possible_indices)

    def _get_sample(self, index):
        i_file = self.possible_indices[index, 0]
        i_pos = self.possible_indices[index, 1]
        x = self.X[i_file, i_pos: i_pos + self.length_sample]
        if self.y is not None:
            if len(self.y[i_file]) == len(self.time[i_file][self.time[i_file] > -1]):
                # case of MABe
                y = np.array([[self.y[i_file][(i_pos + i_pos + self.length_sample) // 2]]])
            else:
                y = np.array([self.y[i_file]])
        else:
            y = None
        m = self.mask[i_file, i_pos: i_pos + self.length_sample]
        z = np.array([self.length_sample])  # we add this to fit the other supervised dataset item format
        sample = {'x': x,
                  'm': m,
                  'z': z,
                  'y': y,
                  'i_file': i_file,
                  'i_pos': i_pos}
        return sample


class ImputeDataset(FullLengthDataset):

    def __init__(self,
                 file: str,
                 dataset_constants: object,
                 transform: list = None,
                 freq: int = 60,
                 skeleton_file: str = None,
                 length_sample: int = 120,  # in number of frames
                 stride: int = 2,  # in number of frames, int
                 padding: tuple = (1, 1),
                 outputdir: str = '',
                 verbose: int = 1, # 0, 1, or 2
                 all_segments: bool = False,
                 **kwargs):
        self.padding = padding
        self.all_segments = all_segments
        super(ImputeDataset, self).__init__(file, dataset_constants, transform, skeleton_file, freq, length_sample,
                                            stride, outputdir, verbose, **kwargs)

    def get_possible_indices(self):
        logging.debug('calling the ImputeDataset get_possible_indices')

        # some recordings are shorter than others, so find the real end point for each recording
        end_times = [np.where(t == -1)[0][0] if -1 in t else len(t) for t in self.time]
        n_imputed = 0
        n_total = 0
        possible_indices = []
        for recording in tqdm(range(self.X.shape[0])):

            # some recordings have gaps in the time
            breakpoints = np.where(np.diff(self.time[recording]) > 1 / self.freq + 1e-9)[0]
            breakpoints = np.insert(breakpoints, 0, 0)  # add first point = index 0
            breakpoints = np.insert(breakpoints, len(breakpoints),
                                    end_times[recording])  # add first point = index 0

            for i in range(len(breakpoints) - 1):
                start = breakpoints[i] + 1
                stop = breakpoints[i + 1]
                if stop <= start + self.padding[0] + self.padding[1]:
                    # not long enough to make a segment with the correct padding on the right and on the left
                    continue
                mask = np.any(np.isnan(self.X[recording]), axis=1)[start: stop]
                holes = find_holes(mask[:, np.newaxis], ['all'], indep=False, target_val=True)
                # logging.debug(f'holes {holes}')
                n_total += np.sum([h[1] for h in holes])

                for ihole in range(len(holes)):
                    if not self.all_segments and len(holes[ihole][1].split(' ')) > 1:
                        # case where we want only one keypoint missing, but several are missing
                        continue
                    # center segment on each hole
                    end = holes[ihole + 1][0] if ihole < len(holes) - 1 else len(mask)
                    begin = holes[ihole - 1][0] + holes[ihole - 1][1] + 1 if ihole >= 1 else 0

                    if holes[ihole][1] <= self.seq_length - self.padding[0] - self.padding[1] and \
                            end - holes[ihole][0] - holes[ihole][1] >= self.padding[1] and \
                            holes[ihole][0] - begin >= self.padding[0] and \
                            np.all(~mask[holes[ihole][0] - self.padding[0]: holes[ihole][0]]) and \
                            np.all(~mask[holes[ihole][0] + holes[ihole][1]:holes[ihole][0] + holes[ihole][1] +
                                                                           self.padding[1]]):

                        # we can make a segment
                        start_segment = holes[ihole][0] - self.padding[0]
                        stop_segment = holes[ihole][0] + holes[ihole][1] + self.padding[1]

                        if np.all(~mask[:holes[ihole][0] - self.padding[0]:]):
                            possible_extension_left = len(mask[:holes[ihole][0] - self.padding[0]][::-1])
                        else:
                            possible_extension_left = np.where(mask[:holes[ihole][0] - self.padding[0]][::-1])[0][0]
                        if np.all(~mask[holes[ihole][0] + holes[ihole][1] + self.padding[1]:]):
                            possible_extension_right = len(mask[holes[ihole][0] + holes[ihole][1] + self.padding[1]:])
                        else:
                            possible_extension_right = np.where(mask[holes[ihole][0] + holes[ihole][1] + self.padding[1]:])[0][0]

                        length_to_split = self.seq_length - (stop_segment - start_segment)

                        if possible_extension_left >= length_to_split // 2:
                            if possible_extension_right >= length_to_split // 2:
                                ## enough space, right and left, center because limitign factor in SEQ_LENGTH
                                start_segment = start_segment - length_to_split // 2
                                stop_segment = stop_segment + length_to_split // 2
                            else:
                                start_segment = start_segment - min(length_to_split // 2 - possible_extension_right,
                                                                    possible_extension_left)
                                stop_segment = stop_segment + possible_extension_right
                        else:
                            if possible_extension_right >= length_to_split // 2:
                                start_segment = start_segment - possible_extension_left
                                stop_segment = stop_segment + min(length_to_split // 2 - possible_extension_left,
                                                                  possible_extension_right)

                            else:
                                start_segment = start_segment - possible_extension_left
                                stop_segment = stop_segment + possible_extension_right

                        len_sample = stop_segment - start_segment
                        # logging.debug(f'{ihole}, {start_segment}, {stop_segment}, {len_sample}')
                        n_imputed += holes[ihole][1]
                        assert len_sample <= self.seq_length
                        assert np.all(~mask[start_segment: start_segment + self.padding[0]])
                        assert np.all(~mask[stop_segment - self.padding[1]: stop_segment])
                        if not np.all(np.isclose(
                                np.diff((self.time[recording, start + start_segment: start + stop_segment] * self.freq)),
                                1)):
                            print('Problem in segment formation, gap in time')
                            print(self.time[recording, start + start_segment: start + stop_segment] * self.freq)
                            print(recording, start, start_segment, stop_segment, breakpoints)
                            import sys
                            sys.exit(1)

                        sample_x = np.zeros((self.seq_length, self.n_keypoints, self.divider))
                        sample_x[:len_sample] = self.X[recording, start + start_segment: start + stop_segment].reshape(
                            (len_sample, self.n_keypoints, self.original_divider))[..., :self.divider]

                        possible_indices.append([recording, start + start_segment, len_sample])

        if n_imputed == 0:
            return np.array([])

        logging.info(
            f'Found {n_imputed} imputable timepoints over the {n_total} total missing timepoints '
            f'({n_imputed / n_total * 100:.1f} %)')
        logging.info(f'Lengths of imputable segments (25th, 50th, 75th percentiles): '
                     f'{np.percentile(np.array(possible_indices)[:, -1], (25, 50, 75))}')
        return np.array(possible_indices)

    def __len__(self):
        return len(self.possible_indices)

    def _get_sample(self, index):
        i_file = self.possible_indices[index, 0]
        i_pos = self.possible_indices[index, 1]
        len_ = self.possible_indices[index, 2]

        x = np.ones((self.seq_length, self.n_keypoints * self.original_divider)) * np.nan
        x[:len_] = self.X[i_file, i_pos: i_pos + len_]
        m = np.zeros((self.seq_length, self.n_keypoints * self.original_divider), dtype=bool)
        m[:len_] = self.mask[i_file, i_pos: i_pos + len_] # False when missing
        logging.debug(f'[WARNING] Updating {i_file} at pos {i_pos + 1}: {i_pos + len_} '
                     f'with vector with {np.sum(np.isnan(x[:len_])) // 4} NaN')
        if not np.all(~np.isnan(x[0])):
            raise ValueError(f'no missing data in sample')
        z = np.array([len_])  # we add this to fit the other supervised dataset item format
        sample = {'x': x,
                  'm': m,
                  'z': z,
                  'i_file': i_file,
                  'i_pos': i_pos,
                  'index': index}
        return sample

    def update_dataset(self, index, new_x, uncertainty=None, threshold=0):
        idx = index
        new_x_np = new_x.reshape(new_x.shape[0], new_x.shape[1], -1)
        i_file = self.possible_indices[idx, 0]
        i_pos = self.possible_indices[idx, 1]
        len_ = self.possible_indices[idx, 2]

        try:
            len(i_file)
            pass
        except Exception:
            i_file = [[i_file], ]
            i_pos = [[i_pos], ]
            len_ = [[len_], ]

        for ii in range(len(i_file)):
            m = self.mask[i_file[ii][0]][i_pos[ii][0]: i_pos[ii][0] + len_[ii][0]]

            if uncertainty is not None:
                unc = np.sum(uncertainty[ii, :len_[ii][0]]) / np.sum(m)
            else:
                unc = None

            if unc is None or unc <= threshold:
                logging.debug(
                f'[WARNING] Updating {i_file[ii][0]} at pos {i_pos[ii][0]}: {i_pos[ii][0] + len_[ii][0]} '
                f'with vector with {np.sum(m)} NaN and uncertainty {unc}')
                if self.original_divider == self.divider:
                    self.X[i_file[ii][0], i_pos[ii][0]: i_pos[ii][0] + len_[ii][0]][~m] = new_x_np[ii][:len_[ii][0]][~m]
                    self.mask[i_file[ii][0], i_pos[ii][0]: i_pos[ii][0] + len_[ii][0]][~m] = True
                else:  # only for Qualisys Mocap
                    dims = [i for i in range(self.X.shape[-1]) if i != 0 and i % self.divider == 0]
                    self.X[i_file[ii][0], i_pos[ii][0]: i_pos[ii][0] + len_[ii][0], dims][~m] = new_x_np[ii][:len_[ii][0]][~m]
                    self.mask[i_file[ii][0], i_pos[ii][0]: i_pos[ii][0] + len_[ii][0]][~m] = True
                    self.X[i_file[ii][0], i_pos[ii][0]: i_pos[ii][0] + len_[ii][0], self.divider::self.original_divider][~m] = 2
            else:
                logging.info(
                f'[WARNING] NOT Updating {i_file[ii][0]} at pos {i_pos[ii][0]}: {i_pos[ii][0] + len_[ii][0]} '
                f'with vector with {np.sum(m)} NaN and uncertainty {unc}')


