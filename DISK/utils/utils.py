import matplotlib.pyplot as plt
import importlib.util
import os
import numpy as np
import math
import logging
import torch
import time


def plot_training(df, offset=10, print_every=1):
    x = np.arange(df.shape[0]) * print_every
    fig, axes = plt.subplots(2, 1, sharex='all')
    axes[0].plot(x[offset:], df[0][offset:], label='train')
    axes[0].plot(x[offset:], df[2][offset:], label='validation')
    axes[0].legend()
    axes[0].set_title('Loss')
    axes[1].plot(x[offset:], df[1][offset:])
    axes[1].plot(x[offset:], df[3][offset:])
    axes[1].set_title('RMSE')


def plot_history(history, every):
    fig, axes = plt.subplots(2, 1, sharex='all')
    x_train = np.arange(len(history['loss']))
    x_val = np.arange(0, len(history['loss']), every)
    pl_train = axes[0].plot(x_train, history['loss'])
    pl_val = axes[0].plot(x_val, history['val_loss'])
    axes[0].set_title('Loss')
    axes[1].plot(x_train, history['accuracy'], c=pl_train[0].get_color())
    axes[1].plot(x_val, history['val_accuracy'], c=pl_val[0].get_color())
    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Epochs')

def save_checkpoint(model, epoch, optimizer, dict_, PATH):
    torch.save({
      'epoch': epoch,
      'model_state_dict': model.state_dict(),
      'optimizer_state_dict': optimizer.state_dict(),
      **dict_
    }, PATH)


def load_checkpoint(model, optimizer, PATH, device):
    data = torch.load(PATH, map_location=torch.device(device))
    model.load_state_dict(data['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(data['optimizer_state_dict'])
        optimizer.__setattr__('lr', data['lr'])
    for key in data.keys():
        if key in ['model_state_dict', 'optimizer_state_dict']:
            continue
        logging.info(f'Loading with {key} = {data[key]}')
        if key == 'print_every':
            print_every = data['print_every']
        else:
            print_every = 1
    return data['epoch'], print_every


def asMinutes(s):
    if s >= 61 * 60:
        h = math.floor(s / 60 / 60)
        s -= h * 60 * 60
        m = math.floor(s / 60)
        s -= m * 60
        return '%dh %02dm %02ds' % (h, m, s)
    else:
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %02ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    # time since beginning:
    s = now - since
    # estimated time per percent:
    if percent != 0:
        es = s / percent
    else:
        es = 0
    # estimated remaining time:
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def read_constant_file(constant_file):
    """import constant file as a python file from its path"""
    spec = importlib.util.spec_from_file_location("module.name", constant_file)
    constants = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(constants)

    try:
        constants.NUM_FEATURES, constants.DIVIDER, constants.KEYPOINTS, constants.SEQ_LENGTH
    except NameError:
        print('constant file should have following keys: NUM_FEATURES, DIVIDER, KEYPOINTS, SEQ_LENGTH')
    constants.N_KEYPOINTS = len(constants.KEYPOINTS)

    return constants


def plot_save(plot_fct, save_bool=True, title='', only_png=False, outputdir=''):
    with plt.style.context('seaborn'): #plt.style.context('dark_background'):
        plot_fct()
        if save_bool:
            if only_png:
                plt.savefig(os.path.join(outputdir, title + '_dark.png'), transparent=True)
            else:
                plt.savefig(os.path.join(outputdir, title + '.svg'), transparent=True)
            plt.close()
    with plt.style.context('seaborn'):
        plot_fct()
        if save_bool:
            plt.savefig(os.path.join(outputdir, title + '.png'))
            plt.close()


def compute_interp(data_with_holes_np, mask_holes_np, keypoints, n_dim):
    # do it all in numpy
    linear_interp_data = np.copy(data_with_holes_np[..., :n_dim])
    for i_sample, (sample, mask_sample) in enumerate(zip(data_with_holes_np, mask_holes_np)):
        out = find_holes(mask_sample, keypoints, indep=True)
        for start, length, kpname in out:
            k = keypoints.index(kpname)
            if start == 0 or start + length >= len(sample):
                print(f'[WARNING] cannot compute interpolation on the segment [{start}, {start + length}]')
                continue
            # if not np.all(sample[start - 1, k, :n_dim] != 0) or not np.all(sample[start + length, k, :n_dim] != 0):
            # here prints when one of the coordinate before or after the hole is actually 0
            #     print(sample[start - 1: start + length + 1, k, :3])
            #     print('stop')
            for i_dim in range(n_dim):
                interp_ = np.linspace(sample[start - 1, k, i_dim],
                                      sample[start + length, k, i_dim], length)
                linear_interp_data[i_sample, start:start + length, k, i_dim] = interp_
    return linear_interp_data


def find_holes(mask, keypoints, target_val=1, indep=True):
    # holes are where mask == target_val
    # data shape (time, keypoints, 3) or (time, keypoints)
    # runs with torch tensor or numpy array
    if type(mask) == np.ndarray:
        if len(mask.shape) == 2:
            mask = mask.reshape((mask.shape[0], len(keypoints), -1))
        module_ = np
    else:
        if len(mask.shape) == 2:
            mask = mask.view((mask.shape[0], len(keypoints), -1))
        module_ = torch
    out = []
    if indep:
        for i_kp in range(len(keypoints)):
            # safer to loop on the keypoints, and process the mask 1D
            # probably slower
            start = 0
            mask_kp = mask[:, i_kp]
            while start < mask_kp.shape[0]:
                if not module_.any(mask_kp[start:] == target_val):
                    break
                index_start_nan = module_.where(mask_kp[start:] == target_val)[0][0]
                if module_.any(~(mask_kp[start + index_start_nan:] == target_val)):
                    length_nan = module_.where(~(mask_kp[start + index_start_nan:] == target_val))[0][0]
                else:
                    # the nans go until the end of the vector
                    length_nan = mask_kp.shape[0] - start - index_start_nan
                out.append((start + index_start_nan, length_nan, keypoints[i_kp]))
                start = start + index_start_nan + length_nan

    else:
        # sets of keypoints
        times, kp, _ = np.where(mask == target_val)
        kp_per_time = [''] * mask.shape[0]
        set_kp = []
        for t in np.unique(times):
            tmp = ' '.join(np.unique([keypoints[v] for v in kp[times == t]]))
            kp_per_time[t] = tmp
            set_kp.append(tmp)
        sets, counts = np.unique(set_kp, return_counts=True)
        kp_per_time = np.array(kp_per_time)
        for i_set in range(len(sets)):
            start = 0
            while start < mask.shape[0]:
                if not np.any(kp_per_time[start:] == sets[i_set]):
                    break
                index_start_nan = np.where(kp_per_time[start:] == sets[i_set])[0][0]
                if np.any(~(kp_per_time[start + index_start_nan:] == sets[i_set])):
                    length_nan = np.where(~(kp_per_time[start + index_start_nan:] == sets[i_set]))[0][0]
                else:
                    # the nans go until the end of the vector
                    length_nan = mask.shape[0] - start - index_start_nan
                out.append((start + index_start_nan, length_nan, sets[i_set]))
                start = start + index_start_nan + length_nan

    return out  # returns a list of tuples (length_nan, keypoint_name)
