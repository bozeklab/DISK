import numpy as np
import os
import sys
import matplotlib

matplotlib.use('Agg')
basedir = '.'
import matplotlib.pyplot as plt
import time
import random
import logging
import pandas as pd
import hydra
from omegaconf import DictConfig

from DISK.utils.dataset_utils import load_datasets
from DISK.utils.utils import read_constant_file, plot_training, timeSince, load_checkpoint, \
    save_checkpoint
from DISK.utils.transforms import init_transforms
from DISK.utils.train_fillmissing import construct_NN_model, feed_forward, compute_loss
from DISK.utils.transformer_lr_scheduler import TransformerLRScheduler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from torch.nn.utils import clip_grad_norm_


@hydra.main(version_base=None, config_path="conf", config_name="conf_missing")
def my_app(_cfg: DictConfig) -> None:
    if _cfg.training.seed:
        torch.manual_seed(_cfg.training.seed)
        random.seed(0)
        np.random.seed(0)

    outputdir = os.getcwd()
    basedir = hydra.utils.get_original_cwd()
    logging.info(f'basedir: {basedir}')

    torch.autograd.set_detect_anomaly(True)
    """ LOGGING AND PATHS """

    logging.info(f'{_cfg}')

    constant_file_path = os.path.join(basedir, 'datasets', _cfg.dataset.name, f'constants.py')
    if not os.path.exists(constant_file_path):
        raise ValueError(f'no constant file found in', constant_file_path)
    dataset_constants = read_constant_file(constant_file_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info("Device: {}".format(device))

    print_every = _cfg.training.get('print_every', 1)

    """ DATA """
    transforms, _ = init_transforms(_cfg, dataset_constants.KEYPOINTS, dataset_constants.DIVIDER,
                                 dataset_constants.SEQ_LENGTH, basedir, outputdir)

    logging.info('Loading datasets...')
    if _cfg.dataset.skeleton_file is not None:
        skeleton_file_path = os.path.join(basedir, 'datasets', _cfg.dataset.skeleton_file)
        if not os.path.exists(skeleton_file_path):
            raise ValueError(f'no skeleton file found in', skeleton_file_path)
    else:
        skeleton_file_path = None

    train_dataset, val_dataset, test_dataset = load_datasets(dataset_name=_cfg.dataset.name,
                                                             dataset_constants=dataset_constants,
                                                             transform=transforms,
                                                             dataset_type='supervised',
                                                             suffix='_w-0-nans',
                                                             root_path=basedir,
                                                             outputdir=outputdir,
                                                             skeleton_file=skeleton_file_path,
                                                             label_type='all',  # don't care, not using
                                                             verbose=_cfg.feed_data.verbose)

    train_loader = DataLoader(train_dataset, batch_size=_cfg.training.batch_size, shuffle=True,
                              num_workers=_cfg.training.n_cpus)
    val_loader = DataLoader(val_dataset, batch_size=_cfg.training.batch_size, shuffle=True,
                            num_workers=_cfg.training.n_cpus)

    """ MODEL INITIALIZATION """
    logging.info('Initializing prediction model...')
    # load model
    model = construct_NN_model(_cfg, dataset_constants, skeleton_file_path, device)

    logging.info(f'{model}')
    logging.info(f'Nb of NN parameters: {np.sum([p.numel() for p in model.parameters() if p.requires_grad])}')

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=_cfg.training.learning_rate)

    if _cfg.training.loss.type == 'l1':
        criterion_seq = nn.L1Loss(reduction='none')
    elif _cfg.training.loss.type == 'l2':
        criterion_seq = nn.MSELoss(reduction='none')
    else:
        raise NotImplementedError(f'[ERROR][MAIN_FILLMISSING] Loss type _cfg.training.loss.type should be "l1" or "l2". '
                                  f'Given: {_cfg.training.loss.type}')

    start = time.time()
    lambda1 = lambda ith_epoch: _cfg.training.model_scheduler.rate ** (ith_epoch // _cfg.training.model_scheduler.steps_epoch)
    if _cfg.training.model_scheduler.type == 'lambdalr':
        model_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    elif _cfg.training.model_scheduler.type == 'transformer':
        total_steps = int(_cfg.training.epochs * len(train_loader))
        warmup_steps = int(total_steps / 10)
        model_scheduler = TransformerLRScheduler(optimizer, init_lr=1e-4, peak_lr=_cfg.training.learning_rate,
                                                 final_lr=1e-6, final_lr_scale=0.05,
                                                 warmup_steps=warmup_steps, decay_steps=total_steps - warmup_steps)
    past_val_rmse = np.inf

    start_epoch = 1
    # Load a saved model
    if _cfg.training.load:
        for item in os.listdir(os.path.join(basedir, _cfg.training.load)):
            if item.startswith('model_epoch') and not item.endswith('txt'):
                # Pull the starting epoch from the file name
                print('Loading model from', item)
                start_epoch, loaded_print_every = load_checkpoint(model, optimizer, os.path.join(basedir, _cfg.training.load, item), device)
                start_epoch += 1
                # found a model, so stop looking in the folders
                break

    if _cfg.training.load:
        file_output = open(f'training_losses.txt', 'a')
        for item in os.listdir(os.path.join(basedir, _cfg.training.load)):
            if item.startswith('training_losses'):
                previous_content = open(os.path.join(basedir, _cfg.training.load, item), 'r').readlines()
                file_output.writelines(previous_content[:(start_epoch - 1) // loaded_print_every])
                # found a model, stop looking in the folders
                break
    else:
        file_output = open(f'training_losses.txt', 'w')


    for ith_epoch in range(start_epoch, start_epoch + _cfg.training.epochs):
        ave_loss_train = 0
        ave_rmse_train = 0

        ### TRAINING LOOP
        for data_dict in train_loader:
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            data_with_holes = data_dict['X'].to(device)
            if 'x_supp' in data_dict and torch.any(torch.isnan(data_dict['x_supp'])):
                print('[MAIN_FILLMISSING][main train loop] nan in input data')
                sys.exit(1)
            data_full = data_dict['x_supp'].to(device)
            mask_holes = data_dict['mask_holes'].to(device)

            de_out, _, total_loss, loss_original, list_rmse = feed_forward(data_with_holes,
                                                                           mask_holes, dataset_constants.DIVIDER,
                                                                           model, _cfg, data_full=data_full,
                                                                           criterion_seq=criterion_seq)
            ave_loss_train += total_loss.item()
            ave_rmse_train += list_rmse.mean().item()

            total_loss.backward()
            clip_grad_norm_(model.parameters(), 25, norm_type=2)
            optimizer.step()
            if torch.stack([torch.isnan(p).any() for p in model.parameters()]).any():
                raise ValueError('[ERROR][MAIN_FILLMISSING][main train loop] Nans in the model weights after optimizer '
                                 'step')

        model_scheduler.step()
        ave_loss_train /= len(train_loader)
        ave_rmse_train /= len(train_loader)

        ### EVALUATION
        if ith_epoch % print_every == 0 and ith_epoch != start_epoch:
            with torch.no_grad():
                ave_loss_eval, ave_rmse_eval, _ = compute_loss(model, val_loader, dataset_constants.DIVIDER,
                                                               criterion_seq, _cfg, device)

                logging.info(f'Epoch {ith_epoch:>3}: TrainLoss {ave_loss_train:.6f} EvalLoss {ave_loss_eval:.6f} ')
                logging.info(f'{"":>11}TrainRMSE {ave_rmse_train:.6f} EvalRMSE {ave_rmse_eval:.6f} ')
                logging.info(f'{"":>11}Time since beginning: {timeSince(start, (ith_epoch - start_epoch) / _cfg.training.epochs)} '
                             f'-- Completed: {(ith_epoch - start_epoch) / _cfg.training.epochs * 100:.1f}% \n')

                file_output.writelines('%.6f %.6f %.6f %.6f %.4f \n' %
                                       (ave_loss_train, ave_rmse_train, ave_loss_eval, ave_rmse_eval,
                                        model_scheduler.get_last_lr()[0]))

                if ave_rmse_eval < past_val_rmse:
                    past_val_rmse = ave_rmse_eval
                    for item in os.listdir(outputdir):
                        if item.startswith('model_epoch') and not item.endswith('txt'):
                            # overwrite and make the file blank instead - ref: https://stackoverflow.com/a/4914288/3553367
                            open(os.path.join(outputdir, item), 'w').close()
                            os.remove(os.path.join(outputdir, item))
                    logging.info('saving model...')
                    path_model = os.path.join(os.path.join(outputdir, f'model_epoch{ith_epoch}'))
                    value_dict = {'ave_loss_train': ave_loss_train,
                                  'ave_rmse_train': ave_rmse_train,
                                  'ave_loss_eval': ave_loss_eval,
                                  'ave_rmse_eval': ave_rmse_eval,
                                  'lr': model_scheduler.get_last_lr()[0],
                                  'print_every': print_every}
                    save_checkpoint(model, ith_epoch, optimizer, value_dict, path_model)

        if ith_epoch % 50 == 0 or ith_epoch == start_epoch + _cfg.training.epochs - 1:  # to flush
            filename = file_output.name
            file_output.close()
            file_output = open(filename, 'a')

    with torch.no_grad():
        # ave_loss_train, ave_rmse_train, _ = compute_loss(model, train_loader, dataset_constants.DIVIDER, criterion_seq, _cfg, device)
        ave_loss_eval, ave_rmse_eval, _ = compute_loss(model, val_loader, dataset_constants.DIVIDER, criterion_seq, _cfg, device)
        value_dict = {'ave_loss_train': ave_loss_train, 'ave_rmse_train': ave_rmse_train, 'ave_loss_eval': ave_loss_eval,
                  'ave_rmse_eval': ave_rmse_eval, 'lr': model_scheduler.get_last_lr()[0]}
        save_checkpoint(model, ith_epoch, optimizer, value_dict,
                        os.path.join(os.path.join(outputdir, f'model_last_epoch{ith_epoch}')))

    file_output.close()

    """Plot training curves"""
    df = pd.read_csv(f'training_losses.txt', sep=' ', header=None)
    if df.shape[0] < 100:
        offset = 0
    else:
        offset = 10
    with plt.style.context('dark_background'):
        plot_training(df, offset=offset, print_every=print_every)
        plt.savefig(os.path.join(outputdir, f'loss_dark.svg'), transparent=True)
    with plt.style.context('seaborn'):
        plot_training(df, offset=offset, print_every=print_every)
        plt.savefig(os.path.join(outputdir, f'loss.svg'))

    return past_val_rmse


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format=f'[%(levelname)s][%(asctime)s] %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S',
                        )
    logger = logging.getLogger(__name__)

    my_app()
