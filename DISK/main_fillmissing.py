import numpy as np
import os
import sys
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import random
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


def train_fillmissing(project_dir, model_dir, dataset_path, skeleton_file, training_seed,
                      load_model, cfg_network,
                      training_batch_size, training_epochs, learning_rate,
                      loss_type, loss_mask, loss_factor,
                      model_scheduler_rate, model_scheduler_type, model_scheduler_steps_epoch,
                      n_cpus,
                      print_every,
                      proba_file, proba_length_file, indep_keypoints,
                      add_missing_pad, viewinvariant,
                      normalize, normalizecube, swap,
                      add_missing,
                      logger, verbose=0) -> None:
    if training_seed:
        torch.manual_seed(training_seed)
        random.seed(0)
        np.random.seed(0)

    torch.autograd.set_detect_anomaly(True)
    """ LOGGING AND PATHS """

    logger.debug(f'[TRAIN FILLMISSING]{training_seed}')

    constant_file_path = os.path.join(dataset_path, f'constants.py')
    if not os.path.exists(constant_file_path):
        raise ValueError(f'no constant file found in', constant_file_path)
    dataset_constants = read_constant_file(constant_file_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info("Device: {}".format(device))

    """ DATA """
    transforms = init_transforms(dataset_constants.KEYPOINTS,
                                 dataset_constants.DIVIDER,
                                 dataset_constants.SEQ_LENGTH,
                                 model_dir,
                                 logger,
                                 add_missing_pad,
                                 viewinvariant,
                                 normalize,
                                 normalizecube,
                                 swap,
                                 proba_file,
                                 proba_length_file,
                                 indep_keypoints,
                                 add_missing,
                                 verbose)

    logger.info('Loading datasets')
    if skeleton_file is not None and skeleton_file != '':
        skeleton_file_path = os.path.join(project_dir, 'DISK-data', skeleton_file)
        if not os.path.exists(skeleton_file_path):
            raise ValueError(f'no skeleton file found in', skeleton_file_path)
    else:
        skeleton_file_path = None

    train_dataset, val_dataset, test_dataset = load_datasets(
        dataset_path=dataset_path,
        keypoints=dataset_constants.KEYPOINTS,
        divider=dataset_constants.DIVIDER,
        transform=transforms,
        dataset_type='supervised',
        suffix='_w-0-nans',
        root_path=project_dir,
        outputdir=model_dir,
        skeleton_file=skeleton_file_path,
        label_type='all',  # don't care, not using
        verbose=verbose,
        logger=logger
    )

    train_loader = DataLoader(train_dataset, batch_size=training_batch_size, shuffle=True,
                              num_workers=n_cpus)
    val_loader = DataLoader(val_dataset, batch_size=training_batch_size, shuffle=True,
                            num_workers=n_cpus)

    """ MODEL INITIALIZATION """
    logger.info('Initializing prediction model')
    # load model
    model = construct_NN_model(cfg_network, dataset_constants.KEYPOINTS, dataset_constants.DIVIDER,
                               dataset_constants.SEQ_LENGTH,
                               skeleton_file_path,
                               device)

    logger.debug(f'Nb of NN parameters: {np.sum([p.numel() for p in model.parameters() if p.requires_grad])}')

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=learning_rate)

    if loss_type == 'l1':
        criterion_seq = nn.L1Loss(reduction='none')
    elif loss_type == 'l2':
        criterion_seq = nn.MSELoss(reduction='none')
    else:
        raise NotImplementedError(f'[ERROR][MAIN_FILLMISSING] Loss type should be "l1" or '
                                  f'"l2". '
                                  f'Given: {loss_type}')

    start = time.time()
    lambda1 = lambda ith_epoch: model_scheduler_rate ** (ith_epoch // model_scheduler_steps_epoch)
    if model_scheduler_type == 'lambdalr':
        model_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    elif model_scheduler_type == 'transformer':
        total_steps = int(training_epochs * len(train_loader))
        warmup_steps = int(total_steps / 10)
        model_scheduler = TransformerLRScheduler(optimizer, init_lr=1e-4, peak_lr=learning_rate,
                                                 final_lr=1e-6, final_lr_scale=0.05,
                                                 warmup_steps=warmup_steps, decay_steps=total_steps - warmup_steps)
    past_val_rmse = np.inf

    start_epoch = 1
    # Load a saved model
    if load_model:
        for item in os.listdir(load_model):
            if item.startswith('model_epoch') and not item.endswith('txt'):
                # Pull the starting epoch from the file name
                print('Loading model from', item)
                start_epoch, loaded_print_every = load_checkpoint(model, optimizer, os.path.join(project_dir, load_model,
                                                                                                 item), device, logger)
                start_epoch += 1
                # found a model, so stop looking in the folders
                break

    if load_model:
        file_output = open(os.path.join(model_dir, f'training_losses.txt'), 'a')
        for item in os.listdir(load_model):
            if item.startswith('training_losses'):
                previous_content = open(os.path.join(load_model, item), 'r').readlines()
                file_output.writelines(previous_content[:(start_epoch - 1) // loaded_print_every])
                # found a model, stop looking in the folders
                break
    else:
        file_output = open(os.path.join(model_dir, f'training_losses.txt'), 'w')

    ith_epoch = 0
    for ith_epoch in range(start_epoch, start_epoch + training_epochs):
        ave_loss_train = 0
        ave_rmse_train = 0

        ### TRAINING LOOP
        for data_dict in train_loader:
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            data_with_holes = data_dict['X'].to(device)
            if torch.any(torch.isnan(data_dict['x_supp'])):
                print('[MAIN_FILLMISSING][main train loop] nan in input data')
                sys.exit(1)
            data_full = data_dict['x_supp'].to(device)
            mask_holes = data_dict['mask_holes'].to(device)

            de_out, _, total_loss, loss_original, list_rmse = feed_forward(data_with_holes,
                                                                           mask_holes, dataset_constants.DIVIDER,
                                                                           model, loss_mask, loss_factor,
                                                                           cfg_network,
                                                                           data_full=data_full,
                                                                           criterion_seq=criterion_seq,
                                                                           logger=logger)
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
                                                               criterion_seq, loss_mask, loss_factor, cfg_network,
                                                               device, logger)

                logger.info(f'Epoch {ith_epoch:>3}: TrainLoss {ave_loss_train:.6f} EvalLoss {ave_loss_eval:.6f} ')
                logger.info(f'{"":>11}TrainRMSE {ave_rmse_train:.6f} EvalRMSE {ave_rmse_eval:.6f} ')
                logger.info(f'{"":>11}Time since beginning: '
                            f'{timeSince(start, (ith_epoch - start_epoch + 1) / training_epochs)} '
                             f'-- Completed: {(ith_epoch - start_epoch + 1) / training_epochs * 100:.1f}% \n')

                file_output.writelines('%.6f %.6f %.6f %.6f %.4f \n' %
                                       (ave_loss_train, ave_rmse_train, ave_loss_eval, ave_rmse_eval,
                                        model_scheduler.get_last_lr()[0]))

                if ave_rmse_eval < past_val_rmse:
                    past_val_rmse = ave_rmse_eval
                    for item in os.listdir(model_dir):
                        if item.startswith('model_epoch') and not item.endswith('txt'):
                            # overwrite and make the file blank instead - ref: https://stackoverflow.com/a/4914288/3553367
                            open(os.path.join(model_dir, item), 'w').close()
                            os.remove(os.path.join(model_dir, item))
                    logger.info('saving model')
                    path_model = os.path.join(os.path.join(model_dir, f'model_epoch{ith_epoch}'))
                    value_dict = {'ave_loss_train': ave_loss_train,
                                  'ave_rmse_train': ave_rmse_train,
                                  'ave_loss_eval': ave_loss_eval,
                                  'ave_rmse_eval': ave_rmse_eval,
                                  'lr': model_scheduler.get_last_lr()[0],
                                  'print_every': print_every}
                    save_checkpoint(model, ith_epoch, optimizer, value_dict, path_model)

        if ith_epoch % 50 == 0 or ith_epoch == start_epoch + training_epochs - 1:  # to flush
            filename = file_output.name
            file_output.close()
            file_output = open(filename, 'a')

    with torch.no_grad():
        # ave_loss_train, ave_rmse_train, _ = compute_loss(model, train_loader, dataset_constants.DIVIDER, criterion_seq, _cfg, device)
        ave_loss_eval, ave_rmse_eval, _ = compute_loss(model, val_loader, dataset_constants.DIVIDER, criterion_seq,
                                                       loss_mask, loss_factor, cfg_network, device, logger)
        value_dict = {
            'ave_loss_train': ave_loss_train,
            'ave_rmse_train': ave_rmse_train,
            'ave_loss_eval': ave_loss_eval,
            'ave_rmse_eval': ave_rmse_eval,
            'lr': model_scheduler.get_last_lr()[0]
                      }
        save_checkpoint(model, ith_epoch, optimizer, value_dict,
                        os.path.join(os.path.join(model_dir, f'model_last_epoch{ith_epoch}')))

    file_output.close()

    """Plot training curves"""
    df = pd.read_csv(os.path.join(model_dir, f'training_losses.txt'), sep=' ', header=None)
    if df.shape[0] < 100:
        offset = 0
    else:
        offset = 10

    with plt.style.context('seaborn'):
        plot_training(df, offset=offset, print_every=print_every)
        plt.savefig(os.path.join(model_dir, f'loss.svg'))

    return past_val_rmse


