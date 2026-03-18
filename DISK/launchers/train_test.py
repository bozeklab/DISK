import logging
from datetime import datetime
import os
import sys
import yaml
import torch

from DISK.utils.logger_setup import setup_custom_logging, copy_config_file, VoidHandler
from DISK.models.graph import Graph
from DISK.utils.config_decorator import config_reader, parse_command_line_args, test_boolean_variable
from DISK.main_fillmissing import train_fillmissing
from DISK.test_fillmissing import test

possible_network_type_values = ('transformer', 'gru', 'st_gcn', 'sts_gcn', 'tcn')

def check_network_type(value: str) -> bool:
    if value in possible_network_type_values:
        return True
    else:
        return False

def find_proba_files(dataset_path: str, suffix: str):
    proba_file = os.path.join(dataset_path, f'proba_missing{suffix}.csv')
    proba_length_file = os.path.join(dataset_path, f'proba_missing_length{suffix}.csv')

    if not os.path.exists(proba_file) or not os.path.exists(proba_length_file):
        proba_file = os.path.join(dataset_path, f'proba_missing_uniform{suffix}.csv')
        proba_length_file = os.path.join(dataset_path, f'proba_missing_length_uniform{suffix}.csv')

        if not os.path.exists(proba_file) or not os.path.exists(proba_length_file):
            return False, None, None

    return True, proba_file, proba_length_file


def check_model_dir(model_dir: str):
    found_checkpoint = False
    found_training_losses = False
    for item in os.listdir(model_dir):
        if item.startswith('model_epoch') and not item.endswith('txt'):
            found_checkpoint = True
        if item.startswith('training_losses'):
            found_training_losses = True
    return found_checkpoint and found_training_losses

def main(project_dir, model_dir, dataset_path, dataset_name, test_dir, skeleton_graph,
         training_seed, load_model_dir, cfg_network, training_batch_size,
         training_epochs, learning_rate, loss_type, loss_mask, loss_factor,
         model_scheduler_rate, model_scheduler_type, model_scheduler_steps_epoch,
         n_cpus, print_every,
         proba_file, proba_length_file, indep_keypoints,
         add_missing_pad, viewinvariant, normalize, normalizecube, swap,
         add_missing,
         test_original_coordinates, test_threshold_pck,
         n_repeat,
         total_n_plots, plot2d_only_holes, plot3d_size, plot3d_azim,
         logger, verbose=0):

    logger.info(f'*********************** TRAINING DISK *********************** \n')
    try:
        train_fillmissing(project_dir, model_dir, dataset_path, skeleton_graph, training_seed,
                          load_model_dir, cfg_network,
                          training_batch_size, training_epochs, learning_rate,
                          loss_type, loss_mask, loss_factor,
                          model_scheduler_rate, model_scheduler_type, model_scheduler_steps_epoch,
                          n_cpus,
                          print_every,
                          proba_file, proba_length_file, indep_keypoints,
                          add_missing_pad, viewinvariant,
                          normalize, normalizecube, swap,
                          add_missing,
                          logger, verbose=verbose)
    except torch.OutOfMemoryError:
        print(f"\n❌ CUDA (GPU) out of memory. Try reducing the --training_batch_size. Got {training_batch_size}")
        sys.exit(1)

    logger.info(f'✅ Successfully trained DISK model {model_dir}.\n')

    logger.info(f'*********************** TESTING DISK TRAINED MODEL *********************** \n')

    add_missing_pad_for_test = (max(1, add_missing_pad[0]), max(1, add_missing_pad[0]))
    pcoef_per_model, err_pck_sup = test(project_dir, test_dir, dataset_path, dataset_name, skeleton_graph,
                             [model_dir, ], training_batch_size, n_cpus,
                             loss_type, loss_mask, loss_factor,
                             proba_file, proba_length_file, indep_keypoints,
                             add_missing_pad_for_test,
                             viewinvariant, normalize, normalizecube, swap, add_missing,
                             test_original_coordinates, test_threshold_pck, n_repeat,
                             total_n_plots, plot2d_only_holes,
                             plot3d_size, plot3d_azim,
                             logger, suffix='', stride=None, verbose=verbose)
    logger.info(f'✅ Successfully tested DISK model {model_dir}.\n')

    if pcoef_per_model[0][0] is not None and pcoef_per_model[0][0] < 0.8:
        logger.info(f"⚠️ The correlation coefficient between the estimation of the error and "
                    f"the real error made by the model is low (corr = {pcoef_per_model[0][0]:.2f}). \n"
                    f"Be cautious when visualizing it in the plots and when using DISK-impute "
                    f"(threshold_error_score).\n")

    if err_pck_sup[0]  == -1:
        logger.info(f"⚠️ The DISK model seems to give poor results. \n"
                    f"No threshold for the estimated error was found "
                    f"to reach at least 80% of correct keypoints.")
    else:
        if err_pck_sup[0] is not None:
            logger.info(f"ℹ️  Based on the test results, we recommend a threshold_error_score of "
                        f"{err_pck_sup[0]:.3f} for the imputation step (based on 80% of PCK@{test_threshold_pck} on "
                        f"the test set).")
        else:
            logger.info(f"ℹ️  The DISK model was trained without module for error estimation. \n"
                        f"No thresholding on the results will be possible at imputation step.")



@config_reader(config_path="../conf/config_train.yaml")
def cli(_cfg) -> None:
    _cfg = parse_command_line_args(_cfg)
    modified_cfg = dict(_cfg.__dict__)

    for key in ('project_path', 'dataset_name'):
        val = _cfg.__dict__[key]
        if val is None or val == '_DEFAULT_':
            print(f'\n❌ No value was passed to parameter {key}. This is a required parameter.'
                  f'\n  Expected syntax:'
                  f'\n  > DISK-train --project_path test_project --dataset_name dataset')
            sys.exit(1)

    ### _CFG PARAMETER CHECK --- REQUIRED PARAMETERS
    if _cfg.project_path is None or type(_cfg.project_path) != str:
        print("\n❌ project_path is a required parameter and should be a "
              "valid path (str) to the config_project.yaml file. "
              f"Got {_cfg.project_path}")
        sys.exit(1)
    elif not os.path.exists(_cfg.project_path) or not os.path.exists(os.path.join(
            _cfg.project_path, 'config_project.yaml')):
        print("\n❌ project_path is a required parameter and should be a "
              "valid path to the config_project.yaml file. "
              f"Got {_cfg.project_path}")
        sys.exit(1)
    else:
        project_path = _cfg.project_path

    ### LOAD PROJECT LOG
    # Load the YAML configuration file
    with open(os.path.join(project_path, 'config_project.yaml'), 'r') as file:
        config = yaml.safe_load(file)

    if _cfg.dataset_name is None or type(_cfg.dataset_name) != str \
            or not os.path.exists(os.path.join(project_path, 'DISK_data', _cfg.dataset_name)):
        print("\n❌ dataset_name is a required parameter and should be the name "
              "of an existing dataset within subfolder DISK_data. "
              f"  Got {_cfg.dataset_name} {os.path.join(project_path, 'DISK_data', _cfg.dataset_name)}")
        sys.exit(1)
    else:
        dataset_name = os.path.basename(_cfg.dataset_name)

    dataset_path = os.path.join(project_path, 'DISK_data', dataset_name)

    final_load_model_path = None
    if _cfg.load_model is not None and type(_cfg.load_model) == str:
        final_load_model_path = os.path.join(project_path, 'DISK_train', _cfg.load_model)
        if not os.path.exists(final_load_model_path) or not check_model_dir(final_load_model_path):
            print(f"\n❌ You provided a load_model entry, but could not find the checkpoint at {final_load_model_path}.")
            sys.exit(1)

    if final_load_model_path is not None:
        if _cfg.model_name == '_DEFAULT_':
            model_name = os.path.basename(final_load_model_path)
            final_model_path = os.path.join(project_path, 'DISK_train', model_name)
        else:
            model_name = _cfg.model_name
            model_path = os.path.join(project_path, 'DISK_train', model_name)
            if model_path == final_load_model_path:
                final_model_path = model_path
            else:
                final_model_path = str(model_path)
                ext_model_path = 1
                while os.path.exists(final_model_path):
                    final_model_path = model_path + f'_{ext_model_path}'
                    ext_model_path += 1

                os.mkdir(final_model_path)
    else:
        if _cfg.model_name == '_DEFAULT_':
            if _cfg.network == 'transformer':
                network_name = 'DISK'
            else:
                network_name = f'DISK-{_cfg.network}'
            model_name = f'{network_name}_{dataset_name}'
        else:
            if _cfg.model_name is None or type(_cfg.model_name) != str:
                print("\n❌ model_name should be a "
                      f"string. Got {_cfg.model_name}")
                sys.exit(1)
            else:
                model_name = _cfg.model_name

        model_path = os.path.join(project_path, 'DISK_train', model_name)
        final_model_path = str(model_path)
        ext_model_path = 1
        while os.path.exists(final_model_path):
            final_model_path = model_path + f'_{ext_model_path}'
            ext_model_path += 1

        os.mkdir(final_model_path)
    modified_cfg['model_name'] = os.path.basename(final_model_path)

    if _cfg.debug:
        logging_flag = logging.DEBUG
        verbose = 1
    else:
        logging_flag = logging.INFO
        verbose = 0

    logging.basicConfig(level=logging_flag, handlers=[VoidHandler()])
    logger = setup_custom_logging(final_model_path, 'train.log', logging_flag)

    if final_load_model_path is not None:
        if final_load_model_path == final_model_path:
            logger.info(f"\n️⚠️ Loading model from and saving in {final_load_model_path}. Is it the desired behavior? [y/n]")
            y_n = input('> ')
            while y_n not in ['y', 'n', 'Y', 'N', 'yes', 'no', 'Yes', 'YES', 'No', 'NO']:
                y_n = input ('Retype y or n: ')
            if y_n in ['y', 'Y', 'yes', 'Yes', 'YES']:
                pass
            else:
                logger.info(f"️⚠️ If you want to save the output in a different folder, then the correct command is:\n"
                            f"DISK-train ... --load_model my_existing_model --model_name a_new_name ...\n")
                exit(0)
        else:
            logger.info(f"ℹ️  Loading model {final_load_model_path}, saving in {final_model_path}.\n")
    else:
        logger.info(f"ℹ️  Model folder is {final_model_path}.\n")


    if (not 'skeleton' in config.keys()) or config['skeleton'] is None or len(config['skeleton']) == 0:
        skeleton_graph = None
    else:
        skeleton_graph = Graph(len(config['keypoints']),
                               config['skeleton_center'],
                               config['skeleton'],
                               config['skeleton_colors'],
                               logger=logger)

    ### _CFG PARAMETER CHECK --- OFTEN CHANGED PARAMETERS
    if _cfg.training_epochs is None or type(_cfg.training_epochs) != int:
        print("\n❌ training_epochs should be a "
              f"strictly positive integer. Got {_cfg.training_epochs}")
        sys.exit(1)
    elif _cfg.training_epochs <= 0:
        print("\n❌ training_epochs should be a "
              f"strictly positive integer. Got {_cfg.training_epochs}")
        sys.exit(1)
    else:
        training_epochs = _cfg.training_epochs

    if _cfg.training_batch_size is None or type(_cfg.training_batch_size) != int:
        print("\n❌ training_batch_size should be an "
              f"integer. Got {_cfg.training_batch_size}")
        sys.exit(1)
    elif _cfg.training_batch_size <= 0:
        print("\n❌ training_batch_size should be a "
              f"strictly positive integer. Got {_cfg.training_batch_size}")
        sys.exit(1)
    else:
        training_batch_size = _cfg.training_batch_size

    if _cfg.n_cpus is None or type(_cfg.n_cpus) != int:
        print(f"\n❌ n_cpus should be a positive integer. Got {_cfg.n_cpus}")
        sys.exit(1)
    else:
        n_cpus = max(0, _cfg.n_cpus)

    modified_cfg['n_cpus'] = n_cpus

    if not check_network_type(_cfg.network):
        print(f"\n❌ network {_cfg.network} is not recognized. "
              f"Should be of type {possible_network_type_values}")
        sys.exit(1)

    script_directory = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(script_directory, f'../conf/network/{_cfg.network}.yaml'), 'r') as file:
        network_config = yaml.safe_load(file)
    modified_cfg['network'] = network_config

    if _cfg.training_learning_rate == '_DEFAULT_':
        if _cfg.network == 'transformer':
            learning_rate = 0.001
        else:
            learning_rate = 0.0001
    else:
        if _cfg.training_learning_rate is None or type(_cfg.training_learning_rate) != float:
            print(f"\n❌ learning_rate should be a positive float. "
                  f"Got {_cfg.training_learning_rate}")
            sys.exit(1)
        else:
            learning_rate = max(0, _cfg.training_learning_rate)

    modified_cfg['training_learning_rate'] = learning_rate

    if _cfg.training_seed is None:
        training_seed = None
    else:
        if _cfg.training_seed == '_DEFAULT_':
            training_seed = None
        elif type(_cfg.training_seed) != int:
            print(f"\n❌ training_seed should be a positive integer. "
                  f"Got {_cfg.training_seed}")
            sys.exit(1)
        else:
            training_seed = max(0, _cfg.training_seed)

    if _cfg.print_every == '_DEFAULT_':
        if training_epochs <= 10:
            print_every = 1
        elif training_epochs <= 50:
            print_every = 2
        else:
            print_every = 5
    else:
        if _cfg.print_every is None or type(_cfg.print_every) != int:
            print(f"\n❌ print_every should be an integer. Got {_cfg.print_every}")
            sys.exit(1)
        else:
            print_every = max(1, min(_cfg.print_every, training_epochs//10))
    modified_cfg['print_every'] = print_every

    if _cfg.transforms_add_missing_pad is None or len(_cfg.transforms_add_missing_pad) != 2 or type(
            _cfg.transforms_add_missing_pad[0]) != int or type(
        _cfg.transforms_add_missing_pad[1]) != int:
        print("\n❌ transforms_add_missing_pad should be an "
              f"a list of two integers. Got {_cfg.transforms_add_missing_pad}")
        sys.exit(1)
    else:
        add_missing_pad = list(_cfg.transforms_add_missing_pad)

    if config['original_missing']:
        indep_keypoints = test_boolean_variable(_cfg.indep_keypoints, 'indep_keypoints')
        merge_keypoints = test_boolean_variable(_cfg.merge_keypoints, 'merge_keypoints')

        suffix = f'_set_keypoints' if not indep_keypoints else ''
        if indep_keypoints:
            if merge_keypoints:
                logger.info(f'️ℹ\n️ merge_keypoints = True is not a valid option when indep_keypoints = True. '
                            f'merge_keypoints would be considered False')
                suffix += f'_merged'

        proba_files_exist, proba_file, proba_length_file = find_proba_files(dataset_path, suffix)

        if not proba_files_exist:
            from DISK.create_proba_missing_files import create_proba_missing_files
            indep_keypoints = False if 'set_keypoints' in suffix else True
            merge_keypoints = True if ('merged' in suffix and not indep_keypoints) else False

            create_proba_missing_files(project_path, dataset_path, indep_keypoints, merge_keypoints, skeleton_graph,
                                       logger)
            logger.info(f'✅ Successfully estimated probabilities of missing keypoints with '
                        f'{["set_keypoints", "indep_keypoints"][int(indep_keypoints)]}.\n')

        proba_files_exist, proba_file, proba_length_file = find_proba_files(dataset_path, suffix)
        if not proba_files_exist:
            print("\n❌ did not find proba_files matching your criterion.")
            sys.exit(1)

    else:
        indep_keypoints = True
        merge_keypoints = False
        suffix = '_uniform'

        proba_files_exist, proba_file, proba_length_file = find_proba_files(dataset_path, suffix)

        if not proba_files_exist:
            from DISK.create_proba_missing_files import create_proba_missing_files
            indep_keypoints = False if 'set_keypoints' in suffix else True
            merge_keypoints = True if ('merged' in suffix and not indep_keypoints) else False

            create_proba_missing_files(project_path, dataset_path, indep_keypoints, merge_keypoints, skeleton_graph,
                                       logger)
            logger.info(f'✅ Successfully estimated probabilities of missing keypoints with '
                        f'{["set_keypoints", "indep_keypoints"][int(indep_keypoints)]}.\n')

        proba_files_exist, proba_file, proba_length_file = find_proba_files(dataset_path, suffix)
        if not proba_files_exist:
            print("\n❌ did not find proba_files matching your criterion.")
            sys.exit(1)

    viewinvariant = test_boolean_variable(_cfg.transforms_viewinvariant, 'transforms_viewinvariant')
    normalize = test_boolean_variable(_cfg.transforms_normalize, 'transforms_normalize')
    normalizecube = test_boolean_variable(_cfg.transforms_normalizecube, 'transforms_normalizecube')

    if _cfg.transforms_swap is None or type(
            _cfg.transforms_swap) != float or _cfg.transforms_swap < 0 or _cfg.transforms_swap > 1:
        print("\n❌ transforms_swap should be a float between 0 and 1 "
              "(probability of swapping during training). "
              f"Got {_cfg.transforms_swap}")
        sys.exit(1)
    else:
        swap = _cfg.transforms_swap

    if _cfg.test_n_plots is None or type(_cfg.test_n_plots) != int:
        print("\n❌ test.n_plots should be a positive integer. "
              f"Got {_cfg.test_n_plots}")
        sys.exit(1)
    else:
        n_plots = max(0, _cfg.test_n_plots)

    if _cfg.test_threshold_pck is None or type(
            _cfg.test_threshold_pck) != float or _cfg.test_threshold_pck < 0 or _cfg.test_threshold_pck > 1:
        print("\n❌ test.threshold_pck should be a "
              f"float between 0 and 1. Got {_cfg.test_threshold_pck}")
        sys.exit(1)
    else:
        threshold_pck = _cfg.test_threshold_pck

    if _cfg.plot_azim3d is None or type(_cfg.plot_azim3d) != int:
        print("\n❌ test.plot3d_azim should be an integer."
              f"Got {_cfg.plot_azim3d}")
        sys.exit(1)
    else:
        plot3d_azim = _cfg.plot_azim3d

    if _cfg.plot_size3d is None or type(_cfg.plot_size3d) not in [float, int]:
        print("\n❌ test.plot3d_size should be a "
              f"float. Got {_cfg.plot_size3d}")
        sys.exit(1)
    else:
        plot3d_size = _cfg.plot_size3d

    plot2d_only_holes = test_boolean_variable(_cfg.plot_only_holes2d, 'test_plot2d_only_holes')
    original_coordinates = test_boolean_variable(_cfg.plot_original_coordinates, 'test_original_coordinates')

    if _cfg.test_n_repeat is None or type(_cfg.test_n_repeat) != int:
        print("\n❌ test.n_repeat should be a string."
              f"Got {_cfg.test_n_repeat}")
        sys.exit(1)
    else:
        n_repeat = max(1, _cfg.test_n_repeat)

    if _cfg.loss_def is None or type(_cfg.loss_def) != str or not _cfg.loss_def in ['l1', 'l2']:
        print("\n❌ loss.type should be l1 or l2."
              f"Got {_cfg.loss_def}")
        sys.exit(1)
    else:
        loss_def = _cfg.loss_def

    loss_mask = test_boolean_variable(_cfg.loss_mask, 'loss_mask')

    if _cfg.loss_factor is None or type(_cfg.loss_factor) != int:
        print("\n❌ loss.factor should be an integer."
              f"Got {_cfg.loss_factor}")
        sys.exit(1)
    else:
        loss_factor = max(1, _cfg.loss_factor)

    if _cfg.model_scheduler_def is None or type(_cfg.model_scheduler_def) != str \
            or _cfg.model_scheduler_def != 'lambdalr':
        print("\n❌ model_scheduler_def should be 'lambdalr'."
              f"Got {_cfg.model_scheduler_def}")
        sys.exit(1)
    else:
        model_scheduler_type = _cfg.model_scheduler_def

    if _cfg.model_scheduler_steps_epoch is None or type(_cfg.model_scheduler_steps_epoch) != int:
        print("\n❌ model_scheduler_steps_epoch should be 'lambdalr'."
              f"Got {_cfg.model_scheduler_steps_epoch}")
        sys.exit(1)
    else:
        model_scheduler_steps_epoch = max(1, _cfg.model_scheduler_steps_epoch)

    if _cfg.model_scheduler_rate is None or type(_cfg.model_scheduler_rate) != float:
        print("\n❌ model_scheduler_rate should be a float between 0 and 1."
              f"Got {_cfg.model_scheduler_rate}")
        sys.exit(1)
    else:
        model_scheduler_rate = max(0, min(1, _cfg.model_scheduler_rate))

    os.makedirs(os.path.join(final_model_path, 'config'), exist_ok=True)

    output_config_file = os.path.join(final_model_path, 'config', f'config_train.yaml')
    copy_config_file(modified_cfg, output_config_file)

    test_dir = os.path.join(final_model_path, f'{datetime.today().strftime("%Y-%m-%d_%H-%M")}_test')
    os.makedirs(test_dir, exist_ok=True)

    os.makedirs(os.path.join(test_dir, 'config'), exist_ok=True)
    output_config_file = os.path.join(test_dir, 'config', f'config_train.yaml')
    copy_config_file(modified_cfg, output_config_file)

    logger.info(f'✅ Successfully loaded configuration.\n')

    add_missing = True
    main(project_path, final_model_path, dataset_path, dataset_name, test_dir,
         skeleton_graph, training_seed,
         final_load_model_path, network_config,
         training_batch_size, training_epochs, learning_rate,
         loss_def, loss_mask, loss_factor,
         model_scheduler_rate, model_scheduler_type, model_scheduler_steps_epoch,
         n_cpus,
         print_every,
         proba_file, proba_length_file, indep_keypoints,
         add_missing_pad, viewinvariant,
         normalize, normalizecube, swap,
         add_missing,
         original_coordinates, threshold_pck, n_repeat,
         n_plots, plot2d_only_holes,
         plot3d_size, plot3d_azim,
         logger, verbose)


if __name__ == '__main__':
    cli()
