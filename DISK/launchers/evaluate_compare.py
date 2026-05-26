import logging
from datetime import datetime
import os
import sys
import yaml
import torch

from DISK.utils.logger_setup import setup_custom_logging, copy_config_file, VoidHandler
from DISK.launchers.train_evaluate import find_proba_files, check_proba_parameters
from DISK.models.graph import Graph
from DISK.utils.config_decorator import config_reader, parse_command_line_args, test_boolean_variable
from DISK.evaluate_fillmissing import evaluate
from DISK.create_proba_missing_files import create_proba_missing_files


def main(project_dir, model_dirs, dataset_path, dataset_name, test_dir, skeleton_graph,
         training_batch_size,
         loss_type, loss_mask, loss_factor,
         n_cpus,
        rerun_create_proba, indep_keypoints, merge_keypoints, suffix_proba_files,
         add_missing_pad, viewinvariant, normalize, normalizecube, swap,
         add_missing,
         test_original_coordinates, pck_threshold,
         n_repeat,
         total_n_plots, plot2d_only_holes, plot3d_size, plot3d_azim,
         logger, suffix='', verbose=0):

    if rerun_create_proba:
        create_proba_missing_files(project_dir, dataset_path, indep_keypoints,
                                   merge_keypoints, suffix_proba_files, skeleton_graph,
                                   logger)
        logger.info(f'✅ Successfully estimated probabilities of missing keypoints with '
                    f'{["set_keypoints", "indep_keypoints"][int(indep_keypoints)]}.\n')

    proba_files_exist, proba_file, proba_length_file = find_proba_files(dataset_path, suffix_proba_files)
    if not proba_files_exist:
        print("\n❌ did not find proba_files matching your criterion.")
        sys.exit(1)

    logger.info(f'*********************** TESTING DISK TRAINED MODEL *********************** \n')
    try:
        pcoeff_per_model, err_sup_PCK = evaluate(project_dir, test_dir, dataset_path, dataset_name, skeleton_graph,
             model_dirs, training_batch_size, n_cpus,
             loss_type, loss_mask, loss_factor,
             proba_file, proba_length_file, indep_keypoints,
             add_missing_pad,
             viewinvariant, normalize, normalizecube, swap, add_missing,
             test_original_coordinates, pck_threshold, n_repeat,
             total_n_plots, plot2d_only_holes,
             plot3d_size, plot3d_azim,
             logger, suffix=suffix, stride=None, verbose=verbose)

        for model_name, err_pck_sup in zip(model_dirs, err_sup_PCK):
            if err_pck_sup == -1:
                logger.info(f"⚠️ The DISK model {os.path.basename(model_name)} seems to give poor results. \n"
                            f"No threshold for the estimated error was found "
                            f"to reach at least 80% of correct keypoints.")
            else:
                if err_pck_sup is not None:
                    logger.info(f"ℹ️  For model {os.path.basename(model_name)}, based on the test results, "
                                f"we recommend a threshold_error_score of "
                                f"{err_pck_sup:.3f} for the imputation step (based on 80% of PCK@{pck_threshold} on "
                                f"the test set).")
                else:
                    logger.info(f"ℹ️  The DISK model {os.path.basename(model_name)} was trained without module for "
                                f"error estimation. \n"
                                f"No thresholding on the results will be possible at imputation step.")

    except RuntimeError as e:
        print(f"\n❌ CUDA (GPU) out of memory ({e}). Try reducing the --training_batch_size. Got {training_batch_size}")
        sys.exit(1)

    logger.info(f'✅ Successfully tested DISK model.\n')


@config_reader(config_path="../conf/config_evaluate.yaml")
def cli(_cfg) -> None:
    print('\n', '*' * 80, sep='')
    print('*' * 30, ' DISK-evaluate START ', '*' * 30)
    print('*' * 80, '\n')

    _cfg = parse_command_line_args(_cfg)
    modified_cfg = dict(_cfg.__dict__)

    for key in ('project_path', 'dataset_name', 'model_name_list'):
        val = _cfg.__dict__[key]
        if val is None or val == '_DEFAULT_':
            print(f'\n❌ No value was passed to parameter {key}. This is a required parameter.'
                  f'\n  Expected syntax:'
                  f'\n  > DISK-evaluate project_path=test_project dataset_name=dataset model_name_list=[model1,'
                  f'model2]\n'
                  f'# careful no space between model names inside the brackets, and after/before "="')

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
        dataset_name = _cfg.dataset_name

    dataset_path = os.path.join(project_path, 'DISK_data', dataset_name)

    model_path_list = []
    if _cfg.model_name_list is None or type(_cfg.model_name_list) != list:
        print("\n❌ model_name_list should be a "
              f"list of strings. Got {_cfg.model_name_list}")
        sys.exit(1)
    else:
        model_path_list = []
        for model_name in _cfg.model_name_list:
            if model_name is None or type(model_name) != str:
                print("\n❌ model_name_list should be a "
                      f"list of strings. Got {_cfg.model_name_list}")
                sys.exit(1)
            model_path = os.path.join(project_path, 'DISK_train', model_name)
            if not os.path.exists(model_path):
                print(f"\n❌ model_name {model_name}  was not found. Please "
                      f"check the name. It should match a folder within the "
                      f"subfolder 'DISK_train' of your project.")
                sys.exit(1)
            model_path_list.append(model_path)


    if _cfg.debug:
        logging_flag = logging.DEBUG
        verbose = 1
    else:
        logging_flag = logging.INFO
        verbose = 0

    if _cfg.name_output_dir == '_DEFAULT_':
        test_dir = os.path.join(project_path, 'DISK_train', f'{datetime.today().strftime("%Y-%m-%d_%H-%M")}_test')
    else:
        if _cfg.name_output_dir is None or type(_cfg.name_output_dir) != str:
            print(f"\n❌ name_output_dir should be a "
                      f"string. Got {_cfg.name_output_dir}")
            sys.exit(1)
        else:
            test_dir = os.path.join(project_path, 'DISK_train', _cfg.name_output_dir)
    os.makedirs(test_dir, exist_ok=True)

    logging.basicConfig(level=logging_flag, handlers=[VoidHandler()])
    logger = setup_custom_logging(test_dir, 'test.log', logging_flag)


    if not ('skeleton' in config.keys() and config['skeleton'] is not None and len(config['skeleton']) == 0):
        skeleton_graph = None
    else:
        skeleton_graph = Graph(len(config['keypoints']),
                             config['center'],
                             config['neighbor_links'],
                             config['neighbor_link_colors'],
                               logger=logger)

    ### _CFG PARAMETER CHECK --- OFTEN CHANGED PARAMETERS

    if _cfg.batch_size is None or type(_cfg.batch_size) != int:
        print("\n❌ batch_size should be an "
              f"integer. Got {_cfg.batch_size}")
        sys.exit(1)
    elif _cfg.batch_size <= 0:
        print("\n❌ batch_size should be a "
              f"strictly positive integer. Got {_cfg.batch_size}")
        sys.exit(1)
    else:
        batch_size = _cfg.batch_size

    if _cfg.n_cpus is None or type(_cfg.n_cpus) != int:
        print(f"\n❌ n_cpus should be a positive integer. Got {_cfg.n_cpus}")
        sys.exit(1)
    else:
        n_cpus = max(0, _cfg.n_cpus)

    modified_cfg['n_cpus'] = n_cpus
    if _cfg.transforms_add_missing_pad is None or len(_cfg.transforms_add_missing_pad) != 2 or type(
            _cfg.transforms_add_missing_pad[0]) != int or type(
        _cfg.transforms_add_missing_pad[1]) != int:
        print("\n❌ transforms_add_missing_pad should be an "
              f"a list of two integers. Got {_cfg.transforms_add_missing_pad}")
        sys.exit(1)
    else:
        add_missing_pad = list(_cfg.transforms_add_missing_pad)

    indep_keypoints, merge_keypoints, suffix_proba_files, rerun_create_proba = check_proba_parameters(dataset_path,
                                                                                                    config, _cfg, logger)

    if not 'original_missing' in config.keys():
        print("\n❌ Problem with `config_project.yaml`. No 'original_missing' key."
              "\nMake sure to run DISK-prepare-data first. \n"
              "If the problem persists, recreate a DISK project from scratch with DISK-create-project")
        sys.exit(1)

    if config['original_missing']:
        indep_keypoints = test_boolean_variable(_cfg.indep_keypoints, 'indep_keypoints')
        merge_keypoints = test_boolean_variable(_cfg.merge_keypoints, 'merge_keypoints')

        suffix_proba_files = f'_set_keypoints' if not indep_keypoints else ''
        if indep_keypoints:
            if merge_keypoints:
                logger.info(f'️ℹ merge_keypoints = True is not a valid option when indep_keypoints = True. '
                            f'merge_keypoints would be considered False')
            merge_keypoints = False
        else:
            if merge_keypoints:
                suffix_proba_files += f'_merged'

        proba_files_exist, _, _ = find_proba_files(dataset_path, suffix_proba_files)
        rerun_create_proba = False if proba_files_exist else True

    else:
        indep_keypoints = True
        merge_keypoints = False
        suffix_proba_files = '_uniform'

        proba_files_exist, _, _ = find_proba_files(dataset_path, suffix_proba_files)
        rerun_create_proba = False if proba_files_exist else True

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

    if _cfg.n_plots is None or type(_cfg.n_plots) != int:
        print("\n❌ n_plots should be a positive integer. "
              f"Got {_cfg.n_plots}")
        sys.exit(1)
    else:
        n_plots = max(0, _cfg.n_plots)

    if _cfg.pck_threshold is None or type(
            _cfg.pck_threshold) != float or _cfg.pck_threshold < 0 or _cfg.pck_threshold > 1:
        print("\n❌ pck_threshold should be a "
              f"float between 0 and 1. Got {_cfg.pck_threshold}")
        sys.exit(1)
    else:
        pck_threshold = _cfg.pck_threshold

    if _cfg.plot_azim3d is None or type(_cfg.plot_azim3d) != int:
        print("\n❌ plot_azim3d should be an integer."
              f"Got {_cfg.plot_azim3d}")
        sys.exit(1)
    else:
        plot3d_azim = _cfg.plot_azim3d

    if _cfg.plot_size3d is None or (type(_cfg.plot_size3d) != float and type(_cfg.plot_size3d) != int):
        print("\n❌ plot_size3d should be a "
              f"float. Got {_cfg.plot_size3d}")
        sys.exit(1)
    else:
        plot3d_size = _cfg.plot_size3d

    plot2d_only_holes = test_boolean_variable(_cfg.plot_only_holes2d, 'plot_only_holes2d')
    original_coordinates = test_boolean_variable(_cfg.plot_original_coordinates, 'plot_original_coordinates')

    if _cfg.n_repeat is None or type(_cfg.n_repeat) != int:
        print("\n❌ n_repeat should be a string."
              f"Got {_cfg.n_repeat}")
        sys.exit(1)
    else:
        n_repeat = max(1, _cfg.n_repeat)

    if _cfg.loss_def is None or type(_cfg.loss_def) != str or not _cfg.loss_def in ['l1', 'l2']:
        print("\n❌ loss_def should be l1 or l2."
              f"Got {_cfg.loss_def}")
        sys.exit(1)
    else:
        loss_def = _cfg.loss_def

    if _cfg.loss_mask is None or type(_cfg.loss_mask) != bool:
        print("\n❌ loss_mask should be a bool."
              f"Got {_cfg.loss_mask}")
        sys.exit(1)
    else:
        loss_mask = _cfg.loss_mask

    if _cfg.loss_factor is None or type(_cfg.loss_factor) != int:
        print("\n❌ loss_factor should be an integer."
              f"Got {_cfg.loss_factor}")
        sys.exit(1)
    else:
        loss_factor = max(1, _cfg.loss_factor)

    os.makedirs(os.path.join(test_dir, 'config'), exist_ok=True)
    output_config_file = os.path.join(test_dir, 'config', f'config_evaluate.yaml')
    copy_config_file(modified_cfg, output_config_file)

    logger.info(f'✅ Successfully loaded configuration.\n')

    add_missing = True
    main(project_path, model_path_list, dataset_path, dataset_name, test_dir,
         skeleton_graph,
         batch_size, loss_def, loss_mask, loss_factor,
         n_cpus,
         rerun_create_proba, indep_keypoints, merge_keypoints, suffix_proba_files,
         add_missing_pad, viewinvariant,
         normalize, normalizecube, swap,
         add_missing,
         original_coordinates, pck_threshold, n_repeat,
         n_plots, plot2d_only_holes,
         plot3d_size, plot3d_azim,
         logger, '', verbose)

    print('\n', '*' * 77, sep='')
    print('*' * 30, ' DISK-evaluate END ', '*' * 30)
    print('*' * 77, '\n')

    return

if __name__ == '__main__':
    cli()
