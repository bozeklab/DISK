import logging
from datetime import datetime
import os
import sys
import yaml

from DISK.utils.logger_setup import setup_custom_logging, copy_config_file
from DISK.launchers.train_test import find_proba_files
from DISK.models.graph import Graph
from DISK.utils.config_decorator import config_reader, parse_command_line_args

def main(project_dir, model_dirs, dataset_path, dataset_name, test_dir, skeleton_graph,
         training_batch_size,
         loss_type, loss_mask, loss_factor,
         n_cpus,
         proba_file, proba_length_file, indep_keypoints,
         add_missing_pad, viewinvariant, normalize, normalizecube, swap,
         add_missing,
         test_original_coordinates, test_threshold_pck,
         n_repeat,
         total_n_plots, plot2d_only_holes, plot3d_size, plot3d_azim,
         logger, verbose=0):

    from DISK.test_fillmissing import test

    logger.info(f'\n*********************** TESTING DISK TRAINED MODEL *********************** \n')

    test(project_dir, test_dir, dataset_path, dataset_name, skeleton_graph,
         model_dirs, training_batch_size, n_cpus,
         loss_type, loss_mask, loss_factor,
         proba_file, proba_length_file, indep_keypoints,
         add_missing_pad,
         viewinvariant, normalize, normalizecube, swap, add_missing,
         test_original_coordinates, test_threshold_pck, n_repeat,
         total_n_plots, plot2d_only_holes,
         plot3d_size, plot3d_azim,
         logger, suffix='', stride=None, verbose=verbose)
    logger.info(f'✅ Successfully tested DISK model.\n')


@config_reader(config_path="../conf/config_test.yaml")
def cli(_cfg) -> None:
    modified_cfg = dict(_cfg.__dict__)

    for key in ('project_path', 'dataset_name', 'model_name_list'):
        val = _cfg.__dict__[key]
        if val is None:
            print(f'\n❌ No value was passed to parameter {key}. This is a required parameter.'
                  f'\n  Expected syntax:'
                  f'\n  > DISK-test project_path=test_project dataset_name=dataset model_name_list=[model1,model2]\n'
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

    if not ('skeleton' in config.keys() and config['skeleton'] is not None and len(config['skeleton']) == 0):
        skeleton_graph = None
    else:
        skeleton_graph = Graph(len(config['keypoints']),
                 config['center'],
                 config['neighbor_links'],
                 config['neighbor_link_colors'])

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
    if _cfg.model_name_list is None or type(_cfg.model_name_list) != listconfig.ListConfig:
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
        test_dir = os.path.join(project_path, 'DISK_train', f'{datetime.today().strftime("%Y-%m-%d")}_test')
    else:
        if _cfg.name_output_dir is None or type(_cfg.name_output_dir) != str:
            print(f"\n❌ name_output_dir should be a "
                      f"string. Got {_cfg.name_output_dir}")
            sys.exit(1)
        else:
            test_dir = os.path.join(project_path, 'DISK_train', _cfg.name_output_dir)
    os.makedirs(test_dir, exist_ok=True)

    logger = setup_custom_logging(test_dir, 'test.log', logging_flag)

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

    if _cfg.transforms_indep_keypoints is None or type(_cfg.transforms_indep_keypoints) != bool:
        print("\n❌ transforms_indep_keypoints should be a "
              f"bool. Got {_cfg.transforms_indep_keypoints}")
        sys.exit(1)
    else:
        indep_keypoints = _cfg.transforms_indep_keypoints

    if _cfg.transforms_merge_keypoints is None or type(_cfg.transforms_merge_keypoints) != bool:
        print("\n❌ transforms_merge_keypoints should be a "
              f"bool. Got {_cfg.transforms_merge_keypoints}")
        sys.exit(1)
    else:
        merge_keypoints = _cfg.transforms_merge_keypoints

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

    if _cfg.transforms_viewinvariant is None or type(_cfg.transforms_viewinvariant) != bool:
        print("\n❌ transforms_viewinvariant should be a "
              f"bool. Got {_cfg.transforms_viewinvariant}")
        sys.exit(1)
    else:
        viewinvariant = _cfg.transforms_viewinvariant

    if _cfg.transforms_normalize is None or type(_cfg.transforms_normalize) != bool:
        print("\n❌ transforms_normalize should be a "
              f"bool. Got {_cfg.transforms_normalize}")
        sys.exit(1)
    else:
        normalize = _cfg.transforms_normalize

    if _cfg.transforms_normalizecube is None or type(_cfg.transforms_normalizecube) != bool:
        print("\n❌ transforms_normalizecube should be a "
              f"bool. Got {_cfg.transforms_normalizecube}")
        sys.exit(1)
    else:
        normalizecube = _cfg.transforms_normalizecube

    if _cfg.transforms_swap is None or type(
            _cfg.transforms_swap) != float or _cfg.transforms_swap < 0 or _cfg.transforms_swap > 1:
        print("\n❌ transforms_swap should be a float between 0 and 1 "
              "(probability of swapping during training). "
              f"Got {_cfg.transforms_swap}")
        sys.exit(1)
    else:
        swap = _cfg.transforms_swap

    if _cfg.test.n_plots is None or type(_cfg.test.n_plots) != int:
        print("\n❌ test.n_plots should be a positive integer. "
              f"Got {_cfg.test.n_plots}")
        sys.exit(1)
    else:
        n_plots = max(0, _cfg.test.n_plots)

    if _cfg.test.threshold_pck is None or type(
            _cfg.test.threshold_pck) != float or _cfg.test.threshold_pck < 0 or _cfg.test.threshold_pck > 1:
        print("\n❌ test.threshold_pck should be a "
              f"float between 0 and 1. Got {_cfg.test.threshold_pck}")
        sys.exit(1)
    else:
        threshold_pck = _cfg.test.threshold_pck

    if _cfg.test.plot3d_azim is None or type(_cfg.test.plot3d_azim) != int:
        print("\n❌ test.plot3d_azim should be an integer."
              f"Got {_cfg.test.plot3d_azim}")
        sys.exit(1)
    else:
        plot3d_azim = _cfg.test.plot3d_azim

    if _cfg.test.plot3d_size is None or type(_cfg.test.plot3d_size) != float:
        print("\n❌ test.plot3d_size should be a "
              f"float. Got {_cfg.test.plot3d_size}")
        sys.exit(1)
    else:
        plot3d_size = _cfg.test.plot3d_size

    if _cfg.test.plot2d_only_holes is None or type(_cfg.test.plot2d_only_holes) != bool:
        print("\n❌ test.plot2d_only_holes should be a "
              f"bool. Got {_cfg.test.plot2d_only_holes}")
        sys.exit(1)
    else:
        plot2d_only_holes = _cfg.test.plot2d_only_holes

    if _cfg.test.original_coordinates is None or type(_cfg.test.original_coordinates) != bool:
        print("\n❌ test.original_coordinates should be a "
              f"bool. Got {_cfg.test.original_coordinates}")
        sys.exit(1)
    else:
        original_coordinates = _cfg.test.original_coordinates


    if _cfg.test.n_repeat is None or type(_cfg.test.n_repeat) != int:
        print("\n❌ test.n_repeat should be a string."
              f"Got {_cfg.test.n_repeat}")
        sys.exit(1)
    else:
        n_repeat = max(1, _cfg.test.n_repeat)

    if _cfg.loss.type is None or type(_cfg.loss.type) != str or not _cfg.loss.type in ['l1', 'l2']:
        print("\n❌ loss.type should be l1 or l2."
              f"Got {_cfg.loss.type}")
        sys.exit(1)
    else:
        loss_type = _cfg.loss.type

    if _cfg.loss.mask is None or type(_cfg.loss.mask) != bool:
        print("\n❌ loss.mask should be a bool."
              f"Got {_cfg.loss.mask}")
        sys.exit(1)
    else:
        loss_mask = max(1, _cfg.loss.mask)

    if _cfg.loss.factor is None or type(_cfg.loss.factor) != int:
        print("\n❌ loss.factor should be an integer."
              f"Got {_cfg.loss.factor}")
        sys.exit(1)
    else:
        loss_factor = max(1, _cfg.loss.factor)

    os.makedirs(os.path.join(test_dir, 'config'), exist_ok=True)
    output_config_file = os.path.join(test_dir, 'config', f'config_test.yaml')
    copy_config_file(modified_cfg, output_config_file)

    logger.info(f'✅ Successfully loaded configuration.\n')

    add_missing = True
    main(project_path, model_path_list, dataset_path, dataset_name, test_dir,
         skeleton_graph,
         batch_size, loss_type, loss_mask, loss_factor,
         n_cpus,
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
