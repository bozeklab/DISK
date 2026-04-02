import logging
from datetime import datetime
import os
import sys
import yaml

from DISK.utils.logger_setup import setup_custom_logging, copy_config_file, VoidHandler
from DISK.models.graph import Graph
from DISK.utils.config_decorator import config_reader, parse_command_line_args

def main(project_dir, impute_dir, plot_dir, file_type, dataset_path, skeleton_graph, checkpoint, batch_size,
         threshold_error_score, total_n_plots, plot_only_holes, missing_pad, logger, verbose=0):

    from DISK.impute_dataset import impute

    if impute(project_dir, impute_dir, plot_dir, file_type, dataset_path, skeleton_graph, checkpoint, batch_size,
           threshold_error_score, total_n_plots, plot_only_holes, missing_pad, verbose=verbose, logger=logger):
        mess = f'✅ Successfully imputed data with DISK model.\n'
        logger.info(mess)
    else:
        mess = (f'❌ Could not find short-enough segments to be imputed by the DISK model.\n'
                f'Re-run DISK-prepare-data with higher value for length.')
        logger.info(mess)

    return mess

@config_reader(config_path="../conf/config_impute.yaml")
def cli(_cfg) -> None:
    print('\n', '*' * 81, sep='')
    print('*' * 30, ' DISK-IMPUTE START ', '*' * 30)
    print('*' * 81, '\n')

    _cfg = parse_command_line_args(_cfg)
    modified_cfg = dict(_cfg.__dict__)

    for key in ('project_path', 'dataset_name', 'model_name'):
        val = _cfg.__dict__[key]
        if val is None or val == '_DEFAULT_':
            print(f'\n❌ No value was passed to parameter {key}. This is a required parameter.'
                  f'\n  Expected syntax:'
                  f'\n  > DISK-impute --project_path test_project --dataset_name dataset --model_name model')
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

    if not config['original_missing']:
        print(f'ℹ️ No Missing keypoints in the original files. DISK will NOT impute.')
        sys.exit(0)

    file_type = config['file_type']

    if _cfg.dataset_name is None or type(_cfg.dataset_name) != str \
            or not os.path.exists(os.path.join(project_path, 'DISK_data', _cfg.dataset_name)):
        print("\n❌ dataset_name is a required parameter and should be the name "
              "of an existing dataset within subfolder DISK_data. "
              f"  Got {_cfg.dataset_name} {os.path.join(project_path, 'DISK_data', _cfg.dataset_name)}")
        sys.exit(1)
    else:
        dataset_name = _cfg.dataset_name

    dataset_path = os.path.join(project_path, 'DISK_data', dataset_name)

    if _cfg.model_name is not None and type(_cfg.model_name) == str:
        model_path = os.path.join(project_path, 'DISK_train', _cfg.model_name)
        if not os.path.exists(model_path):
            print(f"\n❌ model_name is a required parameter, "
                  f"but could not find the checkpoint"
                  f" at {_cfg.model_name}.")
            sys.exit(1)
        model_name = _cfg.model_name
    else:
        print(f"\n❌ model_name is a required parameter (str). "
              f"Got {_cfg.model_name}.")
        sys.exit(1)

    output_path = os.path.join(project_path, 'DISK_impute', model_name)
    os.makedirs(output_path, exist_ok=True)

    if _cfg.debug:
        logging_flag = logging.DEBUG
        verbose = 1
    else:
        logging_flag = logging.INFO
        verbose = 0

    modified_cfg['model_name'] = os.path.basename(output_path)
    logging.basicConfig(level=logging_flag, handlers=[VoidHandler()])

    logger = setup_custom_logging(output_path, 'impute.log', logging_flag)

    if not ('skeleton' in config.keys() and config['skeleton'] is not None and len(config['skeleton']) == 0):
        skeleton_graph = None
    else:
        skeleton_graph = Graph(len(config['keypoints']),
                 config['center'],
                 config['neighbor_links'],
                 config['neighbor_link_colors'], logger=logger)

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

    if _cfg.n_plots is None or type(_cfg.n_plots) != int:
        print("\n❌ n_plots should be a positive integer. "
              f"Got {_cfg.n_plots}")
        sys.exit(1)
    else:
        n_plots = max(0, _cfg.n_plots)

    if _cfg.threshold_error_score is not None:
        if _cfg.threshold_error_score == '_DEFAULT_':
            import numpy as np
            threshold_error_score = np.inf
        elif type(_cfg.threshold_error_score) != float or _cfg.threshold_error_score < 0:
            print("\n❌ threshold_error_score should be a strictly positive "
                  f"float. Got {_cfg.threshold_error_score}")
            sys.exit(1)
        else:
            threshold_error_score = _cfg.threshold_error_score

    else:
        print("\n❌ threshold_error_score should be a strictly positive"
              f"float. Got {_cfg.threshold_error_score}")
        sys.exit(1)

    if _cfg.plot_only_holes is None or type(_cfg.plot_only_holes) != bool:
        print("\n❌ plot_only_holes should be a "
              f"bool. Got {_cfg.plot_only_holes}")
        sys.exit(1)
    else:
        plot_only_holes = _cfg.plot_only_holes

    if _cfg.missing_pad is None or len(_cfg.missing_pad) != 2 or type(
            _cfg.missing_pad[0]) != int or type(
        _cfg.missing_pad[1]) != int or _cfg.missing_pad[0] < 1 or _cfg.missing_pad[1] < 0:
        print("\n❌ missing_pad should be an "
              f"a list of two positive integers, the first one strictly positive. "
              f"Got {_cfg.missing_pad}")
        sys.exit(1)
    else:
        missing_pad = list(_cfg.missing_pad)

    os.makedirs(os.path.join(output_path, 'config'), exist_ok=True)

    output_config_file = os.path.join(output_path, 'config', f'config_train.yaml')
    copy_config_file(modified_cfg, output_config_file)

    impute_plots_dir = os.path.join(output_path, f'{datetime.today().strftime("%Y-%m-%d")}_impute_plots')
    os.makedirs(impute_plots_dir, exist_ok=True)

    logger.info(f'✅ Successfully loaded configuration.\n')

    main(project_path, output_path, impute_plots_dir, file_type, dataset_path, skeleton_graph, model_path,
         batch_size, threshold_error_score, n_plots, plot_only_holes, missing_pad, logger, verbose=verbose)

    print('\n', '*' * 79, sep='')
    print('*' * 30, ' DISK-IMPUTE END ', '*' * 30)
    print('*' * 79, '\n')

    return

if __name__ == '__main__':
    cli()
