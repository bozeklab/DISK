import os
import sys
import yaml
import logging

from DISK.utils.logger_setup import setup_custom_logging, copy_config_file, VoidHandler
from DISK.models.graph import Graph
from DISK.utils.config_decorator import config_reader, parse_command_line_args, test_boolean_variable

def main(project_path, dataset_path, dataset_name, data_files, file_type,
         length, stride, fill_gap, sequential, original_freq, subsampling_freq,
         dlc_likelihood_threshold, discard_beginning, discard_end,
         drop_keypoints, indep_keypoints, merge_keypoints, skeleton_graph,
         logger):

    from DISK.create_dataset import create_dataset
    from DISK.create_proba_missing_files import create_proba_missing_files

    number_samples_train, keypoints, divider = create_dataset(
                                dataset_path,
                                data_files,
                                file_type,
                                length,
                                stride,
                                fill_gap,
                                sequential,
                                original_freq,
                                subsampling_freq,
                                dlc_likelihood_threshold,
                                discard_beginning,
                                discard_end,
                                drop_keypoints,
                                logger
                            )
    logger.info(f'✅ Successfully created dataset {dataset_name}.\n')

    if number_samples_train < 2000:
        logger.info(f'⚠️️⚠️️⚠️ The training set created for DISK has only {number_samples_train}. This risks to be too '
                  f'small for the training. '
              f'\nTry relaunching DISK-prepare-data with a higher fill_gap value and/or '
              f'lower stride value.\n')


    create_proba_missing_files(project_path, dataset_path, indep_keypoints, merge_keypoints, skeleton_graph, logger)
    logger.info(f'✅ Successfully estimated probabilities of missing keypoints for dataset {dataset_name}.\n')
    return keypoints, divider

@config_reader(config_path="../conf/config_prepare_data.yaml")
def cli(_cfg) -> None:
    print(_cfg)
    _cfg = parse_command_line_args(_cfg)
    modified_cfg = dict(_cfg.__dict__)

    for key in ('length', 'project_path'):
        val = _cfg.__dict__[key]
        if val is None:
            print(f'\n❌ No value was passed to parameter {key}. This is a required parameter.'
                  f'\n  Expected syntax:'
                  f'\n  > DISK-prepare-data --project_path test_project --length 60\n')
            sys.exit(1)

    ### _CFG PARAMETER CHECK --- REQUIRED PARAMETERS
    if _cfg.length is None or type(_cfg.length) != int:
        print("\n❌ length is a required parameter and should be an integer. "
              f"  Got {_cfg.length}")
        sys.exit(1)
    else:
        length = _cfg.length

    if _cfg.project_path is None or type(_cfg.project_path) != str:
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

    if not 'skeleton' in config.keys() or config['skeleton'] is None or len(config['skeleton'])\
            == 0:
        skeleton_graph = None
    else:
        skeleton_graph = Graph(len(config['keypoints']),
                 config['skeleton_center'],
                 config['skeleton_links'],
                 config['skeleton_colors'])
    ### _CFG PARAMETER CHECK --- OFTEN CHANGED PARAMETERS

    if _cfg.stride == '_DEFAULT_':
        stride = max(length // 2, 1)
    else:
        if type(_cfg.stride) != int:
            print("\n❌ stride is a required parameter and should be a "
                  f"strictly positive integer. Got {_cfg.stride}")
            sys.exit(1)
        stride = max(_cfg.stride, 1)

    if _cfg.fill_gap is None or type(_cfg.fill_gap) != int:
        print("\n❌ fill_gap is a required parameter and should be an "
              f"integer. Got {_cfg.fill_gap}")
        sys.exit(1)
    else:
        if _cfg.fill_gap > 100 or _cfg.fill_gap > length // 2:
            print(f"\n⚠️️⚠️️⚠️  fill_gap has a value of {_cfg.fill_gap}, "
                  "which is unusually large. Its value depends on the"
                  " sampling frequency, but the rule of thumb is that "
                  "during this duration, no strong / discontinuous movement "
                  "should be observed.")
        fill_gap = _cfg.fill_gap

    data_files = config['data_files']
    number_data_files = len(data_files)
    file_type = config['file_type']
    if _cfg.sequential == '_DEFAULT_':
        if number_data_files <= 6:
            sequential = True
        else:
            sequential = False
    else:
        sequential = test_boolean_variable(_cfg.sequential, 'sequential')

    if _cfg.original_freq == '_DEFAULT_':
        original_freq = 1
        subsampling_freq = 1
    else:
        if _cfg.original_freq is None or type(_cfg.original_freq) != int:
            print("\n❌ original_freq should be an "
                  f"integer or _DEFAULT_. Got {_cfg.original_freq}")
            sys.exit(1)
        else:
            original_freq = _cfg.original_freq

        if _cfg.subsampling_freq == '_DEFAULT_':
            subsampling_freq = int(original_freq)
        else:
            if _cfg.subsampling_freq is None or type(_cfg.subsampling_freq) != int:
                print("\n❌ subsampling_freq should be an "
                      f"integer or _DEFAULT_. Got {_cfg.subsampling_freq}")
                sys.exit(1)
            elif _cfg.subsampling_freq > _cfg.original_freq:
                print("\n❌ subsampling_freq should be _DEFAULT_ or an integer <= to original_freq. "
                      f"Got subsampling_freq: {_cfg.subsampling_freq} > original_freq: {_cfg.original_freq}")
                sys.exit(1)
            else:
                subsampling_freq = _cfg.subsampling_freq

    config['original_freq'] = original_freq
    config['subsampling_freq'] = subsampling_freq

    if _cfg.dataset_name == '_DEFAULT_':
        if subsampling_freq == 1:
            dataset_name = f'dataset_{length}_{stride}'
        else:
            dataset_name = f'dataset_{subsampling_freq}Hz_{length}length_{stride}stride'
    else:
        if _cfg.dataset_name is None or type(_cfg.dataset_name) != str:
            print("\n❌ dataset_name should be a "
                  f"string. Got {_cfg.dataset_name}")
            sys.exit(1)
        else:
            dataset_name = _cfg.dataset_name

    dataset_path = os.path.join(project_path, 'DISK_data', dataset_name,)
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)

    if _cfg.debug:
        logging_flag = logging.DEBUG
    else:
        logging_flag = logging.INFO

    logging.basicConfig(level=logging_flag, handlers=[VoidHandler()])

    logger = setup_custom_logging(dataset_path, 'prepare_data.log', logging_flag)

    if _cfg.dlc_likelihood_threshold is None or type(_cfg.dlc_likelihood_threshold) != float:
        print("\n❌ dlc_likelihood_threshold should be a "
              f"float. Got {_cfg.dlc_likelihood_threshold}")
        sys.exit(1)
    else:
        dlc_likelihood_threshold = _cfg.dlc_likelihood_threshold

    if config['file_type'] in ['dlc_h5', 'dlc_csv']:
        logger.info(f'ℹ️ Using a threshold of {_cfg.dlc_likelihood_threshold} for DLC likelihood. '
                    f'Any coordinate with a likelihood under {_cfg.dlc_likelihood_threshold} will be considered '
              f'missing.\n')

    if _cfg.discard_beginning is None or type(_cfg.discard_beginning) != int:
        print("\n❌ discard_beginning should be an "
              f"integer. Got {_cfg.discard_beginning}")
        sys.exit(1)
    else:
        discard_beginning = _cfg.discard_beginning

    if _cfg.discard_end is None or type(_cfg.discard_end) != int:
        print("\n❌ discard_end should be an "
              f"integer. Got {_cfg.discard_end}")
        sys.exit(1)
    else:
        discard_end = _cfg.discard_end

    if _cfg.drop_keypoints is None:
        print("\n❌ drop_keypoints should be a (empty)"
              f"list. Got {_cfg.drop_keypoints}")
        sys.exit(1)
    else:
        drop_keypoints = list(_cfg.drop_keypoints)

    indep_keypoints = test_boolean_variable(_cfg.indep_keypoints, 'indep_keypoints')
    merge_keypoints = test_boolean_variable(_cfg.merge_keypoints, 'merge_keypoints')

    os.makedirs(os.path.join(dataset_path, 'config'), exist_ok=True)

    output_config_file = os.path.join(dataset_path, 'config', f'config_prepare_data.yaml')
    copy_config_file(modified_cfg, output_config_file)

    logger.info(f'✅ Successfully loaded configuration.\n')

    keypoints, divider = main(project_path, dataset_path, dataset_name, data_files, file_type,
         length, stride, fill_gap, sequential, original_freq, subsampling_freq,
         dlc_likelihood_threshold, discard_beginning, discard_end,
         drop_keypoints, indep_keypoints, merge_keypoints, skeleton_graph,
         logger)

    config['keypoints'] = keypoints
    config['divider'] = divider

    with open(os.path.join(project_path, 'config_project.yaml'), 'w') as file:
        yaml.dump(config, file)

    return

if __name__ == '__main__':
    cli()