import os
import sys
import yaml
import logging

from DISK.launchers.train_evaluate import check_proba_parameters
from DISK.utils.logger_setup import setup_custom_logging, copy_config_file, VoidHandler
from DISK.models.graph import Graph
from DISK.utils.config_decorator import config_reader, parse_command_line_args, test_boolean_variable

def main(project_path, dataset_path, dataset_name, data_files, file_format,
         length, stride, fill_gap, sequential, original_freq, subsampling_freq,
         dlc_likelihood_threshold, discard_beginning, discard_end,
         drop_keypoints, indep_keypoints, merge_keypoints,
         suffix_proba_files, skeleton_graph, logger):

    from DISK.create_dataset import create_dataset
    from DISK.create_proba_missing_files import create_proba_missing_files
    from DISK.utils.transforms import init_transforms
    from DISK.utils.dataset_utils import load_datasets

    number_samples_train, keypoints, divider = create_dataset(
                                dataset_path,
                                data_files,
                                file_format,
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
        warning_signs = '⚠️️⚠️️⚠️ ' if number_samples_train < 1000 else '⚠️️ '
        logger.info(f'{warning_signs}The training set created for DISK has only {number_samples_train}. This risks to be too '
                  f'small for the training. We recommend at least 2000 training samples.'
              f'\nTry relaunching DISK-prepare-data with a higher fill_gap value and/or '
              f'lower stride value.\n')


    no_original_missing, indep_keypoints, merge_keypoints, suffix = create_proba_missing_files(project_path,
                                                                                               dataset_path,
                                                                                               indep_keypoints,
                                                                                               merge_keypoints,
                                                                                               suffix_proba_files,
                                                                                               skeleton_graph, logger)
    logger.info(f'✅ Successfully estimated probabilities of missing keypoints for dataset {dataset_name}.\n')

    logger.info(f'ℹ️ Checking if imputable segments in dataset {dataset_name}.\n')
    transforms = init_transforms(
                                keypoints,
                                divider,
                                length,
                                dataset_path,
                                logger,
                                add_missing=False,
                                viewinvariant=False,
                                normalize=False,
                                normalizecube=False,
                                swap=0)


    # return full length dataset for imputation
    train_dataset, val_dataset, test_dataset = load_datasets(
                                                        dataset_path=dataset_path,
                                                        transform=transforms,
                                                        dataset_type='impute',
                                                        suffix='_w-all-nans',
                                                        root_path=project_path,
                                                        outputdir=dataset_path,
                                                        keypoints=keypoints,
                                                        label_type='all',  # don't care, not using
                                                        verbose=0,
                                                        padding=(1, 0), # minimal padding
                                                        skeleton_graph=skeleton_graph,
                                                        seq_length=length,
                                                        stride=stride,
                                                        freq=subsampling_freq,
                                                        divider=divider,
                                                        logger=logger
                                                )

    if len(train_dataset) == 0 and len(val_dataset) == 0 and len(test_dataset) == 0:
        # if the length of all datasets is null, then nothing to impute
        # in the dataset are only listed samples of "possible_indices" which are gaps
        # which length is <= DISK_model_length - pad_before - pad_after
        logger.info(
            f'⚠️️⚠️️⚠️ It seems the created datasets does not have gaps that will be imputable. \n'
            f'We recommend to increase the --length and --stride values, and if available add more data files. \n'
            f'(This has been computed with pad (1, 0) which is the minimal recommended padding.)\n')

    return keypoints, divider, no_original_missing, indep_keypoints, merge_keypoints, suffix

@config_reader(config_path="../conf/config_prepare_data.yaml")
def cli(_cfg) -> None:
    print('\n', '*' * 87, sep='')
    print('*' * 30, ' DISK-PREPARE-DATA START ', '*' * 30)
    print('*' * 87, '\n')

    _cfg = parse_command_line_args(_cfg)
    modified_cfg = dict(_cfg.__dict__)

    for key in ('length', 'project_path'):
        val = _cfg.__dict__[key]
        if val is None or val == '_DEFAULT_':
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

    ### _CFG PARAMETER CHECK --- OFTEN CHANGED PARAMETERS
    if _cfg.stride == '_DEFAULT_':
        stride = max(length // 2, 1)
    else:
        if type(_cfg.stride) != int:
            print("\n❌ stride is a required parameter and should be a "
                  f"strictly positive integer. Got {_cfg.stride}")
            sys.exit(1)
        stride = max(_cfg.stride, 1)

    if _cfg.fill_gap == '_DEFAULT_':
        fill_gap = 0
    elif _cfg.fill_gap is None or type(_cfg.fill_gap) != int:
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
    file_format = config['file_format']
    if _cfg.sequential == '_DEFAULT_':
        if number_data_files <= 6:
            sequential = True
        else:
            sequential = False
    else:
        sequential = test_boolean_variable(_cfg.sequential, 'sequential')
        if number_data_files < 3 and sequential is False:
            print(f"\n❌ sequential cannot be set to False if there are less than 3 files. "
                  f"Got {_cfg.sequential}")
            sys.exit(1)


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
            dataset_name = f'dataset_length{length}_stride{stride}'
        else:
            dataset_name = f'dataset_{subsampling_freq}Hz_length{length}_stride{stride}'
        if sequential:
            dataset_name += '_sequential'
    else:
        if _cfg.dataset_name is None or type(_cfg.dataset_name) != str:
            print("\n❌ dataset_name should be a "
                  f"string. Got {_cfg.dataset_name}")
            sys.exit(1)
        else:
            dataset_name = _cfg.dataset_name

    dataset_path = os.path.join(project_path, 'DISK_data', dataset_name,)
    if os.path.exists(dataset_path):
        print(f"⚠️  Dataset {dataset_path} already exists. Do you want to rewrite files in the same folder? [y/n]")
        y_n = input('> ')
        while y_n not in ['y', 'n', 'Y', 'N', 'yes', 'no', 'Yes', 'YES', 'No', 'NO']:
            y_n = input('Retype y or n: ')
        if y_n in ['y', 'Y', 'yes', 'Yes', 'YES']:
            pass
        else:
            ext_dataset_path = 1
            final_dataset_path = str(dataset_path)
            while os.path.exists(final_dataset_path):
                final_dataset_path = dataset_path + f'_{ext_dataset_path}'
                ext_dataset_path += 1
            dataset_path = final_dataset_path
            print(f"\nℹ️ Not overwriting, instead creating new folder {dataset_path}.")
    else:
        os.mkdir(dataset_path)

    if _cfg.debug:
        logging_flag = logging.DEBUG
    else:
        logging_flag = logging.INFO

    logging.basicConfig(level=logging_flag, handlers=[VoidHandler()])
    logger = setup_custom_logging(dataset_path, 'prepare_data.log', logging_flag)

    if not 'skeleton' in config.keys() or config['skeleton'] is None or len(config['skeleton'])\
            == 0:
        skeleton_graph = None
    else:
        skeleton_graph = Graph(len(config['keypoints']),
                 config['skeleton_center'],
                 config['skeleton'],
                 config['skeleton_colors'], logger=logger)

    if _cfg.dlc_likelihood_threshold == '_DEFAULT':
        dlc_likelihood_threshold = 0.9
    elif _cfg.dlc_likelihood_threshold is None or type(_cfg.dlc_likelihood_threshold) != float:
        print("\n❌ dlc_likelihood_threshold should be a "
              f"float. Got {_cfg.dlc_likelihood_threshold}")
        sys.exit(1)
    else:
        dlc_likelihood_threshold = _cfg.dlc_likelihood_threshold

    if config['file_format'] in ['dlc_h5', 'dlc_csv']:
        logger.info(f'ℹ️ Using a threshold of {_cfg.dlc_likelihood_threshold} for DLC likelihood. '
                    f'Any coordinate with a likelihood under {_cfg.dlc_likelihood_threshold} will be considered '
              f'missing.\n')

    if _cfg.discard_beginning == '_DEFAULT_':
        discard_beginning = 0
    elif _cfg.discard_beginning is None or type(_cfg.discard_beginning) != int:
        print("\n❌ discard_beginning should be an "
              f"integer. Got {_cfg.discard_beginning}")
        sys.exit(1)
    else:
        discard_beginning = _cfg.discard_beginning

    if _cfg.discard_end == '_DEFAULT_':
        discard_end = -1
    elif _cfg.discard_end is None or type(_cfg.discard_end) != int:
        print("\n❌ discard_end should be an "
              f"integer. Got {_cfg.discard_end}")
        sys.exit(1)
    else:
        discard_end = _cfg.discard_end


    if _cfg.drop_keypoints == '_DEFAULT_':
        drop_keypoints = []
    elif _cfg.drop_keypoints is None:
        print("\n❌ drop_keypoints should be a (empty)"
              f"list. Got {_cfg.drop_keypoints}")
        sys.exit(1)
    else:
        drop_keypoints = list(_cfg.drop_keypoints)

    indep_keypoints, merge_keypoints, suffix_proba_files, rerun_create_proba = check_proba_parameters(dataset_path,
                                                                                                    config, _cfg, logger)

    os.makedirs(os.path.join(dataset_path, 'config'), exist_ok=True)

    output_config_file = os.path.join(dataset_path, 'config', f'config_prepare_data.yaml')
    copy_config_file(modified_cfg, output_config_file)

    logger.info(f'✅ Successfully loaded configuration.\n')

    keypoints, divider, no_original_missing, indep_keypoints, merge_keypoints, suffix = main(project_path,
                                                                                             dataset_path,
                                                                                             dataset_name,
                                                                                             data_files, file_format,
         length, stride, fill_gap, sequential, original_freq, subsampling_freq,
         dlc_likelihood_threshold, discard_beginning, discard_end,
         drop_keypoints, indep_keypoints, merge_keypoints, suffix_proba_files, skeleton_graph,
         logger)

    config['keypoints'] = keypoints
    config['divider'] = divider
    config['original_missing'] = not no_original_missing

    with open(os.path.join(project_path, 'config_project.yaml'), 'w') as file:
        yaml.safe_dump(config, file)

    print('\n', '*' * 85, sep='')
    print('*' * 30, ' DISK-PREPARE-DATA END ', '*' * 30)
    print('*' * 85, '\n')

    return

if __name__ == '__main__':
    cli()