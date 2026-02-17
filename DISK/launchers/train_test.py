import logging
from datetime import datetime
import hydra
from omegaconf import DictConfig, OmegaConf
import os
import sys
import yaml

from DISK.utils.logger_setup import setup_custom_logging, copy_config_file


def main(project_dir, model_dir, dataset_path, dataset_name, test_dir, skeleton_file,
         training_seed, load_model, cfg_network, training_batch_size,
         training_epochs, learning_rate, loss_type, loss_mask, loss_factor,
         model_scheduler_rate, model_scheduler_type, model_scheduler_steps_epoch,
         n_cpus, print_every,
         proba_file, proba_length_file, indep_keypoints,
         add_missing_pad, viewinvariant, normalize, normalizecube, swap,
         add_missing,
         test_name_items, test_merge, test_original_coordinates, test_threshold_pck,
         n_repeat, merge_sets_file,
         total_n_plots, plot2d_only_holes, plot3d_size, plot3d_azim,
         suffix, logger, verbose=0):

    from DISK.main_fillmissing import train_fillmissing
    from DISK.test_fillmissing import test

    logger.info(f'\n*********************** TRAINING DISK *********************** \n')
    train_fillmissing(project_dir, model_dir, dataset_path, skeleton_file, training_seed,
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
                      logger, verbose=verbose)

    logger.info(f'✅ Successfully trained DISK model.\n')

    logger.info(f'\n*********************** TESTING DISK TRAINED MODEL *********************** \n')

    test(project_dir, test_dir, dataset_path, dataset_name, skeleton_file,
         [model_dir, ], training_batch_size, n_cpus,
         loss_mask, loss_factor,
         proba_file, proba_length_file, indep_keypoints,
         add_missing_pad,
         viewinvariant, normalize, normalizecube, swap, add_missing,
         test_name_items, test_merge,
         test_original_coordinates, test_threshold_pck, n_repeat,
         merge_sets_file, total_n_plots, plot2d_only_holes,
         plot3d_size, plot3d_azim,
         logger, suffix=suffix, stride=None, verbose=verbose)
    logger.info(f'✅ Successfully tested DISK model.\n')


@hydra.main(version_base=None, config_path="../conf", config_name="config_train")
def cli(_cfg: DictConfig) -> None:
    modified_cfg = DictConfig(_cfg)

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

    project_name = config['project_name']
    skeleton_file_path = config['skeleton']

    if _cfg.dataset_name is None or type(_cfg.dataset_name) != str \
            or not os.path.exists(os.path.join(project_path, 'DISK_data', _cfg.dataset_name)):
        print("\n❌ dataset_name is a required parameter and should be the name "
              "of an existing dataset within subfolder DISK_data. "
              f"  Got {_cfg.dataset_name} {os.path.join(project_path, 'DISK_data', _cfg.dataset_name)}")
        sys.exit(1)
    else:
        dataset_name = _cfg.dataset_name

    dataset_path = os.path.join(project_path, 'DISK_data', dataset_name)

    final_model_path = None
    load_model = None
    if _cfg.model_name == '_DEFAULT_':
        if _cfg.load_model is not None and type(_cfg.load_model) == str:
            final_model_path = os.path.join(project_path, 'DISK_train', _cfg.load_model)
            if not os.path.exists(final_model_path):
                print(f"\n❌ You provided a load_model entry, but could not find the checkpoint at {final_model_path}.")
                sys.exit(1)
            model_name = _cfg.load_model
        else:
            if _cfg.network.type == 'transformer':
                network_name = 'DISK'
            else:
                network_name = f'DISK-{_cfg.network.type}'
            model_name = f'{dataset_name}_{network_name}'
    else:
        if _cfg.model_name is None or type(_cfg.model_name) != str:
            print("\n❌ model_name should be a "
                  f"string. Got {_cfg.model_name}")
            sys.exit(1)
        else:
            model_name = _cfg.model_name

    if final_model_path is None:
        ext_model_path = 1
        model_path = os.path.join(project_path, 'DISK_train', model_name)
        final_model_path = str(model_path)
        while os.path.exists(final_model_path):
            final_model_path = model_path + f'_{ext_model_path}'
            ext_model_path += 1

        os.mkdir(final_model_path)

    if _cfg.debug:
        logging_flag = logging.DEBUG
        verbose = 1
    else:
        logging_flag = logging.INFO
        verbose = 0

    modified_cfg.model_name = os.path.basename(final_model_path)
    logger = setup_custom_logging(final_model_path, 'train.log', logging_flag)

    ### _CFG PARAMETER CHECK --- OFTEN CHANGED PARAMETERS

    if _cfg.training_epochs is None or type(_cfg.training_epochs) != int:
        print("\n❌ training_epochs is a required parameter and should be a "
              f"strictly positive integer. Got {_cfg.training_epochs}")
        sys.exit(1)
    elif _cfg.training_epochs <= 0:
        print("\n❌ training_epochs is a required parameter and should be a "
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

    modified_cfg.n_cpus = n_cpus

    if _cfg.training_learning_rate == '_DEFAULT_':
        if _cfg.network.type == 'transformer':
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

    modified_cfg.training_learning_rate = learning_rate

    if _cfg.training_seed is None:
        training_seed = None
    else:
        if _cfg.training_seed is None or type(_cfg.training_seed) != int:
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
            print_every = max(1, _cfg.print_every)
    modified_cfg.print_every = print_every

    if _cfg.transforms.add_missing_pad is None or len(_cfg.transforms.add_missing_pad) != 2 or type(
            _cfg.transforms.add_missing_pad[0]) != int or type(
        _cfg.transforms.add_missing_pad[1]) != int:
        print("\n❌ transforms.add_missing_pad should be an "
              f"a list of two integers. Got {_cfg.transforms.add_missing_pad}")
        sys.exit(1)
    else:
        add_missing_pad = list(_cfg.transforms.add_missing_pad)

    if _cfg.transforms.add_missing_indep_keypoints is None or type(_cfg.transforms.add_missing_indep_keypoints) != bool:
        print("\n❌ transforms.add_missing_indep_keypoints should be a "
              f"bool. Got {_cfg.transforms.add_missing_indep_keypoints}")
        sys.exit(1)
    else:
        add_missing_indep_keypoints = _cfg.transforms.add_missing_indep_keypoints

    if _cfg.transforms.add_missing_proba_file_type is None or type(_cfg.transforms.add_missing_proba_file_type) != str:
        print("\n❌ transforms.add_missing_proba_file_type should be a string within the following ('set_keypoints', "
              "'set_keypoints_merged', '', )"
              f"Got {_cfg.transforms.add_missing_proba_file_type}")
        sys.exit(1)
    else:
        suffix = _cfg.transforms.add_missing_proba_file_type
        if len(suffix) > 0 and suffix[0] != '_':
            suffix = '_' + suffix

        proba_file = os.path.join(dataset_path, f'proba_missing{suffix}.csv')
        proba_length_file = os.path.join(dataset_path, f'proba_missing_length{suffix}.csv')

        if not os.path.exists(proba_file) or not os.path.exists(proba_length_file):
            from DISK.create_proba_missing_files import create_proba_missing_files
            indep_keypoints = False if 'set_keypoints' in suffix else True
            merge_keypoints = True if ('merged' in suffix and not indep_keypoints) else True

            create_proba_missing_files(project_path, dataset_path, indep_keypoints, merge_keypoints, skeleton_file_path,
                                       logger)
            logger.info(f'✅ Successfully estimated probabilities of missing keypoints with '
                        f'{["set_keypoints", "indep_keypoints"][int(indep_keypoints)]}.\n')

    if _cfg.transforms.viewinvariant is None or type(_cfg.transforms.viewinvariant) != bool:
        print("\n❌ transforms.viewinvariant should be a "
              f"bool. Got {_cfg.transforms.viewinvariant}")
        sys.exit(1)
    else:
        viewinvariant = _cfg.transforms.viewinvariant

    if _cfg.transforms.normalize is None or type(_cfg.transforms.normalize) != bool:
        print("\n❌ transforms.normalize should be a "
              f"bool. Got {_cfg.transforms.normalize}")
        sys.exit(1)
    else:
        normalize = _cfg.transforms.normalize

    if _cfg.transforms.normalizecube is None or type(_cfg.transforms.normalizecube) != bool:
        print("\n❌ transforms.normalizecube should be a "
              f"bool. Got {_cfg.transforms.normalizecube}")
        sys.exit(1)
    else:
        normalizecube = _cfg.transforms.normalizecube

    if _cfg.transforms.swap is None or type(
            _cfg.transforms.swap) != float or _cfg.transforms.swap < 0 or _cfg.transforms.swap > 1:
        print("\n❌ transforms.swap should be a float between 0 and 1 "
              "(probability of swapping during training). "
              f"Got {_cfg.transforms.swap}")
        sys.exit(1)
    else:
        swap = _cfg.transforms.swap

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

    if _cfg.test.suffix is None or type(_cfg.test.suffix) != str:
        print("\n❌ test.suffix should be a string."
              f"Got {_cfg.test.suffix}")
        sys.exit(1)
    else:
        suffix = _cfg.test.suffix

    if _cfg.test.name_items is None:
        print("\n❌ test.name_items should be a dictionary."
              f"Got {_cfg.test.name_items}")
        sys.exit(1)
    else:
        name_items = _cfg.test.name_items

    if _cfg.test.merge is None or type(_cfg.test.merge) != bool:
        print("\n❌ test.merge should be a bool."
              f"Got {_cfg.test.merge}")
        sys.exit(1)
    else:
        merge = _cfg.test.merge

    if _cfg.test.merge_sets_file is None or type(_cfg.test.merge_sets_file) != str:
        print("\n❌ test.merge_sets_file should be a string."
              f"Got {_cfg.test.merge_sets_file}")
        sys.exit(1)
    else:
        merge_sets_file = _cfg.test.merge_sets_file

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

    if _cfg.model_scheduler.type is None or type(_cfg.model_scheduler.type) != str \
            or _cfg.model_scheduler.type != 'lambdalr':
        print("\n❌ model_scheduler.type should be 'lambdalr'."
              f"Got {_cfg.model_scheduler.type}")
        sys.exit(1)
    else:
        model_scheduler_type = _cfg.model_scheduler.type

    if _cfg.model_scheduler.steps_epoch is None or type(_cfg.model_scheduler.steps_epoch) != int:
        print("\n❌ model_scheduler.steps_epoch should be 'lambdalr'."
              f"Got {_cfg.model_scheduler.steps_epoch}")
        sys.exit(1)
    else:
        model_scheduler_steps_epoch = max(1, _cfg.model_scheduler.steps_epoch)

    if _cfg.model_scheduler.rate is None or type(_cfg.model_scheduler.rate) != float:
        print("\n❌ model_scheduler.rate should be a float between 0 and 1."
              f"Got {_cfg.model_scheduler.rate}")
        sys.exit(1)
    else:
        model_scheduler_rate = max(0, min(1, _cfg.model_scheduler.rate))

    os.makedirs(os.path.join(final_model_path, 'config'), exist_ok=True)

    output_config_file = os.path.join(final_model_path, 'config', f'config_train.yaml')
    copy_config_file(modified_cfg, output_config_file)

    test_dir = os.path.join(final_model_path, f'{datetime.today().strftime("%Y-%m-%d")}_test')
    os.makedirs(test_dir, exist_ok=True)

    os.makedirs(os.path.join(test_dir, 'config'), exist_ok=True)
    output_config_file = os.path.join(test_dir, 'config', f'config_train.yaml')
    copy_config_file(modified_cfg, output_config_file)

    logger.info(f'✅ Successfully loaded configuration.\n')

    add_missing = True
    main(project_path, final_model_path, dataset_path, dataset_name, test_dir,
         skeleton_file_path, training_seed,
         load_model, _cfg.network,
         training_batch_size, training_epochs, learning_rate,
         loss_type, loss_mask, loss_factor,
         model_scheduler_rate, model_scheduler_type, model_scheduler_steps_epoch,
         n_cpus,
         print_every,
         proba_file, proba_length_file, add_missing_indep_keypoints,
         add_missing_pad, viewinvariant,
         normalize, normalizecube, swap,
         add_missing, name_items, merge,
         original_coordinates, threshold_pck, n_repeat,
         merge_sets_file, n_plots, plot2d_only_holes,
         plot3d_size, plot3d_azim, suffix,
         logger, verbose)


if __name__ == '__main__':
    cli()
