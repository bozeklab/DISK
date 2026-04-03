from pathlib import Path
import logging
import pytest

from DISK.launchers import create_project, prepare_data, train_evaluate, evaluate_compare, impute
from DISK.utils.logger_setup import copy_config_file

from shared_assertions import *



data_files = [
    "/home/france/mount_cvg/behavior_data/mocap_dataset/mocap_3djoints_subset/90_15.npy",
    "/home/france/mount_cvg/behavior_data/mocap_dataset/mocap_3djoints_subset/90_23.npy",
    "/home/france/mount_cvg/behavior_data/mocap_dataset/mocap_3djoints_subset/90_29.npy",
    "/home/france/mount_cvg/behavior_data/mocap_dataset/mocap_3djoints_subset/91_03.npy",
    "/home/france/mount_cvg/behavior_data/mocap_dataset/mocap_3djoints_subset/91_08.npy",
    "/home/france/mount_cvg/behavior_data/mocap_dataset/mocap_3djoints_subset/91_12.npy",
    "/home/france/mount_cvg/behavior_data/mocap_dataset/mocap_3djoints_subset/91_16.npy",

    # "/home/france/mount_cvg/behavior_data/mocap_dataset/mocap_3djoints_subset/91_20.npy",
    #
    # "/home/france/mount_cvg/behavior_data/mocap_dataset/mocap_3djoints_subset/91_24.npy",
    # "/home/france/mount_cvg/behavior_data/mocap_dataset/mocap_3djoints_subset/91_28.npy",
    # "/home/france/mount_cvg/behavior_data/mocap_dataset/mocap_3djoints_subset/91_32.npy",
    # "/home/france/mount_cvg/behavior_data/mocap_dataset/mocap_3djoints_subset/91_57.npy",
    # "/home/france/mount_cvg/behavior_data/mocap_dataset/mocap_3djoints_subset/94_02.npy",
    # "/home/france/mount_cvg/behavior_data/mocap_dataset/mocap_3djoints_subset/94_06.npy",
    # "/home/france/mount_cvg/behavior_data/mocap_dataset/mocap_3djoints_subset/94_10.npy",
    # "/home/france/mount_cvg/behavior_data/mocap_dataset/mocap_3djoints_subset/94_14.npy",
    #
    # "/home/france/mount_cvg/behavior_data/mocap_dataset/mocap_3djoints_subset/90_19.npy",
    # "/home/france/mount_cvg/behavior_data/mocap_dataset/mocap_3djoints_subset/90_25.npy",
    # "/home/france/mount_cvg/behavior_data/mocap_dataset/mocap_3djoints_subset/90_30.npy",
    # "/home/france/mount_cvg/behavior_data/mocap_dataset/mocap_3djoints_subset/91_04.npy",
    # "/home/france/mount_cvg/behavior_data/mocap_dataset/mocap_3djoints_subset/91_09.npy",
    # "/home/france/mount_cvg/behavior_data/mocap_dataset/mocap_3djoints_subset/91_13.npy",
    # "/home/france/mount_cvg/behavior_data/mocap_dataset/mocap_3djoints_subset/91_17.npy",
    # "/home/france/mount_cvg/behavior_data/mocap_dataset/mocap_3djoints_subset/91_21.npy",
    #
    # "/home/france/mount_cvg/behavior_data/mocap_dataset/mocap_3djoints_subset/91_25.npy",
    # "/home/france/mount_cvg/behavior_data/mocap_dataset/mocap_3djoints_subset/91_29.npy",
    # "/home/france/mount_cvg/behavior_data/mocap_dataset/mocap_3djoints_subset/91_33.npy",
    # "/home/france/mount_cvg/behavior_data/mocap_dataset/mocap_3djoints_subset/91_62.npy",
    # "/home/france/mount_cvg/behavior_data/mocap_dataset/mocap_3djoints_subset/94_03.npy",
    # "/home/france/mount_cvg/behavior_data/mocap_dataset/mocap_3djoints_subset/94_07.npy",
    # "/home/france/mount_cvg/behavior_data/mocap_dataset/mocap_3djoints_subset/94_11.npy",
    # "/home/france/mount_cvg/behavior_data/mocap_dataset/mocap_3djoints_subset/94_15.npy",
    #
    # "/home/france/mount_cvg/behavior_data/mocap_dataset/mocap_3djoints_subset/90_20.npy",
    # "/home/france/mount_cvg/behavior_data/mocap_dataset/mocap_3djoints_subset/90_26.npy",
    # "/home/france/mount_cvg/behavior_data/mocap_dataset/mocap_3djoints_subset/91_01.npy",
    # "/home/france/mount_cvg/behavior_data/mocap_dataset/mocap_3djoints_subset/91_05.npy",
    # "/home/france/mount_cvg/behavior_data/mocap_dataset/mocap_3djoints_subset/91_10.npy",
    # "/home/france/mount_cvg/behavior_data/mocap_dataset/mocap_3djoints_subset/91_14.npy",
    # "/home/france/mount_cvg/behavior_data/mocap_dataset/mocap_3djoints_subset/91_18.npy",
    # "/home/france/mount_cvg/behavior_data/mocap_dataset/mocap_3djoints_subset/91_22.npy",
    #
    # "/home/france/mount_cvg/behavior_data/mocap_dataset/mocap_3djoints_subset/91_26.npy",
    # "/home/france/mount_cvg/behavior_data/mocap_dataset/mocap_3djoints_subset/91_30.npy",
    # "/home/france/mount_cvg/behavior_data/mocap_dataset/mocap_3djoints_subset/91_34.npy",
    # "/home/france/mount_cvg/behavior_data/mocap_dataset/mocap_3djoints_subset/93_02.npy",
    # "/home/france/mount_cvg/behavior_data/mocap_dataset/mocap_3djoints_subset/94_04.npy",
    # "/home/france/mount_cvg/behavior_data/mocap_dataset/mocap_3djoints_subset/94_08.npy",
    # "/home/france/mount_cvg/behavior_data/mocap_dataset/mocap_3djoints_subset/94_12.npy",
    # "/home/france/mount_cvg/behavior_data/mocap_dataset/mocap_3djoints_subset/94_16.npy",
    # "/home/france/mount_cvg/behavior_data/mocap_dataset/mocap_3djoints_subset/90_21.npy",
    # "/home/france/mount_cvg/behavior_data/mocap_dataset/mocap_3djoints_subset/90_27.npy",
    # "/home/france/mount_cvg/behavior_data/mocap_dataset/mocap_3djoints_subset/91_02.npy",
    # "/home/france/mount_cvg/behavior_data/mocap_dataset/mocap_3djoints_subset/91_07.npy",
    # "/home/france/mount_cvg/behavior_data/mocap_dataset/mocap_3djoints_subset/91_11.npy",
    # "/home/france/mount_cvg/behavior_data/mocap_dataset/mocap_3djoints_subset/91_15.npy",
    # "/home/france/mount_cvg/behavior_data/mocap_dataset/mocap_3djoints_subset/91_19.npy",
    # "/home/france/mount_cvg/behavior_data/mocap_dataset/mocap_3djoints_subset/91_23.npy",
    # "/home/france/mount_cvg/behavior_data/mocap_dataset/mocap_3djoints_subset/91_27.npy",
    # "/home/france/mount_cvg/behavior_data/mocap_dataset/mocap_3djoints_subset/91_31.npy",
    # "/home/france/mount_cvg/behavior_data/mocap_dataset/mocap_3djoints_subset/91_35.npy",
    # "/home/france/mount_cvg/behavior_data/mocap_dataset/mocap_3djoints_subset/94_01.npy",
    # "/home/france/mount_cvg/behavior_data/mocap_dataset/mocap_3djoints_subset/94_05.npy",
    # "/home/france/mount_cvg/behavior_data/mocap_dataset/mocap_3djoints_subset/94_09.npy",
    # "/home/france/mount_cvg/behavior_data/mocap_dataset/mocap_3djoints_subset/94_13.npy"
]

project_name = "DISK_mocap_dataset"
file_format = "npy"

dataset_name = 'DISK_human_mocap'
model_name = 'DISK-GRU'
network_type = 'GRU'

print_every = 1
training_n_plots = 5
training_n_repeat = 1
pck_threshold = 0.1

@pytest.fixture(scope="session")
def create_project_human_mocap_3d(tmp_path_factory):
    # STEP1. CREATE PROJECT
    ## GIVEN
    tmp_path = tmp_path_factory.mktemp("output")
    project_path = tmp_path / project_name
    assert not project_path.is_dir()


    ## WHEN
    create_project.main(
        project_path=project_path,
        data_file_list=data_files,
        file_type=file_format,
    )

    return tmp_path

def test_create_project_human_mocap_3d(create_project_human_mocap_3d):
    project_path = (create_project_human_mocap_3d / project_name)
    ## THEN
    assert_file_creation_after_create_project(project_path)


@pytest.fixture(scope="session")
def prepare_data_human_mocap_3d(create_project_human_mocap_3d):
    # STEP2. PREPARE DATA
    project_path = (create_project_human_mocap_3d / project_name)

    ## GIVEN
    dataset_path = (project_path / f'DISK_data/{dataset_name}')
    assert not dataset_path.is_dir()
    dataset_path.mkdir(exist_ok=True, parents=True)
    assert dataset_path.is_dir()

    logger = logging.getLogger()
    indep_keypoints = True
    merge_keypoints = False

    prepare_data_kwargs = dict(
        project_path=project_path,
        data_files =data_files,
        file_type =file_format,
        dataset_name = dataset_name,
        dataset_path = dataset_path,
        length = 20,
        stride= 10,
        fill_gap = 0,
        sequential = True,
        original_freq = 1,
        subsampling_freq = 1,
        dlc_likelihood_threshold = 0.1,
        discard_beginning = 0,
        discard_end = -1,
        drop_keypoints = [],
        indep_keypoints = indep_keypoints,
        merge_keypoints = merge_keypoints,
        skeleton_graph = None,
        logger = logger,
    )

    ## WHEN
    keypoints, divider, no_original_missing, indep_keypoints, merge_keypoints, suffix = prepare_data.main(
        **prepare_data_kwargs)

    ## THEN

    return suffix, indep_keypoints, merge_keypoints

def test_prepare_data_human_mocap_3d(create_project_human_mocap_3d, prepare_data_human_mocap_3d):
    project_path = (create_project_human_mocap_3d / project_name)
    dataset_path = (project_path / f'DISK_data/{dataset_name}')
    suffix, _, _ = prepare_data_human_mocap_3d
    assert_file_creation_after_prepare_data(dataset_path, suffix)


@pytest.fixture(scope="session")
def train1_human_mocap_3d(create_project_human_mocap_3d, prepare_data_human_mocap_3d):
    ## NORMAL TRAINING
    # STEP3. TRAIN
    ## GIVEN
    project_path = (create_project_human_mocap_3d / project_name)
    suffix, indep_keypoints, merge_keypoints = prepare_data_human_mocap_3d
    dataset_path = (project_path / f'DISK_data/{dataset_name}')
    logger = logging.getLogger()

    ## WHEN / THEN
    network_config = assert_and_get_network_config('gru')

    ## GIVEN
    proba_files_exist, proba_file, proba_length_file = train_evaluate.find_proba_files(dataset_path, suffix)

    model_dir = project_path.joinpath(f'DISK_train/{model_name}')
    assert not model_dir.is_dir()
    model_dir.mkdir(exist_ok=True, parents=True)
    model_dir.joinpath('config').mkdir(exist_ok=True, parents=True)

    test_dir = project_path.joinpath(f'DISK_train/{model_name}/test_folder')
    assert not test_dir.is_dir()
    test_dir.mkdir(exist_ok=True, parents=True)
    test_dir.joinpath('config').mkdir(exist_ok=True, parents=True)

    train_kwargs = dict(
            project_dir=str(project_path),
            model_dir=str(model_dir),
            dataset_path=str(dataset_path),
            dataset_name=dataset_name,
            test_dir=str(test_dir),
            skeleton_graph=None,
            training_seed=None,
            load_model_dir='',
            cfg_network=network_config,
            training_batch_size=8,
            training_epochs=4,
            learning_rate=0.001,
            loss_type='l1',
            loss_mask=True,
            loss_factor=100,
            model_scheduler_rate=0.95,
            model_scheduler_type='lambdalr',
            model_scheduler_steps_epoch=500,
            n_cpus=4,
            print_every=print_every,
            proba_file=proba_file,
            proba_length_file=proba_length_file,
            indep_keypoints=False,
            add_missing_pad=(1,0),
            viewinvariant=True,
            normalize=False,
            normalizecube=True,
            swap=0.5,
            add_missing=True,
            test_original_coordinates=True,
            test_threshold_pck=pck_threshold,
            n_repeat=training_n_repeat,
            total_n_plots=training_n_plots,
            plot2d_only_holes=True,
            plot3d_size=2,
            plot3d_azim=60,
            verbose=0
    )
    output_config_file = os.path.join(model_dir, 'config', f'config_train.yaml')

    # workaround so the config that is written works for the next steps
    modified_cfg = dict(train_kwargs)
    modified_cfg['network'] = network_config
    modified_cfg['network']['type'] = network_type
    modified_cfg['transforms_viewinvariant'] = train_kwargs['viewinvariant']
    modified_cfg['transforms_normalize'] = train_kwargs['normalize']
    modified_cfg['transforms_normalizecube'] = train_kwargs['normalizecube']
    copy_config_file(modified_cfg, output_config_file)

    output_config_file = os.path.join(test_dir, 'config', f'config_train.yaml')
    copy_config_file(modified_cfg, output_config_file)

    ## WHEN
    best_rmse, best_epoch, last_epoch = train_evaluate.main(logger=logger, **train_kwargs)

    return model_dir, test_dir, best_epoch, last_epoch, print_every

def test_train1_human_mocap_3d(train1_human_mocap_3d):
    model_dir, test_dir, best_epoch, last_epoch, print_every = train1_human_mocap_3d
    ## THEN
    assert_file_creation_after_train(model_dir, best_epoch, last_epoch, print_every)
    assert_file_creation_after_evaluate(test_dir, model_name, training_n_plots, training_n_repeat, pck_threshold,
                                        '')

@pytest.fixture(scope="session")
def train2_human_mocap_3d(create_project_human_mocap_3d, prepare_data_human_mocap_3d, train1_human_mocap_3d):
    ## GIVEN
    ## TRAINING WITH LOADING
    project_path = (create_project_human_mocap_3d / project_name)
    suffix, indep_keypoints, merge_keypoints = prepare_data_human_mocap_3d
    dataset_path = (project_path / f'DISK_data/{dataset_name}')
    logger = logging.getLogger()

    ## WHEN / THEN
    network_config = assert_and_get_network_config('gru')

    ## GIVEN
    proba_files_exist, proba_file, proba_length_file = train_evaluate.find_proba_files(dataset_path, suffix)

    model_dir, _, _, _, _ = train1_human_mocap_3d
    test_dir = project_path.joinpath(f'DISK_train/{model_name}/test_folder2')
    assert not test_dir.is_dir()
    test_dir.mkdir(exist_ok=True, parents=True)

    train_kwargs = dict(
        project_dir=str(project_path),
        model_dir=str(model_dir),
        dataset_path=str(dataset_path),
        dataset_name=dataset_name,
        test_dir=str(test_dir),
        skeleton_graph=None,
        training_seed=None,
        load_model_dir=str(model_dir),
        cfg_network=network_config,
        training_batch_size=8,
        training_epochs=4,
        learning_rate=0.001,
        loss_type='l1',
        loss_mask=True,
        loss_factor=100,
        model_scheduler_rate=0.95,
        model_scheduler_type='lambdalr',
        model_scheduler_steps_epoch=500,
        n_cpus=4,
        print_every=print_every,
        proba_file=proba_file,
        proba_length_file=proba_length_file,
        indep_keypoints=False,
        add_missing_pad=(1, 0),
        viewinvariant=True,
        normalize=False,
        normalizecube=True,
        swap=0.5,
        add_missing=True,
        test_original_coordinates=True,
        test_threshold_pck=pck_threshold,
        n_repeat=training_n_repeat,
        total_n_plots=training_n_plots,
        plot2d_only_holes=True,
        plot3d_size=2,
        plot3d_azim=60,
        verbose=0
    )

    ## WHEN
    best_rmse, best_epoch, last_epoch = train_evaluate.main(logger=logger, **train_kwargs)

    return model_dir, test_dir, best_epoch, last_epoch, print_every


def test_train2_human_mocap_3d(train2_human_mocap_3d):
    ## THEN
    model_dir, test_dir, best_epoch, last_epoch, print_every = train2_human_mocap_3d
    print(best_epoch)
    logger = logging.getLogger()

    assert_file_creation_after_train(model_dir, best_epoch, last_epoch, logger)
    assert_file_creation_after_evaluate(test_dir, model_name, training_n_plots, training_n_repeat, pck_threshold,
                                        '')


def test_evaluate_human_mocap_3d(create_project_human_mocap_3d, prepare_data_human_mocap_3d, train2_human_mocap_3d):
    # STEP 3bis. Evaluate
    ## GIVEN
    project_path = (create_project_human_mocap_3d / project_name)
    dataset_path = (project_path / f'DISK_data/{dataset_name}')
    suffix, indep_keypoints, merge_keypoints = prepare_data_human_mocap_3d
    proba_files_exist, proba_file, proba_length_file = train_evaluate.find_proba_files(dataset_path, suffix)

    logger = logging.getLogger()

    model_dir, _, _, _, _ = train2_human_mocap_3d
    test_dir = project_path.joinpath(f'DISK_train/test_folder3')
    n_plots = 6
    n_repeat = 2
    pck_threshold = 0.1
    suffix = ''
    evaluate_kwargs = dict(project_dir=project_path,
                       model_dirs=[model_dir, ],
                       dataset_path=dataset_path,
                       dataset_name=dataset_name,
                       test_dir=test_dir,
                       skeleton_graph=None,
                       training_batch_size=32,
                       loss_type='l1',
                       loss_mask=True,
                       loss_factor=100,
                       n_cpus=4,
                       proba_file=proba_file,
                       proba_length_file=proba_length_file,
                       indep_keypoints=indep_keypoints,
                       add_missing_pad=(1,1),
                       viewinvariant=True,
                       normalize=False,
                       normalizecube=True,
                       swap=0.5,
                       add_missing=True,
                       test_original_coordinates=True,
                       pck_threshold=pck_threshold,
                       n_repeat=n_repeat,
                       total_n_plots=n_plots,
                       plot2d_only_holes=True,
                       plot3d_size=2,
                       plot3d_azim=60,
                       logger=logger,
                           suffix=suffix,
                       verbose=0)
    test_dir.mkdir(exist_ok=True, parents=True)

    evaluate_compare.main(**evaluate_kwargs)

    assert_file_creation_after_evaluate(test_dir, model_name, n_plots, n_repeat, pck_threshold, suffix)


def test_impute_human_mocap_3d(create_project_human_mocap_3d, train2_human_mocap_3d):
    project_path = (create_project_human_mocap_3d / project_name)
    dataset_path = (project_path / f'DISK_data/{dataset_name}')
    model_dir, _, _, _, _ = train2_human_mocap_3d

    impute_dir = project_path.joinpath(f'DISK_impute/Impute_{model_name}')
    impute_dir.mkdir(exist_ok=True, parents=True)
    plot_dir = project_path.joinpath(f'DISK_impute/Impute_{model_name}/plots')
    plot_dir.mkdir(exist_ok=True, parents=True)

    logger = logging.getLogger()

    impute_kwargs = dict(project_dir=project_path,
                         impute_dir=impute_dir,
                         plot_dir=plot_dir,
                         file_type=file_format,
                         dataset_path=dataset_path,
                         skeleton_graph=None,
                         checkpoint=model_dir,
                         batch_size=32,
                         threshold_error_score=1000,
                         total_n_plots=5,
                         plot_only_holes=True,
                         missing_pad=(1, 0),
                         logger=logger,
                         verbose=0)

    mess = impute.main(**impute_kwargs)

    assert_after_impute_no_gaps_found(mess)


def test_create_project_errors_if_project_path_already_exists(tmp_path):
    project_path = tmp_path.joinpath("DISK_mocap_dataset")

    # first time, no problem
    create_project.main(
        project_path=project_path,
        data_file_list=data_files,
        file_type="npy",
    )

    # second time, should a problem
    with pytest.raises(FileExistsError):
        create_project.main(
            project_path=project_path,
            data_file_list=data_files,
            file_type="npy",
        )





from DISK.utils.config_decorator import parse_command_line_args

# def test_parse_cl_args1():
#     # GIVEN
#     cfg_in = {'--project_path': '/path/to/project'}
#
#     # WHEN
#     cfg_out = parse_command_line_args(cfg_in)
#
#     # THEN
#     assert cfg_out['args'] == "something"