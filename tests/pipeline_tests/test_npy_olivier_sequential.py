from pathlib import Path
import logging
import pytest
import yaml

from DISK.launchers import create_project, prepare_data, train_evaluate, evaluate_compare, impute, add_skeleton
from DISK.utils.logger_setup import copy_config_file

from shared_assertions import *



data_file_list = [os.path.join(root_path, "behavior_data/female_2318.npy"), ]

project_name = "DISK_olivier_dataset"
file_format = "npy"

dataset_name = 'DISK_olivier_onefile'
model_name = 'DISK_olivier'
network_type = 'transformer'

length = 240
stride = 30
fill_gap = 10
sequential = True

training_epochs = 2
training_batch_size = 8
n_cpus = 6

print_every = 1
training_n_plots = 5
training_n_repeat = 1
pck_threshold = 0.1

@pytest.fixture(scope="session")
def create_project_olivier_sequential(tmp_path_factory):
    # STEP1. CREATE PROJECT
    ## GIVEN
    tmp_path = tmp_path_factory.mktemp("output")
    project_path = tmp_path / project_name
    assert not project_path.is_dir()

    ## WHEN
    create_project.main(
        project_path=str(project_path),
        data_file_list=data_file_list,
        file_format=file_format,
    )

    return tmp_path


def test_create_project_olivier_sequential(create_project_olivier_sequential):
    ## THEN
    project_path = (create_project_olivier_sequential / project_name)
    assert_file_creation_after_create_project(project_path)


def test_add_skeleton_shouldFail(create_project_olivier_sequential):
    project_path = (create_project_olivier_sequential / project_name)
    with open(str(project_path / 'config_project.yaml'), 'r') as f:
        content = yaml.safe_load(f)
    assert not 'keypoints' in content.keys()

    with pytest.raises(KeyError):
        add_skeleton.create_skeleton(content['keypoints'])

@pytest.fixture(scope="session")
def prepare_data_olivier_sequential_indepFalse(create_project_olivier_sequential):
    # STEP2. PREPARE DATA
    project_path = (create_project_olivier_sequential / project_name)

    ## GIVEN
    dataset_path = (project_path / f'DISK_data/{dataset_name}')
    assert not dataset_path.is_dir()
    dataset_path.mkdir(exist_ok=True, parents=True)
    assert dataset_path.is_dir()

    logger = logging.getLogger()

    indep_keypoints = False
    merge_keypoints = False
    suffix_proba_files = '_set_keypoints'

    prepare_data_kwargs = dict(
        project_path=project_path,
        data_files =data_file_list,
        file_format =file_format,
        dataset_name = dataset_name,
        dataset_path = dataset_path,
        length = length,
        stride= stride,
        fill_gap = fill_gap,
        sequential = sequential,
        original_freq = 1,
        subsampling_freq = 1,
        dlc_likelihood_threshold = 0.1,
        discard_beginning = 0,
        discard_end = -1,
        drop_keypoints = [],
        indep_keypoints = indep_keypoints,
        merge_keypoints = merge_keypoints,
        suffix_proba_files = suffix_proba_files,
        skeleton_graph = None,
        logger = logger,
    )

    ## WHEN
    keypoints, divider, no_original_missing, indep_keypoints, merge_keypoints, suffix = prepare_data.main(
        **prepare_data_kwargs)

    ## THEN

    return suffix, indep_keypoints, merge_keypoints


def test_prepare_data_olivier_sequential_indepFalse(create_project_olivier_sequential,
                                                    prepare_data_olivier_sequential_indepFalse):
    project_path = (create_project_olivier_sequential / project_name)
    suffix, indep_keypoints, merge_keypoints = prepare_data_olivier_sequential_indepFalse
    dataset_path = (project_path / f'DISK_data/{dataset_name}')
    assert_file_creation_after_prepare_data(dataset_path, suffix)


@pytest.fixture(scope="session")
def prepare_data_olivier_sequential_indepTrue(create_project_olivier_sequential,
                                                   prepare_data_olivier_sequential_indepFalse):
    # STEP2. PREPARE DATA
    project_path = (create_project_olivier_sequential / project_name)

    ## GIVEN
    dataset_path = (project_path / f'DISK_data/{dataset_name}')
    assert dataset_path.is_dir()

    logger = logging.getLogger()

    indep_keypoints = True
    merge_keypoints = False
    suffix_proba_files = ''

    prepare_data_kwargs = dict(
        project_path=project_path,
        data_files=data_file_list,
        file_format=file_format,
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        length=length,
        stride=stride,
        fill_gap=fill_gap,
        sequential=sequential,
        original_freq=1,
        subsampling_freq=1,
        dlc_likelihood_threshold=0.1,
        discard_beginning=0,
        discard_end=-1,
        drop_keypoints=[],
        indep_keypoints=indep_keypoints,
        merge_keypoints=merge_keypoints,
        suffix_proba_files=suffix_proba_files,
        skeleton_graph=None,
        logger=logger,
    )

    ## WHEN
    keypoints, divider, no_original_missing, indep_keypoints, merge_keypoints, suffix = prepare_data.main(
        **prepare_data_kwargs)

    config_project_file_path = project_path / 'config_project.yaml'
    with open(config_project_file_path, 'r') as f:
        content = yaml.safe_load(f)
    content['keypoints'] = keypoints
    with open(config_project_file_path, 'w') as f:
        yaml.safe_dump(content, f)

    return suffix, indep_keypoints, merge_keypoints


def test_prepare_data_olivier_sequential_indepTrue(create_project_olivier_sequential,
                                                    prepare_data_olivier_sequential_indepTrue):
    ## THEN
    project_path = (create_project_olivier_sequential / project_name)
    suffix, indep_keypoints, merge_keypoints = prepare_data_olivier_sequential_indepTrue
    dataset_path = (project_path / f'DISK_data/{dataset_name}')

    assert_file_creation_after_prepare_data(dataset_path, suffix)


def test_add_skeleton(create_project_olivier_sequential, prepare_data_olivier_sequential_indepTrue,
                      monkeypatch):
    project_path = (create_project_olivier_sequential / project_name)
    with open(str(project_path / 'config_project.yaml'), 'r') as f:
        content = yaml.safe_load(f)

    inputs = iter(['0,1', '1,2', '2,3', '', '1'])
    monkeypatch.setattr('builtins.input', lambda _: next(inputs))

    add_skeleton.create_skeleton(content['keypoints'])


@pytest.fixture(scope="session")
def train_olivier_sequential(create_project_olivier_sequential,
                                  prepare_data_olivier_sequential_indepTrue):
    # STEP3. TRAIN
    ## GIVEN
    project_path = (create_project_olivier_sequential / project_name)
    suffix, indep_keypoints, merge_keypoints = prepare_data_olivier_sequential_indepTrue
    dataset_path = (project_path / f'DISK_data/{dataset_name}')
    logger = logging.getLogger()

    ## WHEN / THEN
    network_config = assert_and_get_network_config(network_type)

    ## GIVEN
    proba_files_exist, _, _ = train_evaluate.find_proba_files(dataset_path, suffix)
    rerun_create_proba = False if proba_files_exist else True

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
            training_batch_size=training_batch_size,
            training_epochs=training_epochs,
            learning_rate=0.001,
            loss_type='l1',
            loss_mask=True,
            loss_factor=100,
            model_scheduler_rate=0.95,
            model_scheduler_type='lambdalr',
            model_scheduler_steps_epoch=500,
            n_cpus=n_cpus,
            print_every=print_every,
            rerun_create_proba=rerun_create_proba,
            indep_keypoints=indep_keypoints,
            merge_keypoints=merge_keypoints,
            suffix_proba_files=suffix,
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


def test_train_olivier_sequential(prepare_data_olivier_sequential_indepTrue,
                                  train_olivier_sequential):
    model_dir, test_dir, best_epoch, last_epoch, print_every = train_olivier_sequential
    ## THEN
    assert_file_creation_after_train(model_dir, best_epoch, last_epoch, print_every)
    suffix, indep_keypoints, merge_keypoints = prepare_data_olivier_sequential_indepTrue

    assert_file_creation_after_evaluate(test_dir, model_name, training_n_plots, training_n_repeat, pck_threshold,
                                        suffix)


# @pytest.mark.skip(reason="Not implemented yet")
def test_evaluate_olivier_sequential(create_project_olivier_sequential,
                                     prepare_data_olivier_sequential_indepTrue,
                               train_olivier_sequential):
    # STEP 3bis. Evaluate
    ## GIVEN
    project_path = (create_project_olivier_sequential / project_name)
    dataset_path = (project_path / f'DISK_data/{dataset_name}')
    suffix, indep_keypoints, merge_keypoints = prepare_data_olivier_sequential_indepTrue
    proba_files_exist, _, _ = train_evaluate.find_proba_files(dataset_path, suffix)
    rerun_create_proba = False if proba_files_exist else True

    logger = logging.getLogger()

    model_dir, _, _, _, _ = train_olivier_sequential
    test_dir = project_path.joinpath(f'DISK_train/test_folder3')
    n_plots = 6
    n_repeat = 2
    pck_threshold = 0.1

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
                    rerun_create_proba=rerun_create_proba,
                       indep_keypoints=indep_keypoints,
                       merge_keypoints=merge_keypoints,
                       suffix_proba_files=suffix,
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


#@pytest.mark.skip(reason="Not implemented yet")
def test_impute_olivier_sequential(create_project_olivier_sequential, train_olivier_sequential):
    project_path = (create_project_olivier_sequential / project_name)
    dataset_path = (project_path / f'DISK_data/{dataset_name}')
    model_dir, _, _, _, _ = train_olivier_sequential

    impute_dir = project_path.joinpath(f'DISK_impute/Impute_{model_name}')
    impute_dir.mkdir(exist_ok=True, parents=True)
    plot_dir = project_path.joinpath(f'DISK_impute/Impute_{model_name}/plots')
    plot_dir.mkdir(exist_ok=True, parents=True)

    logger = logging.getLogger()

    impute_kwargs = dict(project_dir=project_path,
                         impute_dir=impute_dir,
                         plot_dir=plot_dir,
                         file_format=file_format,
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

    impute.main(**impute_kwargs)


