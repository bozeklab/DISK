from pathlib import Path
import logging
import pytest
from glob import glob

from DISK.launchers import create_project, prepare_data, train_evaluate, evaluate_compare, impute
from DISK.utils.logger_setup import copy_config_file

from shared_assertions import *

project_name = 'DISK_multi_DLC_CSV'
file_format = 'dlc_csv'
data_files = glob('/home/france/mount_cvg/behavior_data/calms21-disk-dlc/calms21-01.01-snapshot195-dlc-csv/*.csv')

dataset_name = 'mouse'
length = 30
stride = 30
fill_gap = 10
sequential = False
dlc_likelihood_threshold = 0.8
discard_beginning = 0
discard_end = -1
drop_keypoints = []
indep_keypoints = False
merge_keypoints = False
original_freq = 60
subsampling_freq = 60

model_name = 'DISK_transformer'
network_type = 'transformer'
print_every = 2
training_epochs = 4
training_n_plots = 5
training_n_repeat = 1
pck_threshold = 0.5
n_cpus = 6

threshoold_error_score = 5


@pytest.fixture(scope="session")
def create_project_multianimal_dlc_csv(tmp_path_factory):
    # STEP1. CREATE PROJECT
    ## GIVEN
    tmp_path = tmp_path_factory.mktemp("output")
    project_path = tmp_path / project_name
    assert not project_path.is_dir()


    ## WHEN
    create_project.main(
        project_path=project_path,
        data_file_list=data_files,
        file_format=file_format,
    )

    return tmp_path

def test_create_project_multianimal_dlc_csv(create_project_multianimal_dlc_csv):
    project_path = (create_project_multianimal_dlc_csv / project_name)
    ## THEN
    assert_file_creation_after_create_project(project_path)


@pytest.fixture(scope="session")
def prepare_data_multianimal_dlc_csv(create_project_multianimal_dlc_csv):
    # STEP2. PREPARE DATA
    project_path = (create_project_multianimal_dlc_csv / project_name)

    ## GIVEN
    dataset_path = (project_path / f'DISK_data/{dataset_name}')
    assert not dataset_path.is_dir()
    dataset_path.mkdir(exist_ok=True, parents=True)
    assert dataset_path.is_dir()

    logger = logging.getLogger()
    indep_keypoints = True
    merge_keypoints = False

    prepare_data_kwargs = dict(
        project_path = project_path,
        data_files = data_files,
        file_format = file_format,
        dataset_name = dataset_name,
        dataset_path = dataset_path,
        length = length,
        stride= stride,
        fill_gap = fill_gap,
        sequential = sequential,
        original_freq = original_freq,
        subsampling_freq = subsampling_freq,
        dlc_likelihood_threshold = dlc_likelihood_threshold,
        discard_beginning = discard_beginning,
        discard_end = discard_end,
        drop_keypoints = drop_keypoints,
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

def test_prepare_data_multianimal_dlc_csv(create_project_multianimal_dlc_csv, prepare_data_multianimal_dlc_csv):
    project_path = (create_project_multianimal_dlc_csv / project_name)
    dataset_path = (project_path / f'DISK_data/{dataset_name}')
    suffix, _, _ = prepare_data_multianimal_dlc_csv
    assert_file_creation_after_prepare_data(dataset_path, suffix)


@pytest.fixture(scope="session")
def train_multianimal_dlc_csv(create_project_multianimal_dlc_csv, prepare_data_multianimal_dlc_csv):
    ## NORMAL TRAINING
    # STEP3. TRAIN
    ## GIVEN
    project_path = (create_project_multianimal_dlc_csv / project_name)
    suffix, indep_keypoints, merge_keypoints = prepare_data_multianimal_dlc_csv
    dataset_path = (project_path / f'DISK_data/{dataset_name}')
    logger = logging.getLogger()

    ## WHEN / THEN
    network_config = assert_and_get_network_config(network_type)

    ## GIVEN
    proba_files_exist, proba_file, proba_length_file = train_evaluate.find_proba_files(dataset_path, suffix)

    model_dir = project_path.joinpath(f'DISK_train/{model_name}')
    model_dir.mkdir(exist_ok=True, parents=True)
    model_dir.joinpath('config').mkdir(exist_ok=True, parents=True)

    test_dir = project_path.joinpath(f'DISK_train/{model_name}/test_folder')
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


def test_train_multianimal_dlc_csv(train_multianimal_dlc_csv):
    model_dir, test_dir, best_epoch, last_epoch, print_every = train_multianimal_dlc_csv
    logger = logging.getLogger()
    ## THEN
    assert_file_creation_after_train(model_dir, best_epoch, last_epoch, print_every)
    assert_file_creation_after_evaluate(test_dir, model_name, training_n_plots, training_n_repeat, pck_threshold,
                                        '')


def test_impute_multianimal_dlc_csv(create_project_multianimal_dlc_csv, train_multianimal_dlc_csv):
    project_path = (create_project_multianimal_dlc_csv / project_name)
    dataset_path = (project_path / f'DISK_data/{dataset_name}')
    model_dir, _, _, _, _ = train_multianimal_dlc_csv

    impute_dir = project_path.joinpath(f'DISK_impute/Impute_{model_name}')
    impute_dir.mkdir(exist_ok=True, parents=True)
    plot_dir = project_path.joinpath(f'DISK_impute/Impute_{model_name}/plots')
    plot_dir.mkdir(exist_ok=True, parents=True)

    logger = logging.getLogger()

    n_plots = 5
    impute_kwargs = dict(project_dir=project_path,
                         impute_dir=impute_dir,
                         plot_dir=plot_dir,
                         file_format=file_format,
                         dataset_path=dataset_path,
                         skeleton_graph=None,
                         checkpoint=model_dir,
                         batch_size=32,
                         threshold_error_score=threshoold_error_score,
                         total_n_plots=n_plots,
                         plot_only_holes=True,
                         missing_pad=(1, 0),
                         logger=logger,
                         verbose=0)

    mess = impute.main(**impute_kwargs)

    assert_after_impute_gaps_found(impute_dir, data_files, mess, n_plots)




