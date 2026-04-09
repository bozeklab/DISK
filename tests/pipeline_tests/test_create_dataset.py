import pytest, logging
from DISK.launchers import create_project, prepare_data
from shared_assertions import *
from functools import partial

project_name = 'DISK_FL2'
file_format = 'mat_qualisys'
data_files = [
'/home/france/mount_cvg/behavior_data/bogna_data/INH1A_open_field_proc/INH1A_S1_M1_MC6_FL2_17_04_2019_proc_bij_6_08_19_A.mat',
'/home/france/mount_cvg/behavior_data/bogna_data/INH1A_open_field_proc/INH1A_S2_M2_MC6_FL2_17_04_2019_proc-bij_6_08_19_C.mat',
'/home/france/mount_cvg/behavior_data/bogna_data/INH1A_open_field_proc/INH1A_S3_M3_MC6_FL2_17_04_2019_proc_bij_7_08_19_B.mat',
'/home/france/mount_cvg/behavior_data/bogna_data/INH1A_open_field_proc/INH1A_S4_M4_MC7_FL2_17_04_2019_proc_bij_7_08_19_A.mat',
'/home/france/mount_cvg/behavior_data/bogna_data/INH1A_open_field_proc/INH1A_S5_M5_MC7_FL2_18_04_2019_proc_bij_6_08_19_C.mat',
'/home/france/mount_cvg/behavior_data/bogna_data/INH1A_open_field_proc/INH1A_S6_M6_MC7_FL2_18_04_2019_proc_bij_8_08_19_B.mat',
'/home/france/mount_cvg/behavior_data/bogna_data/INH1A_open_field_proc/INH1A_S7_M7_MC8_FL2_18_04_2019_proc_bij_8_08_19_A.mat',
'/home/france/mount_cvg/behavior_data/bogna_data/INH1A_open_field_proc/INH1A_S8_M8_MC8_FL2_18_04_2019_proc_bij_8_08_19_C.mat',
'/home/france/mount_cvg/behavior_data/bogna_data/INH1A_open_field_proc/INH1A_S9_M9_MC8_FL2_18_04_2019_proc_bij_8_08_19_B.mat',
'/home/france/mount_cvg/behavior_data/bogna_data/INH1A_open_field_proc/INH1A_S10_M10_MC8_FL2_18_04_2019_proc_nij_8_08_19_C.mat']

dataset_rootname = 'INH_test'
original_freq = 300
subsampling_freq = 60
length = 60
discard_beginning = 5
discard_end = 5
fill_gap = 0
drop_keypoints = []
sequential = False


@pytest.fixture(scope="session")
def create_project_inh(tmp_path_factory):
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


def test_create_project_inh(create_project_inh):
    project_path = (create_project_inh / project_name)
    ## THEN
    assert_file_creation_after_create_project(project_path)


def prepare_data_inh(project_path, dataset_name, indep_keypoints, merge_keypoints):
    # STEP2. PREPARE DATA

    ## GIVEN
    dataset_path = (project_path / f'DISK_data/{dataset_name}')
    print(dataset_path)
    assert not dataset_path.is_dir()
    dataset_path.mkdir(exist_ok=True, parents=True)
    assert dataset_path.is_dir()

    logger = logging.getLogger()

    prepare_data_kwargs = dict(
        project_path=project_path,
        data_files =data_files,
        file_format =file_format,
        dataset_name = dataset_name,
        dataset_path = dataset_path,
        length = length,
        stride= 10,
        fill_gap = fill_gap,
        sequential = True,
        original_freq = original_freq,
        subsampling_freq = subsampling_freq,
        dlc_likelihood_threshold = 0.1,
        discard_beginning = discard_beginning,
        discard_end = discard_end,
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


def test_prepare_data_inh_indepTrue_mergeFalse(create_project_inh):
    project_path = (create_project_inh / project_name)
    indep_kepoints = True
    merge_keypoints = False
    dataset_name = f'{dataset_rootname}_indep{indep_kepoints}_merge{merge_keypoints}'
    dataset_path = (project_path / f'DISK_data/{dataset_name}')
    suffix, indep_kepoints_return, merge_keypoints_return = prepare_data_inh(project_path, dataset_name,
                                                                             indep_kepoints, merge_keypoints)
    print(suffix)
    assert suffix == ''
    assert indep_kepoints_return == indep_kepoints
    assert merge_keypoints_return == merge_keypoints
    assert_file_creation_after_prepare_data(dataset_path, suffix)


def test_prepare_data_inh_indepFalse_mergeFalse(create_project_inh):
    project_path = (create_project_inh / project_name)
    indep_kepoints = False
    merge_keypoints = False
    dataset_name = f'{dataset_rootname}_indep{indep_kepoints}_merge{merge_keypoints}'
    dataset_path = (project_path / f'DISK_data/{dataset_name}')
    suffix, indep_kepoints_return, merge_keypoints_return = prepare_data_inh(project_path, dataset_name, indep_kepoints, merge_keypoints)
    print(suffix)
    assert suffix == '_set_keypoints'
    assert indep_kepoints_return == indep_kepoints
    assert merge_keypoints_return == merge_keypoints
    assert_file_creation_after_prepare_data(dataset_path, suffix)


def test_prepare_data_inh_indepFalse_mergeTrue(create_project_inh):
    project_path = (create_project_inh / project_name)
    indep_kepoints = False
    merge_keypoints = True
    dataset_name = f'{dataset_rootname}_indep{indep_kepoints}_merge{merge_keypoints}'
    dataset_path = (project_path / f'DISK_data/{dataset_name}')
    suffix, indep_kepoints_return, merge_keypoints_return = prepare_data_inh(project_path, dataset_name, indep_kepoints, merge_keypoints)
    print(suffix)
    assert suffix == '_set_keypoints_merged'
    assert indep_kepoints_return == indep_kepoints
    assert merge_keypoints_return == merge_keypoints
    assert_file_creation_after_prepare_data(dataset_path, suffix)




