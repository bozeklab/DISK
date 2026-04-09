import pytest
import logging
from DISK.launchers import prepare_data

data_file_list = [
    "/home/france/mount_cvg/behavior_data/mocap_dataset/mocap_3djoints_subset/90_15.npy",
    "/home/france/mount_cvg/behavior_data/mocap_dataset/mocap_3djoints_subset/90_23.npy",
    "/home/france/mount_cvg/behavior_data/mocap_dataset/mocap_3djoints_subset/90_29.npy",
]


@pytest.mark.skip(reason="Not implemented yet")
def test_prepare_data(tmp_path):
    # GIVEN
    dataset_name = 'test_dataset'
    project_path = tmp_path.joinpath("DISK_mocap_dataset")
    kwargs = dict(
        project_path=project_path,
        data_files =data_file_list,
        file_format ="npy",
        dataset_name = dataset_name,
        dataset_path = project_path.join(f'DISK_data/{dataset_name}'),
        length = 30,
        stride= 15,
        fill_gap = 0,
        sequential = True,
        original_freq = 1,
        subsampling_freq = 1,
        dlc_likelihood_threshold = 0.1,
        discard_beginning = 0,
        discard_end = -1,
        drop_keypoints = [],
        indep_keypoints = True,
        merge_keypoints = False,
        skeleton_graph = None,
        logger = logging.getLogger(),
    )

    # WHEN
    prepare_data.main(**kwargs)


    # THEN

