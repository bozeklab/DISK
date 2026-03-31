from pathlib import Path

import pytest

from DISK.launchers import create_project


def is_empty(path: Path):
    return any(path.iterdir())

data_file_list = [
    "/home/france/mount_cvg/behavior_data/mocap_dataset/mocap_3djoints_subset/90_15.npy",
    "/home/france/mount_cvg/behavior_data/mocap_dataset/mocap_3djoints_subset/90_23.npy",
    "/home/france/mount_cvg/behavior_data/mocap_dataset/mocap_3djoints_subset/90_29.npy",
]

def test_create_project_human_mocap_3d(tmp_path):
    # GIVEN
    project_path = tmp_path.joinpath("DISK_mocap_dataset")
    assert not project_path.is_dir()

    # WHEN
    create_project.main(
        project_path=project_path,
        data_file_list=data_file_list,
        file_type="npy",
    )

    # THEN
    assert project_path.is_dir()

    assert project_path.joinpath("config_project.yaml").is_file()

    assert project_path.joinpath("DISK_data").is_dir()
    assert not is_empty(project_path.joinpath("DISK_data"))

    assert project_path.joinpath("DISK_train").is_dir()
    assert not is_empty(project_path.joinpath("DISK_train"))

    assert project_path.joinpath("DISK_impute").is_dir()
    assert not is_empty(project_path.joinpath("DISK_impute"))

    assert project_path.joinpath("example_configs").is_dir()
    assert is_empty(project_path.joinpath("example_configs"))




def test_create_project_errors_if_project_path_already_exists(tmp_path):
    project_path = tmp_path.joinpath("DISK_mocap_dataset")

    # first time, no problem
    create_project.main(
        project_path=project_path,
        data_file_list=data_file_list,
        file_type="npy",
    )

    # second time, should a problem
    with pytest.raises(FileExistsError):
        create_project.main(
            project_path=project_path,
            data_file_list=data_file_list,
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