import sys
from unittest.mock import Mock, patch

import pytest

from DISK.launchers import create_project


def test_create_project(tmp_path, monkeypatch):

    # GIVEN
    monkeypatch.chdir(tmp_path)  # set working directory to the temp directory for this test

    # data/1.mat, data/2.mat
    tmp_path.joinpath('data').mkdir()
    data_files = []
    for fname in ['1.mat', '2.mat']:
        tmp_path.joinpath(f"data/{fname}").touch()
        data_files.append(str(tmp_path.joinpath(f"data/{fname}")))
    assert not tmp_path.joinpath('my_DISK_project').exists()

    file_format = 'mat_dannce'


    # WHEN
    # main = Mock(spec=create_project.main)

    monkeypatch.setattr(
        sys,
        'argv',
        [
        'DISK-create-project',
         '--project_path', 'my_DISK_project',
         '--file_format', file_format,
         '--data_files', data_files[0], data_files[1],
        ]
    )

    with patch('DISK.launchers.create_project.main') as main:
        create_project.cli()

    # THEN
    main.assert_called_once_with(
        project_path=str(tmp_path.joinpath('my_DISK_project')),
        data_file_list=data_files,
        file_type=file_format,
    )
