import sys
from unittest.mock import Mock, patch
from pathlib import Path

import pytest

from DISK.launchers import create_project


## UTILS

def generate_test_data(root: Path, foldername: str, file_names: list[str]) -> tuple[str, list[str]]:

    data_folder = root.joinpath(foldername)
    data_files = []
    for fname in file_names:
        fpath = data_folder.joinpath(fname)
        fpath.parent.mkdir(parents=True, exist_ok=True)
        fpath.touch()
        data_files.append(fpath)

    return str(data_folder), [str(p) for p in data_files]


def run_create_project(monkeypatch, project_path: str, file_format: str, data_files: list[str]) -> Mock:
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'DISK-create-project',
            '--project_path', project_path,
            '--file_format', file_format,
            '--data_files'
        ] + data_files
    )

    with patch('DISK.launchers.create_project.main') as main:
        create_project.cli()

    return main

def assert_main_create_project_call(main, project_path, file_format, data_files):

    main.assert_called_once()
    args, kwargs = main.call_args
    assert kwargs['project_path'] == project_path
    assert kwargs['file_format'] == file_format
    assert set(kwargs['data_file_list']) == set(data_files)


## TESTS

def test_create_project_mat_files_ok(tmp_path, monkeypatch):
    # vanilla test with project_name as relative path, list of files, and correct format
    # GIVEN
    monkeypatch.chdir(tmp_path)  # set working directory to the temp directory for this test
    project_name = 'my_DISK_project'
    file_format = 'mat_dannce'
    project_path = str(tmp_path.joinpath(project_name))
    file_names = ['1.mat', '2.mat']
    data_folder, data_files = generate_test_data(tmp_path, 'data', file_names)

    # WHEN
    main = run_create_project(monkeypatch, project_path, file_format, data_files)

    # THEN
    assert_main_create_project_call(main, project_path, file_format, data_files)


def test_create_project_mat_folder_ok(tmp_path, monkeypatch):
    # same test but providing folder instead of list of files

    # GIVEN
    monkeypatch.chdir(tmp_path)  # set working directory to the temp directory for this test

    # data/1.mat, data/2.mat
    file_names = ['1.mat', '2.mat']
    file_format = 'mat_dannce'
    project_name = 'my_DISK_project'
    project_path = str(tmp_path.joinpath('my_DISK_project'))

    data_folder, data_files = generate_test_data(tmp_path, 'data', file_names)
    generate_test_data(tmp_path, 'data', ['other_file.txt'])

    # WHEN
    main = run_create_project(monkeypatch, project_name, file_format, [data_folder])

    # THEN
    assert_main_create_project_call(main, project_path, file_format, data_files)
