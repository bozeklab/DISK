import sys
from unittest.mock import Mock, patch
from pathlib import Path
from textwrap import dedent
import pytest
import inspect
from logging import Logger

from DISK.launchers import prepare_data
from test_create_project_cli import generate_test_data


## UTILS
def create_files_and_folders(project_path, files):
    for file in files:
        p = project_path.joinpath(file)
        if file.endswith('/'):
            p.mkdir(exist_ok=True, parents=True)
        else:
            p.parent.mkdir(exist_ok=True, parents=True)
            p.touch()

def assert_prepare_data_main_default_inputs(args):
    """project_path, dataset_path, dataset_name, data_files, file_format,
         length, stride, fill_gap, sequential, original_freq, subsampling_freq,
         dlc_likelihood_threshold, discard_beginning, discard_end,
         drop_keypoints, indep_keypoints, merge_keypoints, skeleton_graph,
         logger"""

    assert type(args['dataset_path']) == str
    assert type(args['dataset_name']) == str
    assert type(args['dataset_files']) == list[str]
    assert type(args['file_format']) == str
    assert type(args['length']) == int and args['length'] > 0
    assert type(args['stride']) == int and args['stride'] > 0
    assert type(args['fill_gap']) == int and args['fill_gap'] > 0
    assert type(args['sequential']) == bool
    assert type(args['original_freq']) == int and args['original_freq'] > 0
    assert type(args['subsampling_freq']) == int and args['subsampling_freq'] > 0
    assert type(args['dlc_likelihood_threshold']) == float and \
           args['dlc_likelihood_threshold'] >= 0 and args['dlc_likelihood_threshold'] <= 1
    assert (type(args['discard_beginning']) == int or type(args['discard_beginning']) == float) \
           and args['discard_beginning'] >= 0
    assert (type(args['discard_end']) == int or type(args['discard_end']) == float) \
           and args['discard_end'] >= -1
    assert type(args['drop_keypoints']) == list
    assert type(args['indep_keypoints']) == bool
    assert type(args['merge_keypoints']) == bool
    from DISK.models.graph import Graph
    assert args['skeleton_graph'] is None or isinstance(args['skeleton_graph'], Graph)
    assert isinstance(args['logger'], Logger)


list_args = {
    'DISK_FL2': dict(
                     length=60,
                     discard_beginning=5,
                     discard_end=5,
                     fill_gap=0,
                     drop_keypoints='',
                     sequential=False,
                     ),
}

## TESTS
@pytest.mark.parametrize("project_name,file_format,file_names,input_args",
                         ['DISK_FL2', 'mat_qualisys', [f'{i}.mat' for i in range(10)], list_args['DISK_FL2']]
                         )
def test_prepare_data_ok(project_name,file_format,file_names, input_args, tmp_path, monkeypatch):
    # GIVEN
    monkeypatch.chdir(tmp_path)  # set working directory to the temp directory for this test

    project_path = tmp_path.joinpath(project_name)
    project_path.mkdir(exist_ok=True, parents=True)

    data_folder, data_files = generate_test_data(tmp_path, 'data', file_names)

    files = [
        f'DISK_train/',
    ]
    create_files_and_folders(project_path, files)

    config_path = project_path.joinpath('config_project.yaml')
    config_path.write_text(dedent(
        f"""
        original_missing: true
        file_format: {input_args['file_format']}
        data_files:
        - {data_files[0]}
        - {data_files[1]}
        """
    ))

    # WHEN
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'DISK-prepare-data',
            '--project_path', str(project_path),

        ] + [[f'--{k}', str(v)] for k,v in input_args.items()]
    )

    with patch('DISK.launchers.prepare_data.main') as main:
        main.return_value = {'keypoints': ['a', 'b'],
                             'divider': 3,
                             'no_original_missing': False,
                             'indep_keypoints': True,
                             'merge_keypoints': False,
                             'suffix': ''}
        prepare_data.cli()

    # THEN
    sig = inspect.signature(prepare_data.main)
    bound_args = sig.bind(*main.call_args.args, **main.call_args.kwargs)
    bound_args.apply_defaults()

    main.assert_called_once()


    ## checks what the main expects
    """project_path, dataset_path, dataset_name, data_files, file_format,
         length, stride, fill_gap, sequential, original_freq, subsampling_freq,
         dlc_likelihood_threshold, discard_beginning, discard_end,
         drop_keypoints, indep_keypoints, merge_keypoints, skeleton_graph,
         logger"""

