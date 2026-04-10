import sys
from textwrap import dedent
from unittest.mock import Mock, patch
from pathlib import Path
import os
import pytest
import inspect
from logging import Logger

from DISK.launchers import impute
from test_create_project_cli import generate_test_data
from DISK.launchers.create_project import possible_file_format_values

## UTILS
def assert_impute_main_default_inputs(args):
    """project_dir, impute_dir, plot_dir, file_format, dataset_path, skeleton_graph, checkpoint, batch_size,
         threshold_error_score, total_n_plots, plot_only_holes, missing_pad, logger, verbose=0):

    """
    assert type(args['project_dir']) == str and Path(args['project_dir']).exists()
    assert type(args['impute_dir']) == str and Path(args['impute_dir']).exists()
    assert type(args['plot_dir']) == str and Path(args['plot_dir']).exists()
    assert type(args['dataset_path']) == str and Path(args['dataset_path']).exists()
    assert type(args['checkpoint']) == str and Path(args['checkpoint']).exists()
    assert type(args['file_format']) == str and args['file_format'] in possible_file_format_values
    # skeleton_graph
    from DISK.models.graph import Graph
    assert args['skeleton_graph'] is None or isinstance(args['skeleton_graph'], Graph)

    assert type(args['batch_size']) == int and args['batch_size'] > 0
    assert (type(args['threshold_error_score']) == float or type(args['threshold_error_score']) == int) \
            and args['threshold_error_score'] >= 0
    assert type(args['total_n_plots']) == int and args['total_n_plots'] >= 0
    assert type(args['plot_only_holes']) == bool
    assert (len(args['missing_pad']) == 2 and type(args['missing_pad'][0]) == int \
            and type(args['missing_pad'][1]) == int)
    assert isinstance(args['logger'], Logger)
    assert type(args['verbose']) == bool or (type(args['verbose']) == int and args['verbose'] >= 0)


## TESTS

# @pytest.mark.skip("")
def test_impute(tmp_path, monkeypatch):
    # GIVEN
    monkeypatch.chdir(tmp_path)  # set working directory to the temp directory for this test

    project_name = 'my_DISK_project'
    dataset_name = 'dataset'
    project_path = tmp_path.joinpath(project_name)
    project_path.mkdir(exist_ok=True, parents=True)
    model_name = 'DISK_model1'

    file_format = 'mat_dannce'
    file_names = ['1.mat', '2.mat']
    data_folder, data_files = generate_test_data(tmp_path, dataset_name, file_names)

    suffix_proba_files = ''
    files = [
        # f'config_project.yaml',
        f'DISK_train/{model_name}/model_epoch14',
        f'DISK_data/{dataset_name}/',
        f'DISK_data/{dataset_name}/proba_missing{suffix_proba_files}.csv',
        f'DISK_data/{dataset_name}/proba_missing_length{suffix_proba_files}.csv',
        'DISK_impute/'
    ]
    for file in files:
        p = project_path.joinpath(file)
        if file.endswith('/'):
            p.mkdir(exist_ok=True, parents=True)
        else:
            p.parent.mkdir(exist_ok=True, parents=True)
            p.touch()

    config_path = project_path.joinpath('config_project.yaml')
    config_path.write_text(dedent(
        f"""
        original_missing: true
        file_format: {file_format}
        """
    ))

    # WHEN
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'DISK-impute',
            '--project_path', str(project_path),
            '--dataset_name', dataset_name,
            '--model_name', model_name
        ]
    )

    with patch('DISK.launchers.impute.main') as main:
        impute.cli()

    # THEN
    sig = inspect.signature(impute.main)
    bound_args = sig.bind(*main.call_args.args, **main.call_args.kwargs)
    bound_args.apply_defaults()

    main.assert_called_once()

    assert bound_args.arguments['project_dir'] == str(project_path) # project_dir
    assert Path(bound_args.arguments['plot_dir']).exists()
    assert os.path.dirname(bound_args.arguments['impute_dir']) == f'{project_path}/DISK_impute'
    assert bound_args.arguments['checkpoint'] == f'{project_path}/DISK_train/{model_name}'
    assert bound_args.arguments['dataset_path'] == str(f'{project_path}/DISK_data/{dataset_name}') # dataset_path
    assert bound_args.arguments['file_format'] == file_format
    ## checks what the main expects
    assert_impute_main_default_inputs(bound_args.arguments)
    """project_dir, impute_dir, plot_dir, file_format, dataset_path, skeleton_graph, checkpoint, batch_size,
         threshold_error_score, total_n_plots, plot_only_holes, missing_pad, logger, verbose=0
    """
