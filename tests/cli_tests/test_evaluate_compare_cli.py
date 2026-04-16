import sys
from textwrap import dedent
from unittest.mock import Mock, patch
from pathlib import Path
import os
import pytest
import inspect
from logging import Logger

from DISK.launchers import evaluate_compare
from tests.cli_tests.test_prepare_data_cli import create_files_and_folders


## UTILS
def assert_evaluate_compare_main_default_inputs(args):
    """(project_dir, model_dirs, dataset_path, dataset_name, test_dir, skeleton_graph,
         training_batch_size,
         loss_type, loss_mask, loss_factor,
         n_cpus,
         rerun_create_proba, indep_keypoints, merge_keypoints,
         add_missing_pad, viewinvariant, normalize, normalizecube, swap,
         add_missing,
         test_original_coordinates, pck_threshold,
         n_repeat,
         total_n_plots, plot2d_only_holes, plot3d_size, plot3d_azim,
         logger, suffix='', verbose=0)
    """
    # skeleton_graph
    from DISK.models.graph import Graph
    assert args['skeleton_graph'] is None or isinstance(args['skeleton_graph'], Graph)
    assert type(args['model_dirs']) == list and \
          ([Path(m).exists() for m in args['model_dirs']])
    assert type(args['loss_type']) == str
    assert type(args['loss_mask']) == bool
    assert (type(args['loss_factor']) == int or type(args['loss_factor']) == float) and args['loss_factor'] > 0
    assert type(args['n_cpus']) == int and args['n_cpus'] >= 0
    assert type(args['rerun_create_proba']) == bool
    assert type(args['indep_keypoints']) == bool
    assert type(args['merge_keypoints']) == bool
    assert (len(args['add_missing_pad']) == 2 and type(args['add_missing_pad'][0]) == int \
            and type(args['add_missing_pad'][1]) == int)
    assert type(args['viewinvariant']) == bool
    assert type(args['normalize']) == bool
    assert type(args['normalizecube']) == bool
    assert type(args['swap']) == float and args['swap'] >= 0 and args['swap'] <= 1
    assert type(args['add_missing']) == bool
    assert type(args['test_original_coordinates']) == bool
    assert (type(args['pck_threshold']) == float and args['pck_threshold'] >= 0 \
            and args['pck_threshold'] <= 1)
    assert type(args['n_repeat']) == int and args['n_repeat'] > 0
    assert type(args['total_n_plots']) == int and args['total_n_plots'] >= 0
    assert type(args['plot2d_only_holes']) == bool
    assert (type(args['plot3d_size']) == int or type(args['plot3d_size']) == float) and args['plot3d_size'] > 0
    assert (type(args['plot3d_azim']) == int or type(args['plot3d_azim']) == float) and args['plot3d_azim'] > 0
    assert isinstance(args['logger'], Logger)
    assert type(args['verbose']) == bool or (type(args['verbose']) == int and args['verbose'] >= 0)
    assert type(args['suffix_proba_files']) == str


list_args = {
    'DISK_human_mocap': dict()
}

## TESTS

@pytest.mark.parametrize("project_name, model_names,input_args",
                         [
                            ['DISK_human_mocap', ['DISK_model1'], list_args['DISK_human_mocap']],
                             ['DISK_human_mocap', ['DISK_model1', 'DISK_model2'], list_args['DISK_human_mocap']],
                         ]
                         )
def test_evaluate_compare(project_name, model_names, input_args, tmp_path, monkeypatch):
    # GIVEN
    monkeypatch.chdir(tmp_path)  # set working directory to the temp directory for this test

    dataset_name = 'dataset'
    project_path = tmp_path.joinpath(project_name)
    project_path.mkdir(exist_ok=True, parents=True)

    proba_files_suffix = ''
    for model_name in model_names:
        files = [
            f'DISK_train/{model_name}/model_epoch14',
            f'DISK_data/{dataset_name}/proba_missing{proba_files_suffix}.csv',
            f'DISK_data/{dataset_name}/proba_missing_length{proba_files_suffix}.csv',
        ]
        create_files_and_folders(project_path, files)

    config_path = project_path.joinpath('config_project.yaml')
    config_path.write_text(dedent(
        """
        original_missing: true
        """
    ))

    # WHEN
    cli = [
        'DISK-evaluate',
            '--project_path', str(project_path),
            '--dataset_name', dataset_name,
            '--model_name_list', model_name,

    ]
    for k, v in input_args.items():
        if type(v) == list:
            cli.append(f'--{k}')
            cli.extend([str(vv) for vv in v])
        else:
            cli.extend([f'--{k}', str(v)])
    monkeypatch.setattr(
        sys,
        'argv',
        cli
    )

    with patch('DISK.launchers.evaluate_compare.main') as main:
        evaluate_compare.cli()

    # THEN
    sig = inspect.signature(evaluate_compare.main)
    bound_args = sig.bind(*main.call_args.args, **main.call_args.kwargs)
    bound_args.apply_defaults()

    main.assert_called_once()

    assert bound_args.arguments['project_dir'] == str(project_path) # project_dir
    assert bound_args.arguments['model_dirs'][0] == f'{project_path}/DISK_train/{model_name}'
    assert bound_args.arguments['dataset_path'] == str(f'{project_path}/DISK_data/{dataset_name}')  # dataset_path
    assert bound_args.arguments['dataset_name'] == dataset_name  # dataset_name
    assert os.path.dirname(bound_args.arguments['test_dir']) == f'{project_path}/DISK_train'

    assert_evaluate_compare_main_default_inputs(bound_args.arguments)
    ## checks what the main expects

    """(project_dir, model_dirs, dataset_path, dataset_name, test_dir, skeleton_graph,
         training_batch_size,
         loss_type, loss_mask, loss_factor,
         n_cpus,
         rerun_create_proba, indep_keypoints, merge_keypoints,
         add_missing_pad, viewinvariant, normalize, normalizecube, swap,
         add_missing,
         test_original_coordinates, pck_threshold,
         n_repeat,
         total_n_plots, plot2d_only_holes, plot3d_size, plot3d_azim,
         logger, suffix='', verbose=0)
    """
