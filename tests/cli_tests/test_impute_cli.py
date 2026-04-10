import sys
from textwrap import dedent
from unittest.mock import Mock, patch
from pathlib import Path
import os
import pytest
import inspect
from logging import Logger

from DISK.launchers import train_evaluate


## UTILS
def assert_train_evaluate_main_default_inputs(args):
    """(project_dir, model_dir, dataset_path, dataset_name, test_dir, skeleton_graph,
         training_seed, load_model_dir, cfg_network, training_batch_size,
         training_epochs, learning_rate, loss_type, loss_mask, loss_factor,
         model_scheduler_rate, model_scheduler_type, model_scheduler_steps_epoch,
         n_cpus, print_every,
         rerun_proba_files, indep_keypoints, merge_keypoints, suffix_proba_files,
         add_missing_pad, viewinvariant, normalize, normalizecube, swap,
         add_missing,
         test_original_coordinates, test_threshold_pck,
         n_repeat,
         total_n_plots, plot2d_only_holes, plot3d_size, plot3d_azim,
         logger, verbose=0)
    """
    # skeleton_graph
    from DISK.models.graph import Graph
    assert args['skeleton_graph'] is None or isinstance(args['skeleton_graph'], Graph)
    assert args['training_seed'] is None or type(args['training_seed']) == int
    assert args['load_model_dir'] is None or (type(args['load_model_dir']) == str and \
                                              (args['load_model_dir'] == '' or Path(args['load_model_dir']).exists()))

    assert type(args['cfg_network']) == dict
    assert type(args['training_batch_size']) == int and args['training_batch_size'] > 0
    assert type(args['training_epochs']) == int and args['training_epochs'] > 0
    assert type(args['learning_rate']) == float and args['learning_rate'] > 0
    assert type(args['loss_type']) == str
    assert type(args['loss_mask']) == bool
    assert (type(args['loss_factor']) == int or type(args['loss_factor']) == float) and args['loss_factor'] > 0
    assert type(args['model_scheduler_rate']) == float and args['model_scheduler_rate'] > 0
    assert type(args['model_scheduler_type']) == str
    assert type(args['model_scheduler_steps_epoch']) == int and args['model_scheduler_steps_epoch'] > 0
    assert type(args['n_cpus']) == int and args['n_cpus'] >= 0
    assert type(args['print_every']) == int and args['print_every'] > 0
    assert type(args['rerun_create_proba']) == bool
    assert type(args['indep_keypoints']) == bool
    assert type(args['merge_keypoints']) == bool
    assert type(args['suffix_proba_files']) == str
    assert (len(args['add_missing_pad']) == 2 and type(args['add_missing_pad'][0]) == int \
            and type(args['add_missing_pad'][1]) == int)
    assert type(args['viewinvariant']) == bool
    assert type(args['normalize']) == bool
    assert type(args['normalizecube']) == bool
    assert type(args['swap']) == float and args['swap'] >= 0 and args['swap'] <= 1
    assert type(args['add_missing']) == bool
    assert type(args['test_original_coordinates']) == bool
    assert (type(args['test_threshold_pck']) == float and args['test_threshold_pck'] >= 0 \
            and args['test_threshold_pck'] <= 1)
    assert type(args['n_repeat']) == int and args['n_repeat'] > 0
    assert type(args['total_n_plots']) == int and args['total_n_plots'] >= 0
    assert type(args['plot2d_only_holes']) == bool
    assert (type(args['plot3d_size']) == int or type(args['plot3d_size']) == float) and args['plot3d_size'] > 0
    assert (type(args['plot3d_azim']) == int or type(args['plot3d_azim']) == float) and args['plot3d_azim'] > 0
    assert isinstance(args['logger'], Logger)
    assert type(args['verbose']) == bool or (type(args['verbose']) == int and args['verbose'] >= 0)


## TESTS

# @pytest.mark.skip("")
def test_train_evaluate(tmp_path, monkeypatch):
    # GIVEN
    monkeypatch.chdir(tmp_path)  # set working directory to the temp directory for this test

    project_name = 'my_DISK_project'
    dataset_name = 'dataset'
    project_path = tmp_path.joinpath(project_name)
    project_path.mkdir(exist_ok=True, parents=True)

    for suffix, rerun in [['', True],
                          ['_set_keypoints_merged', True],
                          ['_set_keypoints', False],
                          ]:
        files = [
            # f'config_project.yaml',
            f'DISK_train/',
            f'DISK_data/',
            f'DISK_data/{dataset_name}/',
            f'DISK_data/{dataset_name}/proba_missing{suffix}.csv',
            f'DISK_data/{dataset_name}/proba_missing_length{suffix}.csv',
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
            """
            original_missing: true
            """
        ))

        # WHEN
        monkeypatch.setattr(
            sys,
            'argv',
            [
                'DISK-train',
                '--project_path', str(project_path),
                '--dataset_name', dataset_name,
            ]
        )

        with patch('DISK.launchers.train_evaluate.main') as main:
            train_evaluate.cli()

        # THEN
        sig = inspect.signature(train_evaluate.main)
        bound_args = sig.bind(*main.call_args.args, **main.call_args.kwargs)
        bound_args.apply_defaults()

        main.assert_called_once()

        assert bound_args.arguments['project_dir'] == str(project_path) # project_dir
        assert os.path.dirname(bound_args.arguments['model_dir']) == f'{project_path}/DISK_train'
        assert os.path.dirname(bound_args.arguments['test_dir']) == bound_args.arguments['model_dir']
        assert bound_args.arguments['dataset_path'] == str(f'{project_path}/DISK_data/{dataset_name}') # dataset_path
        assert bound_args.arguments['dataset_name'] == dataset_name # dataset_name
        assert bound_args.arguments['rerun_create_proba'] == rerun
        assert_train_evaluate_main_default_inputs(bound_args.arguments)
        ## checks what the main expects

    """(project_dir, model_dir, dataset_path, dataset_name, test_dir, skeleton_graph,
         training_seed, load_model_dir, cfg_network, training_batch_size,
         training_epochs, learning_rate, loss_type, loss_mask, loss_factor,
         model_scheduler_rate, model_scheduler_type, model_scheduler_steps_epoch,
         n_cpus, print_every,
         rerun_proba_files, indep_keypoints, merge_keypoints, suffix_proba_files,
         add_missing_pad, viewinvariant, normalize, normalizecube, swap,
         add_missing,
         test_original_coordinates, test_threshold_pck,
         n_repeat,
         total_n_plots, plot2d_only_holes, plot3d_size, plot3d_azim,
         logger, verbose=0)
    """
