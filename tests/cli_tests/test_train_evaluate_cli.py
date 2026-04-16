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
list_args = {
    'GRU_indep_kp_true': dict(
                    training_epochs=4,
                    n_cpus=6,
                    indep_keypoints=True,
                    network='gru'
                     ),
    'GRU_indep_kp_false': dict(
        training_epochs=4,
        n_cpus=6,
        indep_keypoints=False,
        network='gru',
        model_name='GRU_test'
    ),
    'transformer_indep_kp_false': dict(
        training_epochs=4,
        print_every=1,
        n_cpus=0,
        indep_keypoints=False,
        merge_keypoints=False,
        network='transformer'
    ),
    'transformer_indep_kp_false_WRONG_PRINT_EVERY0': dict(
        training_epochs=4,
        print_every=0,
        n_cpus=6,
        indep_keypoints=False,
        network='transformer'
    ),
    'transformer_indep_kp_false_WRONG_PRINT_EVERY-1': dict(
        training_epochs=8,
        print_every=-1,
        n_cpus=6,
        indep_keypoints=False,
        network='transformer',
        transforms_add_missing_pad=[2, 2]
    ),
}
@pytest.mark.parametrize("project_name,dataset_name,suffix,rerun,input_args",
                         [
                             ['GRU_indep_kp_true', 'test_dlc_csv', '', False, list_args['GRU_indep_kp_true']],
                             ['GRU_indep_kp_false', 'test_dlc_csv', '', True, list_args['GRU_indep_kp_false']],
                             ['GRU_indep_kp_true', 'test_dlc_csv', '_set_keypoints', True, list_args[
                                 'GRU_indep_kp_true']],
                             ['GRU_indep_kp_false', 'test_dlc_csv', '_set_keypoints', False, list_args['GRU_indep_kp_false']],
['transformer_indep_kp_false', 'test_dlc_csv', '_set_keypoints', False, list_args['transformer_indep_kp_false']],
['transformer_indep_kp_false_WRONG_PRINT_EVERY0', 'test_dlc_csv', '_set_keypoints', False, list_args[
    'transformer_indep_kp_false_WRONG_PRINT_EVERY0']],
['transformer_indep_kp_false_WRONG_PRINT_EVERY-1', 'test_dlc_csv', '_set_keypoints', False, list_args[
    'transformer_indep_kp_false_WRONG_PRINT_EVERY-1']],
                             # ['DISK_DLC_CSV', 'dlc_csv', ['ex.csv']],
                          # ['DISK_CSV', 'simple_csv', ['ex.csv']],
                          # ['DISK_DLC_H5', 'dlc_h5', ['ex.h5', 'ex2.h5']],
                          # ['DISK_SLEAP_H5', 'sleap_h5', ['ex.h5', 'ex2.h5']],
                          # ['DISK_NPY', 'npy', [f'{i}.npy' for i in range(10)]],
                          # ['DISK_PKL', 'df3d_pkl', [f'{i}.pkl' for i in range(10)]],
                          # pytest.param('DISK_NPY', 'mat_dannce', [f'{i}.npy' for i in range(10)],
                          #              marks=pytest.mark.xfail),
                          ]
                         )
def test_train_evaluate(project_name, dataset_name,suffix, rerun, input_args, tmp_path, monkeypatch):
    # GIVEN
    monkeypatch.chdir(tmp_path)  # set working directory to the temp directory for this test

    project_path = tmp_path.joinpath(project_name)
    project_path.mkdir(exist_ok=True, parents=True)


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
    cli = [
        'DISK-train',
            '--project_path', str(project_path),
            '--dataset_name', dataset_name,

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
