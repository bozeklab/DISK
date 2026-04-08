from pathlib import Path
import numpy as np
import pandas as pd
import os
import yaml
import torch


def is_empty(path: Path):
    return any(path.iterdir())


def assert_file_creation_after_create_project(project_path):
    assert project_path.is_dir()

    assert (project_path / "config_project.yaml").is_file()

    assert (project_path / "DISK_data").is_dir()
    assert not is_empty(project_path / "DISK_data")

    assert (project_path / "DISK_train").is_dir()
    assert not is_empty(project_path / "DISK_train")

    assert (project_path / "DISK_impute").is_dir()
    assert not is_empty(project_path / "DISK_impute")

    assert (project_path / "example_configs").is_dir()
    assert is_empty(project_path / "example_configs")


def assert_file_creation_after_prepare_data(dataset_path, suffix):
    for set in ['train', 'test', 'val']:
        for n_nans in ['w-0-nans', 'w-all-nans']:
            for appendix in ['', 'fulllength_']:
                assert (dataset_path / f"{set}_{appendix}dataset_{n_nans}.npz").is_file()
                data = np.load(dataset_path / f"{set}_{appendix}dataset_{n_nans}.npz")
                assert 'X' in data.keys()
                if appendix == '':
                    assert 'lengths' in data.keys()
                    assert data['X'].shape[0] == data['lengths'].shape[0]
                else:
                    assert 'files' in data.keys()
                    assert 'time' in data.keys()
                    assert data['X'].shape[0] == data['files'].shape[0]
                    assert data['X'].shape[0] == data['time'].shape[0]

    assert (dataset_path / f"proba_missing_length{suffix}.csv").is_file()
    assert (dataset_path / f"proba_missing{suffix}.csv").is_file()
    assert (dataset_path / f"constants.py").is_file()

    df = pd.read_csv(dataset_path.joinpath(f"proba_missing{suffix}.csv"))
    assert 'keypoint' in df.columns and 'proba' in df.columns and len(df.columns) == 2

    df = pd.read_csv(dataset_path.joinpath(f"proba_missing_length{suffix}.csv"))
    assert 'keypoint' in df.columns and 'proba' in df.columns and 'length' in df.columns and len(df.columns) == 3

    from DISK.utils.utils import read_constant_file
    constants = read_constant_file(dataset_path.joinpath(f"constants.py"))
    assert 'NUM_FEATURES' in constants.__dict__.keys()
    assert 'KEYPOINTS' in constants.__dict__.keys()
    assert 'DIVIDER' in constants.__dict__.keys()
    assert 'SEQ_LENGTH' in constants.__dict__.keys()
    assert 'STRIDE' in constants.__dict__.keys()
    assert 'ORIG_FREQ' in constants.__dict__.keys()
    assert 'FREQ' in constants.__dict__.keys()
    assert 'W_RESIDUALS' in constants.__dict__.keys()
    assert 'FILE_TYPE' in constants.__dict__.keys()
    assert 'DLC_LIKELIHOOD_THRESHOLD' in constants.__dict__.keys()


def assert_and_get_network_config(network):
    network_config = None
    script_directory = os.path.dirname(os.path.abspath(__file__))
    if network == 'gru':
        with open(os.path.join(script_directory, f'../../DISK/conf/network/gru.yaml'), 'r') as file:
            network_config = yaml.safe_load(file)

        ## THEN
        assert 'num_layers' in network_config.keys()
        assert 'dropout' in network_config.keys()
        assert 'type' in network_config.keys()
        assert 'size_layer' in network_config.keys()
        assert 'mu_sigma' in network_config.keys()
        assert 'beta_mu_sigma' in network_config.keys()
    elif network == 'transformer':
        with open(os.path.join(script_directory, f'../../DISK/conf/network/transformer.yaml'), 'r') as file:
            network_config = yaml.safe_load(file)

        ## THEN
        assert 'input_type' in network_config.keys()
        assert 'encoding' in network_config.keys()
        assert 'type' in network_config.keys()
        assert 'num_layers' in network_config.keys()
        assert 'dim_ff' in network_config.keys()
        assert 'd_model' in network_config.keys()
        assert 'num_heads' in network_config.keys()
        assert 'activation' in network_config.keys()
        assert 'norm_first' in network_config.keys()
        assert 'attn_type' in network_config.keys()
        assert 'norm' in network_config.keys()
        assert 'dropout' in network_config.keys()
        assert 'mu_sigma' in network_config.keys()
        assert 'beta_mu_sigma' in network_config.keys()
    return network_config


def assert_file_creation_after_train(model_dir, best_epoch, last_epoch, print_every):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    assert model_dir.joinpath(f'model_epoch{best_epoch}').is_file()
    data = torch.load(model_dir.joinpath(f'model_epoch{best_epoch}'), map_location=torch.device(device))
    assert 'model_state_dict' in data.keys()
    assert 'optimizer_state_dict' in data.keys()
    assert 'print_every' in data.keys()
    assert 'lr' in data.keys()
    assert 'epoch' in data.keys()

    assert model_dir.joinpath(f'model_last_epoch{last_epoch}').is_file()
    data = torch.load(model_dir.joinpath(f'model_last_epoch{last_epoch}'), map_location=torch.device(device))
    assert 'model_state_dict' in data.keys()
    assert 'optimizer_state_dict' in data.keys()
    assert 'print_every' in data.keys()
    assert 'lr' in data.keys()
    assert 'epoch' in data.keys()

    assert model_dir.joinpath(f'loss.svg').is_file()

    losses = pd.read_csv(model_dir.joinpath('training_losses.txt'), sep=' ', header=None)
    print(losses)
    assert len(losses.columns) >= 5
    print(len(losses), last_epoch, print_every, last_epoch // print_every)
    assert len(losses) == last_epoch // print_every
    ## we start counting epochs at 1 not at 0 (hence best_epoch - 1)
    assert np.argmin(losses.iloc[:, 3]) == best_epoch // print_every - 1


def assert_file_creation_after_evaluate(test_dir, model_name, n_plots, n_repeat, pck_threshold, suffix):
    assert len(list(test_dir.joinpath('visualize_prediction_val').iterdir())) == n_plots * n_repeat
    if n_plots > 0:
        for f in test_dir.joinpath('visualize_prediction_val').iterdir():
            assert f.is_file()
            assert str(f).endswith('.png')

    for i_repeat in range(n_repeat):
        for metric in ('MPJPE', 'RMSE', f'PCK@{pck_threshold}'):
            assert test_dir.joinpath(f'barplot_comparison_{metric}{suffix}_repeat-{i_repeat}.png').is_file()
            assert test_dir.joinpath(f'comparison_length_hole_kp_vs_{metric}{suffix}_repeat-{i_repeat}.png').is_file()
            assert test_dir.joinpath(f'thresholding_curve_{metric}{suffix}_repeat-{i_repeat}.png').is_file()

        total_metrics_file_path = test_dir.joinpath(f'total_metrics{suffix}_repeat-{i_repeat}.csv')
        assert total_metrics_file_path.is_file()
        total_metrics = pd.read_csv(total_metrics_file_path)
        assert 'index' in total_metrics.columns
        assert 'id_sample' in total_metrics.columns
        assert 'id_hole' in total_metrics.columns
        assert 'keypoint' in total_metrics.columns
        assert 'method' in total_metrics.columns
        assert 'method_param' in total_metrics.columns
        assert 'RMSE' in total_metrics.columns
        assert 'MPJPE' in total_metrics.columns
        assert f'PCK@{pck_threshold}' in total_metrics.columns
        assert 'mean_uncertainty' in total_metrics.columns
        assert 'length_hole' in total_metrics.columns
        assert 'swap_kp_id' in total_metrics.columns
        assert 'swap_length' in total_metrics.columns
        assert 'average_dist_bw_swap_kp' in total_metrics.columns

        assert total_metrics['RMSE'].dtype == float
        assert total_metrics['MPJPE'].dtype == float
        assert total_metrics[f'PCK@{pck_threshold}'].dtype == float
        assert total_metrics['index'].dtype == int
        assert total_metrics['id_sample'].dtype == int
        assert total_metrics['id_hole'].dtype == int

        assert test_dir.joinpath(f'corrplot-model-RMSE-{model_name}{suffix}_repeat-{i_repeat}.png').is_file()

    mean_metric_file_path = test_dir.joinpath(f'mean_metrics{suffix}.csv')
    assert mean_metric_file_path.is_file()
    mean_metrics = pd.read_csv(mean_metric_file_path)

    assert 'method' in mean_metrics.columns
    assert 'method_param' in mean_metrics.columns
    assert f'PCK@{pck_threshold}' in mean_metrics.columns
    assert 'RMSE' in mean_metrics.columns
    assert 'MPJPE' in mean_metrics.columns
    assert 'repeat' in mean_metrics.columns
    assert 'dataset' in mean_metrics.columns

    assert mean_metrics['RMSE'].dtype == float
    assert mean_metrics['MPJPE'].dtype == float
    assert mean_metrics[f'PCK@{pck_threshold}'].dtype == float
    assert mean_metrics['repeat'].dtype == int


def assert_after_impute_no_gaps_found(mess):
    assert "Could not find short-enough segments to be imputed by the DISK model" in mess


def assert_after_impute_gaps_found(impute_dir, data_files, mess, n_plots):
    assert "Successfully imputed data with DISK model" in mess

    assert impute_dir.is_dir()
    assert impute_dir.joinpath("plots").is_dir()
    assert impute_dir.joinpath("plots/hist_uncertainty.png").is_file()

    assert len(list(impute_dir.joinpath('plots').iterdir())) >= n_plots + 1
    if n_plots > 0:
        for f in impute_dir.joinpath('plots').iterdir():
            assert f.is_file()
            assert str(f).endswith('.png')

    for f in data_files:
        assert impute_dir.joinpath(f).is_file()