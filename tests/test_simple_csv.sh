#!/bin/bash
set -e

## TEST 3D
DISK-create-project project_path=DISK_simple_csv file_type=simple_csv data_files=[/home/france/mount_cvg/behavior_data/fish_data/fish_fighting_interpolated_head.csv]

DISK-prepare-data project_path=DISK_simple_csv dataset_name=test_simple_csv indep_keypoints=True merge_keypoints=False original_freq=60 subsampling_freq=60 length=30 stride=30 discard_beginning=0 discard_end=-1 fill_gap=10 drop_keypoints=[] sequential=true

DISK-train network=transformer project_path=DISK_simple_csv dataset_name=test_simple_csv training_epochs=4 n_cpus=0

DISK-impute project_path=DISK_simple_csv dataset_name=test_simple_csv model_name=test_simple_csv_DISK_2

## TEST 2D
DISK-create-project project_path=DISK_simple_csv_2D file_type=simple_csv data_files=[/home/france/mount_cvg/behavior_data/fish_data/fish_fighting_interpolated_head_2D.csv]

DISK-prepare-data project_path=DISK_simple_csv_2D dataset_name=test_simple_csv_2D indep_keypoints=True merge_keypoints=False original_freq=60 subsampling_freq=60 length=30 stride=30 discard_beginning=0 discard_end=-1 fill_gap=10 drop_keypoints=[] sequential=true

DISK-train project_path=DISK_simple_csv_2D dataset_name=test_simple_csv_2D training_epochs=3 n_cpus=1

DISK-impute project_path=DISK_simple_csv_2D dataset_name=test_simple_csv_2D model_name=test_simple_csv_2D_DISK

## 3D non sequential
DISK-create-project project_path=DISK_simple_3csv file_type=simple_csv data_files=[/home/france/mount_cvg/behavior_data/fish_data/Fish_fight_data_v3/FishTank20200824_151740_pp.csv,/home/france/mount_cvg/behavior_data/fish_data/Fish_fight_data_v3/FishTank20200902_160124_pp.csv,/home/france/mount_cvg/behavior_data/fish_data/Fish_fight_data_v3/FishTank20200903_160946_pp.csv]

DISK-prepare-data project_path=DISK_simple_3csv dataset_name=test indep_keypoints=True merge_keypoints=False length=60

DISK-train project_path=DISK_simple_3csv dataset_name=test training_epochs=2 indep_keypoints=true

DISK-impute project_path=DISK_simple_3csv dataset_name=test model_name=