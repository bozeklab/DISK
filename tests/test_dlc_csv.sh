#!/bin/bash
set -e

DISK-create-project --project_path DISK_DLC_CSV --file_format dlc_csv --data_files /home/france/mount_cvg/behavior_data/freewalking_20220811_WTTB_fly1_1DLC_resnet50_FreeWalkingMp4Jul30shuffle1_190000.csv

DISK-prepare-data --project_path DISK_DLC_CSV --dataset_name test_dlc_csv --length 30 --stride 10 --fill_gap 10 --sequential true --dlc_likelihood_threshold 0.01 --discard_beginning 0 --discard_end -1 --drop_keypoints --indep_keypoints True --merge_keypoints False --original_freq 60

DISK-train --project_path DISK_DLC_CSV --dataset_name test_dlc_csv --training_epochs 4 --n_cpus 6 --indep_keypoints true --network gru

DISK-impute --project_path DISK_DLC_CSV --dataset_name test_dlc_csv --model_name DISK-gru_test_dlc_csv
