#!/bin/bash
set -e

## not found
DISK-create-project --project_path DISK_DLC_H5 --file_format dlc_h5 --data_files /home/france/mount_cvg/behavior_data/dlc_project/videos/

DISK-prepare-data --project_path DISK_DLC_H5 --dataset_name test_dlc_h5 --original_freq 60 --subsampling_freq 60 --length 20 --stride 10 --discard_beginning 0 --discard_end -1 --fill_gap 10 --drop_keypoints head --sequential false --dlc_likelihood_threshold 0.8

DISK-train --project_path DISK_DLC_H5 --dataset_name test_dlc_h5 --training_epochs 4 --print_every 2 --n_cpus 6

DISK-impute --project_path DISK_DLC_H5 --dataset_name test_dlc_h5 --model_name DISK_test_dlc_h5 --threshold_error_score 1000
