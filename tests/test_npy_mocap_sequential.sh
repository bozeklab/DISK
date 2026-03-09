#!/bin/bash
set -e

DISK-create-project --project_path /home/france/Documents/DISK_human_mocap_sequential --file_format npy --data_files /home/france/mount_cvg/behavior_data/mocap_dataset/mocap_3djoints

DISK-prepare-data --project_path /home/france/Documents/DISK_human_mocap_sequential --original_freq 12 --subsampling_freq 12 --length 30 --stride 16  --fill_gap 0 --sequential true

DISK-train --project_path /home/france/Documents/DISK_human_mocap_sequential --network gru --dataset_name dataset_12Hz_30length_16stride --training_epochs 4 --indep_keypoints true --transforms-add_missing_pad 2 2

DISK-impute --project_path /home/france/Documents/DISK_human_mocap_sequential --dataset_name dataset_12Hz_30length_16stride --model_name DISK-gru_dataset_12Hz_30length_16stride

