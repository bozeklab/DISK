#!/bin/bash
set -e

DISK-create-project --project_path DISK_human_mocap --file_format npy --data_files /home/france/mount_cvg/behavior_data/mocap_dataset/mocap_3djoints_subset/

DISK-prepare-data --project_path DISK_human_mocap --length 20

DISK-train --project_path DISK_human_mocap --network gru --dataset_name dataset_20_10  --training_epochs 4

yes y | DISK-train --project_path DISK_human_mocap --network gru --dataset_name dataset_20_10  --training_epochs 4 --load_model DISK-gru_dataset_20_10

DISK-test --project_path DISK_human_mocap --dataset_name dataset_20_10  --model_name_list DISK-gru_dataset_20_10

DISK-impute --project_path DISK_human_mocap --dataset_name dataset_20_10  --model_name DISK-gru_dataset_20_10