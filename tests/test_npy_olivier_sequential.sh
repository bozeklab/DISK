#!/bin/bash
set -e

DISK-create-project --project_path DISK_olivier_npy --file_format npy --data_files /home/france/Dropbox/2025_UniBonn/behavior_data/DISK/female_2318.npy

DISK-prepare-data --project_path DISK_olivier_npy --length 240 --stride 30 --fill_gap 10 --sequential true
DISK-add-skeleton --project_path DISK_olivier_npy

DISK-train --project_path DISK_olivier_npy --network transformer --dataset_name dataset_240_30 --training_epochs 2 --n_cpus 6 --indep_keypoints true

DISK-impute --project_path DISK_olivier_npy --dataset_name dataset_240_30 --model_name dataset_240_30_DISK

