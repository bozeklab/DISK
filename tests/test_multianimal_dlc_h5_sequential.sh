#!/bin/bash
set -e

DISK-create-project --project_path DISK_multianimal_dlc_h5_seq --file_format dlc_h5 --data_files /home/france/mount_cvg/behavior_data/calms21-disk-dlc/calms21-01.01-snapshot195-dlc-h5/

DISK-prepare-data --project_path DISK_multianimal_dlc_h5_seq --original_freq 30 --subsampling_freq 30 --length 30 --stride 30 --fill_gap 10 --sequential true --dlc_likelihood_threshold 0.01 --dataset_name dataset1

DISK-train --network gru --project_path DISK_multianimal_dlc_h5_seq --dataset_name dataset1 --training_epochs 4 --indep_keypoints true

DISK-impute --project_path DISK_multianimal_dlc_h5_seq --dataset_name dataset1 --model_name DISK-gru_dataset1