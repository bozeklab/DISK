#!/bin/bash
set -e

DISK-create-project --project_path DISK_test2_multianimal_dlc_h5 --file_format dlc_h5 --data_files /home/france/mount_cvg/behavior_data/calms21-disk-dlc/calms21-01.01-snapshot195-dlc-h5/

DISK-prepare-data --project_path DISK_test2_multianimal_dlc_h5 --dataset_name test2 --indep_keypoints True --merge_keypoints False --length 30 --stride 30 --fill_gap 20 --dlc_likelihood_threshold 0.1

DISK-train --project_path DISK_test2_multianimal_dlc_h5 --dataset_name test2 --indep_keypoints True --merge_keypoints False --training_epochs 8

DISK-impute --project_path DISK_test2_multianimal_dlc_h5 --dataset_name test2 --model_name DISK_test2