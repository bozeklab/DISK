#!/bin/bash
set -e

DISK-create-project --project_path DISK_multianimal_dlc_csv --file_format dlc_csv --data_files /home/france/mount_cvg/behavior_data/calms21-disk-dlc/calms21-01.01-snapshot195-dlc-csv/

DISK-prepare-data --project_path DISK_multianimal_dlc_csv --dlc_likelihood_threshold 0.01 --sequential false --original_freq 60 --subsampling_freq 60 --length 30 --stride 30 --discard_beginning 0 --discard_end -1 --fill_gap 10 --drop_keypoints

DISK-train --project_path DISK_multianimal_dlc_csv --dataset_name dataset_60Hz_30length_30stride --training_epochs 4 --n_cpus 6

DISK-train --project_path DISK_multianimal_dlc_csv --dataset_name dataset_60Hz_30length_30stride --training_epochs 4 --n_cpus 6 --indep_keypoints false --merge_keypoints false

DISK-train --project_path DISK_multianimal_dlc_csv --dataset_name dataset_60Hz_30length_30stride --training_epochs 4 --n_cpus 6 --indep_keypoints false --merge_keypoints true

DISK-test --project_path DISK_multianimal_dlc_csv --dataset_name dataset_60Hz_30length_30stride --indep_keypoints false --merge_keypoints false --model_name_list DISK_dataset_60Hz_30length_30stride DISK_dataset_60Hz_30length_30stride_1

DISK-impute --project_path DISK_multianimal_dlc_csv --dataset_name dataset_60Hz_30length_30stride --model_name DISK_dataset_60Hz_30length_30stride