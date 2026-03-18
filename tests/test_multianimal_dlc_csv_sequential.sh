#!/bin/bash
set -e

#DISK-create-project --project_path DISK_multianimal_dlc_csv --file_format dlc_csv --data_files [/home/france/mount_cvg/behavior_data/calms21-disk-dlc/calms21-01.01-snapshot195-dlc-csv/mouse001_task1_annotator1DLC_DlcrnetStride16Ms5_calms21Jan1shuffle1_snapshot_best-195_el_filtered.csv,/home/france/mount_cvg/behavior_data/calms21-disk-dlc/calms21-01.01-snapshot195-dlc-csv/mouse002_task1_annotator1DLC_DlcrnetStride16Ms5_calms21Jan1shuffle1_snapshot_best-195_el_filtered.csv]

DISK-prepare-data --project_path DISK_multianimal_dlc_csv --original_freq 60 --subsampling_freq 60 --length 30 --stride 30 --fill_gap 10 --drop_keypoints [] --sequential true  --dlc_likelihood_threshold 0.01

DISK-train --network gru --project_path DISK_multianimal_dlc_csv --dataset_name /home/france/Documents/DISK_multianimal_dlc_csv/DISK_data/dataset_60Hz_30length_30stride_sequential --training_epochs 4 --n_cpus 6 --print_every 1 --model_name GRU_test

DISK-impute --project_path DISK_multianimal_dlc_csv --dataset_name /home/france/Documents/DISK_multianimal_dlc_csv/DISK_data/dataset_60Hz_30length_30stride_sequential  --model_name GRU_test