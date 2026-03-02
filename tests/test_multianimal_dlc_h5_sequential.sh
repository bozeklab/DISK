#!/bin/bash
set -e

DISK-create-project project_name=DISK_multianimal_dlc_h5 file_type=dlc_h5 data_files=[/home/france/mount_cvg/behavior_data/calms21-disk-dlc/calms21-01.01-snapshot195-dlc-h5/mouse001_task1_annotator1DLC_DlcrnetStride16Ms5_calms21Jan1shuffle1_snapshot_best-195_el_filtered.h5,/home/france/mount_cvg/behavior_data/calms21-disk-dlc/calms21-01.01-snapshot195-dlc-h5/mouse002_task1_annotator1DLC_DlcrnetStride16Ms5_calms21Jan1shuffle1_snapshot_best-195_el_filtered.h5] working_directory='.'

DISK-prepare-data project_path=DISK_multianimal_dlc_h5 original_freq=30 subsampling_freq=30 length=30 stride=30 fill_gap=10 sequential=true dlc_likelihood_threshold=0.01 dataset_name=dataset1

DISK-train network=gru project_path=DISK_multianimal_dlc_h5 dataset_name=dataset1 training_epochs=4 indep_keypoints=true

DISK-impute project_path=DISK_multianimal_dlc_h5 dataset_name=dataset1 model_name=dataset1_DISK-GRU