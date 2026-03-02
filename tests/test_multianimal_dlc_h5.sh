#!/bin/bash
set -e

DISK-create-project project_name=DISK_test2_multianimal_dlc_h5 file_type=dlc_h5 data_files=[/home/france/mount_cvg/behavior_data/calms21-disk-dlc/calms21-01.01-snapshot195-dlc-h5/mouse001_task1_annotator1DLC_DlcrnetStride16Ms5_calms21Jan1shuffle1_snapshot_best-195_el_filtered.h5,/home/france/mount_cvg/behavior_data/calms21-disk-dlc/calms21-01.01-snapshot195-dlc-h5/mouse002_task1_annotator1DLC_DlcrnetStride16Ms5_calms21Jan1shuffle1_snapshot_best-195_el_filtered.h5,/home/france/mount_cvg/behavior_data/calms21-disk-dlc/calms21-01.01-snapshot195-dlc-h5/mouse004_task1_annotator1DLC_DlcrnetStride16Ms5_calms21Jan1shuffle1_snapshot_best-195_el_filtered.h5] working_directory=.

DISK-prepare-data project_path=DISK_test2_multianimal_dlc_h5 dataset_name=test2 indep_keypoints=True merge_keypoints=False length=30 stride=30 fill_gap=20 dlc_likelihood_threshold=0.1

DISK-train project_path=DISK_test2_multianimal_dlc_h5 dataset_name=test2 indep_keypoints=True merge_keypoints=False training_epochs=8

DISK-impute project_path=DISK_test2_multianimal_dlc_h5 dataset_name=test2 model_name=test2_DISK