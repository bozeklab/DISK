#!/bin/bash
set -e

DISK-create-project --project_path DISK_FL2 --file_format mat_qualisys --data_files /home/france/Mounted_dir/behavior_data/bogna_data/INH1A_open_field_proc/INH1A_S1_M1_MC6_FL2_17_04_2019_proc_bij_6_08_19_A.mat \
/home/france/Mounted_dir/behavior_data/bogna_data/INH1A_open_field_proc/INH1A_S2_M2_MC6_FL2_17_04_2019_proc-bij_6_08_19_C.mat \
/home/france/Mounted_dir/behavior_data/bogna_data/INH1A_open_field_proc/INH1A_S3_M3_MC6_FL2_17_04_2019_proc_bij_7_08_19_B.mat \
/home/france/Mounted_dir/behavior_data/bogna_data/INH1A_open_field_proc/INH1A_S4_M4_MC7_FL2_17_04_2019_proc_bij_7_08_19_A.mat \
/home/france/Mounted_dir/behavior_data/bogna_data/INH1A_open_field_proc/INH1A_S5_M5_MC7_FL2_18_04_2019_proc_bij_6_08_19_C.mat \
/home/france/Mounted_dir/behavior_data/bogna_data/INH1A_open_field_proc/INH1A_S6_M6_MC7_FL2_18_04_2019_proc_bij_8_08_19_B.mat \
/home/france/Mounted_dir/behavior_data/bogna_data/INH1A_open_field_proc/INH1A_S7_M7_MC8_FL2_18_04_2019_proc_bij_8_08_19_A.mat \
/home/france/Mounted_dir/behavior_data/bogna_data/INH1A_open_field_proc/INH1A_S8_M8_MC8_FL2_18_04_2019_proc_bij_8_08_19_C.mat \
/home/france/Mounted_dir/behavior_data/bogna_data/INH1A_open_field_proc/INH1A_S9_M9_MC8_FL2_18_04_2019_proc_bij_8_08_19_B.mat \
/home/france/Mounted_dir/behavior_data/bogna_data/INH1A_open_field_proc/INH1A_S10_M10_MC8_FL2_18_04_2019_proc_nij_8_08_19_C.mat

DISK-prepare-data --project_path /home/france/Documents/DISK_FL2 --original_freq 300 --subsampling_freq 60 --length 60 --discard_beginning 5 --discard_end 5 --fill_gap 0 --drop_keypoints --sequential false

DISK-prepare-data --project_path /home/france/Documents/DISK_FL2 --original_freq 300 --subsampling_freq 60 --length 60 --discard_beginning 5 --discard_end 5 --fill_gap 0 --drop_keypoints --sequential false --indep_keypoints True --merge_keypoints False

DISK-prepare-data --project_path /home/france/Documents/DISK_FL2 --original_freq 300 --subsampling_freq 60 --length 60 --discard_beginning 5 --discard_end 5 --fill_gap 0 --drop_keypoints --sequential false --indep_keypoints False --merge_keypoints False

DISK-prepare-data --project_path /home/france/Documents/DISK_FL2 --original_freq 300 --subsampling_freq 60 --length 60 --discard_beginning 5 --discard_end 5 --fill_gap 0 --drop_keypoints --sequential false --indep_keypoints False --merge_keypoints True

DISK-prepare-data --project_path /home/france/Documents/DISK_FL2 --original_freq 300 --subsampling_freq 60 --length 60 --discard_beginning 5 --discard_end 5 --fill_gap 0 --drop_keypoints --sequential false --indep_keypoints True --merge_keypoints True

