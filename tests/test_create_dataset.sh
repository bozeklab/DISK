#!/bin/bash
set -e

cd /projects/ag-bozek/france/results_behavior
conda activate env_cuda11.3

yes n | python /home/frose1/DISK/DISK/create_dataset.py dataset_name=test_fl2 original_freq=300 subsampling_freq=60 length=60 stride=30 discard_beginning=0 discard_end=0 fill_gap=0 \
drop_keypoints=[] sequential=false file_type=mat_qualisys \
input_files=[/projects/ag-bozek/france/behavior_data/bogna_data/INH1A_open_field_proc/INH1A_S1_M1_MC6_FL2_17_04_2019_proc_bij_6_08_19_A.mat,\
/projects/ag-bozek/france/behavior_data/bogna_data/INH1A_open_field_proc/INH1A_S2_M2_MC6_FL2_17_04_2019_proc-bij_6_08_19_C.mat,\
/projects/ag-bozek/france/behavior_data/bogna_data/INH1A_open_field_proc/INH1A_S3_M3_MC6_FL2_17_04_2019_proc_bij_7_08_19_B.mat,\
/projects/ag-bozek/france/behavior_data/bogna_data/INH1A_open_field_proc/INH1A_S4_M4_MC7_FL2_17_04_2019_proc_bij_7_08_19_A.mat,\
/projects/ag-bozek/france/behavior_data/bogna_data/INH1A_open_field_proc/INH1A_S5_M5_MC7_FL2_18_04_2019_proc_bij_6_08_19_C.mat,\
/projects/ag-bozek/france/behavior_data/bogna_data/INH1A_open_field_proc/INH1A_S6_M6_MC7_FL2_18_04_2019_proc_bij_8_08_19_B.mat,\
/projects/ag-bozek/france/behavior_data/bogna_data/INH1A_open_field_proc/INH1A_S7_M7_MC8_FL2_18_04_2019_proc_bij_8_08_19_A.mat,\
/projects/ag-bozek/france/behavior_data/bogna_data/INH1A_open_field_proc/INH1A_S8_M8_MC8_FL2_18_04_2019_proc_bij_8_08_19_C.mat,\
/projects/ag-bozek/france/behavior_data/bogna_data/INH1A_open_field_proc/INH1A_S9_M9_MC8_FL2_18_04_2019_proc_bij_8_08_19_B.mat,\
/projects/ag-bozek/france/behavior_data/bogna_data/INH1A_open_field_proc/INH1A_S10_M10_MC8_FL2_18_04_2019_proc_nij_8_08_19_C.mat]

python create_proba_missing_files.py dataset_name=test_fl2 indep_keypoints=True merge_keypoints=False
python create_proba_missing_files.py dataset_name=test_fl2 indep_keypoints=False merge_keypoints=False
python create_proba_missing_files.py dataset_name=test_fl2 indep_keypoints=False merge_keypoints=True

yes n | python /home/frose1/DISK/DISK/create_dataset.py dataset_name=test_fl2 original_freq=300 subsampling_freq=60 length=60 stride=30 discard_beginning=5 discard_end=5 fill_gap=10 \
drop_keypoints=[] sequential=false file_type=mat_qualisys \
input_files=[/projects/ag-bozek/france/behavior_data/bogna_data/INH1A_open_field_proc/INH1A_S1_M1_MC6_FL2_17_04_2019_proc_bij_6_08_19_A.mat,\
/projects/ag-bozek/france/behavior_data/bogna_data/INH1A_open_field_proc/INH1A_S2_M2_MC6_FL2_17_04_2019_proc-bij_6_08_19_C.mat,\
/projects/ag-bozek/france/behavior_data/bogna_data/INH1A_open_field_proc/INH1A_S3_M3_MC6_FL2_17_04_2019_proc_bij_7_08_19_B.mat,\
/projects/ag-bozek/france/behavior_data/bogna_data/INH1A_open_field_proc/INH1A_S4_M4_MC7_FL2_17_04_2019_proc_bij_7_08_19_A.mat,\
/projects/ag-bozek/france/behavior_data/bogna_data/INH1A_open_field_proc/INH1A_S5_M5_MC7_FL2_18_04_2019_proc_bij_6_08_19_C.mat,\
/projects/ag-bozek/france/behavior_data/bogna_data/INH1A_open_field_proc/INH1A_S6_M6_MC7_FL2_18_04_2019_proc_bij_8_08_19_B.mat,\
/projects/ag-bozek/france/behavior_data/bogna_data/INH1A_open_field_proc/INH1A_S7_M7_MC8_FL2_18_04_2019_proc_bij_8_08_19_A.mat,\
/projects/ag-bozek/france/behavior_data/bogna_data/INH1A_open_field_proc/INH1A_S8_M8_MC8_FL2_18_04_2019_proc_bij_8_08_19_C.mat,\
/projects/ag-bozek/france/behavior_data/bogna_data/INH1A_open_field_proc/INH1A_S9_M9_MC8_FL2_18_04_2019_proc_bij_8_08_19_B.mat,\
/projects/ag-bozek/france/behavior_data/bogna_data/INH1A_open_field_proc/INH1A_S10_M10_MC8_FL2_18_04_2019_proc_nij_8_08_19_C.mat]
