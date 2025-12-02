#!/bin/bash
set -e

cd /projects/ag-bozek/france/results_behavior
source activate env_cuda11.3 || conda activate env_cuda11.3

yes n | python /home/frose1/DISK/DISK/create_dataset.py dataset_name=test_simple_csv original_freq=60 subsampling_freq=60 length=30 stride=30 discard_beginning=0 discard_end=-1 fill_gap=10 drop_keypoints=[] sequential=true file_type=simple_csv dlc_likelihood_threshold=0.01 input_files=[/projects/ag-bozek/france/behavior_data/fish_data/fish_fighting_interpolated_head.csv]
yes n | python /home/frose1/DISK/DISK/create_dataset.py dataset_name=test_simple_csv_2D original_freq=60 subsampling_freq=60 length=30 stride=30 discard_beginning=0 discard_end=-1 fill_gap=10 drop_keypoints=[] sequential=true file_type=simple_csv dlc_likelihood_threshold=0.01 input_files=[/projects/ag-bozek/france/behavior_data/fish_data/fish_fighting_interpolated_head_2D.csv]
