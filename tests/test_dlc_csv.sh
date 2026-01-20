#!/bin/bash
set -e

cd /projects/ag-bozek/france/results_behavior
source activate env_DISK || conda activate env_DISK
yes n | python /home/frose1/DISK/DISK/create_dataset.py dataset_name=test_dlc_csv original_freq=60 subsampling_freq=60 length=30 stride=30 discard_beginning=0 discard_end=-1 fill_gap=10 drop_keypoints=[] sequential=true file_type=dlc_csv dlc_likelihood_threshold=0.01 input_files=[/projects/ag-bozek/france/behavior_data/freewalking_20220811_WTTB_fly1_1DLC_resnet50_FreeWalkingMp4Jul30shuffle1_190000.csv]
python /home/frose1/DISK/DISK/create_proba_missing_files.py dataset_name=test_dlc_csv indep_keypoints=True merge_keypoints=False
python /home/frose1/DISK/DISK/main_fillmissing.py network=gru hydra.run.dir=outputs/gru_test_dlc_csv dataset.name=test_dlc_csv dataset.skeleton_file=null training.epochs=4 training.n_cpus=6 training.print_every=2 feed_data.transforms.add_missing.indep_keypoints=true feed_data.transforms.add_missing.files=[test_dlc_csv/proba_missing.csv,test_dlc_csv/proba_missing_length.csv]
python /home/frose1/DISK/DISK/test_fillmissing.py hydra.run.dir=outputs/gru_test_dlc_csv/test dataset.name=test_dlc_csv dataset.stride=10 dataset.skeleton_file=null feed_data.transforms.add_missing.files=[test_dlc_csv/proba_missing.csv,test_dlc_csv/proba_missing_length.csv] evaluate.checkpoints=[outputs/gru_test_dlc_csv] evaluate.n_repeat=1 evaluate.n_plots=2
python /home/frose1/DISK/DISK/impute_dataset.py hydra.run.dir=outputs/gru_test_dlc_csv/impute dataset.name=test_dlc_csv dataset.skeleton_file=null evaluate.checkpoint=outputs/gru_test_dlc_csv evaluate.n_plots=2 evaluate.path_to_original_files=../behavior_data