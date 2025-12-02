#!/bin/bash
set -e

cd /projects/ag-bozek/france/results_behavior
source activate env_cuda11.3 || conda activate env_cuda11.3
yes n | python /home/frose1/DISK/DISK/create_dataset.py dataset_name=test_dlc_h5 original_freq=60 subsampling_freq=60 length=10 stride=10 discard_beginning=0 discard_end=-1 fill_gap=10 drop_keypoints=[0] sequential=false file_type=dlc_h5 dlc_likelihood_threshold=0.8 input_files=[/projects/ag-bozek/france/results_behavior/dlc_project/videos/21_11_8_one_mouse.top.irDLC_resnet50_moseq_exampleAug21shuffle1_500000.h5,/projects/ag-bozek/france/results_behavior/dlc_project/videos/21_12_2_def6a_1.top.irDLC_resnet50_moseq_exampleAug21shuffle1_500000.h5,/projects/ag-bozek/france/results_behavior/dlc_project/videos/22_04_26_cage4_0_2.top.irDLC_resnet50_moseq_exampleAug21shuffle1_500000.h5]
python /home/frose1/DISK/DISK/create_proba_missing_files.py dataset_name=test_dlc_h5 indep_keypoints=True merge_keypoints=False
python /home/frose1/DISK/DISK/main_fillmissing.py network=gru hydra.run.dir=outputs/gru_test_dlc_h5 dataset.name=test_dlc_h5 training.epochs=10 training.print_every=2 training.n_cpus=6 feed_data.transforms.add_missing.indep_keypoints=true feed_data.transforms.add_missing.files=[test_dlc_h5/proba_missing.csv,test_dlc_h5/proba_missing_length.csv]
python /home/frose1/DISK/DISK/test_fillmissing.py hydra.run.dir=outputs/gru_test_dlc_h5/test dataset.name=test_dlc_h5 dataset.stride=10 dataset.skeleton_file=null feed_data.transforms.add_missing.files=[test_dlc_h5/proba_missing.csv,test_dlc_h5/proba_missing_length.csv] evaluate.checkpoints=[outputs/gru_test_dlc_h5] evaluate.n_repeat=1 evaluate.n_plots=2
python /home/frose1/DISK/DISK/impute_dataset.py hydra.run.dir=outputs/gru_test_dlc_h5/impute dataset.name=test_dlc_h5 dataset.skeleton_file=null evaluate.checkpoint=outputs/gru_test_dlc_h5 evaluate.n_plots=2 evaluate.path_to_original_files=dlc_project/videos
