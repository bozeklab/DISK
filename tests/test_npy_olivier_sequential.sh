#!/bin/bash
set -e

cd /projects/ag-bozek/france/results_behavior
source activate env_DISK || conda activate env_DISK
yes n | python /home/frose1/DISK/DISK/create_dataset.py dataset_name=test_olivier_npy original_freq=1 subsampling_freq=1 length=240 stride=30 discard_beginning=0 discard_end=-1 fill_gap=10 drop_keypoints=[] sequential=true file_type=npy dlc_likelihood_threshold=0.01 input_files=[/home/france/Dropbox/2025_UniBonn/behavior_data/DISK/female_2318.npy]
python /home/frose1/DISK/DISK/create_proba_missing_files.py dataset_name=test_olivier_npy indep_keypoints=False merge_keypoints=False
python /home/frose1/DISK/DISK/main_fillmissing.py network=transformer hydra.run.dir=outputs/transformer_test_olivier_npy dataset.name=test_olivier_npy dataset.skeleton_file=null training.epochs=1000 training.n_cpus=6 training.print_every=10 feed_data.transforms.add_missing.indep_keypoints=true feed_data.transforms.add_missing.files=[test_olivier_npy/proba_missing_set_keypoints.csv,test_olivier_npy/proba_missing_length_set_keypoints.csv]
python /home/frose1/DISK/DISK/test_fillmissing.py hydra.run.dir=outputs/transformer_test_olivier_npy/test dataset.name=test_olivier_npy dataset.stride=240 dataset.skeleton_file=null feed_data.transforms.add_missing.files=[test_olivier_npy/proba_missing_set_keypoints.csv,test_olivier_npy/proba_missing_length_set_keypoints.csv] evaluate.checkpoints=[outputs/transformer_test_olivier_npy] evaluate.n_repeat=1 evaluate.n_plots=10
python /home/frose1/DISK/DISK/impute_dataset.py hydra.run.dir=outputs/transformer_test_olivier_npy/impute dataset.name=test_olivier_npy dataset.skeleton_file=test_olivier_npy/skeleton.py evaluate.checkpoint=outputs/transformer_test_olivier_npy evaluate.save=true evaluate.save_dataset=true evaluate.n_plots=5 evaluate.path_to_original_files=/home/france/Dropbox/2025_UniBonn/behavior_data/DISK/

