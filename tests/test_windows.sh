mkdir datasets
mkdir models

conda create --name env_DISK python=3.9 -y
conda activate env_DISK
git clone -b debug https://github.com/bozeklab/DISK.git DISK
cd DISK
pip install --quiet

DISK-create-project --project_path DISK_simple_csv_2D --file_format simple_csv --data_files DISK/tests/test_files/fish_fighting_interpolated_head_2D.csv

DISK-prepare-data --project_path DISK_simple_csv_2D --dataset_name test_simple_csv_2D --indep_keypoints True --merge_keypoints False --original_freq 60 --subsampling_freq 60 --length 30 --stride 30 --discard_beginning 0 --discard_end -1 --fill_gap 10 --drop_keypoints [] --sequential true

DISK-train --project_path DISK_simple_csv_2D --dataset_name test_simple_csv_2D --training_epochs 3 --n_cpus 0
DISK-train --project_path DISK_simple_csv_2D --dataset_name test_simple_csv_2D --training_epochs 3 --n_cpus 1
DISK-train --project_path DISK_simple_csv_2D --dataset_name test_simple_csv_2D --training_epochs 3 --n_cpus 2

