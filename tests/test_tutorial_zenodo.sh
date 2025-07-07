#!/bin/bash
set +e

if test -d datasets
then
  echo "Directory datasets exists"
else
  mkdir datasets
fi

if test -d models
then
    echo "Directory models exists"
else
  mkdir models
fi

conda create --name env_impute python=3.9 -y

#{
source activate env_impute || conda activate env_impute
git clone https://github.com/bozeklab/DISK.git cloned-DISK-repo
cd cloned-DISK-repo
pip install -r DISK/requirements.txt -e . --quiet
pip install gdown
cd ..

cd datasets
wget -r "https://zenodo.org/records/15800035/files/Human_DISK_dataset.zip?download=1"
yes N | unzip Human_DISK_dataset.zip
rm Human_DISK_dataset.zip
cd ../

cd models
wget -r "https://zenodo.org/records/15800035/files/Human_transformer_proba.zip?download=1"
yes N | unzip Human_transformer_proba.zip
rm Human_transformer_proba.zip
wget -r "https://zenodo.org/records/15800035/files/Human_GRU.zip?download=1"
yes N | unzip Human_GRU.zip
rm Human_GRU.zip
cd ..

cd datasets
wget -r "https://zenodo.org/records/15800035/files/rat7M_raw_data.zip?download=1"
yes N | unzip Rat7M_seq_DISK_dataset.zip
rm Rat7M_seq_DISK_dataset.zip
wget -r "https://zenodo.org/records/15800035/files/rat7M_raw_data.zip?download=1"
yes N | unzip rat7M_raw_data.zip
rm rat7M_raw_data.zip
cd ..

cd models
wget -r "https://zenodo.org/records/15800035/files/Rat7M_transformer_proba.zip?download=1"
yes N | unzip Rat7M_transformer_proba.zip
cd ..

python cloned-DISK-repo/tests/import_test.py

cp -Rf cloned-DISK-repo/notebooks/*.yaml cloned-DISK-repo/DISK/conf/

python cloned-DISK-repo/DISK/main_fillmissing.py

python cloned-DISK-repo/DISK/test_fillmissing.py

# does not work on my local computer
python cloned-DISK-repo/DISK/embedding_umap.py --batch_size 1 --checkpoint_folder models/03-10-24_transformer_NLL --stride 60 --dataset_path .

# rm -rf datasets/Mocap_keypoints_60_stride30_new


python cloned-DISK-repo/DISK/impute_dataset.py

conda activate base
conda remove --name env_impute --all -y
conda clean --all -y
#} || {
#  conda remove --name env_impute --all -y
#  conda clean --all -y
#}

rm -rf datasets
rm -rf models
rm -rf cloned-DISK-repo
