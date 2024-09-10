#!/bin/bash
set -e

conda create --name env_impute python=3.9 -y

{
  conda activate env_impute || source activate env_impute
  git clone https://github.com/bozeklab/DISK.git cloned-DISK-repo
  cd cloned-DISK-repo
  pip install -r DISK/requirements.txt -e . --quiet
  cd ..

  python cloned-DISK-repo/tests/import_test.py

  if test -d datasets
  then
    echo "Directory datasets exists"
  else
    mkdir datasets
  fi

  cp cloned-DISK-repo/notebooks/*.yaml cloned-DISK-repo/DISK/conf/*

  cd datasets
  gdown https://drive.google.com/uc?id=1PXECUljc5qr8kz9H2LxT4LhS6P4uN4ck
  unzip Human_DISK_dataset.zip
  cd ../

  python cloned-DISK-repo/DISK/main_fillmissing.py


  if test -d models
  then
      echo "Directory models exists"
  else
    mkdir models
  fi

  cd models
  gdown https://drive.google.com/uc?id=1b8Px-lbTddOrMZW9dozJPVLjxh0PjnTp
  unzip Human_transformer_proba.zip
  gdown https://drive.google.com/uc?id=1tGL8eyafpwJS7wdNGB5o_tuABd8qOBHB
  unzip Human_GRU.zip
  cd ..

  python cloned-DISK-repo/DISK/test_fillmissing.py

  python cloned-DISK-repo/DISK/embedding_umap.py --batch_size 1 --checkpoint_folder models/03-10-24_transformer_NLL --stride 60 --dataset_path .

  cd datasets
  gdown https://drive.google.com/uc?id=1dpgBqqdwHWN4fcUzaeVt_Sq1wH-e4lhK
  unzip Rat7M_seq_DISK_dataset.zip
  gdown https://drive.google.com/uc?id=1t_tPwyzNCDK_YUJzzbAJwn3YbtHmDi_z
  unzip rat7M_raw_data.zip
  cd ..

  cd models
  gdown https://drive.google.com/uc?id=1hbEkwTI2ir0T54UywVv4r9GbzfgveCaC
  unzip Rat7M_transformer_proba.zip
  cd ..

  python cloned-DISK-repo/DISK/impute_dataset.py

  conda remove --name env_impute --all -y
  conda clean --all } || {
  conda remove --name env_impute --all -y
  conda clean --all -y
}