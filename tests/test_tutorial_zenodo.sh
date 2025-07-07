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
cd ..

cd datasets
wget --header 'Host: zenodo.org' --user-agent 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:136.0) Gecko/20100101 Firefox/136.0' --header 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8' --header 'Accept-Language: en-US,en;q=0.5' --referer 'https://zenodo.org/records/15828939' --header 'Cookie: cookie_consent=essential; session=a50d7ee07c8ac6d6_6866883c.qzB_SvSWBo17flvWfCWgcfs9ISw; 5569e5a730cade8ff2b54f1e815f3670=a9c91475048ef11a7f0ba86fd6fb1964' --header 'Upgrade-Insecure-Requests: 1' --header 'Sec-Fetch-Dest: document' --header 'Sec-Fetch-Mode: navigate' --header 'Sec-Fetch-Site: same-origin' --header 'Sec-Fetch-User: ?1' 'https://zenodo.org/records/15828939/files/Human_DISK_dataset.zip?download=1' --output-document 'Human_DISK_dataset.zip'
yes N | unzip Human_DISK_dataset.zip
rm Human_DISK_dataset.zip
cd ../

cd models
wget --header 'Host: zenodo.org' --user-agent 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:136.0) Gecko/20100101 Firefox/136.0' --header 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8' --header 'Accept-Language: en-US,en;q=0.5' --referer 'https://zenodo.org/records/15828939' --header 'Cookie: cookie_consent=essential; session=a50d7ee07c8ac6d6_6866883c.qzB_SvSWBo17flvWfCWgcfs9ISw; 5569e5a730cade8ff2b54f1e815f3670=a9c91475048ef11a7f0ba86fd6fb1964' --header 'Upgrade-Insecure-Requests: 1' --header 'Sec-Fetch-Dest: document' --header 'Sec-Fetch-Mode: navigate' --header 'Sec-Fetch-Site: same-origin' --header 'Sec-Fetch-User: ?1' 'https://zenodo.org/records/15828939/files/Human_transformer_proba.zip?download=1' --output-document 'Human_transformer_proba.zip'
yes N | unzip Human_transformer_proba.zip
rm Human_transformer_proba.zip
wget --header 'Host: zenodo.org' --user-agent 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:136.0) Gecko/20100101 Firefox/136.0' --header 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8' --header 'Accept-Language: en-US,en;q=0.5' --referer 'https://zenodo.org/records/15828939' --header 'Cookie: cookie_consent=essential; session=a50d7ee07c8ac6d6_6866883c.qzB_SvSWBo17flvWfCWgcfs9ISw; 5569e5a730cade8ff2b54f1e815f3670=a9c91475048ef11a7f0ba86fd6fb1964' --header 'Upgrade-Insecure-Requests: 1' --header 'Sec-Fetch-Dest: document' --header 'Sec-Fetch-Mode: navigate' --header 'Sec-Fetch-Site: same-origin' --header 'Sec-Fetch-User: ?1' 'https://zenodo.org/records/15828939/files/Human_GRU.zip?download=1' --output-document 'Human_GRU.zip'
yes N | unzip Human_GRU.zip
rm Human_GRU.zip
cd ..

cd datasets
wget --header 'Host: zenodo.org' --user-agent 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:136.0) Gecko/20100101 Firefox/136.0' --header 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8' --header 'Accept-Language: en-US,en;q=0.5' --referer 'https://zenodo.org/records/15828939' --header 'Cookie: cookie_consent=essential; session=a50d7ee07c8ac6d6_6866883c.qzB_SvSWBo17flvWfCWgcfs9ISw; 5569e5a730cade8ff2b54f1e815f3670=a9c91475048ef11a7f0ba86fd6fb1964' --header 'Upgrade-Insecure-Requests: 1' --header 'Sec-Fetch-Dest: document' --header 'Sec-Fetch-Mode: navigate' --header 'Sec-Fetch-Site: same-origin' --header 'Sec-Fetch-User: ?1' 'https://zenodo.org/records/15828939/files/Rat7M_seq_DISK_dataset.zip?download=1' --output-document 'Rat7M_seq_DISK_dataset.zip'
yes N | unzip Rat7M_seq_DISK_dataset.zip
rm Rat7M_seq_DISK_dataset.zip
wget --header 'Host: zenodo.org' --user-agent 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:136.0) Gecko/20100101 Firefox/136.0' --header 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8' --header 'Accept-Language: en-US,en;q=0.5' --referer 'https://zenodo.org/records/15828939' --header 'Cookie: cookie_consent=essential; session=a50d7ee07c8ac6d6_6866883c.qzB_SvSWBo17flvWfCWgcfs9ISw; 5569e5a730cade8ff2b54f1e815f3670=a9c91475048ef11a7f0ba86fd6fb1964' --header 'Upgrade-Insecure-Requests: 1' --header 'Sec-Fetch-Dest: document' --header 'Sec-Fetch-Mode: navigate' --header 'Sec-Fetch-Site: same-origin' --header 'Sec-Fetch-User: ?1' 'https://zenodo.org/records/15828939/files/rat7M_raw_data.zip?download=1' --output-document 'rat7M_raw_data.zip'
yes N | unzip rat7M_raw_data.zip
rm rat7M_raw_data.zip
cd ..

cd models
wget --header 'Host: zenodo.org' --user-agent 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:136.0) Gecko/20100101 Firefox/136.0' --header 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8' --header 'Accept-Language: en-US,en;q=0.5' --referer 'https://zenodo.org/records/15828939' --header 'Cookie: cookie_consent=essential; session=a50d7ee07c8ac6d6_6866883c.qzB_SvSWBo17flvWfCWgcfs9ISw; 5569e5a730cade8ff2b54f1e815f3670=a9c91475048ef11a7f0ba86fd6fb1964' --header 'Upgrade-Insecure-Requests: 1' --header 'Sec-Fetch-Dest: document' --header 'Sec-Fetch-Mode: navigate' --header 'Sec-Fetch-Site: same-origin' --header 'Sec-Fetch-User: ?1' 'https://zenodo.org/records/15828939/files/Rat7M_transformer_proba.zip?download=1' --output-document 'Rat7M_transformer_proba.zip'
yes N | unzip Rat7M_transformer_proba.zip
rm Rat7M_transformer_proba.zip
cd ..

python3 cloned-DISK-repo/tests/import_test.py

cp -Rf cloned-DISK-repo/notebooks/*.yaml cloned-DISK-repo/DISK/conf/

python3 cloned-DISK-repo/DISK/main_fillmissing.py

python3 cloned-DISK-repo/DISK/test_fillmissing.py

# does not work on my local computer
python3 cloned-DISK-repo/DISK/embedding_umap.py --batch_size 1 --checkpoint_folder models/03-10-24_transformer_NLL --stride 60 --dataset_path .

# rm -rf datasets/Mocap_keypoints_60_stride30_new


python3 cloned-DISK-repo/DISK/impute_dataset.py

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
