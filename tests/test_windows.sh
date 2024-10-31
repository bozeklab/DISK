mkdir datasets
mkdir models

conda create --name env_impute python=3.9 -y
conda activate env_impute
git clone https://github.com/bozeklab/DISK.git DISK
cd DISK
pip install -r DISK/requirements.txt -e . --quiet
pip install gdown
cd ..

cp DISK\notebooks\*.yaml DISK\DISK\conf\

python DISK\DISK\main_fillmissing.py training.n_cpus=0
python DISK\DISK\main_fillmissing.py training.n_cpus=1
python DISK\DISK\main_fillmissing.py training.n_cpus=2
