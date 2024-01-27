# Full tutorial with dataset "Mocap"

This dataset contains human motion capture data of different actions executed by human actors with 20 tracked keypoints.

## 1. Download original files

Download and unzip these files: https://drive.google.com/file/d/1-nwek-Iv1CveTvEF_84RW7U9oz200Etm/view?usp=sharing

## 2. Create the dataset

In the file `conf/conf_create_dataset.yaml`, change the following lines:

```yaml
dataset_name: Mocap
length: 60
stride: 30

original_freq: 12
subsampling_freq: 12 # no subsampling, subsampling_freq = original_freq

# no cropping at the beginning and end of the file
discard_beginning: 0
discard_end: -1 

skeleton: human

# no dropped keypoints:
drop_keypoints: # should stay blank

sequential: false # if false, some files left out for the validation and the test sets, 
                  # if true, the beginning of each file is put in the training set, then a portion to the validation set, then the end to the test set

input_files:
# path to the downloaded files
# path_to should be replaced by the correct path on your machine
# the list should be extended to include all the files
  - /path_to/mocap_dataset/mocap_3djoints/01_01.npy
  - /path_to/mocap_dataset/mocap_3djoints/01_02.npy
  - /path_to/mocap_dataset/mocap_3djoints/01_03.npy
  - /path_to/mocap_dataset/mocap_3djoints/01_04.npy
  - ...
```

Launch: `python DISK/create_dataset.py`

## 3. Use uniform probability to create missing holes

Indeed, this human dataset does not have any missing data.
