import pandas as pd
from DISK.utils.transforms import AddMissing_LengthProba
import DISK.utils.transforms
import logging
import numpy as np
from approvaltests.approvals import verify
from approvaltests.reporters import PythonNativeReporter
## for f in ls /home/france/Documents/DISK/tests/approval_tests/*received.txt; do g=${f%%.received.txt}; mv $f $g.approved.txt; done

def test_add_missing_class():
    np.random.seed(0)
    length_proba_df = pd.read_csv('/home/france/Documents/DISK_DLC_H5/DISK_data/test_dlc_h5/proba_missing_length_set_keypoints.csv')
    proba_file = '/home/france/Documents/DISK_DLC_H5/DISK_data/test_dlc_h5/proba_missing_set_keypoints.csv'

    length_proba_df['length'] = length_proba_df['length'].astype('int')
    length_proba_df['keypoint'] = length_proba_df['keypoint'].astype('str')
    init_proba_df = pd.read_csv(proba_file, dtype={'keypoint': str})

    logger = logging.getLogger()
    list_keypoints = ['left_ear', 'nose', 'right_ear', 'spine1', 'spine2', 'spine3', 'spine4', 'tail']
    indep_keypoints = False
    pad = (1, 1)
    proba = 1
    divider = 2
    verbose = 0
    outputdir = ''

    addmissing_instance = AddMissing_LengthProba(length_proba_df, init_proba_df, list_keypoints, logger=logger,
                                                 indep_keypoints=indep_keypoints, pad=pad, proba=proba, divider=divider,
                                                 verbose=verbose, outputdir=outputdir)

    length = 120

    x = np.random.random((length, len(list_keypoints), divider))
    x_prime = addmissing_instance(x, verbose_sample=0)
    verify(x_prime, reporter=PythonNativeReporter())


def test_add_missing_class_uniform():
    np.random.seed(0)
    length_proba_df = pd.read_csv('/home/france/Documents/DISK_human_mocap/DISK_data/dataset_length20_stride10'
                                   '/proba_missing_length_uniform.csv')
    proba_file = '/home/france/Documents/DISK_human_mocap/DISK_data/dataset_length20_stride10/proba_missing_uniform.csv'

    length_proba_df['length'] = length_proba_df['length'].astype('int')
    length_proba_df['keypoint'] = length_proba_df['keypoint'].astype('str')
    init_proba_df = pd.read_csv(proba_file, dtype={'keypoint': str})

    logger = logging.getLogger()
    list_keypoints = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19']
    indep_keypoints = True
    pad = (1, 1)
    proba = 1
    divider = 3
    verbose = 0
    outputdir = ''

    addmissing_instance = AddMissing_LengthProba(length_proba_df, init_proba_df, list_keypoints, logger=logger,
                                                 indep_keypoints=indep_keypoints, pad=pad, proba=proba, divider=divider,
                                                 verbose=verbose, outputdir=outputdir)

    length = 20

    x = np.random.random((length, len(list_keypoints), divider))
    x_prime = addmissing_instance(x, verbose_sample=0)
    verify(x_prime, reporter=PythonNativeReporter())

def test_add_missing_class_nan():
    np.random.seed(0)
    length_proba_df = pd.read_csv('/home/france/Documents/DISK_human_mocap/DISK_data/dataset_length20_stride10/proba_missing_length_uniform.csv')
    proba_file = '/home/france/Documents/DISK_human_mocap/DISK_data/dataset_length20_stride10/proba_missing_uniform.csv'

    length_proba_df['length'] = length_proba_df['length'].astype('int')
    length_proba_df['keypoint'] = length_proba_df['keypoint'].astype('str')
    init_proba_df = pd.read_csv(proba_file, dtype={'keypoint': str})

    logger = logging.getLogger()
    list_keypoints = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19']
    indep_keypoints = True
    pad = (1, 1)
    proba = 1
    divider = 3
    verbose = 0
    outputdir = ''

    addmissing_instance = AddMissing_LengthProba(length_proba_df, init_proba_df, list_keypoints, logger=logger,
                                                 indep_keypoints=indep_keypoints, pad=pad, proba=proba, divider=divider,
                                                 verbose=verbose, outputdir=outputdir)

    length = 20

    x = np.random.random((length, len(list_keypoints), divider))
    x[0,0,0] = np.nan
    x_prime = addmissing_instance(x, verbose_sample=True)
    verify(x_prime, reporter=PythonNativeReporter())

def test_add_missing_class_long(monkeypatch):
    np.random.seed(0)
    length_proba_df = pd.read_csv('/home/france/Documents/DISK_human_mocap/DISK_data/dataset_length20_stride10/proba_missing_length_uniform.csv')
    proba_file = '/home/france/Documents/DISK_human_mocap/DISK_data/dataset_length20_stride10/proba_missing_uniform.csv'

    length_proba_df['length'] = length_proba_df['length'].astype('int')
    length_proba_df['keypoint'] = length_proba_df['keypoint'].astype('str')
    init_proba_df = pd.read_csv(proba_file, dtype={'keypoint': str})

    logger = logging.getLogger()
    list_keypoints = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19']
    indep_keypoints = True
    pad = (1, 1)
    proba = 1
    divider = 3
    verbose = 0
    outputdir = ''

    init_proba_df.loc[:, 'proba'] = 0
    init_proba_df.loc[init_proba_df['keypoint'] == '00', 'proba'] = 1

    addmissing_instance = AddMissing_LengthProba(length_proba_df, init_proba_df, list_keypoints, logger=logger,
                                                 indep_keypoints=indep_keypoints, pad=pad, proba=proba, divider=divider,
                                                 verbose=verbose, outputdir=outputdir)

    length = 20

    x = np.random.random((length, len(list_keypoints), divider))

    x_prime = addmissing_instance(x, verbose_sample=0)
    verify(x_prime, reporter=PythonNativeReporter())