#!/usr/bin/env python

from setuptools import setup, Extension, find_packages
import os
import sys

setup(
    name='DISK',
    version='0.1',
    description="Deep Imputation of SKeleton data",
    long_description="Enter your project description here (long)",
    author='France ROSE',
    author_email="france.rose@wanadoo.fr",
    packages=find_packages("."),
    url="NONE",
    install_requires=['numpy>=1.21,<1.24',
                      'matplotlib>=3.1,<3.8',
                      'tqdm==4.59.0',
                      'plotly==5.5.0',
                      'h5py==3.7.0',
                      'seaborn==0.12.2',
                      'imageio>=2.3,<2.10',
                      'scikit-image>=0.18.1,<0.20',
                      'hydra-core==1.2.0',
                      'pandas==1.4.4',
                      'scikit-learn>=0.24.1',
                      'scipy>=1.10,<1.13',
                      'einops==0.6.1',
                      'umap-learn>=0.5'],
    extras_require={'gpu': ['torch>=1.9.1,<1.13',]
                      },
    package_data={'DISK': ['resources/*']},
    entry_points={'console_scripts': [
        "DISK = DISK.launchers.DISK_launcher:main",
    ]},
)
