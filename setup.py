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
    install_requires=[],
    package_data={'DISK': ['resources/*']},
    entry_points={'console_scripts': [
        "DISK = DISK.launchers.DISK_launcher:main",
    ]},
)
