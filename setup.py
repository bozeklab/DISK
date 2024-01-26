#!/usr/bin/env python

from setuptools import setup, Extension, find_packages
import os
import sys

setup(
    name='ImputeSkeleton',
    version='0.1',
    description="Enter your project description here (short)",
    long_description="Enter your project description here (long)",
    author='NONE',
    author_email="NONE",
    packages=find_packages("."),
    url="NONE",
    install_requires=[],
    package_data={'ImputeSkeleton': ['resources/*']},
    entry_points={'console_scripts': [
        "ImputeSkeleton = ImputeSkeleton.launchers.ImputeSkeleton_launcher:main",
    ]},
)
