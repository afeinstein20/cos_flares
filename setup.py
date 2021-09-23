#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import os
import sys
from setuptools import setup

sys.path.insert(0, "cos_flares")
from version import __version__


long_description = \
    """
cos_flares is a tool to reduce Hubble/COS data and 
model flares found in said data.
"""


setup(
    name='cos_flares',
    version=__version__,
    license='MIT',
    author='Adina D. Feinstein',
    author_email='adina.d.feinstein@gmail.com',
    packages=[
        'cos_flares',
        ],
    include_package_data=True,
    url='http://github.com/afeinstein20/cos_flares',
    description='For analyzing flares in Hubble/COS data',
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_data={'': ['README.md', 'LICENSE']},
    install_requires=[
        'tqdm', 'astropy',
        'setuptools>=41.0.0', 'more-itertools',
        'matplotlib', 'numpy', 'scipy==1.4.1',
        'lightkurve>=1.9.0', 'calcos', 'costools'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.0',
        ],
    )
