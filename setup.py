#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
from setuptools import setup

REQUIRES = [
        'numpy',
        'scipy',
        'emcee',
        'corner',
        'lmfit',
        'seaborn',
        'pandas',
        'tqdm',
        'clint'
]

readme = open('README.md').read()

setup(
    name='epytox',
    version='0.1.0',
    description='A python toolkit for environmental toxicology model model fitting and simulations.',
    author='Raymond Nepstad',
    author_email='raymond.nepstad+epytox@gmail.com',
    url='https://github.com/nepstad/epytox',
    packages=[
        'epytox',
    ],
    package_dir={'epytox': 'epytox'},
    include_package_data=True,
    install_requires=[
    ],
    license=open('LICENSE').read(),
    zip_safe=False,
    keywords='epytox',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.5',
    ],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
)
