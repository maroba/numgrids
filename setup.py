#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os.path
import re
from os.path import exists
from setuptools import setup, find_packages

author = 'Matthias Baer'
email = 'matthias.r.baer@googlemail.com'
description = 'Working with numerical grids made easy.'
name = 'numgrids'
year = '2023'
url = 'https://github.com/maroba/numgrids'

with open(os.path.join('numgrids', '__init__.py'), 'r') as fh:
    match = re.match('__version__ *= *"([^"]+)"', fh.readline())
    if match:
        version = match.group(1)
    else:
        raise ValueError('numgrids.__init__.py must define version string as first line.')

setup(
    name=name,
    author=author,
    author_email=email,
    url=url,
    version=version,
    packages=find_packages(),
    package_dir={name: name},
    include_package_data=True,
    license='MIT',
    description=description,
    long_description=open('docs/pypi_description.md').read() if exists('README.md') else '',
    long_description_content_type="text/markdown",
    install_requires=['numpy', 'scipy>=1.10.1', 'matplotlib', 'findiff'
                      ],
    python_requires=">=3.8",
    classifiers=['Operating System :: OS Independent',
                 'Programming Language :: Python :: 3',
                 ],
    platforms=['ALL'],
)
