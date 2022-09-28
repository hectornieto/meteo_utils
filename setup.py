#!/usr/bin/env python
#
# This file is part of SEN-ET project.
# Copyright 2018 Radoslaw Guzinski, Hector Nieto and contributors listed in the README.md file.

import os
from setuptools import setup

PROJECT_ROOT = os.path.dirname(__file__)


def read_file(filepath, root=PROJECT_ROOT):
    """
    Return the contents of the specified `filepath`.

    * `root` is the base path and it defaults to the `PROJECT_ROOT` directory.
    * `filepath` should be a relative path, starting from `root`.
    """
    try:
        # Python 2.x
        with open(os.path.join(root, filepath)) as fd:
            text = fd.read()
    except UnicodeDecodeError:
        # Python 3.x
        with open(os.path.join(root, filepath), encoding="utf8") as fd:
            text = fd.read()
    return text


LONG_DESCRIPTION = read_file("README.md")
SHORT_DESCRIPTION = "Modules for preparing meteorological inputs from EMCWF data."
REQS = ['numpy>=1.10', 'gdal', 'netCDF4', 'cdsapi', "xarray", "pyyaml",
        "scikit-image", "pyDMS", "pyTSEB"]

setup(
    name                  = "meteo_utils",
    packages              = ['meteo_utils'],
    dependency_links      = ['git+https://github.com/radosuav/pyDMS',
                             'git+https://github.com/hectornieto/pyTSEB'],
    install_requires      = REQS,
    version               = "1.0",
    author                = "Hector Nieto",
    author_email          = "hector.nieto@complutig.com",
    maintainer            = "Hector Nieto",
    maintainer_email      = "hector.nieto@complutig.com",
    description           = SHORT_DESCRIPTION,
    url                   = "https://github.com/hectornieto/sen2agrotig/",
    long_description      = LONG_DESCRIPTION,
    classifiers           = [
        "Development Status :: Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Agricultural Science",
        "Topic :: Scientific/Engineering :: Hydrology",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3"],
    keywords             = ['SEN-ET','Evapotranspiration',
                            'Sentinel-2','Sentinel-3','SLSTR','Remote Sensing'])
