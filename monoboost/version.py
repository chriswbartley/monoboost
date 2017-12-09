from __future__ import absolute_import, division, print_function
from os.path import join as pjoin

# Format expected by setup.py and doc/source/conf.py: string of form "X.Y.Z"
_version_major = 0
_version_minor = 7
_version_micro = ''  # use '' for first of series, number for 1 and above
_version_extra = 'dev'
_version_extra = ''  # Uncomment this for full releases

# Construct full version string from these.
_ver = [_version_major, _version_minor]
if _version_micro:
    _ver.append(_version_micro)
if _version_extra:
    _ver.append(_version_extra)

__version__ = '.'.join(map(str, _ver))

CLASSIFIERS = ["Development Status :: 3 - Alpha",
               "Environment :: Console",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: MIT License",
               "Operating System :: OS Independent",
               "Programming Language :: Python",
               "Topic :: Scientific/Engineering"]

# Description should be a one-liner:
description = "monoboost: a partially monotone instance based classification library"
# Long description will go up on the pypi page
long_description = """

Monoboost
========
Monoboost is a template project for small scientific Python projects.

It contains software implementations of an analysis of some simple data, but
more importantly, it contains infrastructure for testing, documentation,
continuous integration and deployment, which can be easily adapted
to use in other projects.

To get started using these components in your own software, please go to the
repository README_.

.. _README: https://github.com/chriswbartley/monoboost/blob/master/README.md

License
=======
``monoboost`` is licensed under the terms of the MIT license. See the file
"LICENSE" for information on the history of this software, terms & conditions
for usage, and a DISCLAIMER OF ALL WARRANTIES.

All trademarks referenced herein are property of their respective holders.

Copyright (c) 2017--, Christopher Bartley, The University of Western Australia.
"""

NAME = "MonoBoost"
MAINTAINER = "Christopher Bartley"
MAINTAINER_EMAIL = "christopher.bartley@research.uwa.edu.au"
DESCRIPTION = description
LONG_DESCRIPTION = long_description
URL = "http://github.com/chriswbartley/monoboost"
DOWNLOAD_URL = ""
LICENSE = "MIT"
AUTHOR = "Christopher Bartley"
AUTHOR_EMAIL = "christopher.bartley@research.uwa.edu.au"
PLATFORMS = "OS Independent"
MAJOR = _version_major
MINOR = _version_minor
MICRO = _version_micro
VERSION = __version__
PACKAGE_DATA = {'monoboost': [pjoin('data', '*')]}
REQUIRES = ["numpy","scipy","scikit-learn"]
