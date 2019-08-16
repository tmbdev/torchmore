#!/usr/bin/python3
#
# Copyright (c) 2017-2019 TBD. All rights reserved.
# This file is part of TBD (see TBD).
# See the LICENSE file for licensing terms (TBD).
#

from __future__ import print_function

import glob
import sys
from distutils.core import setup  # , Extension, Command

#scripts = glob.glob("wl-*[a-z]")

setup(
    name='torchmore',
    version='v0.0',
    author="Thomas Breuel",
    description="Useful additional layers for PyTorch.",
    packages=["torchmore"],
    # scripts=scripts,
)
