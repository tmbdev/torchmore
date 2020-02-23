#!/usr/bin/python3
#
# Copyright (c) 2017-2019 TBD. All rights reserved.
# This file is part of TBD (see TBD).
# See the LICENSE file for licensing terms (TBD).
#

import setuptools

VERSION = "0.1.0"

setuptools.setup(
    author="Thomas Breuel",
    author_email="tmbdev+removeme@gmail.com",
    description="Useful additional layers for PyTorch.",
    install_requires="simplejson braceexpand msgpack pyyaml numpy torch Pillow torchvision scipy".split(),
    keywords="object store, client, deep learning",
    license="MIT",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    name='torchmore',
    packages=["torchmore"],
    python_requires=">=3.6",
    # scripts=scripts,
    url="http://github.com/tmbdev/torchmore",
    version=VERSION,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7"
    ],
)

