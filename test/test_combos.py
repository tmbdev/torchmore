#
# Copyright (c) 2017-2019 TBD. All rights reserved.
# This file is part of TBD (see TBD).
# See the LICENSE file for licensing terms (TBD).
#
from __future__ import unicode_literals

import pdb
from builtins import next
import sys
import os
import numpy as np

import torch
from torch import nn
from torchmore import layers
from torchmore import flex
from torchmore import combos

def test_UnetLayer():
    mod = combos.UnetLayer(33, sub=nn.Identity(), post=nn.Conv2d(33+11, 7, 3, padding=1))
    a = torch.ones((17, 11, 64, 64))
    b = mod(a)
    assert tuple(b.size()) == (17, 7, 64, 64)

def test_ResnetBlock():
    mod = combos.ResnetBlock(33)
    a = torch.ones((17, 33, 64, 64))
    b = mod(a)
    assert tuple(b.size()) == (17, 33, 64, 64)

def test_ResnetBottleneck():
    mod = combos.ResnetBottleneck(33, 22)
    a = torch.ones((17, 33, 64, 64))
    b = mod(a)
    assert tuple(b.size()) == (17, 33, 64, 64)
