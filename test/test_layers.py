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

def test_weighted_grad():
    pass

def test_Fun():
    mod = layers.Fun("lambda x: x[0]**2")
    a = torch.ones((2, 3, 4))
    b = mod(a)
    assert (b==1).all()
    assert tuple(b.shape) == (3, 4)

def test_Info():
    mod = layers.Info()
    a = torch.ones((2, 3, 4))
    b = mod(a)
    assert (b==a).all()

def test_CheckSizes():
    mod = layers.CheckSizes(2, 3, 4)
    a = torch.ones((2, 3, 4))
    b = mod(a)
    assert (b==a).all()

def test_Device():
    mod = layers.Device("cpu")
    a = torch.ones((2, 3, 4))
    b = mod(a)
    assert (b.cpu()==a.cpu()).all()

def test_CheckRange():
    mod = layers.CheckRange(0, 2)
    a = torch.ones((2, 3, 4))
    b = mod(a)
    assert (b.cpu()==a.cpu()).all()

def test_Input():
    mod = layers.Input("ABCD", sizes=(4, 5, 2, 3))
    a = torch.ones((2, 3, 4, 5))
    a.order = "CDAB"
    b = mod(a)
    assert tuple(b.shape) == (4, 5, 2, 3)

def test_Reorder():
    mod = layers.Reorder("ABC", "CBA")
    a = torch.ones((2, 3, 4))
    b = mod(a)
    assert tuple(b.shape) == (4, 3, 2)

def test_Permute():
    mod = layers.Permute(2, 1, 0)
    a = torch.ones((2, 3, 4))
    b = mod(a)
    assert tuple(b.shape) == (4, 3, 2)

def test_Reshape():
    mod = layers.Reshape(0, (1, 2))
    a = torch.ones((2, 3, 4))
    b = mod(a)
    assert tuple(b.shape) == (2, 12)

def test_Viewer():
    mod = layers.Viewer(6, 4)
    a = torch.ones((2, 3, 4))
    b = mod(a)
    assert tuple(b.shape) == (6, 4)

def test_BDL_LSTM():
    mod = layers.BDL_LSTM(3, 20)
    a = torch.ones((17, 3, 100))
    b = mod(a)
    assert tuple(b.shape) == (17, 20, 100)

def test_BDHW_LSTM():
    mod = layers.BDHW_LSTM(3, 20)
    a = torch.ones((17, 3, 48, 64))
    b = mod(a)
    assert tuple(b.shape) == (17, 40, 48, 64)

def test_BDHW_LSTM_to_BDH():
    mod = layers.BDHW_LSTM_to_BDH(3, 13)
    a = torch.ones((17, 3, 48, 64))
    b = mod(a)
    assert tuple(b.shape) == (17, 13, 48)

def test_NoopSub():
    mod = layers.NoopSub(sub=nn.Identity())
    a = torch.ones((17, 11, 64, 64))
    b = mod(a)
    assert tuple(b.size()) == (17, 11, 64, 64)

def test_KeepSize():
    mod = layers.KeepSize(sub=nn.FractionalMaxPool2d((3, 3), output_ratio=(0.7, 0.9)))
    a = torch.ones((17, 11, 64, 64))
    b = mod(a)
    assert tuple(b.size()) == (17, 11, 64, 64)

def test_Additive():
    mod = layers.Additive(
        nn.Conv2d(11, 17, 3, padding=1),
        nn.Conv2d(11, 17, 5, padding=2)
    )
    a = torch.ones((17, 11, 64, 64))
    b = mod(a)
    assert tuple(b.size()) == (17, 17, 64, 64)

def test_Parallel():
    mod = layers.Parallel(
        nn.Conv2d(1, 4, 3, padding=(1, 1)),
        nn.Conv2d(1, 7, 3, padding=(1, 1)))
    a = torch.ones((17, 1, 64, 64))
    b = mod(a)
    assert tuple(b.size()) == (17, 11, 64, 64)

def test_SimplePooling2d():
    mod = layers.SimplePooling2d([nn.Conv2d(1, 1, 3, padding=(1, 1))])
    a = torch.ones((17, 1, 64, 64))
    b = mod(a)
    assert a.size() == b.size()

def test_AcrossPooling2d():
    mod = layers.AcrossPooling2d(
        [nn.Conv2d(1, 1, 3, padding=(1, 1))],
        [nn.Conv2d(1, 7, 3, padding=(1, 1))])
    a = torch.ones((17, 1, 64, 64))
    b = mod(a)
    assert tuple(b.size()) == (17, 8, 64, 64)

