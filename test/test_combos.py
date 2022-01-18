#
# Copyright (c) 2017-2019 TBD. All rights reserved.
# This file is part of TBD (see TBD).
# See the LICENSE file for licensing terms (TBD).
#

import torch
from torch import nn
from torchmore import combos
from torchmore import flex


def test_UnetLayer():
    mod = combos.UnetLayer(33, sub=nn.Identity(), post=nn.Conv2d(2 * 33, 7, 3, padding=1), dropout=0.5)
    a = torch.ones((17, 11, 64, 64))
    b = mod(a)
    assert tuple(b.size()) == (17, 7, 64, 64)


def test_make_unet():
    mod = combos.make_unet([16, 32, 64], dropout=[0.5, 0.5, 0.5])
    a = torch.ones((17, 11, 64, 64))
    b = mod(a)
    assert tuple(b.size()) == (17, 32, 64, 64)


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
