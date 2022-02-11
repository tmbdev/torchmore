#
# Copyright (c) 2017-2019 TBD. All rights reserved.
# This file is part of TBD (see TBD).
# See the LICENSE file for licensing terms (TBD).
#

import torch
from torch import nn
from torchmore import combos
from torchmore import flex


def test_UnetLayer0():
    mod = combos.UnetLayer0(33, dropout=0.5)
    flex.shape_inference(mod, (17, 11, 64, 64))
    print(mod)
    a = torch.ones((17, 11, 64, 64))
    b = mod(a)
    assert tuple(b.size()) == (17, 66, 64, 64)


def test_UnetLayer1():
    mod = combos.UnetLayer1(33, dropout=0.5)
    flex.shape_inference(mod, (17, 11, 64, 64))
    print(mod)
    a = torch.ones((17, 11, 64, 64))
    b = mod(a)
    assert tuple(b.size()) == (17, 66, 64, 64)


def test_make_unet0():
    mod = combos.make_unet([16, 32, 64], dropout=[0.5, 0.5, 0.5], mode=0)
    a = torch.ones((17, 11, 64, 64))
    b = mod(a)
    assert tuple(b.size()) == (17, 32, 64, 64)


def test_make_unet1():
    mod = combos.make_unet([16, 32, 64], dropout=[0.5, 0.5, 0.5], mode=1)
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
