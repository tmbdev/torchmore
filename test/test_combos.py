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
    for mode in range(100):
        if f"UnetLayer{mode}" not in combos.__dict__:
            continue
        print(f"testting mode {mode}:")
        mod = combos.__dict__[f"UnetLayer{mode}"](33, dropout=0.5)
        flex.shape_inference(mod, (17, 11, 64, 64))
        print(mod)
        a = torch.ones((17, 11, 64, 64))
        b = mod(a)
        assert b.shape[:1] == a.shape[:1]
        assert b.shape[2:] == a.shape[2:]


def test_make_unet():
    for mode in range(100):
        if f"UnetLayer{mode}" not in combos.__dict__:
            continue
        print(f"testting mode {mode}:")
        mod = combos.make_unet([16, 32, 64], dropout=[0.5, 0.5, 0.5], mode=mode)
        a = torch.ones((17, 11, 64, 64))
        b = mod(a)
        assert b.shape[:1] == a.shape[:1]
        assert b.shape[2:] == a.shape[2:]


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
