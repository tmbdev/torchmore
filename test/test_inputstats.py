#
# Copyright (c) 2017-2019 TBD. All rights reserved.
# This file is part of TBD (see TBD).
# See the LICENSE file for licensing terms (TBD).
#
from __future__ import unicode_literals

import pdb
import pytest

import torch
from torch import nn
from torchmore import inputstats


def test_InputStats():
    mod = inputstats.InputStats(name="mystats", error=True)
    for i in range(100):
        a = torch.rand([3, 4, 5])
        mod.forward(a)
    for i in range(100):
        b = torch.rand([3, 4, 6])
        mod.forward(b)
    assert len(mod) == 200
    assert "mystats" in str(mod)
    assert "4.0,4.0" in str(mod)
    mod.train(False)
    mod.forward(a)
    mod.forward(b)
    with pytest.raises(ValueError):
        mod.forward(torch.rand([3, 4, 5]) + 2.0)
    with pytest.raises(ValueError):
        mod.forward(torch.rand([3, 4, 7]))


def test_InputStats2(tmpdir):
    mod = inputstats.InputStats(name="mystats", error=True)
    for i in range(100):
        a = torch.rand([3, 4, 5])
        mod.forward(a)
    assert len(mod) == 100
    assert "mystats" in str(mod)
    assert "4.0,4.0" in str(mod)
    fname = tmpdir.join("test.pth")
    print(mod)
    with open(fname, "wb") as stream:
        torch.save(mod, stream)
    with open(fname, "rb") as stream:
        mod2 = torch.load(stream)
    print(mod2)
    assert len(mod2) == 100
    assert "mystats" in str(mod2)
    assert "4.0,4.0" in str(mod2)
