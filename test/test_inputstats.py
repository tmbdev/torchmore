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
    assert "4,4" in str(mod)
    mod.train(False)
    mod.forward(a)
    mod.forward(b)
    mod.mode = "check_range"
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
    assert "4,4" in str(mod)
    fname = tmpdir.join("test.pth")
    print(mod)
    with open(fname, "wb") as stream:
        state = mod.state_dict()
        print("[[[", state, "]]]")
        torch.save(state, stream)
    mod2 = inputstats.InputStats()
    with open(fname, "rb") as stream:
        state = torch.load(stream)
        mod2.load_state_dict(state)
    print(state)
    print(mod2)
    assert len(mod2) == 100
    assert "4,4" in str(mod2)
