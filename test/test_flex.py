#
# Copyright (c) 2017-2019 TBD. All rights reserved.
# This file is part of TBD (see TBD).
# See the LICENSE file for licensing terms (TBD).
#

import numpy as np
import torch
from torch import autograd, nn
from torch.autograd import Variable
import warnings

from torchmore import flex, layers

kw = dict(kernel_size=3, padding=1)

def test_Linear():
    mod = flex.Linear(17)
    a = torch.zeros((7, 3))
    b = mod(a)
    assert b.size() == (7, 17)
    a = torch.zeros((4, 3))
    b = mod(a)
    assert b.size() == (4, 17)


def test_Conv1d():
    mod = flex.Conv1d(17, 3, padding=1)
    a = torch.zeros((7, 3, 99))
    b = mod(a)
    assert b.size() == (7, 17, 99)
    a = torch.zeros((4, 3, 9))
    b = mod(a)
    assert b.size() == (4, 17, 9)


def test_Conv2d():
    mod = flex.Conv2d(17, 3, padding=1)
    a = torch.zeros((7, 3, 99, 88))
    b = mod(a)
    assert b.size() == (7, 17, 99, 88)
    a = torch.zeros((4, 3, 9, 8))
    b = mod(a)
    assert b.size() == (4, 17, 9, 8)


def test_Conv3d():
    mod = flex.Conv3d(17, 3, padding=1)
    a = torch.zeros((7, 3, 99, 88, 77))
    b = mod(a)
    assert b.size() == (7, 17, 99, 88, 77)
    a = torch.zeros((4, 3, 9, 8, 7))
    b = mod(a)
    assert b.size() == (4, 17, 9, 8, 7)

def test_ConvTranspose1d():
    mod = flex.ConvTranspose1d(17, 3, padding=1)
    a = torch.zeros((7, 3, 99))
    b = mod(a)
    assert b.size() == (7, 17, 99)
    a = torch.zeros((4, 3, 9))
    b = mod(a)
    assert b.size() == (4, 17, 9)


def test_ConvTranspose2d():
    mod = flex.ConvTranspose2d(17, 3, padding=1, stride=2)
    a = torch.zeros((7, 3, 99, 88))
    b = mod(a)
    assert b.size() == (7, 17, 197, 175)
    a = torch.zeros((4, 3, 9, 8))
    b = mod(a)
    assert b.size() == (4, 17, 17, 15)


def test_ConvTranspose3d():
    mod = flex.ConvTranspose3d(17, 3, padding=1)
    a = torch.zeros((7, 3, 99, 88, 77))
    b = mod(a)
    assert b.size() == (7, 17, 99, 88, 77)
    a = torch.zeros((4, 3, 9, 8, 7))
    b = mod(a)
    assert b.size() == (4, 17, 9, 8, 7)


def test_LSTM():
    mod = flex.LSTM(17)
    a = torch.zeros((99, 7, 4))
    b = mod(a)
    assert b.size() == (99, 7, 17)

def test_Lstm1d():
    mod = flex.Lstm1d(17)
    a = torch.zeros((7, 3, 99))
    b = mod(a)
    assert b.size() == (7, 17, 99)
    a = torch.zeros((4, 3, 9))
    b = mod(a)
    assert b.size() == (4, 17, 9)

def test_Lstm2d():
    mod = flex.Lstm2d(17)
    a = torch.zeros((7, 3, 99, 88))
    b = mod(a)
    assert b.size() == (7, 34, 99, 88)
    a = torch.zeros((4, 3, 9, 8))
    b = mod(a)
    assert b.size() == (4, 34, 9, 8)

def test_BatchNorm1d():
    mod = flex.BatchNorm1d(3)
    a = torch.zeros((7, 3, 99))
    b = mod(a)
    assert b.size() == (7, 3, 99)
    a = torch.zeros((4, 3, 9))
    b = mod(a)
    assert b.size() == (4, 3, 9)


def test_BatchNorm2d():
    mod = flex.BatchNorm2d(3)
    a = torch.zeros((7, 3, 99, 88))
    b = mod(a)
    assert b.size() == (7, 3, 99, 88)
    a = torch.zeros((4, 3, 9, 8))
    b = mod(a)
    assert b.size() == (4, 3, 9, 8)


def test_BatchNorm3d():
    mod = flex.BatchNorm3d(3)
    a = torch.zeros((7, 3, 99, 88, 77))
    b = mod(a)
    assert b.size() == (7, 3, 99, 88, 77)
    a = torch.zeros((4, 3, 9, 8, 7))
    b = mod(a)
    assert b.size() == (4, 3, 9, 8, 7)


def test_shape_inference():
    mod = nn.Sequential(flex.Conv1d(3, 3, padding=1))
    print(mod)
    assert "Flex" in repr(mod)
    a = torch.zeros((7, 3, 99))
    b = mod(a)
    assert b.size() == (7, 3, 99)
    assert "Flex" in repr(mod)
    print(mod)
    flex.shape_inference(mod, a.shape)
    print(mod)
    assert "Flex" not in repr(mod)
    a = torch.zeros((4, 3, 9))
    b = mod(a)
    assert b.size() == (4, 3, 9)
