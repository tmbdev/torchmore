#
# Copyright (c) 2017-2019 TBD. All rights reserved.
# This file is part of TBD (see TBD).
# See the LICENSE file for licensing terms (TBD).
#

# TODO transfer the documentation strings to flex creators/instances

import numpy as np
import torch
from torch import autograd, nn
from torch.autograd import Variable
import warnings
from functools import wraps

from . import helpers, layers

verbose = False

class Flex(nn.Module):
    def __init__(self, creator):
        super(Flex, self).__init__()
        self.creator = creator
        self.layer = None

    def forward(self, *args):
        if self.layer is None:
            self.layer = self.creator(*args)
            if verbose: print("# created", self.layer)
        return self.layer.forward(*args)

    def __repr__(self):
        return "Flex:"+repr(self.layer)

    def __str__(self):
        return "Flex:"+str(self.layer)


def Linear(*args, **kw):
    def creator(x):
        assert x.ndimension() == 2
        return nn.Linear(x.size(1), *args, **kw)
    return Flex(creator)


def Conv1d(*args, **kw):
    def creator(x):
        assert x.ndimension() == 3
        d = x.size(1)
        return nn.Conv1d(x.size(1), *args, **kw)
    return Flex(creator)


def Conv2d(*args, **kw):
    def creator(x):
        assert x.ndimension() == 4
        return nn.Conv2d(x.size(1), *args, **kw)
    return Flex(creator)


def Conv3d(*args, **kw):
    def creator(x):
        assert x.ndimension() == 5
        return nn.Conv3d(x.size(1), *args, **kw)
    return Flex(creator)


def ConvTranspose1d(*args, **kw):
    def creator(x):
        assert x.ndimension() == 3
        d = x.size(1)
        return nn.ConvTranspose1d(x.size(1), *args, **kw)
    return Flex(creator)


def ConvTranspose2d(*args, **kw):
    def creator(x):
        assert x.ndimension() == 4
        return nn.ConvTranspose2d(x.size(1), *args, **kw)
    return Flex(creator)


def ConvTranspose3d(*args, **kw):
    def creator(x):
        assert x.ndimension() == 5
        return nn.ConvTranspose3d(x.size(1), *args, **kw)
    return Flex(creator)


def LSTM(*args, **kw):
    def creator(x):
        assert x.ndimension() == 3
        return layers.LSTM(x.size(2), *args, **kw)
    return Flex(creator)

def BDL_LSTM(*args, **kw):
    def creator(x):
        assert x.ndimension() == 3
        return layers.BDL_LSTM(x.size(1), *args, **kw)
    return Flex(creator)

Lstm1 = BDL_LSTM
Lstm1d = BDL_LSTM

def BDHW_LSTM(*args, **kw):
    def creator(x):
        assert x.ndimension() == 4
        return layers.BDHW_LSTM(x.size(1), *args, **kw)
    return Flex(creator)

Lstm2 = BDHW_LSTM
Lstm2d = BDHW_LSTM

def BatchNorm(*args, **kw):
    def creator(x):
        assert x.ndimension() == 2
        assert x.size(0) > 1, f"batch norm requires batch size > 1, got {x.shape}"
        return nn.BatchNorm1d(x.size(1), *args, **kw)
    return Flex(creator)

def BatchNorm1d(*args, **kw):
    def creator(x):
        assert x.ndimension() == 3
        return nn.BatchNorm1d(x.size(1), *args, **kw)
    return Flex(creator)


def BatchNorm2d(*args, **kw):
    def creator(x):
        assert x.ndimension() == 4
        return nn.BatchNorm2d(x.size(1), *args, **kw)
    return Flex(creator)


def BatchNorm3d(*args, **kw):
    def creator(x):
        assert x.ndimension() == 5
        return nn.BatchNorm3d(x.size(1), *args, **kw)
    return Flex(creator)


def replace_modules(model, f):
    for key in list(model._modules.keys()):
        sub = model._modules[key]
        replacement = f(sub)
        if replacement is not None:
            model._modules[key] = replacement
        else:
            replace_modules(sub, f)


def flex_replacer(module):
    if isinstance(module, Flex):
        return module.layer
    else:
        return None

def flex_freeze(model):
    replace_modules(model, flex_replacer)

def freeze(model):
    replace_modules(model, flex_replacer)

def shape_inference(model, tensor, dtype=None):
    if isinstance(tensor, (tuple, list)):
        tensor = torch.zeros(tensor, dtype=dtype)
    model.eval()
    with autograd.no_grad():
        model(tensor)
    replace_modules(model, flex_replacer)

def delete_modules(model, f):
    for key in list(model._modules.keys()):
        if f(model._modules[key]):
            del model._modules[key]
