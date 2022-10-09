#
# Copyright (c) 2017-2019 TBD. All rights reserved.
# This file is part of TBD (see TBD).
# See the LICENSE file for licensing terms (TBD).
#

# TODO transfer the documentation strings to flex creators/instances

import torch
from torch import autograd, nn

from . import layers

verbose = False


def arginfo(l):
    return tuple([p.shape if isinstance(p, torch.Tensor) else p for p in l])

class Flex(nn.Module):
    def __init__(self, creator):
        super(Flex, self).__init__()
        self.creator = creator
        self.layer = None

    def forward(self, *args):
        if self.layer is None:
            self.layer = self.creator(*args)
            creating = True
            if verbose:
                print(f"Flex: {self.creator}{arginfo(args)} -> {self.layer}")
        result = self.layer.forward(*args)
        return result

    def __repr__(self):
        info = repr(self.layer) if self.layer is not None else repr(self.creator)
        return "Flex(%s)" % info

    def __str__(self):
        info = repr(self.layer) if self.layer is not None else repr(self.creator)
        return "Flex(%s)" % info


def Linear(*args, **kw):
    def creator(x):
        assert x.ndimension() == 2
        return nn.Linear(x.size(1), *args, **kw)

    return Flex(creator)


def Conv1d(*args, **kw):
    def creator(x):
        assert x.ndimension() == 3
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


def LSTMn(*args, **kw):
    def creator(x):
        if x.ndimension() == 3:
            return layers.BDL_LSTM(x.size(1), *args, **kw)
        elif x.ndimension() == 4:
            return layers.BDHW_LSTM(x.size(1), *args, **kw)
        else:
            raise ValueError(f"{x.shape}: multidimensional LSTM not implemented for this rank array")

    return Flex(creator)


def BatchNorm(*args, **kw):
    def creator(x):
        if x.ndimension() in [2, 3]:
            return nn.BatchNorm1d(x.size(1), *args, **kw)
        elif x.ndimension() == 4:
            return nn.BatchNorm2d(x.size(1), *args, **kw)
        elif x.ndimension() == 5:
            return nn.BatchNorm3d(x.size(1), *args, **kw)
        else:
            raise ValueError("unsupported dimension")

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


def InstanceNorm1d(*args, **kw):
    def creator(x):
        assert x.ndimension() == 3
        return nn.InstanceNorm1d(x.size(1), *args, **kw)

    return Flex(creator)


def InstanceNorm2d(*args, **kw):
    def creator(x):
        assert x.ndimension() == 4
        return nn.InstanceNorm2d(x.size(1), *args, **kw)

    return Flex(creator)


def InstanceNorm3d(*args, **kw):
    def creator(x):
        assert x.ndimension() == 5
        return nn.InstanceNorm3d(x.size(1), *args, **kw)

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
