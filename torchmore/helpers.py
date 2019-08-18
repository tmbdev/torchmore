#
# Copyright (c) 2017-2019 TBD. All rights reserved.
# This file is part of TBD (see TBD).
# See the LICENSE file for licensing terms (TBD).
#

"""A set of helper functions for dealing uniformly with tensors and
ndarrays."""

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from torch import autograd, nn, optim
from torch.autograd import Variable

def typeas(a, b):
    return a.type(b.dtype).to(b.device)

def bhwd2bdhw(images, depth1=False):
    images = as_torch(images)
    if depth1:
        assert len(shp(images)) == 3, shp(images)
        images = images.unsqueeze(3)
    assert len(shp(images)) == 4, shp(images)
    return images.permute(0, 3, 1, 2)


def bdhw2bhwd(images, depth1=False):
    images = as_torch(images)
    assert len(shp(images)) == 4, shp(images)
    images = images.permute(0, 2, 3, 1)
    if depth1:
        assert images.size(3) == 1
        images = images.index_select(3, 0)
    return images


def reorder(batch, inp, out):
    """Reorder the dimensions of the batch from inp to out order.

    E.g. BHWD -> BDHW.
    """
    if inp is None:
        return batch
    if out is None:
        return batch
    assert isinstance(inp, str)
    assert isinstance(out, str)
    assert len(inp) == len(out), (inp, out)
    assert rank(batch) == len(inp), (rank(batch), inp)
    result = [inp.find(c) for c in out]
    # print ">>>>>>>>>>>>>>>> reorder", result
    for x in result:
        assert x >= 0, result
    if is_tensor(batch):
        return batch.permute(*result)
    elif isinstance(batch, np.ndarray):
        return batch.transpose(*result)


def assign(dest, src, transpose_on_convert=None):
    """Resizes the destination and copies the source."""
    src = as_torch(src, transpose_on_convert)
    if isinstance(dest, Variable):
        dest.data.resize_(*shp(src)).copy_(src)
    elif isinstance(dest, torch.Tensor):
        dest.resize_(*shp(src)).copy_(src)
    else:
        raise ValueError("{}: unknown type".format(type(dest)))

def sequence_is_normalized(a, dim=-1):
    sums = a.detach().sum(dim)
    err = (sums-1.0).abs().max().item()
    return err < 1e-4
    
def ctc_align(prob, target):
    """Perform CTC alignment on torch sequence batches (using ocrolstm)"""
    import cctc2
    prob_ = prob.cpu()
    target = target.cpu()
    b, l, d = prob.size()
    bt, lt, dt = target.size()
    assert bt == b, (bt, b)
    assert dt == d, (dt, d)
    assert sequence_is_normalized(prob, 2), prob
    assert sequence_is_normalized(target, 2), target
    result = cctc2.ctc_align_targets_batch(prob_, target)
    return typeas(result, prob)


def ctc_loss(probs, target):
    """A CTC loss function for BLD sequence training."""
    assert probs.is_contiguous()
    assert target.is_contiguous()
    assert sequence_is_normalized(probs)
    assert sequence_is_normalized(target)
    aligned = ctc_align(probs, target)
    assert aligned.size() == probs.size(), \
        (aligned.size(), probs.size())
    deltas = aligned - probs
    probs.backward(deltas.contiguous())
    return typeas(deltas, probs), typeas(aligned, probs)


class LearningRateSchedule(object):
    def __init__(self, schedule):
        if ":" in schedule:
            self.learning_rates = [
                [float(y) for y in x.split(",")] for x in schedule.split(":")]
            assert self.learning_rates[0][0] == 0
        else:
            lr0 = float(schedule)
            self.learning_rates = [[0, lr0]]

    def __call__(self, count):
        _, lr = self.learning_rates[0]
        for n, l in self.learning_rates:
            if count < n:
                break
            lr = l
        return lr
