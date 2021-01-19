#
# Copyright (c) 2017-2019 TBD. All rights reserved.
# This file is part of TBD (see TBD).
# See the LICENSE file for licensing terms (TBD).
#

"""A set of helper functions for dealing uniformly with tensors and
ndarrays."""

import numpy as np


def typeas(a, b):
    return a.type(b.dtype).to(b.device)


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
    assert batch.ndim == len(inp)
    result = [inp.find(c) for c in out]
    for x in result:
        assert x >= 0, result
    if isinstance(batch, torch.Tensor):
        return batch.permute(*result)
    elif isinstance(batch, np.ndarray):
        return batch.transpose(*result)


def sequence_is_normalized(a, dim=-1):
    sums = a.detach().sum(dim)
    err = (sums - 1.0).abs().max().item()
    return err < 1e-4


class LearningRateSchedule(object):
    """Translate n1,lr1:n2,lr2... into a function that maps steps into learning rates."""
    def __init__(self, schedule):
        if ":" in schedule:
            self.learning_rates = [
                [float(y) for y in x.split(",")] for x in schedule.split(":")
            ]
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


def lr_schedule(s):
    """Converts a string into a learning rate schedule.

    Permitted inputs:
    - a callable (just returned directly)
    - a string of the form "=0,l1:n2,l2:n3,l3"
    - a lambda or other function specification
    """
    if callable(s):
        return s
    if s[0] == "=":
        return LearningRateSchedule(s[1:])
    f = eval(s)
    assert callable(f)
    return f


class Schedule:
    """A quick way of scheduling events at time intervals.

    Usage:

        schedule = Schedule()
        for ...:
            if schedule("eventname", 60):
                ...
    """
    def __init__(self):
        self.jobs = {}

    def __call__(self, key, seconds, initial=False):
        now = time.time()
        last = self.jobs.setdefault(key, 0 if initial else now)
        if now - last > seconds:
            self.jobs[key] = now
            return True
        else:
            return False
