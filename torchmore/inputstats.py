#
# Copyright (c) 2017-2019 TBD. All rights reserved.
# This file is part of TBD (see TBD).
# See the LICENSE file for licensing terms (TBD).
#

import warnings
import torch
from torch import nn
from torch import Tensor


def empty_stats():
    return torch.tensor([1e38, -1e38, 0, 0, 0])


def update_stats(stats: Tensor, x: float, message: str = ""):
    assert len(stats) == 5
    stats[0] = min(stats[0], x)
    stats[1] = max(stats[1], x)
    stats[2] += 1.0
    stats[3] += x
    stats[4] += x * x


def check_range(stats: Tensor, x: float):
    return x >= stats[0] and x <= stats[1]


def check_sigma(stats: Tensor, x: float, sigmas: float = 4.0):
    assert stats[2] > 0
    mean = stats[3] / stats[2]
    std = ((stats[4] / stats[2]) - mean ** 2) ** 0.5
    return x >= mean - sigmas * std and x <= mean + sigmas * std


class InputStats(nn.Module):
    def __init__(self, name: str = "InputStats"):
        super().__init__()
        self.updating = True
        self.name = name
        self.register_buffer(
            "dim_stats", torch.vstack([empty_stats() for _ in range(8)])
        )
        self.register_buffer("min_stats", empty_stats())
        self.register_buffer("max_stats", empty_stats())
        self.register_buffer("mean_stats", empty_stats())
        self.register_buffer("std_stats", empty_stats())

    def train(self, mode=True):
        if mode:
            self.updating = True
        else:
            self.updating = False

    def add_value(self, stats, x: float, message: str):
        if not self.updating:
            return
        update_stats(stats, x)

    def numsamples(self) -> int:
        return int(self.min_stats[2].cpu().detach().item())

    def __len__(self):
        return self.numsamples()

    def forward(self, a):
        for i in range(min(a.ndim, len(self.dim_stats))):
            self.add_value(self.dim_stats[i], float(a.shape[i]), f"dim({i})")
        self.add_value(self.min_stats, a.detach().min().cpu().item(), "min value")
        self.add_value(self.max_stats, a.detach().max().cpu().item(), "max value")
        self.add_value(self.mean_stats, a.detach().mean().cpu().item(), "mean value")
        self.add_value(self.std_stats, a.detach().std().cpu().item(), "std value")
        return a

    def __str__(self):
        result = f"InputStats({self.name}"
        if self.dim_stats is not None and len(self) > 0:
            result += f" (n={len(self)})"
            result += " dims"
            for i in range(len(self.dim_stats)):
                if self.dim_stats[i][0] > self.dim_stats[i][1]:
                    break
                result += f" [{int(self.dim_stats[i][0])},{int(self.dim_stats[i][1])}]"
            result += f" min [{self.min_stats[0]:.3g},{self.min_stats[1]:.3g}]"
            result += f" max [{self.max_stats[0]:.3g},{self.max_stats[1]:.3g}]"
            result += f" mean [{self.mean_stats[0]:.3g},{self.mean_stats[1]:.3g}]"
            result += f" std [{self.std_stats[0]:.3g},{self.std_stats[1]:.3g}]"
        result += ")"
        return result

    def __repr__(self):
        return str(self)


class SampleTensorBase(nn.Module):
    def __init__(self, bufsize: int = 4096):
        super().__init__()
        self.bufsize = bufsize
        self.count = 0
        self.every = 1
        self.register_buffer("buf", None)
        self.register_buffer("counts", None)

    def add_tensor(self, tensor: Tensor):
        self.count += 1
        if (self.count-1) % self.every != 0:
            return
        a = torch.tensor(tensor.detach(), device="cpu")
        if self.buf is None:
            self.buf = a.unsqueeze(0)
            self.counts = torch.tensor([self.count])
            return
        if len(self.buf) >= self.bufsize:
            self.buf = self.buf[::2]
            self.counts = self.counts[::2]
            self.every *= 2
        self.buf = torch.cat([self.buf, a.unsqueeze(0)], dim=0)
        self.counts = torch.cat([self.counts, torch.tensor([self.count])])

    def __len__(self):
        return len(self.buf)

    def __iter__(self):
        return iter(self.buf)

    def __str__(self):
        return f"SampleTensors(bufsize={self.bufsize}, every={self.every}, len={len(self)})"

    def __repr__(self):
        return str(self)


class SampleTensor(SampleTensorBase):
    def __init__(self, bufsize: int = 4096):
        super().__init__(bufsize)
    def forward(self, a):
        self.add_tensor(a)
        return a

class SampleGrandient(SampleTensorBase):
    def __init__(self, bufsize: int = 4096):
        super().__init__(bufsize)
    def backward(self, grad_output):
        self.add_tensor(grad_output)
        return grad_output
