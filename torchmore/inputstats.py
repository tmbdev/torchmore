#
# Copyright (c) 2017-2019 TBD. All rights reserved.
# This file is part of TBD (see TBD).
# See the LICENSE file for licensing terms (TBD).
#

import warnings
import torch
from torch import nn


def empty_stats():
    return torch.tensor([1e38, -1e38, 0, 0, 0])


def update_stats(stats, x, message=None):
    assert len(stats) == 5
    stats[0] = min(stats[0], x)
    stats[1] = max(stats[1], x)
    stats[2] += 1.0
    stats[3] += x
    stats[4] += x * x


def check_range(stats, x):
    return x >= stats[0] and x <= stats[1]


def check_sigma(stats, x, sigmas=4):
    assert stats[2] > 0
    mean = stats[3] / stats[2]
    std = ((stats[4] / stats[2]) - mean ** 2) ** 0.5
    return x >= mean - sigmas * std and x <= mean + sigmas * std


class InputStats(nn.Module):
    def __init__(self, name="InputStats", error=False, mode="check_std"):
        super().__init__()
        self.train()
        self.name = name
        self.error = error
        self.register_buffer("dim_stats", torch.vstack([empty_stats() for _ in range(8)]))
        self.register_buffer("min_stats", empty_stats())
        self.register_buffer("max_stats", empty_stats())
        self.register_buffer("mean_stats", empty_stats())
        self.register_buffer("std_stats", empty_stats())
        self.inference_mode = mode

    def train(self, mode=True):
        if mode:
            self.mode = "update"
        else:
            self.mode = self.inference_mode

    def alert(self, message):
        if self.error:
            raise ValueError(message)
        else:
            warnings.warn(message)

    def __len__(self):
        return int(self.min_stats[2].cpu().detach().item())

    def value(self, stats, x, message):
        if self.mode == "update":
            update_stats(stats, x)
            return
        if len(self) < 2:
            return
        if self.mode == "nocheck":
            return
        elif self.mode == "check_range":
            if not check_range(stats, x):
                self.alert(f"{message}: range error, {x} not in range {stats[0]}, {stats[1]}")
        elif self.mode == "check_std":
            if not check_sigma(stats, x):
                self.alert(f"{message}: {x} is outside 4 sigma of input value")
        elif isinstance(self.mode, (float, int)):
            delta = (stats[1] - stats[0]) * self.mode
            if not check_range([stats[0] - delta, stats[1] + delta], x):
                self.alert(f"{message}: range error, {x} not within {delta} of {stats[0]}, {stats[1]}")
        else:
            raise ValueError(f"{self.mode}: unknown check mode")

    def forward(self, a):
        for i in range(min(a.ndim, len(self.dim_stats))):
            self.value(self.dim_stats[i], a.shape[i], f"dim({i})")
        self.value(self.min_stats, a.detach().min().cpu().item(), "min value")
        self.value(self.max_stats, a.detach().max().cpu().item(), "max value")
        self.value(self.mean_stats, a.detach().mean().cpu().item(), "mean value")
        self.value(self.std_stats, a.detach().std().cpu().item(), "std value")
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
