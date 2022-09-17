#
# Copyright (c) 2017-2019 TBD. All rights reserved.
# This file is part of TBD (see TBD).
# See the LICENSE file for licensing terms (TBD).
#

import sys
import numpy as np
import torch
from torch import Tensor
from torch import autograd, nn
from torch.nn import functional as F
import warnings
from typing import Callable, Tuple, List, Type
from .utils import *


def conform_tensors1(a: Tensor, args: List[Tensor], slop: int = 9999, dims: List[int] = []):
    """Trim the remaining args down to the size of the first."""
    if len(dims) == 0:
        dims = np.arange(a.ndimension())
    if slop is not None:
        # FIXME convert to torch
        target = np.array(a.shape)
        sizes = np.array([a.shape for a in args])
        deltas = np.amax(np.abs(sizes - target[np.newaxis, :]), 0)
        for i in dims:
            assert deltas[i] <= slop, (sizes, deltas)
    box = tuple(slice(j) if i in dims else slice(None) for i, j in enumerate(a.shape))
    return tuple([a[box]] + [arg[box] for arg in args])


def conform1(a, *args, slop=9999, dims=[]):
    return conform_tensors1(a, args, slop=slop, dims=dims)


def conform_tensors(args: List[Tensor], slop: int = 9999, dims: List[int] = []):
    """Trim all args to the size of the smallest arg."""
    if slop is not None:
        # FIXME convert to torch
        sizes = np.array([a.shape for a in args])
        deltas = np.amax(sizes, 0) - np.amin(sizes, 0)
        for i in dims:
            assert deltas[i] <= slop, (sizes, deltas)
    box = map(min, zip(*[a.shape for a in args]))
    box1 = tuple(slice(j) if i in dims else slice(None) for i, j in enumerate(box))
    return tuple([arg[box1] for arg in args])


def conform(*args, slop=9999, dims=[]):
    return conform_tensors(args, slop=slop, dims=dims)


def reorder(x: Tensor, old: str, new: str, set_order: bool = True):
    """Reorder dimensions according to strings.

    E.g., reorder(x, "BLD", "LBD")
    """
    assert isinstance(old, str) and isinstance(new, str)
    for c in old:
        assert new.find(c) >= 0
    for c in new:
        assert old.find(c) >= 0
    permutation = [old.find(c) for c in new]
    assert len(old) == x.ndim, (old, x.size())
    result = x.permute(permutation).contiguous()
    return result


@DEPRECATED
def check_order(x: Tensor, order: str):
    # DEPRECATED
    pass


class WeightedGradFunction(autograd.Function):
    """Reweight the gradient using the given weights."""

    @staticmethod
    def forward(self, input, weights):
        raise Exception("WeightedGradFunction has been deprecated")
        self.weights = weights
        return input

    @staticmethod
    def backward(self, grad_output):
        assert grad_output.shape == self.weights.shape, (
            grad_output.shape,
            self.weights.shape,
        )
        return grad_output * self.weights, None


weighted_grad = WeightedGradFunction.apply


@deprecated
class Fun(nn.Module):
    """Turn an arbitrary function into a layer."""

    def __init__(self, f: str, info=None):
        super().__init__()
        assert isinstance(f, str), type(f)
        self.f = eval(f)
        self.f_str = f
        self.info = info

    def __getnewargs__(self):
        return (self.f_str, self.info)

    def forward(self, x):
        return self.f(x)

    def __repr__(self):
        return "Fun {} {}".format(self.info, self.f_str)


@deprecated
class Fun_(nn.Module):
    """Turn an arbitrary function into a layer."""

    def __init__(self, f: Callable, info=None):
        super().__init__()
        assert callable(f)
        self.f = f
        self.info = info

    def forward(self, x):
        return self.f(x)

    def __repr__(self):
        return "Fun {} {}".format(self.info, self.f)


class Info(nn.Module):
    """Output information for the given input."""

    def __init__(self, info="", every=1000000):
        super().__init__()
        self.info = info
        self.count = 0
        self.every = every

    def forward(self, x: Tensor):
        if self.count % self.every == 0:
            print(("Info", self.info, x.size(), x.min().item(), x.max().item()))
        self.count += 1
        return x

    def __repr__(self):
        return "Info {} ({} % {})".format(self.info, self.count, self.every)


class CheckSizes(nn.Module):
    """Check tensor sizes against the given sizes.

    Specify ranges as integers (exact), tuples (low, high), or -1 (no check).
    """

    def __init__(self, *args, **kw):
        super().__init__()
        self.order = kw.get("order")
        self.name = kw.get("name", "")
        self.limits = [(x, x) if isinstance(x, int) else x for x in args]

    def forward(self, x: Tensor):
        for i in range(x.ndim):
            lo, hi = self.limits[i]
            actual = x.shape[i]
            if lo >= 0 and actual < lo:
                raise Exception(
                    "{} ({}): index {} too low ({} not >= {})".format(self.name, self.order, i, actual, lo)
                )
            if hi >= 0 and actual > hi:
                raise Exception(
                    "{} ({}): index {} too high ({} not <= {})".format(self.name, self.order, i, actual, hi)
                )
        return x

    def __repr__(self):
        return "CheckSizes({})".format(", ".join([repr(x) for x in self.limits]))


class Device(nn.Module):
    """Always transfer tensor to device."""

    def __init__(self, device="cuda"):
        super().__init__()
        self.device = device

    def forward(self, x: Tensor):
        return x.to(self.device)

    def __repr__(self):
        return "Device {}".format(self.device)


class CheckRange(nn.Module):
    """Check that values are within the given range."""

    def __init__(self, lo: float = -1e5, hi: float = 1e5, name=""):
        super().__init__()
        self.name = name
        self.valid = (lo, hi)

    def forward(self, x: Tensor):
        assert x.min().item() >= self.valid[0], (self.name, x.min().item(), self.valid)
        assert x.max().item() <= self.valid[1], (self.name, x.max().item(), self.valid)
        return x

    def __repr__(self):
        return 'CheckRange({}, {}, name="{}")'.format(self.valid[0], self.valid[1], self.name)


@deprecated
class Input(nn.Module):
    assume: str
    reorder: str
    range: Tuple[float, float]
    size: List[Tuple[int, int]]
    device: str
    dtype: str

    def __init__(
        self,
        assume: str,
        reorder: str = "",
        range: Tuple[float, float] = (-1e5, 1e5),
        sizes: List[Tuple[int, int]] = [],
        device: str = "",
        dtype: str = "float",
    ):
        """Declares the input for a network.

        :param order: order of axes (e.g., BDL, BHWD, etc.)
        :param dtype: dtype to convert to
        :param range: tuple giving low/high values
        :param assume: default input order (when tensor doesn't have order attribute; None=required)
        """
        super().__init__()
        assert reorder == ""
        assert dtype == "float"
        self.assume = assume
        self.device = device
        self.dtype = dtype
        self.range = range
        self.param = torch.nn.Parameter(torch.zeros(1))
        self.sizes = sizes

    def forward(self, x):
        if self.range is not None:
            lo = x.min().item()
            hi = x.max().item()
            assert lo >= self.range[0] and hi <= self.range[1], (lo, hi, self.range)
        if self.sizes is not None:
            assert len(self.sizes) == x.ndim, f"Input expects tensor of rank {len(self.sizes)} got {x.ndim}"
            for i, size in enumerate(self.sizes):
                if size is None:
                    continue
                elif isinstance(size, int):
                    assert x.size(i) == size, f"Input dim {i}: expected {size}, got {x.size(i)} ({x.shape})"
                elif isinstance(size, (list, tuple)):
                    lo, hi = size
                    assert (
                        x.size(i) >= lo and x.size(i) <= hi
                    ), f"Input dim {i}: expected {(lo, hi)}, got {x.size(i)} ({x.shape})"
                else:
                    raise ValueError("bad size spec")
        if self.device == "":
            x = x.to(device=self.param.device, dtype=torch.float32)
        else:
            x = x.to(device=self.device, dtype=torch.float32)
        return x

    def __repr__(self):
        autodev = self.param.device if self.device else None
        return f"Input({self.assume} " + f"{self.dtype} {self.range} {autodev} {self.sizes})"


class Reorder(nn.Module):
    """Reorder the dimensions using the given arguments."""

    def __init__(self, old, new):
        self.old = old
        self.new = new
        super().__init__()
        assert isinstance(old, str), old
        assert isinstance(new, str), new
        assert set(old) == set(new), (old, new)
        self.permutation = tuple([old.find(c) for c in new])

    def forward(self, x):
        return x.permute(*self.permutation).contiguous()

    def __repr__(self):
        return 'Reorder("{}", "{}")'.format(self.old, self.new)


@DEPRECATED
class CheckOrder(nn.Module):
    def __init__(self, order=None):
        super().__init__()
        self.order = order

    def forward(self, x):
        if self.order is not None:
            check_order(x, self.order)
        return x


def trim_rest(args: List[int]) -> List[int]:
    while len(args) > 0 and args[-1] == -1:
        args.pop()
    return args


class Permute(nn.Module):
    """Permute the dimensions of the input tensor."""

    def __init__(
        self,
        d0: int,
        d1: int = -1,
        d2: int = -1,
        d3: int = -1,
        d4: int = -1,
        d5: int = -1,
    ):
        super().__init__()
        args = trim_rest([d0, d1, d2, d3, d4, d5])
        self.permutation = args

    def forward(self, x):
        return x.permute(self.permutation).contiguous()

    def __repr__(self):
        return "Permute({})".format(", ".join(list(self.permutation)))


class Reshape(nn.Module):
    """Reshape an input tensor.

    Shapes can be specified as a list of integers and tuples.
    If specified as tuple, the corresponding input dimensions
    are collapsed into a single output dimension.
    """

    def __init__(
        self,
        d0: int,
        d1: int = -1,
        d2: int = -1,
        d3: int = -1,
        d4: int = -1,
        d5: int = -1,
    ):
        super().__init__()
        self.shape = trim_rest([d0, d1, d2, d3, d4, d5])

    def forward(self, x):
        newshape = []
        for s in self.shape:
            if isinstance(s, int):
                newshape.append(int(x.size(s)))
            elif isinstance(s, (tuple, list)):
                total = 1
                for j in s:
                    total *= int(x.size(j))
                newshape.append(total)
            else:
                raise ValueError("shape spec must be either int or tuple, got {}".format(s))
        return x.view(*newshape)

    def __repr__(self):
        return "Reshape({})".format(", ".join([repr(x) for x in self.shape]))


class Collapse(nn.Module):
    """Reshape an input tensor.

    Collapse a range of dimensions.
    """

    start: int
    end: int

    def __init__(self, start: int, end: int):
        super().__init__()
        self.start = start
        self.end = end

    def forward(self, x: Tensor):
        newshape = [x.shape[i] for i in range(0, self.start)]
        d = 1
        for i in range(self.start, self.end + 1):
            d *= x.shape[i]
        newshape += [d]
        newshape += [x.shape[i] for i in range(self.end + 1, len(x.shape))]
        if len(newshape) == 1:
            return x.view(newshape[0])
        if len(newshape) == 2:
            return x.view(newshape[0], newshape[1])
        if len(newshape) == 3:
            return x.view(newshape[0], newshape[1], newshape[2])
        if len(newshape) == 4:
            return x.view(newshape[0], newshape[1], newshape[2], newshape[3])
        raise ValueError("Collapse: unknown shape {newshape}")

    def __repr__(self):
        return "Collapse({}, {})".format(self.start, self.end)


class Viewer(nn.Module):
    """Module equivalent of x.view(*args)"""

    def __init__(
        self,
        d0: int,
        d1: int = -1,
        d2: int = -1,
        d3: int = -1,
        d4: int = -1,
        d5: int = -1,
    ):
        super().__init__()
        self.shape = trim_rest([d0, d1, d2, d3, d4, d5])

    def forward(self, x):
        return x.view(self.shape)

    def __repr__(self):
        return "Viewer({})".format(", ".join([repr(x) for x in self.shape]))


class LSTM(nn.Module):
    """LSTM wrapper that discards the state."""

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        bias=True,
        batch_firts=True,
        dropout=0,
        bidirectional=False,
        proj_size=0,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            bias=bias,
            batch_first=batch_firts,
            dropout=dropout,
            bidirectional=bidirectional,
        )

    def forward(self, x):
        return self.lstm(x)[0]


class BDL_LSTM(nn.Module):
    """A simple bidirectional LSTM.

    All the sequence processing layers use BDL order by default to
    be consistent with 1D convolutions.
    """

    def __init__(
        self,
        ninput: int = None,
        noutput: int = None,
        num_layers: int = 1,
        bidirectional: bool = False,
        batch_first: bool = True,
    ):
        super().__init__()
        assert ninput is not None
        assert noutput is not None
        self.lstm = nn.LSTM(ninput, noutput, num_layers=num_layers, bidirectional=bidirectional)

    def forward(self, seq: Tensor, volatile: bool = False, verbose: bool = False) -> Tensor:
        seq = reorder(seq, "BDL", "LBD")
        output, _ = self.lstm(seq)
        return reorder(output, "LBD", "BDL")


class BDHW_LSTM(nn.Module):
    """A 2D LSTM module.

    Input order as for 2D convolutions.
    """

    def __init__(
        self,
        ninput: int = None,
        noutput: int = None,
        nhidden: int = None,
        num_layers: int = 1,
        bidirectional: bool = True,
    ):
        super().__init__()
        nhidden = nhidden or noutput
        ndir = bidirectional + 1
        self.hlstm = nn.LSTM(ninput, nhidden, num_layers=num_layers, bidirectional=bidirectional)
        self.vlstm = nn.LSTM(nhidden * ndir, noutput, num_layers=num_layers, bidirectional=bidirectional)

    def forward(self, img: Tensor) -> Tensor:
        b, d, h, w = img.shape
        hin = reorder(img, "BDHW", "WHBD").view(w, h * b, d)
        hout, _ = self.hlstm(hin)
        vin = reorder(hout.view(w, h, b, -1), "WHBD", "HWBD").view(h, w * b, -1)
        vout, _ = self.vlstm(vin)
        return reorder(vout.view(h, w, b, -1), "HWBD", "BDHW")


class BDHW_LSTM_to_BDH(nn.Module):
    """An LSTM that summarizes 2D down to 1D along the last dim."""

    def __init__(self, ninput=None, noutput=None):
        super().__init__()
        assert ninput is not None
        assert noutput is not None
        self.lstm = nn.LSTM(ninput, noutput, 1, bidirectional=False)

    def forward(self, img, volatile: bool = False):
        noutput = self.lstm.hidden_size
        b, d, h, w = img.size()
        seq = reorder(img, "BDHW", "WBHD").view(w, b * h, d)
        out, (_, state) = self.lstm(seq)
        assert state.size() == (1, b * h, noutput), ((w, b * h, noutput), state.size())
        return reorder(state.view(b, h, noutput), "BHD", "BDH")


# Wrap-Around Modules


class NoopSub(nn.Module):
    """Noop wrap-around module (for testing/exploration)."""

    def __init__(self, *args, sub=None, **kw):
        super().__init__()
        self.sub = sub

    def forward(self, x: Tensor) -> Tensor:
        return self.sub(x)


class KeepSize(nn.Module):
    """Run layers, then upsample back to the original."""

    dims: List[int]

    def __init__(self, mode: str = "bilinear", sub=None, dims: List[int] = []):
        super().__init__()
        self.sub = sub
        self.mode = mode
        self.dims = dims

    def forward(self, x):
        y = self.sub(x)
        if len(self.dims) == 0:
            size = x.size()[2:]
        else:
            size = [x.size(i) for i in self.dims]
        # kw = dict(align_corners=False) if self.mode != "nearest" else {}
        if self.mode != "nearest":
            return F.interpolate(y, size=size, mode=self.mode, align_corners=False)
        else:
            return F.interpolate(y, size=size, mode=self.mode)


class Additive(nn.Module):
    """Additive wrap-around module for Resnet-style architectures.

    :args: modules whose output is to be added
    :post: module to execute after everything has been added
    """

    def __init__(self, *args, post=None):
        super().__init__()
        self.sub = nn.ModuleList(args)
        self.post = None

    def forward(self, x):
        y = self.sub[0](x)
        for f in self.sub[1:]:
            y = y + f(x)
        if self.post is not None:
            y = self.post(y)
        return y


class Parallel(nn.Module):
    """Run modules in parallel and concatenate the results."""

    def __init__(self, *args, dim=1):
        super().__init__()
        self.args = args
        for i, arg in enumerate(args):
            if isinstance(arg, list):
                arg = nn.Sequential(*arg)
            self.add_module(str(i), arg)
        self.dim = dim

    def forward(self, x):
        results = [f(x) for f in self.args]
        return torch.cat(results, dim=self.dim)


class Shortcut(nn.Module):
    """Run modules in parallel and concatenate the results."""

    def __init__(self, *args, dim=1):
        super().__init__()
        self.block = nn.Sequential(*args)
        self.dim = dim

    def forward(self, x):
        y = self.block(x)
        assert x.shape[: self.dim] == y.shape[: self.dim], (x.shape, y.shape)
        assert x.shape[self.dim + 1 :] == y.shape[self.dim + 1 :], (x.shape, y.shape)
        return torch.cat([x, y], dim=self.dim)


class SimplePooling2d(nn.Module):
    """Perform max pooling/unpooling"""

    def __init__(self, sub, mp=2, **kw):
        super().__init__()
        self.pool = torch.nn.MaxPool2d(mp, return_indices=True, ceil_mode=True)
        self.sub = nn.Sequential(*sub) if isinstance(sub, list) else sub
        self.unpool = torch.nn.MaxUnpool2d(mp)

    def forward(self, x):
        y, indices = self.pool(x)
        z = self.sub(y)
        indices, z = conform1(indices, z, dims=[0, 2, 3])
        assert z.shape == indices.shape, (z.shape, indices.shape)
        return self.unpool(z, indices)

    def __repr__(self):
        return "Pooling2d(\n" + repr(self.sub) + "\n)"


class AcrossPooling2d(nn.Module):
    """Perform max pooling/unpooling with across accumulation."""

    def __init__(self, sub, across, mp=2, **kw):
        super().__init__()
        self.pool = torch.nn.MaxPool2d(mp, return_indices=True, ceil_mode=True)
        self.sub = nn.Sequential(*sub) if isinstance(sub, list) else sub
        self.across = nn.Sequential(*across) if isinstance(across, list) else across
        self.unpool = torch.nn.MaxUnpool2d(mp)

    def forward(self, x):
        y, indices = self.pool(x)
        z = self.sub(y)
        indices, z = conform1(indices, z, slop=2, dims=[0, 2, 3])
        up = self.unpool(z, indices)
        across = self.across(x)
        up, across = conform(up, across, slop=2, dims=[0, 2, 3])
        return torch.cat([across, up], dim=1)


class ModPad(nn.Module):
    def __init__(self, mod=8):
        super().__init__()
        self.mod = mod

    def __str__(self):
        return f"ModPad({self.mod})"

    def __repr__(self):
        return f"ModPad({self.mod})"

    def forward(self, a):
        mod = self.mod
        bs, d, h, w = a.shape
        nh = ((h + mod - 1) // mod) * mod
        nw = ((w + mod - 1) // mod) * mod
        result = nn.functional.pad(a, (0, nw - w, 0, nh - h))
        # print(a.shape, result.shape, file=sys.stderr)
        nbs, nd, nh, nw = result.shape
        assert nh % mod == 0 and nw % mod == 0
        assert nbs == bs and nd == d and nh >= h and nw >= w
        return result


class ModPadded(nn.Module):
    def __init__(self, mod, sub):
        super().__init__()
        self.mod = mod
        self.sub = sub

    def forward(self, a):
        mod = self.mod
        bs, d, h, w = a.shape
        nh = ((h + mod - 1) // mod) * mod
        nw = ((w + mod - 1) // mod) * mod
        input = nn.functional.pad(a, (0, nw - w, 0, nh - h))
        # print(a.shape, input.shape, file=sys.stderr)
        nbs, nd, nh, nw = input.shape
        assert nh % mod == 0 and nw % mod == 0
        assert nbs == bs and nd == d and nh >= h and nw >= w
        output = self.sub(input)
        assert output.ndim == 4
        assert output.shape[0] == bs
        return output[:, :, :h, :w]
