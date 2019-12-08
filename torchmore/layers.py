#
# Copyright (c) 2017-2019 TBD. All rights reserved.
# This file is part of TBD (see TBD).
# See the LICENSE file for licensing terms (TBD).
#

import sys
import numpy as np
import torch
from torch import autograd, nn
from torch.nn import functional as F

def deprecated(f):
    def g(*args, **kw):
        raise Exception("deprecated")
    return g


def conform1(a, *args, slop=None, dims=True):
    """Trim the remaining args down to the size of the first."""
    if dims is True: dims = arange(a.ndimension())
    if slop is not None:
        # FIXME convert to torch
        target = np.array(a.shape)
        sizes = np.array([a.shape for a in args])
        deltas = np.amax(np.abs(sizes-target[np.newaxis,:]), 0)
        for i in dims:
            assert deltas[i] <= slop, (sizes, deltas)
    box = tuple(slice(j) if i in dims else slice(None)
                for i,j in enumerate(a.shape))
    return tuple([a[box]] + [arg[box] for arg in args])


def conform(*args, slop=None, dims=True):
    """Trim all args to the size of the smallest arg."""
    if slop is not None:
        # FIXME convert to torch
        sizes = np.array([a.shape for a in args])
        deltas = np.amax(sizes, 0)-np.amin(sizes, 0)
        for i in dims:
            assert deltas[i] <= slop, (sizes, deltas)
    box = map(min, zip(*[a.shape for a in args]))
    box = tuple(slice(j) if i in dims else slice(None)
                for i,j in enumerate(box))
    return tuple([arg[box] for arg in args])


def reorder(x, old, new, set_order=True):
    """Reorder dimensions according to strings.

    E.g., reorder(x, "BLD", "LBD")
    """
    assert isinstance(old, str) and isinstance(new, str)
    assert set(old)==set(new) and len(old)==len(new) and len(set(old))==len(old), (old, new)
    permutation = tuple([old.find(c) for c in new])
    assert len(old) == x.ndimension(), (old, x.size())
    result = x.permute(permutation).contiguous()
    if set_order:
        result.order = new
    return result

def check_order(x, order):
    if hasattr(x, "order"):
        if x.order != order:
            raise ValueError(f"expected order {order}, got {x.order}")

class WeightedGrad(autograd.Function):
    """Reweight the gradient using the given weights."""

    def forward(self, input, weights):
        self.weights = weights
        return input

    def backward(self, grad_output):
        return grad_output * self.weights, None

def weighted_grad(x, y):
    return WeightedGrad()(x, y)


class Fun(nn.Module):
    """Turn an arbitrary function into a layer."""

    def __init__(self, f, info=None):
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

class Fun_(nn.Module):
    """Turn an arbitrary function into a layer."""

    def __init__(self, f, info=None):
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

    def forward(self, x):
        if self.count % self.every == 0:
            print(("Info", self.info, x.size(),
                  x.min().item(), x.max().item()))
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
        self.name = kw.get("name")
        self.limits = [(x, x) if isinstance(x, int) else x for x in args]

    def forward(self, x):
        for (i, actual), (lo, hi) in zip(enumerate(tuple(x.size())), self.limits):
            if lo >= 0 and actual < lo:
                raise Exception("{} ({}): index {} too low ({} not >= {})"
                                .format(self.name, self.order,
                                        i, actual, lo))
            if hi >= 0 and actual > hi:
                raise Exception("{} ({}): index {} too high ({} not <= {})"
                                .format(self.name, self.order,
                                        i, actual, hi))
        return x

    def __repr__(self):
        return "CheckSizes({})".format(
            ", ".join([repr(x) for x in self.limits]))


class Device(nn.Module):
    """Always transfer tensor to device."""

    def __init__(self, device="cuda"):
        super().__init__()
        self.device = device

    def forward(self, x):
        return x.to(self.device)

    def __repr__(self):
        return "Device {}".format(self.device)


class CheckRange(nn.Module):
    """Check that values are within the given range."""

    def __init__(self, lo=-1e5, hi=1e5, name=""):
        super().__init__()
        self.valid = (lo, hi)

    def forward(self, x):
        assert x.min().item() >= self.valid[0], \
            (self.name, data(x).min(), self.valid)
        assert x.max().item() <= self.valid[1], \
            (self.name, data(x).max(), self.valid)
        return x

    def __repr__(self):
        return "CheckRange({}, {}, name=\"{}\")".format(
                           self.valid[0],
                           self.valid[1],
                           self.name)


class Input(nn.Module):
    def __init__(self, assume, reorder=None, range=None, sizes=None, device=True, dtype=torch.float32):
        """Declares the input for a network.

        :param order: order of axes (e.g., BDL, BHWD, etc.)
        :param dtype: dtype to convert to
        :param range: tuple giving low/high values
        :param device: input device to move to (True = auto)
        :param assume: default input order (when tensor doesn't have order attribute; None=required)
        """
        super().__init__()
        self.assume = assume
        self.reorder = reorder if reorder is not None else assume
        self.dtype = dtype
        self.range = range
        self.device = device
        self.param = torch.nn.Parameter(torch.zeros(1))
        self.sizes = sizes
    def forward(self, x):
        if self.range is not None:
            lo = x.min().item()
            hi = x.max().item()
            assert lo >= self.range[0] and hi <= self.range[1], (lo, hi, self.range)
        if self.reorder is not None:
            if hasattr(x, "order"):
                x = reorder(x, x.order, self.reorder)
            else:
                if self.assume is True or self.assume==self.reorder:
                    pass
                elif self.assume is None:
                    raise ValueError("input is required to have a .order property")
                else:
                    x = reorder(x, self.assume, self.reorder)
        if self.sizes is not None:
            for i, size in enumerate(self.sizes):
                if size is None:
                    continue
                elif isinstance(size, int):
                    assert x.size(i) == size, (i, x.size(i))
                elif isinstance(size, (list, tuple)):
                    lo, hi = size
                    assert x.size(i) >= lo and x.size(i) <= hi, (i, x.size(i), (lo, hi))
                else:
                    raise ValueError("bad size spec")
        if self.device is True:
            x = x.to(device=self.param.device, dtype=self.dtype)
        else:
            x = x.type(self.dtype)
        return x
    def __repr__(self):
        autodev = self.param.device if self.device else None
        return f"Input({self.assume}->{self.reorder} " + \
            f"{self.dtype} {self.range} {autodev} {self.sizes})"


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

class CheckOrder(nn.Module):
    def __init__(self, order=None):
        super().__init__()
        self.order = order
    def forward(self, x):
        if self.order is not None:
            check_order(x, self.order)
        return x

class Permute(nn.Module):
    """Permute the dimensions of the input tensor."""

    def __init__(self, *args):
        super().__init__()
        self.permutation = args

    def forward(self, x):
        return x.permute(*self.permutation).contiguous()

    def __repr__(self):
        return "Permute({})".format(", ".join(list(self.permutation)))


class Reshape(nn.Module):
    """Reshape an input tensor.

    Shapes can be specified as a list of integers and tuples.
    If specified as tuple, the corresponding input dimensions
    are collapsed into a single output dimension.
    """

    def __init__(self, *args):
        super().__init__()
        self.shape = args

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
                raise ValueError(
                    "shape spec must be either int or tuple, got {}".format(s))
        return x.view(*newshape)

    def __repr__(self):
        return "Reshape({})".format(
            ", ".join([repr(x) for x in self.shape]))


class Viewer(nn.Module):
    """Module equivalent of x.view(*args)"""

    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(*self.shape)

    def __repr__(self):
        return "Viewer({})".format(
            ", ".join([repr(x) for x in self.shape]))

class LSTM(nn.Module):
    """LSTM wrapper that discards the state."""
    def __init__(self, *args, **kw):
        super().__init__()
        self.lstm = nn.LSTM(*args, **kw)

    def forward(self, *args, **kw):
        return self.lstm(*args, **kw)[0]

class BDL_LSTM(nn.Module):
    """A simple bidirectional LSTM.

    All the sequence processing layers use BDL order by default to
    be consistent with 1D convolutions.
    """

    def __init__(self, ninput=None, noutput=None, num_layers=1, bidirectional=False, batch_first=True):
        super().__init__()
        assert ninput is not None
        assert noutput is not None
        self.lstm = nn.LSTM(ninput, noutput, num_layers=num_layers,
                            bidirectional=bidirectional)

    def forward(self, seq, volatile=False, verbose=False):
        seq = reorder(seq, "BDL", "LBD")
        output, _ = self.lstm(seq)
        return reorder(output, "LBD", "BDL")


class BDHW_LSTM(nn.Module):
    """A 2D LSTM module.

    Input order as for 2D convolutions.
    """

    def __init__(self, ninput=None, noutput=None, nhidden=None, 
                 num_layers=1, bidirectional=True):
        super().__init__()
        nhidden = nhidden or noutput
        ndir = bidirectional+1
        self.hlstm = nn.LSTM(ninput, nhidden, num_layers=num_layers,
                             bidirectional=bidirectional)
        self.vlstm = nn.LSTM(nhidden*ndir, noutput, num_layers=num_layers,
                             bidirectional=bidirectional)

    def forward(self, img):
        b, d, h, w = img.shape
        hin = reorder(img, "BDHW", "WHBD").view(w, h*b, d)
        hout, _ = self.hlstm(hin)
        vin = reorder(hout.view(w, h, b, -1), "WHBD", "HWBD").view(h, w*b, -1)
        vout, _ = self.vlstm(vin)
        return reorder(vout.view(h, w, b, -1), "HWBD", "BDHW")



class BDHW_LSTM_to_BDH(nn.Module):
    """An LSTM that summarizes 2D down to 1D along the last dim."""
    def __init__(self, ninput=None, noutput=None):
        super().__init__()
        assert ninput is not None
        assert noutput is not None
        self.lstm = nn.LSTM(ninput, noutput, 1, bidirectional=False)

    def forward(self, img, volatile=False):
        noutput = self.lstm.hidden_size
        b, d, h, w = img.size()
        seq = reorder(img, "BDHW", "WBHD").view(w, b*h, d)
        out, (_, state) = self.lstm(seq)
        assert state.size() == (1, b*h, noutput), ((w, b*h, noutput), state.size())
        return reorder(state.view(b, h, noutput), "BHD", "BDH")

### Wrap-Around Modules

class NoopSub(nn.Module):
    """Noop wrap-around module (for testing/exploration)."""
    def __init__(self, *args, sub=None, **kw):
        super().__init__()
        self.sub = sub
    def forward(self, x):
        return self.sub(x)

class KeepSize(nn.Module):
    """Run layers, then upsample back to the original.
    """
    def __init__(self, mode="bilinear", sub=None, dims=None):
        super().__init__()
        if isinstance(sub, list):
            sub = nn.Sequential(*sub)
        self.sub = sub
        self.mode = mode
        self.dims = dims
    def forward(self, x):
        y = self.sub(x)
        if self.dims is None:
            size = x.size()[2:]
        else:
            size = [x.size(i) for i in self.dims]
        kw = dict(align_corners=False) if self.mode != "nearest" else {}
        try:
            return F.interpolate(y, size=size, mode=self.mode, **kw)
        except Exception as exn:
            print("error:", x.size(), y.size(), self.dims, size, self.mode, file=sys.stderr)
            raise exn

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
        return "Pooling2d(\n"+repr(self.sub)+"\n)"


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


