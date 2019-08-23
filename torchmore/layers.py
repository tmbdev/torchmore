#
# Copyright (c) 2017-2019 TBD. All rights reserved.
# This file is part of TBD (see TBD).
# See the LICENSE file for licensing terms (TBD).
#

import numpy as np
import torch
from torch import autograd, nn

BD = "BD"
LBD = "LBD"
LDB = "LDB"
BDL = "BDL"
BLD = "BLD"
BWHD = "BWHD"
BDWH = "BDWH"
BWH = "BWH"


def deprecated(f):
    def g(*args, **kw):
        raise Exception("deprecated")
    return g


def lbd2bdl(x):
    assert len(x.size()) == 3
    return x.permute(1, 2, 0).contiguous()


def bdl2lbd(x):
    assert len(x.size()) == 3
    return x.permute(2, 0, 1).contiguous()


def data(x):
    if isinstance(x, Variable):
        return x.data
    else:
        return x


class Fun(nn.Module):
    """Turn an arbitrary function into a layer."""

    def __init__(self, f, info=None):
        nn.Module.__init__(self)
        assert isinstance(f, str)
        self.f = eval(f)
        self.f_str = f
        self.info = info

    def __getnewargs__(self):
        return (self.f_str, self.info)

    def forward(self, x):
        return self.f(x)

    def __repr__(self):
        return "Fun {} {}".format(self.info, self.f)


class PixelsToBatch(nn.Module):
    """Reshape an image batch so that the pixels are treated as separate samples."""

    def forward(self, x):
        b, d, h, w = x.size()
        return x.permute(0, 2, 3, 1).contiguous().view(b*h*w, d)


class WeightedGrad(autograd.Function):
    """Reweight the gradient using the given weights."""

    def forward(self, input, weights):
        self.weights = weights
        return input

    def backward(self, grad_output):
        return grad_output * self.weights, None


def weighted_grad(x, y):
    return WeightedGrad()(x, y)


class Info(nn.Module):
    """Output information for the given input."""

    def __init__(self, info="", every=1000000):
        nn.Module.__init__(self)
        self.info = info
        self.count = 0
        self.every = every

    def forward(self, x):
        if self.count % self.every == 0:
            print(("Info", self.info, x.size(),
                  x.min().item(), x.max().item()))
        return x

    def __repr__(self):
        return "Info {} ({} % {})".format(self.info, self.count, self.every)


class CheckSizes(nn.Module):
    """Check tensor sizes against the given sizes.

    Specify ranges as integers (exact), tuples (low, high), or -1 (no check).
    """

    def __init__(self, *args, **kw):
        nn.Module.__init__(self)
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
        nn.Module.__init__(self)
        self.device = device

    def forward(self, x):
        return x.to(self.device)

    def __repr__(self):
        return "Device {}".format(self.device)


class CheckRange(nn.Module):
    """Check that values are within the given range."""

    def __init__(self, lo=-1e5, hi=1e5, name=""):
        nn.Module.__init__(self)
        self.valid = (lo, hi)

    def forward(self, x):
        assert data(x).min().item(
        ) >= self.valid[0], (self.name, data(x).min(), self.valid)
        assert data(x).max().item(
        ) <= self.valid[1], (self.name, data(x).max(), self.valid)
        return x

    def __repr__(self):
        return "CheckRange({}, {}, name=\"{}\")".format(
                           self.lo,
                           self.hi,
                           self.name)


def reorder(x, old, new):
    assert isinstance(old, str) and isinstance(new, str)
    assert set(old)==set(new) and len(old)==len(new) and len(set(old))==len(old)
    permutation = tuple([old.find(c) for c in new])
    assert len(old) == x.ndimension()
    return x.permute(permutation).contiguous()


class Input(nn.Module):
    def __init__(self, ndim=None, order=None, dtype=None, range=None, device=True, assume=True):
        """Declares the input for a network.

        :param ndim: required input dimensions
        :param order: order of axes (e.g., BDL, BHWD, etc.)
        :param dtype: dtype to convert to
        :param range: tuple giving low/high values
        :param device: input device to move to (True = auto)
        :param assume: default input order (when tensor doesn't have order attribute; None=required)
        """
        super().__init__()
        self.ndim = ndim
        self.order = order
        self.dtype = dtype
        self.range = range
        self.device = device
        if device is True:
            self.param = torch.nn.Parameter(torch.zeros(1))
        self.assume = assume
    def forward(self, x):
        if self.ndim is not None:
            assert x.ndimension() == self.ndim
        if self.range is not None:
            assert x.min().item() >= self.range[0]
            assert x.max().item() <= self.range[1]
        if self.order is not None:
            if hasattr(x, "order"):
                x = reorder(x, x.order, self.order)
            else:
                if self.assume is True:
                    pass
                elif self.assume is None:
                    raise ValueError("input is required to have a .order property")
                else:
                    x = reorder(x, self.assume, self.order)
        if self.device is True:
            x = x.to(device=self.param.device, dtype=self.dtype)
        else:
            x = x.to(device=self.device, dtype=self.dtype)
        return x


class Reorder(nn.Module):
    """Reorder the dimensions using the given arguments."""

    def __init__(self, old, new):
        self.old = old
        self.new = new
        nn.Module.__init__(self)
        assert isinstance(old, str)
        assert isinstance(new, str)
        assert set(old) == set(new)
        self.permutation = tuple([old.find(c) for c in new])

    def forward(self, x):
        return x.permute(*self.permutation).contiguous()

    def __repr__(self):
        return 'Reorder("{}", "{}")'.format(self.old, self.new)


class Permute(nn.Module):
    """Permute the dimensions of the input tensor."""

    def __init__(self, *args):
        nn.Module.__init__(self)
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
        nn.Module.__init__(self)
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
        nn.Module.__init__(self)
        self.shape = args

    def forward(self, x):
        return x.view(*self.shape)

    def __repr__(self):
        return "Viewer({})".format(
            ", ".join([repr(x) for x in self.shape]))

def lstm_state(shape, seq):
    # FIXME--not needed anymore
    """Create hidden state for LSTM."""
    h0 = torch.zeros(shape, dtype=seq.dtype,
                     device=seq.device, requires_grad=False)
    c0 = torch.zeros(shape, dtype=seq.dtype,
                     device=seq.device, requires_grad=False)
    return h0, c0


class LSTM1BDL(nn.Module):
    """A simple bidirectional LSTM.

    All the sequence processing layers use BDL order by default to
    be consistent with 1D convolutions.
    """


    def __init__(self, ninput=None, noutput=None, num_layers=1, bidirectional=False, batch_first=True):
        nn.Module.__init__(self)
        assert ninput is not None
        assert noutput is not None
        self.lstm = nn.LSTM(ninput,
                            noutput,
                            num_layers=num_layers,
                            bidirectional=bidirectional)

    def forward(self, seq, volatile=False, verbose=False):
        ninput = self.lstm.input_size
        noutput = self.lstm.hidden_size
        # BDL -> LBD
        seq = bdl2lbd(seq)
        l, bs, d = seq.shape
        assert d==ninput
        # FIXME--not needed anymore
        sd = self.lstm.num_layers * (1 + bool(self.lstm.bidirectional))
        h0, c0 = lstm_state((sd, bs, noutput), seq)
        output, _ = self.lstm(seq, (h0, c0))
        return lbd2bdl(output)

    def __repr__(self):
        return "LSTM1({}, {}, bidir={})".format(
                      self.lstm.input_size,
                      self.lstm.hidden_size,
                      self.lstm.bidirectional)



class RowwiseLSTM(nn.Module):
    def __init__(self, ninput=None, noutput=None, bidirectional=True, num_layers=1):
        nn.Module.__init__(self)
        assert ninput is not None
        assert noutput is not None
        self.lstm = nn.LSTM(ninput,
                            noutput,
                            num_layers,
                            bidirectional=bidirectional)

    def forward(self, img):
        ninput = self.lstm.input_size
        noutput = self.lstm.hidden_size
        bidirectional = self.lstm.bidirectional
        ndir = bidirectional+1
        b, d, h, w = img.size()
        # BDHW -> WHBD -> WB'D
        seq = img.permute(3, 2, 0, 1).contiguous().view(w, h * b, d)
        # WB'D
        # FIXME--not needed anymore
        sd = self.lstm.num_layers * (1 + bool(self.lstm.bidirectional))
        h0, c0 = lstm_state((sd, h*b, noutput), seq)
        seqresult, _ = self.lstm(seq, (h0, c0))
        # WB'D' -> BD'HW
        result = seqresult.view(w, h, b, noutput*ndir).permute(2, 3, 1, 0)
        return result

    def __repr__(self):
        return "RowwiseLSTM({}, {}, ndir={})".format(
                      self.lstm.input_size,
                      self.lstm.hidden_size,
                      self.lstm.bidirectional)


class LSTM2BDHW(nn.Module):
    """A 2D LSTM module.

    Input order as for 2D convolutions.
    """

    def __init__(self, ninput=None, noutput=None, nhidden=None, bidirectional=True):
        nn.Module.__init__(self)
        nhidden = nhidden or noutput
        ndir = bidirectional+1
        self.hlstm = RowwiseLSTM(ninput, nhidden, bidirectional=bidirectional)
        self.vlstm = RowwiseLSTM(nhidden * ndir, noutput, bidirectional=bidirectional)

    def forward(self, img):
        horiz = self.hlstm(img)
        horizT = horiz.permute(0, 1, 3, 2).contiguous()
        vert = self.vlstm(horizT)
        vertT = vert.permute(0, 1, 3, 2).contiguous()
        return vertT

    def __repr__(self):
        ndir = 1+self.hlstm.lstm.bidirectional+1
        return "LSTM2({}, {}, nhidden={}, bidirectional={})".format(
                      self.hlstm.lstm.input_size,
                      self.vlstm.lstm.hidden_size//ndir,
                      self.hlstm.lstm.hidden_size//ndir,
                      self.hlstm.lstm.bidirectional)

class FIXME_Pooling2d(nn.Module):
    """Perform max pooling/unpooling"""

    def __init__(self, *args, kernel_size=2, **kw):
        nn.Module.__init__(self)
        self.pool = torch.nn.MaxPool2d(kernel_size, return_indices=True, ceil_mode=True)
        self.f = args[0] if len(args)==1 else nn.Sequential(*args)
        self.unpool = torch.nn.MaxUnpool2d(kernel_size)

    def forward(self, x):
        y, indices = self.pool(x)
        z = self.f(y)
        d0, d1, d2, d3 = indices.shape
        #print(z.shape, indices.shape)
        return self.unpool(z[:d0,:d1,:d2,:d3], indices)

    def __repr__(self):
        return "Pooling2d(\n"+repr(self.f)+"\n)"

class Parallel(nn.Module):
    """Run modules in parallel and concatenate the results."""
    def __init__(self, *args, dim=1):
        nn.Module.__init__(self)
        self.args = args
        for i, arg in enumerate(args):
            self.add_module(str(i), arg)
        self.dim = dim

    def forward(self, x):
        results = [f(x) for f in self.args]
        #print([tuple(r.shape for r in results)])
        return torch.cat(results, dim=self.dim)

    def __repr__(self):
        return "Parallel("+repr(self.args)+")"


class FIXME_LSTM2to1(nn.Module):
    """An LSTM that summarizes one dimension."""
    input_order = BDWH
    output_order = BDL

    def __init__(self, ninput=None, noutput=None):
        nn.Module.__init__(self)
        self.ninput = ninput
        self.noutput = noutput
        self.lstm = nn.LSTM(ninput, noutput, 1, bidirectional=False)

    def forward(self, img, volatile=False):
        # BDWH -> HBWD -> HBsD
        b, d, w, h = img.size()
        seq = img.permute(3, 0, 2, 1).contiguous().view(h, b * w, d)
        bs = b * w
        h0, c0 = lstm_state((1, bs, self.noutput), img)

        # HBsD -> HBsD
        assert seq.size() == (h, b * w, d), (seq.size(), (h, b * w, d))
        post_lstm, _ = self.lstm(seq, (h0, c0))
        assert post_lstm.size() == (h, b * w, self.noutput), (post_lstm.size(),
                                                              (h, b * w, self.noutput))
        # HBsD -> BsD -> BWD
        final = post_lstm.select(0, h - 1).view(b, w, self.noutput)
        assert final.size() == (b, w, self.noutput), (final.size(), (b, w, self.noutput))
        # BWD -> BDW
        final = final.permute(0, 2, 1).contiguous()
        assert final.size() == (b, self.noutput, w), (final.size(),
                                                      (b, self.noutput, self.noutput))
        return final
    def __repr__(self):
        return "LSTM2to1({}, {})".format(
                      self.ninput,
                      self.noutput)


class FIXME_LSTM1to0(nn.Module):
    """An LSTM that summarizes one dimension."""
    input_order = BDL
    output_order = BD

    def __init__(self, ninput=None, noutput=None):
        nn.Module.__init__(self)
        self.ninput = ninput
        self.noutput = noutput
        self.lstm = nn.LSTM(ninput, noutput, 1, bidirectional=False)

    def forward(self, seq):
        volatile = not isinstance(seq, Variable) or seq.volatile
        seq = bdl2lbd(seq)
        l, b, d = seq.size()
        assert d == self.ninput, (d, self.ninput)
        h0, c0 = lstm_state((1, b, self.noutput), seq)
        assert seq.size() == (l, b, d)
        post_lstm, _ = self.lstm(seq, (h0, c0))
        assert post_lstm.size() == (l, b, self.noutput)
        final = post_lstm.select(0, l - 1).view(b, self.noutput)
        return final

    def __repr__(self):
        return "LSTM1to0({}, {})".format(
                      self.ninput,
                      self.noutput)

