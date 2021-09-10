import sys
import numpy as np
import torch
from torch import Tensor
from torch import autograd, nn
from torch.nn import functional as F
import warnings
from typing import Callable, Tuple, List, Type


def DEPRECATED(f: Callable):
    def g(*args, **kw):
        raise Exception("DEPRECATED")
        return f(*args, **kw)

    return g


def deprecated(f: Callable):
    def g(*args, **kw):
        warnings.warn("deprecated", DeprecationWarning)
        return f(*args, **kw)

    return g