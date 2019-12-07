import sys
import numpy as np
import torch
from torch import autograd, nn
from torch.nn import functional as F
from . import layers, flex

def fc_block(sizes, batchnorm=True, nonlin=nn.ReLU, flatten=True):
    assert len(sizes) >= 1, sizes
    result = [nn.Flatten()]
    for i, s in enumerate(sizes[:-1]):
        result += [flex.Linear(s)]
        if batchnorm:
            result += [flex.BatchNorm()]
        result += [nonlin()]
    result += [flex.Linear(sizes[-1])]
    return result

def conv2d_block(d, r=3, mp=None, fmp=None, repeat=1, batchnorm=True, nonlin=nn.ReLU):
    """Generate a conv layer with batchnorm and optional maxpool."""
    result = []
    for i in range(repeat):
        result += [flex.Conv2d(d, r, padding=(r//2, r//2))]
        if batchnorm:
            result += [flex.BatchNorm2d()]
        result += [nonlin()]
    if fmp is not None:
        assert mp is None, (fmp, mp)
        result += [nn.FractionalMaxPool2d(3, output_ratio=fmp)]
    elif mp is not None:
        result += [nn.MaxPool2d(mp)]
    return result

class UnetLayer(nn.Module):
    """Resolution pyramid layer using convolutions and upscaling.
    """
    def __init__(self, d, r=3, sub=None, post=None):
        super().__init__()
        self.down = nn.MaxPool2d(2)
        self.up = flex.ConvTranspose2d(d, r, stride=2, padding=1, output_padding=1)
        if isinstance(sub, list):
            sub = nn.Sequential(*sub)
        self.sub = sub
        self.post = post
    def forward(self, x):
        b, d, h, w = x.size()
        assert h%2==0 and w%2==0, x.size()
        lo = self.down(x)
        lo1 = self.sub(lo)
        hi = self.up(lo1)
        result = torch.cat([x, hi], dim=1)
        if self.post is not None:
            result = self.post(result)
        return result

def ResnetBottleneck(d,  b, r=3, identity=None, post=None):
    return layers.Additive(
        identity or nn.Identity(),
        nn.Sequential(
            nn.Conv2d(d, b, 1),
            nn.BatchNorm2d(b),
            nn.ReLU(),
            nn.Conv2d(b, b, r, padding=r//2),
            nn.BatchNorm2d(b),
            nn.ReLU(),
            nn.Conv2d(b, d, 1)
        ),
        post = post or nn.BatchNorm2d(d)
    )

def ResnetBlock(d, r=3, identity=None, post=None):
    return layers.Additive(
        identity or nn.Identity(),
        nn.Sequential(
            nn.Conv2d(d, d, r, padding=r//2),
            nn.BatchNorm2d(d),
            nn.ReLU(),
            nn.Conv2d(d, d, r, padding=r//2),
            nn.BatchNorm2d(d)
        ),
        post = post or nn.BatchNorm2d(d)
    )

def resnet_blocks(n, d, r=3):
    return [ResnetBlock(d, r) for _ in range(n)]

def make_unet(sizes, r=3, repeat=3, sub=None):
    if len(sizes)==1:
        if sub is None:
            return nn.Sequential(*conv2d_block(sizes[0], r, repeat=repeat))
        else:
            return UnetLayer(sizes[0], sub=sub)
    else:
        return UnetLayer(sizes[0],
            sub=make_unet(sizes[1:], r=r, repeat=repeat, sub=sub)
        )
    

