import torch
from torch import nn
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


# common multi-layer convolution block (e.g., used in VGG models)


def conv2d_block(d, r=3, mp=None, fmp=None, repeat=1, batchnorm=True, nonlin=nn.ReLU):
    """Generate a conv layer with batchnorm and optional maxpool."""
    result = []
    for i in range(repeat):
        result += [flex.Conv2d(d, r, padding=(r // 2, r // 2))]
        if batchnorm:
            result += [flex.BatchNorm2d()]
        result += [nonlin()]
    if fmp is not None:
        assert mp is None, (fmp, mp)
        result += [nn.FractionalMaxPool2d(3, output_ratio=fmp)]
    elif mp is not None:
        result += [nn.MaxPool2d(mp)]
    return result


# Resnet Architecture


def ResnetBottleneck(d, b, r=3, identity=None, post=None):
    return layers.Additive(
        identity or nn.Identity(),
        nn.Sequential(
            nn.Conv2d(d, b, 1),
            nn.BatchNorm2d(b),
            nn.ReLU(),
            nn.Conv2d(b, b, r, padding=r // 2),
            nn.BatchNorm2d(b),
            nn.ReLU(),
            nn.Conv2d(b, d, 1),
        ),
        post=post or nn.BatchNorm2d(d),
    )


def ResnetBlock(d, r=3, identity=None, post=None):
    """Block for Resnet."""
    return layers.Additive(
        identity or nn.Identity(),
        nn.Sequential(
            nn.Conv2d(d, d, r, padding=r // 2),
            nn.BatchNorm2d(d),
            nn.ReLU(),
            nn.Conv2d(d, d, r, padding=r // 2),
            nn.BatchNorm2d(d),
        ),
        post=post or nn.BatchNorm2d(d),
    )


def resnet_blocks(n, d, r=3):
    return [ResnetBlock(d, r) for _ in range(n)]


# Unet Architecture


class UnetLayer(nn.Module):
    """Resolution pyramid layer using convolutions and upscaling.
    """

    def __init__(self, d, sub=None, post=None, dropout=0.0):
        super().__init__()
        self.conv = flex.Conv2d(d, 3, padding=1)
        self.down = nn.MaxPool2d(2)
        self.up = flex.ConvTranspose2d(d, 3, stride=2, padding=1, output_padding=1)
        if isinstance(sub, list):
            sub = nn.Sequential(*sub)
        self.sub = sub
        self.dropout = None if dropout <= 0.0 else nn.Dropout(dropout)
        self.post = post

    def forward(self, x):
        b, d, h, w = x.size()
        assert h % 2 == 0 and w % 2 == 0, x.size()
        xc = self.conv(x)
        lo = self.down(xc)
        if self.dropout is not None:
            lo = self.dropout(lo)
        lo1 = self.sub(lo)
        hi = self.up(lo1)
        result = torch.cat([xc, hi], dim=1)
        if self.post is not None:
            result = self.post(result)
        return result


def make_unet(sizes, sub=None, dropout=[0.0] * 100):
    if len(sizes) == 1:
        if sub is None:
            return nn.Sequential(*conv2d_block(sizes[0]))
        else:
            return UnetLayer(sizes[0], sub=sub)
    else:
        return UnetLayer(sizes[0], sub=make_unet(sizes[1:], sub=sub, dropout=dropout[1:]), dropout=dropout[0])
