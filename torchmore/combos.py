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


# helpers for constructing networks


def resnet_blocks(n, d, r=3):
    return [ResnetBlock(d, r) for _ in range(n)]


def maybe(arg):
    return [arg] if arg is not None else []


def maybexp(arg):
    if isinstance(arg, nn.Sequential):
        return list(arg)
    elif isinstance(arg, (list, tuple)):
        return list(arg)
    elif arg is None:
        return []
    else:
        raise ValueError(f"{arg}: must be None, a list, or nn.Sequential")


def opt(condition, *args):
    if not condition:
        return []
    return list(args)


def ifelse(condition, model1, model2):
    if condition:
        if isinstance(model1, list):
            model1 = nn.Sequential(*model1)
        return model1
    else:
        if isinstance(model2, list):
            model2 = nn.Sequential(*model2)
        return model2


# Unet Architecture


def UnetLayer0(d, sub=None, post=None, dropout=0.0, leaky=0.0, instancenorm=False, relu=nn.ReLU()):
    result = nn.Sequential(
        flex.Conv2d(d, 3, padding=1),
        *maybe(relu),
        layers.Shortcut(
            nn.MaxPool2d(2),
            *maybexp(sub),
            flex.ConvTranspose2d(d, 3, stride=2, padding=1, output_padding=1)
        ),
        *maybexp(post),
    )
    return result


def UnetLayer1(d, sub=None, post=None, dropout=0.0, leaky=0.0, instancenorm=False):
    result = nn.Sequential(
        flex.Conv2d(d, 3, padding=1),
        *opt(instancenorm, flex.InstanceNorm2d()),
        ifelse(leaky == 0.0, nn.ReLU(), nn.LeakyReLU(leaky)),
        layers.Shortcut(
            nn.MaxPool2d(2),
            *maybexp(sub),
            flex.ConvTranspose2d(d, 3, stride=2, padding=1, output_padding=1),
            *opt(instancenorm, flex.InstanceNorm2d()),
        ),
        *opt(dropout > 0.0, nn.Dropout(dropout)),
        *maybexp(post),
    )
    return result


def make_unet(sizes, dropout=[0.0] * 100, mode=0, sub=None):
    if isinstance(dropout, float):
        dropout = [dropout] * len(sizes)
    make_layer = globals()[f"UnetLayer{mode}"]
    if len(sizes) == 1:
        return make_layer(sizes[0], sub=sub)
    else:
        subtree = make_unet(sizes[1:], dropout=dropout[1:], sub=sub)
        return make_layer(sizes[0], sub=subtree, dropout=dropout[0])
