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
        return [arg]


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


def mayberelu(leaky):
    if leaky is None:
        return []
    elif leaky == 0.0:
        return [nn.ReLU()]
    else:
        return [nn.LeakyReLU(leaky)]

def maybedropout(d):
    if d <= 0.0:
        return []
    return [nn.Dropout(d)]


# Unet Architecture


def UnetLayer0(d, sub=None, post=None, dropout=0.0, leaky=0.0, instancenorm=False, relu=None):
    result = nn.Sequential(
        flex.Conv2d(d, 3, padding=1),
        *opt(instancenorm, flex.InstanceNorm2d()),
        *mayberelu(relu),
        layers.Shortcut(
            nn.MaxPool2d(2),
            *opt(dropout > 0, nn.Dropout2d(dropout)),
            *maybexp(sub),
            flex.ConvTranspose2d(d, 3, stride=2, padding=1, output_padding=1)
        ),
        *maybedropout(dropout),
        *maybexp(post),
    )
    return result


def UnetLayer1(d, sub=None, post=None, dropout=0.0, relu=0.2, instancenorm=True, instancenorm2=True, relu2=None):
    # Only instancenorm2 and relu2 are optional in the original
    result = nn.Sequential(
        layers.Shortcut(
            flex.Conv2d(d, kernel_size=4, stride=2, padding=1, bias=nn.InstanceNorm2d),
            *opt(instancenorm, flex.InstanceNorm2d()),
            *mayberelu(relu),
            *maybexp(sub),
            flex.ConvTranspose2d(d, kernel_size=4, stride=2, padding=1, bias=nn.InstanceNorm2d),
            *opt(instancenorm2, flex.InstanceNorm2d()),
            *mayberelu(relu2),
            *maybedropout(dropout),
        ),
        *maybexp(post),
    )
    return result


def UnetLayer2(d, sub=None, pre=2, post=2, dropout=0.0, leaky=0.0):
    prelayers = []
    for i in range(pre):
        prelayers += [flex.Conv2d(d, kernel_size=3, padding=1), nn.ReLU()]
    postlayers = []
    for i in range(post):
        postlayers += [flex.Conv2d(d, kernel_size=3, padding=1), nn.ReLU()]
    result = nn.Sequential(
        *prelayers,
        layers.Shortcut(
            nn.MaxPool2d(2),
            *maybexp(sub),
            flex.ConvTranspose2d(d, 3, stride=2, padding=1, output_padding=1),
            *maybedropout(dropout),
        ),
        *postlayers,
    )
    return result

def make_unet(sizes, dropout=[0.0] * 100, mode=2, sub=2, **kw):
    if isinstance(dropout, float):
        dropout = [dropout] * len(sizes)
    make_layer = globals()[f"UnetLayer{mode}"]
    if len(sizes) < 2:
        raise ValueError("Unet must have at least 2 layers")
    elif len(sizes) == 2:
        if isinstance(sub, int):
            sublayers = []
            for i in range(sub):
                sublayers += [flex.Conv2d(sizes[1], kernel_size=3, padding=1), nn.ReLU()]
            sub = sublayers
        return make_layer(sizes[0], sub=sub, **kw)
    else:
        subtree = make_unet(sizes[1:], dropout=dropout[1:], sub=sub, **kw)
        return make_layer(sizes[0], sub=subtree, dropout=dropout[0], **kw)
