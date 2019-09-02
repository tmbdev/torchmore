# Torchmore

The `torchmore` library is a small library of layers and utilities
for writing PyTorch models for image recognition, OCR, and other applications.


# Flex


The `flex` library performs simple size inference. It does so by wrapping up individual layers in a wrapper that instantiates the layer only when dimensional data is available. The wrappers can be removed later and the model turned into one with only completely standard modules. That looks like this:

    from torch import nn
    from torchmore import layers, flex

    noutput = 10
    
    model = nn.Sequential(
        layers.Input("BDHW"),
        flex.Conv2d(100),
        flex.BatchNorm(),
        nn.ReLU(),
        flex.Conv2d(100),
        flex.BatchNorm(),
        nn.ReLU(),
        layers.Reshape([1, [2, 3, 4]]),
        flex.Full(100),
        flex.BatchNorm(),
        nn.ReLU(),
        flex.Full(noutput)
    )
  
    flex.shape_inference(model, (1, 1, 28, 28))



The `flex` library provides wrappers for the following layers right now:

- `Linear`
- `Conv1d`, `Conv2d`, `Conv3d`
- `ConvTranspose1d`, `ConvTranspose2d`, `ConvTranspose3d`
- `LSTM`, `BDL_LSTM`, `BDHW_LSTM`
- `BatchNorm1d`, `BatchNorm2d`, `BatchNorm3d`
- `BatchNorm`

You can use `Flex` directly. The following two layers are identical:

    layer1 = flex.Conv2d(100)
    layer2 = flex.Flex(lambda x: nn.Conv2d(x.size(1), 100))
    
That is, you can easily turn any layer into a `Flex` layer that way even if it isn't in the library.


# Layers


## layers.Input

The `Input` layer is a handy little layer that reorders input dimensions, checks size ranges and value ranges, and automatically transfers data to the current device on which the model runs.

For example, consider the following `Input` layer:

        layers.Input("BHWD", "BDHW", range=(0, 1), sizes=[None, 1, None, None]),

This says:

- the input is in "BHWD" order and will get reordered to "BDHW"
- input values must be in the interval $[0, 1]$
- input tensors must have $D=1$
- input tensors are transferred to the same device as weights for the model

## The `.order` Attribute

Note that if the input tensor has a `.order` attribute, that will be used to reorder the input dimensions into the desired dimensions. This allows the model to accept inputs in multiple orders. Consider

    model = nn.Sequential(
        layers.Input("BHWD", "BDHW", range=(0, 1), sizes=[None, 1, None, None]),
        ...
    )
    a = torch.rand((1, 100, 150, 1))
    b = a.permute(0, 3, 1, 2)
    b.order = "BDHW"
    
    assert model(a) == model(b)



# layers.Reorder

The `Reorder` layer reorders axes just like `Tensor.permute` does, but it does so in a way that documents better what is going on. Consider the following code fragment:

        layers.Reorder("BDL", "LBD"),
        flex.LSTM(100, bidirectional=True),
        layers.Reorder("LBD", "BDL"),
        flex.Conv1d(noutput, 1),
        layers.Reorder("BDL", "BLD")
        
The letters themselves are arbitrary, but common choices are "BDLHW". This is likely clearer than a sequence of permutations.


## layers.Fun

For module-based networks, it's convenient to add functions. The `Fun` layer permits that, as in:

        layers.Fun("lambda x: x.permute(2, 0, 1)")
        
Note that since functions are specified as strings, this can be pickled.


# LSTM layers

- `layers.LSTM`: a trivial LSTM layer that simply dicards the state output
- `layers.BDL_LSTM`: an LSTM variant that is a drop-in replacement for a `Conv1d` layer
- `layers.BDHW_LSTM`: an MDLSTM variant that is a drop-in replacement for a `Conv2d` layer
- `layers.BDHW_LSTM_to_BDH`: a rowwise LSTM, reducing dimension by 1



# Other Layers

These may be occasionally useful:

- `layers.Info(info="", every=1000000)`: prints info about the activations
- `layers.CheckSizes(...)`: checks the sizes of tensors propagated through
- `layers.CheckRange(...)`: checks the ranges of values
- `layers.Permute(...)`: axis permutation (like x.permute)
- `layers.Reshape(...)`: tensor reshaping, with the option of combining axes
- `layers.View(...)`: equivalent of x.view
- `layers.Parallel`: run two modules in parallel and stack the results
- `layers.SimplePooling2d`: wrapped up max pooling/unpooling
- `layers.AcrossPooling2d`: wrapped up max pooling/unpooling with convolution


```python

```
