# PyTorch cuDNN Convolution

PyTorch extension enabling direct access to the following cuDNN-accelerated C++ functions
that are included in PyTorch:

- `cudnn_convolution`
- `cudnn_convolution_backward_weight`
- `cudnn_convolution_backward_input`

The functions defined here can be called from Python in replacement of
`torch.nn.conv2d`, `torch.nn.grad.conv2d_weight` and `torch.nn.grad.conv2d_input`,
and run significantly faster. See **example.py** for how these functions
are called.

Adapted from the following code posted by *hanspinckaers*:

https://discuss.pytorch.org/t/cuda-error-with-cudnn-convolution-backward-weight-function/41214


# Winograd

References:

1. https://github.com/xuexinghua/Operation-level-FI/blob/main/layer/winograd_layers.py

2. https://github.com/adam-dziedzic/winograd

3. https://github.com/xingyul/sparse-winograd-cnn/tree/master/doc

4. Efficient Point Selection: https://arxiv.org/abs/2201.10369

Also see: [Speedups or not?](https://discuss.pytorch.org/t/why-does-winograd-algorithm-speedup-convolution-given-that-mul-and-add-cost-the-same-clock-cycles-on-gpu/89993)