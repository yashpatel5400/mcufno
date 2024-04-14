from winograd_2d_conv import winconv2d, conv2d, win_conv2d, direct_conv2d
import torch
import torch.nn as nn
import numpy as np

x = torch.randn(1, 64, 32, 32).cuda()
weight = torch.randn(64, 64, 3, 3).cuda()
bias = torch.randn(64).cuda()

tiles = 2

padding=1
stride=1

conv_op = direct_conv2d(x, weight, padding, stride, bias)
win_op = win_conv2d(x, weight, padding, tiles, stride, bias)

np.testing.assert_array_almost_equal(conv_op, 
                                    win_op, 
                                    decimal=5)