import torch
import numpy as np
from torch.utils.cpp_extension import load
from cudnn_convolution import *

# B, F, C = 256, 512, 128
# N, K, O = 32, 5, 32

input  = torch.from_numpy(np.array(range(25))).type(torch.float32).reshape((1, 1, 5, 5)).to('cuda')
weight = torch.ones((1, 1, 2, 2)).type(torch.float32).to('cuda')

output = cudnn_convolution_fwd(
  CudnnConvFwdAlgo.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
  input, weight, padding=1, verbose=True
)

print("Done!")
# # create dummy gradient w.r.t. the output
# grad_output = torch.zeros(128, 64, 14, 14).to('cuda')

# # compute the gradient w.r.t. the weights and input
# grad_weight = cudnn_convolution.convolution_backward_weight(input, weight.shape, grad_output, stride, padding, dilation, groups, False, False, False)
# grad_input  = cudnn_convolution.convolution_backward_input(input.shape, weight, grad_output, stride, padding, dilation, groups, False, False, False)

# print(grad_weight.shape)
# print(grad_input.shape)