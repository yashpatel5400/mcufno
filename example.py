import scipy
import torch
import numpy as np
from torch.utils.cpp_extension import load
from cudnn_convolution import *

# B, F, C = 256, 512, 128
# N, K, O = 32, 5, 32

input  = torch.from_numpy(np.array(range(25))).type(torch.float32).reshape((1, 1, 5, 5)).to('cuda')
weight = torch.ones((1, 1, 2, 2)).type(torch.float32).to('cuda')

conv_method_names = {
  CudnnConvFwdAlgo.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM: "IMPLICIT_GEMM",
  CudnnConvFwdAlgo.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM: "IMPLICIT_PRECOMP_GEMM",
  CudnnConvFwdAlgo.CUDNN_CONVOLUTION_FWD_ALGO_GEMM: "GEMM",
  # CudnnConvFwdAlgo.CUDNN_CONVOLUTION_FWD_ALGO_DIRECT: "DIRECT",
  CudnnConvFwdAlgo.CUDNN_CONVOLUTION_FWD_ALGO_FFT: "FFT",
  CudnnConvFwdAlgo.CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING: "FFT_TILING",
  # CudnnConvFwdAlgo.CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD: "WINOGRAD",
  # CudnnConvFwdAlgo.CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED: "WINOGRAD_NONFUSED",
}

for conv_method in conv_method_names:
  print(f"Performing: {conv_method_names[conv_method]}")
  output = cudnn_convolution_fwd(conv_method, input, weight, padding=1, verbose=True)
print(f"Done!")
# # create dummy gradient w.r.t. the output
# grad_output = torch.zeros(128, 64, 14, 14).to('cuda')

# # compute the gradient w.r.t. the weights and input
# grad_weight = cudnn_convolution.convolution_backward_weight(input, weight.shape, grad_output, stride, padding, dilation, groups, False, False, False)
# grad_input  = cudnn_convolution.convolution_backward_input(input.shape, weight, grad_output, stride, padding, dilation, groups, False, False, False)

# print(grad_weight.shape)
# print(grad_input.shape)