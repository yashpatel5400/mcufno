import pickle

import scipy
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.cpp_extension import load
from cudnn_convolution import *

# B, F, C = 256, 512, 128
# N, K, O = 32, 5, 32

conv_method_names = {
  CudnnConvFwdAlgo.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM: "IMPLICIT_GEMM",
  CudnnConvFwdAlgo.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM: "IMPLICIT_PRECOMP_GEMM",
  CudnnConvFwdAlgo.CUDNN_CONVOLUTION_FWD_ALGO_GEMM: "GEMM",
  # CudnnConvFwdAlgo.CUDNN_CONVOLUTION_FWD_ALGO_DIRECT: "DIRECT", # not supported?
  CudnnConvFwdAlgo.CUDNN_CONVOLUTION_FWD_ALGO_FFT: "FFT",
  CudnnConvFwdAlgo.CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING: "FFT_TILING",
  CudnnConvFwdAlgo.CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD: "WINOGRAD",
  CudnnConvFwdAlgo.CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED: "WINOGRAD_NONFUSED",
}

input = torch.rand((1, 1, 5, 5)).to('cuda')
weight = torch.rand((1, 1, 3, 3)).to('cuda')

output = cudnn_convolution_fwd(CudnnConvFwdAlgo.CUDNN_CONVOLUTION_FWD_ALGO_DIRECT, input, weight)
torch_conv = F.conv2d(input, weight)

print(torch.sum((output - torch_conv) ** 2))