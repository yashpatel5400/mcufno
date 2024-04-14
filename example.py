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
  # CudnnConvFwdAlgo.CUDNN_CONVOLUTION_FWD_ALGO_DIRECT: "DIRECT",
  CudnnConvFwdAlgo.CUDNN_CONVOLUTION_FWD_ALGO_FFT: "FFT",
  CudnnConvFwdAlgo.CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING: "FFT_TILING",
  # CudnnConvFwdAlgo.CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD: "WINOGRAD",
  # CudnnConvFwdAlgo.CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED: "WINOGRAD_NONFUSED",
}

with open("tmp.pkl", "rb") as f:
    input, weight = pickle.load(f)

input = input.contiguous() # torch.from_numpy(input.cpu().detach().numpy().copy()).to('cuda')
# input = torch.rand((1, 20, 130, 130)).to('cuda')
# print(input)

output = cudnn_convolution_fwd(CudnnConvFwdAlgo.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, input, weight)
torch_conv = F.conv2d(input, weight)

# input_np   = input.squeeze().cpu().detach().numpy()
# weight_np  = weight.squeeze().cpu().detach().numpy()[::-1,::-1]
# output_van = scipy.signal.convolve2d(input_np, weight_np, mode="valid")

print(torch.sum((output - torch_conv) ** 2))
# print(output_van)

# # create dummy gradient w.r.t. the output
# grad_output = torch.zeros(128, 64, 14, 14).to('cuda')

# # compute the gradient w.r.t. the weights and input
# grad_weight = cudnn_convolution.convolution_backward_weight(input, weight.shape, grad_output, stride, padding, dilation, groups, False, False, False)
# grad_input  = cudnn_convolution.convolution_backward_input(input.shape, weight, grad_output, stride, padding, dilation, groups, False, False, False)

# print(grad_weight.shape)
# print(grad_input.shape)