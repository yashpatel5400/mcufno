import argparse
import scipy
import torch
import time
import torch.nn as nn
import pandas as pd
import numpy as np
import pickle
import yaml
import torch.nn.functional as F

from cudnn_convolution import *
from fno_utils import FNO2d, FNODatasetSingle

class WNO2d(FNO2d):
    def __init__(
            self, 
            num_channels, 
            method=CudnnConvFwdAlgo.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, 
            modes1=12, 
            modes2=12, 
            width=20, 
            initial_step=10, 
            kernel_size=3):
        super(WNO2d, self).__init__(num_channels, modes1, modes2, width, initial_step, kernel_size)
        self.method = method
        self.conv_padding = (kernel_size - 1) // 2

    def forward(self, x, grid):
        # x dim = [b, x1, x2, t*v]
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        # Pad tensor with boundary condition
        x = F.pad(x, [0, self.padding, 0, self.padding])

        x1 = self.conv0(x)
        if self.method == CudnnConvFwdAlgo.DEFAULT:
            x2 = F.conv2d(x.contiguous(), self.w0.weight, bias=self.w0.bias, padding=self.conv_padding)
        else:
            x2 = cudnn_convolution_fwd(self.method, x.contiguous(), self.w0.weight, padding=self.conv_padding) + self.w0.bias[:, None, None]
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        if self.method == CudnnConvFwdAlgo.DEFAULT:
            x2 = F.conv2d(x.contiguous(), self.w1.weight, bias=self.w1.bias, padding=self.conv_padding)
        else:
            x2 = cudnn_convolution_fwd(self.method, x.contiguous(), self.w1.weight, padding=self.conv_padding) + self.w1.bias[:, None, None]
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        if self.method == CudnnConvFwdAlgo.DEFAULT:
            x2 = F.conv2d(x.contiguous(), self.w2.weight, bias=self.w2.bias, padding=self.conv_padding)
        else:
            x2 = cudnn_convolution_fwd(self.method, x.contiguous(), self.w2.weight, padding=self.conv_padding) + self.w2.bias[:, None, None]
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        if self.method == CudnnConvFwdAlgo.DEFAULT:
            x2 = F.conv2d(x.contiguous(), self.w3.weight, bias=self.w3.bias, padding=self.conv_padding)
        else:
            x2 = cudnn_convolution_fwd(self.method, x.contiguous(), self.w3.weight, padding=self.conv_padding) + self.w3.bias[:, None, None]
        x = x1 + x2

        x = x[..., :-self.padding, :-self.padding] # Unpad the tensor
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)

        return x.unsqueeze(-2)
    
def time_method(cfg, method, kernel_size):
    if kernel_size not in [3,5]:
        print(f"Winograd convolutions only support kernels of size 3x3 or 5x5")
        return None

    pde_name = cfg["filename"].split(".h")[0]
    times = []
    num_trials = 10
    trial_size = 50
    train_data = FNODatasetSingle(filename=cfg["filename"])
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=trial_size)
    model_weights = torch.load(f"{pde_name}_FNO.pt")

    # HACK: augment baseline structure to allow for Winograd 3x3 and 5x5 convs to be tested
    weights_to_expand = ["w0", "w1", "w2", "w3"]
    for weight in weights_to_expand:
        weight_name = f"{weight}.weight"
        tmp = torch.zeros((20, 20, kernel_size, kernel_size))
        tmp[:,:,kernel_size // 2,kernel_size // 2] = model_weights["model_state_dict"][weight_name][:,:,0,0]
        model_weights["model_state_dict"][weight_name] = tmp.clone()

    for _ in range(num_trials):
        fno = WNO2d(
            num_channels=cfg["num_channels"], 
            method=method, 
            modes1=cfg["modes"], 
            modes2=cfg["modes"], 
            width=cfg["width"], 
            initial_step=cfg["initial_step"],
            kernel_size=kernel_size).to("cuda")
        fno.load_state_dict(model_weights["model_state_dict"])

        device = "cuda"
        start_time = time.time()
        for xxbatch, _, gridbatch in train_loader:
            if cfg["training_type"] == "autoregressive":
                inp_shape = list(xxbatch.shape)
                inp_shape = inp_shape[:-2]
                inp_shape.append(-1)
                
                xxbatch = xxbatch.reshape(inp_shape)
                for xidx in range(trial_size):
                    xx = xxbatch[xidx:xidx+1,...].to(device)
                    grid = gridbatch[xidx:xidx+1,...].to(device)
                    yhat = fno(xx, grid)
            else:
                for xidx in range(trial_size):
                    xx = xxbatch[xidx:xidx+1,...].to(device)
                    grid = gridbatch[xidx:xidx+1,...].to(device)
                    yhat = fno(xx[...,0,:], grid)
            break
        end_time = time.time()
        times.append((end_time - start_time) / trial_size)

    debug_plot = False
    if debug_plot:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(y.cpu().detach().numpy()[0,:,:,0,0])
        ax2.imshow(yhat.cpu().detach().numpy()[0,:,:,0,0])
        plt.savefig("result.png")
    return np.array(times)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pde", choices=["Darcy", "diff-react", "rdb"])
    parser.add_argument("--kernel")
    args = parser.parse_args()

    cfg_fn = f"config_{args.pde}.yaml"
    with open(cfg_fn, "r") as f:
        cfg = yaml.safe_load(f)

    method_to_name = {
        CudnnConvFwdAlgo.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM: "IMPLICIT_GEMM",
        CudnnConvFwdAlgo.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM: "IMPLICIT_PRECOMP_GEMM",
        CudnnConvFwdAlgo.CUDNN_CONVOLUTION_FWD_ALGO_GEMM: "GEMM",
        CudnnConvFwdAlgo.CUDNN_CONVOLUTION_FWD_ALGO_FFT: "FFT",
        CudnnConvFwdAlgo.CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING: "FFT_TILING",
        CudnnConvFwdAlgo.CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD: "WINOGRAD",
        CudnnConvFwdAlgo.CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED: "WINOGRAD_NONFUSED",
        CudnnConvFwdAlgo.FASTEST: "FASTEST",
        CudnnConvFwdAlgo.DEFAULT: "DEFAULT",
    }

    # first call always takes longer (some startup cost), so ignore that for comparison
    method_to_time = {}
    for method in method_to_name:
        method_to_time[method_to_name[method]] = time_method(cfg, method, kernel_size=int(args.kernel))
    method_times_df = pd.DataFrame.from_dict(method_to_time)
    
    print(f"--- {args.pde} : {args.kernel} ---")
    for method_name in method_to_name.values():
        print(f"{method_name}: {np.around(method_times_df[method_name].mean(), 5)} ({np.around(method_times_df[method_name].std(), 5)})")