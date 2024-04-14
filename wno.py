from cudnn_convolution import *
import scipy
import torch
import torch.nn as nn
import numpy as np
import pickle
import torch.nn.functional as F

from fno_utils import FNO2d, FNODatasetSingle

class WNO2d(FNO2d):
    def __init__(
            self, 
            num_channels, 
            method=CudnnConvFwdAlgo.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, 
            modes1=12, 
            modes2=12, 
            width=20, 
            initial_step=10):
        super(WNO2d, self).__init__(num_channels, modes1, modes2, width, initial_step)
        self.method = method

    def forward(self, x, grid):
        # x dim = [b, x1, x2, t*v]
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        # Pad tensor with boundary condition
        x = F.pad(x, [0, self.padding, 0, self.padding])

        x1 = self.conv0(x)
        x2 = cudnn_convolution_fwd(self.method, x.contiguous(), self.w0.weight, padding=1) + self.w0.bias[:, None, None]
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = cudnn_convolution_fwd(self.method, x.contiguous(), self.w1.weight, padding=1) + self.w1.bias[:, None, None]
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = cudnn_convolution_fwd(self.method, x.contiguous(), self.w2.weight, padding=1) + self.w2.bias[:, None, None]
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = cudnn_convolution_fwd(self.method, x.contiguous(), self.w3.weight, padding=1) + self.w3.bias[:, None, None]
        x = x1 + x2

        x = x[..., :-self.padding, :-self.padding] # Unpad the tensor
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)

        return x.unsqueeze(-2)
    
def test_correct(pde):
    train_data = FNODatasetSingle(filename=f"{pde}.hdf5")
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=1)

    initial_step = 1
    model_weights = torch.load(f"{pde}_FNO.pt")

    # HACK: augment baseline structure to allow for Winograd 3x3 and 5x5 convs to be tested
    weights_to_expand = ["w0", "w1", "w2", "w3"]
    for weight in weights_to_expand:
        weight_name = f"{weight}.weight"
        tmp = torch.zeros((20, 20, 3, 3))
        tmp[:,:,1,1] = model_weights["model_state_dict"][weight_name][:,:,0,0]
        model_weights["model_state_dict"][weight_name] = tmp.clone()

    method_to_name = {
        CudnnConvFwdAlgo.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM: "IMPLICIT_GEMM",
        CudnnConvFwdAlgo.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM: "IMPLICIT_PRECOMP_GEMM",
        CudnnConvFwdAlgo.CUDNN_CONVOLUTION_FWD_ALGO_GEMM: "GEMM",
        CudnnConvFwdAlgo.CUDNN_CONVOLUTION_FWD_ALGO_FFT: "FFT",
        CudnnConvFwdAlgo.CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING: "FFT_TILING",
        CudnnConvFwdAlgo.CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD: "WINOGRAD",
        CudnnConvFwdAlgo.CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED: "WINOGRAD_NONFUSED",
    }   

    fno = WNO2d(
        num_channels=1, 
        method=CudnnConvFwdAlgo.CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD, 
        modes1=12, 
        modes2=12, 
        width=20, 
        initial_step=initial_step).to("cuda")
    fno.load_state_dict(model_weights["model_state_dict"])

    device  = "cuda"
    t_train = 2  # this right?
    for xx, yy, grid in train_loader:
        # xx: input tensor (first few time steps) [b, x1, ..., xd, t_init, v]
        # yy: target tensor [b, x1, ..., xd, t, v]
        # grid: meshgrid [b, x1, ..., xd, dims]
        xx = xx.to(device)
        yy = yy.to(device)
        grid = grid.to(device)

        x    = xx[..., 0 , :]
        y    = yy[..., t_train-1:t_train, :]
        yhat = fno(x, grid)
        break

    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(y.cpu().detach().numpy()[0,:,:,0,0])
    ax2.imshow(yhat.cpu().detach().numpy()[0,:,:,0,0])
    plt.savefig("result.png")

if __name__ == "__main__":
    test_correct(pde="2D_DarcyFlow_beta10.0_Train")