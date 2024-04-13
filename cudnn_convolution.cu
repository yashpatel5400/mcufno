/**
 * The #include<ATen/cudnn/*.h> needs guards as pointed in
 * https://github.com/pytorch/pytorch/tree/master/aten/src/ATen/cudnn
 */
#include <ATen/cuda/CUDAConfig.h> // for the definition of AT_CUDNN_ENABLED
// #if AT_CUDNN_ENABLED()

#include <cudnn.h>
#include <torch/extension.h>
#include <ATen/cudnn/Handle.h> // for getCudnnHandle
#include "cudnn_utils.h"

/*
PyTorch extension enabling direct access to the following cuDNN-accelerated C++ functions
that are included in PyTorch:

    - cudnn_convolution
    - cudnn_convolution_backward_weight
    - cudnn_convolution_backward_input

The functions defined here can be called from Python in replacement of
torch.nn.conv2d, torch.nn.grad.conv2d_weight and torch.nn.grad.conv2d_input,
and run significantly faster. See 'example.py' for how these functions
are called.

Adapted from code posted by goldsborough/conv.cu:
https://gist.github.com/eduardo4jesus/33ef6d8696e8af70a3046e9f364a65f8#file-conv-cu
*/

#define CUDA_CALL(f) { \
  cudaError_t err = (f); \
  if (err != cudaSuccess) { \
    std::cout \
        << "    Error occurred: " << err << std::endl; \
    std::exit(1); \
  } \
}

#define CUDNN_CALL(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS)                      \
    {                                                        \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }

__global__ void dev_const(float *px, float k) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  px[tid] = k;
}

__global__ void dev_iota(float *px) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  px[tid] = tid;
}

std::ostream &cout_unit(std::ostream &out, const size_t &nbytes) {
  if ((nbytes / (1u<<30)) > 1)
    out << nbytes/(1u<<30) << " GB";
  else if ((nbytes / (1u<<20)) > 1)
    out << nbytes/(1u<<20) << " MB";
  else if ((nbytes / (1u<<10)) > 1)
    out << nbytes/(1u<<10) << " KB";

  out << " (" << nbytes << " Bytes)";
  return out;
}

void print(const float *data, int n, int c, int h, int w) {
  std::vector<float> buffer(1 << 20);
  CUDA_CALL(cudaMemcpy(
        buffer.data(), data,
        n * c * h * w * sizeof(float),
        cudaMemcpyDeviceToHost));
  int a = 0;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < c; ++j) {
      std::cout << "n=" << i << ", c=" << j << ":" << std::endl;
      for (int k = 0; k < h; ++k) {
        for (int l = 0; l < w; ++l) {
          std::cout << std::setw(4) << std::right << buffer[a];
          ++a;
        }
        std::cout << std::endl;
      }
    }
  }
  std::cout << std::endl;
}

at::Tensor convolution(const int fwdAlgo,
                       const at::Tensor &input, const at::Tensor &weight, const at::Tensor &output,
                       c10::ArrayRef<int64_t> stride, c10::ArrayRef<int64_t> padding,
                       c10::ArrayRef<int64_t> dilation, int64_t groups, bool verbose)
{
  std::cout << "updated" << std::endl;

  const cudnnHandle_t cudnn = at::native::getCudnnHandle();

  // input
  const int in_n = 1;
  const int in_c = 1;
  const int in_h = 5;
  const int in_w = 5;
  std::cout << "in_n: " << in_n << std::endl;
  std::cout << "in_c: " << in_c << std::endl;
  std::cout << "in_h: " << in_h << std::endl;
  std::cout << "in_w: " << in_w << std::endl;
  std::cout << std::endl;

  cudnnTensorDescriptor_t in_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(
        in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        in_n, in_c, in_h, in_w));

  float *in_data;
  CUDA_CALL(cudaMalloc(
        &in_data, in_n * in_c * in_h * in_w * sizeof(float)));

  // filter
  const int filt_k = 1;
  const int filt_c = 1;
  const int filt_h = 2;
  const int filt_w = 2;
  std::cout << "filt_k: " << filt_k << std::endl;
  std::cout << "filt_c: " << filt_c << std::endl;
  std::cout << "filt_h: " << filt_h << std::endl;
  std::cout << "filt_w: " << filt_w << std::endl;
  std::cout << std::endl;

  cudnnFilterDescriptor_t filt_desc;
  CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
  CUDNN_CALL(cudnnSetFilter4dDescriptor(
        filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
        filt_k, filt_c, filt_h, filt_w));

  float *filt_data;
  CUDA_CALL(cudaMalloc(
      &filt_data, filt_k * filt_c * filt_h * filt_w * sizeof(float)));

  // convolution
  const int pad_h = 1;
  const int pad_w = 1;
  const int str_h = 1;
  const int str_w = 1;
  const int dil_h = 1;
  const int dil_w = 1;
  std::cout << "pad_h: " << pad_h << std::endl;
  std::cout << "pad_w: " << pad_w << std::endl;
  std::cout << "str_h: " << str_h << std::endl;
  std::cout << "str_w: " << str_w << std::endl;
  std::cout << "dil_h: " << dil_h << std::endl;
  std::cout << "dil_w: " << dil_w << std::endl;
  std::cout << std::endl;

  cudnnConvolutionDescriptor_t conv_desc;
  CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
  CUDNN_CALL(cudnnSetConvolution2dDescriptor(
        conv_desc,
        pad_h, pad_w, str_h, str_w, dil_h, dil_w,
        CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT));

  // output
  int out_n;
  int out_c;
  int out_h;
  int out_w;
  
  CUDNN_CALL(cudnnGetConvolution2dForwardOutputDim(
        conv_desc, in_desc, filt_desc,
        &out_n, &out_c, &out_h, &out_w));

  std::cout << "out_n: " << out_n << std::endl;
  std::cout << "out_c: " << out_c << std::endl;
  std::cout << "out_h: " << out_h << std::endl;
  std::cout << "out_w: " << out_w << std::endl;
  std::cout << std::endl;

  cudnnTensorDescriptor_t out_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(
        out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        out_n, out_c, out_h, out_w));

  float *out_data;
  CUDA_CALL(cudaMalloc(
        &out_data, out_n * out_c * out_h * out_w * sizeof(float)));

  // algorithm
  // CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM         = 0 - Y
  // CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM = 1 - Y
  // CUDNN_CONVOLUTION_FWD_ALGO_GEMM                  = 2 - Y
  // CUDNN_CONVOLUTION_FWD_ALGO_DIRECT                = 3 - N
  // CUDNN_CONVOLUTION_FWD_ALGO_FFT                   = 4 - Y
  // CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING            = 5 - Y
  // CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD              = 6 - N
  // CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED     = 7 - N
  // CUDNN_CONVOLUTION_FWD_ALGO_COUNT                 = 8 - Y

  // CUDNN_CALL(cudnnGetConvolutionForwardAlgorithm(
  //       cudnn,
  //       in_desc, filt_desc, conv_desc, out_desc,
  //       CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo));

  cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_GEMM;
  std::cout << "Convolution algorithm: " << algo << std::endl;
  std::cout << std::endl;

  // workspace
  size_t ws_size;
  CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(
        cudnn, in_desc, filt_desc, conv_desc, out_desc, algo, &ws_size));

  float *ws_data;
  CUDA_CALL(cudaMalloc(&ws_data, ws_size));

  std::cout << "Workspace size: " << ws_size << std::endl;
  std::cout << std::endl;

  // results
  std::cout << "in_data:" << std::endl;
  print(in_data, in_n, in_c, in_h, in_w);
  
  std::cout << "filt_data:" << std::endl;
  print(filt_data, filt_k, filt_c, filt_h, filt_w);

  // perform
  float alpha = 1.f;
  float beta = 0.f;
  dev_iota<<<in_w * in_h, in_n * in_c>>>(in_data);
  dev_const<<<filt_w * filt_h, filt_k * filt_c>>>(filt_data, 1.f);
  CUDNN_CALL(cudnnConvolutionForward(
      cudnn,
      &alpha, in_desc, in_data, filt_desc, filt_data,
      conv_desc, algo, ws_data, ws_size,
      &beta, out_desc, out_data));

  
  std::cout << "out_data:" << std::endl;
  print(out_data, out_n, out_c, out_h, out_w);
}

at::Tensor convolution_backward_weight(const int bwdFilterAlgo,
                                       const at::Tensor &input, const at::Tensor &weight, const at::Tensor &output,
                                       c10::ArrayRef<int64_t> stride, c10::ArrayRef<int64_t> padding, 
                                       c10::ArrayRef<int64_t> dilation, int64_t groups, bool verbose)
{
  const cudnnHandle_t cudnn = at::native::getCudnnHandle();

  /*****************************************************************************
   * 1. Initializing Descriptors
   ****************************************************************************/
  cudnnDescriptors_t desc;
  initialize_descriptors(input, weight, output, stride, padding, dilation, desc);

  /*****************************************************************************
   * 2. Setting BWD Convolution Filter Algo
   ****************************************************************************/
  cudnnConvolutionBwdFilterAlgoPerf_t convolution_algorithm[CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT];
  int returnedAlgoCount;

  if (bwdFilterAlgo == -1)
  {
    if (verbose)
      std::cout << "Trying all" << std::endl;

    checkCUDNN(
        cudnnFindConvolutionBackwardFilterAlgorithm(/*handle*/ cudnn,
                                             /*xDesc*/ desc.input,
                                             /*dyDesc*/ desc.output,
                                             /*convDesc*/ desc.convolution,
                                             /*dwDesc*/ desc.weight,
                                             /*requestedAlgoCount*/ CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT,
                                             /*returnedAlgoCount*/ &returnedAlgoCount,
                                             /*perfResults*/ convolution_algorithm));
    if (verbose)
      for (int i = 0; i < returnedAlgoCount; i++)
        std::cout << convolution_algorithm[i] << std::endl;
  }
  else
  {
    convolution_algorithm[0].algo = static_cast<cudnnConvolutionBwdFilterAlgo_t>(bwdFilterAlgo);
    convolution_algorithm[0].status = static_cast<cudnnStatus_t>(0);
    convolution_algorithm[0].time = -1;
    convolution_algorithm[0].memory = 0;
    convolution_algorithm[0].determinism = static_cast<cudnnDeterminism_t>(-1);
    convolution_algorithm[0].mathType = static_cast<cudnnMathType_t>(0);
    if (verbose)
    {
      std::cout << "Attempt with defined Algo:" << std::endl;
      std::cout << convolution_algorithm[0] << std::endl;
    }
  }

  /*****************************************************************************
   * 3. Get and Allocate Memory for Workspace
   ****************************************************************************/
  if (verbose)
    std::cout << "Allocating Workspace" << std::endl;

  size_t workspace_bytes{0};
  checkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn,
                                                     /*xDesc*/ desc.input,
                                                     /*dyDesc*/ desc.output,
                                                     /*convDesc*/ desc.convolution,
                                                     /*dwDesc*/ desc.weight,
                                                     /*algo*/ convolution_algorithm[0].algo,
                                                     /*sizeInBytes*/ &workspace_bytes));

  /*****************************************************************************
   * 4. Get and Allocate Memory for Workspace
   ****************************************************************************/
  if (verbose) {
    std::cout << "Workspace size: ";
    cout_unit(std::cout, workspace_bytes) << std::endl;
  }

  void *d_workspace{nullptr};
  cudaMalloc(&d_workspace, workspace_bytes);

  if (verbose) {
    std::cout << "Allocated size: ";
    cout_unit(std::cout, workspace_bytes) << std::endl;
  }

  /*****************************************************************************
   * Call CuDNN Convolution
   ****************************************************************************/
  // cudaEvent_t start, stop;
  // cudaEventCreate(&start);
  // cudaEventCreate(&stop);

  // cudaEventRecord(start);
  const float alpha = 1.0f, beta = 0.0f;
  checkCUDNN(cudnnConvolutionBackwardFilter(cudnn,
                                     /*alpha*/ &alpha,
                                     /*xDesc*/ desc.input,
                                     /*x*/ input.data_ptr(),
                                     /*dyDesc*/ desc.output,
                                     /*dy*/ output.data_ptr(),
                                     /*convDesc*/ desc.convolution,
                                     /*algo*/ convolution_algorithm[0].algo,
                                     /*workSpace*/ d_workspace,
                                     /*workSpaceSizeInBytes*/ workspace_bytes,
                                     /*beta*/ &beta,
                                     /*dwDesc*/ desc.weight,
                                     /*dw*/ weight.data_ptr()
                                     ));
  // cudaEventRecord(stop);
  // cudaEventSynchronize(stop);
  // float milliseconds{0};
  // cudaEventElapsedTime(&milliseconds, start, stop);
  // if (verbose)
  //   std::cout << "Elapsed Time: " << milliseconds << " ms" << std::endl;

  /*****************************************************************************
   * 5. Freeing variables
   ****************************************************************************/
  cudaFree(d_workspace);
  return output;
}

at::Tensor convolution_backward_input(const int bwdDataAlgo,
                                      const at::Tensor &input, const at::Tensor &weight, const at::Tensor &output,
                                      c10::ArrayRef<int64_t> stride, c10::ArrayRef<int64_t> padding,
                                      c10::ArrayRef<int64_t> dilation, int64_t groups, bool verbose)
{
  const cudnnHandle_t cudnn = at::native::getCudnnHandle();

  /*****************************************************************************
   * 1. Initializing Descriptors
   ****************************************************************************/
  cudnnDescriptors_t desc;
  initialize_descriptors(input, weight, output, stride, padding, dilation, desc);

  /*****************************************************************************
   * 2. Setting BWD Convolution Data Algo
   ****************************************************************************/
  cudnnConvolutionBwdDataAlgoPerf_t convolution_algorithm[CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT];
  int returnedAlgoCount;

  /**
   * TODO: I frequently get segmentation fault when finding the convolution
   * algorithms. I am not sure how to fix it.
   */
  if (bwdDataAlgo == -1)
  {
    if (verbose)
      std::cout << "Trying all" << std::endl;

    checkCUDNN(
        cudnnFindConvolutionBackwardDataAlgorithm(/*handle*/ cudnn,
                                             /*wDesc*/ desc.weight,
                                             /*dyDesc*/ desc.output,
                                             /*convDesc*/ desc.convolution,
                                             /*dxDesc*/ desc.input,
                                             /*requestedAlgoCount*/ CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT,
                                             /*returnedAlgoCount*/ &returnedAlgoCount,
                                             /*perfResults*/ convolution_algorithm));
    if (verbose)
      for (int i = 0; i < returnedAlgoCount; i++)
        std::cout << convolution_algorithm[i] << std::endl;
  }
  else
  {
    convolution_algorithm[0].algo = static_cast<cudnnConvolutionBwdDataAlgo_t>(bwdDataAlgo);
    convolution_algorithm[0].status = static_cast<cudnnStatus_t>(0);
    convolution_algorithm[0].time = -1;
    convolution_algorithm[0].memory = 0;
    convolution_algorithm[0].determinism = static_cast<cudnnDeterminism_t>(-1);
    convolution_algorithm[0].mathType = static_cast<cudnnMathType_t>(0);
    if (verbose)
    {
      std::cout << "Attempt with defined Algo:" << std::endl;
      std::cout << convolution_algorithm[0] << std::endl;
    }
  }

  /*****************************************************************************
   * 3. Get and Allocate Memory for Workspace
   ****************************************************************************/
  if (verbose)
    std::cout << "Allocating Workspace" << std::endl;

  size_t workspace_bytes{0};
  checkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn,
                                                     /*wDesc*/ desc.weight,
                                                     /*dyDesc*/ desc.output,
                                                     /*convDesc*/ desc.convolution,
                                                     /*dxDesc*/ desc.input,
                                                     /*algo*/ convolution_algorithm[0].algo,
                                                     /*sizeInBytes*/ &workspace_bytes));

  /*****************************************************************************
   * 4. Get and Allocate Memory for Workspace
   ****************************************************************************/
  if (verbose) {
    std::cout << "Workspace size: ";
    cout_unit(std::cout, workspace_bytes) << std::endl;
  }

  void *d_workspace{nullptr};
  cudaMalloc(&d_workspace, workspace_bytes);

  if (verbose) {
    std::cout << "Allocated size: ";
    cout_unit(std::cout, workspace_bytes) << std::endl;
  }

  /*****************************************************************************
   * 5. Call CuDNN Convolution
   ****************************************************************************/
  // cudaEvent_t start, stop;
  // cudaEventCreate(&start);
  // cudaEventCreate(&stop);

  // cudaEventRecord(start);
  const float alpha = 1.0f, beta = 0.0f;
  checkCUDNN(cudnnConvolutionBackwardData(cudnn,
                                     /*alpha*/ &alpha,
                                     /*wDesc*/ desc.weight,
                                     /*w*/ weight.data_ptr(),
                                     /*dyDesc*/ desc.output,
                                     /*dy*/ output.data_ptr(),
                                     /*convDesc*/ desc.convolution,
                                     /*algo*/ convolution_algorithm[0].algo,
                                     /*workSpace*/ d_workspace,
                                     /*workSpaceSizeInBytes*/ workspace_bytes,
                                     /*beta*/ &beta,
                                     /*xDesc*/ desc.input,
                                     /*x*/ input.data_ptr()));
  // cudaEventRecord(stop);
  // cudaEventSynchronize(stop);
  // float milliseconds{0};
  // cudaEventElapsedTime(&milliseconds, start, stop);
  // if (verbose)
  //   std::cout << "Elapsed Time: " << milliseconds << " ms" << std::endl;

  /*****************************************************************************
   * 5. Freeing variables
   ****************************************************************************/
  cudaFree(d_workspace);
  return output;
}


int conv_test() {
  const cudnnHandle_t cudnn = at::native::getCudnnHandle();

// input
  const int in_n = 1;
  const int in_c = 1;
  const int in_h = 5;
  const int in_w = 5;
  std::cout << "in_n: " << in_n << std::endl;
  std::cout << "in_c: " << in_c << std::endl;
  std::cout << "in_h: " << in_h << std::endl;
  std::cout << "in_w: " << in_w << std::endl;
  std::cout << std::endl;

  cudnnTensorDescriptor_t in_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(
        in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        in_n, in_c, in_h, in_w));

  float *in_data;
  CUDA_CALL(cudaMalloc(
        &in_data, in_n * in_c * in_h * in_w * sizeof(float)));

  // filter
  const int filt_k = 1;
  const int filt_c = 1;
  const int filt_h = 2;
  const int filt_w = 2;
  std::cout << "filt_k: " << filt_k << std::endl;
  std::cout << "filt_c: " << filt_c << std::endl;
  std::cout << "filt_h: " << filt_h << std::endl;
  std::cout << "filt_w: " << filt_w << std::endl;
  std::cout << std::endl;

  cudnnFilterDescriptor_t filt_desc;
  CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
  CUDNN_CALL(cudnnSetFilter4dDescriptor(
        filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
        filt_k, filt_c, filt_h, filt_w));

  float *filt_data;
  CUDA_CALL(cudaMalloc(
      &filt_data, filt_k * filt_c * filt_h * filt_w * sizeof(float)));

  // convolution
  const int pad_h = 1;
  const int pad_w = 1;
  const int str_h = 1;
  const int str_w = 1;
  const int dil_h = 1;
  const int dil_w = 1;
  std::cout << "pad_h: " << pad_h << std::endl;
  std::cout << "pad_w: " << pad_w << std::endl;
  std::cout << "str_h: " << str_h << std::endl;
  std::cout << "str_w: " << str_w << std::endl;
  std::cout << "dil_h: " << dil_h << std::endl;
  std::cout << "dil_w: " << dil_w << std::endl;
  std::cout << std::endl;

  cudnnConvolutionDescriptor_t conv_desc;
  CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
  CUDNN_CALL(cudnnSetConvolution2dDescriptor(
        conv_desc,
        pad_h, pad_w, str_h, str_w, dil_h, dil_w,
        CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT));

  // output
  int out_n;
  int out_c;
  int out_h;
  int out_w;
  
  CUDNN_CALL(cudnnGetConvolution2dForwardOutputDim(
        conv_desc, in_desc, filt_desc,
        &out_n, &out_c, &out_h, &out_w));

  std::cout << "out_n: " << out_n << std::endl;
  std::cout << "out_c: " << out_c << std::endl;
  std::cout << "out_h: " << out_h << std::endl;
  std::cout << "out_w: " << out_w << std::endl;
  std::cout << std::endl;

  cudnnTensorDescriptor_t out_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(
        out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        out_n, out_c, out_h, out_w));

  float *out_data;
  CUDA_CALL(cudaMalloc(
        &out_data, out_n * out_c * out_h * out_w * sizeof(float)));

  // algorithm
  // CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM         = 0 - Y
  // CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM = 1 - Y
  // CUDNN_CONVOLUTION_FWD_ALGO_GEMM                  = 2 - Y
  // CUDNN_CONVOLUTION_FWD_ALGO_DIRECT                = 3 - N
  // CUDNN_CONVOLUTION_FWD_ALGO_FFT                   = 4 - Y
  // CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING            = 5 - Y
  // CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD              = 6 - N
  // CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED     = 7 - N
  // CUDNN_CONVOLUTION_FWD_ALGO_COUNT                 = 8 - Y

  // CUDNN_CALL(cudnnGetConvolutionForwardAlgorithm(
  //       cudnn,
  //       in_desc, filt_desc, conv_desc, out_desc,
  //       CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo));

  cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_GEMM;
  std::cout << "Convolution algorithm: " << algo << std::endl;
  std::cout << std::endl;

  // workspace
  size_t ws_size;
  CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(
        cudnn, in_desc, filt_desc, conv_desc, out_desc, algo, &ws_size));

  float *ws_data;
  CUDA_CALL(cudaMalloc(&ws_data, ws_size));

  std::cout << "Workspace size: " << ws_size << std::endl;
  std::cout << std::endl;

  // perform
  float alpha = 1.f;
  float beta = 0.f;
  dev_iota<<<in_w * in_h, in_n * in_c>>>(in_data);
  dev_const<<<filt_w * filt_h, filt_k * filt_c>>>(filt_data, 1.f);

  // results
  std::cout << "in_data:" << std::endl;
  print(in_data, in_n, in_c, in_h, in_w);
  
  std::cout << "filt_data:" << std::endl;
  print(filt_data, filt_k, filt_c, filt_h, filt_w);

  CUDNN_CALL(cudnnConvolutionForward(
      cudnn,
      &alpha, in_desc, in_data, filt_desc, filt_data,
      conv_desc, algo, ws_data, ws_size,
      &beta, out_desc, out_data));
  
  std::cout << "out_data:" << std::endl;
  print(out_data, out_n, out_c, out_h, out_w);

  // finalizing
  CUDA_CALL(cudaFree(ws_data));
  CUDA_CALL(cudaFree(out_data));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(out_desc));
  CUDNN_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));
  CUDA_CALL(cudaFree(filt_data));
  CUDNN_CALL(cudnnDestroyFilterDescriptor(filt_desc));
  CUDA_CALL(cudaFree(in_data));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(in_desc));
  CUDNN_CALL(cudnnDestroy(cudnn));
  return 0;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("conv_test", &conv_test, "conv_test");
  m.def("convolution", &convolution, "convolution");
  m.def("convolution_backward_weight", &convolution_backward_weight, "convolution backward weight");
  m.def("convolution_backward_input", &convolution_backward_input, "convolution backward input");
}

// #endif
