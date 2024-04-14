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
  const cudnnHandle_t cudnn = at::native::getCudnnHandle();

  /*****************************************************************************
   * 1. Initializing Descriptors
   ****************************************************************************/
  cudnnDescriptors_t desc;
  initialize_descriptors(input, weight, output, stride, padding, dilation, desc);

  /*****************************************************************************
   * 2. Setting FWD Convolution Algo
   ****************************************************************************/
  cudnnConvolutionFwdAlgoPerf_t convolution_algorithm[CUDNN_CONVOLUTION_FWD_ALGO_COUNT];
  int returnedAlgoCount;

  if (fwdAlgo == -1)
  {
    if (verbose)
      std::cout << "Trying all" << std::endl;

    checkCUDNN(
        cudnnFindConvolutionForwardAlgorithm(/*handle*/ cudnn,
                                             /*xDesc*/ desc.input,
                                             /*wDesc*/ desc.weight,
                                             /*convDesc*/ desc.convolution,
                                             /*yDesc*/ desc.output,
                                             /*requestedAlgoCount*/ CUDNN_CONVOLUTION_FWD_ALGO_COUNT,
                                             /*returnedAlgoCount*/ &returnedAlgoCount,
                                             /*perfResults*/ convolution_algorithm));
    if (verbose)
      for (int i = 0; i < returnedAlgoCount; i++)
        std::cout << convolution_algorithm[i] << std::endl;
  }
  else
  {
    convolution_algorithm[0].algo = static_cast<cudnnConvolutionFwdAlgo_t>(fwdAlgo);
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
  checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
                                                     /*xDesc*/ desc.input,
                                                     /*wDesc*/ desc.weight,
                                                     /*convDesc*/ desc.convolution,
                                                     /*yDesc*/ desc.output,
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
  checkCUDNN(cudnnConvolutionForward(cudnn,
                                     /*alpha*/ &alpha,
                                     /*xDesc*/ desc.input,
                                     /*x*/ input.data_ptr(),
                                     /*wDesc*/ desc.weight,
                                     /*w*/ weight.data_ptr(),
                                     /*convDesc*/ desc.convolution,
                                     /*algo*/ convolution_algorithm[0].algo,
                                     /*workSpace*/ d_workspace,
                                     /*workSpaceSizeInBytes*/ workspace_bytes,
                                     /*beta*/ &beta,
                                     /*yDesc*/ desc.output,
                                     /*y*/ output.data_ptr()));
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("convolution", &convolution, "convolution");
  m.def("convolution_backward_weight", &convolution_backward_weight, "convolution backward weight");
  m.def("convolution_backward_input", &convolution_backward_input, "convolution backward input");
}

// #endif
