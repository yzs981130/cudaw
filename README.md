# CUDAw

## Goal

CUDAw is developed based on [CUDA wrapper](https://github.com/yzs981130/cuda-wrapper), which implements virtual GPU memory translation to support **pause and resume** of CUDA application without user's awareness. It can not only trace and record every GPU memory usage, like `cudaMalloc` and `cudaMemcpy`, and every kernel function call, helping them use virtual GPU memory managed by administrators correctly, it can also support checkpointing user application, release their GPU memory, hang on kernel function execution and restore in any specified time. 

Based on CUDAw, we have implemented a mechanism for GPU sharing across multiple workloads. Our additional work as follows:

- [x] Anytime pause and resume of CUDA application
- [x] Fine-grained time-sharing of GPU cores and memory
- [x] Fine-grained GPU memory virtualization for CUDA application
- [ ] Deep learning job migration across nodes
- [ ] Co-design of deep learning jobs and scheduler

## Approach

Like [CUDA wrapper](https://github.com/yzs981130/cuda-wrapper), we wrap the NVIDIA runtime API by `dlsym` and captures every memory usage and kernel launch. For general purpose and easier usage, we wrap all possible CUDA calls, which can be used as a fake `libcudart.so`  independently. 

We implement a full GPU memory management and translation for CUDA application. In other words, we implement a full GPU memory abstraction layer. When user application tries to allocate or access GPU memory, our wrapper will handle the request, pass the true GPU address to real `libcudart.so`, and make actual execution on GPU device through CUDA.  

We use some novel techniques to get parameters of kernel functions of  `cudaLaunchKernel`, which instructs us to translate memory addresses. We have realized an automatic discovery of kernel function parameters.

We also implement mechanisms for a better usage and codesign checkpoint notifier with a custom DLT job scheduler on Kubernetes. The feature of any-time checkpoint and migration of jobs will bring unprecedented possibilities to the scheduling strategy and performance of the scheduler.

Some ongoing works are that reduce the overhead of checkpointing and framework and algorithm of scheduler codesign with the techniques. We will release job migration support shortly.

## Usage

To be continued...

