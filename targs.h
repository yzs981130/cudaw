#ifndef __CUDAWTARGS_H__
#define __CUDAWTARGS_H__

extern cudaError_t cudawLaunchKernel(const void * func, dim3 gridDim, dim3 blockDim, void ** args, size_t sharedMem, cudaStream_t stream);

#endif // __CUDAWTARGS_H__
