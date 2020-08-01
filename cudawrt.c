#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <string.h>
#include <pthread.h>
#include <cuda_runtime.h>
#include <sys/types.h>
#include <unistd.h>
#include <time.h>
#include <unistd.h>
#include <assert.h>
#include <dlfcn.h>
#include <errno.h>

#include "cudaw.h"

static const char LIB_STRING[] = "/workspace/libcudart.so.10.0.130";

// DEFSO & LDSYM

#define MAX_FUNC 200
#include "ldsym.h"

#define DEFSR(rtype, func) static int idx_##func; static rtype(*so_##func)
#define DEFSO(func)        static int idx_##func; static cudaError_t(*so_##func)
#define FSWAP(func)        &so_##func,
#define FCOPY(func)        so_##func,

DEFSR(const char*, cudaGetErrorString)(cudaError_t err);
DEFSR(const char*, cudaGetErrorName)(cudaError_t err);
DEFSR(void, __cudaRegisterVar)(void **fatCubinHandle, char *hostVar, char *deviceAddress, const char *deviceName, int ext, int size, int constant, int global);
DEFSR(void, __cudaRegisterTexture)(void **fatCubinHandle, const struct textureReference *hostVar, const void **deviceAddress, const char *deviceName, int dim, int norm, int ext);
DEFSR(void, __cudaRegisterSurface)(void **fatCubinHandle, const struct surfaceReference  *hostVar, const void **deviceAddress, const char *deviceName, int dim, int ext);
DEFSR(void, __cudaRegisterFunction)(void **fatCubinHandle, const char *hostFun, char *deviceFun, const char *deviceName, int thread_limit, uint3 *tid, uint3 *bid, dim3 *bDim, dim3 *gDim, int *wSize);
DEFSR(void**, __cudaRegisterFatBinary)(void* fatCubin);
DEFSR(void, __cudaUnregisterFatBinary)(void** point);
DEFSR(unsigned, __cudaPushCallConfiguration)(dim3 gridDim, dim3 blockDim, size_t sharedMem , void *stream);
DEFSR(void, __cudaRegisterFatBinaryEnd)(void **fatCubinHandle);
DEFSR(struct cudaChannelFormatDesc, cudaCreateChannelDesc)(int  x, int  y, int  z, int  w, enum cudaChannelFormatKind f);
DEFSO(__cudaPopCallConfiguration)(dim3 *gridDim, dim3 *blockDim, size_t *sharedMem, void *stream);
DEFSO(cudaMalloc)(void** devPtr, size_t bytesize);
DEFSO(cudaFree)(void* devPtr);
DEFSO(cudaHostAlloc)(void** pHost, size_t size, unsigned int flags);
DEFSO(cudaFreeHost)(void* ptr);
DEFSO(cudaDeviceGetStreamPriorityRange)(int* leastPriority, int* greatestPriority);
DEFSO(cudaHostGetDevicePointer)(void** pDevice, void* pHost, unsigned int flags);
DEFSO(cudaGetDeviceProperties)(struct cudaDeviceProp* prop, int device);
DEFSO(cudaStreamCreateWithPriority)(cudaStream_t* pStream, unsigned int flags, int priority);
DEFSO(cudaStreamCreateWithFlags)(cudaStream_t* pStream, unsigned int flags);
DEFSO(cudaEventCreateWithFlags)(cudaEvent_t* event, unsigned int flags);
DEFSO(cudaEventDestroy)(cudaEvent_t event);
DEFSO(cudaGetDeviceCount)(int* count);
DEFSO(cudaFuncGetAttributes)(struct cudaFuncAttributes* attr, const void* func);
DEFSO(cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags)(int* numBlocks, const void* func, int blockSize, size_t dynamicSMemSize, unsigned int flags);
DEFSO(cudaStreamSynchronize)(cudaStream_t stream);
DEFSO(cudaMemcpyAsync)(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream);
DEFSO(cudaEventRecord)(cudaEvent_t event, cudaStream_t stream);
DEFSO(cudaDeviceGetAttribute)(int* value, enum cudaDeviceAttr attr, int device);
DEFSO(cudaMemsetAsync)(void* devPtr, int value, size_t count, cudaStream_t stream);
DEFSO(cudaLaunchKernel)(const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream);
DEFSO(cudaGetLastError)();
DEFSO(cudaSetDevice)(int device);
DEFSO(cudaGetDevice)(int* device);
DEFSO(cudaProfilerStop)(void);
DEFSO(cudaProfilerStart)(void);
DEFSO(cudaProfilerInitialize)(const char* configFile, const char* outputFile, cudaOutputMode_t outputMode);
DEFSO(cudaGraphRemoveDependencies)(cudaGraph_t graph, cudaGraphNode_t* from, cudaGraphNode_t* to, size_t numDependencies);
DEFSO(cudaGraphNodeGetType)(cudaGraphNode_t node, enum cudaGraphNodeType * pType);
DEFSO(cudaGraphNodeGetDependentNodes)(cudaGraphNode_t node, cudaGraphNode_t* pDependentNodes, size_t* pNumDependentNodes);
DEFSO(cudaGraphNodeGetDependencies)(cudaGraphNode_t node, cudaGraphNode_t* pDependencies, size_t* pNumDependencies);
DEFSO(cudaGraphNodeFindInClone)(cudaGraphNode_t* pNode, cudaGraphNode_t originalNode, cudaGraph_t clonedGraph);
DEFSO(cudaGraphMemsetNodeSetParams)(cudaGraphNode_t node, const struct cudaMemsetParams* pNodeParams);
DEFSO(cudaGraphMemsetNodeGetParams)(cudaGraphNode_t node, struct cudaMemsetParams* pNodeParams);
DEFSO(cudaGraphMemcpyNodeSetParams)(cudaGraphNode_t node, const struct cudaMemcpy3DParms* pNodeParams);
DEFSO(cudaGraphMemcpyNodeGetParams)(cudaGraphNode_t node, struct cudaMemcpy3DParms* pNodeParams);
DEFSO(cudaGraphLaunch)(cudaGraphExec_t graphExec, cudaStream_t stream);
DEFSO(cudaGraphKernelNodeSetParams)(cudaGraphNode_t node, const struct cudaKernelNodeParams* pNodeParams);
DEFSO(cudaGraphKernelNodeGetParams)(cudaGraphNode_t node, struct cudaKernelNodeParams* pNodeParams);
DEFSO(cudaGraphInstantiate)(cudaGraphExec_t* pGraphExec, cudaGraph_t graph, cudaGraphNode_t* pErrorNode, char* pLogBuffer, size_t bufferSize);
DEFSO(cudaGraphHostNodeSetParams)(cudaGraphNode_t node, const struct cudaHostNodeParams* pNodeParams);
DEFSO(cudaGraphHostNodeGetParams)(cudaGraphNode_t node, struct cudaHostNodeParams* pNodeParams);
DEFSO(cudaGraphGetRootNodes)(cudaGraph_t graph, cudaGraphNode_t* pRootNodes, size_t* pNumRootNodes);
DEFSO(cudaGraphGetNodes)(cudaGraph_t graph, cudaGraphNode_t* nodes, size_t* numNodes);
DEFSO(cudaGraphGetEdges)(cudaGraph_t graph, cudaGraphNode_t* from, cudaGraphNode_t* to, size_t* numEdges);
DEFSO(cudaGraphExecDestroy)(cudaGraphExec_t graphExec);
DEFSO(cudaGraphDestroyNode)(cudaGraphNode_t node);
DEFSO(cudaGraphDestroy)(cudaGraph_t graph);
DEFSO(cudaGraphCreate)(cudaGraph_t* pGraph, unsigned int flags);
DEFSO(cudaGraphClone)(cudaGraph_t* pGraphClone, cudaGraph_t originalGraph);
DEFSO(cudaGraphChildGraphNodeGetGraph)(cudaGraphNode_t node, cudaGraph_t* pGraph);
DEFSO(cudaGraphAddMemsetNode)(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, cudaGraphNode_t* pDependencies, size_t numDependencies, const struct cudaMemsetParams* pMemsetParams);
DEFSO(cudaGraphAddMemcpyNode)(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, cudaGraphNode_t* pDependencies, size_t numDependencies, const struct cudaMemcpy3DParms* pCopyParams);
DEFSO(cudaGraphAddKernelNode)(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, cudaGraphNode_t* pDependencies, size_t numDependencies, const struct cudaKernelNodeParams* pNodeParams);
DEFSO(cudaGraphAddHostNode)(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, cudaGraphNode_t* pDependencies, size_t numDependencies, const struct cudaHostNodeParams* pNodeParams);
DEFSO(cudaGraphAddEmptyNode)(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, cudaGraphNode_t* pDependencies, size_t numDependencies);
DEFSO(cudaGraphAddDependencies)(cudaGraph_t graph, cudaGraphNode_t* from, cudaGraphNode_t* to, size_t numDependencies);
DEFSO(cudaGraphAddChildGraphNode)(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, cudaGraphNode_t* pDependencies, size_t numDependencies, cudaGraph_t childGraph);
DEFSO(cudaRuntimeGetVersion)(int* runtimeVersion);
DEFSO(cudaDriverGetVersion)(int* driverVersion);
DEFSO(cudaGetSurfaceObjectResourceDesc)(struct cudaResourceDesc* pResDesc, cudaSurfaceObject_t surfObject);
DEFSO(cudaDestroySurfaceObject)(cudaSurfaceObject_t surfObject);
DEFSO(cudaCreateSurfaceObject)(cudaSurfaceObject_t* pSurfObject, const struct cudaResourceDesc* pResDesc);
DEFSO(cudaGetTextureObjectTextureDesc)(struct cudaTextureDesc* pTexDesc, cudaTextureObject_t texObject);
DEFSO(cudaGetTextureObjectResourceViewDesc)(struct cudaResourceViewDesc* pResViewDesc, cudaTextureObject_t texObject);
DEFSO(cudaGetTextureObjectResourceDesc)(struct cudaResourceDesc* pResDesc, cudaTextureObject_t texObject);
DEFSO(cudaGetChannelDesc)(struct cudaChannelFormatDesc* desc, cudaArray_const_t array);
DEFSO(cudaDestroyTextureObject)(cudaTextureObject_t texObject);
DEFSO(cudaCreateTextureObject)(cudaTextureObject_t* pTexObject, const struct cudaResourceDesc* pResDesc, const struct cudaTextureDesc* pTexDesc, const struct cudaResourceViewDesc* pResViewDesc);
DEFSO(cudaGraphicsUnregisterResource)(cudaGraphicsResource_t resource);
DEFSO(cudaGraphicsUnmapResources)(int count, cudaGraphicsResource_t* resources, cudaStream_t stream);
DEFSO(cudaGraphicsSubResourceGetMappedArray)(cudaArray_t* array, cudaGraphicsResource_t resource, unsigned int arrayIndex, unsigned int mipLevel);
DEFSO(cudaGraphicsResourceSetMapFlags)(cudaGraphicsResource_t resource, unsigned int flags);
DEFSO(cudaGraphicsResourceGetMappedPointer)(void** devPtr, size_t* size, cudaGraphicsResource_t resource);
DEFSO(cudaGraphicsResourceGetMappedMipmappedArray)(cudaMipmappedArray_t* mipmappedArray, cudaGraphicsResource_t resource);
DEFSO(cudaGraphicsMapResources)(int count, cudaGraphicsResource_t* resources, cudaStream_t stream);
DEFSO(cudaDeviceEnablePeerAccess)(int peerDevice, unsigned int flags);
DEFSO(cudaDeviceDisablePeerAccess)(int peerDevice);
DEFSO(cudaDeviceCanAccessPeer)(int* canAccessPeer, int device, int peerDevice);
DEFSO(cudaPointerGetAttributes)(struct cudaPointerAttributes* attributes, const void* ptr);
DEFSO(cudaMemset3DAsync)(struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent, cudaStream_t stream);
DEFSO(cudaMemset3D)(struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent);
DEFSO(cudaMemset2DAsync)(void* devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream);
DEFSO(cudaMemset2D)(void* devPtr, size_t pitch, int value, size_t width, size_t height);
DEFSO(cudaMemset)(void* devPtr, int value, size_t count);
DEFSO(cudaMemcpyToSymbolAsync)(const void* symbol, const void* src, size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream);
DEFSO(cudaMemcpyToSymbol)(const void* symbol, const void* src, size_t count, size_t offset, enum cudaMemcpyKind kind);
DEFSO(cudaMemcpyPeerAsync)(void* dst, int dstDevice, const void* src, int srcDevice, size_t count, cudaStream_t stream);
DEFSO(cudaMemcpyPeer)(void* dst, int dstDevice, const void* src, int srcDevice, size_t count);
DEFSO(cudaMemcpyFromSymbolAsync)(void* dst, const void* symbol, size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream);
DEFSO(cudaMemcpyFromSymbol)(void* dst, const void* symbol, size_t count, size_t offset , enum cudaMemcpyKind kind);
DEFSO(cudaMemcpy3DPeerAsync)(const struct cudaMemcpy3DPeerParms* p, cudaStream_t stream);
DEFSO(cudaMemcpy3DPeer)(const struct cudaMemcpy3DPeerParms* p);
DEFSO(cudaMemcpy3DAsync)(const struct cudaMemcpy3DParms* p, cudaStream_t stream);
DEFSO(cudaMemcpy3D)(const struct cudaMemcpy3DParms* p);
DEFSO(cudaMemcpy2DToArrayAsync)(cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream);
DEFSO(cudaMemcpy2DToArray)(cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind);
DEFSO(cudaMemcpy2DFromArrayAsync)(void* dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream);
DEFSO(cudaMemcpy2DFromArray)(void* dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind);
DEFSO(cudaMemcpy2DAsync)(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream);
DEFSO(cudaMemcpy2DArrayToArray)(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, enum cudaMemcpyKind kind);
DEFSO(cudaMemcpy2D)(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind);
DEFSO(cudaMemcpy)(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind);
DEFSO(cudaMemRangeGetAttributes)(void** data, size_t* dataSizes, enum cudaMemRangeAttribute *attributes, size_t numAttributes, const void* devPtr, size_t count);
DEFSO(cudaMemRangeGetAttribute)(void* data, size_t dataSize, enum cudaMemRangeAttribute attribute, const void* devPtr, size_t count);
DEFSO(cudaMemPrefetchAsync)(const void* devPtr, size_t count, int dstDevice, cudaStream_t stream);
DEFSO(cudaMemAdvise)(const void* devPtr, size_t count, enum cudaMemoryAdvise advice, int device);
DEFSO(cudaHostUnregister)(void* ptr);
DEFSO(cudaHostRegister)(void* ptr, size_t size, unsigned int flags);
DEFSO(cudaHostGetFlags)(unsigned int* pFlags, void* pHost);
DEFSO(cudaGetSymbolSize)(size_t* size, const void* symbol);
DEFSO(cudaGetSymbolAddress)(void** devPtr, const void* symbol);
DEFSO(cudaGetMipmappedArrayLevel)(cudaArray_t* levelArray, cudaMipmappedArray_const_t mipmappedArray, unsigned int level);
DEFSO(cudaFreeMipmappedArray)(cudaMipmappedArray_t mipmappedArray);
DEFSO(cudaFreeArray)(cudaArray_t array);
DEFSO(cudaArrayGetInfo)(struct cudaChannelFormatDesc* desc, struct cudaExtent* extent, unsigned int* flags, cudaArray_t array);
DEFSO(cudaOccupancyMaxActiveBlocksPerMultiprocessor)(int* numBlocks, const void* func, int blockSize, size_t dynamicSMemSize);
//DEFSO(cudaSetDoubleForHost)(double* d);
//DEFSO(cudaSetDoubleForDevice)(double* d);
DEFSO(cudaLaunchHostFunc)(cudaStream_t stream, cudaHostFn_t fn, void* userData);
DEFSO(cudaLaunchCooperativeKernelMultiDevice)(struct cudaLaunchParams* launchParamsList, unsigned int numDevices, unsigned int flags);
DEFSO(cudaLaunchCooperativeKernel)(const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream);
DEFSO(cudaFuncSetSharedMemConfig)(const void* func, enum cudaSharedMemConfig config);
DEFSO(cudaFuncSetCacheConfig)(const void* func, enum cudaFuncCache cacheConfig);
DEFSO(cudaFuncSetAttribute)(const void* func, enum cudaFuncAttribute attr, int value);
DEFSO(cudaWaitExternalSemaphoresAsync)(const cudaExternalSemaphore_t* extSemArray, const struct cudaExternalSemaphoreWaitParams* paramsArray, unsigned int numExtSems, cudaStream_t stream);
DEFSO(cudaSignalExternalSemaphoresAsync)(const cudaExternalSemaphore_t* extSemArray, const struct cudaExternalSemaphoreSignalParams* paramsArray, unsigned int numExtSems, cudaStream_t stream);
DEFSO(cudaImportExternalSemaphore)(cudaExternalSemaphore_t* extSem_out, const struct cudaExternalSemaphoreHandleDesc* semHandleDesc);
DEFSO(cudaImportExternalMemory)(cudaExternalMemory_t* extMem_out, const struct cudaExternalMemoryHandleDesc* memHandleDesc);
DEFSO(cudaExternalMemoryGetMappedMipmappedArray)(cudaMipmappedArray_t* mipmap, cudaExternalMemory_t extMem, const struct cudaExternalMemoryMipmappedArrayDesc* mipmapDesc);
DEFSO(cudaExternalMemoryGetMappedBuffer)(void** devPtr, cudaExternalMemory_t extMem, const struct cudaExternalMemoryBufferDesc* bufferDesc);
DEFSO(cudaDestroyExternalSemaphore)(cudaExternalSemaphore_t extSem);
DEFSO(cudaDestroyExternalMemory)(cudaExternalMemory_t extMem);
DEFSO(cudaEventSynchronize)(cudaEvent_t event);
DEFSO(cudaEventQuery)(cudaEvent_t event);
DEFSO(cudaEventElapsedTime)(float* ms, cudaEvent_t start, cudaEvent_t end);
DEFSO(cudaEventCreate)(cudaEvent_t* event);
DEFSO(cudaStreamWaitEvent)(cudaStream_t stream, cudaEvent_t event, unsigned int flags);
DEFSO(cudaStreamQuery)(cudaStream_t stream);
DEFSO(cudaStreamIsCapturing)(cudaStream_t stream, enum cudaStreamCaptureStatus *pCaptureStatus);
DEFSO(cudaStreamGetPriority)(cudaStream_t hStream, int* priority);
DEFSO(cudaStreamGetFlags)(cudaStream_t hStream, unsigned int* flags);
DEFSO(cudaStreamEndCapture)(cudaStream_t stream, cudaGraph_t* pGraph);
DEFSO(cudaStreamCreate)(cudaStream_t* pStream);
DEFSO(cudaStreamBeginCapture)(cudaStream_t stream);
DEFSO(cudaStreamAttachMemAsync)(cudaStream_t stream, void* devPtr, size_t length, unsigned int flags);
DEFSO(cudaStreamAddCallback)(cudaStream_t stream, cudaStreamCallback_t callback, void* userData, unsigned int flags);
DEFSO(cudaPeekAtLastError)();
DEFSO(cudaSetValidDevices)(int* device_arr, int len);
DEFSO(cudaSetDeviceFlags)(unsigned int flags);
DEFSO(cudaIpcOpenMemHandle)(void** devPtr, cudaIpcMemHandle_t hand, unsigned int flags);
DEFSO(cudaIpcOpenEventHandle)(cudaEvent_t* event, cudaIpcEventHandle_t hand);
DEFSO(cudaIpcGetMemHandle)(cudaIpcMemHandle_t* handle, void* devPtr);
DEFSO(cudaIpcGetEventHandle)(cudaIpcEventHandle_t* handle, cudaEvent_t event);
DEFSO(cudaIpcCloseMemHandle)(void* devPtr);
DEFSO(cudaGetDeviceFlags)(unsigned int* flags);
DEFSO(cudaDeviceSynchronize)();
DEFSO(cudaDeviceSetSharedMemConfig)(enum cudaSharedMemConfig config);
DEFSO(cudaDeviceSetLimit)(enum cudaLimit limit, size_t value);
DEFSO(cudaDeviceSetCacheConfig)(enum cudaFuncCache cacheConfig);
DEFSO(cudaDeviceReset)();
DEFSO(cudaDeviceGetSharedMemConfig)(enum cudaSharedMemConfig * pConfig);
DEFSO(cudaDeviceGetPCIBusId)(char* pciBusId, int len, int device);
DEFSO(cudaDeviceGetP2PAttribute)(int* value, enum cudaDeviceP2PAttr attr, int srcDevice, int dstDevice);
DEFSO(cudaDeviceGetLimit)(size_t* pValue, enum cudaLimit limit);
DEFSO(cudaDeviceGetCacheConfig)(enum cudaFuncCache * pCacheConfig);
DEFSO(cudaDeviceGetByPCIBusId)(int* device, const char* pciBusId);
DEFSO(cudaChooseDevice)(int* device, const struct cudaDeviceProp* prop);
DEFSO(cudaMallocMipmappedArray)(cudaMipmappedArray_t* mipmappedArray, const struct cudaChannelFormatDesc* desc, struct cudaExtent extent, unsigned int numLevels, unsigned int flags);
DEFSO(cudaMallocArray)(cudaArray_t* array, const struct cudaChannelFormatDesc* desc, size_t width, size_t height, unsigned int flags);
DEFSO(cudaMallocHost)(void** ptr, size_t size);
DEFSO(cudaMalloc3DArray)(cudaArray_t* array, const struct cudaChannelFormatDesc* desc, struct cudaExtent extent, unsigned int flags);
DEFSO(cudaMalloc3D)(struct cudaPitchedPtr* pitchedDevPtr, struct cudaExtent extent);
DEFSO(cudaMallocManaged)(void** devPtr, size_t bytesize, unsigned int flags);
DEFSO(cudaMallocPitch)(void** devPtr, size_t* pitch, size_t width, size_t height);
DEFSO(cudaMemGetInfo)(size_t* free , size_t* total);
DEFSO(cudaStreamDestroy)(cudaStream_t stream);


#define checkCudaErrors(err)  __checkCudaErrors(err, __FILE__, __LINE__, __func__)
static cudaError_t __checkCudaErrors(cudaError_t err, const char *file, const int line , const char *func) {
    if (cudaSuccess != err && 11!=err && 29!=err) {
        fprintf(stderr, 
                "CUDA Runtime API error = %04d from file <%s>, line %i, function %s.\n", 
                err, file, line, func);
        /*if (err == 4||err==77) {
            exit(1);
        }*/
        //exit(-1);
    }
    return err;
}

cudaError_t cudaMalloc(void** devPtr, size_t bytesize) {
    cudaError_t r;
    begin_func(cudaMalloc); 
    r = so_cudaMalloc(devPtr ,(bytesize));
    end_func(cudaMalloc);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaFree(void* devPtr) {
    cudaError_t r;
    begin_func(cudaFree);
    r = so_cudaFree(devPtr);
    end_func(cudaFree);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaHostAlloc(void** pHost, size_t size, unsigned int flags) {
    cudaError_t r;
    begin_func(cudaHostAlloc);
    r = so_cudaHostAlloc(pHost, size, flags);
    end_func(cudaHostAlloc);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaFreeHost(void* ptr) {
    cudaError_t r;
    begin_func(cudaFreeHost);
    r = so_cudaFreeHost(ptr);
    end_func(cudaFreeHost);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaDeviceGetStreamPriorityRange(int* leastPriority, int* greatestPriority) {
    cudaError_t r;
    begin_func(cudaDeviceGetStreamPriorityRange);
    r = so_cudaDeviceGetStreamPriorityRange(leastPriority, greatestPriority);
    end_func(cudaDeviceGetStreamPriorityRange);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaHostGetDevicePointer(void** pDevice, void* pHost, unsigned int  flags) {
    cudaError_t r;
    begin_func(cudaHostGetDevicePointer);
    r = so_cudaHostGetDevicePointer(pDevice, pHost, flags);
    end_func(cudaHostGetDevicePointer);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp* prop, int device) {
    cudaError_t r;
    begin_func(cudaGetDeviceProperties);
    r = so_cudaGetDeviceProperties(prop, device);
    end_func(cudaGetDeviceProperties);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaStreamCreateWithPriority(cudaStream_t* pStream, unsigned int  flags, int  priority) {
    cudaError_t r;
    begin_func(cudaStreamCreateWithPriority);
    r = so_cudaStreamCreateWithPriority(pStream,  flags,  priority);
    end_func(cudaStreamCreateWithPriority);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaStreamCreateWithFlags(cudaStream_t* pStream, unsigned int  flags) {
    cudaError_t r;
    begin_func(cudaStreamCreateWithFlags);
    r = so_cudaStreamCreateWithFlags(pStream,  flags);
    end_func(cudaStreamCreateWithFlags);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int  flags) {
    cudaError_t r;
    begin_func(cudaEventCreateWithFlags);
    r = so_cudaEventCreateWithFlags(event, flags);
    end_func(cudaEventCreateWithFlags);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaEventDestroy(cudaEvent_t event) {
    cudaError_t r;
    begin_func(cudaEventDestroy);
    r = so_cudaEventDestroy(event);
    end_func(cudaEventDestroy);
    checkCudaErrors(r);
    return r;
}

__host__  __device__ cudaError_t cudaStreamDestroy(cudaStream_t stream) {
    cudaError_t r;
    begin_func(cudaStreamDestroy);
    r = so_cudaStreamDestroy(stream);
    end_func(cudaStreamDestroy);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaGetDeviceCount(int* count) {
    cudaError_t r;
    begin_func(cudaGetDeviceCount);
    r = so_cudaGetDeviceCount(count);
    end_func(cudaGetDeviceCount);
    checkCudaErrors(r);
    return r;
}

struct cudaChannelFormatDesc cudaCreateChannelDesc(int  x, int  y, int  z, int  w, enum cudaChannelFormatKind f) {
    struct cudaChannelFormatDesc desc;
    begin_func(cudaCreateChannelDesc);
    desc = so_cudaCreateChannelDesc(x, y, z, w, f);
    end_func(cudaCreateChannelDesc);
    return desc;
}

cudaError_t cudaFuncGetAttributes(struct cudaFuncAttributes* attr, const void* func) {
    cudaError_t r;
    begin_func(cudaFuncGetAttributes);
    r = so_cudaFuncGetAttributes(attr, func);
    end_func(cudaFuncGetAttributes);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int* numBlocks, const void* func, int  blockSize, size_t dynamicSMemSize, unsigned int flags) {
    cudaError_t r;
    begin_func(cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags);
    r = so_cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, func, blockSize, dynamicSMemSize, flags);
    end_func(cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
    cudaError_t r;
    begin_func(cudaStreamSynchronize);
    r = so_cudaStreamSynchronize(stream);
    end_func(cudaStreamSynchronize);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream) {
    cudaError_t r;
    begin_func(cudaMemcpyAsync);
    r = so_cudaMemcpyAsync(dst, src, count, kind, stream);
    end_func(cudaMemcpyAsync);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
    cudaError_t r;
    begin_func(cudaEventRecord);
    r = so_cudaEventRecord(event, stream);
    end_func(cudaEventRecord);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaDeviceGetAttribute(int* value, enum cudaDeviceAttr attr, int  device) {
    cudaError_t r;
    begin_func(cudaDeviceGetAttribute);
    r = so_cudaDeviceGetAttribute(value, attr, device);
    end_func(cudaDeviceGetAttribute);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaMemsetAsync(void* devPtr, int  value, size_t count, cudaStream_t stream) {
    cudaError_t r;
    begin_func(cudaMemsetAsync);
    r = so_cudaMemsetAsync(devPtr, value, count, stream);
    end_func(cudaMemsetAsync);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaLaunchKernel(const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream) {
    cudaError_t r;
    begin_func(cudaLaunchKernel);
    r = so_cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);
    end_func(cudaLaunchKernel);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaGetLastError() {
    cudaError_t r;
    begin_func(cudaGetLastError);
    r = so_cudaGetLastError();
    end_func(cudaGetLastError);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaSetDevice(int device) {
    cudaError_t r;
    begin_func(cudaSetDevice);
    r = so_cudaSetDevice(device);
    end_func(cudaSetDevice);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaGetDevice(int* device) {
    cudaError_t r;
    begin_func(cudaGetDevice);
    r = so_cudaGetDevice(device);
    end_func(cudaGetDevice);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaProfilerStop(void) {
    cudaError_t r;
    begin_func(cudaProfilerStop);
    r = so_cudaProfilerStop();
    end_func(cudaProfilerStop);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaProfilerStart(void) {
    cudaError_t r;
    begin_func(cudaProfilerStart);
    r = so_cudaProfilerStart();
    end_func(cudaProfilerStart);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaProfilerInitialize(const char* configFile, const char* outputFile, cudaOutputMode_t outputMode) {
    cudaError_t r;
    begin_func(cudaProfilerInitialize);
    r = so_cudaProfilerInitialize(configFile,  outputFile, outputMode);
    end_func(cudaProfilerInitialize);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaGraphRemoveDependencies(cudaGraph_t graph, cudaGraphNode_t* from,  cudaGraphNode_t* to, size_t numDependencies) {
    cudaError_t r;
    begin_func(cudaGraphRemoveDependencies);
    r = so_cudaGraphRemoveDependencies(graph, from, to, numDependencies);
    end_func(cudaGraphRemoveDependencies);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaGraphNodeGetType(cudaGraphNode_t node, enum cudaGraphNodeType * pType) {
    cudaError_t r;
    begin_func(cudaGraphNodeGetType);
    r = so_cudaGraphNodeGetType(node, pType);
    end_func(cudaGraphNodeGetType);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaGraphNodeGetDependentNodes(cudaGraphNode_t node, cudaGraphNode_t* pDependentNodes, size_t* pNumDependentNodes) {
    cudaError_t r;
    begin_func(cudaGraphNodeGetDependentNodes);
    r = so_cudaGraphNodeGetDependentNodes(node, pDependentNodes, pNumDependentNodes);
    end_func(cudaGraphNodeGetDependentNodes);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaGraphNodeGetDependencies(cudaGraphNode_t node, cudaGraphNode_t* pDependencies, size_t* pNumDependencies) {
    cudaError_t r;
    begin_func(cudaGraphNodeGetDependencies);
    r = so_cudaGraphNodeGetDependencies(node, pDependencies, pNumDependencies);
    end_func(cudaGraphNodeGetDependencies);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaGraphNodeFindInClone(cudaGraphNode_t* pNode, cudaGraphNode_t originalNode, cudaGraph_t clonedGraph) {
    cudaError_t r;
    begin_func(cudaGraphNodeFindInClone);
    r = so_cudaGraphNodeFindInClone(pNode, originalNode, clonedGraph);
    end_func(cudaGraphNodeFindInClone);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaGraphMemsetNodeSetParams(cudaGraphNode_t node, const struct cudaMemsetParams* pNodeParams) {
    cudaError_t r;
    begin_func(cudaGraphMemsetNodeSetParams);
    r = so_cudaGraphMemsetNodeSetParams(node, pNodeParams);
    end_func(cudaGraphMemsetNodeSetParams);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaGraphMemsetNodeGetParams(cudaGraphNode_t node, struct cudaMemsetParams* pNodeParams) {
    cudaError_t r;
    begin_func(cudaGraphMemsetNodeGetParams);
    r = so_cudaGraphMemsetNodeGetParams(node, pNodeParams);
    end_func(cudaGraphMemsetNodeGetParams);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaGraphMemcpyNodeSetParams(cudaGraphNode_t node, const struct cudaMemcpy3DParms* pNodeParams) {
    cudaError_t r;
    begin_func(cudaGraphMemcpyNodeSetParams);
    r = so_cudaGraphMemcpyNodeSetParams(node, pNodeParams);
    end_func(cudaGraphMemcpyNodeSetParams);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaGraphMemcpyNodeGetParams(cudaGraphNode_t node, struct cudaMemcpy3DParms* pNodeParams) {
    cudaError_t r;
    begin_func(cudaGraphMemcpyNodeGetParams);
    r = so_cudaGraphMemcpyNodeGetParams(node, pNodeParams);
    end_func(cudaGraphMemcpyNodeGetParams);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaGraphLaunch(cudaGraphExec_t graphExec, cudaStream_t stream) {
    cudaError_t r;
    begin_func(cudaGraphLaunch);
    r = so_cudaGraphLaunch(graphExec, stream);
    end_func(cudaGraphLaunch);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaGraphKernelNodeSetParams(cudaGraphNode_t node, const struct cudaKernelNodeParams* pNodeParams) {
    cudaError_t r;
    begin_func(cudaGraphKernelNodeSetParams);
    r = so_cudaGraphKernelNodeSetParams(node, pNodeParams);
    end_func(cudaGraphKernelNodeSetParams);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaGraphKernelNodeGetParams(cudaGraphNode_t node, struct cudaKernelNodeParams* pNodeParams) {
    cudaError_t r;
    begin_func(cudaGraphKernelNodeGetParams);
    r = so_cudaGraphKernelNodeGetParams(node, pNodeParams);
    end_func(cudaGraphKernelNodeGetParams);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaGraphInstantiate(cudaGraphExec_t* pGraphExec, cudaGraph_t graph, cudaGraphNode_t* pErrorNode, char* pLogBuffer, size_t bufferSize) {
    cudaError_t r;
    begin_func(cudaGraphInstantiate);
    r = so_cudaGraphInstantiate(pGraphExec , graph ,  pErrorNode, pLogBuffer, bufferSize);
    end_func(cudaGraphInstantiate);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaGraphHostNodeSetParams(cudaGraphNode_t node, const struct cudaHostNodeParams* pNodeParams) {
    cudaError_t r;
    begin_func(cudaGraphHostNodeSetParams);
    r = so_cudaGraphHostNodeSetParams(node , pNodeParams);
    end_func(cudaGraphHostNodeSetParams);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaGraphHostNodeGetParams(cudaGraphNode_t node, struct cudaHostNodeParams* pNodeParams) {
    cudaError_t r;
    begin_func(cudaGraphHostNodeGetParams);
    r = so_cudaGraphHostNodeGetParams(node , pNodeParams);
    end_func(cudaGraphHostNodeGetParams);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaGraphGetRootNodes(cudaGraph_t graph, cudaGraphNode_t* pRootNodes, size_t* pNumRootNodes) {
    cudaError_t r;
    begin_func(cudaGraphGetRootNodes);
    r = so_cudaGraphGetRootNodes(graph, pRootNodes, pNumRootNodes);
    end_func(cudaGraphGetRootNodes);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaGraphGetNodes(cudaGraph_t graph, cudaGraphNode_t* nodes, size_t* numNodes) {
    cudaError_t r;
    begin_func(cudaGraphGetNodes);
    r = so_cudaGraphGetNodes(graph, nodes, numNodes);
    end_func(cudaGraphGetNodes);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaGraphGetEdges(cudaGraph_t graph, cudaGraphNode_t* from, cudaGraphNode_t* to, size_t* numEdges) {
    cudaError_t r;
    begin_func(cudaGraphGetEdges);
    r = so_cudaGraphGetEdges(graph, from, to, numEdges);
    end_func(cudaGraphGetEdges);
    checkCudaErrors(r);
    return r;
}

/*cudaError_t cudaGraphExecKernelNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const struct cudaKernelNodeParams* pNodeParams) {
    cudaError_t r;
    begin_func();
    printf("cudaGraphExecKernelNodeSetParams\n");
    cudaError_t r =(*(cudaError_t(*)(cudaGraphExec_t , cudaGraphNode_t, const struct cudaKernelNodeParams*))(func[46]))(hGraphExec, node, pNodeParams);
    end_func();
    checkCudaErrors(r);
    return r;
}*/

cudaError_t cudaGraphExecDestroy(cudaGraphExec_t graphExec) {
    cudaError_t r;
    begin_func(cudaGraphExecDestroy);
    r = so_cudaGraphExecDestroy(graphExec);
    end_func(cudaGraphExecDestroy);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaGraphDestroyNode(cudaGraphNode_t node) {
    cudaError_t r;
    begin_func(cudaGraphDestroyNode);
    r = so_cudaGraphDestroyNode(node);
    end_func(cudaGraphDestroyNode);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaGraphDestroy(cudaGraph_t graph) {
    cudaError_t r;
    begin_func(cudaGraphDestroy);
    r = so_cudaGraphDestroy(graph);
    end_func(cudaGraphDestroy);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaGraphCreate(cudaGraph_t* pGraph, unsigned int flags) {
    cudaError_t r;
    begin_func(cudaGraphCreate);
    r = so_cudaGraphCreate(pGraph,  flags);
    end_func(cudaGraphCreate);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaGraphClone(cudaGraph_t* pGraphClone, cudaGraph_t originalGraph) {
    cudaError_t r;
    begin_func(cudaGraphClone);
    r = so_cudaGraphClone(pGraphClone,  originalGraph);
    end_func(cudaGraphClone);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaGraphChildGraphNodeGetGraph(cudaGraphNode_t node, cudaGraph_t* pGraph) {
    cudaError_t r;
    begin_func(cudaGraphChildGraphNodeGetGraph);
    r = so_cudaGraphChildGraphNodeGetGraph(node,  pGraph);
    end_func(cudaGraphChildGraphNodeGetGraph);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaGraphAddMemsetNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, cudaGraphNode_t* pDependencies, size_t numDependencies, const struct cudaMemsetParams* pMemsetParams) {
    cudaError_t r;
    begin_func(cudaGraphAddMemsetNode);
    r = so_cudaGraphAddMemsetNode(pGraphNode,  graph, pDependencies, numDependencies, pMemsetParams);
    end_func(cudaGraphAddMemsetNode);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaGraphAddMemcpyNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, cudaGraphNode_t* pDependencies, size_t numDependencies, const struct cudaMemcpy3DParms* pCopyParams) {
    cudaError_t r;
    begin_func(cudaGraphAddMemcpyNode);
    r = so_cudaGraphAddMemcpyNode(pGraphNode,  graph, pDependencies, numDependencies, pCopyParams);
    end_func(cudaGraphAddMemcpyNode);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaGraphAddKernelNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, cudaGraphNode_t* pDependencies, size_t numDependencies, const struct cudaKernelNodeParams* pNodeParams) {
    cudaError_t r;
    begin_func(cudaGraphAddKernelNode);
    r = so_cudaGraphAddKernelNode(pGraphNode,  graph, pDependencies, numDependencies, pNodeParams);
    end_func(cudaGraphAddKernelNode);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaGraphAddHostNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, cudaGraphNode_t* pDependencies, size_t numDependencies, const struct cudaHostNodeParams* pNodeParams) {
    cudaError_t r;
    begin_func(cudaGraphAddHostNode);
    r = so_cudaGraphAddHostNode(pGraphNode,  graph, pDependencies, numDependencies, pNodeParams);
    end_func(cudaGraphAddHostNode);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaGraphAddEmptyNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, cudaGraphNode_t* pDependencies, size_t numDependencies) {
    cudaError_t r;
    begin_func(cudaGraphAddEmptyNode);
    r = so_cudaGraphAddEmptyNode(pGraphNode,  graph, pDependencies, numDependencies);
    end_func(cudaGraphAddEmptyNode);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaGraphAddDependencies(cudaGraph_t graph, cudaGraphNode_t* from, cudaGraphNode_t* to, size_t numDependencies) {
    cudaError_t r;
    begin_func(cudaGraphAddDependencies);
    r = so_cudaGraphAddDependencies(graph,  from, to, numDependencies);
    end_func(cudaGraphAddDependencies);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaGraphAddChildGraphNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, cudaGraphNode_t* pDependencies, size_t numDependencies, cudaGraph_t childGraph) {
    cudaError_t r;
    begin_func(cudaGraphAddChildGraphNode);
    r = so_cudaGraphAddChildGraphNode(pGraphNode,  graph, pDependencies, numDependencies, childGraph);
    end_func(cudaGraphAddChildGraphNode);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaRuntimeGetVersion(int* runtimeVersion) {
    cudaError_t r;
    begin_func(cudaRuntimeGetVersion);
    r = so_cudaRuntimeGetVersion(runtimeVersion);
    end_func(cudaRuntimeGetVersion);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaDriverGetVersion(int* driverVersion) {
    cudaError_t r;
    begin_func(cudaDriverGetVersion);
    r = so_cudaDriverGetVersion(driverVersion);
    end_func(cudaDriverGetVersion);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaGetSurfaceObjectResourceDesc(struct cudaResourceDesc* pResDesc, cudaSurfaceObject_t surfObject) {
    cudaError_t r;
    begin_func(cudaGetSurfaceObjectResourceDesc);
    r = so_cudaGetSurfaceObjectResourceDesc(pResDesc,  surfObject);
    end_func(cudaGetSurfaceObjectResourceDesc);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaDestroySurfaceObject(cudaSurfaceObject_t surfObject) {
    cudaError_t r;
    begin_func(cudaDestroySurfaceObject);
    r = so_cudaDestroySurfaceObject(surfObject);
    end_func(cudaDestroySurfaceObject);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaCreateSurfaceObject(cudaSurfaceObject_t* pSurfObject, const struct cudaResourceDesc* pResDesc) {
    cudaError_t r;
    begin_func(cudaCreateSurfaceObject);
    r = so_cudaCreateSurfaceObject(pSurfObject, pResDesc);
    end_func(cudaCreateSurfaceObject);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaGetTextureObjectTextureDesc(struct cudaTextureDesc* pTexDesc, cudaTextureObject_t texObject) {
    cudaError_t r;
    begin_func(cudaGetTextureObjectTextureDesc);
    r = so_cudaGetTextureObjectTextureDesc(pTexDesc, texObject);
    end_func(cudaGetTextureObjectTextureDesc);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaGetTextureObjectResourceViewDesc(struct cudaResourceViewDesc* pResViewDesc, cudaTextureObject_t texObject) {
    cudaError_t r;
    begin_func(cudaGetTextureObjectResourceViewDesc);
    r = so_cudaGetTextureObjectResourceViewDesc(pResViewDesc, texObject);
    end_func(cudaGetTextureObjectResourceViewDesc);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaGetTextureObjectResourceDesc(struct cudaResourceDesc* pResDesc, cudaTextureObject_t texObject) {
    cudaError_t r;
    begin_func(cudaGetTextureObjectResourceDesc);
    r = so_cudaGetTextureObjectResourceDesc(pResDesc, texObject);
    end_func(cudaGetTextureObjectResourceDesc);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaGetChannelDesc(struct cudaChannelFormatDesc* desc, cudaArray_const_t array) {
    cudaError_t r;
    begin_func(cudaGetChannelDesc);
    r = so_cudaGetChannelDesc(desc, array);
    end_func(cudaGetChannelDesc);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaDestroyTextureObject(cudaTextureObject_t texObject) {
    cudaError_t r;
    begin_func(cudaDestroyTextureObject);
    r = so_cudaDestroyTextureObject(texObject);
    end_func(cudaDestroyTextureObject);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaCreateTextureObject(cudaTextureObject_t* pTexObject, const struct cudaResourceDesc* pResDesc, const struct cudaTextureDesc* pTexDesc, const struct cudaResourceViewDesc* pResViewDesc) {
    cudaError_t r;
    begin_func(cudaCreateTextureObject);
    r = so_cudaCreateTextureObject(pTexObject, pResDesc, pTexDesc, pResViewDesc);
    end_func(cudaCreateTextureObject);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaGraphicsUnregisterResource(cudaGraphicsResource_t resource) {
    cudaError_t r;
    begin_func(cudaGraphicsUnregisterResource);
    r = so_cudaGraphicsUnregisterResource(resource);
    end_func(cudaGraphicsUnregisterResource);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaGraphicsUnmapResources(int count, cudaGraphicsResource_t* resources, cudaStream_t stream) {
    cudaError_t r;
    begin_func(cudaGraphicsUnmapResources);
    r = so_cudaGraphicsUnmapResources(count, resources, stream);
    end_func(cudaGraphicsUnmapResources);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaGraphicsSubResourceGetMappedArray(cudaArray_t* array, cudaGraphicsResource_t resource, unsigned int arrayIndex, unsigned int mipLevel) {
    cudaError_t r;
    begin_func(cudaGraphicsSubResourceGetMappedArray);
    r = so_cudaGraphicsSubResourceGetMappedArray(array, resource, arrayIndex, mipLevel);
    end_func(cudaGraphicsSubResourceGetMappedArray);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaGraphicsResourceSetMapFlags(cudaGraphicsResource_t resource, unsigned int  flags) {
    cudaError_t r;
    begin_func(cudaGraphicsResourceSetMapFlags);
    r = so_cudaGraphicsResourceSetMapFlags(resource, flags);
    end_func(cudaGraphicsResourceSetMapFlags);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaGraphicsResourceGetMappedPointer(void** devPtr, size_t* size, cudaGraphicsResource_t resource) {
    cudaError_t r;
    begin_func(cudaGraphicsResourceGetMappedPointer);
    r = so_cudaGraphicsResourceGetMappedPointer(devPtr, size, resource);
    end_func(cudaGraphicsResourceGetMappedPointer);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaGraphicsResourceGetMappedMipmappedArray(cudaMipmappedArray_t* mipmappedArray, cudaGraphicsResource_t resource) {
    cudaError_t r;
    begin_func(cudaGraphicsResourceGetMappedMipmappedArray);
    r = so_cudaGraphicsResourceGetMappedMipmappedArray(mipmappedArray, resource);
    end_func(cudaGraphicsResourceGetMappedMipmappedArray);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaGraphicsMapResources(int count, cudaGraphicsResource_t* resources, cudaStream_t stream) {
    cudaError_t r;
    begin_func(cudaGraphicsMapResources);
    r = so_cudaGraphicsMapResources(count, resources, stream);
    end_func(cudaGraphicsMapResources);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags) {
    cudaError_t r;
    begin_func(cudaDeviceEnablePeerAccess);
    r = so_cudaDeviceEnablePeerAccess(peerDevice, flags);
    end_func(cudaDeviceEnablePeerAccess);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaDeviceDisablePeerAccess(int peerDevice) {
    cudaError_t r;
    begin_func(cudaDeviceDisablePeerAccess);
    r = so_cudaDeviceDisablePeerAccess(peerDevice);
    end_func(cudaDeviceDisablePeerAccess);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaDeviceCanAccessPeer(int* canAccessPeer, int device, int peerDevice) {
    cudaError_t r;
    begin_func(cudaDeviceCanAccessPeer);
    r = so_cudaDeviceCanAccessPeer(canAccessPeer, device, peerDevice);
    end_func(cudaDeviceCanAccessPeer);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaPointerGetAttributes(struct cudaPointerAttributes* attributes, const void* ptr) {
    cudaError_t r;
    begin_func(cudaPointerGetAttributes);
    r = so_cudaPointerGetAttributes(attributes, ptr);
    end_func(cudaPointerGetAttributes);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaMemset3DAsync(struct cudaPitchedPtr pitchedDevPtr, int  value, struct cudaExtent extent, cudaStream_t stream) {
    cudaError_t r;
    begin_func(cudaMemset3DAsync);
    r = so_cudaMemset3DAsync(pitchedDevPtr, value, extent, stream);
    end_func(cudaMemset3DAsync);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaMemset3D(struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent) {
    cudaError_t r;
    begin_func(cudaMemset3D);
    r = so_cudaMemset3D(pitchedDevPtr, value, extent);
    end_func(cudaMemset3D);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaMemset2DAsync(void* devPtr, size_t pitch, int  value, size_t width, size_t height, cudaStream_t stream) {
    cudaError_t r;
    begin_func(cudaMemset2DAsync);
    r = so_cudaMemset2DAsync(devPtr, pitch, value, width, height, stream);
    end_func(cudaMemset2DAsync);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaMemset2D(void* devPtr, size_t pitch, int  value, size_t width, size_t height) {
    cudaError_t r;
    begin_func(cudaMemset2D);
    r = so_cudaMemset2D(devPtr, pitch, value, width, height);
    end_func(cudaMemset2D);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaMemset(void* devPtr, int value, size_t count) {
    cudaError_t r;
    begin_func(cudaMemset);
    r = so_cudaMemset(devPtr, value, count);
    end_func(cudaMemset);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaMemcpyToSymbolAsync(const void* symbol, const void* src, size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream) {
    cudaError_t r;
    begin_func(cudaMemcpyToSymbolAsync);
    r = so_cudaMemcpyToSymbolAsync(symbol, src, count, offset, kind, stream);
    end_func(cudaMemcpyToSymbolAsync);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaMemcpyToSymbol(const void* symbol, const void* src, size_t count, size_t offset, enum cudaMemcpyKind kind) {
    cudaError_t r;
    begin_func(cudaMemcpyToSymbol);
    r = so_cudaMemcpyToSymbol(symbol, src, count, offset, kind);
    end_func(cudaMemcpyToSymbol);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaMemcpyPeerAsync(void* dst, int dstDevice, const void* src, int srcDevice, size_t count, cudaStream_t stream) {
    cudaError_t r;
    begin_func(cudaMemcpyPeerAsync);
    r = so_cudaMemcpyPeerAsync(dst, dstDevice, src, srcDevice, count, stream);
    end_func(cudaMemcpyPeerAsync);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaMemcpyPeer(void* dst, int dstDevice, const void* src, int srcDevice, size_t count) {
    cudaError_t r;
    begin_func(cudaMemcpyPeer);
    r = so_cudaMemcpyPeer(dst, dstDevice, src, srcDevice, count);
    end_func(cudaMemcpyPeer);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaMemcpyFromSymbolAsync(void* dst, const void* symbol, size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream) {
    cudaError_t r;
    begin_func(cudaMemcpyFromSymbolAsync);
    r = so_cudaMemcpyFromSymbolAsync(dst, symbol, count, offset,  kind, stream);
    end_func(cudaMemcpyFromSymbolAsync);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaMemcpyFromSymbol(void* dst, const void* symbol, size_t count, size_t offset , enum cudaMemcpyKind kind) {
    cudaError_t r;
    begin_func(cudaMemcpyFromSymbol);
    r = so_cudaMemcpyFromSymbol(dst, symbol, count, offset, kind);
    end_func(cudaMemcpyFromSymbol);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaMemcpy3DPeerAsync(const struct cudaMemcpy3DPeerParms* p, cudaStream_t stream) {
    cudaError_t r;
    begin_func(cudaMemcpy3DPeerAsync);
    r = so_cudaMemcpy3DPeerAsync(p, stream);
    end_func(cudaMemcpy3DPeerAsync);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaMemcpy3DPeer(const struct cudaMemcpy3DPeerParms* p) {
    cudaError_t r;
    begin_func(cudaMemcpy3DPeer);
    r = so_cudaMemcpy3DPeer(p);
    end_func(cudaMemcpy3DPeer);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaMemcpy3DAsync(const struct cudaMemcpy3DParms* p, cudaStream_t stream) {
    cudaError_t r;
    begin_func(cudaMemcpy3DAsync);
    r = so_cudaMemcpy3DAsync(p, stream);
    end_func(cudaMemcpy3DAsync);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaMemcpy3D(const struct cudaMemcpy3DParms* p) {
    cudaError_t r;
    begin_func(cudaMemcpy3D);
    r = so_cudaMemcpy3D(p);
    end_func(cudaMemcpy3D);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaMemcpy2DToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream) {
    cudaError_t r;
    begin_func(cudaMemcpy2DToArrayAsync);
    r = so_cudaMemcpy2DToArrayAsync(dst, wOffset, hOffset, src, spitch, width, height, kind, stream);
    end_func(cudaMemcpy2DToArrayAsync);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaMemcpy2DToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind) {
    cudaError_t r;
    begin_func(cudaMemcpy2DToArray);
    r = so_cudaMemcpy2DToArray(dst, wOffset, hOffset, src, spitch, width, height, kind);
    end_func(cudaMemcpy2DToArray);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaMemcpy2DFromArrayAsync(void* dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream) {
    cudaError_t r;
    begin_func(cudaMemcpy2DFromArrayAsync);
    r = so_cudaMemcpy2DFromArrayAsync(dst, dpitch, src, wOffset, hOffset, width, height, kind, stream);
    end_func(cudaMemcpy2DFromArrayAsync);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaMemcpy2DFromArray(void* dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind) {
    cudaError_t r;
    begin_func(cudaMemcpy2DFromArray);
    r = so_cudaMemcpy2DFromArray(dst, dpitch, src, wOffset, hOffset, width, height, kind);
    end_func(cudaMemcpy2DFromArray);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaMemcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream) {
    cudaError_t r;
    begin_func(cudaMemcpy2DAsync);
    r = so_cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, kind, stream);
    end_func(cudaMemcpy2DAsync);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaMemcpy2DArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, enum cudaMemcpyKind kind) {
    cudaError_t r;
    begin_func(cudaMemcpy2DArrayToArray);
    r = so_cudaMemcpy2DArrayToArray(dst, wOffsetDst, hOffsetDst, src, wOffsetSrc, hOffsetSrc, width, height, kind);
    end_func(cudaMemcpy2DArrayToArray);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind) {
    cudaError_t r;
    begin_func(cudaMemcpy2D);
    r = so_cudaMemcpy2D(dst, dpitch, src, spitch, width, height, kind);
    end_func(cudaMemcpy2D);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind) {
    cudaError_t r;
    begin_func(cudaMemcpy);
    r = so_cudaMemcpy(dst,(const void*)src, count, kind);
    end_func(cudaMemcpy);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaMemRangeGetAttributes(void** data, size_t* dataSizes, enum cudaMemRangeAttribute *attributes, size_t numAttributes, const void* devPtr, size_t count) {
    cudaError_t r;
    begin_func(cudaMemRangeGetAttributes);
    r = so_cudaMemRangeGetAttributes(data, dataSizes, attributes, numAttributes , devPtr, count);
    end_func(cudaMemRangeGetAttributes);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaMemRangeGetAttribute(void* data, size_t dataSize, enum cudaMemRangeAttribute attribute, const void* devPtr, size_t count) {
    cudaError_t r;
    begin_func(cudaMemRangeGetAttribute);
    r = so_cudaMemRangeGetAttribute(data, dataSize, attribute, devPtr, count);
    end_func(cudaMemRangeGetAttribute);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaMemPrefetchAsync(const void* devPtr, size_t count, int  dstDevice, cudaStream_t stream) {
    cudaError_t r;
    begin_func(cudaMemPrefetchAsync);
    r = so_cudaMemPrefetchAsync(devPtr, count, dstDevice, stream);
    end_func(cudaMemPrefetchAsync);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaMemAdvise(const void* devPtr, size_t count, enum cudaMemoryAdvise advice, int  device) {
    cudaError_t r;
    begin_func(cudaMemAdvise);
    r = so_cudaMemAdvise(devPtr, count, advice, device);
    end_func(cudaMemAdvise);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaHostUnregister(void* ptr) {
    cudaError_t r;
    begin_func(cudaHostUnregister);
    r = so_cudaHostUnregister(ptr);
    end_func(cudaHostUnregister);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaHostRegister(void* ptr, size_t size, unsigned int  flags) {
    cudaError_t r;
    begin_func(cudaHostRegister);
    r = so_cudaHostRegister(ptr, size, flags);
    end_func(cudaHostRegister);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaHostGetFlags(unsigned int* pFlags, void* pHost) {
    cudaError_t r;
    begin_func(cudaHostGetFlags);
    r = so_cudaHostGetFlags(pFlags, pHost);
    end_func(cudaHostGetFlags);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaGetSymbolSize(size_t* size, const void* symbol) {
    cudaError_t r;
    begin_func(cudaGetSymbolSize);
    r = so_cudaGetSymbolSize(size, symbol);
    end_func(cudaGetSymbolSize);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaGetSymbolAddress(void** devPtr, const void* symbol) {
    cudaError_t r;
    begin_func(cudaGetSymbolAddress);
    r = so_cudaGetSymbolAddress(devPtr, symbol);
    end_func(cudaGetSymbolAddress);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaGetMipmappedArrayLevel(cudaArray_t* levelArray, cudaMipmappedArray_const_t mipmappedArray, unsigned int level) {
    cudaError_t r;
    begin_func(cudaGetMipmappedArrayLevel);
    r = so_cudaGetMipmappedArrayLevel(levelArray, mipmappedArray, level);
    end_func(cudaGetMipmappedArrayLevel);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaFreeMipmappedArray(cudaMipmappedArray_t mipmappedArray) {
    cudaError_t r;
    begin_func(cudaFreeMipmappedArray);
    r = so_cudaFreeMipmappedArray(mipmappedArray);
    end_func(cudaFreeMipmappedArray);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaFreeArray(cudaArray_t array) {
    cudaError_t r;
    begin_func(cudaFreeArray);
    r = so_cudaFreeArray(array);
    end_func(cudaFreeArray);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaArrayGetInfo(struct cudaChannelFormatDesc* desc, struct cudaExtent* extent, unsigned int* flags, cudaArray_t array) {
    cudaError_t r;
    begin_func(cudaArrayGetInfo);
    r = so_cudaArrayGetInfo(desc, extent, flags, array);
    end_func(cudaArrayGetInfo);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, const void* func, int blockSize, size_t dynamicSMemSize) {
    cudaError_t r;
    begin_func(cudaOccupancyMaxActiveBlocksPerMultiprocessor);
    r = so_cudaOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, func, blockSize, dynamicSMemSize);
    end_func(cudaOccupancyMaxActiveBlocksPerMultiprocessor);
    checkCudaErrors(r);
    return r;
}

/*
cudaError_t cudaSetDoubleForHost(double* d) {
    cudaError_t r;
    begin_func(cudaSetDoubleForHost);
    r = so_cudaSetDoubleForHost(d);
    end_func(cudaSetDoubleForHost);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaSetDoubleForDevice(double* d) {
    cudaError_t r;
    begin_func(cudaSetDoubleForDevice);
    r = so_cudaSetDoubleForDevice(d);
    end_func(cudaSetDoubleForDevice);
    checkCudaErrors(r);
    return r;
}
*/

cudaError_t cudaLaunchHostFunc(cudaStream_t stream, cudaHostFn_t fn, void* userData) {
    cudaError_t r;
    begin_func(cudaLaunchHostFunc);
    r = so_cudaLaunchHostFunc(stream, fn, userData);
    end_func(cudaLaunchHostFunc);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaLaunchCooperativeKernelMultiDevice(struct cudaLaunchParams* launchParamsList, unsigned int  numDevices, unsigned int  flags) {
    cudaError_t r;
    begin_func(cudaLaunchCooperativeKernelMultiDevice);
    assert(0); //TODO
    r = so_cudaLaunchCooperativeKernelMultiDevice(launchParamsList, numDevices, flags);
    end_func(cudaLaunchCooperativeKernelMultiDevice);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaLaunchCooperativeKernel(const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream) {
    cudaError_t r;
    begin_func(cudaLaunchCooperativeKernel);
    assert(0); //TODO
    r = so_cudaLaunchCooperativeKernel(func,  gridDim,  blockDim,  args, sharedMem, stream);
    end_func(cudaLaunchCooperativeKernel);
    checkCudaErrors(r);
    return r;
}

/*void* cudaGetParameterBufferV2(void* func, dim3 gridDimension, dim3 blockDimension, unsigned int  sharedMemSize) {
    cudaError_t r;
    begin_func();
    printf("cudaGetParameterBufferV2\n");
    void* r =(*(void*(*)(void* , dim3 , dim3 , unsigned int))(func2[5]))(func, gridDimension, blockDimension, sharedMemSize);
    end_func();checkCudaErrors(0);
    return r;
}

void* cudaGetParameterBuffer(size_t alignment, size_t size) {
    cudaError_t r;
    begin_func();
    printf("cudaGetParameterBuffer\n");
    void* r =(*(void*(*)(size_t , size_t))(func[125]))(alignment, size);
    end_func();checkCudaErrors(0);
    return r;
}*/

cudaError_t cudaFuncSetSharedMemConfig(const void* func, enum cudaSharedMemConfig config) {
    cudaError_t r;
    begin_func(cudaFuncSetSharedMemConfig);
    r = so_cudaFuncSetSharedMemConfig(func, config);
    end_func(cudaFuncSetSharedMemConfig);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaFuncSetCacheConfig(const void* func, enum cudaFuncCache cacheConfig) {
    cudaError_t r;
    begin_func(cudaFuncSetCacheConfig);
    r = so_cudaFuncSetCacheConfig(func, cacheConfig);
    end_func(cudaFuncSetCacheConfig);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaFuncSetAttribute(const void* func, enum cudaFuncAttribute attr, int  value) {
    cudaError_t r;
    begin_func(cudaFuncSetAttribute);
    r = so_cudaFuncSetAttribute(func, attr, value);
    end_func(cudaFuncSetAttribute);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaWaitExternalSemaphoresAsync(const cudaExternalSemaphore_t* extSemArray, const struct cudaExternalSemaphoreWaitParams* paramsArray, unsigned int numExtSems, cudaStream_t stream) {
    cudaError_t r;
    begin_func(cudaWaitExternalSemaphoresAsync);
    r = so_cudaWaitExternalSemaphoresAsync(extSemArray, paramsArray, numExtSems, stream);
    end_func(cudaWaitExternalSemaphoresAsync);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaSignalExternalSemaphoresAsync(const cudaExternalSemaphore_t* extSemArray, const struct cudaExternalSemaphoreSignalParams* paramsArray, unsigned int numExtSems, cudaStream_t stream) {
    cudaError_t r;
    begin_func(cudaSignalExternalSemaphoresAsync);
    r = so_cudaSignalExternalSemaphoresAsync(extSemArray, paramsArray, numExtSems, stream);
    end_func(cudaSignalExternalSemaphoresAsync);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaImportExternalSemaphore(cudaExternalSemaphore_t* extSem_out, const struct cudaExternalSemaphoreHandleDesc* semHandleDesc) {
    cudaError_t r;
    begin_func(cudaImportExternalSemaphore);
    r = so_cudaImportExternalSemaphore(extSem_out, semHandleDesc);
    end_func(cudaImportExternalSemaphore);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaImportExternalMemory(cudaExternalMemory_t* extMem_out, const struct cudaExternalMemoryHandleDesc* memHandleDesc) {
    cudaError_t r;
    begin_func(cudaImportExternalMemory);
    r = so_cudaImportExternalMemory(extMem_out, memHandleDesc);
    end_func(cudaImportExternalMemory);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaExternalMemoryGetMappedMipmappedArray(cudaMipmappedArray_t* mipmap, cudaExternalMemory_t extMem, const struct cudaExternalMemoryMipmappedArrayDesc* mipmapDesc) {
    cudaError_t r;
    begin_func(cudaExternalMemoryGetMappedMipmappedArray);
    r = so_cudaExternalMemoryGetMappedMipmappedArray(mipmap, extMem, mipmapDesc);
    end_func(cudaExternalMemoryGetMappedMipmappedArray);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaExternalMemoryGetMappedBuffer(void** devPtr, cudaExternalMemory_t extMem, const struct cudaExternalMemoryBufferDesc* bufferDesc) {
    cudaError_t r;
    begin_func(cudaExternalMemoryGetMappedBuffer);
    r = so_cudaExternalMemoryGetMappedBuffer(devPtr, extMem, bufferDesc);
    end_func(cudaExternalMemoryGetMappedBuffer);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaDestroyExternalSemaphore(cudaExternalSemaphore_t extSem) {
    cudaError_t r;
    begin_func(cudaDestroyExternalSemaphore);
    r = so_cudaDestroyExternalSemaphore(extSem);
    end_func(cudaDestroyExternalSemaphore);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaDestroyExternalMemory(cudaExternalMemory_t extMem) {
    cudaError_t r;
    begin_func(cudaDestroyExternalMemory);
    r = so_cudaDestroyExternalMemory(extMem);
    end_func(cudaDestroyExternalMemory);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaEventSynchronize(cudaEvent_t event) {
    cudaError_t r;
    begin_func(cudaEventSynchronize);
    r = so_cudaEventSynchronize(event);
    end_func(cudaEventSynchronize);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaEventQuery(cudaEvent_t event) {
    cudaError_t r;
    begin_func(cudaEventQuery);
    r = so_cudaEventQuery(event);
    end_func(cudaEventQuery);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t end) {
    cudaError_t r;
    begin_func(cudaEventElapsedTime);
    r = so_cudaEventElapsedTime(ms, start, end);
    end_func(cudaEventElapsedTime);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaEventCreate(cudaEvent_t* event) {
    cudaError_t r;
    begin_func(cudaEventCreate);
    r = so_cudaEventCreate(event);
    end_func(cudaEventCreate);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int  flags) {
    cudaError_t r;
    begin_func(cudaStreamWaitEvent);
    r = so_cudaStreamWaitEvent(stream, event, flags);
    end_func(cudaStreamWaitEvent);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaStreamQuery(cudaStream_t stream) {
    cudaError_t r;
    begin_func(cudaStreamQuery);
    r = so_cudaStreamQuery(stream);
    end_func(cudaStreamQuery);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaStreamIsCapturing(cudaStream_t stream, enum cudaStreamCaptureStatus *pCaptureStatus) {
    cudaError_t r;
    begin_func(cudaStreamIsCapturing);
    r = so_cudaStreamIsCapturing(stream, pCaptureStatus);
    end_func(cudaStreamIsCapturing);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaStreamGetPriority(cudaStream_t hStream, int* priority) {
    cudaError_t r;
    begin_func(cudaStreamGetPriority);
    r = so_cudaStreamGetPriority(hStream, priority);
    end_func(cudaStreamGetPriority);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaStreamGetFlags(cudaStream_t hStream, unsigned int* flags) {
    cudaError_t r;
    begin_func(cudaStreamGetFlags);
    r = so_cudaStreamGetFlags(hStream, flags);
    end_func(cudaStreamGetFlags);
    checkCudaErrors(r);
    return r;
}

/*cudaError_t cudaStreamGetCaptureInfo(cudaStream_t stream, enum cudaStreamCaptureStatus ** pCaptureStatus, unsigned long long* pId) {
    cudaError_t r;
    begin_func(cudaStreamGetCaptureInfo);
    cudaError_t r =(*(cudaError_t(*)(cudaStream_t , enum cudaStreamCaptureStatus ** , unsigned long long*))(func[146]))(stream, pCaptureStatus, pId);
    end_func(cudaStreamGetCaptureInfo);
    checkCudaErrors(r);
    return r;
}*/

cudaError_t cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t* pGraph) {
    cudaError_t r;
    begin_func(cudaStreamEndCapture);
    r = so_cudaStreamEndCapture(stream, pGraph);
    end_func(cudaStreamEndCapture);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaStreamCreate(cudaStream_t* pStream) {
    cudaError_t r;
    begin_func(cudaStreamCreate);
    r = so_cudaStreamCreate(pStream);
    end_func(cudaStreamCreate);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaStreamBeginCapture(cudaStream_t stream) {
    cudaError_t r;
    begin_func(cudaStreamBeginCapture);
    r = so_cudaStreamBeginCapture(stream);
    end_func(cudaStreamBeginCapture);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaStreamAttachMemAsync(cudaStream_t stream, void* devPtr, size_t length, unsigned int  flags) {
    cudaError_t r;
    begin_func(cudaStreamAttachMemAsync);
    r = so_cudaStreamAttachMemAsync(stream, devPtr, length, flags);
    end_func(cudaStreamAttachMemAsync);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaStreamAddCallback(cudaStream_t stream, cudaStreamCallback_t callback, void* userData, unsigned int  flags) {
    cudaError_t r;
    begin_func(cudaStreamAddCallback);
    assert(0); //TODO
    r = so_cudaStreamAddCallback(stream,  callback, userData, flags);
    end_func(cudaStreamAddCallback);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaPeekAtLastError() {
    cudaError_t r;
    begin_func(cudaPeekAtLastError);
    r = so_cudaPeekAtLastError();
    end_func(cudaPeekAtLastError);
    checkCudaErrors(r);
    return r;
}

const char* cudaGetErrorString(cudaError_t err) {
    const char* r;
    begin_func(cudaGetErrorString);
    r = so_cudaGetErrorString(err);
    end_func(cudaGetErrorString);
    return r;
}

const char* cudaGetErrorName(cudaError_t err) {
    const char* r;
    begin_func(cudaGetErrorName);
    r = so_cudaGetErrorName(err);
    end_func(cudaGetErrorName);
    return r;
}

cudaError_t cudaSetValidDevices(int* device_arr, int  len) {
    cudaError_t r;
    begin_func(cudaSetValidDevices);
    r = so_cudaSetValidDevices(device_arr, len);
    end_func(cudaSetValidDevices);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaSetDeviceFlags(unsigned int  flags) {
    cudaError_t r;
    begin_func(cudaSetDeviceFlags);
    r = so_cudaSetDeviceFlags(flags);
    end_func(cudaSetDeviceFlags);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaIpcOpenMemHandle(void** devPtr, cudaIpcMemHandle_t hand, unsigned int  flags) {
    cudaError_t r;
    begin_func(cudaIpcOpenMemHandle);
    r = so_cudaIpcOpenMemHandle(devPtr, hand, flags);
    end_func(cudaIpcOpenMemHandle);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaIpcOpenEventHandle(cudaEvent_t* event,  cudaIpcEventHandle_t hand) {
    cudaError_t r;
    begin_func(cudaIpcOpenEventHandle);
    r = so_cudaIpcOpenEventHandle(event, hand);
    end_func(cudaIpcOpenEventHandle);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaIpcGetMemHandle(cudaIpcMemHandle_t* handle, void* devPtr) {
    cudaError_t r;
    begin_func(cudaIpcGetMemHandle);
    r = so_cudaIpcGetMemHandle(handle, devPtr);
    end_func(cudaIpcGetMemHandle);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaIpcGetEventHandle(cudaIpcEventHandle_t* handle, cudaEvent_t event) {
    cudaError_t r;
    begin_func(cudaIpcGetEventHandle);
    r = so_cudaIpcGetEventHandle(handle, event);
    end_func(cudaIpcGetEventHandle);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaIpcCloseMemHandle(void* devPtr) {
    cudaError_t r;
    begin_func(cudaIpcCloseMemHandle);
    r = so_cudaIpcCloseMemHandle(devPtr);
    end_func(cudaIpcCloseMemHandle);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaGetDeviceFlags(unsigned int* flags) {
    cudaError_t r;
    begin_func(cudaGetDeviceFlags);
    r = so_cudaGetDeviceFlags(flags);
    end_func(cudaGetDeviceFlags);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaDeviceSynchronize() {
    cudaError_t r;
    begin_func(cudaDeviceSynchronize);
    r = so_cudaDeviceSynchronize();
    end_func(cudaDeviceSynchronize);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaDeviceSetSharedMemConfig(enum cudaSharedMemConfig config) {
    cudaError_t r;
    begin_func(cudaDeviceSetSharedMemConfig);
    r = so_cudaDeviceSetSharedMemConfig(config);
    end_func(cudaDeviceSetSharedMemConfig);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaDeviceSetLimit(enum cudaLimit limit, size_t value) {
    cudaError_t r;
    begin_func(cudaDeviceSetLimit);
    r = so_cudaDeviceSetLimit(limit, value);
    end_func(cudaDeviceSetLimit);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaDeviceSetCacheConfig(enum cudaFuncCache cacheConfig) {
    cudaError_t r;
    begin_func(cudaDeviceSetCacheConfig);
    r = so_cudaDeviceSetCacheConfig(cacheConfig);
    end_func(cudaDeviceSetCacheConfig);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaDeviceReset() {
    cudaError_t r;
    begin_func(cudaDeviceReset);
    r = so_cudaDeviceReset();
    end_func(cudaDeviceReset);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaDeviceGetSharedMemConfig(enum cudaSharedMemConfig * pConfig) {
    cudaError_t r;
    begin_func(cudaDeviceGetSharedMemConfig);
    r = so_cudaDeviceGetSharedMemConfig(pConfig);
    end_func(cudaDeviceGetSharedMemConfig);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaDeviceGetPCIBusId(char* pciBusId, int len, int device) {
    cudaError_t r;
    begin_func(cudaDeviceGetPCIBusId);
    r = so_cudaDeviceGetPCIBusId(pciBusId, len, device);
    end_func(cudaDeviceGetPCIBusId);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaDeviceGetP2PAttribute(int* value, enum cudaDeviceP2PAttr attr, int srcDevice, int dstDevice) {
    cudaError_t r;
    begin_func(cudaDeviceGetP2PAttribute);
    r = so_cudaDeviceGetP2PAttribute(value, attr, srcDevice, dstDevice);
    end_func(cudaDeviceGetP2PAttribute);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaDeviceGetLimit(size_t* pValue, enum cudaLimit limit) {
    cudaError_t r;
    begin_func(cudaDeviceGetLimit);
    r = so_cudaDeviceGetLimit(pValue, limit);
    end_func(cudaDeviceGetLimit);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaDeviceGetCacheConfig(enum cudaFuncCache * pCacheConfig) {
    cudaError_t r;
    begin_func(cudaDeviceGetCacheConfig);
    r = so_cudaDeviceGetCacheConfig(pCacheConfig);
    end_func(cudaDeviceGetCacheConfig);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaDeviceGetByPCIBusId(int* device, const char* pciBusId) {
    cudaError_t r;
    begin_func(cudaDeviceGetByPCIBusId);
    r = so_cudaDeviceGetByPCIBusId(device, pciBusId);
    end_func(cudaDeviceGetByPCIBusId);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaChooseDevice(int* device, const struct cudaDeviceProp* prop) {
    cudaError_t r;
    begin_func(cudaChooseDevice);
    r = so_cudaChooseDevice(device, prop);
    end_func(cudaChooseDevice);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaMallocMipmappedArray(cudaMipmappedArray_t* mipmappedArray, const struct cudaChannelFormatDesc* desc, struct cudaExtent extent, unsigned int  numLevels, unsigned int flags) {
    cudaError_t r;
    begin_func(cudaMallocMipmappedArray);
    r = so_cudaMallocMipmappedArray(mipmappedArray, desc, extent, numLevels, flags);
    end_func(cudaMallocMipmappedArray);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaMallocArray(cudaArray_t* array, const struct cudaChannelFormatDesc* desc, size_t width, size_t height, unsigned int  flags) {
    cudaError_t r;
    begin_func(cudaMallocArray);
    r = so_cudaMallocArray(array, desc, width, height, flags);
    end_func(cudaMallocArray);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaMallocHost(void** ptr, size_t size) {
    cudaError_t r;
    begin_func(cudaMallocHost);
    r = so_cudaMallocHost(ptr, size);
    end_func(cudaMallocHost);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaMalloc3DArray(cudaArray_t* array, const struct cudaChannelFormatDesc* desc, struct cudaExtent extent, unsigned int  flags) {
    cudaError_t r;
    begin_func(cudaMalloc3DArray);
    r = so_cudaMalloc3DArray(array, desc, extent, flags);
    end_func(cudaMalloc3DArray);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaMalloc3D(struct cudaPitchedPtr* pitchedDevPtr, struct cudaExtent extent) {
    cudaError_t r;
    begin_func(cudaMalloc3D);
    r = so_cudaMalloc3D(pitchedDevPtr, extent);
    end_func(cudaMalloc3D);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaMallocManaged(void** devPtr, size_t bytesize, unsigned int flags) {
    cudaError_t r;
    begin_func(cudaMallocManaged);
    r = so_cudaMallocManaged(devPtr, bytesize, flags);
    end_func(cudaMallocManaged);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaMallocPitch(void** devPtr, size_t* pitch, size_t width, size_t height) {
    cudaError_t r;
    begin_func(cudaMallocPitch);
    r = so_cudaMallocPitch(devPtr, pitch, width, height);
    end_func(cudaMallocPitch);
    checkCudaErrors(r);
    return r;
}

cudaError_t cudaMemGetInfo(size_t* free , size_t* total) {
    cudaError_t r;
    begin_func(cudaMemGetInfo);
    r = so_cudaMemGetInfo(free, total);
    end_func(cudaMemGetInfo);
    checkCudaErrors(r);
    return r;
}

void __cudaRegisterVar(void **fatCubinHandle, char *hostVar, char *deviceAddress, const char *deviceName, int ext, int size, int constant, int global) {
    begin_func(__cudaRegisterVar);
    so___cudaRegisterVar(fatCubinHandle, hostVar, deviceAddress, deviceName, ext, size, constant, global);
    end_func(__cudaRegisterVar);
}

void __cudaRegisterTexture(void **fatCubinHandle, const struct textureReference *hostVar, const void **deviceAddress, const char *deviceName, int dim, int norm, int ext) {
    begin_func(__cudaRegisterTexture);
    so___cudaRegisterTexture(fatCubinHandle, hostVar, deviceAddress, deviceName, dim, norm, ext);
    end_func(__cudaRegisterTexture);
}

void __cudaRegisterSurface(void **fatCubinHandle, const struct surfaceReference  *hostVar, const void **deviceAddress, const char *deviceName, int dim, int ext) {
    begin_func(__cudaRegisterSurface);
    so___cudaRegisterSurface(fatCubinHandle, hostVar, deviceAddress, deviceName, dim, ext);
    end_func(__cudaRegisterSurface);
}

void __cudaRegisterFunction(void **fatCubinHandle, const char *hostFun, char *deviceFun, const char *deviceName, int thread_limit, uint3 *tid, uint3 *bid, dim3 *bDim, dim3 *gDim, int *wSize) {
    begin_func(__cudaRegisterFunction);
    so___cudaRegisterFunction(fatCubinHandle, hostFun, deviceFun, deviceName, thread_limit, tid, bid, bDim, gDim, wSize);
    end_func(__cudaRegisterFunction);
}

/*
void __cudaRegisterShared(void **fatCubinHandle, void **devicePtr) {
    cudaError_t r;
    begin_func(__cudaRegisterShared);
    r = so___cudaRegisterShared(fatCubinHandle, devicePtr);
    end_func(__cudaRegisterShared);
    checkCudaErrors(r);
    return r;
}

void __cudaRegisterSharedVar(void **fatCubinHandle, void **devicePtr, size_t size, size_t alignment, int storage) {
    cudaError_t r;
    begin_func(__cudaRegisterSharedVar);
    r = so___cudaRegisterSharedVar(fatCubinHandle, devicePtr, size, alignment, storage);
    end_func(__cudaRegisterSharedVar);
    checkCudaErrors(r);
    return r;
}

int __cudaSynchronizeThreads(void** one, void* two) {
    int r;
    begin_func(__cudaSynchronizeThreads);
    r = so___cudaSynchronizeThreads(one, two);
    end_func(__cudaSynchronizeThreads);
    return r;
}

void __cudaTextureFetch(const void* tex, void* index, int integer, void* val) {
    begin_func(__cudaTextureFetch);
    so___cudaTextureFetch(tex, index, integer, val);
    end_func(__cudaTextureFetch);
}

void __cudaMutexOperation(int lock) {
    begin_func(__cudaMutexOperation);
    so___cudaMutexOperation(lock);
    end_func(__cudaMutexOperation);
}

cudaError_t __cudaRegisterDeviceFunction() {
    cudaError_t r;
    begin_func(__cudaRegisterDeviceFunction);
    r = so___cudaRegisterDeviceFunction();
    end_func(__cudaRegisterDeviceFunction);
    checkCudaErrors(r);
    return r;
}
*/

void** __cudaRegisterFatBinary(void* fatCubin) {
    void** r;
    begin_func(__cudaRegisterFatBinary);
    r = so___cudaRegisterFatBinary(fatCubin);
    end_func(__cudaRegisterFatBinary);
    return r;
}

void __cudaUnregisterFatBinary(void** point) {
    begin_func(__cudaUnregisterFatBinary);
    so___cudaUnregisterFatBinary(point);
    end_func(__cudaUnregisterFatBinary);
}

cudaError_t __cudaPopCallConfiguration(dim3 *gridDim, dim3 *blockDim, size_t *sharedMem, void *stream) {
    cudaError_t r;
    begin_func(__cudaPopCallConfiguration);
    r = so___cudaPopCallConfiguration(gridDim, blockDim, sharedMem, stream);
    end_func(__cudaPopCallConfiguration);
    return r;
}

unsigned __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim, size_t sharedMem , void *stream) {
    unsigned r;
    begin_func(__cudaPushCallConfiguration);
    r = so___cudaPushCallConfiguration(gridDim, blockDim, sharedMem, stream);
    end_func(__cudaPushCallConfiguration);
    return r;
}

void __cudaRegisterFatBinaryEnd(void **fatCubinHandle) {
    begin_func(__cudaRegisterFatBinaryEnd);
    so___cudaRegisterFatBinaryEnd(fatCubinHandle);
    end_func(__cudaRegisterFatBinaryEnd);
}


static void dlsym_all_funcs() {
    printf("dlsym all funcs for %s\n", so_dli.dli_fname);
    LDSYM(cudaGetErrorString);
    LDSYM(cudaGetErrorName);
    LDSYM(__cudaRegisterVar);
    LDSYM(__cudaRegisterTexture);
    LDSYM(__cudaRegisterSurface);
    LDSYM(__cudaRegisterFunction);
    LDSYM(__cudaRegisterFatBinary);
    LDSYM(__cudaUnregisterFatBinary);
    LDSYM(__cudaPopCallConfiguration);
    LDSYM(__cudaPushCallConfiguration);
    LDSYM(__cudaRegisterFatBinaryEnd);
    LDSYM(cudaStreamDestroy);
    LDSYM(cudaMalloc);
    LDSYM(cudaFree);
    LDSYM(cudaHostAlloc);
    LDSYM(cudaFreeHost);
    LDSYM(cudaDeviceGetStreamPriorityRange);
    LDSYM(cudaHostGetDevicePointer);
    LDSYM(cudaGetDeviceProperties);
    LDSYM(cudaStreamCreateWithPriority);
    LDSYM(cudaStreamCreateWithFlags);
    LDSYM(cudaEventCreateWithFlags);
    LDSYM(cudaEventDestroy);
    LDSYM(cudaGetDeviceCount);
    LDSYM(cudaFuncGetAttributes);
    LDSYM(cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags);
    LDSYM(cudaStreamSynchronize);
    LDSYM(cudaMemcpyAsync);
    LDSYM(cudaEventRecord);
    LDSYM(cudaDeviceGetAttribute);
    LDSYM(cudaMemsetAsync);
    LDSYM(cudaLaunchKernel);
    LDSYM(cudaGetLastError);
    LDSYM(cudaSetDevice);
    LDSYM(cudaGetDevice);
    LDSYM(cudaProfilerStop);
    LDSYM(cudaProfilerStart);
    LDSYM(cudaProfilerInitialize);
    LDSYM(cudaGraphRemoveDependencies);
    LDSYM(cudaGraphNodeGetType);
    LDSYM(cudaGraphNodeGetDependentNodes);
    LDSYM(cudaGraphNodeGetDependencies);
    LDSYM(cudaGraphNodeFindInClone);
    LDSYM(cudaGraphMemsetNodeSetParams);
    LDSYM(cudaGraphMemsetNodeGetParams);
    LDSYM(cudaGraphMemcpyNodeSetParams);
    LDSYM(cudaGraphMemcpyNodeGetParams);
    LDSYM(cudaGraphLaunch);
    LDSYM(cudaGraphKernelNodeSetParams);
    LDSYM(cudaGraphKernelNodeGetParams);
    LDSYM(cudaGraphInstantiate);
    LDSYM(cudaGraphHostNodeSetParams);
    LDSYM(cudaGraphHostNodeGetParams);
    LDSYM(cudaGraphGetRootNodes);
    LDSYM(cudaGraphGetNodes);
    LDSYM(cudaGraphGetEdges);
    LDSYM(cudaGraphExecDestroy);
    LDSYM(cudaGraphDestroyNode);
    LDSYM(cudaGraphDestroy);
    LDSYM(cudaGraphCreate);
    LDSYM(cudaGraphClone);
    LDSYM(cudaGraphChildGraphNodeGetGraph);
    LDSYM(cudaGraphAddMemsetNode);
    LDSYM(cudaGraphAddMemcpyNode);
    LDSYM(cudaGraphAddKernelNode);
    LDSYM(cudaGraphAddHostNode);
    LDSYM(cudaGraphAddEmptyNode);
    LDSYM(cudaGraphAddDependencies);
    LDSYM(cudaGraphAddChildGraphNode);
    LDSYM(cudaRuntimeGetVersion);
    LDSYM(cudaDriverGetVersion);
    LDSYM(cudaGetSurfaceObjectResourceDesc);
    LDSYM(cudaDestroySurfaceObject);
    LDSYM(cudaCreateSurfaceObject);
    LDSYM(cudaGetTextureObjectTextureDesc);
    LDSYM(cudaGetTextureObjectResourceViewDesc);
    LDSYM(cudaGetTextureObjectResourceDesc);
    LDSYM(cudaGetChannelDesc);
    LDSYM(cudaDestroyTextureObject);
    LDSYM(cudaCreateTextureObject);
    LDSYM(cudaGraphicsUnregisterResource);
    LDSYM(cudaGraphicsUnmapResources);
    LDSYM(cudaGraphicsSubResourceGetMappedArray);
    LDSYM(cudaGraphicsResourceSetMapFlags);
    LDSYM(cudaGraphicsResourceGetMappedPointer);
    LDSYM(cudaGraphicsResourceGetMappedMipmappedArray);
    LDSYM(cudaGraphicsMapResources);
    LDSYM(cudaDeviceEnablePeerAccess);
    LDSYM(cudaDeviceDisablePeerAccess);
    LDSYM(cudaDeviceCanAccessPeer);
    LDSYM(cudaPointerGetAttributes);
    LDSYM(cudaMemset3DAsync);
    LDSYM(cudaMemset3D);
    LDSYM(cudaMemset2DAsync);
    LDSYM(cudaMemset2D);
    LDSYM(cudaMemset);
    LDSYM(cudaMemcpyToSymbolAsync);
    LDSYM(cudaMemcpyToSymbol);
    LDSYM(cudaMemcpyPeerAsync);
    LDSYM(cudaMemcpyPeer);
    LDSYM(cudaMemcpyFromSymbolAsync);
    LDSYM(cudaMemcpyFromSymbol);
    LDSYM(cudaMemcpy3DPeerAsync);
    LDSYM(cudaMemcpy3DPeer);
    LDSYM(cudaMemcpy3DAsync);
    LDSYM(cudaMemcpy3D);
    LDSYM(cudaMemcpy2DToArrayAsync);
    LDSYM(cudaMemcpy2DToArray);
    LDSYM(cudaMemcpy2DFromArrayAsync);
    LDSYM(cudaMemcpy2DFromArray);
    LDSYM(cudaMemcpy2DAsync);
    LDSYM(cudaMemcpy2DArrayToArray);
    LDSYM(cudaMemcpy2D);
    LDSYM(cudaMemcpy);
    LDSYM(cudaMemRangeGetAttributes);
    LDSYM(cudaMemRangeGetAttribute);
    LDSYM(cudaMemPrefetchAsync);
    LDSYM(cudaMemAdvise);
    LDSYM(cudaHostUnregister);
    LDSYM(cudaHostRegister);
    LDSYM(cudaHostGetFlags);
    LDSYM(cudaGetSymbolSize);
    LDSYM(cudaGetSymbolAddress);
    LDSYM(cudaGetMipmappedArrayLevel);
    LDSYM(cudaFreeMipmappedArray);
    LDSYM(cudaFreeArray);
    LDSYM(cudaArrayGetInfo);
    LDSYM(cudaOccupancyMaxActiveBlocksPerMultiprocessor);
    //LDSYM(cudaSetDoubleForHost);
    //LDSYM(cudaSetDoubleForDevice);
    LDSYM(cudaLaunchHostFunc);
    LDSYM(cudaLaunchCooperativeKernelMultiDevice);
    LDSYM(cudaLaunchCooperativeKernel);
    LDSYM(cudaFuncSetSharedMemConfig);
    LDSYM(cudaFuncSetCacheConfig);
    LDSYM(cudaFuncSetAttribute);
    LDSYM(cudaWaitExternalSemaphoresAsync);
    LDSYM(cudaSignalExternalSemaphoresAsync);
    LDSYM(cudaImportExternalSemaphore);
    LDSYM(cudaImportExternalMemory);
    LDSYM(cudaExternalMemoryGetMappedMipmappedArray);
    LDSYM(cudaExternalMemoryGetMappedBuffer);
    LDSYM(cudaDestroyExternalSemaphore);
    LDSYM(cudaDestroyExternalMemory);
    LDSYM(cudaEventSynchronize);
    LDSYM(cudaEventQuery);
    LDSYM(cudaEventElapsedTime);
    LDSYM(cudaEventCreate);
    LDSYM(cudaStreamWaitEvent);
    LDSYM(cudaStreamQuery);
    LDSYM(cudaStreamIsCapturing);
    LDSYM(cudaStreamGetPriority);
    LDSYM(cudaStreamGetFlags);
    LDSYM(cudaStreamEndCapture);
    LDSYM(cudaStreamCreate);
    LDSYM(cudaStreamBeginCapture);
    LDSYM(cudaStreamAttachMemAsync);
    LDSYM(cudaStreamAddCallback);
    LDSYM(cudaPeekAtLastError);
    LDSYM(cudaSetValidDevices);
    LDSYM(cudaSetDeviceFlags);
    LDSYM(cudaIpcOpenMemHandle);
    LDSYM(cudaIpcOpenEventHandle);
    LDSYM(cudaIpcGetMemHandle);
    LDSYM(cudaIpcGetEventHandle);
    LDSYM(cudaIpcCloseMemHandle);
    LDSYM(cudaGetDeviceFlags);
    LDSYM(cudaDeviceSynchronize);
    LDSYM(cudaDeviceSetSharedMemConfig);
    LDSYM(cudaDeviceSetLimit);
    LDSYM(cudaDeviceSetCacheConfig);
    LDSYM(cudaDeviceReset);
    LDSYM(cudaDeviceGetSharedMemConfig);
    LDSYM(cudaDeviceGetPCIBusId);
    LDSYM(cudaDeviceGetP2PAttribute);
    LDSYM(cudaDeviceGetLimit);
    LDSYM(cudaDeviceGetCacheConfig);
    LDSYM(cudaDeviceGetByPCIBusId);
    LDSYM(cudaChooseDevice);
    LDSYM(cudaMallocMipmappedArray);
    LDSYM(cudaMallocArray);
    LDSYM(cudaMallocHost);
    LDSYM(cudaMalloc3DArray);
    LDSYM(cudaMalloc3D);
    LDSYM(cudaMallocManaged);
    LDSYM(cudaMallocPitch);
    LDSYM(cudaMemGetInfo);
    LDSYM(cudaCreateChannelDesc);
    printf("rt dlsym all funcs end\n");
}

__attribute((constructor)) void cudawrt_init(void) {
    printf("cudawrt_init\n");
    so_handle = dlopen(LIB_STRING, RTLD_NOW);
    if (!so_handle) {
        dlerrmsg("dlopen(%s) failed.\n", LIB_STRING);
        exit(1);
    }
    dlsym_all_funcs();
    cudaw_so_register_dli(&so_dli);
    // copy func
    void * copy_for_trace[] = {
    };
    cudawrt_so_func_copy(copy_for_trace);
    // copy must be locate before any swap
    void * swap_for_blkcpy[] = {
        FSWAP(cudaMemset)
        FSWAP(cudaMemsetAsync)
        FSWAP(cudaMemcpy)
        FSWAP(cudaMemcpyAsync)
    };
    cudawrt_blkcpy_func_swap(swap_for_blkcpy);
    void * swap_for_trace[] = {
        FSWAP(cudaMalloc)
        FSWAP(cudaFree)
        FSWAP(cudaEventCreate)
        FSWAP(cudaEventCreateWithFlags)
        FSWAP(cudaEventDestroy)
        FSWAP(cudaDeviceSynchronize)
        FSWAP(cudaStreamSynchronize)
        FSWAP(cudaLaunchKernel)
        FSWAP(cudaMemset)
        FSWAP(cudaMemsetAsync)
        FSWAP(cudaMemcpy)
        FSWAP(cudaMemcpyAsync)
    };
    cudawrt_so_func_swap(swap_for_trace);
    /*
    void * swap_for_targs[] = {
        FSWAP(cudaLauchKernel)
    };
    cudaw_targs_func_swap(swap_for_targs);
    */
}

__attribute((destructor)) void cudawrt_fini(void) {
    printf("cudawrt_fini\n");
    if (so_handle) {
        dlclose(so_handle);
    }
    for (int k = 1; k <= so_dli.func_num; ++k) {
        if (so_funcs[k].cnt == 0)
            continue;
        printf("%5d %10lu : %s\n", k, so_funcs[k].cnt, so_funcs[k].func_name);
    }
}

