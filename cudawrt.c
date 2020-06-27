#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <cuda.h>
#include <string.h>
#include <pthread.h>
#include <cuda_runtime.h>
#include <sys/types.h>
#include <unistd.h>
#include <time.h>
#include <unistd.h>
#include <assert.h>

//#include <helper_cuda.h>
#include "vaddr.h"
#include "targs.h"


#define SIZE 10000


//#define printf(...) do { } while (0)

unsigned long long mod = 9973L;

static const char LIB_STRING_RT[] = "/workspace/libcudart.so.10.0.130";
static const char LIB_STRING_BLAS[] = "/workspace/libcublas.so.10.0.130";
static const char CONFIG_STRING[] = "WRAPPER_MAX_MEMORY";
static void* func[200];
static void* func2[10];
static int init_flag = 0;
static char* error;
static unsigned long long offset=0;
static void* add;
static int cnt;

static void * so_handle = NULL;

static void printerr() {
    char *errstr = dlerror();
    if (errstr != NULL) {
        printf ("A dynamic linking error occurred: (%s)\n", errstr);
    }
}

// DEFSO & LDSYM

#define DEFSO(func)  static cudaError_t (*so_##func)

#define LDSYM(func)  do { \
    so_##func = dlsym(so_handle, #func); \
    printerr(); \
} while(0)

static const char* (*so_cudaGetErrorString)(cudaError_t err);
static const char* (*so_cudaGetErrorName)(cudaError_t err);
static void (*so___cudaRegisterVar)(void **fatCubinHandle,char *hostVar,char *deviceAddress,const char *deviceName,int ext,int size,int constant,int global);
static void (*so___cudaRegisterTexture)(void **fatCubinHandle,const struct textureReference *hostVar,const void **deviceAddress,const char *deviceName,int dim,int norm,int ext);
static void (*so___cudaRegisterSurface)(void **fatCubinHandle,const struct surfaceReference  *hostVar,const void **deviceAddress,const char *deviceName,int dim,int ext);
static void (*so___cudaRegisterFunction)(void **fatCubinHandle,const char *hostFun,char *deviceFun,const char *deviceName,int thread_limit,uint3 *tid,uint3 *bid,dim3 *bDim,dim3 *gDim,int *wSize);
static void** (*so___cudaRegisterFatBinary)(void* fatCubin);
static void (*so___cudaUnregisterFatBinary) (void** point);
static unsigned (*so___cudaPushCallConfiguration)(dim3 gridDim, dim3 blockDim, size_t sharedMem , void *stream);
static void (*so___cudaRegisterFatBinaryEnd)(void **fatCubinHandle);
static struct cudaChannelFormatDesc (*so_cudaCreateChannelDesc)(int  x, int  y, int  z, int  w, enum cudaChannelFormatKind f);
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
DEFSO(cudaSetDoubleForHost)(double* d);
DEFSO(cudaSetDoubleForDevice)(double* d);
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

static void dlsym_all_funcs() {
    printf("dlsym all funcs\n");

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
    LDSYM(cudaSetDoubleForHost);
    LDSYM(cudaSetDoubleForDevice);
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

#define checkCudaErrors(err)  __checkCudaErrors (err, __FILE__, __LINE__, __func__)
static cudaError_t __checkCudaErrors(cudaError_t err, const char *file, const int line ,const char *func) {
    if(cudaSuccess != err && 11!=err && 29!=err) {
        fprintf(stderr,
                "CUDA Runtime API error = %04d from file <%s>, line %i,function %s.\n",
                err, file, line, func);
        /*if (err == 4||err==77) {
            exit(1);
        }*/
        //exit(-1);
    }
    return err;
}

#define CUDAW_PRINT_ALL_INVOKE
#ifdef CUDAW_PRINT_ALL_INVOKE

    #define begin_func()    __print_stream_func(stream, __func__)
    #define end_func()      do { } while (0)


static cudaStream_t stream = 0; // for API func without stream arg.

#else // CUDAW_PRINT_ALL_INVOKE

    #define begin_func()    __begin_func(__FILE__, __LINE__, __func__)
    #define end_func()      do { } while (0)

#endif

#define PRINT_MEM_INFO

static void __print_stream_func(cudaStream_t stream, const char * func) {
#ifdef PRINT_MEM_INFO
    if (func != NULL) {
        size_t free, total;
        so_cudaMemGetInfo(&free, &total);
        printf("%lug%lum %lug%lum %s\n", free>>30, (free>>20)&1023, total>>30, (total>>20)&1023, func);
        fflush(stdout);
    }
#endif
    static const char * skip_list[] = {
        "cudaSetDevice",
        "cudaGetDevice",
        "",
    };
    #define max_threads 128 
    #define max_funcs   (512 * max_threads)
    #define max_calls   64
    static struct {
        int tid;
        int last_call_idx;
        int next_call_idx;
        int loop_call_max;
        int loop_calls[max_calls];
        FILE * fout;
        int cnt_funcs;
        struct { 
            int cnt;
            int total;
            int tid;
            char func_name[80];
        } funcs[max_funcs];
    } ts[max_threads] = {0};
	static union { 
        int cnt; 
        float no; 
    } sn = {0x3b000000};
    if (func == NULL) {
        for (int t = 0; t < max_threads; t++) {
            if (ts[t].tid == 0)
                break;
            for (int i = 0; i < ts[t].cnt_funcs; i++) {
                int total = ts[t].funcs[i].total + ts[t].funcs[i].cnt;
                char * func = ts[t].funcs[i].func_name;
        	    printf("--total-- %x %10d %s\n", ts[t].tid, total, func);
            }
        }
        return;
    }
    unsigned int tid = (unsigned int)pthread_self();
    static __thread int t = -1;
    if (t == -1) {
        for (t = 0; t < max_threads; t++) {
            if (ts[t].tid == tid) {
                break;
            }
            if (ts[t].tid == 0) {
                ts[t].tid = tid;
                break;
            }
        }
    }
    for (int k = 0; k < sizeof(skip_list) / sizeof(char *); k++) {
        if (strcmp(func, skip_list[k]) == 0) {
            return;
        }
    }
    // test if hit the last func call
    int i = ts[t].loop_calls[ts[t].last_call_idx];
    if (strcmp(ts[t].funcs[i].func_name, func) == 0) {
        ts[t].funcs[i].cnt++;
        return;
    }
    // test if hit the next func call
    i = ts[t].loop_calls[ts[t].next_call_idx];
    if (strcmp(ts[t].funcs[i].func_name, func) == 0) {
        ts[t].last_call_idx = ts[t].next_call_idx;
        ts[t].next_call_idx = (ts[t].last_call_idx + 1) % ts[t].loop_call_max;
        ts[t].funcs[i].cnt++;
        return;
    }
    ts[t].next_call_idx = ts[t].loop_call_max;
    // lookup the func index
    for (i = 0; i < ts[t].cnt_funcs; i++) {
        if (strcmp(ts[t].funcs[i].func_name, func) == 0) {
            break;
        }
    }
    // put in the new func
    if (i == ts[t].cnt_funcs) {
        assert(strlen(func) < sizeof(ts[t].funcs[i].func_name));
        strcpy(ts[t].funcs[i].func_name, func);
        ts[t].cnt_funcs++;
    }
    else {
        for (int k = 0; k < ts[t].loop_call_max; k++) {
            if (i == ts[t].loop_calls[k]) {
                ts[t].next_call_idx = 0;
                break;
            }
        }
    }
    if (ts[t].next_call_idx < ts[t].loop_call_max || 
        (ts[t].last_call_idx + 1) != ts[t].loop_call_max) {
        // print all previous cnts ... 
        if (ts[t].fout == NULL) {
            char filename[256];
            sprintf(filename, "out-%x.log", t);
            ts[t].fout = fopen(filename, "w");
        }
        for (int k = 0; k < ts[t].loop_call_max; k++) {
            int i = ts[t].loop_calls[k];
            if (ts[t].funcs[i].cnt == 0) {
            printf("(funcs[i].cnt == 0) %d %s %d %d\n", i, ts[t].funcs[i].func_name, k, ts[t].loop_call_max);
            }
            char * func = ts[t].funcs[i].func_name;
            int cnt = ts[t].funcs[i].cnt;
            int total = ts[t].funcs[i].total += ts[t].funcs[i].cnt;
        	fprintf(ts[t].fout, "%12.10f %x %p %s %d %d\n", sn.no, tid, stream, func, total, cnt);
            ts[t].funcs[i].cnt = 0;
        }
        fprintf(ts[t].fout, "-----------------------\n");
        fflush(ts[t].fout);
	    sn.cnt++;
        ts[t].loop_call_max = 0;
    }
    // try add the func to loop calls
    if (ts[t].loop_call_max == 0) {
        ts[t].last_call_idx = ts[t].next_call_idx = 0;
    }
    else {
        assert(ts[t].last_call_idx  + 1 == ts[t].loop_call_max);
        ts[t].next_call_idx = 0;
        ts[t].last_call_idx = ts[t].loop_call_max;
    }
    ts[t].funcs[i].cnt++;
    ts[t].loop_calls[ts[t].loop_call_max++] = i;
}


static void __begin_func(const char *file, const int line , const char *func) {
    if(func[0]=='_') {
        return;
    }
    //printf("%s\n",func);
}

static void __end_func(const char *file, const int line ,const char *func) {
    if(func[0]=='_') {
        return;
	}
    //printf("%s end\n",func);
}

#ifdef VA_TEST_DEV_ADDR

#ifdef begin_func
  #undef begin_func
  #define begin_func() do { \
                cudawMemLock(); \
                /*__print_stream_func(stream, __func__);*/ \
          } while (0)
#endif

#ifdef end_func
  #undef end_func
  #define end_func() cudawMemUnlock()
#endif

#endif // VA_TEST_DEV_ADDR

__attribute ((constructor)) void cudawrt_init(void) {
    printf("cudawrt_init\n");
    so_handle = dlopen (LIB_STRING_RT, RTLD_NOW);
    if (!so_handle) {
        fprintf (stderr, "FAIL: %s\n", dlerror());
        exit(1);
    }
    printerr();
    dlsym_all_funcs();
    // test mem
#ifdef PRINT_MEM_INFO
    size_t free, total;
    so_cudaMemGetInfo(&free, &total);
    printf("so_cudaMemGetInfo %lug%lum %lug%lum\n", free>>30, (free>>20)&1023, total>>30, (total>>20)&1023);
#endif
    // Relocate cuda API wrapped in targs.c
    so_cudaLaunchKernel = cudawLaunchKernel;
    // Reloacte cuda API wrapped in vaddr.c
    so_cudaMemGetInfo = cudawMemGetInfo;
    so_cudaMalloc = cudawMalloc;
    so_cudaFree = cudawFree;
#ifdef VA_TEST_DEV_ADDR
    so_cudaMemset = cudawMemset;
    so_cudaMemsetAsync = cudawMemsetAsync;
    so_cudaMemcpy = cudawMemcpy;
    so_cudaMemcpyAsync = cudawMemcpyAsync;
#endif
}

__attribute ((destructor)) void cudawrt_fini(void) {
    printf("cudawrt_fini\n");
    if (so_handle) {
        dlclose(so_handle);
    }
    __print_stream_func(NULL, NULL);
}

cudaError_t cudaMalloc(void** devPtr, size_t bytesize) {
    begin_func(); 
    // so_cudaMalloc is cudawMalloc 
    cudaError_t r = so_cudaMalloc(devPtr , (bytesize));
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaFree(void* devPtr) {

    begin_func();
    // so_cudaFree is cudawFree
    cudaError_t r = so_cudaFree(devPtr);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaHostAlloc (void** pHost, size_t size, unsigned int flags) {

    begin_func();
    //printf("cudaHostAlloc\n");
    //printf("before:phost:%p,size:%zu,flags:%u\n",*pHost,size,flags);
    cudaError_t r = so_cudaHostAlloc(pHost, size, flags);
    //printf("after:phost:%p,size:%zu,flags:%u,return %04d\n",*pHost,size,flags,r);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaFreeHost (void* ptr) {

    begin_func();
    //printf("cudaFreeHost\n");
    cudaError_t r = so_cudaFreeHost(ptr);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaDeviceGetStreamPriorityRange(int* leastPriority, int* greatestPriority) {

    begin_func();
    //printf("cudaDeviceGetStreamPriorityRange\n");
    cudaError_t r = so_cudaDeviceGetStreamPriorityRange(leastPriority, greatestPriority);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaHostGetDevicePointer (void** pDevice, void* pHost, unsigned int  flags) {

    begin_func();
    //printf("cudaHostGetDevicePointer\n");
    cudaError_t r = so_cudaHostGetDevicePointer(pDevice, pHost, flags);
    //assert(0); // TODO
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp* prop, int device) {

    begin_func();
    //printf("cudaGetDeviceProperties\n");
    cudaError_t r = so_cudaGetDeviceProperties(prop,device);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaStreamCreateWithPriority(cudaStream_t* pStream, unsigned int  flags, int  priority) {

    begin_func();
    //printf("cudaStreamCreateWithPriority\n");
    cudaError_t r = so_cudaStreamCreateWithPriority(pStream,  flags,  priority);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaStreamCreateWithFlags(cudaStream_t* pStream, unsigned int  flags) {

    begin_func();
    //printf("cudaStreamCreateWithFlags\n");
    //printf("before:pStream:%p(CUstream_st *),flags:%u\n",*pStream,flags);
    cudaError_t r = so_cudaStreamCreateWithFlags(pStream,  flags);
    //printf("after:pStream:%p(CUstream_st *),flags:%u,return %d\n",*pStream,flags,r);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaEventCreateWithFlags (cudaEvent_t* event, unsigned int  flags) {

    begin_func();
    cudaError_t r = so_cudaEventCreateWithFlags(event, flags);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaEventDestroy (cudaEvent_t event) {

    begin_func();
    cudaError_t r = so_cudaEventDestroy(event);
    end_func();checkCudaErrors(r);
    return r;
}

__host__  __device__ cudaError_t cudaStreamDestroy (cudaStream_t stream) {

    begin_func();
    //printf("cudaStreamDestroy\n");
    //printf("before:pStream:%p(CUstream_st *)\n",stream);
    //exit(1);
    cudaError_t r = so_cudaStreamDestroy(stream);
    //printf("before:pStream:%p(CUstream_st *),return %d\n",stream,r);
    end_func();checkCudaErrors(r);
    return r;
    //return (*so_cudaStreamDestroy)(stream);
}

cudaError_t cudaGetDeviceCount(int* count) {

    begin_func();
    //printf("cudaGetDeviceCount:\n");
    cudaError_t r = so_cudaGetDeviceCount(count);
    //printf("devicecnt\n");
    end_func();checkCudaErrors(r);
    return r;
}

struct cudaChannelFormatDesc cudaCreateChannelDesc (int  x, int  y, int  z, int  w, enum cudaChannelFormatKind f) {
    begin_func();
    //printf("cudaCreateChannelDesc:\n");
    struct cudaChannelFormatDesc r = so_cudaCreateChannelDesc(x,y,z,w,f);
    end_func();checkCudaErrors(0);
    return r;
}

cudaError_t cudaFuncGetAttributes (struct cudaFuncAttributes* attr, const void* func) {

    begin_func();
    //printf("cudaFuncGetAttributes\n");
    cudaError_t r = so_cudaFuncGetAttributes(attr, func);
    //printf("FuncGet\n");
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags (int* numBlocks, const void* func, int  blockSize, size_t dynamicSMemSize, unsigned int flags) {

    begin_func();
    //printf("cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags\n");
    cudaError_t r = so_cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, func, blockSize, dynamicSMemSize, flags);
    //printf("\n\n");
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaStreamSynchronize (cudaStream_t stream) {

    begin_func();
    //printf("cudaStreamSynchronize:\n");
    cudaError_t r = so_cudaStreamSynchronize(stream);
    //printf("StreamSync\n");
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaMemcpyAsync (void* dst, const void* src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream) {

    begin_func();
    //printf("cudaMemcpyAsync:\n");
    /*if(count==8&&kind==2) {
       printf("before\n");
       //memcpy(dst-8,dst,8);
       //printf("memcpy end\n");
       //void* tmpp=(void*)src;
       //cudaMemcpy(tmpp, src, count, 3);
       //printf("%p %p %zu %d\n",dst,src,count,kind);
    }*/
    VtoR2(src,dst);
    cudaError_t r = so_cudaMemcpyAsync(dst, (const void*)src, count, kind, stream);
    //printf("MemAsync\n");
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaEventRecord (cudaEvent_t event, cudaStream_t stream) {

    begin_func();
    //printf("cudaEventRecord:\n");
    cudaError_t r = so_cudaEventRecord(event, stream);
    //printf("event\n");
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaDeviceGetAttribute (int* value, enum cudaDeviceAttr attr, int  device) {

    begin_func();
    //printf("cudaDeviceGetAttribute\n");
    cudaError_t r = so_cudaDeviceGetAttribute(value, attr, device);
    end_func();checkCudaErrors(r);
    //printf("GetAa\n");
    return r;
}

cudaError_t cudaMemsetAsync (void* devPtr, int  value, size_t count, cudaStream_t stream) {

    begin_func();
    VtoR1(devPtr);
    cudaError_t r =  so_cudaMemsetAsync(devPtr, value, count, stream);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaLaunchKernel (const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream) {
    begin_func();
    //printf("cudaLaunchKernel\n");
    // so_cudaLaunchKernel is cudawLaunchKernel
    cudaError_t r = so_cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);
    //printf("cudaLaunchKernel end\n");
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaGetLastError () {

    begin_func();
    //printf("cudaGetLastError:\n");
    cudaError_t r = so_cudaGetLastError();
    //printf("GetLast\n");
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaSetDevice (int  device) {

    begin_func();
    //printf("cudaSetDevice:%d\n",device);
    cudaError_t r = so_cudaSetDevice(device);
    //printf("Set:%d\n",device);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaGetDevice(int* device) {

    begin_func();
    //printf("cudaGetDevice:%d\n",*device);
    cudaError_t r = so_cudaGetDevice(device);
    //printf("Get:%d\n",*device);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaProfilerStop (void) {

    begin_func();
    //printf("cudaProfilerStop\n");
    cudaError_t r = so_cudaProfilerStop();
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaProfilerStart (void) {

    begin_func();
    //printf("cudaProfilerStart\n");
    cudaError_t r = so_cudaProfilerStart();
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaProfilerInitialize (const char* configFile, const char* outputFile, cudaOutputMode_t outputMode) {

    begin_func();
    //printf("cudaProfilerInitialize\n");
    cudaError_t r = so_cudaProfilerInitialize(configFile,  outputFile, outputMode);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaGraphRemoveDependencies (cudaGraph_t graph, cudaGraphNode_t* from,  cudaGraphNode_t* to, size_t numDependencies) {

    begin_func();
    //printf("cudaGraphRemoveDependencies\n");
    cudaError_t r = so_cudaGraphRemoveDependencies(graph, from, to, numDependencies);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaGraphNodeGetType (cudaGraphNode_t node, enum cudaGraphNodeType * pType) {

    begin_func();
    //printf("cudaGraphNodeGetType\n");
    cudaError_t r = so_cudaGraphNodeGetType(node, pType);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaGraphNodeGetDependentNodes (cudaGraphNode_t node, cudaGraphNode_t* pDependentNodes, size_t* pNumDependentNodes) {

    begin_func();
    //printf("cudaGraphNodeGetDependentNodes\n");
    cudaError_t r = so_cudaGraphNodeGetDependentNodes(node, pDependentNodes, pNumDependentNodes);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaGraphNodeGetDependencies (cudaGraphNode_t node, cudaGraphNode_t* pDependencies, size_t* pNumDependencies) {

    begin_func();
    //printf("cudaGraphNodeGetDependencies\n");
    cudaError_t r = so_cudaGraphNodeGetDependencies(node, pDependencies, pNumDependencies);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaGraphNodeFindInClone (cudaGraphNode_t* pNode, cudaGraphNode_t originalNode, cudaGraph_t clonedGraph) {

    begin_func();
    //printf("cudaGraphNodeFindInClone\n");
    cudaError_t r = so_cudaGraphNodeFindInClone(pNode, originalNode, clonedGraph);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaGraphMemsetNodeSetParams (cudaGraphNode_t node, const struct cudaMemsetParams* pNodeParams) {

    begin_func();
    //printf("cudaGraphMemsetNodeSetParams\n");
    cudaError_t r = so_cudaGraphMemsetNodeSetParams(node, pNodeParams);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaGraphMemsetNodeGetParams (cudaGraphNode_t node, struct cudaMemsetParams* pNodeParams) {

    begin_func();
    //printf("cudaGraphMemsetNodeGetParams\n");
    cudaError_t r = so_cudaGraphMemsetNodeGetParams(node, pNodeParams);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaGraphMemcpyNodeSetParams (cudaGraphNode_t node, const struct cudaMemcpy3DParms* pNodeParams) {

    begin_func();
    //printf("cudaGraphMemcpyNodeSetParams\n");
    cudaError_t r = so_cudaGraphMemcpyNodeSetParams(node, pNodeParams);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaGraphMemcpyNodeGetParams (cudaGraphNode_t node, struct cudaMemcpy3DParms* pNodeParams) {

    begin_func();
    //printf("cudaGraphMemcpyNodeGetParams\n");
    cudaError_t r = so_cudaGraphMemcpyNodeGetParams(node, pNodeParams);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaGraphLaunch (cudaGraphExec_t graphExec, cudaStream_t stream) {

    begin_func();
    //printf("cudaGraphLaunch\n");
    cudaError_t r = so_cudaGraphLaunch(graphExec, stream);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaGraphKernelNodeSetParams (cudaGraphNode_t node, const struct cudaKernelNodeParams* pNodeParams) {

    begin_func();
    //printf("cudaGraphKernelNodeSetParams\n");
    cudaError_t r = so_cudaGraphKernelNodeSetParams(node, pNodeParams);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaGraphKernelNodeGetParams (cudaGraphNode_t node, struct cudaKernelNodeParams* pNodeParams) {

    begin_func();
    //printf("cudaGraphKernelNodeGetParams\n");
    cudaError_t r = so_cudaGraphKernelNodeGetParams(node, pNodeParams);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaGraphInstantiate (cudaGraphExec_t* pGraphExec, cudaGraph_t graph, cudaGraphNode_t* pErrorNode, char* pLogBuffer, size_t bufferSize) {

    begin_func();
    //printf("cudaGraphInstantiate\n");
    cudaError_t r = so_cudaGraphInstantiate(pGraphExec , graph ,  pErrorNode, pLogBuffer, bufferSize);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaGraphHostNodeSetParams (cudaGraphNode_t node, const struct cudaHostNodeParams* pNodeParams) {

    begin_func();
    //printf("cudaGraphHostNodeSetParams\n");
    cudaError_t r = so_cudaGraphHostNodeSetParams(node , pNodeParams);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaGraphHostNodeGetParams(cudaGraphNode_t node, struct cudaHostNodeParams* pNodeParams) {

    begin_func();
    //printf("cudaGraphHostNodeGetParams\n");
    cudaError_t r = so_cudaGraphHostNodeGetParams(node , pNodeParams);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaGraphGetRootNodes (cudaGraph_t graph, cudaGraphNode_t* pRootNodes, size_t* pNumRootNodes) {

    begin_func();
    //printf("cudaGraphGetRootNodes\n");
    cudaError_t r = so_cudaGraphGetRootNodes(graph,pRootNodes, pNumRootNodes);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaGraphGetNodes (cudaGraph_t graph, cudaGraphNode_t* nodes, size_t* numNodes) {

    begin_func();
    //printf("cudaGraphGetNodes\n");
    cudaError_t r = so_cudaGraphGetNodes(graph,nodes, numNodes);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaGraphGetEdges (cudaGraph_t graph, cudaGraphNode_t* from, cudaGraphNode_t* to, size_t* numEdges) {

    begin_func();
    //printf("cudaGraphGetEdges\n");
    cudaError_t r = so_cudaGraphGetEdges(graph, from, to, numEdges);
    end_func();checkCudaErrors(r);
    return r;
}

/*cudaError_t cudaGraphExecKernelNodeSetParams (cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const struct cudaKernelNodeParams* pNodeParams) {

    begin_func();
    printf("cudaGraphExecKernelNodeSetParams\n");
    cudaError_t r = (*(cudaError_t (*)(cudaGraphExec_t ,cudaGraphNode_t, const struct cudaKernelNodeParams*))(func[46]))(hGraphExec, node, pNodeParams);
    end_func();checkCudaErrors(r);
    return r;
}*/

cudaError_t cudaGraphExecDestroy (cudaGraphExec_t graphExec) {

    begin_func();
    //printf("cudaGraphExecDestroy\n");
    cudaError_t r = so_cudaGraphExecDestroy(graphExec);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaGraphDestroyNode (cudaGraphNode_t node) {

    begin_func();
    //printf("cudaGraphDestroyNode\n");
    cudaError_t r = so_cudaGraphDestroyNode(node);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaGraphDestroy (cudaGraph_t graph) {

    begin_func();
    //printf("cudaGraphDestroy\n");
    cudaError_t r = so_cudaGraphDestroy(graph);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaGraphCreate (cudaGraph_t* pGraph, unsigned int flags) {

    begin_func();
    //printf("cudaGraphCreate\n");
    cudaError_t r = so_cudaGraphCreate(pGraph,  flags);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaGraphClone (cudaGraph_t* pGraphClone, cudaGraph_t originalGraph) {

    begin_func();
    //printf("cudaGraphClone\n");
    cudaError_t r = so_cudaGraphClone(pGraphClone,  originalGraph);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaGraphChildGraphNodeGetGraph (cudaGraphNode_t node, cudaGraph_t* pGraph) {

    begin_func();
    //printf("cudaGraphChildGraphNodeGetGraph\n");
    cudaError_t r = so_cudaGraphChildGraphNodeGetGraph(node,  pGraph);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaGraphAddMemsetNode (cudaGraphNode_t* pGraphNode, cudaGraph_t graph, cudaGraphNode_t* pDependencies, size_t numDependencies, const struct cudaMemsetParams* pMemsetParams) {

    begin_func();
    //printf("cudaGraphAddMemsetNode\n");
    cudaError_t r = so_cudaGraphAddMemsetNode(pGraphNode,  graph, pDependencies, numDependencies, pMemsetParams);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaGraphAddMemcpyNode (cudaGraphNode_t* pGraphNode, cudaGraph_t graph, cudaGraphNode_t* pDependencies, size_t numDependencies, const struct cudaMemcpy3DParms* pCopyParams) {

    begin_func();
    //printf("cudaGraphAddMemcpyNode\n");
    cudaError_t r = so_cudaGraphAddMemcpyNode(pGraphNode,  graph, pDependencies, numDependencies, pCopyParams);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaGraphAddKernelNode (cudaGraphNode_t* pGraphNode, cudaGraph_t graph, cudaGraphNode_t* pDependencies, size_t numDependencies, const struct cudaKernelNodeParams* pNodeParams) {

    begin_func();
    //printf("cudaGraphAddKernelNode\n");
    cudaError_t r = so_cudaGraphAddKernelNode(pGraphNode,  graph, pDependencies, numDependencies, pNodeParams);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaGraphAddHostNode (cudaGraphNode_t* pGraphNode, cudaGraph_t graph, cudaGraphNode_t* pDependencies, size_t numDependencies, const struct cudaHostNodeParams* pNodeParams) {

    begin_func();
    //printf("cudaGraphAddHostNode\n");
    cudaError_t r = so_cudaGraphAddHostNode(pGraphNode,  graph, pDependencies, numDependencies, pNodeParams);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaGraphAddEmptyNode (cudaGraphNode_t* pGraphNode, cudaGraph_t graph, cudaGraphNode_t* pDependencies, size_t numDependencies) {

    begin_func();
    //printf("cudaGraphAddEmptyNode\n");
    cudaError_t r = so_cudaGraphAddEmptyNode(pGraphNode,  graph, pDependencies, numDependencies);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaGraphAddDependencies (cudaGraph_t graph, cudaGraphNode_t* from, cudaGraphNode_t* to, size_t numDependencies) {

    begin_func();
    //printf("cudaGraphAddDependencies\n");
    cudaError_t r = so_cudaGraphAddDependencies(graph,  from, to, numDependencies);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaGraphAddChildGraphNode (cudaGraphNode_t* pGraphNode, cudaGraph_t graph, cudaGraphNode_t* pDependencies, size_t numDependencies, cudaGraph_t childGraph) {

    begin_func();
    //printf("cudaGraphAddChildGraphNode\n");
    cudaError_t r = so_cudaGraphAddChildGraphNode(pGraphNode,  graph, pDependencies, numDependencies, childGraph);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaRuntimeGetVersion (int* runtimeVersion) {

    begin_func();
    //printf("cudaRuntimeGetVersion\n");
    cudaError_t r = so_cudaRuntimeGetVersion(runtimeVersion);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaDriverGetVersion (int* driverVersion) {

    begin_func();
    //printf("cudaDriverGetVersion\n");
    cudaError_t r = so_cudaDriverGetVersion(driverVersion);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaGetSurfaceObjectResourceDesc (struct cudaResourceDesc* pResDesc, cudaSurfaceObject_t surfObject) {

    begin_func();
    //printf("cudaGetSurfaceObjectResourceDesc\n");
    cudaError_t r = so_cudaGetSurfaceObjectResourceDesc(pResDesc,  surfObject);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaDestroySurfaceObject (cudaSurfaceObject_t surfObject) {

    begin_func();
    //printf("cudaDestroySurfaceObject\n");
    cudaError_t r = so_cudaDestroySurfaceObject(surfObject);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaCreateSurfaceObject (cudaSurfaceObject_t* pSurfObject, const struct cudaResourceDesc* pResDesc) {

    begin_func();
    //printf("cudaCreateSurfaceObject\n");
    cudaError_t r = so_cudaCreateSurfaceObject(pSurfObject, pResDesc);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaGetTextureObjectTextureDesc (struct cudaTextureDesc* pTexDesc, cudaTextureObject_t texObject) {

    begin_func();
    //printf("cudaGetTextureObjectTextureDesc\n");
    cudaError_t r = so_cudaGetTextureObjectTextureDesc(pTexDesc, texObject);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaGetTextureObjectResourceViewDesc (struct cudaResourceViewDesc* pResViewDesc, cudaTextureObject_t texObject) {

    begin_func();
    //printf("cudaGetTextureObjectResourceViewDesc\n");
    cudaError_t r = so_cudaGetTextureObjectResourceViewDesc(pResViewDesc, texObject);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaGetTextureObjectResourceDesc (struct cudaResourceDesc* pResDesc, cudaTextureObject_t texObject) {

    begin_func();
    //printf("cudaGetTextureObjectResourceDesc\n");
    cudaError_t r = so_cudaGetTextureObjectResourceDesc(pResDesc, texObject);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaGetChannelDesc (struct cudaChannelFormatDesc* desc, cudaArray_const_t array) {

    begin_func();
    //printf("cudaGetChannelDesc\n");
    cudaError_t r = so_cudaGetChannelDesc(desc, array);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaDestroyTextureObject (cudaTextureObject_t texObject) {

    begin_func();
    //printf("cudaDestroyTextureObject\n");
    cudaError_t r = so_cudaDestroyTextureObject(texObject);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaCreateTextureObject (cudaTextureObject_t* pTexObject, const struct cudaResourceDesc* pResDesc, const struct cudaTextureDesc* pTexDesc, const struct cudaResourceViewDesc* pResViewDesc) {

    begin_func();
    //printf("cudaCreateTextureObject\n");
    cudaError_t r = so_cudaCreateTextureObject(pTexObject, pResDesc, pTexDesc, pResViewDesc);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaGraphicsUnregisterResource (cudaGraphicsResource_t resource) {

    begin_func();
    //printf("cudaGraphicsUnregisterResource\n");
    cudaError_t r = so_cudaGraphicsUnregisterResource(resource);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaGraphicsUnmapResources (int count, cudaGraphicsResource_t* resources, cudaStream_t stream) {

    begin_func();
    //printf("cudaGraphicsUnmapResources\n");
    cudaError_t r = so_cudaGraphicsUnmapResources(count, resources, stream);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaGraphicsSubResourceGetMappedArray (cudaArray_t* array, cudaGraphicsResource_t resource, unsigned int arrayIndex, unsigned int mipLevel) {

    begin_func();
    //printf("cudaGraphicsSubResourceGetMappedArray\n");
    cudaError_t r = so_cudaGraphicsSubResourceGetMappedArray(array, resource, arrayIndex, mipLevel);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaGraphicsResourceSetMapFlags (cudaGraphicsResource_t resource, unsigned int  flags) {

    begin_func();
    //printf("cudaGraphicsResourceSetMapFlags\n");
    cudaError_t r = so_cudaGraphicsResourceSetMapFlags(resource, flags);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaGraphicsResourceGetMappedPointer (void** devPtr, size_t* size, cudaGraphicsResource_t resource) {

    begin_func();
    //printf("cudaGraphicsResourceGetMappedPointer\n");
    VtoR1(devPtr);
    cudaError_t r = so_cudaGraphicsResourceGetMappedPointer(devPtr, size, resource);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaGraphicsResourceGetMappedMipmappedArray (cudaMipmappedArray_t* mipmappedArray, cudaGraphicsResource_t resource) {

    begin_func();
    //printf("cudaGraphicsResourceGetMappedMipmappedArray\n");
    cudaError_t r = so_cudaGraphicsResourceGetMappedMipmappedArray(mipmappedArray, resource);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaGraphicsMapResources (int count, cudaGraphicsResource_t* resources, cudaStream_t stream) {

    begin_func();
    //printf("cudaGraphicsMapResources\n");
    cudaError_t r = so_cudaGraphicsMapResources(count, resources, stream);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaDeviceEnablePeerAccess (int peerDevice, unsigned int flags) {

    begin_func();
   //printf("cudaDeviceEnablePeerAccess\n");
    cudaError_t r = so_cudaDeviceEnablePeerAccess(peerDevice,flags);
    //dlclose(handle);
    return r;
}

cudaError_t cudaDeviceDisablePeerAccess (int peerDevice) {

    begin_func();
    //printf("cudaDeviceDisablePeerAccess\n");
    cudaError_t r = so_cudaDeviceDisablePeerAccess(peerDevice);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaDeviceCanAccessPeer (int* canAccessPeer, int device, int peerDevice) {

    begin_func();
    //printf("cudaDeviceCanAccessPeer\n");
    cudaError_t r = so_cudaDeviceCanAccessPeer(canAccessPeer,device, peerDevice);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaPointerGetAttributes (struct cudaPointerAttributes* attributes, const void* ptr) {

    begin_func();
    //printf("cudaPointerGetAttributes\n");
    VtoR1(ptr);
    //printf("add:%p ptr:%p,p:%p\n",add,ptr,p);
    cudaError_t r = so_cudaPointerGetAttributes(attributes,(const void*)ptr);
    //printf("Point12\n");
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaMemset3DAsync (struct cudaPitchedPtr pitchedDevPtr, int  value, struct cudaExtent extent, cudaStream_t stream) {

    begin_func();
    //printf("cudaMemset3DAsync\n");
    cudaError_t r = so_cudaMemset3DAsync(pitchedDevPtr,value, extent, stream);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaMemset3D (struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent) {

    begin_func();
    //printf("cudaMemset3D\n");
    cudaError_t r = so_cudaMemset3D(pitchedDevPtr,value, extent);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaMemset2DAsync(void* devPtr, size_t pitch, int  value, size_t width, size_t height, cudaStream_t stream) {

    begin_func();
    //printf("cudaMemset2DAsync\n");
    VtoR1(devPtr);
    cudaError_t r = so_cudaMemset2DAsync(devPtr,pitch, value,width, height, stream);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaMemset2D (void* devPtr, size_t pitch, int  value, size_t width, size_t height) {

    begin_func();
    //printf("cudaMemset2D\n");
    VtoR1(devPtr);
    cudaError_t r = so_cudaMemset2D(devPtr,pitch, value,width, height);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaMemset (void* devPtr, int value, size_t count) {

    begin_func();
    VtoR1(devPtr);
    cudaError_t r = so_cudaMemset(devPtr, value, count);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaMemcpyToSymbolAsync(const void* symbol, const void* src, size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream) {

    begin_func();
    //printf("cudaMemcpyToSymbolAsync\n");
    if (cudaMemcpyDeviceToDevice == kind)
        VtoR1(src);
    cudaError_t r = so_cudaMemcpyToSymbolAsync(symbol, src, count,offset, kind, stream);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaMemcpyToSymbol (const void* symbol, const void* src, size_t count, size_t offset, enum cudaMemcpyKind kind) {

    begin_func();
    //printf("cudaMemcpyToSymbol\n");
    if (cudaMemcpyDeviceToDevice == kind)
        VtoR1(src);
    cudaError_t r = so_cudaMemcpyToSymbol(symbol, src, count,offset, kind);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaMemcpyPeerAsync(void* dst, int dstDevice, const void* src, int srcDevice, size_t count, cudaStream_t stream) {

    begin_func();
    //printf("cudaMemcpyPeerAsync\n");
    VtoR2(src,dst);
    cudaError_t r = so_cudaMemcpyPeerAsync(dst, dstDevice, src,srcDevice, count, stream);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaMemcpyPeer (void* dst, int dstDevice, const void* src, int srcDevice, size_t count) {

    begin_func();
    //printf("cudaMemcpyPeer\n");
    VtoR2(src,dst);
    cudaError_t r = so_cudaMemcpyPeer(dst, dstDevice, src,srcDevice, count);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaMemcpyFromSymbolAsync (void* dst, const void* symbol, size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream) {

    begin_func();
    //printf("cudaMemcpyFromSymbolAsync\n");
    if  (kind == cudaMemcpyDeviceToDevice)
        VtoR1(dst);
    cudaError_t r = so_cudaMemcpyFromSymbolAsync(dst, symbol, count, offset,  kind, stream);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaMemcpyFromSymbol(void* dst, const void* symbol, size_t count, size_t offset , enum cudaMemcpyKind kind) {

    begin_func();
    //printf("cudaMemcpyFromSymbol\n");
    if  (kind == cudaMemcpyDeviceToDevice)
        VtoR1(dst);
    cudaError_t r = so_cudaMemcpyFromSymbol(dst, symbol, count,offset,kind);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaMemcpy3DPeerAsync (const struct cudaMemcpy3DPeerParms* p, cudaStream_t stream) {

    begin_func();
    //printf("cudaMemcpy3DPeerAsync\n");
    cudaError_t r = so_cudaMemcpy3DPeerAsync(p, stream);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaMemcpy3DPeer (const struct cudaMemcpy3DPeerParms* p) {

    begin_func();
    //printf("cudaMemcpy3DPeer\n");
    cudaError_t r = so_cudaMemcpy3DPeer(p);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaMemcpy3DAsync (const struct cudaMemcpy3DParms* p, cudaStream_t stream) {

    begin_func();
    //printf("cudaMemcpy3DAsync\n");
    cudaError_t r = so_cudaMemcpy3DAsync(p, stream);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaMemcpy3D (const struct cudaMemcpy3DParms* p) {

    begin_func();
    //printf("cudaMemcpy3D\n");
    cudaError_t r = so_cudaMemcpy3D(p);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaMemcpy2DToArrayAsync (cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream) {

    begin_func();
    //printf("cudaMemcpy2DToArrayAsync\n");
    cudaError_t r = so_cudaMemcpy2DToArrayAsync(dst,wOffset, hOffset, src,spitch, width,height, kind, stream);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaMemcpy2DToArray (cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind) {

    begin_func();
    //printf("cudaMemcpy2DToArray\n");
    cudaError_t r = so_cudaMemcpy2DToArray(dst,wOffset, hOffset, src,spitch, width,height, kind);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaMemcpy2DFromArrayAsync (void* dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream) {

    begin_func();
    //printf("cudaMemcpy2DFromArrayAsync\n");
    cudaError_t r = so_cudaMemcpy2DFromArrayAsync(dst, dpitch, src, wOffset, hOffset,width,height, kind, stream);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaMemcpy2DFromArray (void* dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind) {

    begin_func();
    //printf("cudaMemcpy2DFromArray\n");
    cudaError_t r = so_cudaMemcpy2DFromArray(dst, dpitch, src, wOffset, hOffset,width,height, kind);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaMemcpy2DAsync (void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream) {

    begin_func();
    //printf("cudaMemcpy2DAsync\n");
    cudaError_t r = so_cudaMemcpy2DAsync (dst, dpitch, src, spitch,width,height, kind, stream);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaMemcpy2DArrayToArray (cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, enum cudaMemcpyKind kind) {

    begin_func();
    //printf("cudaMemcpy2DArrayToArray\n");
    cudaError_t r = so_cudaMemcpy2DArrayToArray(dst, wOffsetDst, hOffsetDst, src, wOffsetSrc,hOffsetSrc,width,height, kind);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaMemcpy2D (void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind) {

    begin_func();
    //printf("cudaMemcpy2D\n");
    cudaError_t r = so_cudaMemcpy2D(dst, dpitch, src, spitch,width,height, kind);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaMemcpy (void* dst, const void* src, size_t count, enum cudaMemcpyKind kind) {

    begin_func();
    //printf("cudaMemcpy\n");
    VtoR2(src,dst);
    cudaError_t r = so_cudaMemcpy(dst, (const void*)src, count, kind);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaMemRangeGetAttributes (void** data, size_t* dataSizes, enum cudaMemRangeAttribute *attributes, size_t numAttributes, const void* devPtr, size_t count) {

    begin_func();
    //printf("cudaMemRangeGetAttributes\n");
    VtoR1(devPtr);
    cudaError_t r = so_cudaMemRangeGetAttributes(data, dataSizes, attributes, numAttributes , devPtr, count);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaMemRangeGetAttribute (void* data, size_t dataSize, enum cudaMemRangeAttribute attribute, const void* devPtr, size_t count) {

    begin_func();
    //printf("cudaMemRangeGetAttribute\n");
    VtoR1(devPtr);
    cudaError_t r = so_cudaMemRangeGetAttribute(data, dataSize, attribute, devPtr, count);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaMemPrefetchAsync (const void* devPtr, size_t count, int  dstDevice, cudaStream_t stream) {

    begin_func();
    //printf("cudaMemPrefetchAsync\n");
    VtoR1(devPtr);
    cudaError_t r = so_cudaMemPrefetchAsync(devPtr, count, dstDevice, stream);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaMemAdvise (const void* devPtr, size_t count, enum cudaMemoryAdvise advice, int  device) {

    begin_func();
    //printf("cudaMemAdvise\n");
    VtoR1(devPtr);
    cudaError_t r = so_cudaMemAdvise(devPtr, count, advice, device);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaHostUnregister (void* ptr) {

    begin_func();
    //printf("cudaHostUnregister\n");
    cudaError_t r = so_cudaHostUnregister(ptr);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaHostRegister (void* ptr, size_t size, unsigned int  flags) {

    begin_func();
    //printf("cudaHostRegister\n");
    cudaError_t r = so_cudaHostRegister(ptr,size, flags);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaHostGetFlags (unsigned int* pFlags, void* pHost) {

    begin_func();
    //printf("cudaHostGetFlags\n");
    cudaError_t r = so_cudaHostGetFlags(pFlags, pHost);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaGetSymbolSize (size_t* size, const void* symbol) {

    begin_func();
    //printf("cudaGetSymbolSize\n");
    cudaError_t r = so_cudaGetSymbolSize(size, symbol);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaGetSymbolAddress (void** devPtr, const void* symbol) {

    begin_func();
    //printf("cudaGetSymbolAddress\n");
    VtoR1(*devPtr);
    cudaError_t r = so_cudaGetSymbolAddress(devPtr, symbol);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaGetMipmappedArrayLevel (cudaArray_t* levelArray, cudaMipmappedArray_const_t mipmappedArray, unsigned int level) {

    begin_func();
    //printf("cudaGetMipmappedArrayLevel\n");
    cudaError_t r = so_cudaGetMipmappedArrayLevel(levelArray, mipmappedArray, level);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaFreeMipmappedArray (cudaMipmappedArray_t mipmappedArray) {

    begin_func();
    //printf("cudaFreeMipmappedArray\n");
    cudaError_t r = so_cudaFreeMipmappedArray(mipmappedArray);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaFreeArray (cudaArray_t array) {

    begin_func();
    //printf("cudaFreeArray\n");
    cudaError_t r = so_cudaFreeArray(array);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaArrayGetInfo (struct cudaChannelFormatDesc* desc, struct cudaExtent* extent, unsigned int* flags, cudaArray_t array) {

    begin_func();
    //printf("cudaArrayGetInfo\n");
    cudaError_t r = so_cudaArrayGetInfo(desc, extent, flags, array);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor (int* numBlocks, const void* func, int blockSize, size_t dynamicSMemSize) {

    begin_func();
    //printf("cudaOccupancyMaxActiveBlocksPerMultiprocessor\n");
    cudaError_t r = so_cudaOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, func, blockSize, dynamicSMemSize);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaSetDoubleForHost (double* d) {

    begin_func();
    //printf("cudaSetDoubleForHost\n");
    cudaError_t r = so_cudaSetDoubleForHost(d);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaSetDoubleForDevice (double* d) {

    begin_func();
    //printf("cudaSetDoubleForDevice\n");
    cudaError_t r = so_cudaSetDoubleForDevice(d);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaLaunchHostFunc (cudaStream_t stream, cudaHostFn_t fn, void* userData) {

    begin_func();
    //printf("cudaLaunchHostFunc\n");
    cudaError_t r = so_cudaLaunchHostFunc(stream, fn, userData);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaLaunchCooperativeKernelMultiDevice (struct cudaLaunchParams* launchParamsList, unsigned int  numDevices, unsigned int  flags) {

    begin_func();
    //printf("cudaLaunchCooperativeKernelMultiDevice\n");
    assert(0); //TODO
    cudaError_t r = so_cudaLaunchCooperativeKernelMultiDevice(launchParamsList, numDevices, flags);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaLaunchCooperativeKernel(const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream) {

    begin_func();
    //printf("cudaLaunchCooperativeKernel\n");
    assert(0); //TODO
    cudaError_t r = so_cudaLaunchCooperativeKernel(func,  gridDim,  blockDim,  args, sharedMem, stream);
    end_func();checkCudaErrors(r);
    return r;
}

/*void* cudaGetParameterBufferV2 (void* func, dim3 gridDimension, dim3 blockDimension, unsigned int  sharedMemSize) {

    begin_func();
    printf("cudaGetParameterBufferV2\n");
    void* r = (*(void* (*)(void* , dim3 , dim3 , unsigned int))(func2[5]))(func, gridDimension, blockDimension, sharedMemSize);
    end_func();checkCudaErrors(0);
    return r;
}

void* cudaGetParameterBuffer (size_t alignment, size_t size) {

    begin_func();
    printf("cudaGetParameterBuffer\n");
    void* r = (*(void* (*)(size_t , size_t))(func[125]))(alignment, size);
    end_func();checkCudaErrors(0);
    return r;
}*/

cudaError_t cudaFuncSetSharedMemConfig (const void* func, enum cudaSharedMemConfig config) {

    begin_func();
    //printf("cudaFuncSetSharedMemConfig\n");
    cudaError_t r = so_cudaFuncSetSharedMemConfig(func, config);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaFuncSetCacheConfig (const void* func, enum cudaFuncCache cacheConfig) {

    begin_func();
    //printf("cudaFuncSetCacheConfig\n");
    cudaError_t r = so_cudaFuncSetCacheConfig(func, cacheConfig);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaFuncSetAttribute (const void* func, enum cudaFuncAttribute attr, int  value) {

    begin_func();
    //printf("cudaFuncSetAttribute\n");
    cudaError_t r = so_cudaFuncSetAttribute(func, attr, value);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaWaitExternalSemaphoresAsync (const cudaExternalSemaphore_t* extSemArray, const struct cudaExternalSemaphoreWaitParams* paramsArray, unsigned int numExtSems, cudaStream_t stream) {

    begin_func();
    //printf("cudaWaitExternalSemaphoresAsync\n");
    cudaError_t r = so_cudaWaitExternalSemaphoresAsync(extSemArray, paramsArray, numExtSems, stream);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaSignalExternalSemaphoresAsync (const cudaExternalSemaphore_t* extSemArray, const struct cudaExternalSemaphoreSignalParams* paramsArray, unsigned int numExtSems, cudaStream_t stream) {

    begin_func();
    //printf("cudaSignalExternalSemaphoresAsync\n");
    cudaError_t r = so_cudaSignalExternalSemaphoresAsync(extSemArray, paramsArray, numExtSems, stream);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaImportExternalSemaphore (cudaExternalSemaphore_t* extSem_out, const struct cudaExternalSemaphoreHandleDesc* semHandleDesc) {

    begin_func();
    //printf("cudaImportExternalSemaphore\n");
    cudaError_t r = so_cudaImportExternalSemaphore(extSem_out, semHandleDesc);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaImportExternalMemory (cudaExternalMemory_t* extMem_out, const struct cudaExternalMemoryHandleDesc* memHandleDesc) {

    begin_func();
    //printf("cudaImportExternalMemory\n");
    cudaError_t r = so_cudaImportExternalMemory(extMem_out, memHandleDesc);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaExternalMemoryGetMappedMipmappedArray (cudaMipmappedArray_t* mipmap, cudaExternalMemory_t extMem, const struct cudaExternalMemoryMipmappedArrayDesc* mipmapDesc) {

    begin_func();
    //printf("cudaExternalMemoryGetMappedMipmappedArray\n");
    cudaError_t r = so_cudaExternalMemoryGetMappedMipmappedArray(mipmap, extMem, mipmapDesc);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaExternalMemoryGetMappedBuffer (void** devPtr, cudaExternalMemory_t extMem, const struct cudaExternalMemoryBufferDesc* bufferDesc) {

    begin_func();
    //printf("cudaExternalMemoryGetMappedBuffer\n");
    VtoR1(*devPtr);
    cudaError_t r = so_cudaExternalMemoryGetMappedBuffer(devPtr, extMem, bufferDesc);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaDestroyExternalSemaphore (cudaExternalSemaphore_t extSem) {

    begin_func();
    //printf("cudaDestroyExternalSemaphore\n");
    cudaError_t r = so_cudaDestroyExternalSemaphore(extSem);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaDestroyExternalMemory (cudaExternalMemory_t extMem) {

    begin_func();
    //printf("cudaDestroyExternalMemory\n");
    cudaError_t r = so_cudaDestroyExternalMemory(extMem);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaEventSynchronize (cudaEvent_t event) {

    begin_func();
    //printf("cudaEventSynchronize\n");
    cudaError_t r = so_cudaEventSynchronize(event);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaEventQuery (cudaEvent_t event) {

    begin_func();
    //printf("cudaEventQuery\n");
    cudaError_t r = so_cudaEventQuery(event);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaEventElapsedTime (float* ms, cudaEvent_t start, cudaEvent_t end) {

    begin_func();
    //printf("cudaEventElapsedTime\n");
    cudaError_t r = so_cudaEventElapsedTime(ms, start, end);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaEventCreate (cudaEvent_t* event) {

    begin_func();
    //printf("cudaEventCreate:\n");
    cudaError_t r = so_cudaEventCreate(event);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaStreamWaitEvent (cudaStream_t stream, cudaEvent_t event, unsigned int  flags) {

    begin_func();
    //printf("cudaStreamWaitEvent\n");
    cudaError_t r = so_cudaStreamWaitEvent(stream,event,flags);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaStreamQuery (cudaStream_t stream) {

    begin_func();
    //printf("cudaStreamQuery\n");
    cudaError_t r = so_cudaStreamQuery(stream);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaStreamIsCapturing(cudaStream_t stream, enum cudaStreamCaptureStatus *pCaptureStatus) {

    begin_func();
    //printf("cudaStreamIsCapturing\n");
    cudaError_t r = so_cudaStreamIsCapturing(stream, pCaptureStatus);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaStreamGetPriority (cudaStream_t hStream, int* priority) {

    begin_func();
    //printf("cudaStreamGetPriority\n");
    cudaError_t r = so_cudaStreamGetPriority(hStream, priority);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaStreamGetFlags(cudaStream_t hStream, unsigned int* flags) {

    begin_func();
    //printf("cudaStreamGetFlags\n");
    cudaError_t r = so_cudaStreamGetFlags(hStream, flags);
    end_func();checkCudaErrors(r);
    return r;
}

/*cudaError_t cudaStreamGetCaptureInfo (cudaStream_t stream, enum cudaStreamCaptureStatus ** pCaptureStatus, unsigned long long* pId) {

    begin_func();
    //printf("cudaStreamGetCaptureInfo\n");
    cudaError_t r = (*(cudaError_t (*)(cudaStream_t , enum cudaStreamCaptureStatus ** , unsigned long long*))(func[146]))(stream, pCaptureStatus, pId);
    end_func();checkCudaErrors(r);
    return r;
}*/

cudaError_t cudaStreamEndCapture (cudaStream_t stream, cudaGraph_t* pGraph) {

    begin_func();
    //printf("cudaStreamEndCapture\n");
    cudaError_t r = so_cudaStreamEndCapture(stream, pGraph);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaStreamCreate (cudaStream_t* pStream) {

    begin_func();
    //printf("cudaStreamCreate\n");
    cudaError_t r = so_cudaStreamCreate(pStream);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaStreamBeginCapture(cudaStream_t stream) {

    begin_func();
    //printf("cudaStreamBeginCapture\n");
    cudaError_t r = so_cudaStreamBeginCapture(stream);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaStreamAttachMemAsync(cudaStream_t stream, void* devPtr, size_t length, unsigned int  flags) {

    begin_func();
    //printf("cudaStreamAttachMemAsync\n");
    VtoR1(devPtr);
    cudaError_t r = so_cudaStreamAttachMemAsync(stream, devPtr, length, flags);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaStreamAddCallback(cudaStream_t stream, cudaStreamCallback_t callback, void* userData, unsigned int  flags) {

    begin_func();
    //printf("cudaStreamAddCallback\n");
    assert(0); //TODO
    cudaError_t r = so_cudaStreamAddCallback(stream,  callback, userData,flags);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaPeekAtLastError () {

    begin_func();
    //printf("cudaPeekAtLastError\n");
    cudaError_t r = so_cudaPeekAtLastError();
    end_func();checkCudaErrors(r);
    return r;
}

const char* cudaGetErrorString (cudaError_t err) {

    begin_func();
    //printf("cudaGetErrorString\n");
    const char* r = so_cudaGetErrorString(err);
    end_func();checkCudaErrors(0);
    return r;
}

const char* cudaGetErrorName (cudaError_t err) {

    begin_func();
    //printf("cudaGetErrorName\n");
    const char* r = so_cudaGetErrorName(err);
    end_func();checkCudaErrors(0);
    return r;
}

cudaError_t cudaSetValidDevices(int* device_arr, int  len) {

    begin_func();
    //printf("cudaSetValidDevices\n");
    cudaError_t r = so_cudaSetValidDevices(device_arr, len);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaSetDeviceFlags(unsigned int  flags) {

    begin_func();
    //printf("cudaSetDeviceFlags\n");
    cudaError_t r = so_cudaSetDeviceFlags(flags);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaIpcOpenMemHandle(void** devPtr, cudaIpcMemHandle_t hand, unsigned int  flags) {

    begin_func();
    //printf("cudaIpcOpenMemHandle\n");
    VtoR1(*devPtr);
    cudaError_t r = so_cudaIpcOpenMemHandle(devPtr, hand, flags);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaIpcOpenEventHandle(cudaEvent_t* event,  cudaIpcEventHandle_t hand) {

    begin_func();
    //printf("cudaIpcOpenEventHandle\n");
    cudaError_t r = so_cudaIpcOpenEventHandle(event, hand);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaIpcGetMemHandle(cudaIpcMemHandle_t* handle, void* devPtr) {

    begin_func();
    //printf("cudaIpcGetMemHandle\n");
    VtoR1(devPtr);
    cudaError_t r = so_cudaIpcGetMemHandle(handle, devPtr);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaIpcGetEventHandle(cudaIpcEventHandle_t* handle, cudaEvent_t event) {

    begin_func();
    //printf("cudaIpcGetEventHandle\n");
    cudaError_t r = so_cudaIpcGetEventHandle(handle, event);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaIpcCloseMemHandle(void* devPtr) {

    begin_func();
    //printf("cudaIpcCloseMemHandle\n");
    VtoR1(devPtr);
    cudaError_t r = so_cudaIpcCloseMemHandle(devPtr);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaGetDeviceFlags(unsigned int* flags) {

    begin_func();
    //printf("cudaGetDeviceFlags\n");
    cudaError_t r = so_cudaGetDeviceFlags(flags);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaDeviceSynchronize() {

    begin_func();
    //printf("cudaDeviceSynchronize\n");
    cudaError_t r = so_cudaDeviceSynchronize();
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaDeviceSetSharedMemConfig(enum cudaSharedMemConfig config) {

    begin_func();
    //printf("cudaDeviceSetSharedMemConfig\n");
    cudaError_t r = so_cudaDeviceSetSharedMemConfig(config);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaDeviceSetLimit(enum cudaLimit limit, size_t value) {

    begin_func();
    //printf("cudaDeviceSetLimit\n");
    cudaError_t r = so_cudaDeviceSetLimit(limit, value);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaDeviceSetCacheConfig(enum cudaFuncCache cacheConfig) {

    begin_func();
    //printf("cudaDeviceSetCacheConfig\n");
    cudaError_t r = so_cudaDeviceSetCacheConfig(cacheConfig);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaDeviceReset () {

    begin_func();
    //printf("cudaDeviceReset\n");
    cudaError_t r = so_cudaDeviceReset();
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaDeviceGetSharedMemConfig(enum cudaSharedMemConfig * pConfig) {

    begin_func();
    //printf("cudaDeviceGetSharedMemConfig\n");
    cudaError_t r = so_cudaDeviceGetSharedMemConfig(pConfig);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaDeviceGetPCIBusId(char* pciBusId, int len, int device) {

    begin_func();
    //printf("cudaDeviceGetPCIBusId\n");
    cudaError_t r = so_cudaDeviceGetPCIBusId(pciBusId, len, device);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaDeviceGetP2PAttribute(int* value, enum cudaDeviceP2PAttr attr, int srcDevice, int dstDevice) {

    begin_func();
    //printf("cudaDeviceGetP2PAttribute\n");
    cudaError_t r = so_cudaDeviceGetP2PAttribute(value, attr, srcDevice, dstDevice);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaDeviceGetLimit(size_t* pValue, enum cudaLimit limit) {

    begin_func();
    //printf("cudaDeviceGetLimit\n");
    cudaError_t r = so_cudaDeviceGetLimit(pValue, limit);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaDeviceGetCacheConfig(enum cudaFuncCache * pCacheConfig) {

    begin_func();
    //printf("cudaDeviceGetCacheConfig\n");
    cudaError_t r = so_cudaDeviceGetCacheConfig(pCacheConfig);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaDeviceGetByPCIBusId (int* device, const char* pciBusId) {

    begin_func();
    //printf("cudaDeviceGetByPCIBusId\n");
    cudaError_t r = so_cudaDeviceGetByPCIBusId(device, pciBusId);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaChooseDevice (int* device, const struct cudaDeviceProp* prop) {

    begin_func();
    //printf("cudaChooseDevice\n");
    cudaError_t r = so_cudaChooseDevice(device, prop);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaMallocMipmappedArray(cudaMipmappedArray_t* mipmappedArray, const struct cudaChannelFormatDesc* desc, struct cudaExtent extent, unsigned int  numLevels, unsigned int flags) {

    begin_func();
    //printf("cudaMallocMipmappedArray\n");
    cudaError_t r = so_cudaMallocMipmappedArray(mipmappedArray, desc, extent,numLevels, flags);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaMallocArray(cudaArray_t* array, const struct cudaChannelFormatDesc* desc, size_t width, size_t height, unsigned int  flags) {

    begin_func();
    //printf("cudaMallocArray\n");
    cudaError_t r = so_cudaMallocArray(array, desc,width,height,flags);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaMallocHost(void** ptr, size_t size) {

    begin_func();
    //printf("cudaMallocHost\n");
    cudaError_t r = so_cudaMallocHost(ptr, size);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaMalloc3DArray(cudaArray_t* array, const struct cudaChannelFormatDesc* desc, struct cudaExtent extent, unsigned int  flags) {

    begin_func();
    //printf("cudaMalloc3DArray\n");
    cudaError_t r = so_cudaMalloc3DArray(array, desc, extent, flags);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaMalloc3D(struct cudaPitchedPtr* pitchedDevPtr, struct cudaExtent extent) {

    begin_func();
    //printf("cudaMalloc3D\n");
    cudaError_t r = so_cudaMalloc3D(pitchedDevPtr, extent);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaMallocManaged(void** devPtr, size_t bytesize, unsigned int flags) {

    begin_func();
    //printf("cudaMallocManaged\n");
    assert(0);//TODO
    cudaError_t r = so_cudaMallocManaged(devPtr, bytesize, flags);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaMallocPitch(void** devPtr, size_t* pitch, size_t width, size_t height) {

    begin_func();
    //printf("cudaMallocPitch\n");
    assert(0);//TODO
    cudaError_t r = so_cudaMallocPitch(devPtr, pitch, width, height);
    end_func();checkCudaErrors(r);
    return r;
}

cudaError_t cudaMemGetInfo(size_t* free , size_t* total) {

    begin_func();
    //printf("cudaMemGetInfo\n");
    cudaError_t r = so_cudaMemGetInfo(free, total);
    end_func();checkCudaErrors(r);
    return r;
}

void __cudaRegisterVar (void **fatCubinHandle,char *hostVar,char *deviceAddress,const char *deviceName,int ext,int size,int constant,int global) {
    begin_func();
    //printf("__cudaRegisterVar(,hostVar: %s deviceAddress: %s deviceName: %s size: %d\n",
    //        hostVar, deviceAddress, deviceName, size);
    //VtoR1(deviceAddress); //TODO
    so___cudaRegisterVar(fatCubinHandle, hostVar, deviceAddress, deviceName, ext, size, constant, global);
    end_func();checkCudaErrors(0);
}

void __cudaRegisterTexture (void **fatCubinHandle,const struct textureReference *hostVar,const void **deviceAddress,const char *deviceName,int dim,int norm,int ext) {
    begin_func();
    //printf("__cudaRegisterTexture %p %p %s\n", hostVar, deviceAddress, deviceName);
    //TODO
    so___cudaRegisterTexture(fatCubinHandle,hostVar,deviceAddress,deviceName, dim, norm, ext);
    end_func();checkCudaErrors(0);
}

void __cudaRegisterSurface (void **fatCubinHandle,const struct surfaceReference  *hostVar,const void **deviceAddress,const char *deviceName,int dim,int ext) {
    begin_func();
    //printf("__cudaRegisterSurface %p %p %s\n", hostVar, deviceAddress, deviceName);
    //TODO
    so___cudaRegisterSurface(fatCubinHandle,hostVar, deviceAddress, deviceName, dim, ext);
    end_func();checkCudaErrors(0);
}

void __cudaRegisterFunction (void **fatCubinHandle,const char *hostFun,char *deviceFun,const char *deviceName,int thread_limit,uint3 *tid,uint3 *bid,dim3 *bDim,dim3 *gDim,int *wSize) {
    begin_func();
    //printf("__cudaRegisterFunction\n");
    //printf("__cudaRegisterFunction ch: %p hf: %p df: %p\n",
    //                fatCubinHandle, hostFun, deviceFun);

    so___cudaRegisterFunction(fatCubinHandle,hostFun,deviceFun, deviceName, thread_limit, tid, bid, bDim, gDim, wSize);
    end_func();checkCudaErrors(0);

}

/*void __cudaRegisterShared (void **fatCubinHandle,void **devicePtr) {
    begin_func();
    //printf("__cudaRegisterShared\n");

    (*(void (*)(void ** ,void **))(func[187]))(fatCubinHandle,devicePtr);
    end_func();checkCudaErrors(0);

}
void __cudaRegisterSharedVar (void **fatCubinHandle,void **devicePtr,size_t size,size_t alignment, int storage) {
    begin_func();
    //printf("__cudaRegisterSharedVar\n");

    (*(void (*)(void ** ,void ** ,size_t ,size_t , int))(func[188]))(fatCubinHandle, devicePtr,size, alignment, storage);
    end_func();checkCudaErrors(0);


}
int __cudaSynchronizeThreads (void** one,void* two) {
    begin_func();
    //printf("__cudaSynchronizeThreads\n");

    int r = (*(int (*)(void ** ,void **))(func[189]))(one,two);
    end_func();checkCudaErrors(0);

    return r;
}
void __cudaTextureFetch (const void* tex,void* index,int integer,void* val) {
    begin_func();
    //printf("__cudaTextureFetch\n");

    (*(void (*)(const void* ,void* ,int ,void*))(func[190]))(tex, index, integer, val);
    end_func();checkCudaErrors(0);

}
void __cudaMutexOperation (int lock) {
    begin_func();
    void (*__crcudaMutexOperationfake)(int);

    (*(void (*)(int))(func[191]))(lock);
    end_func();checkCudaErrors(0);

}
cudaError_t __cudaRegisterDeviceFunction () {
    begin_func();
    printf("__cudaRegisterDeviceFunction\n");

    cudaError_t r=(*(cudaError_t (*)(void))(func[192]))();
    end_func();checkCudaErrors(0);

    return r;
}*/

void** __cudaRegisterFatBinary (void* fatCubin) {
    begin_func();
    //printf("__cudaRegisterFatBinary\n");
    //printf("before:%p\n",fatCubin);
    void** r=so___cudaRegisterFatBinary(fatCubin);
    //printf("after:%p\n",fatCubin);
    end_func();checkCudaErrors(0);

    return r;
}

void __cudaUnregisterFatBinary (void** point) {
    begin_func();
    //printf("__cudaUnregisterFatBinary\n");
    //printf("before:%p\n",*point);
    so___cudaUnregisterFatBinary(point);
    //printf("after:%p\n",*point);
    end_func();checkCudaErrors(0);

}

cudaError_t __cudaPopCallConfiguration(dim3 *gridDim, dim3 *blockDim, size_t *sharedMem, void *stream) {

    begin_func();
    //printf("__cudaPopCallConfiguration\n");

    cudaError_t r=so___cudaPopCallConfiguration(gridDim, blockDim, sharedMem, stream);
    end_func();checkCudaErrors(0);

    return r;
}

unsigned __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim, size_t sharedMem , void *stream) {

    begin_func();
    //printf("__cudaPushCallConfiguration\n");

    unsigned r=so___cudaPushCallConfiguration(gridDim, blockDim, sharedMem, stream);
    end_func();checkCudaErrors(0);

    return r;
}

void __cudaRegisterFatBinaryEnd(void **fatCubinHandle) {

    begin_func();
    //printf("__cudaRegisterFatBinaryEnd\n");

    so___cudaRegisterFatBinaryEnd(fatCubinHandle);
    end_func();checkCudaErrors(0);

}

