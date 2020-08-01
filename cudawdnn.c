#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <cuda.h>
#include <string.h>
#include <sys/time.h>
#include <cudnn.h>
#include <assert.h>
#include <dlfcn.h>

#include "cudaw.h"


static const char LIB_STRING_DNN[] = "/usr/lib/x86_64-linux-gnu/libcudnn.so.7.6.3";

// DEFSO & LDSYM

#define MAX_FUNC 256
#include "ldsym.h"

#define DEFSO(func)  static int idx_##func; static cudnnStatus_t (*so_##func)
#define FSWAP(func)  &so_##func,

static int idx_cudnnGetErrorString;
static int idx_cudnnGetCudartVersion;
static int idx_cudnnGetVersion;
static const char *(*so_cudnnGetErrorString)(cudnnStatus_t status);
static size_t (*so_cudnnGetCudartVersion)(void);
static size_t (*so_cudnnGetVersion)(void);
DEFSO(cudnnDestroyCTCLossDescriptor)(cudnnCTCLossDescriptor_t ctcLossDesc);
DEFSO(cudnnDeriveBNTensorDescriptor)(cudnnTensorDescriptor_t derivedBnDesc,const cudnnTensorDescriptor_t xDesc,cudnnBatchNormMode_t mode);
DEFSO(cudnnActivationForward)(cudnnHandle_t handle, cudnnActivationDescriptor_t activationDesc, const void *alpha,const cudnnTensorDescriptor_t xDesc,const void *x,const void *beta,const cudnnTensorDescriptor_t   yDesc,void *y);
DEFSO(cudnnDeriveBNTensorDescriptor)(cudnnTensorDescriptor_t derivedBnDesc, const cudnnTensorDescriptor_t xDesc, cudnnBatchNormMode_t mode);        
DEFSO(cudnnActivationForward)(cudnnHandle_t handle, cudnnActivationDescriptor_t activationDesc, const void *alpha, const cudnnTensorDescriptor_t xDesc, const void *x, const void *beta, const cudnnTensorDescriptor_t yDesc, void *y);      
DEFSO(cudnnQueryRuntimeError)(cudnnHandle_t handle, cudnnStatus_t *rstatus, cudnnErrQueryMode_t mode, cudnnRuntimeTag_t *tag);        
DEFSO(cudnnGetProperty)(libraryPropertyType type, int *value);         
DEFSO(cudnnCreate)(cudnnHandle_t *handle);         
DEFSO(cudnnDestroy)(cudnnHandle_t handle);         
DEFSO(cudnnSetStream)(cudnnHandle_t handle, cudaStream_t streamId);         
DEFSO(cudnnGetStream)(cudnnHandle_t handle, cudaStream_t *streamId);         
DEFSO(cudnnCreateTensorDescriptor)(cudnnTensorDescriptor_t *tensorDesc);         
DEFSO(cudnnSetTensor4dDescriptor)(cudnnTensorDescriptor_t tensorDesc, cudnnTensorFormat_t format, cudnnDataType_t dataType, int n, int c, int h, int w);       
DEFSO(cudnnSetTensor4dDescriptorEx)(cudnnTensorDescriptor_t tensorDesc, cudnnDataType_t dataType, int n, int c, int h, int w,int nStride, int cStride, int hStride, int wStride);       
DEFSO(cudnnGetTensor4dDescriptor)(const cudnnTensorDescriptor_t tensorDesc, cudnnDataType_t *dataType, int *n, int *c, int *h, int *w, int *nStride, int *cStride, int *hStride, int *wStride);      
DEFSO(cudnnSetTensorNdDescriptor)(cudnnTensorDescriptor_t tensorDesc, cudnnDataType_t dataType, int nbDims, const int dimA[], const int strideA[]);        
DEFSO(cudnnSetTensorNdDescriptorEx)(cudnnTensorDescriptor_t tensorDesc, cudnnTensorFormat_t format, cudnnDataType_t dataType, int nbDims, const int dimA[]);        
DEFSO(cudnnGetTensorNdDescriptor)(const cudnnTensorDescriptor_t tensorDesc, int nbDimsRequested, cudnnDataType_t *dataType, int *nbDims, int dimA[], int strideA[]);        
DEFSO(cudnnGetTensorSizeInBytes)(const cudnnTensorDescriptor_t tensorDesc, size_t *size);        
DEFSO(cudnnDestroyTensorDescriptor)(cudnnTensorDescriptor_t tensorDesc);         
DEFSO(cudnnInitTransformDest)(const cudnnTensorTransformDescriptor_t transformDesc, const cudnnTensorDescriptor_t srcDesc, cudnnTensorDescriptor_t destDesc, size_t *destSizeInBytes);        
DEFSO(cudnnCreateTensorTransformDescriptor)(cudnnTensorTransformDescriptor_t *transformDesc);         
DEFSO(cudnnSetTensorTransformDescriptor)(cudnnTensorTransformDescriptor_t transformDesc, const uint32_t nbDims, const cudnnTensorFormat_t destFormat, const int32_t padBeforeA[], const int32_t padAfterA[], const uint32_t foldA[], const cudnnFoldingDirection_t direction);      
DEFSO(cudnnGetTensorTransformDescriptor)(cudnnTensorTransformDescriptor_t transformDesc, uint32_t nbDimsRequested, cudnnTensorFormat_t *destFormat, int32_t padBeforeA[], int32_t padAfterA[], uint32_t foldA[], cudnnFoldingDirection_t *direction);        
DEFSO(cudnnDestroyTensorTransformDescriptor)(cudnnTensorTransformDescriptor_t transformDesc);         
DEFSO(cudnnTransformTensor)(cudnnHandle_t handle, const void *alpha, const cudnnTensorDescriptor_t xDesc, const void *x, const void *beta, const cudnnTensorDescriptor_t yDesc, void *y);      
DEFSO(cudnnTransformTensorEx)(cudnnHandle_t handle, const cudnnTensorTransformDescriptor_t transDesc, const void *alpha, const cudnnTensorDescriptor_t srcDesc, const void *srcData, const void *beta, const cudnnTensorDescriptor_t destDesc, void *destData);      
DEFSO(cudnnGetFoldedConvBackwardDataDescriptors)(const cudnnHandle_t handle, const cudnnFilterDescriptor_t filterDesc, const cudnnTensorDescriptor_t diffDesc, const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t gradDesc, const cudnnTensorFormat_t transformFormat, cudnnFilterDescriptor_t foldedFilterDesc, cudnnTensorDescriptor_t paddedDiffDesc, cudnnConvolutionDescriptor_t foldedConvDesc, cudnnTensorDescriptor_t foldedGradDesc, cudnnTensorTransformDescriptor_t filterFoldTransDesc, cudnnTensorTransformDescriptor_t diffPadTransDesc, cudnnTensorTransformDescriptor_t gradFoldTransDesc, cudnnTensorTransformDescriptor_t gradUnfoldTransDesc);    
DEFSO(cudnnAddTensor)(cudnnHandle_t handle, const void *alpha, const cudnnTensorDescriptor_t aDesc, const void *A, const void *beta, const cudnnTensorDescriptor_t cDesc, void *C);      
DEFSO(cudnnCreateOpTensorDescriptor)(cudnnOpTensorDescriptor_t *opTensorDesc);         
DEFSO(cudnnSetOpTensorDescriptor)(cudnnOpTensorDescriptor_t opTensorDesc, cudnnOpTensorOp_t opTensorOp, cudnnDataType_t opTensorCompType, cudnnNanPropagation_t opTensorNanOpt);        
DEFSO(cudnnGetOpTensorDescriptor)(const cudnnOpTensorDescriptor_t opTensorDesc, cudnnOpTensorOp_t *opTensorOp, cudnnDataType_t *opTensorCompType, cudnnNanPropagation_t *opTensorNanOpt);        
DEFSO(cudnnDestroyOpTensorDescriptor)(cudnnOpTensorDescriptor_t opTensorDesc);         
DEFSO(cudnnOpTensor)(cudnnHandle_t handle, const cudnnOpTensorDescriptor_t opTensorDesc, const void *alpha1, const cudnnTensorDescriptor_t aDesc, const void *A, const void *alpha2, const cudnnTensorDescriptor_t bDesc, const void *B, const void *beta, const cudnnTensorDescriptor_t cDesc, void *C);    
DEFSO(cudnnCreateReduceTensorDescriptor)(cudnnReduceTensorDescriptor_t *reduceTensorDesc);         
DEFSO(cudnnSetReduceTensorDescriptor)(cudnnReduceTensorDescriptor_t reduceTensorDesc, cudnnReduceTensorOp_t reduceTensorOp, cudnnDataType_t reduceTensorCompType, cudnnNanPropagation_t reduceTensorNanOpt, cudnnReduceTensorIndices_t reduceTensorIndices, cudnnIndicesType_t reduceTensorIndicesType);        
DEFSO(cudnnGetReduceTensorDescriptor)(const cudnnReduceTensorDescriptor_t reduceTensorDesc, cudnnReduceTensorOp_t *reduceTensorOp, cudnnDataType_t *reduceTensorCompType, cudnnNanPropagation_t *reduceTensorNanOpt, cudnnReduceTensorIndices_t *reduceTensorIndices, cudnnIndicesType_t *reduceTensorIndicesType);        
DEFSO(cudnnDestroyReduceTensorDescriptor)(cudnnReduceTensorDescriptor_t reduceTensorDesc);         
DEFSO(cudnnGetReductionIndicesSize)(cudnnHandle_t handle, const cudnnReduceTensorDescriptor_t reduceTensorDesc, const cudnnTensorDescriptor_t aDesc, const cudnnTensorDescriptor_t cDesc, size_t *sizeInBytes);       
DEFSO(cudnnGetReductionWorkspaceSize)(cudnnHandle_t handle, const cudnnReduceTensorDescriptor_t reduceTensorDesc, const cudnnTensorDescriptor_t aDesc, const cudnnTensorDescriptor_t cDesc, size_t *sizeInBytes);       
DEFSO(cudnnReduceTensor)(cudnnHandle_t handle, const cudnnReduceTensorDescriptor_t reduceTensorDesc, void *indices, size_t indicesSizeInBytes, void *workspace, size_t workspaceSizeInBytes, const void *alpha, const cudnnTensorDescriptor_t aDesc, const void *A, const void *beta, const cudnnTensorDescriptor_t cDesc, void *C);     
DEFSO(cudnnSetTensor)(cudnnHandle_t handle, const cudnnTensorDescriptor_t yDesc, void *y, const void *valuePtr);       
DEFSO(cudnnScaleTensor)(cudnnHandle_t handle, const cudnnTensorDescriptor_t yDesc, void *y, const void *alpha);       
DEFSO(cudnnCreateFilterDescriptor)(cudnnFilterDescriptor_t *filterDesc);         
DEFSO(cudnnSetFilter4dDescriptor)(cudnnFilterDescriptor_t filterDesc, cudnnDataType_t dataType, cudnnTensorFormat_t format, int k, int c, int h, int w);       
DEFSO(cudnnGetFilter4dDescriptor)(const cudnnFilterDescriptor_t filterDesc, cudnnDataType_t *dataType, cudnnTensorFormat_t *format, int *k, int *c, int *h, int *w);       
DEFSO(cudnnSetFilterNdDescriptor)(cudnnFilterDescriptor_t filterDesc, cudnnDataType_t dataType, cudnnTensorFormat_t format, int nbDims, const int filterDimA[]);        
DEFSO(cudnnGetFilterNdDescriptor)(const cudnnFilterDescriptor_t filterDesc, int nbDimsRequested, cudnnDataType_t *dataType, cudnnTensorFormat_t *format, int *nbDims, int filterDimA[]);       
DEFSO(cudnnGetFilterSizeInBytes)(const cudnnFilterDescriptor_t filterDesc, size_t *size);        
DEFSO(cudnnTransformFilter)(cudnnHandle_t handle, const cudnnTensorTransformDescriptor_t transDesc, const void *alpha, const cudnnFilterDescriptor_t srcDesc, const void *srcData, const void *beta, const cudnnFilterDescriptor_t destDesc, void *destData);      
DEFSO(cudnnDestroyFilterDescriptor)(cudnnFilterDescriptor_t filterDesc);         
DEFSO(cudnnReorderFilterAndBias)(cudnnHandle_t handle, const cudnnFilterDescriptor_t filterDesc, cudnnReorderType_t reorderType, const void *filterData, void *reorderedFilterData, int reorderBias, const void *biasData, void *reorderedBiasData);       
DEFSO(cudnnCreateConvolutionDescriptor)(cudnnConvolutionDescriptor_t *convDesc);         
DEFSO(cudnnSetConvolutionMathType)(cudnnConvolutionDescriptor_t convDesc, cudnnMathType_t mathType);         
DEFSO(cudnnGetConvolutionMathType)(cudnnConvolutionDescriptor_t convDesc, cudnnMathType_t *mathType);         
DEFSO(cudnnSetConvolutionGroupCount)(cudnnConvolutionDescriptor_t convDesc, int groupCount);         
DEFSO(cudnnGetConvolutionGroupCount)(cudnnConvolutionDescriptor_t convDesc, int *groupCount);         
DEFSO(cudnnSetConvolutionReorderType)(cudnnConvolutionDescriptor_t convDesc, cudnnReorderType_t reorderType);         
DEFSO(cudnnGetConvolutionReorderType)(cudnnConvolutionDescriptor_t convDesc, cudnnReorderType_t *reorderType);         
DEFSO(cudnnSetConvolution2dDescriptor)(cudnnConvolutionDescriptor_t convDesc, int pad_h, int pad_w, int u, int v, int dilation_h, int dilation_w, cudnnConvolutionMode_t mode, cudnnDataType_t computeType);      
DEFSO(cudnnGetConvolution2dDescriptor)(const cudnnConvolutionDescriptor_t convDesc, int *pad_h, int *pad_w, int *u, int *v, int *dilation_h, int *dilation_w, cudnnConvolutionMode_t *mode, cudnnDataType_t *computeType);      
DEFSO(cudnnGetConvolution2dForwardOutputDim)(const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t inputTensorDesc, const cudnnFilterDescriptor_t filterDesc, int *n, int *c, int *h, int *w);       
DEFSO(cudnnSetConvolutionNdDescriptor)(cudnnConvolutionDescriptor_t convDesc, int arrayLength, const int padA[], const int filterStrideA[], const int dilationA[], cudnnConvolutionMode_t mode, cudnnDataType_t computeType);       
DEFSO(cudnnGetConvolutionNdDescriptor)(const cudnnConvolutionDescriptor_t convDesc, int arrayLengthRequested, int *arrayLength, int padA[], int strideA[], int dilationA[], cudnnConvolutionMode_t *mode, cudnnDataType_t *computeType);       
DEFSO(cudnnGetConvolutionNdForwardOutputDim)(const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t inputTensorDesc, const cudnnFilterDescriptor_t filterDesc, int nbDims, int tensorOuputDimA[]);       
DEFSO(cudnnDestroyConvolutionDescriptor)(cudnnConvolutionDescriptor_t convDesc);         
DEFSO(cudnnGetConvolutionForwardAlgorithmMaxCount)(cudnnHandle_t handle, int *count);         
DEFSO(cudnnFindConvolutionForwardAlgorithm)(cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc, const cudnnFilterDescriptor_t wDesc, const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t yDesc, const int requestedAlgoCount, int *returnedAlgoCount, cudnnConvolutionFwdAlgoPerf_t *perfResults);      
DEFSO(cudnnFindConvolutionForwardAlgorithmEx)(cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc, const void *x, const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t yDesc, void *y, const int requestedAlgoCount, int *returnedAlgoCount, cudnnConvolutionFwdAlgoPerf_t *perfResults, void *workSpace, size_t workSpaceSizeInBytes);    
DEFSO(cudnnGetConvolutionForwardAlgorithm)(cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc, const cudnnFilterDescriptor_t wDesc, const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t yDesc, cudnnConvolutionFwdPreference_t preference, size_t memoryLimitInBytes, cudnnConvolutionFwdAlgo_t *algo);      
DEFSO(cudnnGetConvolutionForwardAlgorithm_v7)(cudnnHandle_t handle, const cudnnTensorDescriptor_t srcDesc, const cudnnFilterDescriptor_t filterDesc, const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t destDesc, const int requestedAlgoCount, int *returnedAlgoCount, cudnnConvolutionFwdAlgoPerf_t *perfResults);      
DEFSO(cudnnGetConvolutionForwardWorkspaceSize)(cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc, const cudnnFilterDescriptor_t wDesc, const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t yDesc, cudnnConvolutionFwdAlgo_t algo, size_t *sizeInBytes);       
DEFSO(cudnnConvolutionForward)(cudnnHandle_t handle, const void *alpha, const cudnnTensorDescriptor_t xDesc, const void *x, const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnConvolutionDescriptor_t convDesc, cudnnConvolutionFwdAlgo_t algo, void *workSpace, size_t workSpaceSizeInBytes, const void *beta, const cudnnTensorDescriptor_t yDesc, void *y);    
DEFSO(cudnnConvolutionBiasActivationForward)(cudnnHandle_t handle, const void *alpha1, const cudnnTensorDescriptor_t xDesc, const void *x, const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnConvolutionDescriptor_t convDesc, cudnnConvolutionFwdAlgo_t algo, void *workSpace, size_t workSpaceSizeInBytes, const void *alpha2, const cudnnTensorDescriptor_t zDesc, const void *z, const cudnnTensorDescriptor_t biasDesc, const void *bias, const cudnnActivationDescriptor_t activationDesc, const cudnnTensorDescriptor_t yDesc, void *y);  
DEFSO(cudnnConvolutionBackwardBias)(cudnnHandle_t handle, const void *alpha, const cudnnTensorDescriptor_t dyDesc, const void *dy, const void *beta, const cudnnTensorDescriptor_t dbDesc, void *db);      
DEFSO(cudnnGetConvolutionBackwardFilterAlgorithmMaxCount)(cudnnHandle_t handle, int *count);         
DEFSO(cudnnFindConvolutionBackwardFilterAlgorithm)(cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc, const cudnnTensorDescriptor_t dyDesc, const cudnnConvolutionDescriptor_t convDesc, const cudnnFilterDescriptor_t dwDesc, const int requestedAlgoCount, int *returnedAlgoCount, cudnnConvolutionBwdFilterAlgoPerf_t *perfResults);      
DEFSO(cudnnFindConvolutionBackwardFilterAlgorithmEx)(cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc, const void *x, const cudnnTensorDescriptor_t dyDesc, const void *y, const cudnnConvolutionDescriptor_t convDesc, const cudnnFilterDescriptor_t dwDesc, void *dw, const int requestedAlgoCount, int *returnedAlgoCount, cudnnConvolutionBwdFilterAlgoPerf_t *perfResults, void *workSpace, size_t workSpaceSizeInBytes);    
DEFSO(cudnnGetConvolutionBackwardFilterAlgorithm)(cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc, const cudnnTensorDescriptor_t dyDesc, const cudnnConvolutionDescriptor_t convDesc, const cudnnFilterDescriptor_t dwDesc, cudnnConvolutionBwdFilterPreference_t preference, size_t memoryLimitInBytes, cudnnConvolutionBwdFilterAlgo_t *algo);      
DEFSO(cudnnGetConvolutionBackwardFilterAlgorithm_v7)(cudnnHandle_t handle, const cudnnTensorDescriptor_t srcDesc, const cudnnTensorDescriptor_t diffDesc, const cudnnConvolutionDescriptor_t convDesc, const cudnnFilterDescriptor_t gradDesc, const int requestedAlgoCount, int *returnedAlgoCount, cudnnConvolutionBwdFilterAlgoPerf_t *perfResults);      
DEFSO(cudnnGetConvolutionBackwardFilterWorkspaceSize)(cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc, const cudnnTensorDescriptor_t dyDesc, const cudnnConvolutionDescriptor_t convDesc, const cudnnFilterDescriptor_t gradDesc, cudnnConvolutionBwdFilterAlgo_t algo, size_t *sizeInBytes);       
DEFSO(cudnnConvolutionBackwardFilter)(cudnnHandle_t handle, const void *alpha, const cudnnTensorDescriptor_t xDesc, const void *x, const cudnnTensorDescriptor_t dyDesc, const void *dy, const cudnnConvolutionDescriptor_t convDesc, cudnnConvolutionBwdFilterAlgo_t algo, void *workSpace, size_t workSpaceSizeInBytes, const void *beta, const cudnnFilterDescriptor_t dwDesc, void *dw);    
DEFSO(cudnnGetConvolutionBackwardDataAlgorithmMaxCount)(cudnnHandle_t handle, int *count);         
DEFSO(cudnnFindConvolutionBackwardDataAlgorithm)(cudnnHandle_t handle, const cudnnFilterDescriptor_t wDesc, const cudnnTensorDescriptor_t dyDesc, const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t dxDesc, const int requestedAlgoCount, int *returnedAlgoCount, cudnnConvolutionBwdDataAlgoPerf_t *perfResults);      
DEFSO(cudnnFindConvolutionBackwardDataAlgorithmEx)(cudnnHandle_t handle, const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnTensorDescriptor_t dyDesc, const void *dy, const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t dxDesc, void *dx, const int requestedAlgoCount, int *returnedAlgoCount, cudnnConvolutionBwdDataAlgoPerf_t *perfResults, void *workSpace, size_t workSpaceSizeInBytes);    
DEFSO(cudnnGetConvolutionBackwardDataAlgorithm)(cudnnHandle_t handle, const cudnnFilterDescriptor_t wDesc, const cudnnTensorDescriptor_t dyDesc, const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t dxDesc, cudnnConvolutionBwdDataPreference_t preference, size_t memoryLimitInBytes, cudnnConvolutionBwdDataAlgo_t *algo);      
DEFSO(cudnnGetConvolutionBackwardDataAlgorithm_v7)(cudnnHandle_t handle, const cudnnFilterDescriptor_t filterDesc, const cudnnTensorDescriptor_t diffDesc, const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t gradDesc, const int requestedAlgoCount, int *returnedAlgoCount, cudnnConvolutionBwdDataAlgoPerf_t *perfResults);      
DEFSO(cudnnGetConvolutionBackwardDataWorkspaceSize)(cudnnHandle_t handle, const cudnnFilterDescriptor_t wDesc, const cudnnTensorDescriptor_t dyDesc, const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t dxDesc, cudnnConvolutionBwdDataAlgo_t algo, size_t *sizeInBytes);       
DEFSO(cudnnConvolutionBackwardData)(cudnnHandle_t handle, const void *alpha, const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnTensorDescriptor_t dyDesc, const void *dy, const cudnnConvolutionDescriptor_t convDesc, cudnnConvolutionBwdDataAlgo_t algo, void *workSpace, size_t workSpaceSizeInBytes, const void *beta, const cudnnTensorDescriptor_t dxDesc, void *dx);    
DEFSO(cudnnIm2Col)(cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc, const void *x, const cudnnFilterDescriptor_t wDesc, const cudnnConvolutionDescriptor_t convDesc, void *colBuffer);       
DEFSO(cudnnSoftmaxForward)(cudnnHandle_t handle, cudnnSoftmaxAlgorithm_t algo, cudnnSoftmaxMode_t mode, const void *alpha, const cudnnTensorDescriptor_t xDesc, const void *x, const void *beta, const cudnnTensorDescriptor_t yDesc, void *y);      
DEFSO(cudnnSoftmaxBackward)(cudnnHandle_t handle, cudnnSoftmaxAlgorithm_t algo, cudnnSoftmaxMode_t mode, const void *alpha, const cudnnTensorDescriptor_t yDesc, const void *y, const cudnnTensorDescriptor_t dyDesc, const void *dy, const void *beta, const cudnnTensorDescriptor_t dxDesc, void *dx);     
DEFSO(cudnnCreatePoolingDescriptor)(cudnnPoolingDescriptor_t *poolingDesc);         
DEFSO(cudnnSetPooling2dDescriptor)(cudnnPoolingDescriptor_t poolingDesc, cudnnPoolingMode_t mode, cudnnNanPropagation_t maxpoolingNanOpt, int windowHeight, int windowWidth, int verticalPadding, int horizontalPadding, int verticalStride, int horizontalStride);       
DEFSO(cudnnGetPooling2dDescriptor)(const cudnnPoolingDescriptor_t poolingDesc, cudnnPoolingMode_t *mode, cudnnNanPropagation_t *maxpoolingNanOpt, int *windowHeight, int *windowWidth, int *verticalPadding, int *horizontalPadding, int *verticalStride, int *horizontalStride);       
DEFSO(cudnnSetPoolingNdDescriptor)(cudnnPoolingDescriptor_t poolingDesc, const cudnnPoolingMode_t mode, const cudnnNanPropagation_t maxpoolingNanOpt, int nbDims, const int windowDimA[], const int paddingA[], const int strideA[]);      
DEFSO(cudnnGetPoolingNdDescriptor)(const cudnnPoolingDescriptor_t poolingDesc, int nbDimsRequested, cudnnPoolingMode_t *mode, cudnnNanPropagation_t *maxpoolingNanOpt, int *nbDims, int windowDimA[], int paddingA[], int strideA[]);       
DEFSO(cudnnGetPoolingNdForwardOutputDim)(const cudnnPoolingDescriptor_t poolingDesc, const cudnnTensorDescriptor_t inputTensorDesc, int nbDims, int outputTensorDimA[]);        
DEFSO(cudnnGetPooling2dForwardOutputDim)(const cudnnPoolingDescriptor_t poolingDesc, const cudnnTensorDescriptor_t inputTensorDesc, int *n, int *c, int *h, int *w);       
DEFSO(cudnnDestroyPoolingDescriptor)(cudnnPoolingDescriptor_t poolingDesc);         
DEFSO(cudnnPoolingForward)(cudnnHandle_t handle, const cudnnPoolingDescriptor_t poolingDesc, const void *alpha, const cudnnTensorDescriptor_t xDesc, const void *x, const void *beta, const cudnnTensorDescriptor_t yDesc, void *y);      
DEFSO(cudnnPoolingBackward)(cudnnHandle_t handle, const cudnnPoolingDescriptor_t poolingDesc, const void *alpha, const cudnnTensorDescriptor_t yDesc, const void *y, const cudnnTensorDescriptor_t dyDesc, const void *dy, const cudnnTensorDescriptor_t xDesc, const void *x, const void *beta, const cudnnTensorDescriptor_t dxDesc, void *dx);    
DEFSO(cudnnCreateActivationDescriptor)(cudnnActivationDescriptor_t *activationDesc);         
DEFSO(cudnnSetActivationDescriptor)(cudnnActivationDescriptor_t activationDesc, cudnnActivationMode_t mode, cudnnNanPropagation_t reluNanOpt, double coef);        
DEFSO(cudnnGetActivationDescriptor)(const cudnnActivationDescriptor_t activationDesc, cudnnActivationMode_t *mode, cudnnNanPropagation_t *reluNanOpt, double *coef);        
DEFSO(cudnnDestroyActivationDescriptor)(cudnnActivationDescriptor_t activationDesc);         
DEFSO(cudnnActivationBackward)(cudnnHandle_t handle, cudnnActivationDescriptor_t activationDesc, const void *alpha, const cudnnTensorDescriptor_t yDesc, const void *y, const cudnnTensorDescriptor_t dyDesc, const void *dy, const cudnnTensorDescriptor_t xDesc, const void *x, const void *beta, const cudnnTensorDescriptor_t dxDesc, void *dx);    
DEFSO(cudnnCreateLRNDescriptor)(cudnnLRNDescriptor_t *normDesc);         
DEFSO(cudnnSetLRNDescriptor)(cudnnLRNDescriptor_t normDesc, unsigned lrnN, double lrnAlpha, double lrnBeta, double lrnK);       
DEFSO(cudnnGetLRNDescriptor)(cudnnLRNDescriptor_t normDesc, unsigned *lrnN, double *lrnAlpha, double *lrnBeta, double *lrnK);       
DEFSO(cudnnDestroyLRNDescriptor)(cudnnLRNDescriptor_t lrnDesc);         
DEFSO(cudnnLRNCrossChannelForward)(cudnnHandle_t handle, cudnnLRNDescriptor_t normDesc, cudnnLRNMode_t lrnMode, const void *alpha, const cudnnTensorDescriptor_t xDesc, const void *x, const void *beta, const cudnnTensorDescriptor_t yDesc, void *y);      
DEFSO(cudnnLRNCrossChannelBackward)(cudnnHandle_t handle, cudnnLRNDescriptor_t normDesc, cudnnLRNMode_t lrnMode, const void *alpha, const cudnnTensorDescriptor_t yDesc, const void *y, const cudnnTensorDescriptor_t dyDesc, const void *dy, const cudnnTensorDescriptor_t xDesc, const void *x, const void *beta, const cudnnTensorDescriptor_t dxDesc, void *dx);    
DEFSO(cudnnDivisiveNormalizationForward)(cudnnHandle_t handle, cudnnLRNDescriptor_t normDesc, cudnnDivNormMode_t mode, const void *alpha, const cudnnTensorDescriptor_t xDesc, const void *x, const void *means, void *temp, void *temp2, const void *beta, const cudnnTensorDescriptor_t yDesc, void *y);     
DEFSO(cudnnDivisiveNormalizationBackward)(cudnnHandle_t handle, cudnnLRNDescriptor_t normDesc, cudnnDivNormMode_t mode, const void *alpha, const cudnnTensorDescriptor_t xDesc, const void *x, const void *means, const void *dy, void *temp, void *temp2, const void *beta, const cudnnTensorDescriptor_t dXdMeansDesc, void *dx, void *dMeans);   
DEFSO(cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize)(cudnnHandle_t handle, cudnnBatchNormMode_t mode, cudnnBatchNormOps_t bnOps, const cudnnTensorDescriptor_t xDesc, const cudnnTensorDescriptor_t zDesc, const cudnnTensorDescriptor_t yDesc, const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const cudnnActivationDescriptor_t activationDesc, size_t *sizeInBytes);      
DEFSO(cudnnGetBatchNormalizationBackwardExWorkspaceSize)(cudnnHandle_t handle, cudnnBatchNormMode_t mode, cudnnBatchNormOps_t bnOps, const cudnnTensorDescriptor_t xDesc, const cudnnTensorDescriptor_t yDesc, const cudnnTensorDescriptor_t dyDesc, const cudnnTensorDescriptor_t dzDesc, const cudnnTensorDescriptor_t dxDesc, const cudnnTensorDescriptor_t dBnScaleBiasDesc, const cudnnActivationDescriptor_t activationDesc, size_t *sizeInBytes);     
DEFSO(cudnnGetBatchNormalizationTrainingExReserveSpaceSize)(cudnnHandle_t handle, cudnnBatchNormMode_t mode, cudnnBatchNormOps_t bnOps, const cudnnActivationDescriptor_t activationDesc, const cudnnTensorDescriptor_t xDesc, size_t *sizeInBytes);       
DEFSO(cudnnBatchNormalizationForwardTraining)(cudnnHandle_t handle, cudnnBatchNormMode_t mode, const void *alpha, const void *beta, const cudnnTensorDescriptor_t xDesc, const void *x, const cudnnTensorDescriptor_t yDesc, void *y, const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const void *bnScale, const void *bnBias, double exponentialAverageFactor, void *resultRunningMean, void *resultRunningVariance, double epsilon, void *resultSaveMean, void *resultSaveInvVariance);  
DEFSO(cudnnBatchNormalizationForwardTrainingEx)(cudnnHandle_t handle, cudnnBatchNormMode_t mode, cudnnBatchNormOps_t bnOps, const void *alpha, const void *beta, const cudnnTensorDescriptor_t xDesc, const void *xData, const cudnnTensorDescriptor_t zDesc, const void *zData, const cudnnTensorDescriptor_t yDesc, void *yData, const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const void *bnScale, const void *bnBias, double exponentialAverageFactor, void *resultRunningMean, void *resultRunningVariance, double epsilon, void *resultSaveMean, void *resultSaveInvVariance, cudnnActivationDescriptor_t activationDesc, void *workspace, size_t workSpaceSizeInBytes, void *reserveSpace, size_t reserveSpaceSizeInBytes); 
DEFSO(cudnnBatchNormalizationForwardInference)(cudnnHandle_t handle, cudnnBatchNormMode_t mode, const void *alpha, const void *beta, const cudnnTensorDescriptor_t xDesc, const void *x, const cudnnTensorDescriptor_t yDesc, void *y, const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const void *bnScale, const void *bnBias, const void *estimatedMean, const void *estimatedVariance, double epsilon);   
DEFSO(cudnnBatchNormalizationBackward)(cudnnHandle_t handle, cudnnBatchNormMode_t mode, const void *alphaDataDiff, const void *betaDataDiff, const void *alphaParamDiff, const void *betaParamDiff, const cudnnTensorDescriptor_t xDesc, const void *x, const cudnnTensorDescriptor_t dyDesc, const void *dy, const cudnnTensorDescriptor_t dxDesc, void *dx, const cudnnTensorDescriptor_t dBnScaleBiasDesc, const void *bnScale, void *dBnScaleResult, void *dBnBiasResult, double epsilon, const void *savedMean, const void *savedInvVariance); 
DEFSO(cudnnBatchNormalizationBackwardEx)(cudnnHandle_t handle,cudnnBatchNormMode_t mode,cudnnBatchNormOps_t bnOps,const void *alphaDataDiff,const void *betaDataDiff,const void *alphaParamDiff,const void *betaParamDiff,const cudnnTensorDescriptor_t xDesc,const void *xData,const cudnnTensorDescriptor_t yDesc,const void *yData,const cudnnTensorDescriptor_t dyDesc,const void *dyData,const cudnnTensorDescriptor_t dzDesc,void *dzData,const cudnnTensorDescriptor_t dxDesc,void *dxData,const cudnnTensorDescriptor_t dBnScaleBiasDesc,const void *bnScaleData,const void *bnBiasData,void *dBnScaleData,void *dBnBiasData,double epsilon, const void *savedMean,const void *savedInvVariance,cudnnActivationDescriptor_t activationDesc,void *workSpace,size_t workSpaceSizeInBytes,void *reserveSpace,size_t reserveSpaceSizeInBytes);
DEFSO(cudnnCreateSpatialTransformerDescriptor)(cudnnSpatialTransformerDescriptor_t *stDesc);         
DEFSO(cudnnSetSpatialTransformerNdDescriptor)(cudnnSpatialTransformerDescriptor_t stDesc, cudnnSamplerType_t samplerType, cudnnDataType_t dataType, const int nbDims, const int dimA[]);        
DEFSO(cudnnDestroySpatialTransformerDescriptor)(cudnnSpatialTransformerDescriptor_t stDesc);         
DEFSO(cudnnSpatialTfGridGeneratorForward)(cudnnHandle_t handle, const cudnnSpatialTransformerDescriptor_t stDesc, const void *theta, void *grid);        
DEFSO(cudnnSpatialTfGridGeneratorBackward)(cudnnHandle_t handle, const cudnnSpatialTransformerDescriptor_t stDesc, const void *dgrid, void *dtheta);        
DEFSO(cudnnSpatialTfSamplerForward)(cudnnHandle_t handle, cudnnSpatialTransformerDescriptor_t stDesc, const void *alpha, const cudnnTensorDescriptor_t xDesc, const void *x, const void *grid, const void *beta, cudnnTensorDescriptor_t yDesc, void *y);      
DEFSO(cudnnSpatialTfSamplerBackward)(cudnnHandle_t handle, cudnnSpatialTransformerDescriptor_t stDesc, const void *alpha, const cudnnTensorDescriptor_t xDesc, const void *x, const void *beta, const cudnnTensorDescriptor_t dxDesc, void *dx, const void *alphaDgrid, const cudnnTensorDescriptor_t dyDesc, const void *dy, const void *grid, const void *betaDgrid, void *dgrid);   
DEFSO(cudnnCreateDropoutDescriptor)(cudnnDropoutDescriptor_t *dropoutDesc);         
DEFSO(cudnnDestroyDropoutDescriptor)(cudnnDropoutDescriptor_t dropoutDesc);         
DEFSO(cudnnDropoutGetStatesSize)(cudnnHandle_t handle, size_t *sizeInBytes);         
DEFSO(cudnnDropoutGetReserveSpaceSize)(cudnnTensorDescriptor_t xdesc, size_t *sizeInBytes);         
DEFSO(cudnnSetDropoutDescriptor)(cudnnDropoutDescriptor_t dropoutDesc, cudnnHandle_t handle, float dropout, void *states, size_t stateSizeInBytes, unsigned long long seed);       
DEFSO(cudnnRestoreDropoutDescriptor)(cudnnDropoutDescriptor_t dropoutDesc, cudnnHandle_t handle, float dropout, void *states, size_t stateSizeInBytes, unsigned long long seed);       
DEFSO(cudnnGetDropoutDescriptor)(cudnnDropoutDescriptor_t dropoutDesc, cudnnHandle_t handle, float *dropout, void **states, unsigned long long *seed);        
DEFSO(cudnnDropoutForward)(cudnnHandle_t handle, const cudnnDropoutDescriptor_t dropoutDesc, const cudnnTensorDescriptor_t xdesc, const void *x, const cudnnTensorDescriptor_t ydesc, void *y, void *reserveSpace, size_t reserveSpaceSizeInBytes);      
DEFSO(cudnnDropoutBackward)(cudnnHandle_t handle, const cudnnDropoutDescriptor_t dropoutDesc, const cudnnTensorDescriptor_t dydesc, const void *dy, const cudnnTensorDescriptor_t dxdesc, void *dx, void *reserveSpace, size_t reserveSpaceSizeInBytes);      
DEFSO(cudnnCreateRNNDescriptor)(cudnnRNNDescriptor_t *rnnDesc);         
DEFSO(cudnnDestroyRNNDescriptor)(cudnnRNNDescriptor_t rnnDesc);         
DEFSO(cudnnSetRNNDescriptor)(cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc, const int hiddenSize, const int numLayers, cudnnDropoutDescriptor_t dropoutDesc, cudnnRNNInputMode_t inputMode, cudnnDirectionMode_t direction, cudnnRNNMode_t mode, cudnnRNNAlgo_t algo, cudnnDataType_t mathPrec);      
DEFSO(cudnnGetRNNDescriptor)(cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc, int *hiddenSize, int *numLayers, cudnnDropoutDescriptor_t *dropoutDesc, cudnnRNNInputMode_t *inputMode, cudnnDirectionMode_t *direction, cudnnRNNMode_t *mode, cudnnRNNAlgo_t *algo, cudnnDataType_t *mathPrec);       
DEFSO(cudnnSetRNNMatrixMathType)(cudnnRNNDescriptor_t rnnDesc, cudnnMathType_t mType);         
DEFSO(cudnnGetRNNMatrixMathType)(cudnnRNNDescriptor_t rnnDesc, cudnnMathType_t *mType);         
DEFSO(cudnnSetRNNBiasMode)(cudnnRNNDescriptor_t rnnDesc, cudnnRNNBiasMode_t biasMode);         
DEFSO(cudnnGetRNNBiasMode)(cudnnRNNDescriptor_t rnnDesc, cudnnRNNBiasMode_t *biasMode);         
DEFSO(cudnnRNNSetClip)(cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc, cudnnRNNClipMode_t clipMode, cudnnNanPropagation_t clipNanOpt, double lclip, double rclip);        
DEFSO(cudnnRNNGetClip)(cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc, cudnnRNNClipMode_t *clipMode, cudnnNanPropagation_t *clipNanOpt, double *lclip, double *rclip);        
DEFSO(cudnnSetRNNProjectionLayers)(cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc, const int recProjSize, const int outProjSize);        
DEFSO(cudnnGetRNNProjectionLayers)(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, int *recProjSize, int *outProjSize);        
DEFSO(cudnnCreatePersistentRNNPlan)(cudnnRNNDescriptor_t rnnDesc, const int minibatch, const cudnnDataType_t dataType, cudnnPersistentRNNPlan_t *plan);        
DEFSO(cudnnDestroyPersistentRNNPlan)(cudnnPersistentRNNPlan_t plan);         
DEFSO(cudnnSetPersistentRNNPlan)(cudnnRNNDescriptor_t rnnDesc, cudnnPersistentRNNPlan_t plan);         
DEFSO(cudnnGetRNNWorkspaceSize)(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const int seqLength, const cudnnTensorDescriptor_t *xDesc, size_t *sizeInBytes);       
DEFSO(cudnnGetRNNTrainingReserveSize)(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const int seqLength, const cudnnTensorDescriptor_t *xDesc, size_t *sizeInBytes);       
DEFSO(cudnnGetRNNParamsSize)(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const cudnnTensorDescriptor_t xDesc, size_t *sizeInBytes, cudnnDataType_t dataType);        
DEFSO(cudnnGetRNNLinLayerMatrixParams)(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const int pseudoLayer, const cudnnTensorDescriptor_t xDesc, const cudnnFilterDescriptor_t wDesc, const void *w, const int linLayerID, cudnnFilterDescriptor_t linLayerMatDesc, void **linLayerMat);      
DEFSO(cudnnGetRNNLinLayerBiasParams)(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const int pseudoLayer, const cudnnTensorDescriptor_t xDesc, const cudnnFilterDescriptor_t wDesc, const void *w, const int linLayerID, cudnnFilterDescriptor_t linLayerBiasDesc, void **linLayerBias);     
DEFSO(cudnnRNNForwardInference)(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const int seqLength, const cudnnTensorDescriptor_t *xDesc, const void *x, const cudnnTensorDescriptor_t hxDesc, const void *hx, const cudnnTensorDescriptor_t cxDesc, const void *cx, const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnTensorDescriptor_t *yDesc, void *y, const cudnnTensorDescriptor_t hyDesc, void *hy, const cudnnTensorDescriptor_t cyDesc, void *cy, void *workspace, size_t workSpaceSizeInBytes); 
DEFSO(cudnnRNNForwardTraining)(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const int seqLength, const cudnnTensorDescriptor_t *xDesc, const void *x, const cudnnTensorDescriptor_t hxDesc, const void *hx, const cudnnTensorDescriptor_t cxDesc, const void *cx, const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnTensorDescriptor_t *yDesc, void *y, const cudnnTensorDescriptor_t hyDesc, void *hy, const cudnnTensorDescriptor_t cyDesc, void *cy, void *workspace, size_t workSpaceSizeInBytes, void *reserveSpace, size_t reserveSpaceSizeInBytes); 
DEFSO(cudnnRNNBackwardData)(cudnnHandle_t handle,const cudnnRNNDescriptor_t rnnDesc,const int seqLength,const cudnnTensorDescriptor_t *yDesc,const void *y,const cudnnTensorDescriptor_t *dyDesc,const void *dy,const cudnnTensorDescriptor_t dhyDesc,const void *dhy,const cudnnTensorDescriptor_t dcyDesc,const void *dcy,const cudnnFilterDescriptor_t wDesc,const void *w,const cudnnTensorDescriptor_t hxDesc,const void *hx,const cudnnTensorDescriptor_t cxDesc,const void *cx,const cudnnTensorDescriptor_t *dxDesc,void *dx,const cudnnTensorDescriptor_t dhxDesc,void *dhx,const cudnnTensorDescriptor_t dcxDesc,void *dcx,void *workspace,size_t workSpaceSizeInBytes,void *reserveSpace,size_t reserveSpaceSizeInBytes);
DEFSO(cudnnRNNBackwardWeights)(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const int seqLength, const cudnnTensorDescriptor_t *xDesc, const void *x, const cudnnTensorDescriptor_t hxDesc, const void *hx, const cudnnTensorDescriptor_t *yDesc, const void *y, const void *workspace, size_t workSpaceSizeInBytes, const cudnnFilterDescriptor_t dwDesc, void *dw, const void *reserveSpace, size_t reserveSpaceSizeInBytes);   
DEFSO(cudnnSetRNNPaddingMode)(cudnnRNNDescriptor_t rnnDesc, cudnnRNNPaddingMode_t paddingMode);         
DEFSO(cudnnGetRNNPaddingMode)(cudnnRNNDescriptor_t rnnDesc, cudnnRNNPaddingMode_t *paddingMode);         
DEFSO(cudnnCreateRNNDataDescriptor)(cudnnRNNDataDescriptor_t *rnnDataDesc);         
DEFSO(cudnnDestroyRNNDataDescriptor)(cudnnRNNDataDescriptor_t rnnDataDesc);         
DEFSO(cudnnSetRNNDataDescriptor)(cudnnRNNDataDescriptor_t rnnDataDesc, cudnnDataType_t dataType, cudnnRNNDataLayout_t layout, int maxSeqLength, int batchSize, int vectorSize, const int seqLengthArray[], void *paddingFill);       
DEFSO(cudnnGetRNNDataDescriptor)(cudnnRNNDataDescriptor_t rnnDataDesc, cudnnDataType_t *dataType, cudnnRNNDataLayout_t *layout, int *maxSeqLength, int *batchSize, int *vectorSize, int arrayLengthRequested, int seqLengthArray[], void *paddingFill);       
DEFSO(cudnnRNNForwardTrainingEx)(cudnnHandle_t handle,const cudnnRNNDescriptor_t rnnDesc,const cudnnRNNDataDescriptor_t xDesc,const void *x,const cudnnTensorDescriptor_t hxDesc,const void *hx,const cudnnTensorDescriptor_t cxDesc,const void *cx,const cudnnFilterDescriptor_t wDesc,const void *w,const cudnnRNNDataDescriptor_t yDesc,void *y,const cudnnTensorDescriptor_t hyDesc,void *hy,const cudnnTensorDescriptor_t cyDesc,void *cy,const cudnnRNNDataDescriptor_t kDesc,const void *keys,const cudnnRNNDataDescriptor_t cDesc, void *cAttn,const cudnnRNNDataDescriptor_t iDesc,void *iAttn,const cudnnRNNDataDescriptor_t qDesc, void *queries,void *workSpace,size_t workSpaceSizeInBytes,void *reserveSpace,size_t reserveSpaceSizeInBytes);
DEFSO(cudnnRNNForwardInferenceEx)(cudnnHandle_t handle,const cudnnRNNDescriptor_t rnnDesc,const cudnnRNNDataDescriptor_t xDesc,const void *x,const cudnnTensorDescriptor_t hxDesc,const void *hx,const cudnnTensorDescriptor_t cxDesc,const void *cx,const cudnnFilterDescriptor_t wDesc,const void *w,const cudnnRNNDataDescriptor_t yDesc,void *y,const cudnnTensorDescriptor_t hyDesc,void *hy,const cudnnTensorDescriptor_t cyDesc,void *cy,const cudnnRNNDataDescriptor_t kDesc,const void *keys,const cudnnRNNDataDescriptor_t cDesc,void *cAttn,const cudnnRNNDataDescriptor_t iDesc,void *iAttn,const cudnnRNNDataDescriptor_t qDesc,void *queries,void *workSpace,size_t workSpaceSizeInBytes);
DEFSO(cudnnRNNBackwardDataEx)(cudnnHandle_t handle,const cudnnRNNDescriptor_t rnnDesc,const cudnnRNNDataDescriptor_t yDesc,const void *y,const cudnnRNNDataDescriptor_t dyDesc,const void *dy,const cudnnRNNDataDescriptor_t dcDesc,const void *dcAttn,const cudnnTensorDescriptor_t dhyDesc,const void *dhy,const cudnnTensorDescriptor_t dcyDesc,const void *dcy,const cudnnFilterDescriptor_t wDesc,const void *w,const cudnnTensorDescriptor_t hxDesc,const void *hx,const cudnnTensorDescriptor_t cxDesc,const void *cx,const cudnnRNNDataDescriptor_t dxDesc,void *dx,const cudnnTensorDescriptor_t dhxDesc,void *dhx,const cudnnTensorDescriptor_t dcxDesc,void *dcx,const cudnnRNNDataDescriptor_t dkDesc,void *dkeys, void *workSpace,size_t workSpaceSizeInBytes,void *reserveSpace,size_t reserveSpaceSizeInBytes);
DEFSO(cudnnRNNBackwardWeightsEx)(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const cudnnRNNDataDescriptor_t xDesc, const void *x, const cudnnTensorDescriptor_t hxDesc, const void *hx, const cudnnRNNDataDescriptor_t yDesc, const void *y, void *workSpace, size_t workSpaceSizeInBytes, const cudnnFilterDescriptor_t dwDesc, void *dw, void *reserveSpace, size_t reserveSpaceSizeInBytes);    
DEFSO(cudnnSetRNNAlgorithmDescriptor)(cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc, cudnnAlgorithmDescriptor_t algoDesc);        
DEFSO(cudnnGetRNNForwardInferenceAlgorithmMaxCount)(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, int *count);        
DEFSO(cudnnFindRNNForwardInferenceAlgorithmEx)(cudnnHandle_t handle,const cudnnRNNDescriptor_t rnnDesc,const int seqLength,const cudnnTensorDescriptor_t *xDesc,const void *x,const cudnnTensorDescriptor_t hxDesc,const void *hx,const cudnnTensorDescriptor_t cxDesc,const void *cx,const cudnnFilterDescriptor_t wDesc,const void *w,const cudnnTensorDescriptor_t *yDesc,void *y,const cudnnTensorDescriptor_t hyDesc,void *hy,const cudnnTensorDescriptor_t cyDesc,void *cy,const float findIntensity,const int requestedAlgoCount,int *returnedAlgoCount,cudnnAlgorithmPerformance_t *perfResults,void *workspace,size_t workSpaceSizeInBytes);
DEFSO(cudnnGetRNNForwardTrainingAlgorithmMaxCount)(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, int *count);        
DEFSO(cudnnFindRNNForwardTrainingAlgorithmEx)(cudnnHandle_t handle,const cudnnRNNDescriptor_t rnnDesc,const int seqLength,const cudnnTensorDescriptor_t *xDesc,const void *x,const cudnnTensorDescriptor_t hxDesc,const void *hx,const cudnnTensorDescriptor_t cxDesc,const void *cx,const cudnnFilterDescriptor_t wDesc,const void *w,const cudnnTensorDescriptor_t *yDesc,void *y,const cudnnTensorDescriptor_t hyDesc,void *hy,const cudnnTensorDescriptor_t cyDesc,void *cy,const float findIntensity,const int requestedAlgoCount,int *returnedAlgoCount,cudnnAlgorithmPerformance_t *perfResults,void *workspace,size_t workSpaceSizeInBytes,void *reserveSpace,size_t reserveSpaceSizeInBytes);
DEFSO(cudnnGetRNNBackwardDataAlgorithmMaxCount)(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, int *count);        
DEFSO(cudnnFindRNNBackwardDataAlgorithmEx)(cudnnHandle_t handle,const cudnnRNNDescriptor_t rnnDesc,const int seqLength,const cudnnTensorDescriptor_t *yDesc,const void *y,const cudnnTensorDescriptor_t *dyDesc,const void *dy,const cudnnTensorDescriptor_t dhyDesc,const void *dhy,const cudnnTensorDescriptor_t dcyDesc,const void *dcy,const cudnnFilterDescriptor_t wDesc,const void *w,const cudnnTensorDescriptor_t hxDesc,const void *hx,const cudnnTensorDescriptor_t cxDesc,const void *cx,const cudnnTensorDescriptor_t *dxDesc,void *dx,const cudnnTensorDescriptor_t dhxDesc,void *dhx,const cudnnTensorDescriptor_t dcxDesc,void *dcx,const float findIntensity,const int requestedAlgoCount,int *returnedAlgoCount,cudnnAlgorithmPerformance_t *perfResults,void *workspace,size_t workSpaceSizeInBytes,void *reserveSpace,size_t reserveSpaceSizeInBytes);
DEFSO(cudnnGetRNNBackwardWeightsAlgorithmMaxCount)(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, int *count);        
DEFSO(cudnnFindRNNBackwardWeightsAlgorithmEx)(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const int seqLength, const cudnnTensorDescriptor_t *xDesc, const void *x, const cudnnTensorDescriptor_t hxDesc, const void *hx, const cudnnTensorDescriptor_t *yDesc, const void *y, const float findIntensity, const int requestedAlgoCount, int *returnedAlgoCount, cudnnAlgorithmPerformance_t *perfResults, const void *workspace, size_t workSpaceSizeInBytes, const cudnnFilterDescriptor_t dwDesc, void *dw, const void *reserveSpace, size_t reserveSpaceSizeInBytes); 
DEFSO(cudnnCreateSeqDataDescriptor)(cudnnSeqDataDescriptor_t *seqDataDesc);         
DEFSO(cudnnDestroySeqDataDescriptor)(cudnnSeqDataDescriptor_t seqDataDesc);         
DEFSO(cudnnSetSeqDataDescriptor)(cudnnSeqDataDescriptor_t seqDataDesc, cudnnDataType_t dataType, int nbDims, const int dimA[], const cudnnSeqDataAxis_t axes[], size_t seqLengthArraySize, const int seqLengthArray[], void *paddingFill);       
DEFSO(cudnnGetSeqDataDescriptor)(const cudnnSeqDataDescriptor_t seqDataDesc, cudnnDataType_t *dataType, int *nbDims, int nbDimsRequested, int dimA[], cudnnSeqDataAxis_t axes[], size_t *seqLengthArraySize, size_t seqLengthSizeRequested, int seqLengthArray[], void *paddingFill);       
DEFSO(cudnnCreateAttnDescriptor)(cudnnAttnDescriptor_t *attnDesc);         
DEFSO(cudnnDestroyAttnDescriptor)(cudnnAttnDescriptor_t attnDesc);         
DEFSO(cudnnSetAttnDescriptor)(cudnnAttnDescriptor_t attnDesc, unsigned attnMode, int nHeads, double smScaler, cudnnDataType_t dataType, cudnnDataType_t computePrec, cudnnMathType_t mathType, cudnnDropoutDescriptor_t attnDropoutDesc, cudnnDropoutDescriptor_t postDropoutDesc, int qSize, int kSize, int vSize, int qProjSize, int kProjSize, int vProjSize, int oProjSize, int qoMaxSeqLength, int kvMaxSeqLength, int maxBatchSize, int maxBeamSize);    
DEFSO(cudnnGetAttnDescriptor)(cudnnAttnDescriptor_t attnDesc, unsigned *attnMode, int *nHeads, double *smScaler, cudnnDataType_t *dataType, cudnnDataType_t *computePrec, cudnnMathType_t *mathType, cudnnDropoutDescriptor_t *attnDropoutDesc, cudnnDropoutDescriptor_t *postDropoutDesc, int *qSize, int *kSize, int *vSize, int *qProjSize, int *kProjSize, int *vProjSize, int *oProjSize, int *qoMaxSeqLength, int *kvMaxSeqLength, int *maxBatchSize, int *maxBeamSize);    
DEFSO(cudnnGetMultiHeadAttnBuffers)(cudnnHandle_t handle, const cudnnAttnDescriptor_t attnDesc, size_t *weightSizeInBytes, size_t *workSpaceSizeInBytes, size_t *reserveSpaceSizeInBytes);        
DEFSO(cudnnGetMultiHeadAttnWeights)(cudnnHandle_t handle, const cudnnAttnDescriptor_t attnDesc, cudnnMultiHeadAttnWeightKind_t wKind, size_t weightSizeInBytes, const void *weights, cudnnTensorDescriptor_t wDesc, void **wAddr);       
DEFSO(cudnnMultiHeadAttnForward)(cudnnHandle_t handle, const cudnnAttnDescriptor_t attnDesc, int currIdx, const int *loWinIdx, const int *hiWinIdx, const int *seqLengthArrayQRO, const int *seqLengthArrayKV, const cudnnSeqDataDescriptor_t qDesc, const void *queries, const void *residuals, const cudnnSeqDataDescriptor_t kDesc, const void *keys, const cudnnSeqDataDescriptor_t vDesc, const void *values, const cudnnSeqDataDescriptor_t oDesc, void *out, size_t weightSizeInBytes, const void *weights, size_t workSpaceSizeInBytes, void *workSpace, size_t reserveSpaceSizeInBytes, void *reserveSpace);
DEFSO(cudnnMultiHeadAttnBackwardData)(cudnnHandle_t handle,const cudnnAttnDescriptor_t attnDesc,const int *loWinIdx,const int *hiWinIdx,const int *seqLengthArrayDQDO,const int *seqLengthArrayDKDV,const cudnnSeqDataDescriptor_t doDesc,const void *dout,const cudnnSeqDataDescriptor_t dqDesc,void *dqueries,const void *queries,const cudnnSeqDataDescriptor_t dkDesc,void *dkeys,const void *keys,const cudnnSeqDataDescriptor_t dvDesc,void *dvalues,const void *values,size_t weightSizeInBytes,const void *weights,size_t workSpaceSizeInBytes,void *workSpace,size_t reserveSpaceSizeInBytes,void *reserveSpace);
DEFSO(cudnnMultiHeadAttnBackwardWeights)(cudnnHandle_t handle, const cudnnAttnDescriptor_t attnDesc, cudnnWgradMode_t addGrad, const cudnnSeqDataDescriptor_t qDesc, const void *queries, const cudnnSeqDataDescriptor_t kDesc, const void *keys, const cudnnSeqDataDescriptor_t vDesc, const void *values, const cudnnSeqDataDescriptor_t doDesc, const void *dout, size_t weightSizeInBytes, const void *weights, void *dweights, size_t workSpaceSizeInBytes, void *workSpace, size_t reserveSpaceSizeInBytes, void *reserveSpace);  
DEFSO(cudnnCreateCTCLossDescriptor)(cudnnCTCLossDescriptor_t *ctcLossDesc);         
DEFSO(cudnnSetCTCLossDescriptor)(cudnnCTCLossDescriptor_t ctcLossDesc, cudnnDataType_t compType);         
DEFSO(cudnnSetCTCLossDescriptorEx)(cudnnCTCLossDescriptor_t ctcLossDesc, cudnnDataType_t compType, cudnnLossNormalizationMode_t normMode, cudnnNanPropagation_t gradMode);        
DEFSO(cudnnGetCTCLossDescriptor)(cudnnCTCLossDescriptor_t ctcLossDesc, cudnnDataType_t *compType);         
DEFSO(cudnnGetCTCLossDescriptorEx)(cudnnCTCLossDescriptor_t ctcLossDesc, cudnnDataType_t *compType, cudnnLossNormalizationMode_t *normMode, cudnnNanPropagation_t *gradMode);        
DEFSO(cudnnCTCLoss)(cudnnHandle_t handle, const cudnnTensorDescriptor_t probsDesc, const void *probs, const int *labels, const int *labelLengths, const int *inputLengths, void *costs, const cudnnTensorDescriptor_t gradientsDesc, const void *gradients, cudnnCTCLossAlgo_t algo, cudnnCTCLossDescriptor_t ctcLossDesc, void *workspace, size_t workSpaceSizeInBytes);   
DEFSO(cudnnGetCTCLossWorkspaceSize)(cudnnHandle_t handle, const cudnnTensorDescriptor_t probsDesc, const cudnnTensorDescriptor_t gradientsDesc, const int *labels, const int *labelLengths, const int *inputLengths, cudnnCTCLossAlgo_t algo, cudnnCTCLossDescriptor_t ctcLossDesc, size_t *sizeInBytes);      
DEFSO(cudnnCreateAlgorithmDescriptor)(cudnnAlgorithmDescriptor_t *algoDesc);         
DEFSO(cudnnSetAlgorithmDescriptor)(cudnnAlgorithmDescriptor_t algoDesc, cudnnAlgorithm_t algorithm);         
DEFSO(cudnnGetAlgorithmDescriptor)(const cudnnAlgorithmDescriptor_t algoDesc, cudnnAlgorithm_t *algorithm);        
DEFSO(cudnnCopyAlgorithmDescriptor)(const cudnnAlgorithmDescriptor_t src, cudnnAlgorithmDescriptor_t dest);        
DEFSO(cudnnDestroyAlgorithmDescriptor)(cudnnAlgorithmDescriptor_t algoDesc);         
DEFSO(cudnnCreateAlgorithmPerformance)(cudnnAlgorithmPerformance_t *algoPerf, int numberToCreate);         
DEFSO(cudnnSetAlgorithmPerformance)(cudnnAlgorithmPerformance_t algoPerf, cudnnAlgorithmDescriptor_t algoDesc, cudnnStatus_t status, float time, size_t memory);        
DEFSO(cudnnGetAlgorithmPerformance)(const cudnnAlgorithmPerformance_t algoPerf, cudnnAlgorithmDescriptor_t *algoDesc, cudnnStatus_t *status, float *time, size_t *memory);        
DEFSO(cudnnDestroyAlgorithmPerformance)(cudnnAlgorithmPerformance_t *algoPerf, int numberToDestroy);         
DEFSO(cudnnGetAlgorithmSpaceSize)(cudnnHandle_t handle, cudnnAlgorithmDescriptor_t algoDesc, size_t *algoSpaceSizeInBytes);        
DEFSO(cudnnSaveAlgorithm)(cudnnHandle_t handle, cudnnAlgorithmDescriptor_t algoDesc, void *algoSpace, size_t algoSpaceSizeInBytes);        
DEFSO(cudnnRestoreAlgorithm)(cudnnHandle_t handle, void *algoSpace, size_t algoSpaceSizeInBytes, cudnnAlgorithmDescriptor_t algoDesc);        
DEFSO(cudnnSetCallback)(unsigned mask, void *udata, cudnnCallback_t fptr);        
DEFSO(cudnnGetCallback)(unsigned *mask, void **udata, cudnnCallback_t *fptr);        
DEFSO(cudnnCreateFusedOpsConstParamPack)(cudnnFusedOpsConstParamPack_t *constPack, cudnnFusedOps_t ops);         
DEFSO(cudnnDestroyFusedOpsConstParamPack)(cudnnFusedOpsConstParamPack_t constPack);         
DEFSO(cudnnSetFusedOpsConstParamPackAttribute)(cudnnFusedOpsConstParamPack_t constPack, cudnnFusedOpsConstParamLabel_t paramLabel, const void *param);        
DEFSO(cudnnGetFusedOpsConstParamPackAttribute)(const cudnnFusedOpsConstParamPack_t constPack, cudnnFusedOpsConstParamLabel_t paramLabel, void *param, int *isNULL);        
DEFSO(cudnnCreateFusedOpsVariantParamPack)(cudnnFusedOpsVariantParamPack_t *varPack, cudnnFusedOps_t ops);         
DEFSO(cudnnDestroyFusedOpsVariantParamPack)(cudnnFusedOpsVariantParamPack_t varPack);         
DEFSO(cudnnSetFusedOpsVariantParamPackAttribute)(cudnnFusedOpsVariantParamPack_t varPack, cudnnFusedOpsVariantParamLabel_t paramLabel, void *ptr);         
DEFSO(cudnnGetFusedOpsVariantParamPackAttribute)(const cudnnFusedOpsVariantParamPack_t varPack, cudnnFusedOpsVariantParamLabel_t paramLabel, void *ptr);        
DEFSO(cudnnCreateFusedOpsPlan)(cudnnFusedOpsPlan_t *plan, cudnnFusedOps_t ops);         
DEFSO(cudnnDestroyFusedOpsPlan)(cudnnFusedOpsPlan_t plan);         
DEFSO(cudnnMakeFusedOpsPlan)(cudnnHandle_t handle, cudnnFusedOpsPlan_t plan, const cudnnFusedOpsConstParamPack_t constPack, size_t *workspaceSizeInBytes);        
DEFSO(cudnnFusedOpsExecute)(cudnnHandle_t handle, const cudnnFusedOpsPlan_t plan, cudnnFusedOpsVariantParamPack_t varPack);        
DEFSO(cudnnSetRNNDescriptor_v6)(cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc, const int hiddenSize, const int numLayers, cudnnDropoutDescriptor_t dropoutDesc, cudnnRNNInputMode_t inputMode, cudnnDirectionMode_t direction, cudnnRNNMode_t mode, cudnnRNNAlgo_t algo, cudnnDataType_t mathPrec);      
DEFSO(cudnnSetRNNDescriptor_v5)(cudnnRNNDescriptor_t rnnDesc, int hiddenSize, int numLayers, cudnnDropoutDescriptor_t dropoutDesc, cudnnRNNInputMode_t inputMode, cudnnDirectionMode_t direction, cudnnRNNMode_t mode, cudnnDataType_t mathPrec);       


static void dlsym_all_funcs() {
    printf("dlsym all funcs\n");
    
    LDSYM(cudnnGetErrorString);
    LDSYM(cudnnGetCudartVersion);
    LDSYM(cudnnGetVersion);
    LDSYM(cudnnDestroyCTCLossDescriptor); 
    LDSYM(cudnnDeriveBNTensorDescriptor);
    LDSYM(cudnnActivationForward);
    LDSYM(cudnnDeriveBNTensorDescriptor);
    LDSYM(cudnnActivationForward);
    LDSYM(cudnnQueryRuntimeError);
    LDSYM(cudnnGetProperty);
    LDSYM(cudnnCreate);
    LDSYM(cudnnDestroy);
    LDSYM(cudnnSetStream);
    LDSYM(cudnnGetStream);
    LDSYM(cudnnCreateTensorDescriptor);
    LDSYM(cudnnSetTensor4dDescriptor);
    LDSYM(cudnnSetTensor4dDescriptorEx);
    LDSYM(cudnnGetTensor4dDescriptor);
    LDSYM(cudnnSetTensorNdDescriptor);
    LDSYM(cudnnSetTensorNdDescriptorEx);
    LDSYM(cudnnGetTensorNdDescriptor);
    LDSYM(cudnnGetTensorSizeInBytes);
    LDSYM(cudnnDestroyTensorDescriptor);
    LDSYM(cudnnInitTransformDest);
    LDSYM(cudnnCreateTensorTransformDescriptor);
    LDSYM(cudnnSetTensorTransformDescriptor);
    LDSYM(cudnnGetTensorTransformDescriptor);
    LDSYM(cudnnDestroyTensorTransformDescriptor);
    LDSYM(cudnnTransformTensor);
    LDSYM(cudnnTransformTensorEx);
    LDSYM(cudnnGetFoldedConvBackwardDataDescriptors);
    LDSYM(cudnnAddTensor);
    LDSYM(cudnnCreateOpTensorDescriptor);
    LDSYM(cudnnSetOpTensorDescriptor);
    LDSYM(cudnnGetOpTensorDescriptor);
    LDSYM(cudnnDestroyOpTensorDescriptor);
    LDSYM(cudnnOpTensor);
    LDSYM(cudnnCreateReduceTensorDescriptor);
    LDSYM(cudnnSetReduceTensorDescriptor);
    LDSYM(cudnnGetReduceTensorDescriptor);
    LDSYM(cudnnDestroyReduceTensorDescriptor);
    LDSYM(cudnnGetReductionIndicesSize);
    LDSYM(cudnnGetReductionWorkspaceSize);
    LDSYM(cudnnReduceTensor);
    LDSYM(cudnnSetTensor);
    LDSYM(cudnnScaleTensor);
    LDSYM(cudnnCreateFilterDescriptor);
    LDSYM(cudnnSetFilter4dDescriptor);
    LDSYM(cudnnGetFilter4dDescriptor);
    LDSYM(cudnnSetFilterNdDescriptor);
    LDSYM(cudnnGetFilterNdDescriptor);
    LDSYM(cudnnGetFilterSizeInBytes);
    LDSYM(cudnnTransformFilter);
    LDSYM(cudnnDestroyFilterDescriptor);
    LDSYM(cudnnReorderFilterAndBias);
    LDSYM(cudnnCreateConvolutionDescriptor);
    LDSYM(cudnnSetConvolutionMathType);
    LDSYM(cudnnGetConvolutionMathType);
    LDSYM(cudnnSetConvolutionGroupCount);
    LDSYM(cudnnGetConvolutionGroupCount);
    LDSYM(cudnnSetConvolutionReorderType);
    LDSYM(cudnnGetConvolutionReorderType);
    LDSYM(cudnnSetConvolution2dDescriptor);
    LDSYM(cudnnGetConvolution2dDescriptor);
    LDSYM(cudnnGetConvolution2dForwardOutputDim);
    LDSYM(cudnnSetConvolutionNdDescriptor);
    LDSYM(cudnnGetConvolutionNdDescriptor);
    LDSYM(cudnnGetConvolutionNdForwardOutputDim);
    LDSYM(cudnnDestroyConvolutionDescriptor);
    LDSYM(cudnnGetConvolutionForwardAlgorithmMaxCount);
    LDSYM(cudnnFindConvolutionForwardAlgorithm);
    LDSYM(cudnnFindConvolutionForwardAlgorithmEx);
    LDSYM(cudnnGetConvolutionForwardAlgorithm);
    LDSYM(cudnnGetConvolutionForwardAlgorithm_v7);
    LDSYM(cudnnGetConvolutionForwardWorkspaceSize);
    LDSYM(cudnnConvolutionForward);
    LDSYM(cudnnConvolutionBiasActivationForward);
    LDSYM(cudnnConvolutionBackwardBias);
    LDSYM(cudnnGetConvolutionBackwardFilterAlgorithmMaxCount);
    LDSYM(cudnnFindConvolutionBackwardFilterAlgorithm);
    LDSYM(cudnnFindConvolutionBackwardFilterAlgorithmEx);
    LDSYM(cudnnGetConvolutionBackwardFilterAlgorithm);
    LDSYM(cudnnGetConvolutionBackwardFilterAlgorithm_v7);
    LDSYM(cudnnGetConvolutionBackwardFilterWorkspaceSize);
    LDSYM(cudnnConvolutionBackwardFilter);
    LDSYM(cudnnGetConvolutionBackwardDataAlgorithmMaxCount);
    LDSYM(cudnnFindConvolutionBackwardDataAlgorithm);
    LDSYM(cudnnFindConvolutionBackwardDataAlgorithmEx);
    LDSYM(cudnnGetConvolutionBackwardDataAlgorithm);
    LDSYM(cudnnGetConvolutionBackwardDataAlgorithm_v7);
    LDSYM(cudnnGetConvolutionBackwardDataWorkspaceSize);
    LDSYM(cudnnConvolutionBackwardData);
    LDSYM(cudnnIm2Col);
    LDSYM(cudnnSoftmaxForward);
    LDSYM(cudnnSoftmaxBackward);
    LDSYM(cudnnCreatePoolingDescriptor);
    LDSYM(cudnnSetPooling2dDescriptor);
    LDSYM(cudnnGetPooling2dDescriptor);
    LDSYM(cudnnSetPoolingNdDescriptor);
    LDSYM(cudnnGetPoolingNdDescriptor);
    LDSYM(cudnnGetPoolingNdForwardOutputDim);
    LDSYM(cudnnGetPooling2dForwardOutputDim);
    LDSYM(cudnnDestroyPoolingDescriptor);
    LDSYM(cudnnPoolingForward);
    LDSYM(cudnnPoolingBackward);
    LDSYM(cudnnCreateActivationDescriptor);
    LDSYM(cudnnSetActivationDescriptor);
    LDSYM(cudnnGetActivationDescriptor);
    LDSYM(cudnnDestroyActivationDescriptor);
    LDSYM(cudnnActivationBackward);
    LDSYM(cudnnCreateLRNDescriptor);
    LDSYM(cudnnSetLRNDescriptor);
    LDSYM(cudnnGetLRNDescriptor);
    LDSYM(cudnnDestroyLRNDescriptor);
    LDSYM(cudnnLRNCrossChannelForward);
    LDSYM(cudnnLRNCrossChannelBackward);
    LDSYM(cudnnDivisiveNormalizationForward);
    LDSYM(cudnnDivisiveNormalizationBackward);
    LDSYM(cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize);
    LDSYM(cudnnGetBatchNormalizationBackwardExWorkspaceSize);
    LDSYM(cudnnGetBatchNormalizationTrainingExReserveSpaceSize);
    LDSYM(cudnnBatchNormalizationForwardTraining);
    LDSYM(cudnnBatchNormalizationForwardTrainingEx);
    LDSYM(cudnnBatchNormalizationForwardInference);
    LDSYM(cudnnBatchNormalizationBackward);
    LDSYM(cudnnBatchNormalizationBackwardEx);
    LDSYM(cudnnCreateSpatialTransformerDescriptor);
    LDSYM(cudnnSetSpatialTransformerNdDescriptor);
    LDSYM(cudnnDestroySpatialTransformerDescriptor);
    LDSYM(cudnnSpatialTfGridGeneratorForward);
    LDSYM(cudnnSpatialTfGridGeneratorBackward);
    LDSYM(cudnnSpatialTfSamplerForward);
    LDSYM(cudnnSpatialTfSamplerBackward);
    LDSYM(cudnnCreateDropoutDescriptor);
    LDSYM(cudnnDestroyDropoutDescriptor);
    LDSYM(cudnnDropoutGetStatesSize);
    LDSYM(cudnnDropoutGetReserveSpaceSize);
    LDSYM(cudnnSetDropoutDescriptor);
    LDSYM(cudnnRestoreDropoutDescriptor);
    LDSYM(cudnnGetDropoutDescriptor);
    LDSYM(cudnnDropoutForward);
    LDSYM(cudnnDropoutBackward);
    LDSYM(cudnnCreateRNNDescriptor);
    LDSYM(cudnnDestroyRNNDescriptor);
    LDSYM(cudnnSetRNNDescriptor);
    LDSYM(cudnnGetRNNDescriptor);
    LDSYM(cudnnSetRNNMatrixMathType);
    LDSYM(cudnnGetRNNMatrixMathType);
    LDSYM(cudnnSetRNNBiasMode);
    LDSYM(cudnnGetRNNBiasMode);
    LDSYM(cudnnRNNSetClip);
    LDSYM(cudnnRNNGetClip);
    LDSYM(cudnnSetRNNProjectionLayers);
    LDSYM(cudnnGetRNNProjectionLayers);
    LDSYM(cudnnCreatePersistentRNNPlan);
    LDSYM(cudnnDestroyPersistentRNNPlan);
    LDSYM(cudnnSetPersistentRNNPlan);
    LDSYM(cudnnGetRNNWorkspaceSize);
    LDSYM(cudnnGetRNNTrainingReserveSize);
    LDSYM(cudnnGetRNNParamsSize);
    LDSYM(cudnnGetRNNLinLayerMatrixParams);
    LDSYM(cudnnGetRNNLinLayerBiasParams);
    LDSYM(cudnnRNNForwardInference);
    LDSYM(cudnnRNNForwardTraining);
    LDSYM(cudnnRNNBackwardData);
    LDSYM(cudnnRNNBackwardWeights);
    LDSYM(cudnnSetRNNPaddingMode);
    LDSYM(cudnnGetRNNPaddingMode);
    LDSYM(cudnnCreateRNNDataDescriptor);
    LDSYM(cudnnDestroyRNNDataDescriptor);
    LDSYM(cudnnSetRNNDataDescriptor);
    LDSYM(cudnnGetRNNDataDescriptor);
    LDSYM(cudnnRNNForwardTrainingEx);
    LDSYM(cudnnRNNForwardInferenceEx);
    LDSYM(cudnnRNNBackwardDataEx);
    LDSYM(cudnnRNNBackwardWeightsEx);
    LDSYM(cudnnSetRNNAlgorithmDescriptor);
    LDSYM(cudnnGetRNNForwardInferenceAlgorithmMaxCount);
    LDSYM(cudnnFindRNNForwardInferenceAlgorithmEx);
    LDSYM(cudnnGetRNNForwardTrainingAlgorithmMaxCount);
    LDSYM(cudnnFindRNNForwardTrainingAlgorithmEx);
    LDSYM(cudnnGetRNNBackwardDataAlgorithmMaxCount);
    LDSYM(cudnnFindRNNBackwardDataAlgorithmEx);
    LDSYM(cudnnGetRNNBackwardWeightsAlgorithmMaxCount);
    LDSYM(cudnnFindRNNBackwardWeightsAlgorithmEx);
    LDSYM(cudnnCreateSeqDataDescriptor);
    LDSYM(cudnnDestroySeqDataDescriptor);
    LDSYM(cudnnSetSeqDataDescriptor);
    LDSYM(cudnnGetSeqDataDescriptor);
    LDSYM(cudnnCreateAttnDescriptor);
    LDSYM(cudnnDestroyAttnDescriptor);
    LDSYM(cudnnSetAttnDescriptor);
    LDSYM(cudnnGetAttnDescriptor);
    LDSYM(cudnnGetMultiHeadAttnBuffers);
    LDSYM(cudnnGetMultiHeadAttnWeights);
    LDSYM(cudnnMultiHeadAttnForward);
    LDSYM(cudnnMultiHeadAttnBackwardData);
    LDSYM(cudnnMultiHeadAttnBackwardWeights);
    LDSYM(cudnnCreateCTCLossDescriptor);
    LDSYM(cudnnSetCTCLossDescriptor);
    LDSYM(cudnnSetCTCLossDescriptorEx);
    LDSYM(cudnnGetCTCLossDescriptor);
    LDSYM(cudnnGetCTCLossDescriptorEx);
    LDSYM(cudnnCTCLoss);
    LDSYM(cudnnGetCTCLossWorkspaceSize);
    LDSYM(cudnnCreateAlgorithmDescriptor);
    LDSYM(cudnnSetAlgorithmDescriptor);
    LDSYM(cudnnGetAlgorithmDescriptor);
    LDSYM(cudnnCopyAlgorithmDescriptor);
    LDSYM(cudnnDestroyAlgorithmDescriptor);
    LDSYM(cudnnCreateAlgorithmPerformance);
    LDSYM(cudnnSetAlgorithmPerformance);
    LDSYM(cudnnGetAlgorithmPerformance);
    LDSYM(cudnnDestroyAlgorithmPerformance);
    LDSYM(cudnnGetAlgorithmSpaceSize);
    LDSYM(cudnnSaveAlgorithm);
    LDSYM(cudnnRestoreAlgorithm);
    LDSYM(cudnnSetCallback);
    LDSYM(cudnnGetCallback);
    LDSYM(cudnnCreateFusedOpsConstParamPack);
    LDSYM(cudnnDestroyFusedOpsConstParamPack);
    LDSYM(cudnnSetFusedOpsConstParamPackAttribute);
    LDSYM(cudnnGetFusedOpsConstParamPackAttribute);
    LDSYM(cudnnCreateFusedOpsVariantParamPack);
    LDSYM(cudnnDestroyFusedOpsVariantParamPack);
    LDSYM(cudnnSetFusedOpsVariantParamPackAttribute);
    LDSYM(cudnnGetFusedOpsVariantParamPackAttribute);
    LDSYM(cudnnCreateFusedOpsPlan);
    LDSYM(cudnnDestroyFusedOpsPlan);
    LDSYM(cudnnMakeFusedOpsPlan);
    LDSYM(cudnnFusedOpsExecute);
    LDSYM(cudnnSetRNNDescriptor_v6);
    LDSYM(cudnnSetRNNDescriptor_v5);

    printf("dnn dlsym all funcs end\n");
}

__attribute ((constructor)) void cudnn_init(void) {
    printf("cudnn_init\n"); 
    so_handle = dlopen (LIB_STRING_DNN, RTLD_NOW);
    if (!so_handle) {
        fprintf (stderr, "%s\n", dlerror());
        exit(1);
    }
    dlsym_all_funcs();
    cudaw_so_register_dli(&so_dli);
    void * pp_for_trace[] = {
        FSWAP(cudnnAddTensor)
        FSWAP(cudnnConvolutionBackwardBias)
        FSWAP(cudnnConvolutionBackwardData)
        FSWAP(cudnnConvolutionBackwardFilter)
        FSWAP(cudnnConvolutionForward)
        FSWAP(cudnnSetStream)
        FSWAP(cudnnGetStream)
    };
    cudawdnn_so_func_swap(pp_for_trace);
}

__attribute ((destructor)) void cudnn_fini(void) {
    printf("cudnn_fini\n");
    if (so_handle) {
        dlclose(so_handle);
    }
    for (int k = 1; k <= so_dli.func_num; ++k) {
        if (so_funcs[k].cnt == 0)
            continue;
        printf("%5d %10lu : %s\n", k, so_funcs[k].cnt, so_funcs[k].func_name);
    }
}

#define checkCudnnErrors(err)  __checkCudnnErrors (err, __FILE__, __LINE__)
static cudnnStatus_t __checkCudnnErrors( cudnnStatus_t err, const char *file, const int line )
{
    if( CUDNN_STATUS_SUCCESS != err) {
        fprintf(stderr, 
        "CUDA dnn API error = %04d from file <%s>, line %i, function \n", 
        err, file, line );
        //exit(-1);
    }
    return err;
}

cudnnStatus_t cudnnDeriveBNTensorDescriptor(cudnnTensorDescriptor_t derivedBnDesc,const cudnnTensorDescriptor_t xDesc,cudnnBatchNormMode_t mode) {
	cudnnStatus_t r;
    begin_func(cudnnDeriveBNTensorDescriptor);
	r = so_cudnnDeriveBNTensorDescriptor(derivedBnDesc,xDesc,mode);
    end_func(cudnnDeriveBNTensorDescriptor);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnActivationForward(cudnnHandle_t handle, cudnnActivationDescriptor_t activationDesc, const void *alpha,const cudnnTensorDescriptor_t xDesc,const void *x,const void *beta,const cudnnTensorDescriptor_t   yDesc,void *y) {
    cudnnStatus_t r
    begin_func(cudnnActivationForward);
    r = so_cudnnActivationForward(handle,activationDesc,alpha,xDesc,x,beta,yDesc,y);
    end_func(cudnnActivationForward);
    checkCudnnErrors(r);
    return r;
}

size_t cudnnGetVersion(void) {
    size_t r;
    begin_func(cudnnGetVersion);
    r = so_cudnnGetVersion();
    end_func(cudnnGetVersion);
    checkCudnnErrors(0);
    return r;
}

size_t cudnnGetCudartVersion(void) {
    size_t r;
    begin_func(cudnnGetCudartVersion);
    r = so_cudnnGetCudartVersion();
    end_func(cudnnGetCudartVersion);
    checkCudnnErrors(0);
    return r;
}

const char *cudnnGetErrorString(cudnnStatus_t status) {
    begin_func(cudnnGetErrorString);
    const char * r = so_cudnnGetErrorString(status);
    end_func(cudnnGetErrorString);
    checkCudnnErrors(0);
    return r;
}

cudnnStatus_t cudnnQueryRuntimeError(cudnnHandle_t handle, cudnnStatus_t *rstatus, cudnnErrQueryMode_t mode, cudnnRuntimeTag_t *tag) {
    cudnnStatus_t r;
    begin_func(cudnnQueryRuntimeError);
    r = so_cudnnQueryRuntimeError(handle, rstatus, mode, tag);
    end_func(cudnnQueryRuntimeError);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnGetProperty(libraryPropertyType type, int *value) {
    cudnnStatus_t r;
    begin_func(cudnnGetProperty);
    r = so_cudnnGetProperty( type, value);
    end_func(cudnnGetProperty);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnCreate(cudnnHandle_t *handle) {
    cudnnStatus_t r;
    begin_func(cudnnCreate);
    r = so_cudnnCreate(handle);
    end_func(cudnnCreate);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnDestroy(cudnnHandle_t handle) {
    cudnnStatus_t r;
    begin_func(cudnnDestroy);
    r = so_cudnnDestroy(handle);
    end_func(cudnnDestroy);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnSetStream(cudnnHandle_t handle, cudaStream_t streamId) {
    cudnnStatus_t r;
    begin_func(cudnnSetStream);
    r = so_cudnnSetStream(handle,streamId);
    end_func(cudnnSetStream);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnGetStream(cudnnHandle_t handle, cudaStream_t *streamId) {
    cudnnStatus_t r;
    begin_func(cudnnGetStream);
    r = so_cudnnGetStream(handle, streamId);
    end_func(cudnnGetStream);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnCreateTensorDescriptor(cudnnTensorDescriptor_t *tensorDesc) {
    cudnnStatus_t r;
    begin_func(cudnnCreateTensorDescriptor);
    r = so_cudnnCreateTensorDescriptor(tensorDesc);
    end_func(cudnnCreateTensorDescriptor);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnSetTensor4dDescriptor(cudnnTensorDescriptor_t tensorDesc,cudnnTensorFormat_t format,cudnnDataType_t dataType, int n,int c,int h,int w) {
    cudnnStatus_t r;
    begin_func(cudnnSetTensor4dDescriptor);
    r = so_cudnnSetTensor4dDescriptor(tensorDesc,format,dataType,n,c,h,w);
    end_func(cudnnSetTensor4dDescriptor);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnSetTensor4dDescriptorEx(cudnnTensorDescriptor_t tensorDesc,cudnnDataType_t dataType, int n, int c,int h,int w,int nStride,int cStride,int hStride,int wStride) {
    cudnnStatus_t r;
    begin_func(cudnnSetTensor4dDescriptorEx);
    r = so_cudnnSetTensor4dDescriptorEx(tensorDesc,dataType, n, c,h,w,nStride,cStride,hStride,wStride);
    end_func(cudnnSetTensor4dDescriptorEx);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnGetTensor4dDescriptor(const cudnnTensorDescriptor_t tensorDesc,cudnnDataType_t *dataType, int *n,int *c,int *h,int *w,int *nStride,int *cStride,int *hStride,int *wStride) {
    cudnnStatus_t r;
    begin_func(cudnnGetTensor4dDescriptor);
    r = so_cudnnGetTensor4dDescriptor(tensorDesc,dataType, n,c,h,w,nStride,cStride,hStride,wStride);
    end_func(cudnnGetTensor4dDescriptor);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnSetTensorNdDescriptor(cudnnTensorDescriptor_t tensorDesc,cudnnDataType_t dataType,int nbDims,const int dimA[],const int strideA[]) {
    cudnnStatus_t r;
    begin_func(cudnnSetTensorNdDescriptor);
    r = so_cudnnSetTensorNdDescriptor(tensorDesc,dataType,nbDims,dimA,strideA);
    end_func(cudnnSetTensorNdDescriptor);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnSetTensorNdDescriptorEx(cudnnTensorDescriptor_t tensorDesc,cudnnTensorFormat_t format,cudnnDataType_t dataType,int nbDims,const int dimA[]) {
    cudnnStatus_t r;
    begin_func(cudnnSetTensorNdDescriptorEx);
    r = so_cudnnSetTensorNdDescriptorEx( tensorDesc, format, dataType,nbDims,dimA);
    end_func(cudnnSetTensorNdDescriptorEx);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnGetTensorNdDescriptor(const cudnnTensorDescriptor_t tensorDesc,int nbDimsRequested,cudnnDataType_t *dataType,int *nbDims,int dimA[],int strideA[]) {
    cudnnStatus_t r;
    begin_func(cudnnGetTensorNdDescriptor);
    r = so_cudnnGetTensorNdDescriptor(  tensorDesc, nbDimsRequested,dataType,nbDims,dimA,strideA);
    end_func(cudnnGetTensorNdDescriptor);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnGetTensorSizeInBytes(const cudnnTensorDescriptor_t tensorDesc, size_t *size) {
    cudnnStatus_t r;
    begin_func(cudnnGetTensorSizeInBytes);
    r = so_cudnnGetTensorSizeInBytes(  tensorDesc, size);
    end_func(cudnnGetTensorSizeInBytes);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnDestroyTensorDescriptor(cudnnTensorDescriptor_t tensorDesc) {
    cudnnStatus_t r;
    begin_func(cudnnDestroyTensorDescriptor);
    r = so_cudnnDestroyTensorDescriptor( tensorDesc);
    end_func(cudnnDestroyTensorDescriptor);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnInitTransformDest(const cudnnTensorTransformDescriptor_t transformDesc,const cudnnTensorDescriptor_t srcDesc,cudnnTensorDescriptor_t destDesc,size_t *destSizeInBytes) {
    cudnnStatus_t r;
    begin_func(cudnnInitTransformDest);
    r = so_cudnnInitTransformDest(  transformDesc,  srcDesc, destDesc,destSizeInBytes);
    end_func(cudnnInitTransformDest);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnCreateTensorTransformDescriptor(cudnnTensorTransformDescriptor_t *transformDesc) {
    cudnnStatus_t r;
    begin_func(cudnnCreateTensorTransformDescriptor);
    r = so_cudnnCreateTensorTransformDescriptor(transformDesc);
    end_func(cudnnCreateTensorTransformDescriptor);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnSetTensorTransformDescriptor(cudnnTensorTransformDescriptor_t transformDesc,const uint32_t nbDims,const cudnnTensorFormat_t destFormat,const int32_t padBeforeA[],const int32_t padAfterA[],const uint32_t foldA[],const cudnnFoldingDirection_t direction) {
    cudnnStatus_t r;
    begin_func(cudnnSetTensorTransformDescriptor);
    r = so_cudnnSetTensorTransformDescriptor( transformDesc,nbDims,  destFormat,  padBeforeA,  padAfterA,foldA,direction);
    end_func(cudnnSetTensorTransformDescriptor);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnGetTensorTransformDescriptor(cudnnTensorTransformDescriptor_t transformDesc,uint32_t nbDimsRequested,cudnnTensorFormat_t *destFormat,int32_t padBeforeA[],int32_t padAfterA[],uint32_t foldA[],cudnnFoldingDirection_t *direction) {
    cudnnStatus_t r;
    begin_func(cudnnGetTensorTransformDescriptor);
    r = so_cudnnGetTensorTransformDescriptor( transformDesc,nbDimsRequested,  destFormat,  padBeforeA,  padAfterA,foldA,direction);
    end_func(cudnnGetTensorTransformDescriptor);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnDestroyTensorTransformDescriptor(cudnnTensorTransformDescriptor_t transformDesc) {
    cudnnStatus_t r;
    begin_func(cudnnDestroyTensorTransformDescriptor);
    r = so_cudnnDestroyTensorTransformDescriptor(transformDesc);
    end_func(cudnnDestroyTensorTransformDescriptor);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnTransformTensor(cudnnHandle_t handle,const void *alpha,const cudnnTensorDescriptor_t xDesc,const void *x,const void *beta,const cudnnTensorDescriptor_t yDesc,void *y) {
    cudnnStatus_t r;
    begin_func(cudnnTransformTensor);
    r = so_cudnnTransformTensor( handle,alpha,  xDesc,x, beta, yDesc,y);
    end_func(cudnnTransformTensor);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnTransformTensorEx(cudnnHandle_t handle,const cudnnTensorTransformDescriptor_t transDesc,const void *alpha,const cudnnTensorDescriptor_t srcDesc,const void *srcData,const void *beta,const cudnnTensorDescriptor_t destDesc,void *destData) {
    cudnnStatus_t r;
    begin_func(cudnnTransformTensorEx);
    r = so_cudnnTransformTensorEx( handle,transDesc,  alpha,srcDesc,srcData, beta, destDesc,destData);
    end_func(cudnnTransformTensorEx);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnGetFoldedConvBackwardDataDescriptors(const cudnnHandle_t handle,const cudnnFilterDescriptor_t filterDesc,const cudnnTensorDescriptor_t diffDesc,const cudnnConvolutionDescriptor_t convDesc,const cudnnTensorDescriptor_t gradDesc,const cudnnTensorFormat_t transformFormat,cudnnFilterDescriptor_t foldedFilterDesc,cudnnTensorDescriptor_t paddedDiffDesc,cudnnConvolutionDescriptor_t foldedConvDesc,cudnnTensorDescriptor_t foldedGradDesc,cudnnTensorTransformDescriptor_t filterFoldTransDesc,cudnnTensorTransformDescriptor_t diffPadTransDesc,cudnnTensorTransformDescriptor_t gradFoldTransDesc,cudnnTensorTransformDescriptor_t gradUnfoldTransDesc) {
    cudnnStatus_t r;
    begin_func(cudnnGetFoldedConvBackwardDataDescriptors);
    r = so_cudnnGetFoldedConvBackwardDataDescriptors(handle,filterDesc,diffDesc,convDesc,gradDesc,transformFormat,foldedFilterDesc,paddedDiffDesc,foldedConvDesc,foldedGradDesc,filterFoldTransDesc,diffPadTransDesc,gradFoldTransDesc,gradUnfoldTransDesc);
    end_func(cudnnGetFoldedConvBackwardDataDescriptors);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnAddTensor(cudnnHandle_t handle,const void *alpha,const cudnnTensorDescriptor_t aDesc,const void *A,const void *beta,const cudnnTensorDescriptor_t cDesc,void *C) {
    cudnnStatus_t r;
    begin_func(cudnnAddTensor);
    r = so_cudnnAddTensor(handle,alpha,aDesc,A,beta,cDesc,C);
    end_func(cudnnAddTensor);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnCreateOpTensorDescriptor(cudnnOpTensorDescriptor_t *opTensorDesc) {
    cudnnStatus_t r;
    begin_func(cudnnCreateOpTensorDescriptor);
    r = so_cudnnCreateOpTensorDescriptor(opTensorDesc);
    end_func(cudnnCreateOpTensorDescriptor);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnSetOpTensorDescriptor(cudnnOpTensorDescriptor_t opTensorDesc,cudnnOpTensorOp_t opTensorOp,cudnnDataType_t opTensorCompType,cudnnNanPropagation_t opTensorNanOpt) {
    cudnnStatus_t r;
    begin_func(cudnnSetOpTensorDescriptor);
    r = so_cudnnSetOpTensorDescriptor(opTensorDesc,opTensorOp,opTensorCompType,opTensorNanOpt);
    end_func(cudnnSetOpTensorDescriptor);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnGetOpTensorDescriptor(const cudnnOpTensorDescriptor_t opTensorDesc,cudnnOpTensorOp_t *opTensorOp,cudnnDataType_t *opTensorCompType,cudnnNanPropagation_t *opTensorNanOpt) {
    cudnnStatus_t r;
    begin_func(cudnnGetOpTensorDescriptor);
    r = so_cudnnGetOpTensorDescriptor(opTensorDesc,opTensorOp,opTensorCompType,opTensorNanOpt);
    end_func(cudnnGetOpTensorDescriptor);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnDestroyOpTensorDescriptor(cudnnOpTensorDescriptor_t opTensorDesc) {
    cudnnStatus_t r;
    begin_func(cudnnDestroyOpTensorDescriptor);
    r = so_cudnnDestroyOpTensorDescriptor(opTensorDesc);
    end_func(cudnnDestroyOpTensorDescriptor);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnOpTensor(cudnnHandle_t handle,const cudnnOpTensorDescriptor_t opTensorDesc,const void *alpha1,const cudnnTensorDescriptor_t aDesc,const void *A,const void *alpha2,const cudnnTensorDescriptor_t bDesc,const void *B,const void *beta,const cudnnTensorDescriptor_t cDesc,void *C) {
    cudnnStatus_t r;
    begin_func(cudnnOpTensor);
    r = so_cudnnOpTensor(handle,opTensorDesc,opTensorDesc,aDesc,A,alpha2,bDesc,B,beta,cDesc,C);
    end_func(cudnnOpTensor);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnCreateReduceTensorDescriptor(cudnnReduceTensorDescriptor_t *reduceTensorDesc) {
    cudnnStatus_t r;
    begin_func(cudnnCreateReduceTensorDescriptor);
    r = so_cudnnCreateReduceTensorDescriptor(reduceTensorDesc);
    end_func(cudnnCreateReduceTensorDescriptor);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnSetReduceTensorDescriptor(cudnnReduceTensorDescriptor_t reduceTensorDesc,cudnnReduceTensorOp_t reduceTensorOp,cudnnDataType_t reduceTensorCompType,cudnnNanPropagation_t reduceTensorNanOpt,cudnnReduceTensorIndices_t reduceTensorIndices,cudnnIndicesType_t reduceTensorIndicesType) {
    cudnnStatus_t r;
    begin_func(cudnnSetReduceTensorDescriptor);
    r = so_cudnnSetReduceTensorDescriptor(reduceTensorDesc,reduceTensorOp,reduceTensorCompType,reduceTensorNanOpt,reduceTensorIndices,reduceTensorIndicesType);
    end_func(cudnnSetReduceTensorDescriptor);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnGetReduceTensorDescriptor(const cudnnReduceTensorDescriptor_t reduceTensorDesc,cudnnReduceTensorOp_t *reduceTensorOp,cudnnDataType_t *reduceTensorCompType,cudnnNanPropagation_t *reduceTensorNanOpt,cudnnReduceTensorIndices_t *reduceTensorIndices,cudnnIndicesType_t *reduceTensorIndicesType) {
    cudnnStatus_t r;
    begin_func(cudnnGetReduceTensorDescriptor);
    r = so_cudnnGetReduceTensorDescriptor(reduceTensorDesc,reduceTensorOp,reduceTensorCompType,reduceTensorNanOpt,reduceTensorIndices,reduceTensorIndicesType);
    end_func(cudnnGetReduceTensorDescriptor);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnDestroyReduceTensorDescriptor(cudnnReduceTensorDescriptor_t reduceTensorDesc) {
    cudnnStatus_t r;
    begin_func(cudnnDestroyReduceTensorDescriptor);
    r = so_cudnnDestroyReduceTensorDescriptor(reduceTensorDesc);
    end_func(cudnnDestroyReduceTensorDescriptor);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnGetReductionIndicesSize(cudnnHandle_t handle,const cudnnReduceTensorDescriptor_t reduceTensorDesc,const cudnnTensorDescriptor_t aDesc,const cudnnTensorDescriptor_t cDesc,size_t *sizeInBytes) {
    cudnnStatus_t r;
    begin_func(cudnnGetReductionIndicesSize);
    r = so_cudnnGetReductionIndicesSize(handle,reduceTensorDesc,aDesc,cDesc,sizeInBytes);
    end_func(cudnnGetReductionIndicesSize);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnGetReductionWorkspaceSize(cudnnHandle_t handle,const cudnnReduceTensorDescriptor_t reduceTensorDesc,const cudnnTensorDescriptor_t aDesc,const cudnnTensorDescriptor_t cDesc,size_t *sizeInBytes) {
    cudnnStatus_t r;
    begin_func(cudnnGetReductionWorkspaceSize);
    r = so_cudnnGetReductionWorkspaceSize(handle,reduceTensorDesc,aDesc,cDesc,sizeInBytes);
    end_func(cudnnGetReductionWorkspaceSize);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnReduceTensor(cudnnHandle_t handle,const cudnnReduceTensorDescriptor_t reduceTensorDesc,void *indices,size_t indicesSizeInBytes,void *workspace,size_t workspaceSizeInBytes,const void *alpha,const cudnnTensorDescriptor_t aDesc,const void *A,const void *beta,const cudnnTensorDescriptor_t cDesc,void *C) {
    cudnnStatus_t r;
    begin_func(cudnnReduceTensor);
    r = so_cudnnReduceTensor(handle,reduceTensorDesc,indices,indicesSizeInBytes,workspace,workspaceSizeInBytes,alpha,aDesc,A,beta,cDesc,C);
    end_func(cudnnReduceTensor);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnSetTensor(cudnnHandle_t handle, const cudnnTensorDescriptor_t yDesc, void *y, const void *valuePtr) {
    cudnnStatus_t r;
    begin_func(cudnnSetTensor);
    r = so_cudnnSetTensor( handle,   yDesc, y, valuePtr);
    end_func(cudnnSetTensor);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnScaleTensor(cudnnHandle_t handle, const cudnnTensorDescriptor_t yDesc, void *y, const void *alpha) {
    cudnnStatus_t r;
    begin_func(cudnnScaleTensor);
    r = so_cudnnScaleTensor( handle,   yDesc, y, alpha);
    end_func(cudnnScaleTensor);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnCreateFilterDescriptor(cudnnFilterDescriptor_t *filterDesc) {
    cudnnStatus_t r;
    begin_func(cudnnCreateFilterDescriptor);
    r = so_cudnnCreateFilterDescriptor(filterDesc);
    end_func(cudnnCreateFilterDescriptor);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnSetFilter4dDescriptor(cudnnFilterDescriptor_t filterDesc,cudnnDataType_t dataType, cudnnTensorFormat_t format,int k, int c,  int h, int w) {
    cudnnStatus_t r;
    begin_func(cudnnSetFilter4dDescriptor);
    r = so_cudnnSetFilter4dDescriptor(filterDesc,dataType,format,k,c,h,w);
    end_func(cudnnSetFilter4dDescriptor);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnGetFilter4dDescriptor(const cudnnFilterDescriptor_t filterDesc,cudnnDataType_t *dataType,cudnnTensorFormat_t *format,int *k, int *c, int *h,int *w) {
    cudnnStatus_t r;
    begin_func(cudnnGetFilter4dDescriptor);
    r = so_cudnnGetFilter4dDescriptor(filterDesc,dataType,format,k,c,h,w);
    end_func(cudnnGetFilter4dDescriptor);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnSetFilterNdDescriptor(cudnnFilterDescriptor_t filterDesc,cudnnDataType_t dataType, cudnnTensorFormat_t format,int nbDims,const int filterDimA[]) {
    cudnnStatus_t r;
    begin_func(cudnnSetFilterNdDescriptor);
    r = so_cudnnSetFilterNdDescriptor(filterDesc,dataType,format,nbDims,filterDimA);
    end_func(cudnnSetFilterNdDescriptor);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnGetFilterNdDescriptor(const cudnnFilterDescriptor_t filterDesc,int nbDimsRequested,cudnnDataType_t *dataType, cudnnTensorFormat_t *format,int *nbDims,int filterDimA[]) {
    cudnnStatus_t r;
    begin_func(cudnnGetFilterNdDescriptor);
    r = so_cudnnGetFilterNdDescriptor(filterDesc,nbDimsRequested,dataType,format,nbDims,filterDimA);
    end_func(cudnnGetFilterNdDescriptor);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnGetFilterSizeInBytes(const cudnnFilterDescriptor_t filterDesc, size_t *size) {
    cudnnStatus_t r;
    begin_func(cudnnGetFilterSizeInBytes);
    r = so_cudnnGetFilterSizeInBytes(filterDesc,size);
    end_func(cudnnGetFilterSizeInBytes);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnTransformFilter(cudnnHandle_t handle,const cudnnTensorTransformDescriptor_t transDesc,const void *alpha,const cudnnFilterDescriptor_t srcDesc,const void *srcData,const void *beta,const cudnnFilterDescriptor_t destDesc,void *destData) {
    cudnnStatus_t r;
    begin_func(cudnnTransformFilter);
    r = so_cudnnTransformFilter(handle,transDesc,alpha,srcDesc,srcData,beta,destDesc,destData);
    end_func(cudnnTransformFilter);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnDestroyFilterDescriptor(cudnnFilterDescriptor_t filterDesc) {
    cudnnStatus_t r;
    begin_func(cudnnDestroyFilterDescriptor);
    r = so_cudnnDestroyFilterDescriptor(filterDesc);
    end_func(cudnnDestroyFilterDescriptor);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnReorderFilterAndBias(cudnnHandle_t handle,const cudnnFilterDescriptor_t filterDesc,cudnnReorderType_t reorderType,const void *filterData,void *reorderedFilterData,int reorderBias,const void *biasData,void *reorderedBiasData) {
    cudnnStatus_t r;
    begin_func(cudnnReorderFilterAndBias);
    r = so_cudnnReorderFilterAndBias(handle,filterDesc,reorderType,filterData,reorderedFilterData,reorderBias,biasData,reorderedBiasData);
    end_func(cudnnReorderFilterAndBias);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnCreateConvolutionDescriptor(cudnnConvolutionDescriptor_t *convDesc) {
    cudnnStatus_t r;
    begin_func(cudnnCreateConvolutionDescriptor);
    r = so_cudnnCreateConvolutionDescriptor(convDesc);
    end_func(cudnnCreateConvolutionDescriptor);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnSetConvolutionMathType(cudnnConvolutionDescriptor_t convDesc, cudnnMathType_t mathType) {
    cudnnStatus_t r;
    begin_func(cudnnSetConvolutionMathType);
    r = so_cudnnSetConvolutionMathType(convDesc,mathType);
    end_func(cudnnSetConvolutionMathType);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnGetConvolutionMathType(cudnnConvolutionDescriptor_t convDesc, cudnnMathType_t *mathType) {
    cudnnStatus_t r;
    begin_func(cudnnGetConvolutionMathType);
    r = so_cudnnGetConvolutionMathType(convDesc,mathType);
    end_func(cudnnGetConvolutionMathType);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnSetConvolutionGroupCount(cudnnConvolutionDescriptor_t convDesc, int groupCount) {
    cudnnStatus_t r;
    begin_func(cudnnSetConvolutionGroupCount);
    r = so_cudnnSetConvolutionGroupCount(convDesc,groupCount);
    end_func(cudnnSetConvolutionGroupCount);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnGetConvolutionGroupCount(cudnnConvolutionDescriptor_t convDesc, int *groupCount) {
    cudnnStatus_t r;
    begin_func(cudnnGetConvolutionGroupCount);
    r = so_cudnnGetConvolutionGroupCount(convDesc,groupCount);
    end_func(cudnnGetConvolutionGroupCount);
    checkCudnnErrors(r);
    return r;

}

cudnnStatus_t cudnnSetConvolutionReorderType(cudnnConvolutionDescriptor_t convDesc, cudnnReorderType_t reorderType) {
    cudnnStatus_t r;
    begin_func(cudnnSetConvolutionReorderType);
    r = so_cudnnSetConvolutionReorderType(convDesc,reorderType);
    end_func(cudnnSetConvolutionReorderType);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnGetConvolutionReorderType(cudnnConvolutionDescriptor_t convDesc, cudnnReorderType_t *reorderType) {
    cudnnStatus_t r;
    begin_func();
    r = so_cudnnGetConvolutionReorderType(convDesc,reorderType);
    end_func();checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnSetConvolution2dDescriptor(cudnnConvolutionDescriptor_t convDesc,int pad_h, int pad_w, int u, int v,          int dilation_h, int dilation_w, cudnnConvolutionMode_t mode,cudnnDataType_t computeType) {
    cudnnStatus_t r;
    begin_func(cudnnSetConvolution2dDescriptor);
    r = so_cudnnSetConvolution2dDescriptor(convDesc,pad_h,pad_w,u,v,dilation_h,dilation_w,mode,computeType);
    end_func(cudnnSetConvolution2dDescriptor);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnGetConvolution2dDescriptor(const cudnnConvolutionDescriptor_t convDesc,int *pad_h,int *pad_w, int *u, int *v,int *dilation_h, int *dilation_w, cudnnConvolutionMode_t *mode,cudnnDataType_t *computeType) {
    cudnnStatus_t r;
    begin_func(cudnnGetConvolution2dDescriptor);
    r = so_cudnnGetConvolution2dDescriptor(convDesc,pad_h,pad_w,u,v,dilation_h,dilation_w,mode,computeType);
    end_func(cudnnGetConvolution2dDescriptor);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnGetConvolution2dForwardOutputDim(const cudnnConvolutionDescriptor_t convDesc,const cudnnTensorDescriptor_t inputTensorDesc,const cudnnFilterDescriptor_t filterDesc,int *n,int *c,int *h,int *w) {
    cudnnStatus_t r;
    begin_func(cudnnGetConvolution2dForwardOutputDim);
    r = so_cudnnGetConvolution2dForwardOutputDim(convDesc,inputTensorDesc,filterDesc,n,c,h,w);
    end_func(cudnnGetConvolution2dForwardOutputDim);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnSetConvolutionNdDescriptor(cudnnConvolutionDescriptor_t convDesc,int arrayLength,const int padA[],const int filterStrideA[],const int dilationA[],cudnnConvolutionMode_t mode,cudnnDataType_t computeType) {
    cudnnStatus_t r;
    begin_func(cudnnSetConvolutionNdDescriptor);
    r = so_cudnnSetConvolutionNdDescriptor(convDesc,arrayLength,padA,filterStrideA,dilationA,mode,computeType);
    end_func(cudnnSetConvolutionNdDescriptor);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnGetConvolutionNdDescriptor(const cudnnConvolutionDescriptor_t convDesc,int arrayLengthRequested,int *arrayLength,int padA[],int strideA[],int dilationA[],cudnnConvolutionMode_t *mode,cudnnDataType_t *computeType) {
    cudnnStatus_t r;
    begin_func(cudnnGetConvolutionNdDescriptor);
    r = so_cudnnGetConvolutionNdDescriptor(convDesc,arrayLengthRequested,arrayLength,padA,strideA,dilationA,mode,computeType);
    end_func(cudnnGetConvolutionNdDescriptor);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnGetConvolutionNdForwardOutputDim(const cudnnConvolutionDescriptor_t convDesc,const cudnnTensorDescriptor_t inputTensorDesc,const cudnnFilterDescriptor_t filterDesc,int nbDims,int tensorOuputDimA[]) {
    cudnnStatus_t r;
    begin_func(cudnnGetConvolutionNdForwardOutputDim);
    r = so_cudnnGetConvolutionNdForwardOutputDim(convDesc,inputTensorDesc,filterDesc,nbDims,tensorOuputDimA);
    end_func(cudnnGetConvolutionNdForwardOutputDim);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnDestroyConvolutionDescriptor(cudnnConvolutionDescriptor_t convDesc) {
    cudnnStatus_t r;
    begin_func(cudnnDestroyConvolutionDescriptor);
    r = so_cudnnDestroyConvolutionDescriptor(convDesc);
    end_func(cudnnDestroyConvolutionDescriptor);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnGetConvolutionForwardAlgorithmMaxCount(cudnnHandle_t handle, int *count) {
    cudnnStatus_t r;
    begin_func(cudnnGetConvolutionForwardAlgorithmMaxCount);
    r = so_cudnnGetConvolutionForwardAlgorithmMaxCount(handle,count);
    end_func(cudnnGetConvolutionForwardAlgorithmMaxCount);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnFindConvolutionForwardAlgorithm(cudnnHandle_t handle,const cudnnTensorDescriptor_t xDesc,const cudnnFilterDescriptor_t wDesc,const cudnnConvolutionDescriptor_t convDesc,const cudnnTensorDescriptor_t yDesc,const int requestedAlgoCount,int *returnedAlgoCount,cudnnConvolutionFwdAlgoPerf_t *perfResults) {
    cudnnStatus_t r;
    begin_func(cudnnFindConvolutionForwardAlgorithm);
    r = so_cudnnFindConvolutionForwardAlgorithm(handle,xDesc,wDesc,convDesc,yDesc,requestedAlgoCount,returnedAlgoCount,perfResults);
    end_func(cudnnFindConvolutionForwardAlgorithm);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnFindConvolutionForwardAlgorithmEx(cudnnHandle_t handle,const cudnnTensorDescriptor_t xDesc,const void *x,const cudnnFilterDescriptor_t wDesc,const void *w,const cudnnConvolutionDescriptor_t convDesc,const cudnnTensorDescriptor_t yDesc,void *y,const int requestedAlgoCount,int *returnedAlgoCount,cudnnConvolutionFwdAlgoPerf_t *perfResults,void *workSpace,size_t workSpaceSizeInBytes) {
    cudnnStatus_t r;
    begin_func(cudnnFindConvolutionForwardAlgorithmEx);
    r = so_cudnnFindConvolutionForwardAlgorithmEx(handle,xDesc,x,wDesc,w,convDesc,yDesc,y,requestedAlgoCount,returnedAlgoCount,perfResults,workSpace,workSpaceSizeInBytes);
    end_func(cudnnFindConvolutionForwardAlgorithmEx);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnGetConvolutionForwardAlgorithm(cudnnHandle_t handle,const cudnnTensorDescriptor_t xDesc,const cudnnFilterDescriptor_t wDesc,const cudnnConvolutionDescriptor_t convDesc,const cudnnTensorDescriptor_t yDesc,cudnnConvolutionFwdPreference_t preference,size_t memoryLimitInBytes,cudnnConvolutionFwdAlgo_t *algo) {
    cudnnStatus_t r;
    begin_func(cudnnGetConvolutionForwardAlgorithm);
    r = so_cudnnGetConvolutionForwardAlgorithm(handle,xDesc,wDesc,convDesc,yDesc,preference,memoryLimitInBytes,algo);
    end_func(cudnnGetConvolutionForwardAlgorithm);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnGetConvolutionForwardAlgorithm_v7(cudnnHandle_t handle,const cudnnTensorDescriptor_t srcDesc,const cudnnFilterDescriptor_t filterDesc,const cudnnConvolutionDescriptor_t convDesc,const cudnnTensorDescriptor_t destDesc,const int requestedAlgoCount,int *returnedAlgoCount,cudnnConvolutionFwdAlgoPerf_t *perfResults) {
    cudnnStatus_t r;
    begin_func(cudnnGetConvolutionForwardAlgorithm_v7);
    r = so_cudnnGetConvolutionForwardAlgorithm_v7(handle,srcDesc,filterDesc,convDesc,destDesc,requestedAlgoCount,returnedAlgoCount,perfResults);
    end_func(cudnnGetConvolutionForwardAlgorithm_v7);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle_t handle,const cudnnTensorDescriptor_t xDesc,const cudnnFilterDescriptor_t wDesc,const cudnnConvolutionDescriptor_t convDesc,const cudnnTensorDescriptor_t yDesc,cudnnConvolutionFwdAlgo_t algo,size_t *sizeInBytes) {
    cudnnStatus_t r;
    begin_func(cudnnGetConvolutionForwardWorkspaceSize);
    r = so_cudnnGetConvolutionForwardWorkspaceSize(handle,xDesc,wDesc,convDesc,yDesc,algo,sizeInBytes);
    end_func(cudnnGetConvolutionForwardWorkspaceSize);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnConvolutionForward(cudnnHandle_t handle,const void *alpha,const cudnnTensorDescriptor_t xDesc,const void *x,const cudnnFilterDescriptor_t wDesc,const void *w,const cudnnConvolutionDescriptor_t convDesc,cudnnConvolutionFwdAlgo_t algo,void *workSpace,size_t workSpaceSizeInBytes,const void *beta,const cudnnTensorDescriptor_t yDesc,void *y) {
    cudnnStatus_t r;
    begin_func(cudnnConvolutionForward);
    r = so_cudnnConvolutionForward(handle,alpha,xDesc,x,wDesc,w,convDesc,algo,workSpace,workSpaceSizeInBytes,beta,yDesc,y);
    end_func(cudnnConvolutionForward);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnConvolutionBiasActivationForward(cudnnHandle_t handle,const void *alpha1,const cudnnTensorDescriptor_t xDesc,const void *x,const cudnnFilterDescriptor_t wDesc,const void *w,const cudnnConvolutionDescriptor_t convDesc,cudnnConvolutionFwdAlgo_t algo,void *workSpace,size_t workSpaceSizeInBytes,const void *alpha2,const cudnnTensorDescriptor_t zDesc,const void *z,const cudnnTensorDescriptor_t biasDesc,const void *bias,const cudnnActivationDescriptor_t activationDesc,const cudnnTensorDescriptor_t yDesc,void *y) {
    cudnnStatus_t r;
    begin_func(cudnnConvolutionBiasActivationForward);
    r = so_cudnnConvolutionBiasActivationForward(handle,alpha1,xDesc,x,wDesc,w,convDesc,algo,workSpace,workSpaceSizeInBytes,alpha2,zDesc,z,biasDesc,bias,activationDesc,yDesc,y);
    end_func(cudnnConvolutionBiasActivationForward);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnConvolutionBackwardBias(cudnnHandle_t handle,const void *alpha,const cudnnTensorDescriptor_t dyDesc,const void *dy,const void *beta,const cudnnTensorDescriptor_t dbDesc,void *db) {
    cudnnStatus_t r;
    begin_func(cudnnConvolutionBackwardBias);
    r = so_cudnnConvolutionBackwardBias(handle,alpha,dyDesc,dy,beta,dbDesc,db);
    end_func(cudnnConvolutionBackwardBias);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(cudnnHandle_t handle, int *count) {
    cudnnStatus_t r;
    begin_func(cudnnGetConvolutionBackwardFilterAlgorithmMaxCount);
    r = so_cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(handle,count);
    end_func(cudnnGetConvolutionBackwardFilterAlgorithmMaxCount);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnFindConvolutionBackwardFilterAlgorithm(cudnnHandle_t handle,const cudnnTensorDescriptor_t xDesc,const cudnnTensorDescriptor_t dyDesc,const cudnnConvolutionDescriptor_t convDesc,const cudnnFilterDescriptor_t dwDesc,const int requestedAlgoCount,int *returnedAlgoCount,cudnnConvolutionBwdFilterAlgoPerf_t *perfResults) {
    cudnnStatus_t r;
    begin_func(cudnnFindConvolutionBackwardFilterAlgorithm);
    r = so_cudnnFindConvolutionBackwardFilterAlgorithm(handle,xDesc,dyDesc,convDesc,dwDesc,requestedAlgoCount,returnedAlgoCount,perfResults);
    end_func(cudnnFindConvolutionBackwardFilterAlgorithm);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnFindConvolutionBackwardFilterAlgorithmEx(cudnnHandle_t handle,const cudnnTensorDescriptor_t xDesc,const void *x,const cudnnTensorDescriptor_t dyDesc,const void *y,const cudnnConvolutionDescriptor_t convDesc,const cudnnFilterDescriptor_t dwDesc,void *dw,const int requestedAlgoCount,int *returnedAlgoCount,cudnnConvolutionBwdFilterAlgoPerf_t *perfResults,void *workSpace,size_t workSpaceSizeInBytes) {
    cudnnStatus_t r;
    begin_func(cudnnFindConvolutionBackwardFilterAlgorithmEx);
    r = so_cudnnFindConvolutionBackwardFilterAlgorithmEx(handle,xDesc,x,dyDesc,y,convDesc,dwDesc,dw,requestedAlgoCount,returnedAlgoCount,perfResults,workSpace,workSpaceSizeInBytes);
    end_func(cudnnFindConvolutionBackwardFilterAlgorithmEx);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnGetConvolutionBackwardFilterAlgorithm(cudnnHandle_t handle,const cudnnTensorDescriptor_t xDesc,const cudnnTensorDescriptor_t dyDesc,const cudnnConvolutionDescriptor_t convDesc,const cudnnFilterDescriptor_t dwDesc,cudnnConvolutionBwdFilterPreference_t preference,size_t memoryLimitInBytes,cudnnConvolutionBwdFilterAlgo_t *algo) {
    cudnnStatus_t r;
    begin_func(cudnnGetConvolutionBackwardFilterAlgorithm);
    r = so_cudnnGetConvolutionBackwardFilterAlgorithm(handle,xDesc,dyDesc,convDesc,dwDesc,preference,memoryLimitInBytes,algo);
    end_func(cudnnGetConvolutionBackwardFilterAlgorithm);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnGetConvolutionBackwardFilterAlgorithm_v7(cudnnHandle_t handle,const cudnnTensorDescriptor_t srcDesc,const cudnnTensorDescriptor_t diffDesc,const cudnnConvolutionDescriptor_t convDesc,const cudnnFilterDescriptor_t gradDesc,const int requestedAlgoCount,int *returnedAlgoCount,cudnnConvolutionBwdFilterAlgoPerf_t *perfResults) {
    cudnnStatus_t r;
    begin_func(cudnnGetConvolutionBackwardFilterAlgorithm_v7);
    r = so_cudnnGetConvolutionBackwardFilterAlgorithm_v7(handle,srcDesc,diffDesc,convDesc,gradDesc,requestedAlgoCount,returnedAlgoCount,perfResults);
    end_func(cudnnGetConvolutionBackwardFilterAlgorithm_v7);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnnHandle_t handle,const cudnnTensorDescriptor_t xDesc,const cudnnTensorDescriptor_t dyDesc,const cudnnConvolutionDescriptor_t convDesc,const cudnnFilterDescriptor_t gradDesc,cudnnConvolutionBwdFilterAlgo_t algo,size_t *sizeInBytes) {
    cudnnStatus_t r;
    begin_func(cudnnGetConvolutionBackwardFilterWorkspaceSize);
    r = so_cudnnGetConvolutionBackwardFilterWorkspaceSize(handle,xDesc,dyDesc,convDesc,gradDesc,algo,sizeInBytes);
    end_func(cudnnGetConvolutionBackwardFilterWorkspaceSize);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnConvolutionBackwardFilter(cudnnHandle_t handle,const void *alpha,const cudnnTensorDescriptor_t xDesc,const void *x,const cudnnTensorDescriptor_t dyDesc,const void *dy,const cudnnConvolutionDescriptor_t convDesc,cudnnConvolutionBwdFilterAlgo_t algo,void *workSpace,size_t workSpaceSizeInBytes,const void *beta,const cudnnFilterDescriptor_t dwDesc,void *dw) {
    cudnnStatus_t r;
    begin_func(cudnnConvolutionBackwardFilter);
    r = so_cudnnConvolutionBackwardFilter(handle,alpha,xDesc,x,dyDesc,dy,convDesc,algo,workSpace,workSpaceSizeInBytes,beta,dwDesc,dw);
    end_func(cudnnConvolutionBackwardFilter);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnGetConvolutionBackwardDataAlgorithmMaxCount(cudnnHandle_t handle, int *count) {
    cudnnStatus_t r;
    begin_func(cudnnGetConvolutionBackwardDataAlgorithmMaxCount);
    r = so_cudnnGetConvolutionBackwardDataAlgorithmMaxCount(handle,count);
    end_func(cudnnGetConvolutionBackwardDataAlgorithmMaxCount);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnFindConvolutionBackwardDataAlgorithm(cudnnHandle_t handle,const cudnnFilterDescriptor_t wDesc,const cudnnTensorDescriptor_t dyDesc,const cudnnConvolutionDescriptor_t convDesc,const cudnnTensorDescriptor_t dxDesc,const int requestedAlgoCount,int *returnedAlgoCount,cudnnConvolutionBwdDataAlgoPerf_t *perfResults) {
    cudnnStatus_t r;
    begin_func(cudnnFindConvolutionBackwardDataAlgorithm);
    r = so_cudnnFindConvolutionBackwardDataAlgorithm(handle,wDesc,dyDesc,convDesc,dxDesc,requestedAlgoCount,returnedAlgoCount,perfResults);
    end_func(cudnnFindConvolutionBackwardDataAlgorithm);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnFindConvolutionBackwardDataAlgorithmEx(cudnnHandle_t handle,const cudnnFilterDescriptor_t wDesc,const void *w,const cudnnTensorDescriptor_t dyDesc,const void *dy,const cudnnConvolutionDescriptor_t convDesc,const cudnnTensorDescriptor_t dxDesc,void *dx,const int requestedAlgoCount,int *returnedAlgoCount,cudnnConvolutionBwdDataAlgoPerf_t *perfResults,void *workSpace,size_t workSpaceSizeInBytes) {
    cudnnStatus_t r;
    begin_func(cudnnFindConvolutionBackwardDataAlgorithmEx);
    r = so_cudnnFindConvolutionBackwardDataAlgorithmEx(handle,wDesc,w,dyDesc,dy,convDesc,dxDesc,dx,requestedAlgoCount,returnedAlgoCount,perfResults,workSpace,workSpaceSizeInBytes);
    end_func(cudnnFindConvolutionBackwardDataAlgorithmEx);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnGetConvolutionBackwardDataAlgorithm(cudnnHandle_t handle,const cudnnFilterDescriptor_t wDesc,const cudnnTensorDescriptor_t dyDesc,const cudnnConvolutionDescriptor_t convDesc,const cudnnTensorDescriptor_t dxDesc,cudnnConvolutionBwdDataPreference_t preference,size_t memoryLimitInBytes,cudnnConvolutionBwdDataAlgo_t *algo) {
    cudnnStatus_t r;
    begin_func(cudnnGetConvolutionBackwardDataAlgorithm);
    r = so_cudnnGetConvolutionBackwardDataAlgorithm(handle,wDesc,dyDesc,convDesc,dxDesc,preference,memoryLimitInBytes,algo);
    end_func(cudnnGetConvolutionBackwardDataAlgorithm);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnGetConvolutionBackwardDataAlgorithm_v7(cudnnHandle_t handle,const cudnnFilterDescriptor_t filterDesc,const cudnnTensorDescriptor_t diffDesc,const cudnnConvolutionDescriptor_t convDesc,const cudnnTensorDescriptor_t gradDesc,const int requestedAlgoCount,int *returnedAlgoCount,cudnnConvolutionBwdDataAlgoPerf_t *perfResults) {
    cudnnStatus_t r;
    begin_func(cudnnGetConvolutionBackwardDataAlgorithm_v7);
    r = so_cudnnGetConvolutionBackwardDataAlgorithm_v7(handle,filterDesc,diffDesc,convDesc,gradDesc,requestedAlgoCount,returnedAlgoCount,perfResults);
    end_func(cudnnGetConvolutionBackwardDataAlgorithm_v7);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnGetConvolutionBackwardDataWorkspaceSize(cudnnHandle_t handle,const cudnnFilterDescriptor_t wDesc,const cudnnTensorDescriptor_t dyDesc,const cudnnConvolutionDescriptor_t convDesc,const cudnnTensorDescriptor_t dxDesc,cudnnConvolutionBwdDataAlgo_t algo,size_t *sizeInBytes) {
    cudnnStatus_t r;
    begin_func(cudnnGetConvolutionBackwardDataWorkspaceSize);
    r = so_cudnnGetConvolutionBackwardDataWorkspaceSize(handle,wDesc,dyDesc,convDesc,dxDesc,algo,sizeInBytes);
    end_func(cudnnGetConvolutionBackwardDataWorkspaceSize);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnConvolutionBackwardData(cudnnHandle_t handle,const void *alpha,const cudnnFilterDescriptor_t wDesc,const void *w,const cudnnTensorDescriptor_t dyDesc,const void *dy,const cudnnConvolutionDescriptor_t convDesc,cudnnConvolutionBwdDataAlgo_t algo,void *workSpace,size_t workSpaceSizeInBytes,const void *beta,const cudnnTensorDescriptor_t dxDesc,void *dx) {
    cudnnStatus_t r;
    begin_func(cudnnConvolutionBackwardData);
    r = so_cudnnConvolutionBackwardData(handle,alpha,wDesc,w,dyDesc,dy,convDesc,algo,workSpace,workSpaceSizeInBytes,beta,dxDesc,dx);
    end_func(cudnnConvolutionBackwardData);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnIm2Col(cudnnHandle_t handle,const cudnnTensorDescriptor_t xDesc,const void *x,const cudnnFilterDescriptor_t wDesc,const cudnnConvolutionDescriptor_t convDesc,void *colBuffer) {    
    cudnnStatus_t r;
    begin_func(cudnnIm2Col);
    r = so_cudnnIm2Col(handle,xDesc,x,wDesc,convDesc,colBuffer);
    end_func(cudnnIm2Col);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnSoftmaxForward(cudnnHandle_t handle,cudnnSoftmaxAlgorithm_t algo,cudnnSoftmaxMode_t mode,const void *alpha,const cudnnTensorDescriptor_t xDesc,const void *x,const void *beta,const cudnnTensorDescriptor_t yDesc,void *y) {
    cudnnStatus_t r;
    begin_func(cudnnSoftmaxForward);
    r = so_cudnnSoftmaxForward(handle,algo,mode,alpha,xDesc,x,beta,yDesc,y);
    end_func(cudnnSoftmaxForward);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnSoftmaxBackward(cudnnHandle_t handle,cudnnSoftmaxAlgorithm_t algo,cudnnSoftmaxMode_t mode,const void *alpha,const cudnnTensorDescriptor_t yDesc,const void *y,const cudnnTensorDescriptor_t dyDesc,const void *dy,const void *beta,const cudnnTensorDescriptor_t dxDesc,void *dx) {
    cudnnStatus_t r;
    begin_func(cudnnSoftmaxBackward);
    r = so_cudnnSoftmaxBackward(handle,algo,mode,alpha,yDesc,y,dyDesc,dy,beta,dxDesc,dx);
    end_func(cudnnSoftmaxBackward);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnCreatePoolingDescriptor(cudnnPoolingDescriptor_t *poolingDesc) {
    cudnnStatus_t r;
    begin_func(cudnnCreatePoolingDescriptor);
    r = so_cudnnCreatePoolingDescriptor(poolingDesc);
    end_func(cudnnCreatePoolingDescriptor);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnSetPooling2dDescriptor(cudnnPoolingDescriptor_t poolingDesc,cudnnPoolingMode_t mode,cudnnNanPropagation_t maxpoolingNanOpt,int windowHeight,int windowWidth,int verticalPadding,int horizontalPadding,int verticalStride,int horizontalStride) {    
    cudnnStatus_t r;
    begin_func(cudnnSetPooling2dDescriptor);
    r = so_cudnnSetPooling2dDescriptor(poolingDesc,mode,maxpoolingNanOpt,windowHeight,windowWidth,verticalPadding,horizontalPadding,verticalStride,horizontalStride);
    end_func(cudnnSetPooling2dDescriptor);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnGetPooling2dDescriptor(const cudnnPoolingDescriptor_t poolingDesc,cudnnPoolingMode_t *mode,cudnnNanPropagation_t *maxpoolingNanOpt,int *windowHeight,int *windowWidth,int *verticalPadding,int *horizontalPadding,int *verticalStride,int *horizontalStride) {
    cudnnStatus_t r;
    begin_func(cudnnGetPooling2dDescriptor);
    r = so_cudnnGetPooling2dDescriptor(poolingDesc,mode,maxpoolingNanOpt,windowHeight,windowWidth,verticalPadding,horizontalPadding,verticalStride,horizontalStride);
    end_func(cudnnGetPooling2dDescriptor);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnSetPoolingNdDescriptor(cudnnPoolingDescriptor_t poolingDesc,const cudnnPoolingMode_t mode,const cudnnNanPropagation_t maxpoolingNanOpt,int nbDims,const int windowDimA[],const int paddingA[],const int strideA[]) {
    cudnnStatus_t r;
    begin_func(cudnnSetPoolingNdDescriptor);
    r = so_cudnnSetPoolingNdDescriptor(poolingDesc,mode,maxpoolingNanOpt,nbDims,windowDimA,paddingA,strideA);
    end_func(cudnnSetPoolingNdDescriptor);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnGetPoolingNdDescriptor(const cudnnPoolingDescriptor_t poolingDesc,int nbDimsRequested,cudnnPoolingMode_t *mode,cudnnNanPropagation_t *maxpoolingNanOpt,int *nbDims,int windowDimA[],int paddingA[],int strideA[]) {
    cudnnStatus_t r;
    begin_func(cudnnGetPoolingNdDescriptor);
    r = so_cudnnGetPoolingNdDescriptor(poolingDesc,nbDimsRequested,mode,maxpoolingNanOpt,nbDims,windowDimA,paddingA,strideA);
    end_func(cudnnGetPoolingNdDescriptor);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnGetPoolingNdForwardOutputDim(const cudnnPoolingDescriptor_t poolingDesc,const cudnnTensorDescriptor_t inputTensorDesc,int nbDims,int outputTensorDimA[]) {
    cudnnStatus_t r;
    begin_func(cudnnGetPoolingNdForwardOutputDim);
    r = so_cudnnGetPoolingNdForwardOutputDim(poolingDesc,inputTensorDesc,nbDims,outputTensorDimA);
    end_func(cudnnGetPoolingNdForwardOutputDim);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnGetPooling2dForwardOutputDim(const cudnnPoolingDescriptor_t poolingDesc,const cudnnTensorDescriptor_t inputTensorDesc,int *n,int *c,int *h,int *w) {
    cudnnStatus_t r;
    begin_func(cudnnGetPooling2dForwardOutputDim);
    r = so_cudnnGetPooling2dForwardOutputDim(poolingDesc,inputTensorDesc,n,c,h,w);
    end_func(cudnnGetPooling2dForwardOutputDim);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnDestroyPoolingDescriptor(cudnnPoolingDescriptor_t poolingDesc) {
    cudnnStatus_t r;
    begin_func(cudnnDestroyPoolingDescriptor);
    r = so_cudnnDestroyPoolingDescriptor(poolingDesc);
    end_func(cudnnDestroyPoolingDescriptor);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnPoolingForward(cudnnHandle_t handle,const cudnnPoolingDescriptor_t poolingDesc,const void *alpha,const cudnnTensorDescriptor_t xDesc,const void *x,const void *beta,const cudnnTensorDescriptor_t yDesc,void *y) {
    cudnnStatus_t r;
    begin_func(cudnnPoolingForward);
    r = so_cudnnPoolingForward(handle,poolingDesc,alpha,xDesc,x,beta,yDesc,y);
    end_func(cudnnPoolingForward);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnPoolingBackward(cudnnHandle_t handle,const cudnnPoolingDescriptor_t poolingDesc,const void *alpha,const cudnnTensorDescriptor_t yDesc,const void *y,const cudnnTensorDescriptor_t dyDesc,const void *dy,const cudnnTensorDescriptor_t xDesc,const void *x,const void *beta,const cudnnTensorDescriptor_t dxDesc,void *dx) {
    cudnnStatus_t r;
    begin_func(cudnnPoolingBackward);
    r = so_cudnnPoolingBackward(handle,poolingDesc,alpha,yDesc,y,dyDesc,dy,xDesc,x,beta,dxDesc,dx);
    end_func(cudnnPoolingBackward);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnCreateActivationDescriptor(cudnnActivationDescriptor_t *activationDesc) {
    cudnnStatus_t r;
    begin_func(cudnnCreateActivationDescriptor);
    r = so_cudnnCreateActivationDescriptor(activationDesc);
    end_func(cudnnCreateActivationDescriptor);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnSetActivationDescriptor(cudnnActivationDescriptor_t activationDesc,cudnnActivationMode_t mode,cudnnNanPropagation_t reluNanOpt,double coef) {
    cudnnStatus_t r;
    begin_func(cudnnSetActivationDescriptor);
    r = so_cudnnSetActivationDescriptor(activationDesc,mode,reluNanOpt,coef);
    end_func(cudnnSetActivationDescriptor);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnGetActivationDescriptor(const cudnnActivationDescriptor_t activationDesc,cudnnActivationMode_t *mode,cudnnNanPropagation_t *reluNanOpt,double *coef) {
    cudnnStatus_t r;
    begin_func(cudnnGetActivationDescriptor);
    r = so_cudnnGetActivationDescriptor(activationDesc,mode,reluNanOpt,coef);
    end_func(cudnnGetActivationDescriptor);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnDestroyActivationDescriptor(cudnnActivationDescriptor_t activationDesc) {
    cudnnStatus_t r;
    begin_func(cudnnDestroyActivationDescriptor);
    r = so_cudnnDestroyActivationDescriptor(activationDesc);
    end_func(cudnnDestroyActivationDescriptor);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnActivationBackward(cudnnHandle_t handle,cudnnActivationDescriptor_t activationDesc,const void *alpha,const cudnnTensorDescriptor_t yDesc,const void *y,const cudnnTensorDescriptor_t dyDesc,const void *dy,const cudnnTensorDescriptor_t xDesc,const void *x,const void *beta,const cudnnTensorDescriptor_t dxDesc,void *dx) {
    cudnnStatus_t r;
    begin_func(cudnnActivationBackward);
    r = so_cudnnActivationBackward(handle,activationDesc,alpha,yDesc,y,dyDesc,dy,xDesc,x,beta,dxDesc,dx);
    end_func(cudnnActivationBackward);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnCreateLRNDescriptor(cudnnLRNDescriptor_t *normDesc) {
    cudnnStatus_t r;
    begin_func(cudnnCreateLRNDescriptor);
    r = so_cudnnCreateLRNDescriptor(normDesc);
    end_func(cudnnCreateLRNDescriptor);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnSetLRNDescriptor(cudnnLRNDescriptor_t normDesc, unsigned lrnN, double lrnAlpha, double lrnBeta, double lrnK) {
    cudnnStatus_t r;
    begin_func(cudnnSetLRNDescriptor);
    r = so_cudnnSetLRNDescriptor(normDesc,lrnN,lrnAlpha,lrnBeta,lrnK);
    end_func(cudnnSetLRNDescriptor);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnGetLRNDescriptor(cudnnLRNDescriptor_t normDesc, unsigned *lrnN, double *lrnAlpha, double *lrnBeta, double *lrnK) {
    cudnnStatus_t r;
    begin_func(cudnnGetLRNDescriptor);
    r = so_cudnnGetLRNDescriptor(normDesc,lrnN,lrnAlpha,lrnBeta,lrnK);
    end_func(cudnnGetLRNDescriptor);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnDestroyLRNDescriptor(cudnnLRNDescriptor_t lrnDesc) {
    cudnnStatus_t r;
    begin_func(cudnnDestroyLRNDescriptor);
    r = so_cudnnDestroyLRNDescriptor(lrnDesc);
    end_func(cudnnDestroyLRNDescriptor);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnLRNCrossChannelForward(cudnnHandle_t handle,cudnnLRNDescriptor_t normDesc,cudnnLRNMode_t lrnMode,const void *alpha,const cudnnTensorDescriptor_t xDesc,const void *x,const void *beta,const cudnnTensorDescriptor_t yDesc,void *y) {
    cudnnStatus_t r;
    begin_func(cudnnLRNCrossChannelForward);
    r = so_cudnnLRNCrossChannelForward(handle,normDesc,lrnMode,alpha,xDesc,x,beta,yDesc,y);
    end_func(cudnnLRNCrossChannelForward);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnLRNCrossChannelBackward(cudnnHandle_t handle,cudnnLRNDescriptor_t normDesc,cudnnLRNMode_t lrnMode,const void *alpha,const cudnnTensorDescriptor_t yDesc,const void *y,const cudnnTensorDescriptor_t dyDesc,const void *dy,const cudnnTensorDescriptor_t xDesc,const void *x,const void *beta,const cudnnTensorDescriptor_t dxDesc,void *dx) {
    cudnnStatus_t r;
    begin_func(cudnnLRNCrossChannelBackward);
    r = so_cudnnLRNCrossChannelBackward(handle,normDesc,lrnMode,alpha,yDesc,y,dyDesc,dy,xDesc,x,beta,dxDesc,dx);
    end_func(cudnnLRNCrossChannelBackward);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnDivisiveNormalizationForward(cudnnHandle_t handle,cudnnLRNDescriptor_t normDesc,cudnnDivNormMode_t mode,const void *alpha,const cudnnTensorDescriptor_t xDesc,const void *x,const void *means, void *temp,void *temp2,const void *beta,const cudnnTensorDescriptor_t yDesc,void *y) {
    cudnnStatus_t r;
    begin_func(cudnnDivisiveNormalizationForward);
    r = so_cudnnDivisiveNormalizationForward(handle,normDesc,mode,alpha,xDesc,x,means,temp,temp2,beta,yDesc,y);
    end_func(cudnnDivisiveNormalizationForward);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnDivisiveNormalizationBackward(cudnnHandle_t handle,cudnnLRNDescriptor_t normDesc,cudnnDivNormMode_t mode,const void *alpha,const cudnnTensorDescriptor_t xDesc, const void *x,const void *means, const void *dy,void *temp,void *temp2,const void *beta,const cudnnTensorDescriptor_t dXdMeansDesc, void *dx,void *dMeans) {
    cudnnStatus_t r;
    begin_func(cudnnDivisiveNormalizationBackward);
    r = so_cudnnDivisiveNormalizationBackward(handle,normDesc,mode,alpha,xDesc,x,means,dy,temp,temp2,beta,dXdMeansDesc,dx,dMeans);
    end_func(cudnnDivisiveNormalizationBackward);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(cudnnHandle_t handle,cudnnBatchNormMode_t mode,cudnnBatchNormOps_t bnOps,const cudnnTensorDescriptor_t xDesc,const cudnnTensorDescriptor_t zDesc,const cudnnTensorDescriptor_t yDesc,const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc,const cudnnActivationDescriptor_t activationDesc,size_t *sizeInBytes) {
    cudnnStatus_t r;
    begin_func(cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize);
    r = so_cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(handle,mode,bnOps,xDesc,zDesc,yDesc,bnScaleBiasMeanVarDesc,activationDesc,sizeInBytes);
    end_func(cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnGetBatchNormalizationBackwardExWorkspaceSize(cudnnHandle_t handle,cudnnBatchNormMode_t mode,cudnnBatchNormOps_t bnOps,const cudnnTensorDescriptor_t xDesc,const cudnnTensorDescriptor_t yDesc,const cudnnTensorDescriptor_t dyDesc,const cudnnTensorDescriptor_t dzDesc,const cudnnTensorDescriptor_t dxDesc,const cudnnTensorDescriptor_t dBnScaleBiasDesc,const cudnnActivationDescriptor_t activationDesc,size_t *sizeInBytes) {
    cudnnStatus_t r;
    begin_func(cudnnGetBatchNormalizationBackwardExWorkspaceSize);
    r = so_cudnnGetBatchNormalizationBackwardExWorkspaceSize(handle,mode,bnOps,xDesc,yDesc,dyDesc,dzDesc,dxDesc,dBnScaleBiasDesc,activationDesc,sizeInBytes);
    end_func(cudnnGetBatchNormalizationBackwardExWorkspaceSize);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnGetBatchNormalizationTrainingExReserveSpaceSize(cudnnHandle_t handle,cudnnBatchNormMode_t mode,cudnnBatchNormOps_t bnOps,const cudnnActivationDescriptor_t activationDesc,const cudnnTensorDescriptor_t xDesc,size_t *sizeInBytes) {
    cudnnStatus_t r;
    begin_func(cudnnGetBatchNormalizationTrainingExReserveSpaceSize);
    r = so_cudnnGetBatchNormalizationTrainingExReserveSpaceSize(handle,mode,bnOps,activationDesc,xDesc,sizeInBytes);
    end_func(cudnnGetBatchNormalizationTrainingExReserveSpaceSize);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnBatchNormalizationForwardTraining(cudnnHandle_t handle,cudnnBatchNormMode_t mode,const void *alpha, const void *beta,const cudnnTensorDescriptor_t xDesc,const void *x, const cudnnTensorDescriptor_t yDesc,void *y, const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc,const void *bnScale,const void *bnBias,double exponentialAverageFactor,void *resultRunningMean,void *resultRunningVariance,double epsilon,void *resultSaveMean,void *resultSaveInvVariance) {
    cudnnStatus_t r;
    begin_func(cudnnBatchNormalizationForwardTraining);
    r = so_cudnnBatchNormalizationForwardTraining(handle,mode,alpha,beta,xDesc,x,yDesc,y,bnScaleBiasMeanVarDesc,bnScale,bnBias,exponentialAverageFactor,resultRunningMean,resultRunningVariance,epsilon,resultSaveMean,resultSaveInvVariance);
    end_func(cudnnBatchNormalizationForwardTraining);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnBatchNormalizationForwardTrainingEx(cudnnHandle_t handle,cudnnBatchNormMode_t mode,cudnnBatchNormOps_t bnOps,const void *alpha,const void *beta,const cudnnTensorDescriptor_t xDesc,const void *xData,const cudnnTensorDescriptor_t zDesc,const void *zData,const cudnnTensorDescriptor_t yDesc,void *yData,const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc,const void *bnScale,const void *bnBias,double exponentialAverageFactor,void *resultRunningMean,void *resultRunningVariance,double epsilon,void *resultSaveMean,void *resultSaveInvVariance,cudnnActivationDescriptor_t activationDesc,void *workspace,size_t workSpaceSizeInBytes,void *reserveSpace,size_t reserveSpaceSizeInBytes) {
    cudnnStatus_t r;
    begin_func(cudnnBatchNormalizationForwardTrainingEx);
    r = so_cudnnBatchNormalizationForwardTrainingEx(handle,mode,bnOps,alpha,beta,xDesc,xData,zDesc,zData,yDesc,yData,bnScaleBiasMeanVarDesc,bnScale,bnBias,exponentialAverageFactor,resultRunningMean,resultRunningVariance,epsilon,resultSaveMean,resultSaveInvVariance,activationDesc,workspace,workSpaceSizeInBytes,reserveSpace,reserveSpaceSizeInBytes);
    end_func(cudnnBatchNormalizationForwardTrainingEx);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnBatchNormalizationForwardInference(cudnnHandle_t handle,cudnnBatchNormMode_t mode,const void *alpha,const void *beta,const cudnnTensorDescriptor_t xDesc,const void *x,const cudnnTensorDescriptor_t yDesc,void *y,const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc,const void *bnScale,const void *bnBias,const void *estimatedMean,const void *estimatedVariance,double epsilon) {
    cudnnStatus_t r;
    begin_func(cudnnBatchNormalizationForwardInference);
    r = so_cudnnBatchNormalizationForwardInference(handle,mode,alpha,beta,xDesc,x,yDesc,y,bnScaleBiasMeanVarDesc,bnScale,bnBias,estimatedMean,estimatedVariance,epsilon);
    end_func(cudnnBatchNormalizationForwardInference);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnBatchNormalizationBackward(cudnnHandle_t handle,cudnnBatchNormMode_t mode,const void *alphaDataDiff,const void *betaDataDiff,const void *alphaParamDiff,const void *betaParamDiff,const cudnnTensorDescriptor_t xDesc, const void *x,const cudnnTensorDescriptor_t dyDesc,const void *dy,const cudnnTensorDescriptor_t dxDesc,void *dx,const cudnnTensorDescriptor_t dBnScaleBiasDesc,const void *bnScale,void *dBnScaleResult,void *dBnBiasResult,double epsilon,const void *savedMean,const void *savedInvVariance) {
    cudnnStatus_t r;
    begin_func(cudnnBatchNormalizationBackward);
    r = so_cudnnBatchNormalizationBackward(handle,mode,alphaDataDiff,betaDataDiff,alphaParamDiff,betaParamDiff,xDesc,x,dyDesc,dy,dxDesc,dx,dBnScaleBiasDesc,bnScale,dBnScaleResult,dBnBiasResult,epsilon,savedMean,savedInvVariance);
    end_func(cudnnBatchNormalizationBackward);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnBatchNormalizationBackwardEx(cudnnHandle_t handle,cudnnBatchNormMode_t mode,cudnnBatchNormOps_t bnOps,const void *alphaDataDiff,const void *betaDataDiff,const void *alphaParamDiff,const void *betaParamDiff,const cudnnTensorDescriptor_t xDesc,const void *xData,const cudnnTensorDescriptor_t yDesc,const void *yData,const cudnnTensorDescriptor_t dyDesc,const void *dyData,const cudnnTensorDescriptor_t dzDesc,void *dzData,const cudnnTensorDescriptor_t dxDesc,void *dxData,const cudnnTensorDescriptor_t dBnScaleBiasDesc,const void *bnScaleData,const void *bnBiasData,void *dBnScaleData,void *dBnBiasData,double epsilon, const void *savedMean,const void *savedInvVariance,cudnnActivationDescriptor_t activationDesc,void *workSpace,size_t workSpaceSizeInBytes,void *reserveSpace,size_t reserveSpaceSizeInBytes) {
    cudnnStatus_t r;
    begin_func(cudnnBatchNormalizationBackwardEx);
    r = so_cudnnBatchNormalizationBackwardEx(handle,mode,bnOps,alphaDataDiff,betaDataDiff,alphaParamDiff,betaParamDiff,xDesc,xData,yDesc,yData,dyDesc,dyData,dzDesc,dzData,dxDesc,dxData,dBnScaleBiasDesc,bnScaleData,bnBiasData,dBnScaleData,dBnBiasData,epsilon,savedMean,savedInvVariance,activationDesc,workSpace,workSpaceSizeInBytes,reserveSpace,reserveSpaceSizeInBytes);
    end_func(cudnnBatchNormalizationBackwardEx);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnCreateSpatialTransformerDescriptor(cudnnSpatialTransformerDescriptor_t *stDesc) {
    cudnnStatus_t r;
    begin_func(cudnnCreateSpatialTransformerDescriptor);
    r = so_cudnnCreateSpatialTransformerDescriptor(stDesc);
    end_func(cudnnCreateSpatialTransformerDescriptor);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnSetSpatialTransformerNdDescriptor(cudnnSpatialTransformerDescriptor_t stDesc,cudnnSamplerType_t samplerType,cudnnDataType_t dataType,const int nbDims,const int dimA[]) {
    cudnnStatus_t r;
    begin_func(cudnnSetSpatialTransformerNdDescriptor);
    r = so_cudnnSetSpatialTransformerNdDescriptor(stDesc,samplerType,dataType,nbDims,dimA);
    end_func(cudnnSetSpatialTransformerNdDescriptor);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnDestroySpatialTransformerDescriptor(cudnnSpatialTransformerDescriptor_t stDesc) {
    cudnnStatus_t r;
    begin_func(cudnnDestroySpatialTransformerDescriptor);
    r = so_cudnnDestroySpatialTransformerDescriptor(stDesc);
    end_func(cudnnDestroySpatialTransformerDescriptor);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnSpatialTfGridGeneratorForward(cudnnHandle_t handle,const cudnnSpatialTransformerDescriptor_t stDesc,const void *theta,void *grid) {
    cudnnStatus_t r;
    begin_func(cudnnSpatialTfGridGeneratorForward);
    r = so_cudnnSpatialTfGridGeneratorForward(handle,stDesc,theta,grid);
    end_func(cudnnSpatialTfGridGeneratorForward);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnSpatialTfGridGeneratorBackward(cudnnHandle_t handle,const cudnnSpatialTransformerDescriptor_t stDesc,const void *dgrid,void *dtheta) {
    cudnnStatus_t r;
    begin_func(cudnnSpatialTfGridGeneratorBackward);
    r = so_cudnnSpatialTfGridGeneratorBackward(handle,stDesc,dgrid,dtheta);
    end_func(cudnnSpatialTfGridGeneratorBackward);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnSpatialTfSamplerForward(cudnnHandle_t handle,cudnnSpatialTransformerDescriptor_t stDesc,const void *alpha,const cudnnTensorDescriptor_t xDesc,const void *x,const void *grid,const void *beta,cudnnTensorDescriptor_t yDesc,void *y) {
    cudnnStatus_t r;
    begin_func(cudnnSpatialTfSamplerForward);
    r = so_cudnnSpatialTfSamplerForward(handle,stDesc,alpha,xDesc,x,grid,beta,yDesc,y);
    end_func(cudnnSpatialTfSamplerForward);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnSpatialTfSamplerBackward(cudnnHandle_t handle,cudnnSpatialTransformerDescriptor_t stDesc,const void *alpha,const cudnnTensorDescriptor_t xDesc,const void *x,const void *beta,const cudnnTensorDescriptor_t dxDesc,void *dx,const void *alphaDgrid,const cudnnTensorDescriptor_t dyDesc,const void *dy,const void *grid,const void *betaDgrid,void *dgrid) {
    cudnnStatus_t r;
    begin_func(cudnnSpatialTfSamplerBackward);
    r = so_cudnnSpatialTfSamplerBackward(handle,stDesc,alpha,xDesc,x,beta,dxDesc,dx,alphaDgrid,dyDesc,dy,grid,betaDgrid,dgrid);
    end_func(cudnnSpatialTfSamplerBackward);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnCreateDropoutDescriptor(cudnnDropoutDescriptor_t *dropoutDesc) {
    cudnnStatus_t r;
    begin_func(cudnnCreateDropoutDescriptor);
    r = so_cudnnCreateDropoutDescriptor(dropoutDesc);
    end_func(cudnnCreateDropoutDescriptor);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnDestroyDropoutDescriptor(cudnnDropoutDescriptor_t dropoutDesc) {
    cudnnStatus_t r;
    begin_func(cudnnDestroyDropoutDescriptor);
    r = so_cudnnDestroyDropoutDescriptor(dropoutDesc);
    end_func(cudnnDestroyDropoutDescriptor);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnDropoutGetStatesSize(cudnnHandle_t handle, size_t *sizeInBytes) {
    cudnnStatus_t r;
    begin_func(cudnnDropoutGetStatesSize);
    r = so_cudnnDropoutGetStatesSize(handle,sizeInBytes);
    end_func(cudnnDropoutGetStatesSize);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnDropoutGetReserveSpaceSize(cudnnTensorDescriptor_t xdesc, size_t *sizeInBytes) {
    cudnnStatus_t r;
    begin_func(cudnnDropoutGetReserveSpaceSize);
    r = so_cudnnDropoutGetReserveSpaceSize(xdesc,sizeInBytes);
    end_func(cudnnDropoutGetReserveSpaceSize);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnSetDropoutDescriptor(cudnnDropoutDescriptor_t dropoutDesc,cudnnHandle_t handle,float dropout,void *states,size_t stateSizeInBytes,unsigned long long seed) {
    cudnnStatus_t r;
    begin_func(cudnnSetDropoutDescriptor);
    r = so_cudnnSetDropoutDescriptor(dropoutDesc,handle,dropout,states,stateSizeInBytes,seed);
    end_func(cudnnSetDropoutDescriptor);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnRestoreDropoutDescriptor(cudnnDropoutDescriptor_t dropoutDesc,cudnnHandle_t handle,float dropout,void *states,size_t stateSizeInBytes,unsigned long long seed) {
    cudnnStatus_t r;
    begin_func(cudnnRestoreDropoutDescriptor);
    r = so_cudnnRestoreDropoutDescriptor(dropoutDesc,handle,dropout,states,stateSizeInBytes,seed);
    end_func(cudnnRestoreDropoutDescriptor);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnGetDropoutDescriptor(cudnnDropoutDescriptor_t dropoutDesc,cudnnHandle_t handle,float *dropout,void **states,unsigned long long *seed) {
    cudnnStatus_t r;
    begin_func(cudnnGetDropoutDescriptor);
    r = so_cudnnGetDropoutDescriptor(dropoutDesc,handle,dropout,states,seed);
    end_func(cudnnGetDropoutDescriptor);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnDropoutForward(cudnnHandle_t handle,const cudnnDropoutDescriptor_t dropoutDesc,const cudnnTensorDescriptor_t xdesc,const void *x,const cudnnTensorDescriptor_t ydesc,void *y,void *reserveSpace,size_t reserveSpaceSizeInBytes) {
    cudnnStatus_t r;
    begin_func(cudnnDropoutForward);
    r = so_cudnnDropoutForward(handle,dropoutDesc,xdesc,x,ydesc,y,reserveSpace,reserveSpaceSizeInBytes);
    end_func(cudnnDropoutForward);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnDropoutBackward(cudnnHandle_t handle,const cudnnDropoutDescriptor_t dropoutDesc,const cudnnTensorDescriptor_t dydesc,const void *dy,const cudnnTensorDescriptor_t dxdesc,void *dx,void *reserveSpace,size_t reserveSpaceSizeInBytes) {
    cudnnStatus_t r;
    begin_func(cudnnDropoutBackward);
    r = so_cudnnDropoutBackward(handle,dropoutDesc,dydesc,dy,dxdesc,dx,reserveSpace,reserveSpaceSizeInBytes);
    end_func(cudnnDropoutBackward);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnCreateRNNDescriptor(cudnnRNNDescriptor_t *rnnDesc) {
    cudnnStatus_t r;
    begin_func(cudnnCreateRNNDescriptor);
    r = so_cudnnCreateRNNDescriptor(rnnDesc);
    end_func(cudnnCreateRNNDescriptor);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnDestroyRNNDescriptor(cudnnRNNDescriptor_t rnnDesc) {
    cudnnStatus_t r;
    begin_func(cudnnDestroyRNNDescriptor);
    r = so_cudnnDestroyRNNDescriptor(rnnDesc);
    end_func(cudnnDestroyRNNDescriptor);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnSetRNNDescriptor(cudnnHandle_t handle,cudnnRNNDescriptor_t rnnDesc,const int hiddenSize,const int numLayers,cudnnDropoutDescriptor_t dropoutDesc,cudnnRNNInputMode_t inputMode,cudnnDirectionMode_t direction,cudnnRNNMode_t mode,cudnnRNNAlgo_t algo,cudnnDataType_t mathPrec) {
    cudnnStatus_t r;
    begin_func(cudnnSetRNNDescriptor);
    r = so_cudnnSetRNNDescriptor(handle,rnnDesc,hiddenSize,numLayers,dropoutDesc,inputMode,direction,mode,algo,mathPrec);
    end_func(cudnnSetRNNDescriptor);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnGetRNNDescriptor(cudnnHandle_t handle,cudnnRNNDescriptor_t rnnDesc,int *hiddenSize,int *numLayers,cudnnDropoutDescriptor_t *dropoutDesc,cudnnRNNInputMode_t *inputMode,cudnnDirectionMode_t *direction,cudnnRNNMode_t *mode,cudnnRNNAlgo_t *algo,cudnnDataType_t *mathPrec) {
    cudnnStatus_t r;
    begin_func(cudnnGetRNNDescriptor);
    r = so_cudnnGetRNNDescriptor(handle,rnnDesc,hiddenSize,numLayers,dropoutDesc,inputMode,direction,mode,algo,mathPrec);
    end_func(cudnnGetRNNDescriptor);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnSetRNNMatrixMathType(cudnnRNNDescriptor_t rnnDesc, cudnnMathType_t mType) {
    cudnnStatus_t r;
    begin_func(cudnnSetRNNMatrixMathType);
    r = so_cudnnSetRNNMatrixMathType(rnnDesc,mType);
    end_func(cudnnSetRNNMatrixMathType);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnGetRNNMatrixMathType(cudnnRNNDescriptor_t rnnDesc, cudnnMathType_t *mType) {
    cudnnStatus_t r;
    begin_func(cudnnGetRNNMatrixMathType);
    r = so_cudnnGetRNNMatrixMathType(rnnDesc,mType);
    end_func(cudnnGetRNNMatrixMathType);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnSetRNNBiasMode(cudnnRNNDescriptor_t rnnDesc, cudnnRNNBiasMode_t biasMode) {
    cudnnStatus_t r;
    begin_func(cudnnSetRNNBiasMode);
    r = so_cudnnSetRNNBiasMode(rnnDesc,biasMode);
    end_func(cudnnSetRNNBiasMode);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnGetRNNBiasMode(cudnnRNNDescriptor_t rnnDesc, cudnnRNNBiasMode_t *biasMode) {
    cudnnStatus_t r;
    begin_func(cudnnGetRNNBiasMode);
    r = so_cudnnGetRNNBiasMode(rnnDesc,biasMode);
    end_func(cudnnGetRNNBiasMode);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnRNNSetClip(cudnnHandle_t handle,cudnnRNNDescriptor_t rnnDesc,cudnnRNNClipMode_t clipMode,cudnnNanPropagation_t clipNanOpt,double lclip,double rclip) {
    cudnnStatus_t r;
    begin_func(cudnnRNNSetClip);
    r = so_cudnnRNNSetClip(handle,rnnDesc,clipMode,clipNanOpt,lclip,rclip);
    end_func(cudnnRNNSetClip);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnRNNGetClip(cudnnHandle_t handle,cudnnRNNDescriptor_t rnnDesc,cudnnRNNClipMode_t *clipMode,cudnnNanPropagation_t *clipNanOpt,double *lclip,double *rclip) {
    cudnnStatus_t r;
    begin_func(cudnnRNNGetClip);
    r = so_cudnnRNNGetClip(handle,rnnDesc,clipMode,clipNanOpt,lclip,rclip);
    end_func(cudnnRNNGetClip);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnSetRNNProjectionLayers(cudnnHandle_t handle,cudnnRNNDescriptor_t rnnDesc,const int recProjSize,const int outProjSize) {
    cudnnStatus_t r;
    begin_func(cudnnSetRNNProjectionLayers);
    r = so_cudnnSetRNNProjectionLayers(handle,rnnDesc,recProjSize,outProjSize);
    end_func(cudnnSetRNNProjectionLayers);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnGetRNNProjectionLayers(cudnnHandle_t handle,const cudnnRNNDescriptor_t rnnDesc,int *recProjSize,int *outProjSize) {
    cudnnStatus_t r;
    begin_func(cudnnGetRNNProjectionLayers);
    r = so_cudnnGetRNNProjectionLayers(handle,rnnDesc,recProjSize,outProjSize);
    end_func(cudnnGetRNNProjectionLayers);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnCreatePersistentRNNPlan(cudnnRNNDescriptor_t rnnDesc,const int minibatch,const cudnnDataType_t dataType,cudnnPersistentRNNPlan_t *plan) {
    cudnnStatus_t r;
    begin_func(cudnnCreatePersistentRNNPlan);
    r = so_cudnnCreatePersistentRNNPlan(rnnDesc,minibatch,dataType,plan);
    end_func(cudnnCreatePersistentRNNPlan);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnDestroyPersistentRNNPlan(cudnnPersistentRNNPlan_t plan) {
    cudnnStatus_t r;
    begin_func(cudnnDestroyPersistentRNNPlan);
    r = so_cudnnDestroyPersistentRNNPlan(plan);
    end_func(cudnnDestroyPersistentRNNPlan);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnSetPersistentRNNPlan(cudnnRNNDescriptor_t rnnDesc, cudnnPersistentRNNPlan_t plan) {
    cudnnStatus_t r;
    begin_func(cudnnSetPersistentRNNPlan);
    r = so_cudnnSetPersistentRNNPlan(rnnDesc,plan);
    end_func(cudnnSetPersistentRNNPlan);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnGetRNNWorkspaceSize(cudnnHandle_t handle,const cudnnRNNDescriptor_t rnnDesc,const int seqLength,const cudnnTensorDescriptor_t *xDesc,size_t *sizeInBytes) {
    cudnnStatus_t r;
    begin_func(cudnnGetRNNWorkspaceSize);
    r = so_cudnnGetRNNWorkspaceSize(handle,rnnDesc,seqLength,xDesc,sizeInBytes);
    end_func(cudnnGetRNNWorkspaceSize);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnGetRNNTrainingReserveSize(cudnnHandle_t handle,const cudnnRNNDescriptor_t rnnDesc,const int seqLength,const cudnnTensorDescriptor_t *xDesc,size_t *sizeInBytes) {
    cudnnStatus_t r;
    begin_func(cudnnGetRNNTrainingReserveSize);
    r = so_cudnnGetRNNTrainingReserveSize(handle,rnnDesc,seqLength,xDesc,sizeInBytes);
    end_func(cudnnGetRNNTrainingReserveSize);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnGetRNNParamsSize(cudnnHandle_t handle,const cudnnRNNDescriptor_t rnnDesc,const cudnnTensorDescriptor_t xDesc,size_t *sizeInBytes,cudnnDataType_t dataType) {
    cudnnStatus_t r;
    begin_func(cudnnGetRNNParamsSize);
    r = so_cudnnGetRNNParamsSize(handle,rnnDesc,xDesc,sizeInBytes,dataType);
    end_func(cudnnGetRNNParamsSize);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnGetRNNLinLayerMatrixParams(cudnnHandle_t handle,const cudnnRNNDescriptor_t rnnDesc,const int pseudoLayer,const cudnnTensorDescriptor_t xDesc,const cudnnFilterDescriptor_t wDesc,const void *w,const int linLayerID,cudnnFilterDescriptor_t linLayerMatDesc,void **linLayerMat) {
    cudnnStatus_t r;
    begin_func(cudnnGetRNNLinLayerMatrixParams);
    r = so_cudnnGetRNNLinLayerMatrixParams(handle,rnnDesc,pseudoLayer,xDesc,wDesc,w,linLayerID,linLayerMatDesc,linLayerMat);
    end_func(cudnnGetRNNLinLayerMatrixParams);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnGetRNNLinLayerBiasParams(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,const int pseudoLayer,const cudnnTensorDescriptor_t xDesc,const cudnnFilterDescriptor_t wDesc,const void *w,const int linLayerID,cudnnFilterDescriptor_t linLayerBiasDesc,void **linLayerBias) {
    cudnnStatus_t r;
    begin_func(cudnnGetRNNLinLayerBiasParams);
    r = so_cudnnGetRNNLinLayerBiasParams(handle,rnnDesc,pseudoLayer,xDesc,wDesc,w,linLayerID,linLayerBiasDesc,linLayerBias);
    end_func(cudnnGetRNNLinLayerBiasParams);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnRNNForwardInference(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,const int seqLength,const cudnnTensorDescriptor_t *xDesc,const void *x,const cudnnTensorDescriptor_t hxDesc,const void *hx,const cudnnTensorDescriptor_t cxDesc,const void *cx,const cudnnFilterDescriptor_t wDesc,const void *w,const cudnnTensorDescriptor_t *yDesc,void *y,const cudnnTensorDescriptor_t hyDesc,void *hy,const cudnnTensorDescriptor_t cyDesc,void *cy,void *workspace,size_t workSpaceSizeInBytes) {
    cudnnStatus_t r;
    begin_func(cudnnRNNForwardInference);
    r = so_cudnnRNNForwardInference(handle,rnnDesc,seqLength,xDesc,x,hxDesc,hx,cxDesc,cx,wDesc,w,yDesc,y,hyDesc,hy,cyDesc,cy,workspace,workSpaceSizeInBytes);
    end_func(cudnnRNNForwardInference);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnRNNForwardTraining(cudnnHandle_t handle,const cudnnRNNDescriptor_t rnnDesc,const int seqLength,const cudnnTensorDescriptor_t *xDesc,const void *x,const cudnnTensorDescriptor_t hxDesc,const void *hx,const cudnnTensorDescriptor_t cxDesc,const void *cx,const cudnnFilterDescriptor_t wDesc,const void *w,const cudnnTensorDescriptor_t *yDesc,void *y,const cudnnTensorDescriptor_t hyDesc,void *hy,const cudnnTensorDescriptor_t cyDesc,void *cy,void *workspace,size_t workSpaceSizeInBytes,void *reserveSpace,size_t reserveSpaceSizeInBytes) {
    cudnnStatus_t r;
    begin_func(cudnnRNNForwardTraining);
    r = so_cudnnRNNForwardTraining(handle,rnnDesc,seqLength,xDesc,x,hxDesc,hx,cxDesc,cx,wDesc,w,yDesc,y,hyDesc,hy,cyDesc,cy,workspace,workSpaceSizeInBytes,reserveSpace,reserveSpaceSizeInBytes);
    end_func(cudnnRNNForwardTraining);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnRNNBackwardData(cudnnHandle_t handle,const cudnnRNNDescriptor_t rnnDesc,const int seqLength,const cudnnTensorDescriptor_t *yDesc,const void *y,const cudnnTensorDescriptor_t *dyDesc,const void *dy,const cudnnTensorDescriptor_t dhyDesc,const void *dhy,const cudnnTensorDescriptor_t dcyDesc,const void *dcy,const cudnnFilterDescriptor_t wDesc,const void *w,const cudnnTensorDescriptor_t hxDesc,const void *hx,const cudnnTensorDescriptor_t cxDesc,const void *cx,const cudnnTensorDescriptor_t *dxDesc,void *dx,const cudnnTensorDescriptor_t dhxDesc,void *dhx,const cudnnTensorDescriptor_t dcxDesc,void *dcx,void *workspace,size_t workSpaceSizeInBytes,void *reserveSpace,size_t reserveSpaceSizeInBytes) {
    cudnnStatus_t r;
    begin_func(cudnnRNNBackwardData);
    r = so_cudnnRNNBackwardData(handle,rnnDesc,seqLength,yDesc,y,dyDesc,dy,dhyDesc,dhy,dcyDesc,dcy,wDesc,w,hxDesc,hx,cxDesc,cx,dxDesc,dx,dhxDesc,dhx,dcxDesc,dcx,workspace,workSpaceSizeInBytes,reserveSpace,reserveSpaceSizeInBytes);
    end_func(cudnnRNNBackwardData);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnRNNBackwardWeights(cudnnHandle_t handle,const cudnnRNNDescriptor_t rnnDesc,const int seqLength,const cudnnTensorDescriptor_t *xDesc,const void *x,const cudnnTensorDescriptor_t hxDesc,const void *hx,const cudnnTensorDescriptor_t *yDesc,const void *y,const void *workspace,size_t workSpaceSizeInBytes,const cudnnFilterDescriptor_t dwDesc,void *dw,const void *reserveSpace,size_t reserveSpaceSizeInBytes) {
    cudnnStatus_t r;
    begin_func(cudnnRNNBackwardWeights);
    r = so_cudnnRNNBackwardWeights(handle,rnnDesc,seqLength,xDesc,x,hxDesc,hx,yDesc,y,workspace,workSpaceSizeInBytes,dwDesc,dw,reserveSpace,reserveSpaceSizeInBytes);
    end_func(cudnnRNNBackwardWeights);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnSetRNNPaddingMode(cudnnRNNDescriptor_t rnnDesc, cudnnRNNPaddingMode_t paddingMode) {
    cudnnStatus_t r;
    begin_func(cudnnSetRNNPaddingMode);
    r = so_cudnnSetRNNPaddingMode(rnnDesc,paddingMode);
    end_func(cudnnSetRNNPaddingMode);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnGetRNNPaddingMode(cudnnRNNDescriptor_t rnnDesc, cudnnRNNPaddingMode_t *paddingMode) {
    cudnnStatus_t r;
    begin_func(cudnnGetRNNPaddingMode);
    r = so_cudnnGetRNNPaddingMode(rnnDesc,paddingMode);
    end_func(cudnnGetRNNPaddingMode);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnCreateRNNDataDescriptor(cudnnRNNDataDescriptor_t *rnnDataDesc) {
    cudnnStatus_t r;
    begin_func(cudnnCreateRNNDataDescriptor);
    r = so_cudnnCreateRNNDataDescriptor(rnnDataDesc);
    end_func(cudnnCreateRNNDataDescriptor);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnDestroyRNNDataDescriptor(cudnnRNNDataDescriptor_t rnnDataDesc) {
    cudnnStatus_t r;
    begin_func(cudnnDestroyRNNDataDescriptor);
    r = so_cudnnDestroyRNNDataDescriptor(rnnDataDesc);
    end_func(cudnnDestroyRNNDataDescriptor);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnSetRNNDataDescriptor(cudnnRNNDataDescriptor_t rnnDataDesc,cudnnDataType_t dataType,cudnnRNNDataLayout_t layout,int maxSeqLength,int batchSize,int vectorSize,const int seqLengthArray[], void *paddingFill) {
    cudnnStatus_t r;
    begin_func(cudnnSetRNNDataDescriptor);
    r = so_cudnnSetRNNDataDescriptor(rnnDataDesc,dataType,layout,maxSeqLength,batchSize,vectorSize,seqLengthArray,paddingFill);
    end_func(cudnnSetRNNDataDescriptor);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnGetRNNDataDescriptor(cudnnRNNDataDescriptor_t rnnDataDesc,cudnnDataType_t *dataType,cudnnRNNDataLayout_t *layout,int *maxSeqLength,int *batchSize,int *vectorSize,int arrayLengthRequested,int seqLengthArray[],void *paddingFill) {
    cudnnStatus_t r;
    begin_func(cudnnGetRNNDataDescriptor);
    r = so_cudnnGetRNNDataDescriptor(rnnDataDesc,dataType,layout,maxSeqLength,batchSize,vectorSize,arrayLengthRequested,seqLengthArray,paddingFill);
    end_func(cudnnGetRNNDataDescriptor);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnRNNForwardTrainingEx(cudnnHandle_t handle,const cudnnRNNDescriptor_t rnnDesc,const cudnnRNNDataDescriptor_t xDesc,const void *x,const cudnnTensorDescriptor_t hxDesc,const void *hx,const cudnnTensorDescriptor_t cxDesc,const void *cx,const cudnnFilterDescriptor_t wDesc,const void *w,const cudnnRNNDataDescriptor_t yDesc,void *y,const cudnnTensorDescriptor_t hyDesc,void *hy,const cudnnTensorDescriptor_t cyDesc,void *cy,const cudnnRNNDataDescriptor_t kDesc,const void *keys,const cudnnRNNDataDescriptor_t cDesc, void *cAttn,const cudnnRNNDataDescriptor_t iDesc,void *iAttn,const cudnnRNNDataDescriptor_t qDesc, void *queries,void *workSpace,size_t workSpaceSizeInBytes,void *reserveSpace,size_t reserveSpaceSizeInBytes) {
    cudnnStatus_t r;
    begin_func(cudnnRNNForwardTrainingEx);
    r = so_cudnnRNNForwardTrainingEx(handle,rnnDesc,xDesc,x,hxDesc,hx,cxDesc,cx,wDesc,w,yDesc,y,hyDesc,hy,cyDesc,cy,kDesc,keys,cDesc,cAttn,iDesc,iAttn,qDesc,queries,workSpace,workSpaceSizeInBytes,reserveSpace,reserveSpaceSizeInBytes);
    end_func(cudnnRNNForwardTrainingEx);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnRNNForwardInferenceEx(cudnnHandle_t handle,const cudnnRNNDescriptor_t rnnDesc,const cudnnRNNDataDescriptor_t xDesc,const void *x,const cudnnTensorDescriptor_t hxDesc,const void *hx,const cudnnTensorDescriptor_t cxDesc,const void *cx,const cudnnFilterDescriptor_t wDesc,const void *w,const cudnnRNNDataDescriptor_t yDesc,void *y,const cudnnTensorDescriptor_t hyDesc,void *hy,const cudnnTensorDescriptor_t cyDesc,void *cy,const cudnnRNNDataDescriptor_t kDesc,const void *keys,const cudnnRNNDataDescriptor_t cDesc,void *cAttn,const cudnnRNNDataDescriptor_t iDesc,void *iAttn,const cudnnRNNDataDescriptor_t qDesc,void *queries,void *workSpace,size_t workSpaceSizeInBytes) {
    cudnnStatus_t r;
    begin_func(cudnnRNNForwardInferenceEx);
    r = so_cudnnRNNForwardInferenceEx(handle,rnnDesc,xDesc,x,hxDesc,hx,cxDesc,cx,wDesc,w,yDesc,y,hyDesc,hy,cyDesc,cy,kDesc,keys,cDesc,cAttn,iDesc,iAttn,qDesc,queries,workSpace,workSpaceSizeInBytes);
    end_func(cudnnRNNForwardInferenceEx);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnRNNBackwardDataEx(cudnnHandle_t handle,const cudnnRNNDescriptor_t rnnDesc,const cudnnRNNDataDescriptor_t yDesc,const void *y,const cudnnRNNDataDescriptor_t dyDesc,const void *dy,const cudnnRNNDataDescriptor_t dcDesc,const void *dcAttn,const cudnnTensorDescriptor_t dhyDesc,const void *dhy,const cudnnTensorDescriptor_t dcyDesc,const void *dcy,const cudnnFilterDescriptor_t wDesc,const void *w,const cudnnTensorDescriptor_t hxDesc,const void *hx,const cudnnTensorDescriptor_t cxDesc,const void *cx,const cudnnRNNDataDescriptor_t dxDesc,void *dx,const cudnnTensorDescriptor_t dhxDesc,void *dhx,const cudnnTensorDescriptor_t dcxDesc,void *dcx,const cudnnRNNDataDescriptor_t dkDesc,void *dkeys, void *workSpace,size_t workSpaceSizeInBytes,void *reserveSpace,size_t reserveSpaceSizeInBytes) {
    cudnnStatus_t r;
    begin_func(cudnnRNNBackwardDataEx);
    r = so_cudnnRNNBackwardDataEx(handle,rnnDesc,yDesc,y,dyDesc,dy,dcDesc,dcAttn,dhyDesc,dhy,dcyDesc,dcy,wDesc,w,hxDesc,hx,cxDesc,cx,dxDesc,dx,dhxDesc,dhx,dcxDesc,dcx,dkDesc,dkeys,workSpace,workSpaceSizeInBytes,reserveSpace,reserveSpaceSizeInBytes);
    end_func(cudnnRNNBackwardDataEx);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnRNNBackwardWeightsEx(cudnnHandle_t handle,const cudnnRNNDescriptor_t rnnDesc,const cudnnRNNDataDescriptor_t xDesc,const void *x,const cudnnTensorDescriptor_t hxDesc,const void *hx,const cudnnRNNDataDescriptor_t yDesc,const void *y,void *workSpace,size_t workSpaceSizeInBytes,const cudnnFilterDescriptor_t dwDesc,void *dw,void *reserveSpace,size_t reserveSpaceSizeInBytes) {
    cudnnStatus_t r;
    begin_func(cudnnRNNBackwardWeightsEx);
    r = so_cudnnRNNBackwardWeightsEx(handle,rnnDesc,xDesc,x,hxDesc,hx,yDesc,y,workSpace,workSpaceSizeInBytes,dwDesc,dw,reserveSpace,reserveSpaceSizeInBytes);
    end_func(cudnnRNNBackwardWeightsEx);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnSetRNNAlgorithmDescriptor(cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc, cudnnAlgorithmDescriptor_t algoDesc) {
    cudnnStatus_t r;
    begin_func(cudnnSetRNNAlgorithmDescriptor);
    r = so_cudnnSetRNNAlgorithmDescriptor(handle,rnnDesc,algoDesc);
    end_func(cudnnSetRNNAlgorithmDescriptor);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnGetRNNForwardInferenceAlgorithmMaxCount(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, int *count) {
    cudnnStatus_t r;
    begin_func(cudnnGetRNNForwardInferenceAlgorithmMaxCount);
    r = so_cudnnGetRNNForwardInferenceAlgorithmMaxCount(handle,rnnDesc,count);
    end_func(cudnnGetRNNForwardInferenceAlgorithmMaxCount);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnFindRNNForwardInferenceAlgorithmEx(cudnnHandle_t handle,const cudnnRNNDescriptor_t rnnDesc,const int seqLength,const cudnnTensorDescriptor_t *xDesc,const void *x,const cudnnTensorDescriptor_t hxDesc,const void *hx,const cudnnTensorDescriptor_t cxDesc,const void *cx,const cudnnFilterDescriptor_t wDesc,const void *w,const cudnnTensorDescriptor_t *yDesc,void *y,const cudnnTensorDescriptor_t hyDesc,void *hy,const cudnnTensorDescriptor_t cyDesc,void *cy,const float findIntensity,const int requestedAlgoCount,int *returnedAlgoCount,cudnnAlgorithmPerformance_t *perfResults,void *workspace,size_t workSpaceSizeInBytes) {
    cudnnStatus_t r;
    begin_func(cudnnFindRNNForwardInferenceAlgorithmEx);
    r = so_cudnnFindRNNForwardInferenceAlgorithmEx(handle,rnnDesc,seqLength,xDesc,x,hxDesc,hx,cxDesc,cx,wDesc,w,yDesc,y,hyDesc,hy,cyDesc,cy,findIntensity,requestedAlgoCount,returnedAlgoCount,perfResults,workspace,workSpaceSizeInBytes);
    end_func(cudnnFindRNNForwardInferenceAlgorithmEx);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnGetRNNForwardTrainingAlgorithmMaxCount(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, int *count) {
    cudnnStatus_t r;
    begin_func(cudnnGetRNNForwardTrainingAlgorithmMaxCount);
    r = so_cudnnGetRNNForwardTrainingAlgorithmMaxCount(handle,rnnDesc,count);
    end_func(cudnnGetRNNForwardTrainingAlgorithmMaxCount);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnFindRNNForwardTrainingAlgorithmEx(cudnnHandle_t handle,const cudnnRNNDescriptor_t rnnDesc,const int seqLength,const cudnnTensorDescriptor_t *xDesc,const void *x,const cudnnTensorDescriptor_t hxDesc,const void *hx,const cudnnTensorDescriptor_t cxDesc,const void *cx,const cudnnFilterDescriptor_t wDesc,const void *w,const cudnnTensorDescriptor_t *yDesc,void *y,const cudnnTensorDescriptor_t hyDesc,void *hy,const cudnnTensorDescriptor_t cyDesc,void *cy,const float findIntensity,const int requestedAlgoCount,int *returnedAlgoCount,cudnnAlgorithmPerformance_t *perfResults,void *workspace,size_t workSpaceSizeInBytes,void *reserveSpace,size_t reserveSpaceSizeInBytes) {
    cudnnStatus_t r;
    begin_func(cudnnFindRNNForwardTrainingAlgorithmEx);
    r = so_cudnnFindRNNForwardTrainingAlgorithmEx(handle,rnnDesc,seqLength,xDesc,x,hxDesc,hx,cxDesc,cx,wDesc,w,yDesc,y,hyDesc,hy,cyDesc,cy,findIntensity,requestedAlgoCount,returnedAlgoCount,perfResults,workspace,workSpaceSizeInBytes,reserveSpace,reserveSpaceSizeInBytes);
    end_func(cudnnFindRNNForwardTrainingAlgorithmEx);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnGetRNNBackwardDataAlgorithmMaxCount(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, int *count) {
    cudnnStatus_t r;
    begin_func(cudnnGetRNNBackwardDataAlgorithmMaxCount);
    r = so_cudnnGetRNNBackwardDataAlgorithmMaxCount(handle,rnnDesc,count);
    end_func(cudnnGetRNNBackwardDataAlgorithmMaxCount);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnFindRNNBackwardDataAlgorithmEx(cudnnHandle_t handle,const cudnnRNNDescriptor_t rnnDesc,const int seqLength,const cudnnTensorDescriptor_t *yDesc,const void *y,const cudnnTensorDescriptor_t *dyDesc,const void *dy,const cudnnTensorDescriptor_t dhyDesc,const void *dhy,const cudnnTensorDescriptor_t dcyDesc,const void *dcy,const cudnnFilterDescriptor_t wDesc,const void *w,const cudnnTensorDescriptor_t hxDesc,const void *hx,const cudnnTensorDescriptor_t cxDesc,const void *cx,const cudnnTensorDescriptor_t *dxDesc,void *dx,const cudnnTensorDescriptor_t dhxDesc,void *dhx,const cudnnTensorDescriptor_t dcxDesc,void *dcx,const float findIntensity,const int requestedAlgoCount,int *returnedAlgoCount,cudnnAlgorithmPerformance_t *perfResults,void *workspace,size_t workSpaceSizeInBytes,void *reserveSpace,size_t reserveSpaceSizeInBytes) {
    cudnnStatus_t r;
    begin_func(cudnnFindRNNBackwardDataAlgorithmEx);
    r = so_cudnnFindRNNBackwardDataAlgorithmEx(handle,rnnDesc,seqLength,yDesc,y,dyDesc,dy,dhyDesc,dhy,dcyDesc,dcy,wDesc,w,hxDesc,hx,cxDesc,cx,dxDesc,dx,dhxDesc,dhx,dcxDesc,dcx,findIntensity,requestedAlgoCount,returnedAlgoCount,perfResults,workspace,workSpaceSizeInBytes,reserveSpace,reserveSpaceSizeInBytes);
    end_func(cudnnFindRNNBackwardDataAlgorithmEx);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnGetRNNBackwardWeightsAlgorithmMaxCount(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, int *count) {
    cudnnStatus_t r;
    begin_func(cudnnGetRNNBackwardWeightsAlgorithmMaxCount);
    r = so_cudnnGetRNNBackwardWeightsAlgorithmMaxCount(handle,rnnDesc,count);
    end_func(cudnnGetRNNBackwardWeightsAlgorithmMaxCount);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnFindRNNBackwardWeightsAlgorithmEx(cudnnHandle_t handle,const cudnnRNNDescriptor_t rnnDesc,const int seqLength,const cudnnTensorDescriptor_t *xDesc,const void *x,const cudnnTensorDescriptor_t hxDesc,const void *hx,const cudnnTensorDescriptor_t *yDesc,const void *y,const float findIntensity,const int requestedAlgoCount,int *returnedAlgoCount,cudnnAlgorithmPerformance_t *perfResults,const void *workspace,size_t workSpaceSizeInBytes,const cudnnFilterDescriptor_t dwDesc,void *dw,const void *reserveSpace,size_t reserveSpaceSizeInBytes) {
    cudnnStatus_t r;
    begin_func(cudnnFindRNNBackwardWeightsAlgorithmEx);
    r = so_cudnnFindRNNBackwardWeightsAlgorithmEx(handle,rnnDesc,seqLength,xDesc,x,hxDesc,hx,yDesc,y,findIntensity,requestedAlgoCount,returnedAlgoCount,perfResults,workspace,workSpaceSizeInBytes,dwDesc,dw,reserveSpace,reserveSpaceSizeInBytes);
    end_func(cudnnFindRNNBackwardWeightsAlgorithmEx);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnCreateSeqDataDescriptor(cudnnSeqDataDescriptor_t *seqDataDesc) {
    cudnnStatus_t r;
    begin_func(cudnnCreateSeqDataDescriptor);
    r = so_cudnnCreateSeqDataDescriptor(seqDataDesc);
    end_func(cudnnCreateSeqDataDescriptor);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnDestroySeqDataDescriptor(cudnnSeqDataDescriptor_t seqDataDesc) {
    cudnnStatus_t r;
    begin_func(cudnnDestroySeqDataDescriptor);
    r = so_cudnnDestroySeqDataDescriptor(seqDataDesc);
    end_func(cudnnDestroySeqDataDescriptor);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnSetSeqDataDescriptor(cudnnSeqDataDescriptor_t seqDataDesc,cudnnDataType_t dataType,int nbDims,const int dimA[],const cudnnSeqDataAxis_t axes[],size_t seqLengthArraySize,const int seqLengthArray[],void *paddingFill) {
    cudnnStatus_t r;
    begin_func(cudnnSetSeqDataDescriptor);
    r = so_cudnnSetSeqDataDescriptor(seqDataDesc,dataType,nbDims,dimA,axes,seqLengthArraySize,seqLengthArray,paddingFill);
    end_func(cudnnSetSeqDataDescriptor);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnGetSeqDataDescriptor(const cudnnSeqDataDescriptor_t seqDataDesc,cudnnDataType_t *dataType,int *nbDims,int nbDimsRequested,int dimA[],cudnnSeqDataAxis_t axes[],size_t *seqLengthArraySize,size_t seqLengthSizeRequested,int seqLengthArray[],void *paddingFill) {
    cudnnStatus_t r;
    begin_func(cudnnGetSeqDataDescriptor);
    r = so_cudnnGetSeqDataDescriptor(seqDataDesc,dataType,nbDims,nbDimsRequested,dimA,axes,seqLengthArraySize,seqLengthSizeRequested,seqLengthArray,paddingFill);
    end_func(cudnnGetSeqDataDescriptor);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnCreateAttnDescriptor(cudnnAttnDescriptor_t *attnDesc) {
    cudnnStatus_t r;
    begin_func(cudnnCreateAttnDescriptor);
    r = so_cudnnCreateAttnDescriptor(attnDesc);
    end_func(cudnnCreateAttnDescriptor);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnDestroyAttnDescriptor(cudnnAttnDescriptor_t attnDesc) {
    cudnnStatus_t r;
    begin_func(cudnnDestroyAttnDescriptor);
    r = so_cudnnDestroyAttnDescriptor(attnDesc);
    end_func(cudnnDestroyAttnDescriptor);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnSetAttnDescriptor(cudnnAttnDescriptor_t attnDesc,unsigned attnMode,int nHeads,double smScaler,cudnnDataType_t dataType,cudnnDataType_t computePrec,cudnnMathType_t mathType,cudnnDropoutDescriptor_t attnDropoutDesc,cudnnDropoutDescriptor_t postDropoutDesc,int qSize,int kSize,int vSize,int qProjSize,int kProjSize,int vProjSize,int oProjSize,int qoMaxSeqLength,int kvMaxSeqLength,int maxBatchSize,int maxBeamSize) {
    cudnnStatus_t r;
    begin_func(cudnnSetAttnDescriptor);
    r = so_cudnnSetAttnDescriptor(attnDesc,attnMode,nHeads,smScaler,dataType,computePrec,mathType,attnDropoutDesc,postDropoutDesc,qSize,kSize,vSize,qProjSize,kProjSize,vProjSize,oProjSize,qoMaxSeqLength,kvMaxSeqLength,maxBatchSize,maxBeamSize);
    end_func(cudnnSetAttnDescriptor);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnGetAttnDescriptor(cudnnAttnDescriptor_t attnDesc,unsigned *attnMode,int *nHeads,double *smScaler,cudnnDataType_t *dataType,cudnnDataType_t *computePrec,cudnnMathType_t *mathType,cudnnDropoutDescriptor_t *attnDropoutDesc,cudnnDropoutDescriptor_t *postDropoutDesc,int *qSize,int *kSize,int *vSize,int *qProjSize,int *kProjSize,int *vProjSize,int *oProjSize,int *qoMaxSeqLength,int *kvMaxSeqLength,int *maxBatchSize,int *maxBeamSize) {
    cudnnStatus_t r;
    begin_func(cudnnGetAttnDescriptor);
    r = so_cudnnGetAttnDescriptor(attnDesc,attnMode,nHeads,smScaler,dataType,computePrec,mathType,attnDropoutDesc,postDropoutDesc,qSize,kSize,vSize,qProjSize,kProjSize,vProjSize,oProjSize,qoMaxSeqLength,kvMaxSeqLength,maxBatchSize,maxBeamSize);
    end_func(cudnnGetAttnDescriptor);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnGetMultiHeadAttnBuffers(cudnnHandle_t handle,const cudnnAttnDescriptor_t attnDesc,size_t *weightSizeInBytes,size_t *workSpaceSizeInBytes,size_t *reserveSpaceSizeInBytes) {
    cudnnStatus_t r;
    begin_func(cudnnGetMultiHeadAttnBuffers);
    r = so_cudnnGetMultiHeadAttnBuffers(handle,attnDesc,weightSizeInBytes,workSpaceSizeInBytes,reserveSpaceSizeInBytes);
    end_func(cudnnGetMultiHeadAttnBuffers);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnGetMultiHeadAttnWeights(cudnnHandle_t handle,const cudnnAttnDescriptor_t attnDesc,cudnnMultiHeadAttnWeightKind_t wKind,size_t weightSizeInBytes,const void *weights,cudnnTensorDescriptor_t wDesc,void **wAddr) {
    cudnnStatus_t r;
    begin_func(cudnnGetMultiHeadAttnWeights);
    r = so_cudnnGetMultiHeadAttnWeights(handle,attnDesc,wKind,weightSizeInBytes,weights,wDesc,wAddr);
    end_func(cudnnGetMultiHeadAttnWeights);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnMultiHeadAttnForward(cudnnHandle_t handle,const cudnnAttnDescriptor_t attnDesc,int currIdx,const int *loWinIdx,const int *hiWinIdx,const int *seqLengthArrayQRO,const int *seqLengthArrayKV,const cudnnSeqDataDescriptor_t qDesc,const void *queries,const void *residuals,const cudnnSeqDataDescriptor_t kDesc,const void *keys,const cudnnSeqDataDescriptor_t vDesc,const void *values,const cudnnSeqDataDescriptor_t oDesc,void *out,size_t weightSizeInBytes,const void *weights,size_t workSpaceSizeInBytes,void *workSpace,size_t reserveSpaceSizeInBytes,void *reserveSpace) {
    cudnnStatus_t r;
    begin_func(cudnnMultiHeadAttnForward);
    r = so_cudnnMultiHeadAttnForward(handle,attnDesc,currIdx,loWinIdx,hiWinIdx,seqLengthArrayQRO,seqLengthArrayKV,qDesc,queries,residuals,kDesc,keys,vDesc,values,oDesc,out,weightSizeInBytes,weights,workSpaceSizeInBytes,workSpace,reserveSpaceSizeInBytes,reserveSpace);
    end_func(cudnnMultiHeadAttnForward);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnMultiHeadAttnBackwardData(cudnnHandle_t handle,const cudnnAttnDescriptor_t attnDesc,const int *loWinIdx,const int *hiWinIdx,const int *seqLengthArrayDQDO,const int *seqLengthArrayDKDV,const cudnnSeqDataDescriptor_t doDesc,const void *dout,const cudnnSeqDataDescriptor_t dqDesc,void *dqueries,const void *queries,const cudnnSeqDataDescriptor_t dkDesc,void *dkeys,const void *keys,const cudnnSeqDataDescriptor_t dvDesc,void *dvalues,const void *values,size_t weightSizeInBytes,const void *weights,size_t workSpaceSizeInBytes,void *workSpace,size_t reserveSpaceSizeInBytes,void *reserveSpace) {
    cudnnStatus_t r;
    begin_func(cudnnMultiHeadAttnBackwardData);
    r = so_cudnnMultiHeadAttnBackwardData(handle,attnDesc,loWinIdx,hiWinIdx,seqLengthArrayDQDO,seqLengthArrayDKDV,doDesc,dout,dqDesc,dqueries,queries,dkDesc,dkeys,keys,dvDesc,dvalues,values,weightSizeInBytes,weights,workSpaceSizeInBytes,workSpace,reserveSpaceSizeInBytes,reserveSpace);
    end_func(cudnnMultiHeadAttnBackwardData);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnMultiHeadAttnBackwardWeights(cudnnHandle_t handle,const cudnnAttnDescriptor_t attnDesc,cudnnWgradMode_t addGrad,const cudnnSeqDataDescriptor_t qDesc,const void *queries,const cudnnSeqDataDescriptor_t kDesc,const void *keys,const cudnnSeqDataDescriptor_t vDesc,const void *values,const cudnnSeqDataDescriptor_t doDesc,const void *dout,size_t weightSizeInBytes,const void *weights,void *dweights,size_t workSpaceSizeInBytes,void *workSpace,size_t reserveSpaceSizeInBytes,void *reserveSpace) {
    cudnnStatus_t r;
    begin_func(cudnnMultiHeadAttnBackwardWeights);
    r = so_cudnnMultiHeadAttnBackwardWeights(handle,attnDesc,addGrad,qDesc,queries,kDesc,keys,vDesc,values,doDesc,dout,weightSizeInBytes,weights,dweights,workSpaceSizeInBytes,workSpace,reserveSpaceSizeInBytes,reserveSpace);
    end_func(cudnnMultiHeadAttnBackwardWeights);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnCreateCTCLossDescriptor(cudnnCTCLossDescriptor_t *ctcLossDesc) {
    cudnnStatus_t r;
    begin_func(cudnnCreateCTCLossDescriptor);
    r = so_cudnnCreateCTCLossDescriptor(ctcLossDesc);
    end_func(cudnnCreateCTCLossDescriptor);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnSetCTCLossDescriptor(cudnnCTCLossDescriptor_t ctcLossDesc, cudnnDataType_t compType) {
    cudnnStatus_t r;
    begin_func(cudnnSetCTCLossDescriptor);
    r = so_cudnnSetCTCLossDescriptor(ctcLossDesc,compType);
    end_func(cudnnSetCTCLossDescriptor);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnSetCTCLossDescriptorEx(cudnnCTCLossDescriptor_t ctcLossDesc,cudnnDataType_t compType,cudnnLossNormalizationMode_t normMode,cudnnNanPropagation_t gradMode) {
    cudnnStatus_t r;
    begin_func(cudnnSetCTCLossDescriptorEx);
    r = so_cudnnSetCTCLossDescriptorEx(ctcLossDesc,compType,normMode,gradMode);
    end_func(cudnnSetCTCLossDescriptorEx);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnGetCTCLossDescriptor(cudnnCTCLossDescriptor_t ctcLossDesc, cudnnDataType_t *compType) {
    cudnnStatus_t r;
    begin_func(cudnnGetCTCLossDescriptor);
    r = so_cudnnGetCTCLossDescriptor(ctcLossDesc,compType);
    end_func(cudnnGetCTCLossDescriptor);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnGetCTCLossDescriptorEx(cudnnCTCLossDescriptor_t ctcLossDesc,cudnnDataType_t *compType,cudnnLossNormalizationMode_t *normMode,cudnnNanPropagation_t *gradMode) {
    cudnnStatus_t r;
    begin_func(cudnnGetCTCLossDescriptorEx);
    r = so_cudnnGetCTCLossDescriptorEx(ctcLossDesc,compType,normMode,gradMode);
    end_func(cudnnGetCTCLossDescriptorEx);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnDestroyCTCLossDescriptor(cudnnCTCLossDescriptor_t ctcLossDesc) {
    cudnnStatus_t r;
    begin_func(cudnnDestroyCTCLossDescriptor);
    r = so_cudnnDestroyCTCLossDescriptor(ctcLossDesc);
    end_func(cudnnDestroyCTCLossDescriptor);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnCTCLoss(cudnnHandle_t handle,const cudnnTensorDescriptor_t probsDesc,const void *probs,const int *labels,const int *labelLengths, const int *inputLengths, void *costs,const cudnnTensorDescriptor_t gradientsDesc, const void *gradients, cudnnCTCLossAlgo_t algo,cudnnCTCLossDescriptor_t ctcLossDesc,void *workspace,size_t workSpaceSizeInBytes) {
    cudnnStatus_t r;
    begin_func(cudnnCTCLoss);
    r = so_cudnnCTCLoss(handle,probsDesc,probs,labels,labelLengths,inputLengths,costs,gradientsDesc,gradients,algo,ctcLossDesc,workspace,workSpaceSizeInBytes);
    end_func(cudnnCTCLoss);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnGetCTCLossWorkspaceSize(cudnnHandle_t handle,const cudnnTensorDescriptor_t probsDesc,const cudnnTensorDescriptor_t gradientsDesc,const int *labels,const int *labelLengths,const int *inputLengths,cudnnCTCLossAlgo_t algo,cudnnCTCLossDescriptor_t ctcLossDesc,size_t *sizeInBytes) {
    cudnnStatus_t r;
    begin_func(cudnnGetCTCLossWorkspaceSize);
    r = so_cudnnGetCTCLossWorkspaceSize(handle,probsDesc,gradientsDesc,labels,labelLengths,inputLengths,algo,ctcLossDesc,sizeInBytes);
    end_func(cudnnGetCTCLossWorkspaceSize);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnCreateAlgorithmDescriptor(cudnnAlgorithmDescriptor_t *algoDesc) {
    cudnnStatus_t r;
    begin_func(cudnnCreateAlgorithmDescriptor);
    r = so_cudnnCreateAlgorithmDescriptor(algoDesc);
    end_func(cudnnCreateAlgorithmDescriptor);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnSetAlgorithmDescriptor(cudnnAlgorithmDescriptor_t algoDesc, cudnnAlgorithm_t algorithm) {
    cudnnStatus_t r;
    begin_func(cudnnSetAlgorithmDescriptor);
    r = so_cudnnSetAlgorithmDescriptor(algoDesc,algorithm);
    end_func(cudnnSetAlgorithmDescriptor);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnGetAlgorithmDescriptor(const cudnnAlgorithmDescriptor_t algoDesc, cudnnAlgorithm_t *algorithm) {
    cudnnStatus_t r;
    begin_func(cudnnGetAlgorithmDescriptor);
    r = so_cudnnGetAlgorithmDescriptor(algoDesc,algorithm);
    end_func(cudnnGetAlgorithmDescriptor);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnCopyAlgorithmDescriptor(const cudnnAlgorithmDescriptor_t src, cudnnAlgorithmDescriptor_t dest) {
    cudnnStatus_t r;
    begin_func(cudnnCopyAlgorithmDescriptor);
    r = so_cudnnCopyAlgorithmDescriptor(src,dest);
    end_func(cudnnCopyAlgorithmDescriptor);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnDestroyAlgorithmDescriptor(cudnnAlgorithmDescriptor_t algoDesc) {
    cudnnStatus_t r;
    begin_func(cudnnDestroyAlgorithmDescriptor);
    r = so_cudnnDestroyAlgorithmDescriptor(algoDesc);
    end_func(cudnnDestroyAlgorithmDescriptor);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnCreateAlgorithmPerformance(cudnnAlgorithmPerformance_t *algoPerf, int numberToCreate) {
    cudnnStatus_t r;
    begin_func(cudnnCreateAlgorithmPerformance);
    r = so_cudnnCreateAlgorithmPerformance(algoPerf,numberToCreate);
    end_func(cudnnCreateAlgorithmPerformance);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnSetAlgorithmPerformance(cudnnAlgorithmPerformance_t algoPerf,cudnnAlgorithmDescriptor_t algoDesc,cudnnStatus_t status,float time,size_t memory) {
    cudnnStatus_t r;
    begin_func(cudnnSetAlgorithmPerformance);
    r = so_cudnnSetAlgorithmPerformance(algoPerf,algoDesc,status,time,memory);
    end_func(cudnnSetAlgorithmPerformance);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnGetAlgorithmPerformance(const cudnnAlgorithmPerformance_t algoPerf,cudnnAlgorithmDescriptor_t *algoDesc,cudnnStatus_t *status,float *time,size_t *memory) {
    cudnnStatus_t r;
    begin_func(cudnnGetAlgorithmPerformance);
    r = so_cudnnGetAlgorithmPerformance(algoPerf,algoDesc,status,time,memory);
    end_func(cudnnGetAlgorithmPerformance);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnDestroyAlgorithmPerformance(cudnnAlgorithmPerformance_t *algoPerf, int numberToDestroy) {
    cudnnStatus_t r;
    begin_func(cudnnDestroyAlgorithmPerformance);
    r = so_cudnnDestroyAlgorithmPerformance(algoPerf,numberToDestroy);
    end_func(cudnnDestroyAlgorithmPerformance);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnGetAlgorithmSpaceSize(cudnnHandle_t handle, cudnnAlgorithmDescriptor_t algoDesc, size_t *algoSpaceSizeInBytes) {
    cudnnStatus_t r;
    begin_func(cudnnGetAlgorithmSpaceSize);
    r = so_cudnnGetAlgorithmSpaceSize(handle,algoDesc,algoSpaceSizeInBytes);
    end_func(cudnnGetAlgorithmSpaceSize);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnSaveAlgorithm(cudnnHandle_t handle,cudnnAlgorithmDescriptor_t algoDesc,void *algoSpace,size_t algoSpaceSizeInBytes) {
    cudnnStatus_t r;
    begin_func(cudnnSaveAlgorithm);
    r = so_cudnnSaveAlgorithm(handle,algoDesc,algoSpace,algoSpaceSizeInBytes);
    end_func(cudnnSaveAlgorithm);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnRestoreAlgorithm(cudnnHandle_t handle,void *algoSpace,size_t algoSpaceSizeInBytes,cudnnAlgorithmDescriptor_t algoDesc) {
    cudnnStatus_t r;
    begin_func(cudnnRestoreAlgorithm);
    r = so_cudnnRestoreAlgorithm(handle,algoSpace,algoSpaceSizeInBytes,algoDesc);
    end_func(cudnnRestoreAlgorithm);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnSetCallback(unsigned mask, void *udata, cudnnCallback_t fptr) {
    cudnnStatus_t r;
    begin_func(cudnnSetCallback);
    r = so_cudnnSetCallback(mask,udata,fptr);
    end_func(cudnnSetCallback);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnGetCallback(unsigned *mask, void **udata, cudnnCallback_t *fptr) {
    cudnnStatus_t r;
    begin_func(cudnnGetCallback);
    r = so_cudnnGetCallback(mask,udata,fptr);
    end_func(cudnnGetCallback);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnCreateFusedOpsConstParamPack(cudnnFusedOpsConstParamPack_t *constPack, cudnnFusedOps_t ops) {
    cudnnStatus_t r;
    begin_func(cudnnCreateFusedOpsConstParamPack);
    r = so_cudnnCreateFusedOpsConstParamPack(constPack,ops);
    end_func(cudnnCreateFusedOpsConstParamPack);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnDestroyFusedOpsConstParamPack(cudnnFusedOpsConstParamPack_t constPack) {
    cudnnStatus_t r;
    begin_func(cudnnDestroyFusedOpsConstParamPack);
    r = so_cudnnDestroyFusedOpsConstParamPack(constPack);
    end_func(cudnnDestroyFusedOpsConstParamPack);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnSetFusedOpsConstParamPackAttribute(cudnnFusedOpsConstParamPack_t constPack,cudnnFusedOpsConstParamLabel_t paramLabel,const void *param) {
    cudnnStatus_t r;
    begin_func(cudnnSetFusedOpsConstParamPackAttribute);
    r = so_cudnnSetFusedOpsConstParamPackAttribute(constPack,paramLabel,param);
    end_func(cudnnSetFusedOpsConstParamPackAttribute);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnGetFusedOpsConstParamPackAttribute(const cudnnFusedOpsConstParamPack_t constPack,cudnnFusedOpsConstParamLabel_t paramLabel,void *param,int *isNULL) {
    cudnnStatus_t r;
    begin_func(cudnnGetFusedOpsConstParamPackAttribute);
    r = so_cudnnGetFusedOpsConstParamPackAttribute(constPack,paramLabel,param,isNULL);
    end_func(cudnnGetFusedOpsConstParamPackAttribute);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnCreateFusedOpsVariantParamPack(cudnnFusedOpsVariantParamPack_t *varPack, cudnnFusedOps_t ops) {
    cudnnStatus_t r;
    begin_func(cudnnCreateFusedOpsVariantParamPack);
    r = so_cudnnCreateFusedOpsVariantParamPack(varPack,ops);
    end_func(cudnnCreateFusedOpsVariantParamPack);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnDestroyFusedOpsVariantParamPack(cudnnFusedOpsVariantParamPack_t varPack) {
    cudnnStatus_t r;
    begin_func(cudnnDestroyFusedOpsVariantParamPack);
    r = so_cudnnDestroyFusedOpsVariantParamPack(varPack);
    end_func(cudnnDestroyFusedOpsVariantParamPack);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnSetFusedOpsVariantParamPackAttribute(cudnnFusedOpsVariantParamPack_t varPack,cudnnFusedOpsVariantParamLabel_t paramLabel,void *ptr) {
    cudnnStatus_t r;
    begin_func(cudnnSetFusedOpsVariantParamPackAttribute);
    r = so_cudnnSetFusedOpsVariantParamPackAttribute(varPack,paramLabel,ptr);
    end_func(cudnnSetFusedOpsVariantParamPackAttribute);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnGetFusedOpsVariantParamPackAttribute(const cudnnFusedOpsVariantParamPack_t varPack,cudnnFusedOpsVariantParamLabel_t paramLabel,void *ptr) {
    cudnnStatus_t r;
    begin_func(cudnnGetFusedOpsVariantParamPackAttribute);
    r = so_cudnnGetFusedOpsVariantParamPackAttribute(varPack,paramLabel,ptr);
    end_func(cudnnGetFusedOpsVariantParamPackAttribute);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnCreateFusedOpsPlan(cudnnFusedOpsPlan_t *plan, cudnnFusedOps_t ops) {
    cudnnStatus_t r;
    begin_func(cudnnCreateFusedOpsPlan);
    r = so_cudnnCreateFusedOpsPlan(plan,ops);
    end_func(cudnnCreateFusedOpsPlan);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnDestroyFusedOpsPlan(cudnnFusedOpsPlan_t plan) {
    cudnnStatus_t r;
    begin_func(cudnnDestroyFusedOpsPlan);
    r = so_cudnnDestroyFusedOpsPlan(plan);
    end_func(cudnnDestroyFusedOpsPlan);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnMakeFusedOpsPlan(cudnnHandle_t handle,cudnnFusedOpsPlan_t plan,const cudnnFusedOpsConstParamPack_t constPack,size_t *workspaceSizeInBytes) {
    cudnnStatus_t r;
    begin_func(cudnnMakeFusedOpsPlan);
    r = so_cudnnMakeFusedOpsPlan(handle,plan,constPack,workspaceSizeInBytes);
    end_func(cudnnMakeFusedOpsPlan);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnFusedOpsExecute(cudnnHandle_t handle, const cudnnFusedOpsPlan_t plan, cudnnFusedOpsVariantParamPack_t varPack) {
    cudnnStatus_t r;
    begin_func(cudnnFusedOpsExecute);
    r = so_cudnnFusedOpsExecute(handle,plan,varPack);
    end_func(cudnnFusedOpsExecute);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnSetRNNDescriptor_v6(cudnnHandle_t handle,cudnnRNNDescriptor_t rnnDesc,const int hiddenSize,const int numLayers,cudnnDropoutDescriptor_t dropoutDesc,cudnnRNNInputMode_t inputMode,cudnnDirectionMode_t direction,cudnnRNNMode_t mode,cudnnRNNAlgo_t algo,cudnnDataType_t mathPrec) {
    cudnnStatus_t r;
    begin_func(cudnnSetRNNDescriptor_v6);
    r = so_cudnnSetRNNDescriptor_v6(handle,rnnDesc,hiddenSize,numLayers,dropoutDesc,inputMode,direction,mode,algo,mathPrec);
    end_func(cudnnSetRNNDescriptor_v6);
    checkCudnnErrors(r);
    return r;
}

cudnnStatus_t cudnnSetRNNDescriptor_v5(cudnnRNNDescriptor_t rnnDesc,int hiddenSize,int numLayers,cudnnDropoutDescriptor_t dropoutDesc,cudnnRNNInputMode_t inputMode,cudnnDirectionMode_t direction,cudnnRNNMode_t mode,cudnnDataType_t mathPrec) {
    cudnnStatus_t r;
    begin_func(cudnnSetRNNDescriptor_v5);
    r = so_cudnnSetRNNDescriptor_v5(rnnDesc,hiddenSize,numLayers,dropoutDesc,inputMode,direction,mode,mathPrec);
    end_func(cudnnSetRNNDescriptor_v5);
    checkCudnnErrors(r);
    return r;
}
