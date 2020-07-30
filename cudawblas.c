#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <cuda.h>
#include <string.h>
#include <sys/time.h>
#include <cublas_v2.h>
#include <cublas_api.h>
#include <assert.h>
#include <dlfcn.h>

#include "cudaw.h"

static const char LIB_STRING[] = "/workspace/libcublas.so.10.0.130";

// DEFSO & LDSYM

#define MAX_FUNC 256
#include "ldsym.h"

#define DEFSO(func)  static int idx_##func; static cublasStatus_t (*so_##func)
#define FSWAP(func)  &so_##func,

// define all so_clblasXXXs
DEFSO(cublasCreate_v2)(cublasHandle_t *handle);
DEFSO(cublasDestroy_v2)(cublasHandle_t handle);
DEFSO(cublasGetVersion_v2)(cublasHandle_t handle, int *version);
DEFSO(cublasGetProperty)(libraryPropertyType type, int *value);
DEFSO(cublasSetStream_v2)(cublasHandle_t handle, cudaStream_t streamId);
DEFSO(cublasGetStream_v2)(cublasHandle_t handle, cudaStream_t *streamId);
DEFSO(cublasGetPointerMode_v2)(cublasHandle_t handle, cublasPointerMode_t *mode);
DEFSO(cublasSetPointerMode_v2)(cublasHandle_t handle, cublasPointerMode_t mode);
DEFSO(cublasSetVector)(int n, int elemSize, const void *x, int incx, void *y, int incy);
DEFSO(cublasGetVector)(int n, int elemSize, const void *x, int incx, void *y, int incy);
DEFSO(cublasSetMatrix)(int rows, int cols, int elemSize, const void *A, int lda, void *B, int ldb);
DEFSO(cublasGetMatrix)(int rows, int cols, int elemSize, const void *A, int lda, void *B, int ldb);
DEFSO(cublasSetVectorAsync)(int n, int elemSize, const void *hostPtr, int incx, void *devicePtr, int incy, cudaStream_t stream);
DEFSO(cublasGetVectorAsync)(int n, int elemSize, const void *devicePtr, int incx, void *hostPtr, int incy, cudaStream_t stream);
DEFSO(cublasSetMatrixAsync)(int rows, int cols, int elemSize, const void *A, int lda, void *B, int ldb, cudaStream_t stream);
DEFSO(cublasGetMatrixAsync)(int rows, int cols, int elemSize, const void *A, int lda, void *B, int ldb, cudaStream_t stream);
DEFSO(cublasSetAtomicsMode)(cublasHandle_t handle, cublasAtomicsMode_t mode);
DEFSO(cublasGetAtomicsMode)(cublasHandle_t handle, cublasAtomicsMode_t *mode);
DEFSO(cublasSetMathMode)(cublasHandle_t handle, cublasMath_t mode);
DEFSO(cublasGetMathMode)(cublasHandle_t handle, cublasMath_t *mode);
DEFSO(cublasLoggerConfigure)(int logIsOn, int logToStdOut, int logToStdErr, const char* logFileName);
DEFSO(cublasGetLoggerCallback)(cublasLogCallback* userCallback);
DEFSO(cublasSetLoggerCallback)(cublasLogCallback userCallback);
DEFSO(cublasIsamax_v2)(cublasHandle_t handle, int n, const float *x, int incx, int *result);
DEFSO(cublasIdamax_v2)(cublasHandle_t handle, int n, const double *x, int incx, int *result);
DEFSO(cublasIcamax_v2)(cublasHandle_t handle, int n, const cuComplex *x, int incx, int *result);
DEFSO(cublasIzamax_v2)(cublasHandle_t handle, int n, const cuDoubleComplex *x, int incx, int *result);
DEFSO(cublasIsamin_v2)(cublasHandle_t handle, int n, const float *x, int incx, int *result);
DEFSO(cublasIdamin_v2)(cublasHandle_t handle, int n, const double *x, int incx, int *result);
DEFSO(cublasIcamin_v2)(cublasHandle_t handle, int n, const cuComplex *x, int incx, int *result);
DEFSO(cublasIzamin_v2)(cublasHandle_t handle, int n, const cuDoubleComplex *x, int incx, int *result);
DEFSO(cublasScopy_v2)(cublasHandle_t handle, int n, const float *x, int incx, float *y, int incy);
DEFSO(cublasDcopy_v2)(cublasHandle_t handle, int n, const double *x, int incx, double *y, int incy);
DEFSO(cublasCcopy_v2)(cublasHandle_t handle, int n, const cuComplex *x, int incx, cuComplex *y, int incy);
DEFSO(cublasZcopy_v2)(cublasHandle_t handle, int n, const cuDoubleComplex *x, int incx, cuDoubleComplex *y, int incy);
DEFSO(cublasSdot_v2)(cublasHandle_t handle, int n, const float *x, int incx, const float *y, int incy, float *result);
DEFSO(cublasDdot_v2)(cublasHandle_t handle, int n, const double *x, int incx, const double *y, int incy, double *result);
DEFSO(cublasCdotu_v2)(cublasHandle_t handle, int n, const cuComplex *x, int incx, const cuComplex *y, int incy, cuComplex *result);
DEFSO(cublasCdotc_v2)(cublasHandle_t handle, int n, const cuComplex *x, int incx, const cuComplex *y, int incy, cuComplex *result);
DEFSO(cublasZdotu_v2)(cublasHandle_t handle, int n, const cuDoubleComplex *x, int incx, const cuDoubleComplex *y, int incy, cuDoubleComplex *result);
DEFSO(cublasZdotc_v2)(cublasHandle_t handle, int n, const cuDoubleComplex *x, int incx, const cuDoubleComplex *y, int incy, cuDoubleComplex *result);
DEFSO(cublasSnrm2_v2)(cublasHandle_t handle, int n, const float *x, int incx, float *result);
DEFSO(cublasDnrm2_v2)(cublasHandle_t handle, int n, const double *x, int incx, double *result);
DEFSO(cublasScnrm2_v2)(cublasHandle_t handle, int n, const cuComplex *x, int incx, float *result);
DEFSO(cublasDznrm2_v2)(cublasHandle_t handle, int n, const cuDoubleComplex *x, int incx, double *result);
DEFSO(cublasSrot_v2)(cublasHandle_t handle, int n, float *x, int incx, float *y, int incy, const float *c, const float *s);
DEFSO(cublasDrot_v2)(cublasHandle_t handle, int n, double *x, int incx, double *y, int incy, const double *c, const double *s);
DEFSO(cublasCrot_v2)(cublasHandle_t handle, int n, cuComplex *x, int incx, cuComplex *y, int incy, const float *c, const cuComplex *s);
DEFSO(cublasCsrot_v2)(cublasHandle_t handle, int n, cuComplex *x, int incx, cuComplex *y, int incy, const float *c, const float *s);
DEFSO(cublasZrot_v2)(cublasHandle_t handle, int n, cuDoubleComplex *x, int incx, cuDoubleComplex *y, int incy, const double *c, const cuDoubleComplex *s);
DEFSO(cublasZdrot_v2)(cublasHandle_t handle, int n, cuDoubleComplex *x, int incx, cuDoubleComplex *y, int incy, const double *c, const double *s);
DEFSO(cublasSrotg_v2)(cublasHandle_t handle, float *a, float *b, float *c, float *s);
DEFSO(cublasDrotg_v2)(cublasHandle_t handle, double *a, double *b, double *c, double *s);
DEFSO(cublasCrotg_v2)(cublasHandle_t handle, cuComplex *a, cuComplex *b, float *c, cuComplex *s);
DEFSO(cublasZrotg_v2)(cublasHandle_t handle, cuDoubleComplex *a, cuDoubleComplex *b, double *c, cuDoubleComplex *s);
DEFSO(cublasSrotm_v2)(cublasHandle_t handle, int n, float *x, int incx, float *y, int incy, const float* param);
DEFSO(cublasDrotm_v2)(cublasHandle_t handle, int n, double *x, int incx, double *y, int incy, const double* param);
DEFSO(cublasSrotmg_v2)(cublasHandle_t handle, float *d1, float *d2, float *x1, const float *y1, float* param);
DEFSO(cublasDrotmg_v2)(cublasHandle_t handle, double *d1, double *d2, double *x1, const double *y1, double* param);
DEFSO(cublasSgbmv_v2)(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int kl, int ku, const float *alpha, const float *A, int lda, const float *x, int incx, const float *beta, float *y, int incy);
DEFSO(cublasDgbmv_v2)(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int kl, int ku, const double *alpha, const double *A, int lda, const double *x, int incx, const double *beta, double *y, int incy);
DEFSO(cublasCgbmv_v2)(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int kl, int ku, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *x, int incx, const cuComplex *beta, cuComplex *y, int incy);
DEFSO(cublasZgbmv_v2)(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int kl, int ku, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *x, int incx, const cuDoubleComplex *beta, cuDoubleComplex *y, int incy);
DEFSO(cublasSgemv_v2)(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const float *alpha, const float *A, int lda, const float *x, int incx, const float *beta, float *y, int incy);
DEFSO(cublasDgemv_v2)(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const double *alpha, const double *A, int lda, const double *x, int incx, const double *beta, double *y, int incy);
DEFSO(cublasCgemv_v2)(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *x, int incx, const cuComplex *beta, cuComplex *y, int incy);
DEFSO(cublasZgemv_v2)(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *x, int incx, const cuDoubleComplex *beta, cuDoubleComplex *y, int incy);
DEFSO(cublasSger_v2)(cublasHandle_t handle, int m, int n, const float *alpha, const float *x, int incx, const float *y, int incy, float *A, int lda);
DEFSO(cublasDger_v2)(cublasHandle_t handle, int m, int n, const double *alpha, const double *x, int incx, const double *y, int incy, double *A, int lda);
DEFSO(cublasCgeru_v2)(cublasHandle_t handle, int m, int n, const cuComplex *alpha, const cuComplex *x, int incx, const cuComplex *y, int incy, cuComplex *A, int lda);
DEFSO(cublasCgerc_v2)(cublasHandle_t handle, int m, int n, const cuComplex *alpha, const cuComplex *x, int incx, const cuComplex *y, int incy, cuComplex *A, int lda);
DEFSO(cublasZgeru_v2)(cublasHandle_t handle, int m, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *x, int incx, const cuDoubleComplex *y, int incy, cuDoubleComplex *A, int lda);
DEFSO(cublasZgerc_v2)(cublasHandle_t handle, int m, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *x, int incx, const cuDoubleComplex *y, int incy, cuDoubleComplex *A, int lda);
DEFSO(cublasSsbmv_v2)(cublasHandle_t handle, cublasFillMode_t uplo, int n, int k, const float *alpha, const float *A, int lda, const float *x, int incx, const float *beta, float *y, int incy);
DEFSO(cublasDsbmv_v2)(cublasHandle_t handle, cublasFillMode_t uplo, int n, int k, const double *alpha, const double *A, int lda, const double *x, int incx, const double *beta, double *y, int incy);
DEFSO(cublasSspmv_v2)(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float *alpha, const float *AP, const float *x, int incx, const float *beta, float *y, int incy);
DEFSO(cublasDspmv_v2)(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double *alpha, const double *AP, const double *x, int incx, const double *beta, double *y, int incy);
DEFSO(cublasSspr_v2)(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float *alpha, const float *x, int incx, float *AP);
DEFSO(cublasDspr_v2)(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double *alpha, const double *x, int incx, double *AP);
DEFSO(cublasSspr2_v2)(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float *alpha, const float *x, int incx, const float *y, int incy, float* AP);
DEFSO(cublasDspr2_v2)(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double *alpha, const double *x, int incx, const double *y, int incy, double* AP);
DEFSO(cublasSsymv_v2)(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float *alpha, const float *A, int lda, const float *x, int incx, const float *beta, float *y, int incy);
DEFSO(cublasDsymv_v2)(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double *alpha, const double *A, int lda, const double *x, int incx, const double *beta, double *y, int incy);
DEFSO(cublasCsymv_v2)(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *x, int incx, const cuComplex *beta, cuComplex *y, int incy);
DEFSO(cublasZsymv_v2)(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *x, int incx, const cuDoubleComplex *beta, cuDoubleComplex *y, int incy);
DEFSO(cublasSsyr_v2)(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float *alpha, const float *x, int incx, float *A, int lda);
DEFSO(cublasDsyr_v2)(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double *alpha, const double *x, int incx, double *A, int lda);
DEFSO(cublasCsyr_v2)(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex *alpha, const cuComplex *x, int incx, cuComplex *A, int lda);
DEFSO(cublasZsyr_v2)(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *x, int incx, cuDoubleComplex *A, int lda);
DEFSO(cublasSsyr2_v2)(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float *alpha, const float *x, int incx, const float *y, int incy, float *A, int lda);
DEFSO(cublasDsyr2_v2)(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double *alpha, const double *x, int incx, const double *y, int incy, double *A, int lda);
DEFSO(cublasCsyr2_v2)(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex *alpha, const cuComplex *x, int incx, const cuComplex *y, int incy, cuComplex *A, int lda);
DEFSO(cublasZsyr2_v2)(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *x, int incx, const cuDoubleComplex *y, int incy, cuDoubleComplex *A, int lda);
DEFSO(cublasStbmv_v2)(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const float *A, int lda, float *x, int incx);
DEFSO(cublasDtbmv_v2)(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const double *A, int lda, double *x, int incx);
DEFSO(cublasCtbmv_v2)(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const cuComplex *A, int lda, cuComplex *x, int incx);
DEFSO(cublasZtbmv_v2)(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const cuDoubleComplex *A, int lda, cuDoubleComplex *x, int incx);
DEFSO(cublasStbsv_v2)(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const float *A, int lda, float *x, int incx);
DEFSO(cublasDtbsv_v2)(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const double *A, int lda, double *x, int incx);
DEFSO(cublasCtbsv_v2)(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const cuComplex *A, int lda, cuComplex *x, int incx);
DEFSO(cublasZtbsv_v2)(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const cuDoubleComplex *A, int lda, cuDoubleComplex *x, int incx);
DEFSO(cublasStpmv_v2)(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const float *AP, float *x, int incx);
DEFSO(cublasDtpmv_v2)(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const double *AP, double *x, int incx);
DEFSO(cublasCtpmv_v2)(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuComplex *AP, cuComplex *x, int incx);
DEFSO(cublasZtpmv_v2)(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuDoubleComplex *AP, cuDoubleComplex *x, int incx);
DEFSO(cublasStpsv_v2)(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const float *AP, float *x, int incx);
DEFSO(cublasDtpsv_v2)(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const double *AP, double *x, int incx);
DEFSO(cublasCtpsv_v2)(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuComplex *AP, cuComplex *x, int incx);
DEFSO(cublasZtpsv_v2)(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuDoubleComplex *AP, cuDoubleComplex *x, int incx);
DEFSO(cublasStrmv_v2)(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const float *A, int lda, float *x, int incx);
DEFSO(cublasDtrmv_v2)(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const double *A, int lda, double *x, int incx);
DEFSO(cublasCtrmv_v2)(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuComplex *A, int lda, cuComplex *x, int incx);
DEFSO(cublasZtrmv_v2)(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuDoubleComplex *A, int lda, cuDoubleComplex *x, int incx);
DEFSO(cublasStrsv_v2)(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const float *A, int lda, float *x, int incx);
DEFSO(cublasDtrsv_v2)(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const double *A, int lda, double *x, int incx);
DEFSO(cublasCtrsv_v2)(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuComplex *A, int lda, cuComplex *x, int incx);
DEFSO(cublasZtrsv_v2)(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuDoubleComplex *A, int lda, cuDoubleComplex *x, int incx);
DEFSO(cublasChemv_v2)(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *x, int incx, const cuComplex *beta, cuComplex *y, int incy);
DEFSO(cublasZhemv_v2)(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *x, int incx, const cuDoubleComplex *beta, cuDoubleComplex *y, int incy);
DEFSO(cublasChbmv_v2)(cublasHandle_t handle, cublasFillMode_t uplo, int n, int k, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *x, int incx, const cuComplex *beta, cuComplex *y, int incy);
DEFSO(cublasZhbmv_v2)(cublasHandle_t handle, cublasFillMode_t uplo, int n, int k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *x, int incx, const cuDoubleComplex *beta, cuDoubleComplex *y, int incy);
DEFSO(cublasChpmv_v2)(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex *alpha, const cuComplex *AP, const cuComplex *x, int incx, const cuComplex *beta, cuComplex *y, int incy);
DEFSO(cublasZhpmv_v2)(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *AP, const cuDoubleComplex *x, int incx, const cuDoubleComplex *beta, cuDoubleComplex *y, int incy);
DEFSO(cublasCher_v2)(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float *alpha, const cuComplex *x, int incx, cuComplex *A, int lda);
DEFSO(cublasZher_v2)(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double *alpha, const cuDoubleComplex *x, int incx, cuDoubleComplex *A, int lda);
DEFSO(cublasCher2_v2)(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex *alpha, const cuComplex *x, int incx, const cuComplex *y, int incy, cuComplex *A, int lda);
DEFSO(cublasZher2_v2)(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *x, int incx, const cuDoubleComplex *y, int incy, cuDoubleComplex *A, int lda);
DEFSO(cublasChpr_v2)(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float *alpha, const cuComplex *x, int incx, cuComplex *AP);
DEFSO(cublasZhpr_v2)(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double *alpha, const cuDoubleComplex *x, int incx, cuDoubleComplex *AP);
DEFSO(cublasChpr2_v2)(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex *alpha, const cuComplex *x, int incx, const cuComplex *y, int incy, cuComplex *AP);
DEFSO(cublasZhpr2_v2)(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *x, int incx, const cuDoubleComplex *y, int incy, cuDoubleComplex *AP);
DEFSO(cublasSgemm_v2)(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc);
DEFSO(cublasDgemm_v2)(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const double *alpha, const double *A, int lda, const double *B, int ldb, const double *beta, double *C, int ldc);
DEFSO(cublasCgemm_v2)(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *B, int ldb, const cuComplex *beta, cuComplex *C, int ldc);
DEFSO(cublasZgemm_v2)(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb, const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc);
DEFSO(cublasCgemm3m)(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *B, int ldb, const cuComplex *beta, cuComplex *C, int ldc);
DEFSO(cublasZgemm3m)(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb, const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc);
DEFSO(cublasSgemmBatched)(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha, const float *const Aarray[], int lda, const float *const Barray[], int ldb, const float *beta, float *const Carray[], int ldc, int batchCount);
DEFSO(cublasDgemmBatched)(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const double *alpha, const double *const Aarray[], int lda, const double *const Barray[], int ldb, const double *beta, double *const Carray[], int ldc, int batchCount);
DEFSO(cublasCgemmBatched)(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuComplex *alpha, const cuComplex *const Aarray[], int lda, const cuComplex *const Barray[], int ldb, const cuComplex *beta, cuComplex *const Carray[], int ldc, int batchCount);
DEFSO(cublasZgemmBatched)(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuDoubleComplex *alpha, const cuDoubleComplex *const Aarray[], int lda, const cuDoubleComplex *const Barray[], int ldb, const cuDoubleComplex *beta, cuDoubleComplex *const Carray[], int ldc, int batchCount);
DEFSO(cublasSgemmStridedBatched)(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha, const float *A, int lda, long long int strideA, const float *B, int ldb, long long int strideB, const float *beta, float *C, int ldc, long long int strideC, int batchCount);
DEFSO(cublasDgemmStridedBatched)(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const double *alpha, const double *A, int lda, long long int strideA, const double *B, int ldb, long long int strideB, const double *beta, double *C, int ldc, long long int strideC, int batchCount);
DEFSO(cublasCgemmStridedBatched)(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuComplex *alpha, const cuComplex *A, int lda, long long int strideA, const cuComplex *B, int ldb, long long int strideB, const cuComplex *beta, cuComplex *C, int ldc, long long int strideC, int batchCount);
DEFSO(cublasZgemmStridedBatched)(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, long long int strideA, const cuDoubleComplex *B, int ldb, long long int strideB, const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc, long long int strideC, int batchCount);
DEFSO(cublasSsymm_v2)(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc);
DEFSO(cublasDsymm_v2)(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const double *alpha, const double *A, int lda, const double *B, int ldb, const double *beta, double *C, int ldc);
DEFSO(cublasCsymm_v2)(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *B, int ldb, const cuComplex *beta, cuComplex *C, int ldc);
DEFSO(cublasZsymm_v2)(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb, const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc);
DEFSO(cublasSsyrk_v2)(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const float *alpha, const float *A, int lda, const float *beta, float *C, int ldc);
DEFSO(cublasDsyrk_v2)(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const double *alpha, const double *A, int lda, const double *beta, double *C, int ldc);
DEFSO(cublasCsyrk_v2)(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *beta, cuComplex *C, int ldc);
DEFSO(cublasZsyrk_v2)(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc);
DEFSO(cublasSsyr2k_v2)(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc);
DEFSO(cublasDsyr2k_v2)(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const double *alpha, const double *A, int lda, const double *B, int ldb, const double *beta, double *C, int ldc);
DEFSO(cublasCsyr2k_v2)(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *B, int ldb, const cuComplex *beta, cuComplex *C, int ldc);
DEFSO(cublasZsyr2k_v2)(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb, const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc);
DEFSO(cublasSsyrkx)(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc);
DEFSO(cublasDsyrkx)(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const double *alpha, const double *A, int lda, const double *B, int ldb, const double *beta, double *C, int ldc);
DEFSO(cublasCsyrkx)(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *B, int ldb, const cuComplex *beta, cuComplex *C, int ldc);
DEFSO(cublasZsyrkx)(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb, const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc);
DEFSO(cublasStrmm_v2)(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const float *alpha, const float *A, int lda, const float *B, int ldb, float *C, int ldc);
DEFSO(cublasDtrmm_v2)(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const double *alpha, const double *A, int lda, const double *B, int ldb, double *C, int ldc);
DEFSO(cublasCtrmm_v2)(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *B, int ldb, cuComplex *C, int ldc);
DEFSO(cublasZtrmm_v2)(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb, cuDoubleComplex *C, int ldc);
DEFSO(cublasStrsm_v2)(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const float *alpha, const float *A, int lda, float *B, int ldb);
DEFSO(cublasDtrsm_v2)(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const double *alpha, const double *A, int lda, double *B, int ldb);
DEFSO(cublasCtrsm_v2)(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuComplex *alpha, const cuComplex *A, int lda, cuComplex *B, int ldb);
DEFSO(cublasZtrsm_v2)(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, cuDoubleComplex *B, int ldb);
DEFSO(cublasStrsmBatched)(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const float *alpha, const float *const A[], int lda, float *const B[], int ldb, int batchCount);
DEFSO(cublasDtrsmBatched)(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const double *alpha, const double *const A[], int lda, double *const B[], int ldb, int batchCount);
DEFSO(cublasCtrsmBatched)(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuComplex *alpha, const cuComplex *const A[], int lda, cuComplex *const B[], int ldb, int batchCount);
DEFSO(cublasZtrsmBatched)(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *const A[], int lda, cuDoubleComplex *const B[], int ldb, int batchCount);
DEFSO(cublasChemm_v2)(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *B, int ldb, const cuComplex *beta, cuComplex *C, int ldc);
DEFSO(cublasZhemm_v2)(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb, const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc);
DEFSO(cublasCherk_v2)(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const float *alpha, const cuComplex *A, int lda, const float *beta, cuComplex *C, int ldc);
DEFSO(cublasZherk_v2)(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const double *alpha, const cuDoubleComplex *A, int lda, const double *beta, cuDoubleComplex *C, int ldc);
DEFSO(cublasCher2k_v2)(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *B, int ldb, const float *beta, cuComplex *C, int ldc);
DEFSO(cublasZher2k_v2)(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb, const double *beta, cuDoubleComplex *C, int ldc);
DEFSO(cublasCherkx)(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *B, int ldb, const float *beta, cuComplex *C, int ldc);
DEFSO(cublasZherkx)(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb, const double *beta, cuDoubleComplex *C, int ldc);
DEFSO(cublasSgeam)(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, const float *alpha, const float *A, int lda, const float *beta, const float *B, int ldb, float *C, int ldc);
DEFSO(cublasDgeam)(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, const double *alpha, const double *A, int lda, const double *beta, const double *B, int ldb, double *C, int ldc);
DEFSO(cublasCgeam)(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *beta, const cuComplex *B, int ldb, cuComplex *C, int ldc);
DEFSO(cublasZgeam)(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *beta, const cuDoubleComplex *B, int ldb, cuDoubleComplex *C, int ldc);
DEFSO(cublasSdgmm)(cublasHandle_t handle, cublasSideMode_t mode, int m, int n, const float *A, int lda, const float *x, int incx, float *C, int ldc);
DEFSO(cublasDdgmm)(cublasHandle_t handle, cublasSideMode_t mode, int m, int n, const double *A, int lda, const double *x, int incx, double *C, int ldc);
DEFSO(cublasCdgmm)(cublasHandle_t handle, cublasSideMode_t mode, int m, int n, const cuComplex *A, int lda, const cuComplex *x, int incx, cuComplex *C, int ldc);
DEFSO(cublasZdgmm)(cublasHandle_t handle, cublasSideMode_t mode, int m, int n, const cuDoubleComplex *A, int lda, const cuDoubleComplex *x, int incx, cuDoubleComplex *C, int ldc);
DEFSO(cublasSgetrfBatched)(cublasHandle_t handle, int n, float *const Aarray[], int lda, int *PivotArray, int *infoArray, int batchSize);
DEFSO(cublasDgetrfBatched)(cublasHandle_t handle, int n, double *const Aarray[], int lda, int *PivotArray, int *infoArray, int batchSize);
DEFSO(cublasCgetrfBatched)(cublasHandle_t handle, int n, cuComplex *const Aarray[], int lda, int *PivotArray, int *infoArray, int batchSize);
DEFSO(cublasZgetrfBatched)(cublasHandle_t handle, int n, cuDoubleComplex *const Aarray[], int lda, int *PivotArray, int *infoArray, int batchSize);
DEFSO(cublasSgetrsBatched)(cublasHandle_t handle, cublasOperation_t trans, int n, int nrhs, const float *const Aarray[], int lda, const int *devIpiv, float *const Barray[], int ldb, int *info, int batchSize);
DEFSO(cublasDgetrsBatched)(cublasHandle_t handle, cublasOperation_t trans, int n, int nrhs, const double *const Aarray[], int lda, const int *devIpiv, double *const Barray[], int ldb, int *info, int batchSize);
DEFSO(cublasCgetrsBatched)(cublasHandle_t handle, cublasOperation_t trans, int n, int nrhs, const cuComplex *const Aarray[], int lda, const int *devIpiv, cuComplex *const Barray[], int ldb, int *info, int batchSize);
DEFSO(cublasZgetrsBatched)(cublasHandle_t handle, cublasOperation_t trans, int n, int nrhs, const cuDoubleComplex *const Aarray[], int lda, const int *devIpiv, cuDoubleComplex *const Barray[], int ldb, int *info, int batchSize);
DEFSO(cublasSgetriBatched)(cublasHandle_t handle, int n, const float *const Aarray[], int lda, const int *PivotArray, float *const Carray[], int ldc, int *infoArray, int batchSize);
DEFSO(cublasDgetriBatched)(cublasHandle_t handle, int n, const double *const Aarray[], int lda, const int *PivotArray, double *const Carray[], int ldc, int *infoArray, int batchSize);
DEFSO(cublasCgetriBatched)(cublasHandle_t handle, int n, const cuComplex *const Aarray[], int lda, const int *PivotArray, cuComplex *const Carray[], int ldc, int *infoArray, int batchSize);
DEFSO(cublasZgetriBatched)(cublasHandle_t handle, int n, const cuDoubleComplex *const Aarray[], int lda, const int *PivotArray, cuDoubleComplex *const Carray[], int ldc, int *infoArray, int batchSize);
DEFSO(cublasSmatinvBatched)(cublasHandle_t handle, int n, const float *const A[], int lda, float *const Ainv[], int lda_inv, int *info, int batchSize);
DEFSO(cublasDmatinvBatched)(cublasHandle_t handle, int n, const double *const A[], int lda, double *const Ainv[], int lda_inv, int *info, int batchSize);
DEFSO(cublasCmatinvBatched)(cublasHandle_t handle, int n, const cuComplex *const A[], int lda, cuComplex *const Ainv[], int lda_inv, int *info, int batchSize);
DEFSO(cublasZmatinvBatched)(cublasHandle_t handle, int n, const cuDoubleComplex *const A[], int lda, cuDoubleComplex *const Ainv[], int lda_inv, int *info, int batchSize);
DEFSO(cublasSgeqrfBatched)(cublasHandle_t handle, int m, int n, float *const Aarray[], int lda, float *const TauArray[], int *info, int batchSize);
DEFSO(cublasDgeqrfBatched)(cublasHandle_t handle, int m, int n, double *const Aarray[], int lda, double *const TauArray[], int *info, int batchSize);
DEFSO(cublasCgeqrfBatched)(cublasHandle_t handle, int m, int n, cuComplex *const Aarray[], int lda, cuComplex *const TauArray[], int *info, int batchSize);
DEFSO(cublasZgeqrfBatched)(cublasHandle_t handle, int m, int n, cuDoubleComplex *const Aarray[], int lda, cuDoubleComplex *const TauArray[], int *info, int batchSize);
DEFSO(cublasSgelsBatched)(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int nrhs, float *const Aarray[], int lda, float *const Carray[], int ldc, int *info, int *devInfoArray, int batchSize);
DEFSO(cublasDgelsBatched)(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int nrhs, double *const Aarray[], int lda, double *const Carray[], int ldc, int *info, int *devInfoArray, int batchSize);
DEFSO(cublasCgelsBatched)(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int nrhs, cuComplex *const Aarray[], int lda, cuComplex *const Carray[], int ldc, int *info, int *devInfoArray, int batchSize);
DEFSO(cublasZgelsBatched)(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int nrhs, cuDoubleComplex *const Aarray[], int lda, cuDoubleComplex *const Carray[], int ldc, int *info, int *devInfoArray, int batchSize);
DEFSO(cublasStpttr)(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float *AP, float *A, int lda);
DEFSO(cublasDtpttr)(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double *AP, double *A, int lda);
DEFSO(cublasCtpttr)(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex *AP, cuComplex *A, int lda);
DEFSO(cublasZtpttr)(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex *AP, cuDoubleComplex *A, int lda);
DEFSO(cublasStrttp)(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float *A, int lda, float *AP);
DEFSO(cublasDtrttp)(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double *A, int lda, double *AP);
DEFSO(cublasCtrttp)(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex *A, int lda, cuComplex *AP);
DEFSO(cublasZtrttp)(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex *A, int lda, cuDoubleComplex *AP);
DEFSO(cublasSgemmEx)(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha, const void *A, cudaDataType Atype, int lda, const void *B, cudaDataType Btype, int ldb, const float *beta, void *C, cudaDataType Ctype, int ldc);
DEFSO(cublasCgemmEx)(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuComplex *alpha, const void *A, cudaDataType Atype, int lda, const void *B, cudaDataType Btype, int ldb, const cuComplex *beta, void *C, cudaDataType Ctype, int ldc);
DEFSO(cublasGemmEx)(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const void *alpha, const void *A, cudaDataType Atype, int lda, const void *B, cudaDataType Btype, int ldb, const void *beta, void *C, cudaDataType Ctype, int ldc, cudaDataType computeType, cublasGemmAlgo_t algo);
DEFSO(cublasGemmBatchedEx)(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const void *alpha, const void *const Aarray[], cudaDataType Atype, int lda, const void *const Barray[], cudaDataType Btype, int ldb, const void *beta, void *const Carray[], cudaDataType Ctype, int ldc, int batchCount, cudaDataType computeType, cublasGemmAlgo_t algo);
DEFSO(cublasGemmStridedBatchedEx)(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const void *alpha, const void *A, cudaDataType Atype, int lda, long long int strideA, const void *B, cudaDataType Btype, int ldb, long long int strideB, const void *beta, void *C, cudaDataType Ctype, int ldc, long long int strideC, int batchCount, cudaDataType computeType, cublasGemmAlgo_t algo);
DEFSO(cublasCsyrkEx)(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex *alpha, const void *A, cudaDataType Atype, int lda, const cuComplex *beta, void *C, cudaDataType Ctype, int ldc);
DEFSO(cublasCsyrk3mEx)(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex *alpha, const void *A, cudaDataType Atype, int lda, const cuComplex *beta, void *C, cudaDataType Ctype, int ldc);
DEFSO(cublasCherkEx)(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const float *alpha, const void *A, cudaDataType Atype, int lda, const float *beta, void *C, cudaDataType Ctype, int ldc);
DEFSO(cublasCherk3mEx)(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const float *alpha, const void *A, cudaDataType Atype, int lda, const float *beta, void *C, cudaDataType Ctype, int ldc);
DEFSO(cublasNrm2Ex)(cublasHandle_t handle, int n, const void *x, cudaDataType xType, int incx, void *result, cudaDataType resultType, cudaDataType executionType);
DEFSO(cublasAxpyEx)(cublasHandle_t handle, int n, const void *alpha, cudaDataType alphaType, const void *x, cudaDataType xType, int incx, void *y, cudaDataType yType, int incy, cudaDataType executiontype);
DEFSO(cublasDotEx)(cublasHandle_t handle, int n, const void *x, cudaDataType xType, int incx, const void *y, cudaDataType yType, int incy, void *result, cudaDataType resultType, cudaDataType executionType);
DEFSO(cublasDotcEx)(cublasHandle_t handle, int n, const void *x, cudaDataType xType, int incx, const void *y, cudaDataType yType, int incy, void *result, cudaDataType resultType, cudaDataType executionType);
DEFSO(cublasScalEx)(cublasHandle_t handle, int n, const void *alpha, cudaDataType alphaType, void *x, cudaDataType xType, int incx, cudaDataType executionType);
DEFSO(cublasSasum_v2)(cublasHandle_t handle, int n, const float *x, int incx, float  *result);
DEFSO(cublasZswap_v2)(cublasHandle_t handle, int n, cuDoubleComplex *x, int incx, cuDoubleComplex *y, int incy);
DEFSO(cublasCswap_v2)(cublasHandle_t handle, int n, cuComplex *x, int incx, cuComplex *y, int incy);
DEFSO(cublasSswap_v2)(cublasHandle_t handle, int n, float *x, int incx, float *y, int incy);
DEFSO(cublasDswap_v2)(cublasHandle_t handle, int n, double *x, int incx, double *y, int incy);
DEFSO(cublasZdscal_v2)(cublasHandle_t handle, int n, const double *alpha, cuDoubleComplex *x, int incx);
DEFSO(cublasZscal_v2)(cublasHandle_t handle, int n, const cuDoubleComplex *alpha, cuDoubleComplex *x, int incx);
DEFSO(cublasCsscal_v2)(cublasHandle_t handle, int n, const float *alpha, cuComplex *x, int incx);
DEFSO(cublasCscal_v2)(cublasHandle_t handle, int n, const cuComplex *alpha, cuComplex *x, int incx);
DEFSO(cublasDscal_v2)(cublasHandle_t handle, int n, const double *alpha, double *x, int incx);
DEFSO(cublasSscal_v2)(cublasHandle_t handle, int n, const float *alpha, float *x, int incx);
DEFSO(cublasZaxpy_v2)(cublasHandle_t handle, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *x, int incx, cuDoubleComplex *y, int incy);
DEFSO(cublasCaxpy_v2)(cublasHandle_t handle, int n, const cuComplex *alpha, const cuComplex *x, int incx, cuComplex *y, int incy);
DEFSO(cublasDaxpy_v2)(cublasHandle_t handle, int n, const double *alpha, const double *x, int incx, double *y, int incy);
DEFSO(cublasSaxpy_v2)(cublasHandle_t handle, int n, const float *alpha, const float *x, int incx, float *y, int incy);
DEFSO(cublasDzasum_v2)(cublasHandle_t handle, int n, const cuDoubleComplex *x, int incx, double *result);
DEFSO(cublasScasum_v2)(cublasHandle_t handle, int n, const cuComplex *x, int incx, float *result);
DEFSO(cublasDasum_v2)(cublasHandle_t handle, int n, const double *x, int incx, double *result);

static void dlsym_all_funcs() {
    printf("dlsym all funcs\n");

    LDSYM(cublasDasum_v2);
    LDSYM(cublasScasum_v2);
    LDSYM(cublasDzasum_v2);
    LDSYM(cublasSaxpy_v2);
    LDSYM(cublasDaxpy_v2);
    LDSYM(cublasCaxpy_v2);
    LDSYM(cublasZaxpy_v2);
    LDSYM(cublasSscal_v2);
    LDSYM(cublasDscal_v2);
    LDSYM(cublasCscal_v2);
    LDSYM(cublasCsscal_v2);
    LDSYM(cublasZscal_v2);
    LDSYM(cublasZdscal_v2);
    LDSYM(cublasDswap_v2);
    LDSYM(cublasSswap_v2);
    LDSYM(cublasCswap_v2);
    LDSYM(cublasCreate_v2);
    LDSYM(cublasDestroy_v2);
    LDSYM(cublasGetVersion_v2);
    LDSYM(cublasGetProperty);
    LDSYM(cublasSetStream_v2);
    LDSYM(cublasGetStream_v2);
    LDSYM(cublasGetPointerMode_v2);
    LDSYM(cublasSetPointerMode_v2);
    LDSYM(cublasSetVector);
    LDSYM(cublasGetVector);
    LDSYM(cublasSetMatrix);
    LDSYM(cublasGetMatrix);
    LDSYM(cublasSetVectorAsync);
    LDSYM(cublasGetVectorAsync);
    LDSYM(cublasSetMatrixAsync);
    LDSYM(cublasGetMatrixAsync);
    LDSYM(cublasSetAtomicsMode);
    LDSYM(cublasGetAtomicsMode);
    LDSYM(cublasSetMathMode);
    LDSYM(cublasGetMathMode);
    LDSYM(cublasLoggerConfigure);
    LDSYM(cublasGetLoggerCallback);
    LDSYM(cublasSetLoggerCallback);
    LDSYM(cublasIsamax_v2);
    LDSYM(cublasIdamax_v2);
    LDSYM(cublasIcamax_v2);
    LDSYM(cublasIzamax_v2);
    LDSYM(cublasIsamin_v2);
    LDSYM(cublasIdamin_v2);
    LDSYM(cublasIcamin_v2);
    LDSYM(cublasIzamin_v2);
    LDSYM(cublasScopy_v2);
    LDSYM(cublasDcopy_v2);
    LDSYM(cublasCcopy_v2);
    LDSYM(cublasZcopy_v2);
    LDSYM(cublasSdot_v2);
    LDSYM(cublasDdot_v2);
    LDSYM(cublasCdotu_v2);
    LDSYM(cublasCdotc_v2);
    LDSYM(cublasZdotu_v2);
    LDSYM(cublasZdotc_v2);
    LDSYM(cublasSnrm2_v2);
    LDSYM(cublasDnrm2_v2);
    LDSYM(cublasScnrm2_v2);
    LDSYM(cublasDznrm2_v2);
    LDSYM(cublasSrot_v2);
    LDSYM(cublasDrot_v2);
    LDSYM(cublasCrot_v2);
    LDSYM(cublasCsrot_v2);
    LDSYM(cublasZrot_v2);
    LDSYM(cublasZdrot_v2);
    LDSYM(cublasSrotg_v2);
    LDSYM(cublasDrotg_v2);
    LDSYM(cublasCrotg_v2);
    LDSYM(cublasZrotg_v2);
    LDSYM(cublasSrotm_v2);
    LDSYM(cublasDrotm_v2);
    LDSYM(cublasSrotmg_v2);
    LDSYM(cublasDrotmg_v2);
    LDSYM(cublasSgbmv_v2);
    LDSYM(cublasDgbmv_v2);
    LDSYM(cublasCgbmv_v2);
    LDSYM(cublasZgbmv_v2);
    LDSYM(cublasSgemv_v2);
    LDSYM(cublasDgemv_v2);
    LDSYM(cublasCgemv_v2);
    LDSYM(cublasZgemv_v2);
    LDSYM(cublasSger_v2);
    LDSYM(cublasDger_v2);
    LDSYM(cublasCgeru_v2);
    LDSYM(cublasCgerc_v2);
    LDSYM(cublasZgeru_v2);
    LDSYM(cublasZgerc_v2);
    LDSYM(cublasSsbmv_v2);
    LDSYM(cublasDsbmv_v2);
    LDSYM(cublasSspmv_v2);
    LDSYM(cublasDspmv_v2);
    LDSYM(cublasSspr_v2);
    LDSYM(cublasDspr_v2);
    LDSYM(cublasSspr2_v2);
    LDSYM(cublasDspr2_v2);
    LDSYM(cublasSsymv_v2);
    LDSYM(cublasDsymv_v2);
    LDSYM(cublasCsymv_v2);
    LDSYM(cublasZsymv_v2);
    LDSYM(cublasSsyr_v2);
    LDSYM(cublasDsyr_v2);
    LDSYM(cublasCsyr_v2);
    LDSYM(cublasZsyr_v2);
    LDSYM(cublasSsyr2_v2);
    LDSYM(cublasDsyr2_v2);
    LDSYM(cublasCsyr2_v2);
    LDSYM(cublasZsyr2_v2);
    LDSYM(cublasStbmv_v2);
    LDSYM(cublasDtbmv_v2);
    LDSYM(cublasCtbmv_v2);
    LDSYM(cublasZtbmv_v2);
    LDSYM(cublasStbsv_v2);
    LDSYM(cublasDtbsv_v2);
    LDSYM(cublasCtbsv_v2);
    LDSYM(cublasZtbsv_v2);
    LDSYM(cublasStpmv_v2);
    LDSYM(cublasDtpmv_v2);
    LDSYM(cublasCtpmv_v2);
    LDSYM(cublasZtpmv_v2);
    LDSYM(cublasStpsv_v2);
    LDSYM(cublasDtpsv_v2);
    LDSYM(cublasCtpsv_v2);
    LDSYM(cublasZtpsv_v2);
    LDSYM(cublasStrmv_v2);
    LDSYM(cublasDtrmv_v2);
    LDSYM(cublasCtrmv_v2);
    LDSYM(cublasZtrmv_v2);
    LDSYM(cublasStrsv_v2);
    LDSYM(cublasDtrsv_v2);
    LDSYM(cublasCtrsv_v2);
    LDSYM(cublasZtrsv_v2);
    LDSYM(cublasChemv_v2);
    LDSYM(cublasZhemv_v2);
    LDSYM(cublasChbmv_v2);
    LDSYM(cublasZhbmv_v2);
    LDSYM(cublasChpmv_v2);
    LDSYM(cublasZhpmv_v2);
    LDSYM(cublasCher_v2);
    LDSYM(cublasZher_v2);
    LDSYM(cublasCher2_v2);
    LDSYM(cublasZher2_v2);
    LDSYM(cublasChpr_v2);
    LDSYM(cublasZhpr_v2);
    LDSYM(cublasChpr2_v2);
    LDSYM(cublasZhpr2_v2);
    LDSYM(cublasSgemm_v2);
    LDSYM(cublasDgemm_v2);
    LDSYM(cublasCgemm_v2);
    LDSYM(cublasZgemm_v2);
    LDSYM(cublasCgemm3m);
    LDSYM(cublasZgemm3m);
    LDSYM(cublasSgemmBatched);
    LDSYM(cublasDgemmBatched);
    LDSYM(cublasCgemmBatched);
    LDSYM(cublasZgemmBatched);
    LDSYM(cublasSgemmStridedBatched);
    LDSYM(cublasDgemmStridedBatched);
    LDSYM(cublasCgemmStridedBatched);
    LDSYM(cublasZgemmStridedBatched);
    LDSYM(cublasSsymm_v2);
    LDSYM(cublasDsymm_v2);
    LDSYM(cublasCsymm_v2);
    LDSYM(cublasZsymm_v2);
    LDSYM(cublasSsyrk_v2);
    LDSYM(cublasDsyrk_v2);
    LDSYM(cublasCsyrk_v2);
    LDSYM(cublasZsyrk_v2);
    LDSYM(cublasSsyr2k_v2);
    LDSYM(cublasDsyr2k_v2);
    LDSYM(cublasCsyr2k_v2);
    LDSYM(cublasZsyr2k_v2);
    LDSYM(cublasSsyrkx);
    LDSYM(cublasDsyrkx);
    LDSYM(cublasCsyrkx);
    LDSYM(cublasZsyrkx);
    LDSYM(cublasStrmm_v2);
    LDSYM(cublasDtrmm_v2);
    LDSYM(cublasCtrmm_v2);
    LDSYM(cublasZtrmm_v2);
    LDSYM(cublasStrsm_v2);
    LDSYM(cublasDtrsm_v2);
    LDSYM(cublasCtrsm_v2);
    LDSYM(cublasZtrsm_v2);
    LDSYM(cublasStrsmBatched);
    LDSYM(cublasDtrsmBatched);
    LDSYM(cublasCtrsmBatched);
    LDSYM(cublasZtrsmBatched);
    LDSYM(cublasChemm_v2);
    LDSYM(cublasZhemm_v2);
    LDSYM(cublasCherk_v2);
    LDSYM(cublasZherk_v2);
    LDSYM(cublasCher2k_v2);
    LDSYM(cublasZher2k_v2);
    LDSYM(cublasCherkx);
    LDSYM(cublasZherkx);
    LDSYM(cublasSgeam);
    LDSYM(cublasDgeam);
    LDSYM(cublasCgeam);
    LDSYM(cublasZgeam);
    LDSYM(cublasSdgmm);
    LDSYM(cublasDdgmm);
    LDSYM(cublasCdgmm);
    LDSYM(cublasZdgmm);
    LDSYM(cublasSgetrfBatched);
    LDSYM(cublasDgetrfBatched);
    LDSYM(cublasCgetrfBatched);
    LDSYM(cublasZgetrfBatched);
    LDSYM(cublasSgetrsBatched);
    LDSYM(cublasDgetrsBatched);
    LDSYM(cublasCgetrsBatched);
    LDSYM(cublasZgetrsBatched);
    LDSYM(cublasSgetriBatched);
    LDSYM(cublasDgetriBatched);
    LDSYM(cublasCgetriBatched);
    LDSYM(cublasZgetriBatched);
    LDSYM(cublasSmatinvBatched);
    LDSYM(cublasDmatinvBatched);
    LDSYM(cublasCmatinvBatched);
    LDSYM(cublasZmatinvBatched);
    LDSYM(cublasSgeqrfBatched);
    LDSYM(cublasDgeqrfBatched);
    LDSYM(cublasCgeqrfBatched);
    LDSYM(cublasZgeqrfBatched);
    LDSYM(cublasSgelsBatched);
    LDSYM(cublasDgelsBatched);
    LDSYM(cublasCgelsBatched);
    LDSYM(cublasZgelsBatched);
    LDSYM(cublasStpttr);
    LDSYM(cublasDtpttr);
    LDSYM(cublasCtpttr);
    LDSYM(cublasZtpttr);
    LDSYM(cublasStrttp);
    LDSYM(cublasDtrttp);
    LDSYM(cublasCtrttp);
    LDSYM(cublasZtrttp);
    LDSYM(cublasSgemmEx);
    LDSYM(cublasCgemmEx);
    LDSYM(cublasGemmEx);
    LDSYM(cublasGemmBatchedEx);
    LDSYM(cublasGemmStridedBatchedEx);
    LDSYM(cublasCsyrkEx);
    LDSYM(cublasCsyrk3mEx);
    LDSYM(cublasCherkEx);
    LDSYM(cublasCherk3mEx);
    LDSYM(cublasNrm2Ex);
    LDSYM(cublasAxpyEx);
    LDSYM(cublasDotEx);
    LDSYM(cublasDotcEx);
    LDSYM(cublasScalEx);
    LDSYM(cublasSasum_v2);
    LDSYM(cublasZswap_v2);
    printf("blas dlsym all funcs end\n");
}

__attribute ((constructor)) void cublas_init(void) {
    printf("cublas_init\n");
    so_handle = dlopen (LIB_STRING, RTLD_NOW);
    if (!so_handle) {
        fprintf(stderr, "%s\n", dlerror());
        exit(1);
    }
    dlsym_all_funcs();
    cudaw_so_register_dli(&so_dli);
    void * pp_for_trace[] = {
        FSWAP(cublasSgemm_v2)
        FSWAP(cublasSgemv_v2)
    };
    cudawblas_so_func_swap(pp_for_trace);
}

__attribute ((destructor)) void cublas_fini(void) {
    printf("cublas_fini\n");
    if (so_handle) {
        dlclose(so_handle);
    }
    for (int k = 1; k <= so_dli.func_num; ++k) {
        if (so_funcs[k].cnt == 0)
            continue;
        printf("%5d %10lu : %s\n", k, so_funcs[k].cnt, so_funcs[k].func_name);
    }
}

#define checkCublasErrors(err)  __checkCublasErrors (err, __FILE__, __LINE__)
static cublasStatus_t __checkCublasErrors(cublasStatus_t err, const char *file, const int line) {
    if( CUDA_SUCCESS != err) {
        fprintf(stderr,
                "CUDA Blas API error = %04d from file <%s>, line %i, function \n",
                err, file, line );
        //exit(-1);
    }
    return err;
}

cublasStatus_t cublasCreate_v2(cublasHandle_t *handle) {
    cublasStatus_t r;
    begin_func(cublasCreate_v2);
    r = so_cublasCreate_v2(handle);
    end_func(cublasCreate_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasDestroy_v2(cublasHandle_t handle) {
    cublasStatus_t r;
    begin_func(cublasDestroy_v2);
    r = so_cublasDestroy_v2(handle);
    end_func(cublasDestroy_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasGetVersion_v2(cublasHandle_t handle, int *version) {
    cublasStatus_t r;
    begin_func(cublasGetVersion_v2);
    r = so_cublasGetVersion_v2(handle, version);
    end_func(cublasGetVersion_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasGetProperty(libraryPropertyType type, int *value) {
    cublasStatus_t r;
    begin_func(cublasGetProperty);
    r = so_cublasGetProperty(type, value);
    end_func(cublasGetProperty);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasSetStream_v2(cublasHandle_t handle, cudaStream_t streamId) {
    cublasStatus_t r;
    begin_func(cublasSetStream_v2);
    r = so_cublasSetStream_v2(handle, streamId);
    end_func(cublasSetStream_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasGetStream_v2(cublasHandle_t handle, cudaStream_t *streamId) {
    cublasStatus_t r;
    begin_func(cublasGetStream_v2);
    r = so_cublasGetStream_v2(handle, streamId);
    end_func(cublasGetStream_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasGetPointerMode_v2(cublasHandle_t handle, cublasPointerMode_t *mode) {
    cublasStatus_t r;
    begin_func(cublasGetPointerMode_v2);
    r = so_cublasGetPointerMode_v2(handle, mode);
    end_func(cublasGetPointerMode_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasSetPointerMode_v2(cublasHandle_t handle, cublasPointerMode_t mode) {
    cublasStatus_t r;
    begin_func(cublasSetPointerMode_v2);
    r = so_cublasSetPointerMode_v2(handle, mode);
    end_func(cublasSetPointerMode_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasSetVector(int n, int elemSize, const void *x, int incx, void *y, int incy) {
    cublasStatus_t r;
    begin_func(cublasSetVector);
    r = so_cublasSetVector(n, elemSize, x, incx, y, incy);
    end_func(cublasSetVector);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasGetVector(int n, int elemSize, const void *x, int incx, void *y, int incy) {
    cublasStatus_t r;
    begin_func(cublasGetVector);
    r = so_cublasGetVector(n, elemSize, x, incx, y, incy);
    end_func(cublasGetVector);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasSetMatrix(int rows, int cols, int elemSize, const void *A, int lda, void *B, int ldb) {
    cublasStatus_t r;
    begin_func(cublasSetMatrix);
    r = so_cublasSetMatrix(rows, cols, elemSize, A, lda, B, ldb);
    end_func(cublasSetMatrix);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasGetMatrix(int rows, int cols, int elemSize, const void *A, int lda, void *B, int ldb) {
    cublasStatus_t r;
    begin_func(cublasGetMatrix);
    r = so_cublasGetMatrix(rows, cols, elemSize, A, lda, B, ldb);
    end_func(cublasGetMatrix);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasSetVectorAsync(int n, int elemSize, const void *hostPtr, int incx, void *devicePtr, int incy, cudaStream_t stream) {
    cublasStatus_t r;
    begin_func(cublasSetVectorAsync);
    r = so_cublasSetVectorAsync(n, elemSize, hostPtr, incx, devicePtr, incy, stream);
    end_func(cublasSetVectorAsync);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasGetVectorAsync(int n, int elemSize, const void *devicePtr, int incx, void *hostPtr, int incy, cudaStream_t stream) {
    cublasStatus_t r;
    begin_func(cublasGetVectorAsync);
    r = so_cublasGetVectorAsync(n, elemSize, devicePtr, incx, hostPtr, incy, stream);
    end_func(cublasGetVectorAsync);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasSetMatrixAsync(int rows, int cols, int elemSize, const void *A, int lda, void *B, int ldb, cudaStream_t stream) {
    cublasStatus_t r;
    begin_func(cublasSetMatrixAsync);
    r = so_cublasSetMatrixAsync(rows, cols, elemSize, A, lda, B, ldb, stream);
    end_func(cublasSetMatrixAsync);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasGetMatrixAsync(int rows, int cols, int elemSize, const void *A, int lda, void *B, int ldb, cudaStream_t stream) {
    cublasStatus_t r;
    begin_func(cublasGetMatrixAsync);
    r = so_cublasGetMatrixAsync(rows, cols, elemSize, A, lda, B, ldb, stream);
    end_func(cublasGetMatrixAsync);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasSetAtomicsMode(cublasHandle_t handle, cublasAtomicsMode_t mode) {
    cublasStatus_t r;
    begin_func(cublasSetAtomicsMode);
    r = so_cublasSetAtomicsMode(handle, mode);
    end_func(cublasSetAtomicsMode);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasGetAtomicsMode(cublasHandle_t handle, cublasAtomicsMode_t *mode) {
    cublasStatus_t r;
    begin_func(cublasGetAtomicsMode);
    r = so_cublasGetAtomicsMode(handle, mode);
    end_func(cublasGetAtomicsMode);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasSetMathMode(cublasHandle_t handle, cublasMath_t mode) {
    cublasStatus_t r;
    begin_func(cublasSetMathMode);
    r = so_cublasSetMathMode(handle, mode);
    end_func(cublasSetMathMode);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasGetMathMode(cublasHandle_t handle, cublasMath_t *mode) {
    cublasStatus_t r;
    begin_func(cublasGetMathMode);
    r = so_cublasGetMathMode(handle, mode);
    end_func(cublasGetMathMode);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasLoggerConfigure(int logIsOn, int logToStdOut, int logToStdErr, const char* logFileName) {
    cublasStatus_t r;
    begin_func(cublasLoggerConfigure);
    r = so_cublasLoggerConfigure(logIsOn, logToStdOut, logToStdErr, logFileName);
    end_func(cublasLoggerConfigure);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasGetLoggerCallback(cublasLogCallback* userCallback) {
    cublasStatus_t r;
    begin_func(cublasGetLoggerCallback);
    r = so_cublasGetLoggerCallback(userCallback );
    end_func(cublasGetLoggerCallback);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasSetLoggerCallback(cublasLogCallback userCallback) {
    cublasStatus_t r;
    begin_func(cublasSetLoggerCallback);
    r = so_cublasSetLoggerCallback(userCallback);
    end_func(cublasSetLoggerCallback);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasIsamax_v2(cublasHandle_t handle, int n, const float *x, int incx, int *result) {
    cublasStatus_t r;
    begin_func(cublasIsamax_v2);
    r = so_cublasIsamax_v2(handle, n, x, incx, result);
    end_func(cublasIsamax_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasIdamax_v2(cublasHandle_t handle, int n, const double *x, int incx, int *result) {
    cublasStatus_t r;
    begin_func(cublasIdamax_v2);
    r = so_cublasIdamax_v2(handle, n, x, incx, result);
    end_func(cublasIdamax_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasIcamax_v2(cublasHandle_t handle, int n, const cuComplex *x, int incx, int *result) {
    cublasStatus_t r;
    begin_func(cublasIcamax_v2);
    r = so_cublasIcamax_v2(handle, n, x, incx, result);
    end_func(cublasIcamax_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasIzamax_v2(cublasHandle_t handle, int n, const cuDoubleComplex *x, int incx, int *result) {
    cublasStatus_t r;
    begin_func(cublasIzamax_v2);
    r = so_cublasIzamax_v2(handle, n, x, incx, result);
    end_func(cublasIzamax_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasIsamin_v2(cublasHandle_t handle, int n, const float *x, int incx, int *result) {
    cublasStatus_t r;
    begin_func(cublasIsamin_v2);
    r = so_cublasIsamin_v2(handle, n, x, incx, result );
    end_func(cublasIsamin_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasIdamin_v2(cublasHandle_t handle, int n, const double *x, int incx, int *result) {
    cublasStatus_t r;
    begin_func(cublasIdamin_v2);
    r = so_cublasIdamin_v2(handle, n, x, incx, result );
    end_func(cublasIdamin_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasIcamin_v2(cublasHandle_t handle, int n, const cuComplex *x, int incx, int *result) {
    cublasStatus_t r;
    begin_func(cublasIcamin_v2);
    r = so_cublasIcamin_v2(handle, n, x, incx, result );
    end_func(cublasIcamin_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasIzamin_v2(cublasHandle_t handle, int n, const cuDoubleComplex *x, int incx, int *result) {
    cublasStatus_t r;
    begin_func(cublasIzamin_v2);
    r = so_cublasIzamin_v2(handle, n, x, incx, result );
    end_func(cublasIzamin_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasSasum_v2(cublasHandle_t handle, int n, const float *x, int incx, float  *result) {
    cublasStatus_t r;
    begin_func(cublasSasum_v2);
    r = so_cublasSasum_v2(handle, n, x, incx, result );
    end_func(cublasSasum_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasDasum_v2(cublasHandle_t handle, int n, const double *x, int incx, double *result) {
    cublasStatus_t r;
    begin_func(cublasDasum_v2);
    r = so_cublasDasum_v2(handle, n, x, incx, result );
    end_func(cublasDasum_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasScasum_v2(cublasHandle_t handle, int n, const cuComplex *x, int incx, float *result) {
    cublasStatus_t r;
    begin_func(cublasScasum_v2);
    r = so_cublasScasum_v2(handle, n, x, incx, result );
    end_func(cublasScasum_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasDzasum_v2(cublasHandle_t handle, int n, const cuDoubleComplex *x, int incx, double *result) {
    cublasStatus_t r;
    begin_func(cublasDzasum_v2);
    r = so_cublasDzasum_v2(handle, n, x, incx, result );
    end_func(cublasDzasum_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasSaxpy_v2(cublasHandle_t handle, int n, const float *alpha, const float *x, int incx, float *y, int incy) {
    cublasStatus_t r;
    begin_func(cublasSaxpy_v2);
    r = so_cublasSaxpy_v2(handle, n, alpha, x, incx, y, incy);
    end_func(cublasSaxpy_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasDaxpy_v2(cublasHandle_t handle, int n, const double *alpha, const double *x, int incx, double *y, int incy) {
    cublasStatus_t r;
    begin_func(cublasDaxpy_v2);
    r = so_cublasDaxpy_v2(handle, n, alpha, x, incx, y, incy);
    end_func(cublasDaxpy_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasCaxpy_v2(cublasHandle_t handle, int n, const cuComplex *alpha, const cuComplex *x, int incx, cuComplex *y, int incy) {
    cublasStatus_t r;
    begin_func(cublasCaxpy_v2);
    r = so_cublasCaxpy_v2(handle, n, alpha, x, incx, y, incy);
    end_func(cublasCaxpy_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasZaxpy_v2(cublasHandle_t handle, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *x, int incx, cuDoubleComplex *y, int incy) {
    cublasStatus_t r;
    begin_func(cublasZaxpy_v2);
    r = so_cublasZaxpy_v2(handle, n, alpha, x, incx, y, incy);
    end_func(cublasZaxpy_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasScopy_v2(cublasHandle_t handle, int n, const float *x, int incx, float *y, int incy) {
    cublasStatus_t r;
    begin_func(cublasScopy_v2);
    r = so_cublasScopy_v2(handle, n, x, incx, y, incy);
    end_func(cublasScopy_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasDcopy_v2(cublasHandle_t handle, int n, const double *x, int incx, double *y, int incy) {
    cublasStatus_t r;
    begin_func(cublasDcopy_v2);
    r = so_cublasDcopy_v2(handle, n, x, incx, y, incy);
    end_func(cublasDcopy_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasCcopy_v2(cublasHandle_t handle, int n, const cuComplex *x, int incx, cuComplex *y, int incy) {
    cublasStatus_t r;
    begin_func(cublasCcopy_v2);
    r = so_cublasCcopy_v2(handle, n, x, incx, y, incy);
    end_func(cublasCcopy_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasZcopy_v2(cublasHandle_t handle, int n, const cuDoubleComplex *x, int incx, cuDoubleComplex *y, int incy) {
    cublasStatus_t r;
    begin_func(cublasZcopy_v2);
    r = so_cublasZcopy_v2(handle, n, x, incx, y, incy);
    end_func(cublasZcopy_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasSdot_v2(cublasHandle_t handle, int n, const float *x, int incx, const float *y, int incy, float *result) {
    cublasStatus_t r;
    begin_func(cublasSdot_v2);
    r = so_cublasSdot_v2(handle, n, x, incx, y, incy, result);
    end_func(cublasSdot_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasDdot_v2(cublasHandle_t handle, int n, const double *x, int incx, const double *y, int incy, double *result) {
    cublasStatus_t r;
    begin_func(cublasDdot_v2);
    r = so_cublasDdot_v2(handle, n, x, incx, y, incy, result);
    end_func(cublasDdot_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasCdotu_v2(cublasHandle_t handle, int n, const cuComplex *x, int incx, const cuComplex *y, int incy, cuComplex *result) {
    cublasStatus_t r;
    begin_func(cublasCdotu_v2);
    r = so_cublasCdotu_v2(handle, n, x, incx, y, incy, result);
    end_func(cublasCdotu_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasCdotc_v2(cublasHandle_t handle, int n, const cuComplex *x, int incx, const cuComplex *y, int incy, cuComplex *result) {
    cublasStatus_t r;
    begin_func(cublasCdotc_v2);
    r = so_cublasCdotc_v2(handle, n, x, incx, y, incy, result);
    end_func(cublasCdotc_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasZdotu_v2(cublasHandle_t handle, int n, const cuDoubleComplex *x, int incx, const cuDoubleComplex *y, int incy, cuDoubleComplex *result) {
    cublasStatus_t r;
    begin_func(cublasZdotu_v2);
    r = so_cublasZdotu_v2(handle, n, x, incx, y, incy, result);
    end_func(cublasZdotu_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasZdotc_v2(cublasHandle_t handle, int n, const cuDoubleComplex *x, int incx, const cuDoubleComplex *y, int incy, cuDoubleComplex *result) {
    cublasStatus_t r;
    begin_func(cublasZdotc_v2);
    r = so_cublasZdotc_v2(handle, n, x, incx, y, incy, result);
    end_func(cublasZdotc_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasSnrm2_v2(cublasHandle_t handle, int n, const float *x, int incx, float *result) {
    cublasStatus_t r;
    begin_func(cublasSnrm2_v2);
    r = so_cublasSnrm2_v2(handle, n, x, incx, result);
    end_func(cublasSnrm2_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasDnrm2_v2(cublasHandle_t handle, int n, const double *x, int incx, double *result) {
    cublasStatus_t r;
    begin_func(cublasDnrm2_v2);
    r = so_cublasDnrm2_v2(handle, n, x, incx, result);
    end_func(cublasDnrm2_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasScnrm2_v2(cublasHandle_t handle, int n, const cuComplex *x, int incx, float *result) {
    cublasStatus_t r;
    begin_func(cublasScnrm2_v2);
    r = so_cublasScnrm2_v2(handle, n, x, incx, result);
    end_func(cublasScnrm2_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasDznrm2_v2(cublasHandle_t handle, int n, const cuDoubleComplex *x, int incx, double *result) {
    cublasStatus_t r;
    begin_func(cublasDznrm2_v2);
    r = so_cublasDznrm2_v2(handle, n, x, incx, result);
    end_func(cublasDznrm2_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasSrot_v2(cublasHandle_t handle, int n, float *x, int incx, float *y, int incy, const float *c, const float *s) {
    cublasStatus_t r;
    begin_func(cublasSrot_v2);
    r = so_cublasSrot_v2(handle, n, x, incx, y, incy, c, s);
    end_func(cublasSrot_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasDrot_v2(cublasHandle_t handle, int n, double *x, int incx, double *y, int incy, const double *c, const double *s) {
    cublasStatus_t r;
    begin_func(cublasDrot_v2);
    r = so_cublasDrot_v2(handle, n, x, incx, y, incy, c, s);
    end_func(cublasDrot_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasCrot_v2(cublasHandle_t handle, int n, cuComplex *x, int incx, cuComplex *y, int incy, const float *c, const cuComplex *s) {
    cublasStatus_t r;
    begin_func(cublasCrot_v2);
    r = so_cublasCrot_v2(handle, n, x, incx, y, incy, c, s);
    end_func(cublasCrot_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasCsrot_v2(cublasHandle_t handle, int n, cuComplex *x, int incx, cuComplex *y, int incy, const float *c, const float *s) {
    cublasStatus_t r;
    begin_func(cublasCsrot_v2);
    r = so_cublasCsrot_v2(handle, n, x, incx, y, incy, c, s);
    end_func(cublasCsrot_v2);
    checkCublasErrors(r);
    return r;
}


cublasStatus_t cublasZrot_v2(cublasHandle_t handle, int n, cuDoubleComplex *x, int incx, cuDoubleComplex *y, int incy, const double *c, const cuDoubleComplex *s) {
    cublasStatus_t r;
    begin_func(cublasZrot_v2);
    r = so_cublasZrot_v2(handle, n, x, incx, y, incy, c, s);
    end_func(cublasZrot_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasZdrot_v2(cublasHandle_t handle, int n, cuDoubleComplex *x, int incx, cuDoubleComplex *y, int incy, const double *c, const double *s) {
    cublasStatus_t r;
    begin_func(cublasZdrot_v2);
    r = so_cublasZdrot_v2(handle, n, x, incx, y, incy, c, s);
    end_func(cublasZdrot_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasSrotg_v2(cublasHandle_t handle, float *a, float *b, float *c, float *s) {
    cublasStatus_t r;
    begin_func(cublasSrotg_v2);
    r = so_cublasSrotg_v2(handle, a, b, c, s);
    end_func(cublasSrotg_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasDrotg_v2(cublasHandle_t handle, double *a, double *b, double *c, double *s) {
    cublasStatus_t r;
    begin_func(cublasDrotg_v2);
    r = so_cublasDrotg_v2(handle, a, b, c, s);
    end_func(cublasDrotg_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasCrotg_v2(cublasHandle_t handle, cuComplex *a, cuComplex *b, float *c, cuComplex *s) {
    cublasStatus_t r;
    begin_func(cublasCrotg_v2);
    r = so_cublasCrotg_v2(handle, a, b, c, s);
    end_func(cublasCrotg_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasZrotg_v2(cublasHandle_t handle, cuDoubleComplex *a, cuDoubleComplex *b, double *c, cuDoubleComplex *s) {
    cublasStatus_t r;
    begin_func(cublasZrotg_v2);
    r = so_cublasZrotg_v2(handle, a, b, c, s);
    end_func(cublasZrotg_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasSrotm_v2(cublasHandle_t handle, int n, float *x, int incx, float *y, int incy, const float* param) {
    cublasStatus_t r;
    begin_func(cublasSrotm_v2);
    r = so_cublasSrotm_v2(handle, n, x, incx, y, incy, param);
    end_func(cublasSrotm_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasDrotm_v2(cublasHandle_t handle, int n, double *x, int incx, double *y, int incy, const double* param) {
    cublasStatus_t r;
    begin_func(cublasDrotm_v2);
    r = so_cublasDrotm_v2(handle, n, x, incx, y, incy, param);
    end_func(cublasDrotm_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasSrotmg_v2(cublasHandle_t handle, float *d1, float *d2, float *x1, const float *y1, float* param) {
    cublasStatus_t r;
    begin_func(cublasSrotmg_v2);
    r = so_cublasSrotmg_v2(handle, d1, d2, x1, y1, param);
    end_func(cublasSrotmg_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasDrotmg_v2(cublasHandle_t handle, double *d1, double *d2, double *x1, const double *y1, double* param) {
    cublasStatus_t r;
    begin_func(cublasDrotmg_v2);
    r = so_cublasDrotmg_v2(handle, d1, d2, x1, y1, param);
    end_func(cublasDrotmg_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasSscal_v2(cublasHandle_t handle, int n, const float *alpha, float *x, int incx) {
    cublasStatus_t r;
    begin_func(cublasSscal_v2);
    r = so_cublasSscal_v2(handle, n, alpha, x, incx);
    end_func(cublasSscal_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasDscal_v2(cublasHandle_t handle, int n, const double *alpha, double *x, int incx) {
    cublasStatus_t r;
    begin_func(cublasDscal_v2);
    r = so_cublasDscal_v2(handle, n, alpha, x, incx);
    end_func(cublasDscal_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasCscal_v2(cublasHandle_t handle, int n, const cuComplex *alpha, cuComplex *x, int incx) {
    cublasStatus_t r;
    begin_func(cublasCscal_v2);
    r = so_cublasCscal_v2(handle, n, alpha, x, incx);
    end_func(cublasCscal_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasCsscal_v2(cublasHandle_t handle, int n, const float *alpha, cuComplex *x, int incx) {
    cublasStatus_t r;
    begin_func(cublasCsscal_v2);
    r = so_cublasCsscal_v2(handle, n, alpha, x, incx);
    end_func(cublasCsscal_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasZscal_v2(cublasHandle_t handle, int n, const cuDoubleComplex *alpha, cuDoubleComplex *x, int incx) {
    cublasStatus_t r;
    begin_func(cublasZscal_v2);
    r = so_cublasZscal_v2(handle, n, alpha, x, incx);
    end_func(cublasZscal_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasZdscal_v2(cublasHandle_t handle, int n, const double *alpha, cuDoubleComplex *x, int incx) {
    cublasStatus_t r;
    begin_func(cublasZdscal_v2);
    r = so_cublasZdscal_v2(handle, n, alpha, x, incx);
    end_func(cublasZdscal_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasSswap_v2(cublasHandle_t handle, int n, float *x, int incx, float *y, int incy) {
    cublasStatus_t r;
    begin_func(cublasSswap_v2);
    r = so_cublasSswap_v2(handle, n, x, incx, y, incy);
    end_func(cublasSswap_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasDswap_v2(cublasHandle_t handle, int n, double *x, int incx, double *y, int incy) {
    cublasStatus_t r;
    begin_func(cublasDswap_v2);
    r = so_cublasDswap_v2(handle, n, x, incx, y, incy);
    end_func(cublasDswap_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasCswap_v2(cublasHandle_t handle, int n, cuComplex *x, int incx, cuComplex *y, int incy) {
    cublasStatus_t r;
    begin_func(cublasCswap_v2);
    r = so_cublasCswap_v2(handle, n, x, incx, y, incy);
    end_func(cublasCswap_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasZswap_v2(cublasHandle_t handle, int n, cuDoubleComplex *x, int incx, cuDoubleComplex *y, int incy) {
    cublasStatus_t r;
    begin_func(cublasZswap_v2);
    r = so_cublasZswap_v2(handle, n, x, incx, y, incy);
    end_func(cublasZswap_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasSgbmv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int kl, int ku, const float *alpha, const float *A, int lda, const float *x, int incx, const float *beta, float *y, int incy) {
    cublasStatus_t r;
    begin_func(cublasSgbmv_v2);
    r = so_cublasSgbmv_v2(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy);
    end_func(cublasSgbmv_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasDgbmv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int kl, int ku, const double *alpha, const double *A, int lda, const double *x, int incx, const double *beta, double *y, int incy) {
    cublasStatus_t r;
    begin_func(cublasDgbmv_v2);
    r = so_cublasDgbmv_v2(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy);
    end_func(cublasDgbmv_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasCgbmv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int kl, int ku, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *x, int incx, const cuComplex *beta, cuComplex *y, int incy) {
    cublasStatus_t r;
    begin_func(cublasCgbmv_v2);
    r = so_cublasCgbmv_v2(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy);
    end_func(cublasCgbmv_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasZgbmv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int kl, int ku, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *x, int incx, const cuDoubleComplex *beta, cuDoubleComplex *y, int incy) {
    cublasStatus_t r;
    begin_func(cublasZgbmv_v2);
    r = so_cublasZgbmv_v2(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy);
    end_func(cublasZgbmv_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasSgemv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const float *alpha, const float *A, int lda, const float *x, int incx, const float *beta, float *y, int incy) {
    cublasStatus_t r;
    begin_func(cublasSgemv_v2);
    r = so_cublasSgemv_v2(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
    end_func(cublasSgemv_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasDgemv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const double *alpha, const double *A, int lda, const double *x, int incx, const double *beta, double *y, int incy) {
    cublasStatus_t r;
    begin_func(cublasDgemv_v2);
    r = so_cublasDgemv_v2(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
    end_func(cublasDgemv_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasCgemv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *x, int incx, const cuComplex *beta, cuComplex *y, int incy) {
    cublasStatus_t r;
    begin_func(cublasCgemv_v2);
    r = so_cublasCgemv_v2(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
    end_func(cublasCgemv_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasZgemv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *x, int incx, const cuDoubleComplex *beta, cuDoubleComplex *y, int incy) {
    cublasStatus_t r;
    begin_func(cublasZgemv_v2);
    r = so_cublasZgemv_v2(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
    end_func(cublasZgemv_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasSger_v2(cublasHandle_t handle, int m, int n, const float *alpha, const float *x, int incx, const float *y, int incy, float *A, int lda) {
    cublasStatus_t r;
    begin_func(cublasSger_v2);
    r = so_cublasSger_v2(handle, m, n, alpha, x, incx, y, incy, A, lda);
    end_func(cublasSger_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasDger_v2(cublasHandle_t handle, int m, int n, const double *alpha, const double *x, int incx, const double *y, int incy, double *A, int lda) {
    cublasStatus_t r;
    begin_func(cublasDger_v2);
    r = so_cublasDger_v2(handle, m, n, alpha, x, incx, y, incy, A, lda);
    end_func(cublasDger_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasCgeru_v2(cublasHandle_t handle, int m, int n, const cuComplex *alpha, const cuComplex *x, int incx, const cuComplex *y, int incy, cuComplex *A, int lda) {
    cublasStatus_t r;
    begin_func(cublasCgeru_v2);
    r = so_cublasCgeru_v2(handle, m, n, alpha, x, incx, y, incy, A, lda);
    end_func(cublasCgeru_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasCgerc_v2(cublasHandle_t handle, int m, int n, const cuComplex *alpha, const cuComplex *x, int incx, const cuComplex *y, int incy, cuComplex *A, int lda) {
    cublasStatus_t r;
    begin_func(cublasCgerc_v2);
    r = so_cublasCgerc_v2(handle, m, n, alpha, x, incx, y, incy, A, lda);
    end_func(cublasCgerc_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasZgeru_v2(cublasHandle_t handle, int m, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *x, int incx, const cuDoubleComplex *y, int incy, cuDoubleComplex *A, int lda) {
    cublasStatus_t r;
    begin_func(cublasZgeru_v2);
    r = so_cublasZgeru_v2(handle, m, n, alpha, x, incx, y, incy, A, lda);
    end_func(cublasZgeru_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasZgerc_v2(cublasHandle_t handle, int m, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *x, int incx, const cuDoubleComplex *y, int incy, cuDoubleComplex *A, int lda) {
    cublasStatus_t r;
    begin_func(cublasZgerc_v2);
    r = so_cublasZgerc_v2(handle, m, n, alpha, x, incx, y, incy, A, lda);
    end_func(cublasZgerc_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasSsbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, int k, const float *alpha, const float  *A, int lda, const float  *x, int incx, const float  *beta, float *y, int incy) {
    cublasStatus_t r;
    begin_func(cublasSsbmv_v2);
    r = so_cublasSsbmv_v2(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy);
    end_func(cublasSsbmv_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasDsbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, int k, const double *alpha, const double  *A, int lda, const double  *x, int incx, const double  *beta, double *y, int incy) {
    cublasStatus_t r;
    begin_func(cublasDsbmv_v2);
    r = so_cublasDsbmv_v2(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy);
    end_func(cublasDsbmv_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasSspmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float *alpha, const float  *AP, const float  *x, int incx, const float *beta, float *y, int incy) {
    cublasStatus_t r;
    begin_func(cublasSspmv_v2);
    r = so_cublasSspmv_v2(handle, uplo, n, alpha, AP, x, incx, beta, y, incy);
    end_func(cublasSspmv_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasDspmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double *alpha, const double  *AP, const double  *x, int incx, const double *beta, double *y, int incy) {
    cublasStatus_t r;
    begin_func(cublasDspmv_v2);
    r = so_cublasDspmv_v2(handle, uplo, n, alpha, AP, x, incx, beta, y, incy);
    end_func(cublasDspmv_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasSspr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float *alpha, const float *x, int incx, float  *AP) {
    cublasStatus_t r;
    begin_func(cublasSspr_v2);
    r = so_cublasSspr_v2(handle, uplo, n, alpha, x, incx, AP);
    end_func(cublasSspr_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasDspr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double *alpha, const double *x, int incx, double  *AP) {
    cublasStatus_t r;
    begin_func(cublasDspr_v2);
    r = so_cublasDspr_v2(handle, uplo, n, alpha, x, incx, AP);
    end_func(cublasDspr_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasSspr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float *alpha, const float  *x, int incx, const float *y, int incy, float* AP) {
    cublasStatus_t r;
    begin_func(cublasSspr2_v2);
    r = so_cublasSspr2_v2(handle, uplo, n, alpha, x, incx, y, incy, AP);
    end_func(cublasSspr2_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasDspr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double *alpha, const double  *x, int incx, const double *y, int incy, double* AP) {
    cublasStatus_t r;
    begin_func(cublasDspr2_v2);
    r = so_cublasDspr2_v2(handle, uplo, n, alpha, x, incx, y, incy, AP);
    end_func(cublasDspr2_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasSsymv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float *alpha, const float *A, int lda, const float  *x, int incx, const float *beta, float *y, int incy) {
    cublasStatus_t r;
    begin_func(cublasSsymv_v2);
    r = so_cublasSsymv_v2(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy);
    end_func(cublasSsymv_v2);
    checkCublasErrors(r);
    return r;
}


cublasStatus_t cublasDsymv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double *alpha, const double *A, int lda, const double  *x, int incx, const double *beta, double *y, int incy) {
    cublasStatus_t r;
    begin_func(cublasDsymv_v2);
    r = so_cublasDsymv_v2(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy);
    end_func(cublasDsymv_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasCsymv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex  *x, int incx, const cuComplex *beta, cuComplex *y, int incy) {
    cublasStatus_t r;
    begin_func(cublasCsymv_v2);
    r = so_cublasCsymv_v2(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy);
    end_func(cublasCsymv_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasZsymv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex  *x, int incx, const cuDoubleComplex *beta, cuDoubleComplex *y, int incy) {
    cublasStatus_t r;
    begin_func(cublasZsymv_v2);
    r = so_cublasZsymv_v2(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy);
    end_func(cublasZsymv_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasSsyr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float *alpha, const float  *x, int incx, float *A, int lda) {
    cublasStatus_t r;
    begin_func(cublasSsyr_v2);
    r = so_cublasSsyr_v2(handle, uplo, n, alpha, x, incx, A, lda);
    end_func(cublasSsyr_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasDsyr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double *alpha, const double  *x, int incx, double *A, int lda) {
    cublasStatus_t r;
    begin_func(cublasDsyr_v2);
    r = so_cublasDsyr_v2(handle, uplo, n, alpha, x, incx, A, lda);
    end_func(cublasDsyr_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasCsyr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex *alpha, const cuComplex  *x, int incx, cuComplex *A, int lda) {
    cublasStatus_t r;
    begin_func(cublasCsyr_v2);
    r = so_cublasCsyr_v2(handle, uplo, n, alpha, x, incx, A, lda);
    end_func(cublasCsyr_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasZsyr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex *alpha, const cuDoubleComplex  *x, int incx, cuDoubleComplex *A, int lda) {
    cublasStatus_t r;
    begin_func(cublasZsyr_v2);
    r = so_cublasZsyr_v2(handle, uplo, n, alpha, x, incx, A, lda);
    end_func(cublasZsyr_v2);
    checkCublasErrors(r);
    return r;
}


cublasStatus_t cublasSsyr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float *alpha, const float  *x, int incx, const float  *y, int incy, float *A, int lda) {
    cublasStatus_t r;
    begin_func(cublasSsyr2_v2);
    r = so_cublasSsyr2_v2(handle, uplo, n, alpha, x, incx, y, incy, A, lda);
    end_func(cublasSsyr2_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasDsyr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double *alpha, const double  *x, int incx, const double  *y, int incy, double *A, int lda) {
    cublasStatus_t r;
    begin_func(cublasDsyr2_v2);
    r = so_cublasDsyr2_v2(handle, uplo, n, alpha, x, incx, y, incy, A, lda);
    end_func(cublasDsyr2_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasCsyr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex *alpha, const cuComplex  *x, int incx, const cuComplex  *y, int incy, cuComplex *A, int lda) {
    cublasStatus_t r;
    begin_func(cublasCsyr2_v2);
    r = so_cublasCsyr2_v2(handle, uplo, n, alpha, x, incx, y, incy, A, lda);
    end_func(cublasCsyr2_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasZsyr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex *alpha, const cuDoubleComplex  *x, int incx, const cuDoubleComplex  *y, int incy, cuDoubleComplex *A, int lda) {
    cublasStatus_t r;
    begin_func(cublasZsyr2_v2);
    r = so_cublasZsyr2_v2(handle, uplo, n, alpha, x, incx, y, incy, A, lda);
    end_func(cublasZsyr2_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasStbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const float *A, int lda, float *x, int incx) {
    cublasStatus_t r;
    begin_func(cublasStbmv_v2);
    r = so_cublasStbmv_v2(handle, uplo, trans, diag, n, k, A, lda, x, incx);
    end_func(cublasStbmv_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasDtbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const double *A, int lda, double *x, int incx) {
    cublasStatus_t r;
    begin_func(cublasDtbmv_v2);
    r = so_cublasDtbmv_v2(handle, uplo, trans, diag, n, k, A, lda, x, incx);
    end_func(cublasDtbmv_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasCtbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const cuComplex *A, int lda, cuComplex *x, int incx) {
    cublasStatus_t r;
    begin_func(cublasCtbmv_v2);
    r = so_cublasCtbmv_v2(handle, uplo, trans, diag, n, k, A, lda, x, incx);
    end_func(cublasCtbmv_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasZtbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const cuDoubleComplex *A, int lda, cuDoubleComplex *x, int incx) {
    cublasStatus_t r;
    begin_func(cublasZtbmv_v2);
    r = so_cublasZtbmv_v2(handle, uplo, trans, diag, n, k, A, lda, x, incx);
    end_func(cublasZtbmv_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasStbsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const float *A, int lda, float *x, int incx) {
    cublasStatus_t r;
    begin_func(cublasStbsv_v2);
    r = so_cublasStbsv_v2(handle, uplo, trans, diag, n, k, A, lda, x, incx);
    end_func(cublasStbsv_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasDtbsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const double *A, int lda, double *x, int incx) {
    cublasStatus_t r;
    begin_func(cublasDtbsv_v2);
    r = so_cublasDtbsv_v2(handle, uplo, trans, diag, n, k, A, lda, x, incx);
    end_func(cublasDtbsv_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasCtbsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const cuComplex *A, int lda, cuComplex *x, int incx) {
    cublasStatus_t r;
    begin_func(cublasCtbsv_v2);
    r = so_cublasCtbsv_v2(handle, uplo, trans, diag, n, k, A, lda, x, incx);
    end_func(cublasCtbsv_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasZtbsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const cuDoubleComplex *A, int lda, cuDoubleComplex *x, int incx) {
    cublasStatus_t r;
    begin_func(cublasZtbsv_v2);
    r = so_cublasZtbsv_v2(handle, uplo, trans, diag, n, k, A, lda, x, incx);
    end_func(cublasZtbsv_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasStpmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const float *AP, float *x, int incx) {
    cublasStatus_t r;
    begin_func(cublasStpmv_v2);
    r = so_cublasStpmv_v2(handle, uplo, trans, diag, n, AP, x, incx);
    end_func(cublasStpmv_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasDtpmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const double *AP, double *x, int incx) {
    cublasStatus_t r;
    begin_func(cublasDtpmv_v2);
    r = so_cublasDtpmv_v2(handle, uplo, trans, diag, n, AP, x, incx);
    end_func(cublasDtpmv_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasCtpmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuComplex *AP, cuComplex *x, int incx) {
    cublasStatus_t r;
    begin_func(cublasCtpmv_v2);
    r = so_cublasCtpmv_v2(handle, uplo, trans, diag, n, AP, x, incx);
    end_func(cublasCtpmv_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasZtpmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuDoubleComplex *AP, cuDoubleComplex *x, int incx) {
    cublasStatus_t r;
    begin_func(cublasZtpmv_v2);
    r = so_cublasZtpmv_v2(handle, uplo, trans, diag, n, AP, x, incx);
    end_func(cublasZtpmv_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasStpsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const float *AP, float *x, int incx) {
    cublasStatus_t r;
    begin_func(cublasStpsv_v2);
    r = so_cublasStpsv_v2(handle, uplo, trans, diag, n, AP, x, incx);
    end_func(cublasStpsv_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasDtpsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const double *AP, double *x, int incx) {
    cublasStatus_t r;
    begin_func(cublasDtpsv_v2);
    r = so_cublasDtpsv_v2(handle, uplo, trans, diag, n, AP, x, incx);
    end_func(cublasDtpsv_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasCtpsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuComplex *AP, cuComplex *x, int incx) {
    cublasStatus_t r;
    begin_func(cublasCtpsv_v2);
    r = so_cublasCtpsv_v2(handle, uplo, trans, diag, n, AP, x, incx);
    end_func(cublasCtpsv_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasZtpsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuDoubleComplex *AP, cuDoubleComplex *x, int incx) {
    cublasStatus_t r;
    begin_func(cublasZtpsv_v2);
    r = so_cublasZtpsv_v2(handle, uplo, trans, diag, n, AP, x, incx);
    end_func(cublasZtpsv_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasStrmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const float *A, int lda, float *x, int incx) {
    cublasStatus_t r;
    begin_func(cublasStrmv_v2);
    r = so_cublasStrmv_v2(handle, uplo, trans, diag, n, A, lda, x, incx);
    end_func(cublasStrmv_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasDtrmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const double *A, int lda, double *x, int incx) {
    cublasStatus_t r;
    begin_func(cublasDtrmv_v2);
    r = so_cublasDtrmv_v2(handle, uplo, trans, diag, n, A, lda, x, incx);
    end_func(cublasDtrmv_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasCtrmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuComplex *A, int lda, cuComplex *x, int incx) {
    cublasStatus_t r;
    begin_func(cublasCtrmv_v2);
    r = so_cublasCtrmv_v2(handle, uplo, trans, diag, n, A, lda, x, incx);
    end_func(cublasCtrmv_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasZtrmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuDoubleComplex *A, int lda, cuDoubleComplex *x, int incx) {
    cublasStatus_t r;
    begin_func(cublasZtrmv_v2);
    r = so_cublasZtrmv_v2(handle, uplo, trans, diag, n, A, lda, x, incx);
    end_func(cublasZtrmv_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasStrsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const float *A, int lda, float *x, int incx) {
    cublasStatus_t r;
    begin_func(cublasStrsv_v2);
    r = so_cublasStrsv_v2(handle, uplo, trans, diag, n, A, lda, x, incx);
    end_func(cublasStrsv_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasDtrsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const double *A, int lda, double *x, int incx) {
    cublasStatus_t r;
    begin_func(cublasDtrsv_v2);
    r = so_cublasDtrsv_v2(handle, uplo, trans, diag, n, A, lda, x, incx);
    end_func(cublasDtrsv_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasCtrsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuComplex *A, int lda, cuComplex *x, int incx) {
    cublasStatus_t r;
    begin_func(cublasCtrsv_v2);
    r = so_cublasCtrsv_v2(handle, uplo, trans, diag, n, A, lda, x, incx);
    end_func(cublasCtrsv_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasZtrsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuDoubleComplex *A, int lda, cuDoubleComplex *x, int incx) {
    cublasStatus_t r;
    begin_func(cublasZtrsv_v2);
    r = so_cublasZtrsv_v2(handle, uplo, trans, diag, n, A, lda, x, incx);
    end_func(cublasZtrsv_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasChemv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex  *x, int incx, const cuComplex *beta, cuComplex *y, int incy) {
    cublasStatus_t r;
    begin_func(cublasChemv_v2);
    r = so_cublasChemv_v2(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy);
    end_func(cublasChemv_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasZhemv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex  *x, int incx, const cuDoubleComplex *beta, cuDoubleComplex *y, int incy) {
    cublasStatus_t r;
    begin_func(cublasZhemv_v2);
    r = so_cublasZhemv_v2(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy);
    end_func(cublasZhemv_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasChbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, int k, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex  *x, int incx, const cuComplex *beta, cuComplex *y, int incy) {
    cublasStatus_t r;
    begin_func(cublasChbmv_v2);
    r = so_cublasChbmv_v2(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy);
    end_func(cublasChbmv_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasZhbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, int k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex  *x, int incx, const cuDoubleComplex *beta, cuDoubleComplex *y, int incy) {
    cublasStatus_t r;
    begin_func(cublasZhbmv_v2);
    r = so_cublasZhbmv_v2(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy);
    end_func(cublasZhbmv_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasChpmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex *alpha, const cuComplex *AP, const cuComplex  *x, int incx, const cuComplex *beta, cuComplex *y, int incy) {
    cublasStatus_t r;
    begin_func(cublasChpmv_v2);
    r = so_cublasChpmv_v2(handle, uplo, n, alpha, AP, x, incx, beta, y, incy);
    end_func(cublasChpmv_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasZhpmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *AP, const cuDoubleComplex  *x, int incx, const cuDoubleComplex *beta, cuDoubleComplex *y, int incy) {
    cublasStatus_t r;
    begin_func(cublasZhpmv_v2);
    r = so_cublasZhpmv_v2(handle, uplo, n, alpha, AP, x, incx, beta, y, incy);
    end_func(cublasZhpmv_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasCher_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float *alpha, const cuComplex  *x, int incx, cuComplex *A, int lda) {
    cublasStatus_t r;
    begin_func(cublasCher_v2);
    r = so_cublasCher_v2(handle, uplo, n, alpha, x, incx, A, lda);
    end_func(cublasCher_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasZher_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double *alpha, const cuDoubleComplex  *x, int incx, cuDoubleComplex *A, int lda) {
    cublasStatus_t r;
    begin_func(cublasZher_v2);
    r = so_cublasZher_v2(handle, uplo, n, alpha, x, incx, A, lda);
    end_func(cublasZher_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasCher2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex *alpha, const cuComplex  *x, int incx, const cuComplex  *y, int incy, cuComplex *A, int lda) {
    cublasStatus_t r;
    begin_func(cublasCher2_v2);
    r = so_cublasCher2_v2(handle, uplo, n, alpha, x, incx, y, incy, A, lda);
    end_func(cublasCher2_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasZher2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex *alpha, const cuDoubleComplex  *x, int incx, const cuDoubleComplex  *y, int incy, cuDoubleComplex *A, int lda) {
    cublasStatus_t r;
    begin_func(cublasZher2_v2);
    r = so_cublasZher2_v2(handle, uplo, n, alpha, x, incx, y, incy, A, lda);
    end_func(cublasZher2_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasChpr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float *alpha, const cuComplex *x, int incx, cuComplex *AP) {
    cublasStatus_t r;
    begin_func(cublasChpr_v2);
    r = so_cublasChpr_v2(handle, uplo, n, alpha, x, incx, AP);
    end_func(cublasChpr_v2);
    checkCublasErrors(r);
    return r;
}


cublasStatus_t cublasZhpr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double *alpha, const cuDoubleComplex *x, int incx, cuDoubleComplex *AP) {
    cublasStatus_t r;
    begin_func(cublasZhpr_v2);
    r = so_cublasZhpr_v2(handle, uplo, n, alpha, x, incx, AP);
    end_func(cublasZhpr_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasChpr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex *alpha, const cuComplex  *x, int incx, const cuComplex *y, int incy, cuComplex *AP) {
    cublasStatus_t r;
    begin_func(cublasChpr2_v2);
    r = so_cublasChpr2_v2(handle, uplo, n, alpha, x, incx, y, incy, AP);
    end_func(cublasChpr2_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasZhpr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex *alpha, const cuDoubleComplex  *x, int incx, const cuDoubleComplex *y, int incy, cuDoubleComplex *AP) {
    cublasStatus_t r;
    begin_func(cublasZhpr2_v2);
    r = so_cublasZhpr2_v2(handle, uplo, n, alpha, x, incx, y, incy, AP);
    end_func(cublasZhpr2_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasSgemm_v2(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc) {
    cublasStatus_t r;
    begin_func(cublasSgemm_v2);
    r = so_cublasSgemm_v2(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    end_func(cublasSgemm_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasDgemm_v2(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const double *alpha, const double *A, int lda, const double *B, int ldb, const double *beta, double *C, int ldc) {
    cublasStatus_t r;
    begin_func(cublasDgemm_v2);
    r = so_cublasDgemm_v2(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    end_func(cublasDgemm_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasCgemm_v2(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *B, int ldb, const cuComplex *beta, cuComplex *C, int ldc) {
    cublasStatus_t r;
    begin_func(cublasCgemm_v2);
    r = so_cublasCgemm_v2(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    end_func(cublasCgemm_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasZgemm_v2(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb, const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc) {
    cublasStatus_t r;
    begin_func(cublasZgemm_v2);
    r = so_cublasZgemm_v2(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    end_func(cublasZgemm_v2);
    checkCublasErrors(r);
    return r;
}

/*cublasStatus_t cublasHgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const __half *alpha, const __half *A, int lda, const __half *B, int ldb, const __half *beta, __half *C, int ldc) {
    cublasStatus_t r;
    begin_func(cublasHgemm);
    cublasStatus_t r = (*(cublasStatus_t (*)(cublasHandle_t , cublasOperation_t, cublasOperation_t , int, int , int , const __half *, const __half *, int, const __half *, int, const __half *, __half *, int ))(funcs[153]))(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    end_func();
    checkCublasErrors(r);
    return r;
}*/

cublasStatus_t cublasCgemm3m(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *B, int ldb, const cuComplex *beta, cuComplex *C, int ldc) {
    cublasStatus_t r;
    begin_func(cublasCgemm3m);
    r = so_cublasCgemm3m(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    end_func(cublasCgemm3m);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasZgemm3m(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb, const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc) {
    cublasStatus_t r;
    begin_func(cublasZgemm3m);
    r = so_cublasZgemm3m(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    end_func(cublasZgemm3m);
    checkCublasErrors(r);
    return r;
}

/*cublasStatus_t cublasHgemmBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const __half *alpha, const __half *Aarray[], int lda, const __half *Barray[], int ldb, const __half *beta, __half *Carray[], int ldc, int batchCount) {
    cublasStatus_t r;
    begin_func(cublasHgemmBatched);
    cublasStatus_t r = (*(cublasStatus_t (*)(cublasHandle_t , cublasOperation_t, cublasOperation_t , int, int , int , const __half *, const __half *, int, const __half *, int, const __half *, __half *, int, int ))(funcs[156]))(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount);
    end_func();
    checkCublasErrors(r);
    return r;
}*/

cublasStatus_t cublasSgemmBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha, const float *const Aarray[], int lda, const float *const Barray[], int ldb, const float *beta, float *const Carray[], int ldc, int batchCount) {
    cublasStatus_t r;
    begin_func(cublasSgemmBatched);
    r = so_cublasSgemmBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount);
    end_func(cublasSgemmBatched);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasDgemmBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const double *alpha, const double *const Aarray[], int lda, const double *const Barray[], int ldb, const double *beta, double *const Carray[], int ldc, int batchCount) {
    cublasStatus_t r;
    begin_func(cublasDgemmBatched);
    r = so_cublasDgemmBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount);
    end_func(cublasDgemmBatched);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasCgemmBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuComplex *alpha, const cuComplex *const Aarray[], int lda, const cuComplex *const Barray[], int ldb, const cuComplex *beta, cuComplex *const Carray[], int ldc, int batchCount) {
    cublasStatus_t r;
    begin_func(cublasCgemmBatched);
    r = so_cublasCgemmBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount);
    end_func(cublasCgemmBatched);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasZgemmBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuDoubleComplex *alpha, const cuDoubleComplex *const Aarray[], int lda, const cuDoubleComplex *const Barray[], int ldb, const cuDoubleComplex *beta, cuDoubleComplex *const Carray[], int ldc, int batchCount) {
    cublasStatus_t r;
    begin_func(cublasZgemmBatched);
    r = so_cublasZgemmBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount);
    end_func(cublasZgemmBatched);
    checkCublasErrors(r);
    return r;
}

/*cublasStatus_t cublasHgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const __half *alpha, const __half *A, int lda, long long int strideA, const __half *B, int ldb, long long int strideB, const __half *beta, __half *C, int ldc, long long int strideC, int batchCount) {
    cublasStatus_t r;
    begin_func(cublasHgemmStridedBatched);
    cublasStatus_t r = (*(cublasStatus_t (*)(cublasHandle_t , cublasOperation_t, cublasOperation_t , int, int , int , const __half *, const __half *, int, long long int, const __half *, int, long long int, const __half *, __half *, int, long long int, int ))(funcs[161]))(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
    end_func();
    checkCublasErrors(r);
    return r;
}*/

cublasStatus_t cublasSgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha, const float *A, int lda, long long int strideA, const float *B, int ldb, long long int strideB, const float *beta, float *C, int ldc, long long int strideC, int batchCount) {
    cublasStatus_t r;
    begin_func(cublasSgemmStridedBatched);
    r = so_cublasSgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
    end_func(cublasSgemmStridedBatched);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasDgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const double *alpha, const double *A, int lda, long long int strideA, const double *B, int ldb, long long int strideB, const double *beta, double *C, int ldc, long long int strideC, int batchCount) {
    cublasStatus_t r;
    begin_func(cublasDgemmStridedBatched);
    r = so_cublasDgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
    end_func(cublasDgemmStridedBatched);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasCgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuComplex *alpha, const cuComplex *A, int lda, long long int strideA, const cuComplex *B, int ldb, long long int strideB, const cuComplex *beta, cuComplex *C, int ldc, long long int strideC, int batchCount) {
    cublasStatus_t r;
    begin_func(cublasCgemmStridedBatched);
    r = so_cublasCgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
    end_func(cublasCgemmStridedBatched);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasZgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, long long int strideA, const cuDoubleComplex *B, int ldb, long long int strideB, const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc, long long int strideC, int batchCount) {
    cublasStatus_t r;
    begin_func(cublasZgemmStridedBatched);
    r = so_cublasZgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
    end_func(cublasZgemmStridedBatched);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasSsymm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc) {
    cublasStatus_t r;
    begin_func(cublasSsymm_v2);
    r = so_cublasSsymm_v2(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
    end_func(cublasSsymm_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasDsymm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const double *alpha, const double *A, int lda, const double *B, int ldb, const double *beta, double *C, int ldc) {
    cublasStatus_t r;
    begin_func(cublasDsymm_v2);
    r = so_cublasDsymm_v2(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
    end_func(cublasDsymm_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasCsymm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *B, int ldb, const cuComplex *beta, cuComplex *C, int ldc) {
    cublasStatus_t r;
    begin_func(cublasCsymm_v2);
    r = so_cublasCsymm_v2(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
    end_func(cublasCsymm_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasZsymm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb, const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc) {
    cublasStatus_t r;
    begin_func(cublasZsymm_v2);
    r = so_cublasZsymm_v2(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
    end_func(cublasZsymm_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasSsyrk_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const float *alpha, const float *A, int lda, const float *beta, float *C, int ldc) {
    cublasStatus_t r;
    begin_func(cublasSsyrk_v2);
    r = so_cublasSsyrk_v2(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
    end_func(cublasSsyrk_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasDsyrk_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const double *alpha, const double *A, int lda, const double *beta, double *C, int ldc) {
    cublasStatus_t r;
    begin_func(cublasDsyrk_v2);
    r = so_cublasDsyrk_v2(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
    end_func(cublasDsyrk_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasCsyrk_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *beta, cuComplex *C, int ldc) {
    cublasStatus_t r;
    begin_func(cublasCsyrk_v2);
    r = so_cublasCsyrk_v2(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
    end_func(cublasCsyrk_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasZsyrk_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc) {
    cublasStatus_t r;
    begin_func(cublasZsyrk_v2);
    r = so_cublasZsyrk_v2(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
    end_func(cublasZsyrk_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasSsyr2k_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc) {
    cublasStatus_t r;
    begin_func(cublasSsyr2k_v2);
    r = so_cublasSsyr2k_v2(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    end_func(cublasSsyr2k_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasDsyr2k_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const double *alpha, const double *A, int lda, const double *B, int ldb, const double *beta, double *C, int ldc) {
    cublasStatus_t r;
    begin_func(cublasDsyr2k_v2);
    r = so_cublasDsyr2k_v2(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    end_func(cublasDsyr2k_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasCsyr2k_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *B, int ldb, const cuComplex *beta, cuComplex *C, int ldc) {
    cublasStatus_t r;
    begin_func(cublasCsyr2k_v2);
    r = so_cublasCsyr2k_v2(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    end_func(cublasCsyr2k_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasZsyr2k_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb, const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc) {
    cublasStatus_t r;
    begin_func(cublasZsyr2k_v2);
    r = so_cublasZsyr2k_v2(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    end_func(cublasZsyr2k_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasSsyrkx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc) {
    cublasStatus_t r;
    begin_func(cublasSsyrkx);
    r = so_cublasSsyrkx(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    end_func(cublasSsyrkx);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasDsyrkx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const double *alpha, const double *A, int lda, const double *B, int ldb, const double *beta, double *C, int ldc) {
    cublasStatus_t r;
    begin_func(cublasDsyrkx);
    r = so_cublasDsyrkx(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    end_func(cublasDsyrkx);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasCsyrkx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *B, int ldb, const cuComplex *beta, cuComplex *C, int ldc) {
    cublasStatus_t r;
    begin_func(cublasCsyrkx);
    r = so_cublasCsyrkx(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    end_func(cublasCsyrkx);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasZsyrkx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb, const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc) {
    cublasStatus_t r;
    begin_func(cublasZsyrkx);
    r = so_cublasZsyrkx(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    end_func(cublasZsyrkx);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasStrmm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const float *alpha, const float *A, int lda, const float *B, int ldb, float *C, int ldc) {
    cublasStatus_t r;
    begin_func(cublasStrmm_v2);
    r = so_cublasStrmm_v2(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc);
    end_func(cublasStrmm_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasDtrmm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const double *alpha, const double *A, int lda, const double *B, int ldb, double *C, int ldc) {
    cublasStatus_t r;
    begin_func(cublasDtrmm_v2);
    r = so_cublasDtrmm_v2(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc);
    end_func(cublasDtrmm_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasCtrmm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *B, int ldb, cuComplex *C, int ldc) {
    cublasStatus_t r;
    begin_func(cublasCtrmm_v2);
    r = so_cublasCtrmm_v2(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc);
    end_func(cublasCtrmm_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasZtrmm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb, cuDoubleComplex *C, int ldc) {
    cublasStatus_t r;
    begin_func(cublasZtrmm_v2);
    r = so_cublasZtrmm_v2(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc);
    end_func(cublasZtrmm_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasStrsm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const float *alpha, const float *A, int lda, float *B, int ldb) {
    cublasStatus_t r;
    begin_func(cublasStrsm_v2);
    r = so_cublasStrsm_v2(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
    end_func(cublasStrsm_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasDtrsm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const double *alpha, const double *A, int lda, double *B, int ldb) {
    cublasStatus_t r;
    begin_func(cublasDtrsm_v2);
    r = so_cublasDtrsm_v2(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
    end_func(cublasDtrsm_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasCtrsm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuComplex *alpha, const cuComplex *A, int lda, cuComplex *B, int ldb) {
    cublasStatus_t r;
    begin_func(cublasCtrsm_v2);
    r = so_cublasCtrsm_v2(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
    end_func(cublasCtrsm_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasZtrsm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, cuDoubleComplex *B, int ldb) {
    cublasStatus_t r;
    begin_func(cublasZtrsm_v2);
    r = so_cublasZtrsm_v2(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
    end_func(cublasZtrsm_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasStrsmBatched(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const float *alpha, const float *const A[], int lda, float *const B[], int ldb, int batchCount) {
    cublasStatus_t r;
    begin_func(cublasStrsmBatched);
    r = so_cublasStrsmBatched(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount);
    end_func(cublasStrsmBatched);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasDtrsmBatched(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const double *alpha, const double *const A[], int lda, double *const B[], int ldb, int batchCount) {
    cublasStatus_t r;
    begin_func(cublasDtrsmBatched);
    r = so_cublasDtrsmBatched(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount);
    end_func(cublasDtrsmBatched);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasCtrsmBatched(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuComplex *alpha, const cuComplex *const A[], int lda, cuComplex *const B[], int ldb, int batchCount) {
    cublasStatus_t r;
    begin_func(cublasCtrsmBatched);
    r = so_cublasCtrsmBatched(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount);
    end_func(cublasCtrsmBatched);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasZtrsmBatched(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *const A[], int lda, cuDoubleComplex *const B[], int ldb, int batchCount) {
    cublasStatus_t r;
    begin_func(cublasZtrsmBatched);
    r = so_cublasZtrsmBatched(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount);
    end_func(cublasZtrsmBatched);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasChemm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *B, int ldb, const cuComplex *beta, cuComplex *C, int ldc) {
    cublasStatus_t r;
    begin_func(cublasChemm_v2);
    r = so_cublasChemm_v2(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
    end_func(cublasChemm_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasZhemm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb, const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc) {
    cublasStatus_t r;
    begin_func(cublasZhemm_v2);
    r = so_cublasZhemm_v2(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
    end_func(cublasZhemm_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasCherk_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const float *alpha, const cuComplex *A, int lda, const float *beta, cuComplex *C, int ldc) {
    cublasStatus_t r;
    begin_func(cublasCherk_v2);
    r = so_cublasCherk_v2(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
    end_func(cublasCherk_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasZherk_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const double *alpha, const cuDoubleComplex *A, int lda, const double *beta, cuDoubleComplex *C, int ldc) {
    cublasStatus_t r;
    begin_func(cublasZherk_v2);
    r = so_cublasZherk_v2(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
    end_func(cublasZherk_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasCher2k_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *B, int ldb, const float *beta, cuComplex *C, int ldc) {
    cublasStatus_t r;
    begin_func(cublasCher2k_v2);
    r = so_cublasCher2k_v2(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    end_func(cublasCher2k_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasZher2k_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb, const double *beta, cuDoubleComplex *C, int ldc) {
    cublasStatus_t r;
    begin_func(cublasZher2k_v2);
    r = so_cublasZher2k_v2(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    end_func(cublasZher2k_v2);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasCherkx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *B, int ldb, const float *beta, cuComplex *C, int ldc) {
    cublasStatus_t r;
    begin_func(cublasCherkx);
    r = so_cublasCherkx(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    end_func(cublasCherkx);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasZherkx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb, const double *beta, cuDoubleComplex *C, int ldc) {
    cublasStatus_t r;
    begin_func(cublasZherkx);
    r = so_cublasZherkx(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    end_func(cublasZherkx);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasSgeam(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, const float *alpha, const float *A, int lda, const float *beta, const float *B, int ldb, float *C, int ldc) {
    cublasStatus_t r;
    begin_func(cublasSgeam);
    r = so_cublasSgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
    end_func(cublasSgeam);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasDgeam(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, const double *alpha, const double *A, int lda, const double *beta, const double *B, int ldb, double *C, int ldc) {
    cublasStatus_t r;
    begin_func(cublasDgeam);
    r = so_cublasDgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
    end_func(cublasDgeam);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasCgeam(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *beta, const cuComplex *B, int ldb, cuComplex *C, int ldc) {
    cublasStatus_t r;
    begin_func(cublasCgeam);
    r = so_cublasCgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
    end_func(cublasCgeam);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasZgeam(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *beta, const cuDoubleComplex *B, int ldb, cuDoubleComplex *C, int ldc) {
    cublasStatus_t r;
    begin_func(cublasZgeam);
    r = so_cublasZgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
    end_func(cublasZgeam);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasSdgmm(cublasHandle_t handle, cublasSideMode_t mode, int m, int n, const float *A, int lda, const float *x, int incx, float *C, int ldc) {
    cublasStatus_t r;
    begin_func(cublasSdgmm);
    r = so_cublasSdgmm(handle, mode, m, n, A, lda, x, incx, C, ldc);
    end_func(cublasSdgmm);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasDdgmm(cublasHandle_t handle, cublasSideMode_t mode, int m, int n, const double *A, int lda, const double *x, int incx, double *C, int ldc) {
    cublasStatus_t r;
    begin_func(cublasDdgmm);
    r = so_cublasDdgmm(handle, mode, m, n, A, lda, x, incx, C, ldc);
    end_func(cublasDdgmm);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasCdgmm(cublasHandle_t handle, cublasSideMode_t mode, int m, int n, const cuComplex *A, int lda, const cuComplex *x, int incx, cuComplex *C, int ldc) {
    cublasStatus_t r;
    begin_func(cublasCdgmm);
    r = so_cublasCdgmm(handle, mode, m, n, A, lda, x, incx, C, ldc);
    end_func(cublasCdgmm);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasZdgmm(cublasHandle_t handle, cublasSideMode_t mode, int m, int n, const cuDoubleComplex *A, int lda, const cuDoubleComplex *x, int incx, cuDoubleComplex *C, int ldc) {
    cublasStatus_t r;
    begin_func(cublasZdgmm);
    r = so_cublasZdgmm(handle, mode, m, n, A, lda, x, incx, C, ldc);
    end_func(cublasZdgmm);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasSgetrfBatched(cublasHandle_t handle, int n, float *const Aarray[], int lda, int *PivotArray, int *infoArray, int batchSize) {
    cublasStatus_t r;
    begin_func(cublasSgetrfBatched);
    r = so_cublasSgetrfBatched(handle, n, Aarray, lda, PivotArray, infoArray, batchSize );
    end_func(cublasSgetrfBatched);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasDgetrfBatched(cublasHandle_t handle, int n, double *const Aarray[], int lda, int *PivotArray, int *infoArray, int batchSize) {
    cublasStatus_t r;
    begin_func(cublasDgetrfBatched);
    r = so_cublasDgetrfBatched(handle, n, Aarray, lda, PivotArray, infoArray, batchSize );
    end_func(cublasDgetrfBatched);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasCgetrfBatched(cublasHandle_t handle, int n, cuComplex *const Aarray[], int lda, int *PivotArray, int *infoArray, int batchSize) {
    cublasStatus_t r;
    begin_func(cublasCgetrfBatched);
    r = so_cublasCgetrfBatched(handle, n, Aarray, lda, PivotArray, infoArray, batchSize );
    end_func(cublasCgetrfBatched);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasZgetrfBatched(cublasHandle_t handle, int n, cuDoubleComplex *const Aarray[], int lda, int *PivotArray, int *infoArray, int batchSize) {
    cublasStatus_t r;
    begin_func(cublasZgetrfBatched);
    r = so_cublasZgetrfBatched(handle, n, Aarray, lda, PivotArray, infoArray, batchSize );
    end_func(cublasZgetrfBatched);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasSgetrsBatched(cublasHandle_t handle, cublasOperation_t trans, int n, int nrhs, const float *const Aarray[], int lda, const int *devIpiv, float *const Barray[], int ldb, int *info, int batchSize) {
    cublasStatus_t r;
    begin_func(cublasSgetrsBatched);
    r = so_cublasSgetrsBatched(handle, trans, n, nrhs, Aarray, lda, devIpiv, Barray, ldb, info, batchSize );
    end_func(cublasSgetrsBatched);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasDgetrsBatched(cublasHandle_t handle, cublasOperation_t trans, int n, int nrhs, const double *const Aarray[], int lda, const int *devIpiv, double *const Barray[], int ldb, int *info, int batchSize) {
    cublasStatus_t r;
    begin_func(cublasDgetrsBatched);
    r = so_cublasDgetrsBatched(handle, trans, n, nrhs, Aarray, lda, devIpiv, Barray, ldb, info, batchSize );
    end_func(cublasDgetrsBatched);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasCgetrsBatched(cublasHandle_t handle, cublasOperation_t trans, int n, int nrhs, const cuComplex *const Aarray[], int lda, const int *devIpiv, cuComplex *const Barray[], int ldb, int *info, int batchSize) {
    cublasStatus_t r;
    begin_func(cublasCgetrsBatched);
    r = so_cublasCgetrsBatched(handle, trans, n, nrhs, Aarray, lda, devIpiv, Barray, ldb, info, batchSize );
    end_func(cublasCgetrsBatched);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasZgetrsBatched(cublasHandle_t handle, cublasOperation_t trans, int n, int nrhs, const cuDoubleComplex *const Aarray[], int lda, const int *devIpiv, cuDoubleComplex *const Barray[], int ldb, int *info, int batchSize) {
    cublasStatus_t r;
    begin_func(cublasZgetrsBatched);
    r = so_cublasZgetrsBatched(handle, trans, n, nrhs, Aarray, lda, devIpiv, Barray, ldb, info, batchSize );
    end_func(cublasZgetrsBatched);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasSgetriBatched(cublasHandle_t handle, int n, const float *const Aarray[], int lda, const int *PivotArray, float *const Carray[], int ldc, int *infoArray, int batchSize) {
    cublasStatus_t r;
    begin_func(cublasSgetriBatched);
    r = so_cublasSgetriBatched(handle, n, Aarray, lda, PivotArray, Carray, ldc, infoArray, batchSize );
    end_func(cublasSgetriBatched);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasDgetriBatched(cublasHandle_t handle, int n, const double *const Aarray[], int lda, const int *PivotArray, double *const Carray[], int ldc, int *infoArray, int batchSize) {
    cublasStatus_t r;
    begin_func(cublasDgetriBatched);
    r = so_cublasDgetriBatched(handle, n, Aarray, lda, PivotArray, Carray, ldc, infoArray, batchSize );
    end_func(cublasDgetriBatched);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasCgetriBatched(cublasHandle_t handle, int n, const cuComplex *const Aarray[], int lda, const int *PivotArray, cuComplex *const Carray[], int ldc, int *infoArray, int batchSize) {
    cublasStatus_t r;
    begin_func(cublasCgetriBatched);
    r = so_cublasCgetriBatched(handle, n, Aarray, lda, PivotArray, Carray, ldc, infoArray, batchSize );
    end_func(cublasCgetriBatched);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasZgetriBatched(cublasHandle_t handle, int n, const cuDoubleComplex *const Aarray[], int lda, const int *PivotArray, cuDoubleComplex *const Carray[], int ldc, int *infoArray, int batchSize) {
    cublasStatus_t r;
    begin_func(cublasZgetriBatched);
    r = so_cublasZgetriBatched(handle, n, Aarray, lda, PivotArray, Carray, ldc, infoArray, batchSize );
    end_func(cublasZgetriBatched);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasSmatinvBatched(cublasHandle_t handle, int n, const float *const A[], int lda, float *const Ainv[], int lda_inv, int *info, int batchSize) {
    cublasStatus_t r;
    begin_func(cublasSmatinvBatched);
    r = so_cublasSmatinvBatched(handle, n, A, lda, Ainv, lda_inv, info, batchSize );
    end_func(cublasSmatinvBatched);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasDmatinvBatched(cublasHandle_t handle, int n, const double *const A[], int lda, double *const Ainv[], int lda_inv, int *info, int batchSize) {
    cublasStatus_t r;
    begin_func(cublasDmatinvBatched);
    r = so_cublasDmatinvBatched(handle, n, A, lda, Ainv, lda_inv, info, batchSize );
    end_func(cublasDmatinvBatched);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasCmatinvBatched(cublasHandle_t handle, int n, const cuComplex *const A[], int lda, cuComplex *const Ainv[], int lda_inv, int *info, int batchSize) {
    cublasStatus_t r;
    begin_func(cublasCmatinvBatched);
    r = so_cublasCmatinvBatched(handle, n, A, lda, Ainv, lda_inv, info, batchSize );
    end_func(cublasCmatinvBatched);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasZmatinvBatched(cublasHandle_t handle, int n, const cuDoubleComplex *const A[], int lda, cuDoubleComplex *const Ainv[], int lda_inv, int *info, int batchSize) {
    cublasStatus_t r;
    begin_func(cublasZmatinvBatched);
    r = so_cublasZmatinvBatched(handle, n, A, lda, Ainv, lda_inv, info, batchSize );
    end_func(cublasZmatinvBatched);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasSgeqrfBatched(cublasHandle_t handle, int m, int n, float *const Aarray[], int lda, float *const TauArray[], int *info, int batchSize) {
    cublasStatus_t r;
    begin_func(cublasSgeqrfBatched);
    r = so_cublasSgeqrfBatched(handle, m, n, Aarray, lda, TauArray, info, batchSize );
    end_func(cublasSgeqrfBatched);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasDgeqrfBatched(cublasHandle_t handle, int m, int n, double *const Aarray[], int lda, double *const TauArray[], int *info, int batchSize) {
    cublasStatus_t r;
    begin_func(cublasDgeqrfBatched);
    r = so_cublasDgeqrfBatched(handle, m, n, Aarray, lda, TauArray, info, batchSize );
    end_func(cublasDgeqrfBatched);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasCgeqrfBatched(cublasHandle_t handle, int m, int n, cuComplex *const Aarray[], int lda, cuComplex *const TauArray[], int *info, int batchSize) {
    cublasStatus_t r;
    begin_func(cublasCgeqrfBatched);
    r = so_cublasCgeqrfBatched(handle, m, n, Aarray, lda, TauArray, info, batchSize );
    end_func(cublasCgeqrfBatched);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasZgeqrfBatched(cublasHandle_t handle, int m, int n, cuDoubleComplex *const Aarray[], int lda, cuDoubleComplex *const TauArray[], int *info, int batchSize) {
    cublasStatus_t r;
    begin_func(cublasZgeqrfBatched);
    r = so_cublasZgeqrfBatched(handle, m, n, Aarray, lda, TauArray, info, batchSize );
    end_func(cublasZgeqrfBatched);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasSgelsBatched( cublasHandle_t handle, cublasOperation_t trans, int m, int n, int nrhs, float *const Aarray[], int lda, float *const Carray[], int ldc, int *info, int *devInfoArray, int batchSize) {
    cublasStatus_t r;
    begin_func(cublasSgelsBatched);
    r = so_cublasSgelsBatched(handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc, info, devInfoArray, batchSize);
    end_func(cublasSgelsBatched);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasDgelsBatched( cublasHandle_t handle, cublasOperation_t trans, int m, int n, int nrhs, double *const Aarray[], int lda, double *const Carray[], int ldc, int *info, int *devInfoArray, int batchSize) {
    cublasStatus_t r;
    begin_func(cublasDgelsBatched);
    r = so_cublasDgelsBatched(handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc, info, devInfoArray, batchSize);
    end_func(cublasDgelsBatched);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasCgelsBatched( cublasHandle_t handle, cublasOperation_t trans, int m, int n, int nrhs, cuComplex *const Aarray[], int lda, cuComplex *const Carray[], int ldc, int *info, int *devInfoArray, int batchSize) {
    cublasStatus_t r;
    begin_func(cublasCgelsBatched);
    r = so_cublasCgelsBatched(handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc, info, devInfoArray, batchSize);
    end_func(cublasCgelsBatched);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasZgelsBatched( cublasHandle_t handle, cublasOperation_t trans, int m, int n, int nrhs, cuDoubleComplex *const Aarray[], int lda, cuDoubleComplex *const Carray[], int ldc, int *info, int *devInfoArray, int batchSize) {
    cublasStatus_t r;
    begin_func(cublasZgelsBatched);
    r = so_cublasZgelsBatched(handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc, info, devInfoArray, batchSize);
    end_func(cublasZgelsBatched);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasStpttr(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float *AP, float *A, int lda) {
    cublasStatus_t r;
    begin_func(cublasStpttr);
    r = so_cublasStpttr(handle, uplo, n, AP, A, lda);
    end_func(cublasStpttr);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasDtpttr(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double *AP, double *A, int lda) {
    cublasStatus_t r;
    begin_func(cublasDtpttr);
    r = so_cublasDtpttr(handle, uplo, n, AP, A, lda);
    end_func(cublasDtpttr);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasCtpttr(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex *AP, cuComplex *A, int lda) {
    cublasStatus_t r;
    begin_func(cublasCtpttr);
    r = so_cublasCtpttr(handle, uplo, n, AP, A, lda);
    end_func(cublasCtpttr);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasZtpttr(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex *AP, cuDoubleComplex *A, int lda) {
    cublasStatus_t r;
    begin_func(cublasZtpttr);
    r = so_cublasZtpttr(handle, uplo, n, AP, A, lda);
    end_func(cublasZtpttr);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasStrttp( cublasHandle_t handle, cublasFillMode_t uplo, int n, const float *A, int lda, float *AP) {
    cublasStatus_t r;
    begin_func(cublasStrttp);
    r = so_cublasStrttp(handle, uplo, n, A, lda, AP);
    end_func(cublasStrttp);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasDtrttp( cublasHandle_t handle, cublasFillMode_t uplo, int n, const double *A, int lda, double *AP) {
    cublasStatus_t r;
    begin_func(cublasDtrttp);
    r = so_cublasDtrttp(handle, uplo, n, A, lda, AP);
    end_func(cublasDtrttp);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasCtrttp( cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex *A, int lda, cuComplex *AP) {
    cublasStatus_t r;
    begin_func(cublasCtrttp);
    r = so_cublasCtrttp(handle, uplo, n, A, lda, AP);
    end_func(cublasCtrttp);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasZtrttp( cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex *A, int lda, cuDoubleComplex *AP) {
    cublasStatus_t r;
    begin_func(cublasZtrttp);
    r = so_cublasZtrttp(handle, uplo, n, A, lda, AP);
    end_func(cublasZtrttp);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasSgemmEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha, const void *A, cudaDataType Atype, int lda, const void *B, cudaDataType Btype, int ldb, const float *beta, void *C, cudaDataType Ctype, int ldc) {
    cublasStatus_t r;
    begin_func(cublasSgemmEx);
    r = so_cublasSgemmEx(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc);
    end_func(cublasSgemmEx);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasCgemmEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuComplex *alpha, const void *A, cudaDataType Atype, int lda, const void *B, cudaDataType Btype, int ldb, const cuComplex *beta, void *C, cudaDataType Ctype, int ldc) {
    cublasStatus_t r;
    begin_func(cublasCgemmEx);
    r = so_cublasCgemmEx(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc);
    end_func(cublasCgemmEx);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasGemmEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const void *alpha, const void *A, cudaDataType Atype, int lda, const void *B, cudaDataType Btype, int ldb, const void *beta, void *C, cudaDataType Ctype, int ldc, cudaDataType computeType, cublasGemmAlgo_t algo) {
    cublasStatus_t r;
    begin_func(cublasGemmEx);
    r = so_cublasGemmEx(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc, computeType, algo);
    end_func(cublasGemmEx);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasGemmBatchedEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const void *alpha, const void *const Aarray[], cudaDataType Atype, int lda, const void *const Barray[], cudaDataType Btype, int ldb, const void *beta, void *const Carray[], cudaDataType Ctype, int ldc, int batchCount, cudaDataType computeType, cublasGemmAlgo_t algo) {
    cublasStatus_t r;
    begin_func(cublasGemmBatchedEx);
    r = so_cublasGemmBatchedEx(handle, transa, transb, m, n, k, alpha, Aarray, Atype, lda, Barray, Btype, ldb, beta, Carray, Ctype, ldc, batchCount, computeType, algo);
    end_func(cublasGemmBatchedEx);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasGemmStridedBatchedEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const void *alpha, const void *A, cudaDataType Atype, int lda, long long int strideA, const void *B, cudaDataType Btype, int ldb, long long int strideB, const void *beta, void *C, cudaDataType Ctype, int ldc, long long int strideC, int batchCount, cudaDataType computeType, cublasGemmAlgo_t algo) {
    cublasStatus_t r;
    begin_func(cublasGemmStridedBatchedEx);
    r = so_cublasGemmStridedBatchedEx(handle, transa, transb, m, n, k, alpha, A, Atype, lda, strideA, B, Btype, ldb, strideB, beta, C, Ctype, ldc, strideC, batchCount, computeType, algo);
    end_func(cublasGemmStridedBatchedEx);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasCsyrkEx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex *alpha, const void *A, cudaDataType Atype, int lda, const cuComplex *beta, void *C, cudaDataType Ctype, int ldc) {
    cublasStatus_t r;
    begin_func(cublasCsyrkEx);
    r = so_cublasCsyrkEx(handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc);
    end_func(cublasCsyrkEx);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasCsyrk3mEx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex *alpha, const void *A, cudaDataType Atype, int lda, const cuComplex *beta, void *C, cudaDataType Ctype, int ldc) {
    cublasStatus_t r;
    begin_func(cublasCsyrk3mEx);
    r = so_cublasCsyrk3mEx(handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc);
    end_func(cublasCsyrk3mEx);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasCherkEx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const float *alpha, const void *A, cudaDataType Atype, int lda, const float *beta, void *C, cudaDataType Ctype, int ldc) {
    cublasStatus_t r;
    begin_func(cublasCherkEx);
    r = so_cublasCherkEx(handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc);
    end_func(cublasCherkEx);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasCherk3mEx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const float *alpha, const void *A, cudaDataType Atype, int lda, const float *beta, void *C, cudaDataType Ctype, int ldc) {
    cublasStatus_t r;
    begin_func(cublasCherk3mEx);
    r = so_cublasCherk3mEx(handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc);
    end_func(cublasCherk3mEx);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasNrm2Ex(cublasHandle_t handle, int n, const void *x, cudaDataType xType, int incx, void *result, cudaDataType resultType, cudaDataType executionType) {
    cublasStatus_t r;
    begin_func(cublasNrm2Ex);
    r = so_cublasNrm2Ex(handle, n, x, xType, incx, result, resultType, executionType);
    end_func(cublasNrm2Ex);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasAxpyEx(cublasHandle_t handle, int n, const void *alpha, cudaDataType alphaType, const void *x, cudaDataType xType, int incx, void *y, cudaDataType yType, int incy, cudaDataType executiontype) {
    cublasStatus_t r;
    begin_func(cublasAxpyEx);
    r = so_cublasAxpyEx(handle, n, alpha, alphaType, x, xType, incx, y, yType, incy, executiontype);
    end_func(cublasAxpyEx);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasDotEx(cublasHandle_t handle, int n, const void *x, cudaDataType xType, int incx, const void *y, cudaDataType yType, int incy, void *result, cudaDataType resultType, cudaDataType executionType) {
    cublasStatus_t r;
    begin_func(cublasDotEx);
    r = so_cublasDotEx(handle, n, x, xType, incx, y, yType, incy, result, resultType, executionType);
    end_func(cublasDotEx);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasDotcEx(cublasHandle_t handle, int n, const void *x, cudaDataType xType, int incx, const void *y, cudaDataType yType, int incy, void *result, cudaDataType resultType, cudaDataType executionType) {
    cublasStatus_t r;
    begin_func(cublasDotcEx);
    r = so_cublasDotcEx(handle, n, x, xType, incx, y, yType, incy, result, resultType, executionType);
    end_func(cublasDotcEx);
    checkCublasErrors(r);
    return r;
}

cublasStatus_t cublasScalEx(cublasHandle_t handle, int n, const void *alpha, cudaDataType alphaType, void *x, cudaDataType xType, int incx, cudaDataType executionType) {
    cublasStatus_t r;
    begin_func(cublasScalEx);
    r = so_cublasScalEx(handle, n, alpha, alphaType, x, xType, incx, executionType);
    end_func(cublasScalEx);
    checkCublasErrors(r);
    return r;
}
