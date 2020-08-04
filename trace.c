#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <string.h>
#include <pthread.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublas_api.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <unistd.h>
#include <time.h>
#include <assert.h>
#include <dlfcn.h>
#include <semaphore.h>
#include <signal.h>
#include <cudnn.h>

#include "cudaw.h"

#define DEFSO(func) static cudaError_t(*so_##func)
#define FSWAP(func) do {void **pp=pfuncs[i++]; so_##func=*pp; *pp=trace_##func;} while(0);
#define FCOPY(func) do {void *p=funcs[i++]; so_##func=p;} while(0);

DEFSO(cudaMalloc)(void** devPtr, size_t size);
DEFSO(cudaFree)(void* devPtr);
DEFSO(cudaEventCreate)(cudaEvent_t* event);
DEFSO(cudaEventCreateWithFlags)(cudaEvent_t* event, unsigned int flags);
DEFSO(cudaEventDestroy)(cudaEvent_t event);
DEFSO(cudaDeviceSynchronize)(void);
DEFSO(cudaStreamSynchronize)(cudaStream_t stream);
DEFSO(cudaLaunchKernel)(const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream);
DEFSO(cudaMemset)(void* devPtr, int value, size_t count);
DEFSO(cudaMemsetAsync)(void* devPtr, int value, size_t count, cudaStream_t stream);
DEFSO(cudaMemcpy)(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind);
DEFSO(cudaMemcpyAsync)(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream);

DEFSO(cublasSgemm_v2)(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc);
DEFSO(cublasSgemv_v2)(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const float *alpha, const float *A, int lda, const float *x, int incx, const float *beta, float *y, int incy);

/*DEFSO(cudnnAddTensor)(cudnnHandle_t handle, const void *alpha, const cudnnTensorDescriptor_t aDesc, const void *A, const void *beta, const cudnnTensorDescriptor_t cDesc, void *C);      
DEFSO(cudnnConvolutionBackwardBias)(cudnnHandle_t handle, const void *alpha, const cudnnTensorDescriptor_t dyDesc, const void *dy, const void *beta, const cudnnTensorDescriptor_t dbDesc, void *db);      
DEFSO(cudnnConvolutionBackwardData)(cudnnHandle_t handle, const void *alpha, const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnTensorDescriptor_t dyDesc, const void *dy, const cudnnConvolutionDescriptor_t convDesc, cudnnConvolutionBwdDataAlgo_t algo, void *workSpace, size_t workSpaceSizeInBytes, const void *beta, const cudnnTensorDescriptor_t dxDesc, void *dx);    
DEFSO(cudnnConvolutionBackwardFilter)(cudnnHandle_t handle, const void *alpha, const cudnnTensorDescriptor_t xDesc, const void *x, const cudnnTensorDescriptor_t dyDesc, const void *dy, const cudnnConvolutionDescriptor_t convDesc, cudnnConvolutionBwdFilterAlgo_t algo, void *workSpace, size_t workSpaceSizeInBytes, const void *beta, const cudnnFilterDescriptor_t dwDesc, void *dw);    
DEFSO(cudnnConvolutionForward)(cudnnHandle_t handle, const void *alpha, const cudnnTensorDescriptor_t xDesc, const void *x, const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnConvolutionDescriptor_t convDesc, cudnnConvolutionFwdAlgo_t algo, void *workSpace, size_t workSpaceSizeInBytes, const void *beta, const cudnnTensorDescriptor_t yDesc, void *y);    
DEFSO(cudnnSetStream)(cudnnHandle_t handle, cudaStream_t streamId);
DEFSO(cudnnGetStream)(cudnnHandle_t handle, cudaStream_t *streamId);*/

typedef struct so_invoke_t {
    uint16_t invoke_idx;
	uint16_t milliseconds;
	uint8_t  thread_idx;
	uint8_t  dli_idx;
	uint16_t func_idx;
} so_invoke_t;

typedef struct so_tls_t {
	uint8_t         thread_idx;
    uint8_t         async;
    uint8_t         event;
    uint32_t        trace_idx;
    uint64_t        invoke_idx;
    struct timeval  timestamp;
    char            sbuf[1024];
} so_tls_t;

static time_t    base_sec;            // the base time of seconds when start
static uint32_t  next_thread_idx = 0; // next index of new found thread
static uint64_t  next_trace_idx = 0;  // next index of traced invoke in .trace
static uint8_t   next_dli_idx = 0;    // next index of registered dynamic lib
static uint64_t  last_forward_trace_len = 0;
static uint64_t  last_forward_trace_idx = 0;
static uint64_t  last_backward_trace_len = 0;
static uint64_t  last_backward_trace_idx = 0;
static uint64_t  last_memcpy_d2h_trace_idx = 0;
static uint64_t  diff_sync_d2h_to_backward = 0;
static uint64_t  last_sync_d2h_to_backward = 0;

enum {
    SIGPAUSE   = SIGUSR1, // 10
    SIGMIGRATE = SIGUSR2, // 12
    SIGRESUME  = SIGCONT, // 18
};

static volatile sig_atomic_t so_signal_command = 0;
static volatile sig_atomic_t so_signal_last_command = 0;
static volatile sig_atomic_t so_signal_repeat = 0;

#define next_idx(pidx) __sync_add_and_fetch(pidx, 1)
#define idx_next(pidx) __sync_fetch_and_add(pidx, 1)

#define sync_set(ptr, v)   __sync_fetch_and_or(ptr, v)
#define sync_clear(ptr, v) __sync_fetch_and_and(ptr, ~(v))


static const char *fn_trace_dir = ".trace";
static const char *fn_recover_dir = ".recover";
static const char *fn_invoke = "invoke.trace";
static const char *fn_allocfree = "allocfree.trace";
static const char *fn_memcpy = "memcpy.trace";
static const char *fn_kernel = "kernel.trace";

static int fd_trace_dir = -1;
static int fd_recover_dir = -1;
static int fd_invoke = -1;
static int fd_allocfree = -1;
static int fd_memcpy = -1;
static int fd_kernel = -1;

static so_invoke_t *so_invokes = NULL;
static size_t       invoke_size = 10 * 1024 * 1024;

/*
We lock so_func_rwlock for read in cudaw_so_begin_func and unlock it in cudaw_so_end_func.
When we found a new thread, we let the new thread waiting for a write lock on 
so_func_rwlock, so as that the first invoke in the new thread is known after other invokes.
We use so_func_rwlock to guarantee that we do not make a checkpoint during any invokes.
*/
static pthread_rwlock_t so_func_rwlock;

static uint64_t so_checkpoint_thread_bits = 0;

static uint8_t so_backward_thread_idx = 0xff;
static uint8_t so_forward_thread_idx = 1;
static uint8_t so_memcpy_kind = 0; // last kind of memcpy in forward right after backward
static int     so_request_for_checkpoint = 0;
static sem_t   so_signal_sem;
static sem_t   so_pause_sem;
static sem_t   so_checkpoint_sem;

static __thread so_tls_t so_tls = {0};    // Thread local storage

static so_dl_info_t *so_dlips[16] = {0};  // Registration of all traced dynamic libs


static void so_time(struct timeval *pnow) {
	static struct timeval now = {0};
    if (gettimeofday(pnow, NULL) == 0) {
		now = *pnow;
	    return;
	}
	*pnow = now;
}

static void so_init_base_usec(void) {
	struct timeval now;
	so_time(&now);
	base_sec = now.tv_sec;
}

static uint32_t so_msec(struct timeval *pnow) {
    //return (pnow->tv_sec - base_sec) << 10 + (pnow->tv_usec >> 10)
	return (pnow->tv_sec - base_sec) * 1000 + pnow->tv_usec / 1000;
}

static int so_open(int fdir, const char *fname) {
    int oflag = O_RDWR | O_CREAT | O_TRUNC;
    int fd = openat(fdir, fname, oflag, 0660);
    if (fd == -1) {
        errmsg("%s\n", fname);
        exit(1);
	}
	return fd;
}

static void *so_mmap(int fd, size_t size) {
	int prot  = PROT_READ | PROT_WRITE;
	int flags = MAP_SHARED;
    if (ftruncate(fd, size) == -1) {
        errmsg("%d\n", fd);
        exit(1);
        return NULL;
    }
	void *ptr = mmap(NULL, size, prot, flags, fd, 0);
	if (ptr == MAP_FAILED) {
		errmsg("%d\n", fd);
        exit(1);
		return NULL;
	}
	return ptr;
}

static so_func_info_t * so_lookup_func_info(so_dl_info_t *dlip, void * func) {
    so_func_info_t * pfi = &dlip->funcs[1];
    while (pfi->func) {
        if (pfi->func == func) {
            return pfi;
        }
        pfi++;
    }
    static so_func_info_t nil = {0};
    return &nil;
}

static void so_try_pause_for_checkpoint(const void * func) {
	if (so_request_for_checkpoint &&
			so_tls.thread_idx == so_forward_thread_idx &&
			so_tls.async == 0 &&
            last_backward_trace_idx > 0 &&
            last_sync_d2h_to_backward > 0 &&
            diff_sync_d2h_to_backward >= last_sync_d2h_to_backward) {
		cudaError_t r = so_cudaDeviceSynchronize();
		if (r == cudaSuccess) {
            printf("checkpoint: sem_post(&so_pause_sem)\n");
			sem_post(&so_pause_sem);
			sem_wait(&so_checkpoint_sem);
            printf("checkpoint: sem_wait(&so_checkpoint_sem)\n");
            so_request_for_checkpoint = 0;
		}
	}
}

static void * so_ckeckpoint_deamon(void *data) {
    for (;;) {
        static uint64_t last_paused_idx = 0;
        sem_wait(&so_pause_sem);
        printf("paused at %ld for checkpoint[sig=%d] (%ld) (%ld)(%ld)\n", 
                        next_trace_idx, 
                        so_signal_last_command,
                        next_trace_idx-last_paused_idx,
                        last_forward_trace_len, 
                        last_backward_trace_len);
        last_paused_idx = next_trace_idx;
        so_signal_last_command = 0;
        for (;;) {
            if (so_signal_last_command == SIGRESUME ||
                    so_signal_command == SIGRESUME) {
                so_signal_last_command = 0;
                so_signal_command = 0;
                break;
            }
            usleep(100*1000);
        }
        sem_post(&so_checkpoint_sem);
    }
    return data;
}

static void so_start_checkpoint_deamon(void) {
    pthread_t deamon;
    int r = pthread_create(&deamon, NULL, so_ckeckpoint_deamon, NULL);
    if (r != 0) {
        errmsg("pthread_create(so_ckeckpoint_deamon)");
        exit(1);
    }
    pthread_detach(deamon);
}

static void * so_run_func_thread(void *data) {
    void (*func)() = data;
    func();
    return data;
}

static void so_run_func(void *func) {
    pthread_t thread;
    int r = pthread_create(&thread, NULL, so_run_func_thread, func);
    if (r != 0) {
        errmsg("pthread_create(so_run_func_thread)");
        return;
    }
    pthread_detach(thread);
}

static void so_set_request_for_checkpoint(void) {
    pthread_rwlock_wrlock(&so_func_rwlock);
    so_request_for_checkpoint = 1;
    pthread_rwlock_unlock(&so_func_rwlock);
}

static void signal_notifier_func(int sig) {
    if (so_signal_command == sig) {
        so_signal_repeat++;
    }
    else {
        so_signal_command = sig;
        so_signal_repeat = 0;
    }
    if (so_signal_repeat == 0) {
        printf("receiving signal %d\n", sig);
    }
    else {
        printf("receiving signal %d +%d\n", sig, so_signal_repeat);
    }
    sem_post(&so_signal_sem);
}

static void * so_signal_deamon(void *data) {
    for (;;) {
        sem_wait(&so_signal_sem);
        int sig = so_signal_command;
        if (so_signal_last_command != 0) {
            continue;
        }
        switch (sig) {
            case SIGRESUME:
                so_signal_last_command = sig;
                so_signal_command = 0;
                break;
            case SIGPAUSE:
                so_signal_last_command = sig;
                so_signal_command = 0;
                so_run_func(so_set_request_for_checkpoint);
                break;
            case SIGMIGRATE:
                so_signal_last_command = sig;
                so_signal_command = 0;
                so_run_func(so_set_request_for_checkpoint);
                break;
            default:
                so_signal_command = 0;
                break;
        }
    }
}

static void so_start_signal_deamon(void) {
    pthread_t deamon;
    int r = pthread_create(&deamon, NULL, so_signal_deamon, NULL);
    if (r != 0) {
        errmsg("pthread_create(so_signal_deamon)");
        exit(1);
    }
    pthread_detach(deamon);
}

// ----------------------------------------------------
// Swapped DL APIs
//

static cudaError_t trace_cudaMalloc(void** devPtr, size_t size) {
    cudaError_t r = so_cudaMalloc(devPtr, size);
    return r;
}

static cudaError_t trace_cudaFree(void* devPtr) {
    cudaError_t r = so_cudaFree(devPtr);
    return r;
}

static cudaError_t trace_cudaEventCreate(cudaEvent_t* event) {
    cudaError_t r = so_cudaEventCreate(event);
    if (r == cudaSuccess) {
        so_tls.event++;
        sync_set(&so_checkpoint_thread_bits, (1 << so_tls.thread_idx));
		sprintf(so_tls.sbuf, "(%p)", *event);
    }
    return r;
}

static cudaError_t trace_cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int  flags) {
    cudaError_t r = so_cudaEventCreateWithFlags(event, flags);
    if (r == cudaSuccess) {
        so_tls.event++;
        sync_set(&so_checkpoint_thread_bits, (1 << so_tls.thread_idx));
		sprintf(so_tls.sbuf, "(%p)", *event);
    }
    return r;
}

static cudaError_t trace_cudaEventDestroy(cudaEvent_t event) {
    cudaError_t r = so_cudaEventDestroy(event);
    if (r == cudaSuccess) {
        so_tls.event--;
        if (!(so_tls.event || so_tls.async)) {
            sync_clear(&so_checkpoint_thread_bits, (1 << so_tls.thread_idx));
        }
		sprintf(so_tls.sbuf, "(%p)", event);
    }
    return r;
}

static void trace_post_sync(cudaError_t r) {
	if (so_memcpy_kind == cudaMemcpyDeviceToHost && 
			so_tls.thread_idx == so_forward_thread_idx) {
		diff_sync_d2h_to_backward = so_tls.trace_idx - last_backward_trace_idx;
	}
    so_tls.async = 0;
    if (!(so_tls.event || so_tls.async)) {
        sync_clear(&so_checkpoint_thread_bits, (1 << so_tls.thread_idx));
    }
}

static cudaError_t trace_cudaDeviceSynchronize(void) {
    cudaError_t r = so_cudaDeviceSynchronize();
    if (r == cudaSuccess) {
		trace_post_sync(r);
    }
    return r;
}

static cudaError_t trace_cudaStreamSynchronize(cudaStream_t stream) {
    cudaError_t r = so_cudaStreamSynchronize(stream);
    if (!stream && r == cudaSuccess) {
		trace_post_sync(r);
    }
    return r;
}

static cudaError_t trace_cudaLaunchKernel(const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream) {
    cudaError_t r = so_cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);
    static int fcnt = 0;
	static const void * func_updateGradInput = NULL;
	#define __MAX_FUNCS 128
    static const void * funcs[__MAX_FUNCS] = {0};
	static const char * names[__MAX_FUNCS] = {0};
	#undef __MAX_FUNCS
	if (func == func_updateGradInput) {
		so_backward_thread_idx = so_tls.thread_idx;
	}
    for (int i=0; i<fcnt; i++) {
        if (funcs[i] == func) {
			if (names[i] == NULL) {
	            sprintf(so_tls.sbuf, "func=%p", func);
			}
			else {
	            sprintf(so_tls.sbuf, "func=%p name=%s", func, names[i]);
			}
            return r;
        }
    }
    Dl_info dli;
    if (dladdr(func, &dli) != 0) {
        funcs[fcnt] = func;
		size_t offset = (func-dli.dli_fbase);
		switch (offset) {
			case 0x4eca120: 
				names[fcnt] = "updateGradInput";
				func_updateGradInput = func;
				so_backward_thread_idx = so_tls.thread_idx;
				break;
			case 0x4ec9f80: 
				names[fcnt] = "updateOutput"; 
				break;
			// main.py
			case 0x4fffd40: names[fcnt] = "col2im"; break;
			case 0x5000280: names[fcnt] = "im2col"; break;
			case 0x45b2fd0: names[fcnt] = "TensorFillOp"; break;
			case 0x50567f0: names[fcnt] = "threshold"; break;
			case 0x563e850: names[fcnt] = "max_pool_forward_nchw"; break;
			case 0x5616c90: names[fcnt] = "copy_device_to_device"; break;
			case 0x5b01f50: names[fcnt] = "softmax_warp_forward"; break;
			case 0x5742a90: names[fcnt] = "fill_a90"; break;
			case 0x5af8d60: names[fcnt] = "softmax_warp_backward"; break;
			case 0x5a6feb0: names[fcnt] = "reduce:sum_eb0"; break;
			case 0x5742950: names[fcnt] = "fill_950"; break;
			case 0x563edb0: names[fcnt] = "max_pool_backward_nchw"; break;
			case 0x5173340: names[fcnt] = "add_340"; break;
			case 0x51705a0: names[fcnt] = "mul"; break;
			case 0x5a6f7f0: names[fcnt] = "reduce:ArgMax"; break;
			case 0x52de410: names[fcnt] = "eq"; break;
			case 0x5616fb0: names[fcnt] = "copy_device_to_device"; break;
			case 0x5a6ffd0: names[fcnt] = "reduce:sum_fd0"; break;
			// world
			case 0x471aef0: names[fcnt] = "indexSelectLargeIndex"; break;
			case 0x5718800: names[fcnt] = "Dropout:fused_dropout"; break;
			case 0x5a448c0: names[fcnt] = "RNN:lstm_cell_forward"; break;
			case 0x45afaf0: names[fcnt] = "CatArrayBatchedCopy"; break;
			case 0x5173480: names[fcnt] = "add_480"; break;
			case 0x5b03e40: names[fcnt] = "SoftMax:cunn_SoftMaxForward"; break;
			case 0x5b03900: names[fcnt] = "SoftMax:cunn_SoftMaxBackward"; break;
			case 0x571a8a0: names[fcnt] = "Dropout:masked_scale"; break;
			case 0x5a45980: names[fcnt] = "RNN:lstm_cell_backward"; break;
			case 0x572e210: names[fcnt] = "Embedding:embedding_backward_feature"; break;
			case 0x5a6eb00: names[fcnt] = "reduce:Norm"; break;
			default: break;
		}
		if (names[fcnt] != NULL) {
        	sprintf(so_tls.sbuf, "func=%p offset=0x%lx name=%s [%s]", 
                            func, offset, names[fcnt], dli.dli_sname);
		}
		else {
        	sprintf(so_tls.sbuf, "func=%p offset=0x%lx name=%s", 
                            func, offset, dli.dli_sname);
		}
		fcnt++;
    }
    else {
        sprintf(so_tls.sbuf, "func=%p", func);
    }
    return r;
}

static cudaError_t trace_cudaMemset(void* devPtr, int  value, size_t count) {
    cudaError_t r = so_cudaMemset(devPtr, value, count);
    sprintf(so_tls.sbuf, "ptr: %p val: %d cnt: %ld", devPtr, value, count);
    return r;
}

static cudaError_t trace_cudaMemsetAsync(void* devPtr, int  value, size_t count, cudaStream_t stream) {
    cudaError_t r = so_cudaMemsetAsync(devPtr, value, count, stream);
    sprintf(so_tls.sbuf, "ptr: %p val: %d cnt: %ld (%p)", devPtr, value, count, stream);
    return r;
}

static const char * memcpyKinds[] = {
    "Host -> Host",
    "Host -> Device",
    "Device -> Host",
    "Device -> Device",
    "cudaMemcpyDefault"
};

static cudaError_t trace_cudaMemcpy(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind) {
	if (so_request_for_checkpoint &&
            kind == cudaMemcpyHostToDevice &&
			so_tls.thread_idx == so_forward_thread_idx) {
        so_try_pause_for_checkpoint(cudaMemcpy);
    }
	cudaError_t r = so_cudaMemcpy(dst, src, count, kind);
	if (so_tls.thread_idx == so_forward_thread_idx) {
		so_memcpy_kind = kind;
	}
	so_tls.async = 1;
    sprintf(so_tls.sbuf, "dst: %p src: %p cnt: %lu kind: %s",
           	dst, src, count, memcpyKinds[kind]);
	return r;
}

static cudaError_t trace_cudaMemcpyAsync(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream) {
	if (so_request_for_checkpoint &&
            kind == cudaMemcpyHostToDevice &&
			so_tls.thread_idx == so_forward_thread_idx) {
        so_try_pause_for_checkpoint(cudaMemcpyAsync);
    }
	cudaError_t r = so_cudaMemcpyAsync(dst, src, count, kind, stream);
	if (so_tls.thread_idx == so_forward_thread_idx) {
		so_memcpy_kind = kind;
	}
	so_tls.async = 1;
    sprintf(so_tls.sbuf, "dst: %p src: %p cnt: %lu kind: %s (%p)",
           	dst, src, count, memcpyKinds[kind], stream);
	return r;
}

static cublasStatus_t trace_cublasSgemm_v2(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc) {
    cublasStatus_t r;
    r = so_cublasSgemm_v2(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    sprintf(so_tls.sbuf, "alpah=%p A=%p B=%p beta=%p C=%p", alpha, A, B, beta, C);
    return r;
}

static cublasStatus_t trace_cublasSgemv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const float *alpha, const float *A, int lda, const float *x, int incx, const float *beta, float *y, int incy) {
    cublasStatus_t r;
    r = so_cublasSgemv_v2(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
    sprintf(so_tls.sbuf, "alpah=%p A=%p x=%p beta=%p y=%p", alpha, A, x, beta, y);
    return r;
}

/*static cudnnStatus_t trace_cudnnAddTensor(cudnnHandle_t handle,const void *alpha,const cudnnTensorDescriptor_t aDesc,const void *A,const void *beta,const cudnnTensorDescriptor_t cDesc,void *C) {
    cudnnStatus_t r;
    r = so_cudnnAddTensor(handle,alpha,aDesc,A,beta,cDesc,C);
    sprintf(so_tls.sbuf, "alpah=%p A=%p beta=%p C=%p", alpha, A, beta, C);
    return r;
}

static cudnnStatus_t trace_cudnnConvolutionBackwardBias(cudnnHandle_t handle,const void *alpha,const cudnnTensorDescriptor_t dyDesc,const void *dy,const void *beta,const cudnnTensorDescriptor_t dbDesc,void *db) {
    cudnnStatus_t r;
    r = so_cudnnConvolutionBackwardBias(handle,alpha,dyDesc,dy,beta,dbDesc,db);
    sprintf(so_tls.sbuf, "alpah=%p dy=%p beta=%p db=%p", alpha, dy, beta, db);
    return r;
}

static cudnnStatus_t trace_cudnnConvolutionBackwardData(cudnnHandle_t handle,const void *alpha,const cudnnFilterDescriptor_t wDesc,const void *w,const cudnnTensorDescriptor_t dyDesc,const void *dy,const cudnnConvolutionDescriptor_t convDesc,cudnnConvolutionBwdDataAlgo_t algo,void *workSpace,size_t workSpaceSizeInBytes,const void *beta,const cudnnTensorDescriptor_t dxDesc,void *dx) {
    cudnnStatus_t r;
    r = so_cudnnConvolutionBackwardData(handle,alpha,wDesc,w,dyDesc,dy,convDesc,algo,workSpace,workSpaceSizeInBytes,beta,dxDesc,dx);
    sprintf(so_tls.sbuf, "alpah=%p w=%p dy=%p workSpace=%p beta=%p dx=%p", alpha, w, dy, workSpace, beta, dx);
    return r;
}

static cudnnStatus_t trace_cudnnConvolutionBackwardFilter(cudnnHandle_t handle,const void *alpha,const cudnnTensorDescriptor_t xDesc,const void *x,const cudnnTensorDescriptor_t dyDesc,const void *dy,const cudnnConvolutionDescriptor_t convDesc,cudnnConvolutionBwdFilterAlgo_t algo,void *workSpace,size_t workSpaceSizeInBytes,const void *beta,const cudnnFilterDescriptor_t dwDesc,void *dw) {
    cudnnStatus_t r;
    r = so_cudnnConvolutionBackwardFilter(handle,alpha,xDesc,x,dyDesc,dy,convDesc,algo,workSpace,workSpaceSizeInBytes,beta,dwDesc,dw);
    sprintf(so_tls.sbuf, "alpah=%p x=%p dy=%p workSpace=%p beta=%p dw=%p", alpha, x, dy, workSpace, beta, dw);
    return r;
}

static cudnnStatus_t trace_cudnnConvolutionForward(cudnnHandle_t handle,const void *alpha,const cudnnTensorDescriptor_t xDesc,const void *x,const cudnnFilterDescriptor_t wDesc,const void *w,const cudnnConvolutionDescriptor_t convDesc,cudnnConvolutionFwdAlgo_t algo,void *workSpace,size_t workSpaceSizeInBytes,const void *beta,const cudnnTensorDescriptor_t yDesc,void *y) {
    cudnnStatus_t r;
    r = so_cudnnConvolutionForward(handle,alpha,xDesc,x,wDesc,w,convDesc,algo,workSpace,workSpaceSizeInBytes,beta,yDesc,y);
    sprintf(so_tls.sbuf, "alpah=%p x=%p w=%p workSpace=%p beta=%p y=%p", alpha, x, w, workSpace, beta, y);
    return r;
}

static cudnnStatus_t trace_cudnnSetStream(cudnnHandle_t handle, cudaStream_t streamId) {
    
    cudnnStatus_t r = so_cudnnSetStream(handle,streamId);
    if (r == CUDNN_STATUS_SUCCESS) {
        sprintf(so_tls.sbuf, "handle:%p, streamId:%p", handle, streamId);
    }
    return r;
}

static cudnnStatus_t trace_cudnnGetStream(cudnnHandle_t handle, cudaStream_t *streamId) {
    
    cudnnStatus_t r = so_cudnnGetStream(handle, streamId);
    if (r == CUDNN_STATUS_SUCCESS) {
        sprintf(so_tls.sbuf, "handle:%p, streamId:%p", handle, *streamId);
    }
    return r;
}*/

// ----------------------------------------------------
//
//

static void so_print_invoke(uint32_t idx, so_invoke_t * p, uint32_t cnt) {
    int thread_idx = (p->thread_idx & 0x3f);
    int dli_idx = (p->dli_idx & 0xf);
    so_func_info_t *pfi = &so_dlips[dli_idx]->funcs[p->func_idx];
    int thread_checkpoint = !!(p->thread_idx & 0x40);
    int global_checkpoint = !!(p->thread_idx & 0x80);
    int async = (p->dli_idx & 0x80);
    int event = (p->dli_idx & 0x40);
    printf("%10u %6u %6u _%u_ ", 
            idx, p->invoke_idx, p->milliseconds, thread_idx);
    if (global_checkpoint) {
        printf("-g-");
    }
    else if (thread_checkpoint) {
        printf("-t-");
    }
    else {
        printf("-.-");
    }
    printf("%c-%c-", async?'a':'.', event?'e':'.');
    printf("%c- ", pfi->flags.known?'.':'u');
    printf("%d:%s", dli_idx, pfi->func_name);
    if (cnt > 1)
        printf(" +%u\n", cnt);
    else if (cnt == 1) {
        printf("\n");
    }
    else {
        printf(" %s\n", so_tls.sbuf);
    }
}

static void so_update_func_info(so_dl_info_t *dlip) {
    extern void (__cudaPushCallConfiguration)(void);
    extern void (__cudaPopCallConfiguration)(void);
    extern void (__cudaRegisterVar)(void);
    extern void (__cudaRegisterTexture)(void);
    extern void (__cudaRegisterFunction)(void);
    extern void (__cudaRegisterFatBinary)(void);
    extern void (__cudaUnregisterFatBinary)(void);
    void * checkpoint_funcs[] = {
        cudaMemset,
        cudaMemsetAsync,
        cudaLaunchKernel,
    };
    for (int i=0; i<sizeof(checkpoint_funcs)/sizeof(void*); i++) {
        void * func = checkpoint_funcs[i];
        so_lookup_func_info(dlip, func)->flags.checkpoint = 1;
    }
    void * sync_funcs[] = {
        cudaStreamSynchronize,
    };
    for (int i=0; i<sizeof(sync_funcs)/sizeof(void*); i++) {
        void * func = sync_funcs[i];
        so_lookup_func_info(dlip, func)->flags.sync = 1;
    }
    void * event_funcs[] = {
        cudaEventCreate,
        cudaEventCreateWithFlags,
        cudaEventRecord,
        cudaEventQuery,
        cudaEventDestroy,
    };
    for (int i=0; i<sizeof(event_funcs)/sizeof(void*); i++) {
        void * func = event_funcs[i];
        so_lookup_func_info(dlip, func)->flags.event = 1;
    }
    void * rwlock_funcs[] = {
        cudaMalloc,
        cudaFree,
    };
    for (int i=0; i<sizeof(rwlock_funcs)/sizeof(void*); i++) {
        void * func = rwlock_funcs[i];
        so_lookup_func_info(dlip, func)->flags.wrlock = 1;
    }
    void * async_funcs[] = {
        cudaMemset,
        cudaMemsetAsync,
        cudaMemcpy,
        cudaMemcpyAsync,
        cudaLaunchKernel,
        cublasSgemm_v2,
        cublasSgemv_v2,
    };
    for (int i=0; i<sizeof(async_funcs)/sizeof(void*); i++) {
        void * func = async_funcs[i];
        so_lookup_func_info(dlip, func)->flags.async = 1;
    }
    void * notrace_funcs[] = {
        cublasCreate_v2,
        cublasSetStream_v2,
        cudaSetDevice,
        cudaGetDevice,
        cudaGetLastError,
        __cudaPushCallConfiguration,
        __cudaPopCallConfiguration,
        __cudaRegisterVar,
        __cudaRegisterTexture,
        __cudaRegisterFunction,
        __cudaRegisterFatBinary,
        __cudaUnregisterFatBinary,
        cudaCreateChannelDesc,
        cudaGetDeviceCount,
        cudaGetDeviceProperties,
        cudaPointerGetAttributes,
        cudaHostAlloc,
    };
    for (int i=0; i<sizeof(notrace_funcs)/sizeof(void*); i++) {
        void * func = notrace_funcs[i];
        so_lookup_func_info(dlip, func)->flags.notrace = 1;
    }
}

void cudaw_so_begin_func(so_dl_info_t *dlip, int idx) {
    int wrlock = 0;
    const so_func_flags_t flags = dlip->funcs[idx].flags;
    if (so_tls.thread_idx == 0) {
        pthread_rwlock_wrlock(&so_func_rwlock);
        so_tls.thread_idx = next_idx(&next_thread_idx);
		wrlock = 1;
    }
    else if (flags.wrlock || so_request_for_checkpoint) {
        pthread_rwlock_wrlock(&so_func_rwlock);
		wrlock = 1;
    }
    else {
        pthread_rwlock_rdlock(&so_func_rwlock);
    }
    if (flags.notrace) {
    	so_tls.invoke_idx++;
		return;
	}
    if (so_request_for_checkpoint && !wrlock) {
	    pthread_rwlock_unlock(&so_func_rwlock);
       	pthread_rwlock_wrlock(&so_func_rwlock);
        wrlock = 1;
    }
	so_tls.invoke_idx++;
    so_tls.trace_idx = idx_next(&next_trace_idx);
    so_time(&so_tls.timestamp);
    if (so_request_for_checkpoint && 
            so_tls.thread_idx == so_forward_thread_idx &&
            flags.checkpoint) {
		so_try_pause_for_checkpoint(dlip->funcs[idx].func);
	}
}

void cudaw_so_end_func(so_dl_info_t *dlip, int idx) {
	if (so_tls.thread_idx == so_backward_thread_idx) {
        if (last_forward_trace_idx > last_backward_trace_idx) {
            last_forward_trace_len = last_forward_trace_idx - last_backward_trace_idx;
        }
		last_backward_trace_idx = so_tls.trace_idx;
		if (diff_sync_d2h_to_backward > 0) {
			last_sync_d2h_to_backward = diff_sync_d2h_to_backward;
			diff_sync_d2h_to_backward = 0;
			so_memcpy_kind = 0;
		}
	}
    else if (so_tls.thread_idx == so_forward_thread_idx) {
        if (last_backward_trace_idx > last_forward_trace_idx) {
            last_backward_trace_len = last_backward_trace_idx - last_forward_trace_idx;
        }
        last_forward_trace_idx = so_tls.trace_idx;
    }
    pthread_rwlock_unlock(&so_func_rwlock);
    next_idx(&dlip->funcs[idx].cnt);
    const so_func_flags_t flags = dlip->funcs[idx].flags;
    if (flags.notrace) {
        return;
    }
    if (flags.async || !flags.known) {
        so_tls.async = 1;
        sync_set(&so_checkpoint_thread_bits, (1 << so_tls.thread_idx));
    }
    uint8_t dli_idx = dlip->dli_idx;
    if (so_tls.async) {
        dli_idx |= 0x80;
    }
    else if (so_tls.event) {
        dli_idx |= 0x40;
    }
    uint8_t thread_idx = so_tls.thread_idx;
    if (!so_checkpoint_thread_bits) {
        thread_idx |= 0x80; // global_checkpoint
    }
    else if (!(so_checkpoint_thread_bits & (1 << so_tls.thread_idx))) {
        thread_idx |= 0x40; // thread_checkpoint
    }
    if (so_tls.trace_idx >= invoke_size) {
        size_t size = invoke_size * sizeof(so_invoke_t);
        munmap(so_invokes, size);
        invoke_size *= 2;
        size *= 2;
        so_invokes = so_mmap(fd_invoke, size);
    }
    so_invoke_t * p = &so_invokes[so_tls.trace_idx];
    p->invoke_idx = so_tls.invoke_idx;
    p->milliseconds = so_msec(&so_tls.timestamp) % 60000;
    p->thread_idx = thread_idx;
    p->dli_idx = dli_idx;
    p->func_idx = idx;
    so_print_invoke(so_tls.trace_idx, p, 0);
    so_tls.sbuf[0] = 0;
    if (1) {
        #define __step 40000
        static uint64_t next_checkpoint_idx = __step;
        if (so_tls.trace_idx == next_checkpoint_idx) {
            next_checkpoint_idx += __step;
            kill(getpid(), SIGPAUSE);
        }
        #undef __step
    }
}

void cudaw_so_register_dli(so_dl_info_t *dlip) {
    dlip->dli_idx = next_idx(&next_dli_idx);
    if (dlip->dli_idx >= sizeof(so_dlips)/sizeof(void*)) {
        errmsg("FAIL: too many dynamic libs (%d)!\n", dlip->dli_idx);
        exit(1);
        return;
    }
    so_dlips[dlip->dli_idx] = dlip;
    so_update_func_info(dlip);
}

void cudawrt_so_func_copy(void *funcs[]) {
    int i = 0;
    do {
        FCOPY(cudaMalloc)
        FCOPY(cudaFree)
    } while(0);
};

void cudawrt_so_func_swap(void *pfuncs[]) {
    int i = 0;
    do {
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
    } while(0);
};

void cudawblas_so_func_swap(void *pfuncs[]) {
    int i = 0;
    do {
        FSWAP(cublasSgemm_v2)
        FSWAP(cublasSgemv_v2)
    } while(0);
};

/*void cudawdnn_so_func_swap(void *pfuncs[]) {
    int i = 0;
    do {
        FSWAP(cudnnAddTensor)
        FSWAP(cudnnConvolutionBackwardBias)
        FSWAP(cudnnConvolutionBackwardData)
        FSWAP(cudnnConvolutionBackwardFilter)
        FSWAP(cudnnConvolutionForward)
        FSWAP(cudnnSetStream)
        FSWAP(cudnnGetStream)
    
    } while(0);
};*/

__attribute ((constructor)) void cudaw_trace_init(void) {
    printf("cudaw_trace_init\n");
    int r = pthread_rwlock_init(&so_func_rwlock, NULL);
    if (r != 0) {
        errmsg("init(&so_func_rwlock)\n");
        exit(1);
    }
	r = sem_init(&so_signal_sem, 0, 0);
    if (r != 0) {
        errmsg("sem_init(&so_signal_sem)\n");
        exit(1);
    }
	r = sem_init(&so_pause_sem, 0, 0);
    if (r != 0) {
        errmsg("sem_init(&so_pause_sem)\n");
        exit(1);
    }
	r = sem_init(&so_checkpoint_sem, 0, 0);
    if (r != 0) {
        errmsg("sem_init(&so_checkpoint_sem)\n");
        exit(1);
    }
    so_init_base_usec();
    mkdir(fn_trace_dir, 0777);
    fd_trace_dir = open(fn_trace_dir, O_DIRECTORY);
    if (fd_trace_dir == -1) {
        errmsg("%s\n", fn_trace_dir);
        exit(1);
    }
    fd_invoke = so_open(fd_trace_dir, fn_invoke);
    size_t size = invoke_size * sizeof(so_invoke_t);
    so_invokes = so_mmap(fd_invoke, size);
    so_start_checkpoint_deamon();
    so_start_signal_deamon();
    signal(SIGUSR1, signal_notifier_func);
    signal(SIGUSR2, signal_notifier_func);
    signal(SIGCONT, signal_notifier_func);
}

__attribute ((destructor)) void cudaw_trace_fini(void) {
    printf("cudaw_trace_fini\n");
    for (uint32_t i = 0; i < next_trace_idx; i++) {
        uint32_t cnt = 1;
        /*
        so_invoke_t * p = &so_invokes[i];
        for (uint32_t j = i+cnt; j < next_invoke_idx; j++) {
            so_invoke_t * q = &so_invokes[j];
            if (p->thread_idx != q->thread_idx ||
                p->dli_idx != q->dli_idx ||
                p->func_idx != q->func_idx)
                break;
            cnt++;
            i++;
            p = q;
        }
        so_print_invoke(i, &so_invokes[i], cnt);
        */
    }
    size_t size = invoke_size * sizeof(so_invoke_t);
    munmap(so_invokes, size);
    close(fd_invoke);
    close(fd_trace_dir);
	sem_destroy(&so_checkpoint_sem);
	sem_destroy(&so_pause_sem);
    // Don't destroy the so_signal_sem used by checkpoint deamon!
	// sem_destroy(&so_signal_sem);
    // Don't destroy the so_func_rwlock used by checkpoint deamon!
    // pthread_rwlock_destroy(&so_func_rwlock); 
}
