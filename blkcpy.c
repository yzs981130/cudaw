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
#include <semaphore.h>
#include <errno.h>
#include <assert.h>
#include <nvml.h>

#include "cudaw.h"

#define M BLK_SIZE

#define DEFSO(func)  static cudaError_t (*so_##func)
#define FSWAP(func) do {void **pp=pfuncs[i++]; so_##func=*pp; *pp=blkcpy_##func;} while(0);

DEFSO(cudaMemset)(void* devPtr, int value, size_t count);
DEFSO(cudaMemsetAsync)(void* devPtr, int value, size_t count, cudaStream_t stream);
DEFSO(cudaMemcpy)(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind);
DEFSO(cudaMemcpyAsync)(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream);

cudaError_t blkcpy_cudaMemset(void* devPtr, int value, size_t count) {
    cudaError_t r = cudaSuccess;
    void * ptr = devPtr;
    size_t cnt = count;
    int i = 0;
    if ((unsigned long long)ptr & (M - 1)) {
        size_t head = M - ((unsigned long long)ptr & (M - 1));
        if (head > cnt) {
            head = cnt;
        }
        r = so_cudaMemset(ptr, value, head);
        if (r != cudaSuccess) {
            return r;
        }
        cnt -= head;
        ptr += head;
    }
    for (; i < cnt/M; i++) {
        r = so_cudaMemset(ptr, value, M);
        if (r != cudaSuccess) {
            return r;
        }
        ptr += M;
    }
    if (r == cudaSuccess && cnt%M > 0) {
        r = so_cudaMemset(ptr, value, cnt%M);
    }
    return r;
}

cudaError_t blkcpy_cudaMemsetAsync(void* devPtr, int  value, size_t count, 
                                cudaStream_t stream) {
    cudaError_t r = cudaSuccess;
    void * ptr = devPtr;
    size_t cnt = count;
    int i = 0;
    if ((unsigned long long)ptr & (M - 1)) {
        size_t head = M - ((unsigned long long)ptr & (M - 1));
        if (head > cnt) {
            head = cnt;
        }
        r = so_cudaMemsetAsync(ptr, value, head, stream);
        if (r != cudaSuccess) {
            return r;
        }
        cnt -= head;
        ptr += head;
    }
    for (; i < cnt/M; i++) {
        r = so_cudaMemsetAsync(ptr, value, M, stream);
        if (r != cudaSuccess) {
            return r;
        }
        ptr += M;
    }
    if (r == cudaSuccess && cnt%M > 0) {
        r = so_cudaMemsetAsync(ptr, value, cnt%M, stream);
    }
    return r;
}

cudaError_t blkcpy_cudaMemcpy(void* dst, const void* src, size_t count, 
                                enum cudaMemcpyKind kind) {
    cudaError_t r = cudaSuccess;
    if (kind == cudaMemcpyDefault || kind == cudaMemcpyDeviceToDevice) {
        do {
            size_t cnt = count;
            if ((((size_t)dst & (M-1)) + cnt > M) &&
                (((size_t)src & (M-1)) + cnt > M)) {
                size_t cntdst = M - ((size_t)dst & (M-1));
                size_t cntsrc = M - ((size_t)src & (M-1));
                cnt = cntdst < cntsrc ? cntdst : cntsrc;
            }
            else if (((size_t)dst & (M-1)) + cnt > M) {
                cnt = M - ((size_t)dst & (M-1));
            }
            else if (((size_t)src & (M-1)) + cnt > M) {
                cnt = M - ((size_t)src & (M-1));
            }
            r = so_cudaMemcpy(dst, src, cnt, kind);
            if (r != cudaSuccess) {
                break;
            }
            count -= cnt;
            dst += cnt;
            src += cnt;
        } while (count > 0);
        return r;
    }
    if (kind == cudaMemcpyHostToHost) {
        r = so_cudaMemcpy(dst, src, count, kind);
        return r;
    }
    do {
        const void * ptr = (kind == cudaMemcpyDeviceToHost) ? src : dst;
        if ((unsigned long long)ptr & (M - 1)) {
            size_t head = M - ((unsigned long long)ptr & (M - 1));
            if (head > count) {
                head = count;
            }
            r = so_cudaMemcpy(dst, src, head, kind);
            if (r != cudaSuccess) {
                break;
            }
            count -= head;
            dst += head;
            src += head;
        }
        for (int i = 0; i < count/M; i++) {
            r = so_cudaMemcpy(dst, src, M, kind);
            if (r != cudaSuccess) {
                break;
            }
            dst += M;
            src += M;
        }
        if (r == cudaSuccess && count%M > 0) {
            r = so_cudaMemcpy(dst, src, count%M, kind);
        }
    } while (0);
    return r;
}

cudaError_t blkcpy_cudaMemcpyAsync(void* dst, const void* src, size_t count, 
                            enum cudaMemcpyKind kind, cudaStream_t stream) {
    cudaError_t r = cudaSuccess;
    if (kind == cudaMemcpyDefault || kind == cudaMemcpyDeviceToDevice) {
        do {
            size_t cnt = count;
            if ((((size_t)dst & (M-1)) + cnt > M) &&
                (((size_t)src & (M-1)) + cnt > M)) {
                size_t cntdst = M - ((size_t)dst & (M-1));
                size_t cntsrc = M - ((size_t)src & (M-1));
                cnt = cntdst < cntsrc ? cntdst : cntsrc;
            }
            else if (((size_t)dst & (M-1)) + cnt > M) {
                cnt = M - ((size_t)dst & (M-1));
            }
            else if (((size_t)src & (M-1)) + cnt > M) {
                cnt = M - ((size_t)src & (M-1));
            }
            r = so_cudaMemcpyAsync(dst, src, cnt, kind, stream);
            if (r != cudaSuccess) {
                break;
            }
            count -= cnt;
            dst += cnt;
            src += cnt;
        } while (count > 0);
        return r;
    }
    if (kind == cudaMemcpyHostToHost) {
        r = so_cudaMemcpyAsync(dst, src, count, kind, stream);
        return r;
    }
    do {
        const void * ptr = (kind == cudaMemcpyDeviceToHost) ? src : dst;
        if ((unsigned long long)ptr & (M - 1)) {
            size_t head = M - ((unsigned long long)ptr & (M - 1));
            if (head > count) {
                head = count;
            }
            r = so_cudaMemcpyAsync(dst, src, head, kind, stream);
            if (r != cudaSuccess) {
                break;
            }
            count -= head;
            dst += head;
            src += head;
        }
        for (int i = 0; i < count/M; i++) {
            r = so_cudaMemcpyAsync(dst, src, M, kind, stream);
            if (r != cudaSuccess) {
                break;
            }
            dst += M;
            src += M;
        }
        if (r == cudaSuccess && count%M > 0) {
            r = so_cudaMemcpyAsync(dst, src, count%M, kind, stream);
        }
    } while (0);
    return r;
}

void cudawrt_blkcpy_func_swap(void *pfuncs[]) {
    int i = 0;
    do {
        FSWAP(cudaMemset)
        FSWAP(cudaMemsetAsync)
        FSWAP(cudaMemcpy)
        FSWAP(cudaMemcpyAsync)
    } while(0);
};

