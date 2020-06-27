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


#include "cudawrt.h"
#include "vaddr.h"

#define ADDR_MASK 0x7fffffffffffffffull
#define ADDR_FLAG 0x8000000000000000ull


#define DEFSO(func)  static cudaError_t (*so_##func)

#define LDSYM(func)  do { \
    so_##func = dlsym(so_handle, #func); \
    printerr(); \
} while(0)

DEFSO(cudaMalloc)(void** devPtr, size_t bytesize);
DEFSO(cudaFree)(void* devPtr);
DEFSO(cudaMemGetInfo)(size_t* free , size_t* total);
DEFSO(cudaMallocHost)(void** ptr, size_t size);
DEFSO(cudaFreeHost)(void* ptr);
DEFSO(cudaMemset)(void* devPtr, int value, size_t count);
DEFSO(cudaMemsetAsync)(void* devPtr, int value, size_t count, cudaStream_t stream);
DEFSO(cudaMemcpy)(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind);
DEFSO(cudaMemcpyAsync)(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream);
DEFSO(cudaDeviceSynchronize)();

static void * so_handle = NULL;
static pthread_rwlock_t va_rwlock;

static void printerr() {
    char *errstr = dlerror();
    if (errstr != NULL) {
        printf ("A dynamic linking error occurred: (%s)\n", errstr);
    }
}

__attribute ((constructor)) void cudaw_vaddr_init(void) {
    printf("cudaw_vaddr_init\n");
    int r = pthread_rwlock_init(&va_rwlock, NULL);
    if (r != 0) {
        int eno = errno;
        fprintf(stderr, "FAIL: pthread_rwlock_init return %d. (errno=%d) %s\n",
            r, eno, strerror(eno));
        exit(r);
    }
    so_handle = dlopen(LIB_STRING_RT, RTLD_NOW);
    if (!so_handle) {
        fprintf(stderr, "FAIL: %s\n", dlerror());
        exit(1);
    }
    LDSYM(cudaMemGetInfo);
    LDSYM(cudaMalloc);
    LDSYM(cudaFree);
    LDSYM(cudaMallocHost);
    LDSYM(cudaFreeHost);
    LDSYM(cudaMemset);
    LDSYM(cudaMemsetAsync);
    LDSYM(cudaMemcpy);
    LDSYM(cudaMemcpyAsync);
    LDSYM(cudaDeviceSynchronize);

    //void *p;
    //cudaMalloc(&p, 0);
}

__attribute ((destructor)) void cudaw_vaddr_fini(void) {
    printf("cudaw_vaddr_fini\n");
    if (so_handle) {
        dlclose(so_handle);
    }
    pthread_rwlock_destroy(&va_rwlock);
}

void * cudawVirAddrToDev(void * virAddr) {
    if ((unsigned long long)virAddr & ADDR_FLAG) {
        return (void*)((unsigned long long)virAddr & ADDR_MASK);
    }
    return virAddr;
}

void * cudawDevAddrToVir(void * devAddr) {
    return (void*)((unsigned long long)devAddr | ADDR_FLAG);
}

#ifndef VA_TEST_DEV_ADDR
  #define VA_TEST_DEV_ADDR
#endif

#ifdef VA_TEST_DEV_ADDR

#ifndef VA_DEV_TOTAL_BYTES
  //#define VA_DEV_TOTAL_BYTES    0x100000000ull  // 4GB
  #define VA_DEV_TOTAL_BYTES    0x200000000ull  // 8GB
#endif

#ifndef VA_DEV_LOCK_BYTES
  //#define VA_DEV_LOCK_BYTES     0x0ull  // 0B
  //#define VA_DEV_LOCK_BYTES     0x40000000ull  // 1GB
  #define VA_DEV_LOCK_BYTES     0x10000000ull  // 256MB
#endif

#ifndef VA_ALIGNMENT
  //#define VA_ALIGNMENT          0x100000000ull // 4GB
  //#define VA_ALIGNMENT          0x10000000ull // 256MB
  #define VA_ALIGNMENT          0x40000000ull // 1GB
  //#define VA_ALIGNMENT          0x80000000ull // 2GB
  //#define VA_ALIGNMENT          0x2000000ull // 32MB
#endif

#ifndef VA_MALLOC_BLOCK
  #define VA_MALLOC_BLOCK       0x2000000ull // 32MB
#endif

// TODO to support multiple devices

static void * devBaseAddr =     NULL;
static void * devOldBaseAddr =  NULL;
static size_t devUsedBytes =    0x0ull;
static size_t devLockBytes =    0x0ull;
static size_t devTotalBytes =   VA_DEV_TOTAL_BYTES; // 4GB

static int vaIsAlignedBaseAddr(void * ptr) {
    if (devOldBaseAddr != NULL)
        return (ptr == devOldBaseAddr);
    return (((unsigned long long)ptr & (VA_ALIGNMENT-1)) == 0x0ull);
}

int cudawIsDevAddr(const void * devAddr) {
    printf("cudawIsDevAddr: %p <= %p < %p\n", 
                devBaseAddr, devAddr, devBaseAddr + devUsedBytes);
    if (devAddr >= devBaseAddr && devAddr < devBaseAddr + devUsedBytes)
        return 1;
    return 0;
}

static int vaMallocBlocks(void * dps[], int n) {
    cudaError_t r = cudaSuccess;
    void * min_devptr = NULL;
    void * max_devptr = NULL;
    for (int i = 0; i < n; i++) {
        void * devptr;
        r = so_cudaMalloc(&devptr, VA_MALLOC_BLOCK);
        if (r == cudaSuccess) {
            dps[i] = devptr;
            if (max_devptr <= devptr)
                max_devptr = devptr + VA_MALLOC_BLOCK;
            if (min_devptr == NULL || devptr < min_devptr)
                min_devptr = devptr;
            continue;
        }
        else if (r == cudaErrorInvalidValue) {
            fprintf(stderr, "vaMallocBlocks: %s return %s\n",
                            "so_cudaMalloc", "cudaErrorInvalidValue");
        }
        else if (r == cudaErrorMemoryAllocation) {
            fprintf(stderr, "vaMallocBlocks: %s return %s\n",
                            "so_cudaMalloc", "cudaErrorMemoryAllocation");
        }
        else {
            fprintf(stderr, "vaMallocBlocks: %s return %d\n",
                            "so_cudaMalloc", r);
        }
        n = i;
        break;
    }
    printf("vaMallocBlocks: min: %p max: %p cnt: %d\n",
                min_devptr, max_devptr, n);
    return n;
}

static int compare_ptr(const void * pa, const void * pb) {
    if (*(void **)pb < *(void **)pa)
        return 1;
    else if (*(void **)pb > *(void **)pa)
        return -1;
    else
        return 0;
}

static void vaSortBlocks(void * dps[], int n) {
    qsort(dps, n, sizeof(void *), compare_ptr);
}

static cudaError_t vaReallocBlock(void * blkptr) {
    cudaError_t r = cudaSuccess;
    size_t free, total;
    r = so_cudaMemGetInfo(&free, &total);
    if (r != cudaSuccess) {
        return r;
    }
    int n = (int)(free / VA_MALLOC_BLOCK);
    void * dps[n];
    n = vaMallocBlocks(dps, n);
    for (int i = 0; i < n; i++) {
        if (dps[i] != blkptr) {
            so_cudaFree(dps[i]);
        }
    }
    return r;
}

static cudaError_t vaPreMalloc(void) {
    cudaError_t r = cudaSuccess;
    if (devBaseAddr == NULL) {
        size_t free, total;
        r = so_cudaMemGetInfo(&free, &total);
        if (r != cudaSuccess) {
            return r;
        }
        int n = (int)(free / VA_MALLOC_BLOCK);
        void * dps[n];
        n = vaMallocBlocks(dps, n);
        vaSortBlocks(dps, n);
        for (int i = 0; i < n; i++) {
            if (vaIsAlignedBaseAddr(dps[i])) {
//printf("=== -- %3d %p\n", i, dps[i]); fflush(stdout);
                devBaseAddr = dps[i];
                int m = (devTotalBytes + VA_MALLOC_BLOCK - 1) / VA_MALLOC_BLOCK;
                for (int k = 1; k < m; k++) {
//printf("=== .. %3d %p\n", k, dps[i]); fflush(stdout);
                    if (dps[i+k] != devBaseAddr + (VA_MALLOC_BLOCK * k)) {
                        devBaseAddr = NULL;
                        break;
                    }
                }
                if (devBaseAddr != NULL) {
printf("=== == %3d %p\n", i, dps[i]); fflush(stdout);
                    break;
                }
            }
        }
        for (int i = 0; i < n; i++) {
            if (dps[i] < devBaseAddr) {
                so_cudaFree(dps[i]);
            }
            else if (dps[i] >= (devBaseAddr + devTotalBytes)) {
                so_cudaFree(dps[i]);
            }
        }
        if (devBaseAddr == NULL) {
            fprintf(stderr, "FAIL: vaPreMalloc(%lx)\n", devTotalBytes);
            for (int i = 0; i < n; i++) {
                fprintf(stderr, "%d %p\n", i, dps[i]);
            }
            r = cudaErrorMemoryAllocation;
        }
    }
    if (devBaseAddr != NULL) {
        if (devOldBaseAddr == NULL) {
           devOldBaseAddr = devBaseAddr;
        }
    }
    return r;
}

cudaError_t vaMalloc(void ** devPtr, size_t bytesize) {
    cudaError_t r = cudaSuccess;
    pthread_rwlock_wrlock(&va_rwlock);
    if (devBaseAddr == NULL) {
        r = vaPreMalloc();
        if (r != cudaSuccess) {
            pthread_rwlock_unlock(&va_rwlock);
            return r;
        }
    }
    if (devUsedBytes + bytesize <= devTotalBytes) {
        if (bytesize < 0x100000) {
            devUsedBytes = ((devUsedBytes + 0xfff) & ~0xfffull);
        }
        else {
            devUsedBytes = ((devUsedBytes + 0xfffff) & ~0xfffffull);
        }
        *devPtr = devBaseAddr + devUsedBytes;
        devUsedBytes += bytesize;
    }
    else {
        r = cudaErrorMemoryAllocation;
    }
    pthread_rwlock_unlock(&va_rwlock);
    return r;
}

void vaFreeAndRealloc(void) {
    cudaError_t r = cudaSuccess;
    pthread_rwlock_wrlock(&va_rwlock);
    so_cudaDeviceSynchronize();
    int m = (devTotalBytes + VA_MALLOC_BLOCK - 1) / VA_MALLOC_BLOCK;
    void *p = NULL;
    void *used = NULL;
    if (devUsedBytes > 0) {
        r = so_cudaMallocHost(&used, devUsedBytes);
        if (r != cudaSuccess) {
            fprintf(stderr, "FAIL: vaFreeAndRealloc - cudaMallocHost %p %lu\n",
                        used, devUsedBytes);
            //exit(r);
        }
        r = cudawMemcpy(used, devBaseAddr, devUsedBytes, 
                        cudaMemcpyDeviceToHost);
        if (r != cudaSuccess) {
            fprintf(stderr, "FAIL: vaFreeAndRealloc -> cudaMemcpy %p %lx %d\n",
                        used, devUsedBytes, r);
            //exit(r);
        }
    }
    do {
        if (devBaseAddr != NULL) {
            for (int k = 0; k < m; k++) {
                void * dp = devBaseAddr + (VA_MALLOC_BLOCK * k);
                so_cudaFree(dp);
//fprintf(stderr, "%d - %p %lx\n", k, dp, devLockBytes);
            }
            //so_cudaFree(devBaseAddr + VA_MALLOC_BLOCK);
            //so_cudaFree(devBaseAddr + VA_MALLOC_BLOCK*2);
            //so_cudaFree(devBaseAddr + VA_MALLOC_BLOCK*3);
            devBaseAddr = NULL;
        }

        sleep(30);


        if (cudaSuccess != vaPreMalloc()) {
            fprintf(stderr, "FAIL: vaFreeAndRealloc %p %p\n",
                        devBaseAddr, devOldBaseAddr);
            continue;
        }
        if (devBaseAddr == devOldBaseAddr) {
printf("=== xx %p %p\n", devBaseAddr, devOldBaseAddr);
            break;
        }
printf("=== ++ %p %p\n", devBaseAddr, devOldBaseAddr);
    }
    while (1);
    if (used != NULL) {
        r = cudawMemcpy(devBaseAddr, used, devUsedBytes, 
                          cudaMemcpyHostToDevice);
        if (r != cudaSuccess) {
            fprintf(stderr, "FAIL: vaFreeAndRealloc <- cudaMemcpy %p %lx %d\n",
                        used, devUsedBytes, r);
            //exit(r);
        }
        so_cudaFreeHost(used);
    }
    pthread_rwlock_unlock(&va_rwlock);
}

#endif // VA_TEST_DEV_ADDR


cudaError_t cudawMalloc(void ** devPtr, size_t bytesize) {
    cudaError_t r = cudaSuccess;
    static int cnt = 0;
    cnt++;

#ifdef VA_TEST_DEV_ADDR
    pthread_rwlock_unlock(&va_rwlock);
    r = vaMalloc(devPtr, bytesize);
    //vaFreeAndRealloc();
    pthread_rwlock_rdlock(&va_rwlock);
#else
    r = so_cudaMalloc(devPtr, bytesize);
#endif

#ifdef VA_ENABLE_VIR_ADDR
    *devPtr = cudawDevAddrToVir(*devPtr);
#endif

    printf("cudawMalloc: %p %dm %d cnt: %d r: %d (%lx)\n", *devPtr, 
                (int)(bytesize>>20), (int)(bytesize & 0xffffful), cnt, r,
                devUsedBytes);
    return r;
}

cudaError_t cudawFree(void* devPtr) {
#ifdef VA_TEST_DEV_ADDR
    printf("WARN: cudawFree: %p\n", devPtr);
    return cudaSuccess;
#else
    return so_cudaFree(devPtr);
#endif
}

cudaError_t cudawMemGetInfo(size_t* free , size_t* total) {
#ifdef VA_TEST_DEV_ADDR
    *free = devTotalBytes - devUsedBytes;
    *total = devTotalBytes;
    return cudaSuccess;
#else
    return so_cudaMemGetInfo(free, total);
#endif
}

cudaError_t cudawMemset(void* devPtr, int value, size_t count) {
    cudaError_t r = cudaSuccess;
    printf("cudaMemset ptr: %p val:%d cnt: %ld\n", devPtr, value, count);
    do {
        size_t M = VA_MALLOC_BLOCK;
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
                break;
            }
            cnt -= head;
            ptr += head;
        }
        for (; i < cnt/M; i++) {
            r = so_cudaMemset(ptr, value, M);
            if (r != cudaSuccess) {
                break;
            }
            ptr += M;
        }
        if (r == cudaSuccess && cnt%M > 0) {
            r = so_cudaMemset(ptr, value, cnt%M);
        }
    } while (0);
    return r;
}

cudaError_t cudawMemsetAsync(void* devPtr, int  value, size_t count, 
                                cudaStream_t stream) {
    cudaError_t r = cudaSuccess;
    printf("cudaMemsetAsync ptr: %p val:%d cnt: %ld %p\n", 
                devPtr, value, count, stream);
    do {
        size_t M = VA_MALLOC_BLOCK;
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
                break;
            }
            cnt -= head;
            ptr += head;
        }
        for (; i < cnt/M; i++) {
            r = so_cudaMemsetAsync(ptr, value, M, stream);
            if (r != cudaSuccess) {
                break;
            }
            ptr += M;
        }
        if (r == cudaSuccess && cnt%M > 0) {
            r = so_cudaMemsetAsync(ptr, value, cnt%M, stream);
        }
    } while (0);
    return r;
}

static const char * memcpyKinds[] = {
    "Host -> Host",
    "Host -> Device",
    "Device -> Host",
    "Device -> Device",
    "cudaMemcpyDefault"
};

cudaError_t cudawMemcpy(void* dst, const void* src, size_t count, 
                                enum cudaMemcpyKind kind) {
    cudaError_t r = cudaSuccess;
    printf("cudaMemcpy dst: %p, src: %p, cnt: %lu, kind: %s\n", 
                       dst, src, count, memcpyKinds[kind]);
    int devdst = cudawIsDevAddr(dst);
    int devsrc = cudawIsDevAddr(src);
    if (devdst && devsrc) {
        size_t M = VA_MALLOC_BLOCK;
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
    }
    if (!devdst && !devsrc) {
        r = so_cudaMemcpy(dst, src, count, kind);
    }
    else do {
        size_t M = VA_MALLOC_BLOCK;
        const void * ptr = devdst ? dst : src;
        int i = 0;
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
        for (; i < count/M; i++) {
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

cudaError_t cudawMemcpyAsync(void* dst, const void* src, size_t count, 
                            enum cudaMemcpyKind kind, cudaStream_t stream) {
    cudaError_t r = cudaSuccess;
    printf("cudaMemcpyAsync dst: %p, src: %p, cnt: %lu, kind: %s (%p)\n", 
                             dst, src, count, memcpyKinds[kind], stream);
    int devdst = cudawIsDevAddr(dst);
    int devsrc = cudawIsDevAddr(src);
    if (devdst && devsrc) {
        size_t M = VA_MALLOC_BLOCK;
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
    }
    if (!devdst && !devsrc) {
        r = so_cudaMemcpyAsync(dst, src, count, kind, stream);
    }
    else do {
        size_t M = VA_MALLOC_BLOCK;
        const void * ptr = devdst ? dst : src;
        int i = 0;
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
        for (; i < count/M; i++) {
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

void cudawMemLock(void) {
    pthread_rwlock_rdlock(&va_rwlock);
}

void cudawMemUnlock(void) {
    pthread_rwlock_unlock(&va_rwlock);
}
