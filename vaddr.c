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

#include "cudawrt.h"

#define ADDR_MASK 0x7fffffffffffffffull
#define ADDR_FLAG 0x8000000000000000ull


#define DEFSO(func)  static cudaError_t (*so_##func)

#define LDSYM(func)  do { \
    so_##func = dlsym(so_handle, #func); \
    printerr(); \
} while(0)

DEFSO(cudaMalloc)(void** devPtr, size_t bytesize);
DEFSO(cudaFree)(void* devPtr);

static void * so_handle = NULL;

static void printerr() {
    char *errstr = dlerror();
    if (errstr != NULL) {
        printf ("A dynamic linking error occurred: (%s)\n", errstr);
    }
}

__attribute ((constructor)) void cudaw_vaddr_init(void) {
    printf("cudaw_vaddr_init\n");
    so_handle = dlopen (LIB_STRING_RT, RTLD_NOW);
    if (!so_handle) {
        fprintf (stderr, "FAIL: %s\n", dlerror());
        exit(1);
    }
    LDSYM(cudaMalloc);
    LDSYM(cudaFree);
}

__attribute ((destructor)) void cudaw_vaddr_fini(void) {
    printf("cudaw_vaddr_fini\n");
    if (so_handle) {
        dlclose(so_handle);
    }
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

#ifdef VA_TEST_DEV_ADDR
static void * baseDevAddr = NULL;
static size_t usedBytes = 0;
static size_t totalBytes = 1024 * 1024 * 1024;

int cudawIsDevAddr(const void * devAddr) {
    printf("cudawIsDevAddr: %p <= %p < %p\n", baseDevAddr, devAddr, baseDevAddr + usedBytes);
    if (devAddr >= baseDevAddr && devAddr < baseDevAddr + usedBytes)
        return 1;
    return 0;
}
#endif

cudaError_t cudawMalloc(void ** devPtr, size_t bytesize) {
    static int cnt = 0;
    cnt++;
    //printf("cudaMalloc:\n");
    //printf("before:devPtr:%p,size:%zu\n",*devPtr,bytesize);
    cudaError_t r = 0;
#ifdef VA_TEST_DEV_ADDR
    if (baseDevAddr == NULL) {
        void * dps[64];
        for (int i = 0; i < 64; i++) {
            r = so_cudaMalloc(dps+i, (32 * 1024 * 1024));
            if (((unsigned long long)dps[i] & 0x3fffffffull) == 0x0ull) {
                baseDevAddr = dps[i];
                 printf("cudaMalloc %p\n", baseDevAddr);
            }
        }
        int n = 64;
        for (int i = 0; i < 64; i++) {
            if (dps[i] < baseDevAddr) {
                so_cudaFree(dps[i]);
                n--;
            }
            if (dps[i] > (baseDevAddr+0x3fffffffull)) {
                so_cudaFree(dps[i]);
                n--;
            }
        }
        if (n != 32) {
            printf("cudaMalloc %p %d\n", baseDevAddr, n);
            return cudaErrorMemoryAllocation;
        }
        *devPtr = baseDevAddr;
        usedBytes += bytesize;
    } else {
        *devPtr = baseDevAddr + usedBytes;
        usedBytes += bytesize;
    }
#else
    r = so_cudaMalloc(devPtr, bytesize);
#endif
    printf("after:no:%d,devPtr:%p,size:%zu, return %d\n",cnt,*devPtr,bytesize,r);
    //add[cnt++]=*devPtr;
    //*devPt1=(*devPtr+offset);

#ifdef VA_ENABLE_VIR_ADDR
    if(cnt>=0) {
        *devPtr = cudawDevAddrToVir(*devPtr);
        //printf("add:%p,devPtr:%p\n",add,*devPtr);
    }
#endif

    return r;
}

cudaError_t cudawFree(void* devPtr) {
    printf("cudawFree: %p (before)\n", devPtr);
    //cudaError_t r = so_cudaFree(devPtr);
    //printf("cudawFree: %p (after)\n ", devPtr);
    //return r;
    return cudaSuccess;
}

