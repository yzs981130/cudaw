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

#include "cudaw.h"

typedef struct trace_invoke_t {
    uint16_t invoke_idx;
	uint16_t milliseconds;
	uint8_t  thread_idx;
	uint8_t  dli_idx;
	uint16_t func_idx;
} trace_invoke_t;

typedef struct trace_alloc_t {
    void    *devptr;
    uint8_t  flags;
    uint8_t  total;
    uint16_t used;
    uint32_t size;
} trace_alloc_t;

typedef struct trace_tls_t {
	uint8_t         thread_idx;
    uint8_t         wrlock;
    uint8_t         async;
    uint8_t         event;
    uint32_t        trace_idx;
    uint32_t        recover_idx;
    uint64_t        invoke_idx;
    struct timeval  timestamp;
    char            sbuf[1024];
} trace_tls_t;

static time_t    base_sec;            // the base time of seconds when start
static uint32_t  next_thread_idx = 0; // next index of new found thread
static uint64_t  next_trace_idx = 0;  // next index of traced invoke in .trace
static uint8_t   next_dli_idx = 0;    // next index of registered dynamic lib
static uint64_t  last_forward_len = 0;
static uint64_t  last_forward_idx = 0;
static uint64_t  last_backward_len = 0;
static uint64_t  last_backward_idx = 0;
static uint64_t  diff_sync_to_backward = 0;
static uint64_t  last_sync_to_backward = 0;

enum {
    SIGPAUSE   = SIGUSR1, // 10
    SIGMIGRATE = SIGUSR2, // 12
    SIGRESUME  = SIGCONT, // 18
};

static volatile sig_atomic_t sig_command = 0;
static volatile sig_atomic_t sig_last_command = 0;
static volatile sig_atomic_t sig_repeat = 0;

#define next_idx(pidx) __sync_add_and_fetch(pidx, 1)
#define idx_next(pidx) __sync_fetch_and_add(pidx, 1)

#define sync_set(ptr, v)   __sync_fetch_and_or(ptr, v)
#define sync_clear(ptr, v) __sync_fetch_and_and(ptr, ~(v))


static const char *fn_trace_dir = ".trace";
static const char *fn_recover_dir = ".recover";
static const char *fn_invoke = "invoke.trace";
static const char *fn_alloc = "alloc.trace";
static const char *fn_memcpy = "memcpy.trace";
static const char *fn_kernel = "kernel.trace";
static const char *fn_devmem = "devmem.data";

static int fd_trace_dir = -1;
static int fd_recover_dir = -1;
static int fd_invoke = -1;
static int fd_alloc = -1;
static int fd_memcpy = -1;
static int fd_kernel = -1;
static int fd_devmem = -1;

static int             recover_mode = 0;
static int             checkpoint_mode = 0;

static trace_invoke_t *ti_invokes = NULL;
static size_t          ti_max = 1024 * 1024;
static size_t          ti_size = 1024 * 1024 * sizeof(trace_invoke_t);
static size_t          recover_trace_idx = 0;

// data, size and pos of alloc.trace
static void    *ta_data = NULL;
static size_t   ta_size = 4096;
static size_t   ta_pos = 0;
static size_t   ta_recover = 0;
static size_t   ta_freed = 0;

// data, size and pos of devmem.data
static void    *dm_data = NULL;
static size_t   dm_size = 0x800000000lu;
static size_t   dm_pos = 0;
static size_t   dm_recover = 0;

/*
We lock trace_rwlock for read in _begin_func and unlock it in _end_func.
When we found a new thread, we let the new thread waiting for a write lock on 
trace_rwlock, so as that the first invoke in the new thread is known after 
other invokes. We use trace_rwlock to guarantee that we do not make a 
checkpoint during any invokes.
*/
static pthread_rwlock_t trace_rwlock;

static uint64_t checkpoint_thread_bits = 0;

static uint8_t backward_thread_idx = 0xff;
static uint8_t forward_thread_idx = 1;
static uint8_t trace_memcpy_kind = 0; // last memcpy kind in forward round
static int     request_for_checkpoint = 0;
static sem_t   sem_signal;
static sem_t   sem_pause;
static sem_t   sem_checkpoint;

static __thread trace_tls_t so_tls = {0}; // Thread local storage

static cudaEvent_t  so_events[1024] = {0};
static int          so_eventc       = 0;

static so_dl_info_t *so_dlips[16] = {0};  // Registration of all traced dynamic libs

#define DEFSO(func) static cudaError_t(*so_##func)
#define FSWAP(func) do {void **pp=pfuncs[i++]; so_##func=*pp; *pp=trace_##func;} while(0);
#define FCOPY(func) do {void *p=funcs[i++]; so_##func=p;} while(0);

DEFSO(cudaGetLastError)(void);
DEFSO(cudaMemGetInfo)(size_t* free , size_t* total);
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


#define TA_ALLOC    0x1u
#define TA_FREE     0x2u
#define TA_FREED    0x4u
#define TA_REUSED   0x8u
#define TA_BLOCK    0x10u
#define TA_FRACTION 0x20u
#define TA_INUSE    0x40u
#define TA_OOM      0x80u

#define TA_BLOCK_SIZE     (32 * 1024 * 1024lu)
#define TA_FRACTION_SIZE  (32 * 1024lu)

#define ta_total_to_val(x) (((size_t)(0x1))<<(x))
#define ta_size_to_val(x)  ((x)&0x80000000?(size_t)((x)&0x7fffffff)<<16:(x))
#define ta_val_to_size(v) ((v)>=0x80000000u?(((v)+0xffffu)>>16)+0x80000000:(v))

static size_t ta_used_to_val(uint16_t used) {
    if (used < 0x800) {
        return used;
    }
    else {
        return ((size_t)(used & 0x3ff) + 0x400) << ((used >> 10) - 1);
    }
}

static uint16_t ta_val_to_used(size_t v) {
    if (v < (1<<11)) {
        return (uint16_t)(v);
    }
    uint16_t e = 0;
    while (v >= (1<<11)) {
        v = (v + 1) >> 1;
        e++;
    }
    return (uint16_t)((e << 10) + v);
}

static uint8_t ta_val_to_total(size_t v) {
    uint8_t e = 0;
    uint64_t ev = 1;
    while (ev<v) {
        ev <<= 1;
        e++;
    }
    return e;
}

static size_t ta_total_val(size_t val) {
    uint8_t total = ta_val_to_total(val);
    return ta_total_to_val(total);
}

static size_t ta_used_val(size_t val) {
    uint16_t used = ta_val_to_used(val);
    return ta_used_to_val(used);
}

static size_t ta_size_val(size_t val) {
    size_t size = ta_val_to_size(val);
    return ta_size_to_val(size);
}

static void trace_time(struct timeval *pnow) {
	static struct timeval now = {0};
    if (gettimeofday(pnow, NULL) == 0) {
		now = *pnow;
	    return;
	}
	*pnow = now;
}

static void init_base_usec(void) {
	struct timeval now;
	trace_time(&now);
	base_sec = now.tv_sec;
}

static uint32_t trace_msec(struct timeval *pnow) {
    uint32_t msec = (pnow->tv_sec - base_sec) * 1000;
    msec += pnow->tv_usec / 1000;
    return msec % 60000;
}

static FILE * trace_fcreate(int fdir, const char *fname) {
    int oflag = O_RDWR | O_CREAT | O_TRUNC;
    int fd = openat(fdir, fname, oflag, 0660);
	FILE *file = NULL;
    if (fd != -1) {
		file = fdopen(fd, "w+");
		if (file == NULL) {
			close(fd);
		}
	}
	return file;
}

static FILE * trace_fopen(int fdir, const char *fname) {
    int oflag = O_RDWR;
    int fd = openat(fdir, fname, oflag, 0660);
	FILE *file = NULL;
    if (fd != -1) {
		file = fdopen(fd, "r+");
		if (file == NULL) {
			close(fd);
		}
	}
	return file;
}

static int trace_open(int fdir, const char *fname) {
    int oflag = O_RDWR | O_CREAT/* | O_TRUNC*/;
    int fd = openat(fdir, fname, oflag, 0660);
    if (fd == -1) {
        errmsg("%s\n", fname);
        exit(1);
	}
	return fd;
}

static void *trace_mmap(int fd, size_t size) {
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

static void * ta_next_element(void) {
    assert(so_tls.wrlock);
    if (ta_pos >= ta_size) {
        munmap(ta_data, ta_size);
        ta_size *= 2;
        ta_data = trace_mmap(fd_alloc, ta_size);
    }
    void * p = ta_data + ta_pos;
    ta_pos += sizeof(trace_alloc_t);
    return p;
}

static void **   dm_dps = NULL;
static uint8_t * dm_flags = NULL;
static int       dm_max = 0;
static int       dm_dpc = 0;
static int       dm_blk = 0;


static char    dirmem[8 * 4096] = {0};
static uint8_t devmem[8 * 4096] = {0};
static void *  devptrs[32 * 1024] = {0};

#define DM_DPC        128
#define DM_SHIFT      25
#define DM_MASK       0xfffe000000lu
#define dm_idx(a)     (((size_t)(a)&DM_MASK)>>DM_SHIFT)

static void * minp = (void *)0xfffffffffffflu;
static void * maxp = NULL;
extern void print_mem_maps(void * start, void * end);

static void check_re_cudaMalloc(void) {
    cudaError_t r;
    static int entries = 0;
    int i, n, m, M = 1;
    if (dm_dps == NULL) {
        size_t free, total;
        so_cudaMemGetInfo(&free, &total);
        dm_max = (total + TA_BLOCK_SIZE-1) / TA_BLOCK_SIZE;
        dm_dps = malloc(sizeof(void*) * dm_max);
        memset(dm_dps, 0, sizeof(void*) * dm_max);
        dm_flags = malloc(dm_max);
        memset(dm_flags, 0, dm_max);
        M = 2;
    }
    for (m=M; m>0; m--) {
        uint8_t dm[32 * 1024] = {0};
        void ** dps = dm_dps;
        int miss_match = 0;
        for (i=dm_dpc, n=dm_max; i<n; i++) {
            void * devptr = NULL;
            r = so_cudaMalloc(&devptr, TA_BLOCK_SIZE);
            if (r == cudaErrorMemoryAllocation) {
                n = i;
                break;
            }
            if (devptr > maxp) maxp = devptr;
            if (devptr < minp) minp = devptr;
            if (!miss_match && dps[i] != NULL && dps[i] != devptr) {
                printf("dm_dps_miss_match: %d oldptr %p newptr %p\n",
                            i, dps[i], devptr);
                miss_match = 1;
            }
            dps[i] = devptr;
        }
        for (i=dm_dpc; i+1<n; i++) {
            if (i == 0) i++;
            int ip = dm_idx(dps[i-1]);
            int ii = dm_idx(dps[i]);
            int in = dm_idx(dps[i+1]);
            if (ip + 1 == ii && ii + 1 == in) {
                dm_flags[i] = 1;
            }
            else if (ip - 1 == ii && ii - 1 == in) {
                dm_flags[i] = 2;
            }
            else {
                dm_flags[i] = 0;
            }
        }
        dm_flags[n-1] = 0;
        int direction = dm_flags[dm_dpc];
        for (i=dm_dpc; i<n; i++) {
            if (dm_flags[i] == 0 || i==dm_dpc) {
                if (direction == 2) {
                    printf("dm_dps[%d] - %p\n", i, dps[i]);
                }
                else if (direction == 1) {
                    printf("dm_dps[%d] + %p\n", i, dps[i]);
                }
                else {
                    printf("dm_dps[%d] : %p\n", i, dps[i]);
                }
            }
            direction = dm_flags[i];
        }
        if (m == 1) {
            int old_dpc = dm_dpc;
            dm_dpc = (dm_blk + DM_DPC) / DM_DPC * DM_DPC;
            if (dm_dpc > n) {
                dm_dpc = n;
            }
            for (int i=old_dpc; i<dm_dpc; i++) {
                dm_flags[i] = TA_BLOCK;
            }
        }
        for (i=n-1; i>=dm_dpc; i--) {
            so_cudaFree(dps[i]);
        }
        if (miss_match) {
            print_mem_maps(minp, maxp);
        }
    }
    so_cudaGetLastError();
    entries++;
}

static int dm_save(void) {
    cudaError_t r = cudaSuccess;
    trace_alloc_t * begin = ta_data;
    trace_alloc_t * end = (ta_data + ta_pos);
    dm_pos = 0;
    int kind = cudaMemcpyDeviceToHost;
    for (trace_alloc_t * p = begin; p < end; p++) {
        if ((p->flags & TA_BLOCK) && p->used) {
            size_t used = ta_used_to_val(p->used);
            r = so_cudaMemcpy(dm_data+dm_pos, p->devptr, used, kind);
            if (r != cudaSuccess) {
                break;
            }
            printf("dm_save: copy %lx to %lx\n", used, dm_pos);
            dm_pos += used;
        }
    }
    if (r == cudaSuccess) {
        r = so_cudaDeviceSynchronize();
    }
    if (r != cudaSuccess) {
        // TODO
    }
    fflush(stdout);
    return r;
}

static void dm_free(void) {
    cudaError_t r = cudaSuccess;
    for (int i = dm_dpc-1; i>=0; i--) {
        r = so_cudaFree(dm_dps[i]);
        if (r != cudaSuccess) {
            so_cudaGetLastError();
        }
        printf("dm_free: %p (%d) ", dm_dps[i], i);
        for (;0;) {
            void * p = mmap(dm_dps[i], BLK_SIZE, PROT_NONE, 
                            MAP_ANONYMOUS|MAP_PRIVATE|MAP_FIXED|MAP_NORESERVE,
                            -1, 0);
            if (p == dm_dps[i]) {
                printf(" mmap ok");
                break;
            }
            else if (p != MAP_FAILED) {
                munmap(p, BLK_SIZE);
                printf("*");
            }
            else {
                printf(".");
            }
        }
        printf("\n");
    }
    /*
    trace_alloc_t * begin = ta_data;
    trace_alloc_t * end = (ta_data + ta_pos);
    for (trace_alloc_t * p = end-1; p >= begin; p--) {
        if (p->flags & TA_BLOCK) {
            r = so_cudaFree(p->devptr);
            if (r != cudaSuccess) {
                break;
            }
            p->flags |= TA_FREED;
        }
    }
    */
    print_mem_maps(minp, maxp);
    fflush(stdout);
}

static int dm_realloc(void) {
    cudaError_t r = cudaSuccess;
    int miss_match = 0;
    for (int i=dm_dpc; i<dm_dpc; i++) {
        munmap(dm_dps[i], BLK_SIZE);
    }
    printf("dm_realloc-start\n");
    print_mem_maps(minp, maxp);
    for (int i=0; i<dm_dpc; i++) {
        void * devptr = NULL;
        for (int i=0; i<10; i++) {
            r = so_cudaMalloc(&devptr, TA_BLOCK_SIZE);
            if (r == cudaErrorMemoryAllocation) {
                if (devptr == dm_dps[i]) {
                    break;
                }
                continue;
            }
            break;
        }
        if (devptr == NULL) {
            printf("dm_realloc: fail for %p (%d)\n", dm_dps[i], i);
            dm_dpc = i;
            return r;
        }
        else if (dm_dps[i] != devptr) {
            printf("dm_realloc: miss match: %p -> %p (%d)\n",
                            dm_dps[i], devptr, i);
            dm_dps[i] = devptr;
            miss_match++;
        }
        else {
            printf("dm_realloc: match: %p (%d)\n", devptr, i);
        }
    }
    if (miss_match) {
        print_mem_maps(minp, maxp);
    }
    if (r != cudaSuccess) {
        so_cudaGetLastError();
    }
    return cudaSuccess;
    size_t free, total;
    so_cudaMemGetInfo(&free, &total);
    const int N = (total + TA_BLOCK_SIZE-1) / TA_BLOCK_SIZE;
    void * dps[N];
    int n = 0;
    trace_alloc_t * begin = ta_data;
    trace_alloc_t * end = (ta_data + ta_pos);
    for (trace_alloc_t * p = begin; p < end; p++) {
        if ((p->flags & TA_BLOCK) && (p->flags & TA_FREED)) {
            void * devptr = NULL;
            r = so_cudaMalloc(&devptr, TA_BLOCK_SIZE);
            if (devptr == p->devptr) {
                p->flags &= ~TA_FREED;
                printf("dm_realloc: match: %p (%ld)\n", p->devptr, p-begin);
                continue;
            }
            if (r == cudaErrorMemoryAllocation) {
                continue;
            }
            if (devptr != NULL) {
                dps[n++] = devptr;
            }
        }
    }
    miss_match = n;
    for (int k = 0; (k < 10) && (miss_match > 0); k++) {
        miss_match = 0;
        so_cudaMemGetInfo(&free, &total);
        while (free >= TA_BLOCK_SIZE) {
            void * devptr = NULL;
            r = so_cudaMalloc(&devptr, TA_BLOCK_SIZE);
            if (r == cudaErrorMemoryAllocation) {
                break;
            }
            dps[n++] = devptr;
            so_cudaMemGetInfo(&free, &total);
        }
        for (trace_alloc_t * p = begin; p < end; p++) {
            if ((p->flags & TA_BLOCK) && (p->flags & TA_FREED)) {
                for (int i = 0; i < n; i++) {
                    if (p->devptr == dps[i]) {
                        dps[i] = NULL;
                        p->flags &= ~TA_FREED;
                        printf("dm_realloc: match:+ %p (%ld)\n", 
                                    p->devptr, p-begin);
                        break;
                    }
                }

                if (p->flags & TA_FREED) {
                    miss_match++;
                    printf("dm_realloc: miss match: %p (%ld)\n", 
                                    p->devptr, p-begin);
                }
            }
        }
        while (n--) {
            if (dps[n] != NULL) {
                so_cudaFree(dps[n]);
            }
        }
    }
    fflush(stdout);
    return miss_match;
}

static int dm_restore(void) {
    cudaError_t r = cudaSuccess;
    trace_alloc_t * begin = ta_data;
    trace_alloc_t * end = (ta_data + ta_pos);
    dm_pos = 0;
    int kind = cudaMemcpyHostToDevice;
    for (trace_alloc_t * p = begin; p < end; p++) {
        if ((p->flags & TA_BLOCK) && (p->flags & TA_FREED)) {
            size_t used = ta_used_to_val(p->used);
            p->flags &= ~TA_BLOCK;
            printf("dm_restore: no copy %lx from %lx\n", used, dm_pos);
            dm_pos += used;
        }
        if ((p->flags & TA_BLOCK) && p->used) {
            size_t used = ta_used_to_val(p->used);
            r = so_cudaMemcpy(p->devptr, dm_data+dm_pos, used, kind);
            if (r != cudaSuccess) {
                break;
            }
            printf("dm_restore: copy %lx from %lx\n", used, dm_pos);
            dm_pos += used;
        }
    }
    if (r == cudaSuccess) {
        r = so_cudaDeviceSynchronize();
    }
    if (r != cudaSuccess) {
        // TODO
    }
    fflush(stdout);
    return r;
}

static so_func_info_t * lookup_func_info(so_dl_info_t *dlip, void * func) {
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

static void do_recover_load_info(void) {
    char buf[256];
	FILE * f = trace_fopen(fd_trace_dir, "trace.info");
    fscanf(f, "%ld %ld %s", &dm_recover, &dm_size, buf);
    fscanf(f, "%ld %ld %s", &ta_recover, &ta_size, buf);
    fscanf(f, "%ld %ld %s", &recover_trace_idx, &ti_size, buf);
    ti_max = ti_size / sizeof(trace_invoke_t);
    fclose(f);
}

static void do_checkpoint_info(void) {
	FILE * f = trace_fcreate(fd_trace_dir, "trace.info");
	fprintf(f, "%ld\t%ld\t%s\n", dm_pos, dm_size, fn_devmem);
	fprintf(f, "%ld\t%ld\t%s\n", ta_pos, ta_size, fn_alloc);
	fprintf(f, "%ld\t%ld\t%s\n", next_trace_idx-1, ti_size, fn_invoke);
	fflush(f);
	fclose(f);
    rename(fn_trace_dir, fn_recover_dir);
}

static void do_checkpoint(void) {
    msync(dm_data, dm_size, MS_SYNC);
    munmap(dm_data, dm_size);
    close(fd_devmem);
    msync(ta_data, ta_size, MS_SYNC);
    munmap(ta_data, ta_size);
    close(fd_alloc);
    msync(ti_invokes, ti_size, MS_SYNC);
    munmap(ti_invokes, ti_size);
    close(fd_invoke);
	do_checkpoint_info();
}

static void wait_and_exit_for_checkpoint(void) {
	do_checkpoint();
    printf("wait_and_exit_for_checkpoint(%d);\n", so_tls.thread_idx);
    fflush(stdout);
    kill(0, SIGKILL);
    printf("wait_and_exit_for_checkpoint() called kill;\n");
    exit(1);
}

static void try_pause_for_checkpoint(const void * func) {
	if (request_for_checkpoint &&
			so_tls.thread_idx == forward_thread_idx &&
			//so_tls.async == 0 &&
            so_eventc == 0 &&
            last_backward_idx > 0 &&
            last_sync_to_backward > 0 &&
            diff_sync_to_backward >= last_sync_to_backward) {
		cudaError_t r = so_cudaDeviceSynchronize();
		if (r == cudaSuccess) {
            printf("checkpoint: sem_post(&sem_pause)\n");
			sem_post(&sem_pause);
			sem_wait(&sem_checkpoint);
            printf("checkpoint: sem_wait(&sem_checkpoint)\n");
            request_for_checkpoint = 0;
		}
	}
}

static void checkpoint_wait_signal(void) {
    for (;;) {
        if (sig_last_command == SIGRESUME ||
                sig_command == SIGRESUME) {
            sig_last_command = 0;
            if (sig_command == SIGRESUME) {
                sig_command = 0;
            }
            break;
        }
        usleep(100*1000);
    }
}

static void * ckeckpoint_deamon(void *data) {
    for (;;) {
        static uint64_t last_paused_idx = 0;
        sem_wait(&sem_pause);
        printf("paused at next_trace_idx=%ld for checkpoint[sig=%d] (%ld) (%ld)(%ld)\n", 
                        next_trace_idx, 
                        sig_last_command,
                        next_trace_idx-last_paused_idx,
                        last_forward_len, 
                        last_backward_len);
        last_paused_idx = next_trace_idx;
        switch (sig_last_command) {
            case SIGMIGRATE:
                if (ta_freed == 0) {
                    checkpoint_mode = 1;
                }
            case SIGPAUSE:
                sig_last_command = 0;
                check_re_cudaMalloc();
                dm_save();
                dm_free();
                if (checkpoint_mode) {
                    wait_and_exit_for_checkpoint();
                    kill(0, SIGKILL);
                    exit(1);
                    //sem_post(&sem_checkpoint);
                    //return data;
                }
                checkpoint_wait_signal();
                dm_realloc();
                dm_restore();
                check_re_cudaMalloc();
                break;
            default:
                sig_last_command = 0;
                break;
        }
        sem_post(&sem_checkpoint);
    }
    return data;
}

static void start_checkpoint_deamon(void) {
    pthread_t deamon;
    int r = pthread_create(&deamon, NULL, ckeckpoint_deamon, NULL);
    if (r != 0) {
        errmsg("pthread_create(ckeckpoint_deamon)");
        exit(1);
    }
    pthread_detach(deamon);
}

static void * run_func_thread(void *data) {
    void (*func)() = data;
    func();
    return data;
}

static void run_func(void *func) {
    pthread_t thread;
    int r = pthread_create(&thread, NULL, run_func_thread, func);
    if (r != 0) {
        errmsg("pthread_create(run_func_thread(%p))", func);
        return;
    }
    pthread_detach(thread);
}

static void set_request_for_checkpoint(void) {
    if (request_for_checkpoint) {
        return;
    }
    pthread_rwlock_wrlock(&trace_rwlock);
    request_for_checkpoint = 1;
    pthread_rwlock_unlock(&trace_rwlock);
}

static void signal_notifier_func(int sig) {
    if (sig_command == sig) {
        sig_repeat++;
    }
    else {
        sig_command = sig;
        sig_repeat = 0;
    }
    if (sig_repeat == 0) {
        printf("receiving signal %d\n", sig);
    }
    else {
        printf("receiving signal %d +%d\n", sig, sig_repeat);
    }
    sem_post(&sem_signal);
}

static void * signal_deamon(void *data) {
    for (;;) {
        sem_wait(&sem_signal);
        int sig = sig_command;
        if (sig_last_command != 0 && sig_last_command != SIGRESUME) {
            continue;
        }
        switch (sig) {
            case SIGRESUME:
                sig_last_command = sig;
                sig_command = 0;
                break;
            case SIGPAUSE:
                sig_last_command = sig;
                sig_command = 0;
                set_request_for_checkpoint();
                break;
            case SIGMIGRATE:
                sig_last_command = sig;
                sig_command = 0;
                set_request_for_checkpoint();
                break;
            default:
                sig_last_command = 0;
                sig_command = 0;
                break;
        }
    }
}

static void start_signal_deamon(void) {
    pthread_t deamon;
    int r = pthread_create(&deamon, NULL, signal_deamon, NULL);
    if (r != 0) {
        errmsg("pthread_create(signal_deamon)");
        exit(1);
    }
    pthread_detach(deamon);
}

// ----------------------------------------------------
// Swapped DL APIs
//

static cudaError_t re_cudaMalloc(void** devPtr, size_t size) {
    cudaError_t r = cudaSuccess;
    check_re_cudaMalloc();
    if (dm_blk < dm_dpc) {
        *devPtr = dm_dps[dm_blk];
        dm_blk++;
        return cudaSuccess;
    }
    r = so_cudaMalloc(devPtr, size);
    if  (r == cudaErrorMemoryAllocation) {
        return r;
    }
    if (r != cudaSuccess) {
        so_cudaGetLastError();
    }
    if (*devPtr == dm_dps[dm_dpc]) {
        printf("re_cudaMalloc: OK %p\n", *devPtr);
        dm_dpc++;
    }
    else {
        printf("re_cudaMalloc: Bad %p -> %p\n", dm_dps[dm_dpc], *devPtr);
        dm_dps[dm_dpc] = *devPtr;
        dm_dpc++;
    }
    return cudaSuccess;
}

static cudaError_t trace_cudaMalloc(void** devPtr, size_t size) {
    cudaError_t r = cudaSuccess;
    printf("trace_cudaMalloc: %ld\n", size);
    assert(so_tls.wrlock);
    size = ta_size_val(size);
    *devPtr = NULL;
    if (ta_pos > 0) { // do best fit search
        trace_alloc_t * begin = ta_data;
        trace_alloc_t * end = (ta_data + ta_pos);
        trace_alloc_t * found = NULL;
        size_t min_diff = ~((size_t)(0));
        for (trace_alloc_t * p = begin; p < end; p++) {
            if ((p->flags & TA_FREED) && 
                    !(p->flags & TA_REUSED) &&
                    (p->size == ta_val_to_size(size))) {
                trace_alloc_t * tap = ta_next_element();
                tap->devptr = p->devptr;
                tap->size = p->size;
                tap->flags = TA_ALLOC | TA_INUSE | TA_OOM;
                tap->total = 0;
                tap->used = 0;
                p->flags |= TA_REUSED;
                *devPtr = tap->devptr;
                printf("cudaMalloc: reuse freed for 0x%lx\n", size);
                return cudaSuccess;
            }
printf("best fit: %x %x\n", p->total, p->used);
            if (p->flags & TA_OOM) {
                continue;
            }
            size_t free = ta_total_to_val(p->total) - ta_used_to_val(p->used);
printf("best fit: %x %x - %lx %lx\n", p->total, p->used, free, size);
            if (free >= size) {
                size_t diff = free - size;
                if (diff < min_diff) {
                    min_diff = diff;
                    found = p;
                }
            }
        }
        if (found != NULL) {
            size_t total = ta_total_to_val(found->total);
            size_t used = ta_used_to_val(found->used);
            void * devptr = (found->devptr + used);
            used = ta_used_val(used + size);
            trace_alloc_t * tap = ta_next_element();
            tap->devptr = devptr;
            tap->size = ta_val_to_size(size);
            tap->flags = TA_ALLOC | TA_INUSE | TA_OOM;
            tap->total = 0;
            tap->used = 0;
            if (used >= total) {
                found->flags |= TA_OOM;
            }
            found->used = ta_val_to_used(used);
            *devPtr = devptr;
            printf("cudaMalloc: found 0x%lx:0x%lx (free=0x%lx) for 0x%lx\n",
                    total, used, (total-used), size);
            return cudaSuccess;
        }
    }
    size_t total_size = ta_total_val(size);
    if (total_size < TA_BLOCK_SIZE) {
        total_size = ta_total_val(TA_BLOCK_SIZE);
    }
  if (0) {
    if ((total_size - size) * 2 > size) {
        total_size *= 2;
    }
    else if ((total_size - size) * 4 > size) {
        total_size *= 4;
    }
  }
    if (total_size == TA_BLOCK_SIZE) {
        r = re_cudaMalloc(devPtr, total_size);
        if (r == cudaSuccess) {
            size_t fraction_size = ta_total_val(size);
            if (fraction_size < TA_FRACTION_SIZE) {
                fraction_size = TA_FRACTION_SIZE;
            }
            // a block record for pause and resume
            trace_alloc_t * tap = ta_next_element();
            tap->devptr = *devPtr;
            tap->size = 0;
            tap->flags = TA_BLOCK;
            tap->total = ta_val_to_total(total_size);
            tap->used = ta_val_to_used(fraction_size);
            if (fraction_size == total_size) {
                tap->flags |= TA_OOM;
            }
            // an alloc record for alloc and free
            tap = ta_next_element();
            tap->devptr = *devPtr;
            tap->size = ta_val_to_size(size);
            tap->flags = TA_ALLOC | TA_FRACTION | TA_INUSE;
            tap->total = ta_val_to_total(fraction_size);
            tap->used = ta_val_to_used(size);
            if (fraction_size == size) {
                tap->flags |= TA_OOM;
            }
            printf("cudaMalloc: alloc 0x%lx for 0x%lx with used (%x) 0x%lx\n",
                    total_size, size, tap->used, ta_used_val(size));
        }
        return r;
    }
    const int blknum = (size + TA_BLOCK_SIZE - 1) / TA_BLOCK_SIZE;
    trace_alloc_t * begin = ta_data;
    trace_alloc_t * last = (ta_data + ta_pos);
    trace_alloc_t * tap = NULL;
    trace_alloc_t * mark = NULL;
    while (last > begin &&
            (last-1)->size == 0 &&
            (last-1)->flags == TA_BLOCK &&
            (last-1)->total == ta_val_to_total(TA_BLOCK_SIZE) &&
            (last-1)->used == 0) {
        last--;
    }
    do {
        tap = ta_next_element();
        r = re_cudaMalloc(&tap->devptr, TA_BLOCK_SIZE);
        if (r == cudaErrorMemoryAllocation) {
            return r;
        }
        tap->size = 0;
        tap->flags = TA_BLOCK;
        tap->total = ta_val_to_total(TA_BLOCK_SIZE);
        tap->used = 0;
        printf("blks[%ld]: %p of %lx\n", tap-begin, tap->devptr, TA_BLOCK_SIZE);
        int nblk = 0;
        for (trace_alloc_t * p = last; p < tap; p++) {
            if (p->devptr + TA_BLOCK_SIZE == (p+1)->devptr) {
                if (nblk++ == blknum) {
                   mark = p - (blknum - 1);
                   for (int i = 0; i < blknum; i++) {
                       trace_alloc_t * p = mark + i;
                       printf("mark[%d]: %p of %lx\n",
                                    i, p->devptr, TA_BLOCK_SIZE);
                       p->flags |= TA_OOM;
                       p->used = ta_val_to_used(TA_BLOCK_SIZE);
                   }
                   break;
                }
            }
            else nblk = 0;
        }
        if (mark != NULL) {
            break;
        }
        for (trace_alloc_t * p = tap; p > last; p--) {
            if (p->devptr + TA_BLOCK_SIZE == (p-1)->devptr) {
                if (nblk-- == -blknum) {
                   mark = p + (blknum - 1);
                   for (int i = 0; i < blknum; i++) {
                       trace_alloc_t * p = mark - i;
                       printf("mark[%d]: %p of %lx\n",
                                    i, p->devptr, TA_BLOCK_SIZE);
                       p->flags |= TA_OOM;
                       p->used = ta_val_to_used(TA_BLOCK_SIZE);
                   }
                   break;
                }
            }
            else nblk = 0;
        }
    } while (mark == NULL);
    tap = ta_next_element();
    tap->devptr = mark->devptr;
    tap->size = size;
    tap->flags = TA_ALLOC | TA_INUSE | TA_OOM;
    tap->total = 0;
    tap->used = 0;
    if (size % TA_BLOCK_SIZE != 0) {
        size_t free_size = TA_BLOCK_SIZE - size % TA_BLOCK_SIZE;
        size_t fraction_size = ta_val_to_total(free_size);
        size_t used_size = ta_used_val(fraction_size - free_size);
        if (used_size < fraction_size) {
            tap = ta_next_element();
            void * devptr = mark->devptr;
            devptr += TA_BLOCK_SIZE * blknum;
            devptr -= fraction_size;
            tap->devptr = devptr;
            tap->size = 0;
            tap->flags = TA_FRACTION;
            tap->total = ta_val_to_total(fraction_size);
            tap->used = ta_val_to_used(used_size);
        }
    }
    *devPtr = mark->devptr;
    return cudaSuccess;
}

static cudaError_t trace_cudaFree(void* devPtr) {
    assert(so_tls.wrlock);
    trace_alloc_t * begin = ta_data;
    trace_alloc_t * end = (ta_data + ta_pos);
    next_idx(ta_freed);
    for (trace_alloc_t * p = end - 1; p >= begin; p--) {
        if ((p->flags & TA_INUSE) && (p->devptr == devPtr)) {
            trace_alloc_t * tap = ta_next_element();
            tap->devptr = devPtr;
            tap->size = 0;
            tap->flags = TA_FREE;
            tap->total = 0;
            tap->used = 0;
            p->flags &= ~TA_INUSE;
            p->flags |= TA_FREED;
            return cudaSuccess;
        }
    }
    return cudaErrorInvalidValue;
}

static cudaError_t trace_cudaEventCreate(cudaEvent_t* event) {
    cudaError_t r = so_cudaEventCreate(event);
    if (r == cudaSuccess) {
        so_events[so_eventc++] = *event;
		sprintf(so_tls.sbuf, "(%p) (%d)", *event, so_eventc);
    }
    return r;
}

static cudaError_t trace_cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int  flags) {
    cudaError_t r = so_cudaEventCreateWithFlags(event, flags);
    if (r == cudaSuccess) {
        so_events[so_eventc++] = *event;
		sprintf(so_tls.sbuf, "(%p) (%d)", *event, so_eventc);
    }
    return r;
}

static cudaError_t trace_cudaEventDestroy(cudaEvent_t event) {
    cudaError_t r = so_cudaEventDestroy(event);
    if (r == cudaSuccess) {
        for (int i = 0; i < so_eventc; i++) {
            if (event == so_events[i]) {
                for (; i < so_eventc; i++) {
                    so_events[i] = so_events[i+1];
                }
                so_eventc--;
                break;
            }
        }
		sprintf(so_tls.sbuf, "(%p) (%d)", event, so_eventc);
    }
    return r;
}

static void trace_post_sync(cudaError_t r) {
	if (trace_memcpy_kind == cudaMemcpyDeviceToHost && 
			so_tls.thread_idx == forward_thread_idx) {
		diff_sync_to_backward = so_tls.trace_idx - last_backward_idx;
	}
    so_tls.async = 0;
    if (!(so_tls.event || so_tls.async)) {
        sync_clear(&checkpoint_thread_bits, (1 << so_tls.thread_idx));
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
    cudaError_t r = cudaSuccess;
    r = so_cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);
    static int fcnt = 0;
	static const void * func_updateGradInput = NULL;
	#define __MAX_FUNCS 128
    static const void * funcs[__MAX_FUNCS] = {0};
	static const char * names[__MAX_FUNCS] = {0};
	#undef __MAX_FUNCS
	if (func == func_updateGradInput) {
		backward_thread_idx = so_tls.thread_idx;
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
				backward_thread_idx = so_tls.thread_idx;
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
            case 0x7f39d80: names[fcnt] = "cudnn:reduced_divisor"; break;
            case 0x7f59710: names[fcnt] = "cudnn:curandStateXORWOW"; break;
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
    cudaError_t r = cudaSuccess;
    r = so_cudaMemset(devPtr, value, count);
    sprintf(so_tls.sbuf, "ptr: %p val: %d cnt: %ld", devPtr, value, count);
    return r;
}

static cudaError_t trace_cudaMemsetAsync(void* devPtr, int  value, size_t count, cudaStream_t stream) {
    cudaError_t r = cudaSuccess;
    r = so_cudaMemsetAsync(devPtr, value, count, stream);
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
	if (request_for_checkpoint &&
            kind == cudaMemcpyHostToDevice &&
			so_tls.thread_idx == forward_thread_idx) {
        try_pause_for_checkpoint(cudaMemcpy);
    }
	cudaError_t r = cudaSuccess;
    r = so_cudaMemcpy(dst, src, count, kind);
	if (so_tls.thread_idx == forward_thread_idx) {
		trace_memcpy_kind = kind;
	}
	so_tls.async = 1;
    sprintf(so_tls.sbuf, "dst: %p src: %p cnt: %lu kind: %s",
           	dst, src, count, memcpyKinds[kind]);
	return r;
}

static cudaError_t trace_cudaMemcpyAsync(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream) {
	if (request_for_checkpoint &&
            kind == cudaMemcpyHostToDevice &&
			so_tls.thread_idx == forward_thread_idx) {
        try_pause_for_checkpoint(cudaMemcpyAsync);
    }
	cudaError_t r = cudaSuccess;
    r = so_cudaMemcpyAsync(dst, src, count, kind, stream);
	if (so_tls.thread_idx == forward_thread_idx) {
		trace_memcpy_kind = kind;
	}
	so_tls.async = 1;
    sprintf(so_tls.sbuf, "dst: %p src: %p cnt: %lu kind: %s (%p)",
           	dst, src, count, memcpyKinds[kind], stream);
	return r;
}

static cublasStatus_t trace_cublasSgemm_v2(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc) {
    cublasStatus_t r = CUBLAS_STATUS_SUCCESS;
    r = so_cublasSgemm_v2(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    sprintf(so_tls.sbuf, "alpah=%p A=%p B=%p beta=%p C=%p", alpha, A, B, beta, C);
    return r;
}

static cublasStatus_t trace_cublasSgemv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const float *alpha, const float *A, int lda, const float *x, int incx, const float *beta, float *y, int incy) {
    cublasStatus_t r = CUBLAS_STATUS_SUCCESS;
    r = so_cublasSgemv_v2(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
    sprintf(so_tls.sbuf, "alpah=%p A=%p x=%p beta=%p y=%p", alpha, A, x, beta, y);
    return r;
}

// ----------------------------------------------------
//
//

static void trace_print_invoke(uint32_t idx, trace_invoke_t * p, uint32_t cnt) {
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

static void update_func_info(so_dl_info_t *dlip) {
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
        lookup_func_info(dlip, func)->flags.checkpoint = 1;
    }
    void * sync_funcs[] = {
        cudaStreamSynchronize,
    };
    for (int i=0; i<sizeof(sync_funcs)/sizeof(void*); i++) {
        void * func = sync_funcs[i];
        lookup_func_info(dlip, func)->flags.sync = 1;
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
        lookup_func_info(dlip, func)->flags.event = 1;
    }
    void * rwlock_funcs[] = {
        cudaMalloc,
        cudaFree,
        cudaEventCreate,
        cudaEventCreateWithFlags,
        cudaEventDestroy,
    };
    for (int i=0; i<sizeof(rwlock_funcs)/sizeof(void*); i++) {
        void * func = rwlock_funcs[i];
        lookup_func_info(dlip, func)->flags.wrlock = 1;
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
        lookup_func_info(dlip, func)->flags.async = 1;
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
        lookup_func_info(dlip, func)->flags.notrace = 1;
    }
}

void cudaw_so_begin_func(so_dl_info_t *dlip, int idx) {
    const so_func_flags_t flags = dlip->funcs[idx].flags;
    if (so_tls.thread_idx == 0) {
        pthread_rwlock_wrlock(&trace_rwlock);
        so_tls.thread_idx = next_idx(&next_thread_idx);
		so_tls.wrlock = 1;
    }
    else if (flags.wrlock || request_for_checkpoint) {
        pthread_rwlock_wrlock(&trace_rwlock);
		so_tls.wrlock = 1;
    }
    else {
        pthread_rwlock_rdlock(&trace_rwlock);
    }
    if (flags.notrace && so_tls.invoke_idx) {
    	so_tls.invoke_idx++;
		return;
	}
    if (request_for_checkpoint && !so_tls.wrlock) {
	    pthread_rwlock_unlock(&trace_rwlock);
       	pthread_rwlock_wrlock(&trace_rwlock);
        so_tls.wrlock = 1;
    }
	so_tls.invoke_idx++;
    so_tls.trace_idx = idx_next(&next_trace_idx);
    trace_time(&so_tls.timestamp);
    if (request_for_checkpoint && 
            so_tls.thread_idx == forward_thread_idx &&
            flags.checkpoint) {
		try_pause_for_checkpoint(dlip->funcs[idx].func);
	}
}

void cudaw_so_end_func(so_dl_info_t *dlip, int idx) {
	if (so_tls.thread_idx == backward_thread_idx) {
        if (last_forward_idx > last_backward_idx) {
            last_forward_len = last_forward_idx - last_backward_idx;
        }
		last_backward_idx = so_tls.trace_idx;
		if (diff_sync_to_backward > 0) {
			last_sync_to_backward = diff_sync_to_backward;
			diff_sync_to_backward = 0;
			trace_memcpy_kind = 0;
		}
	}
    else if (so_tls.thread_idx == forward_thread_idx) {
        if (last_backward_idx > last_forward_idx) {
            last_backward_len = last_backward_idx - last_forward_idx;
        }
        last_forward_idx = so_tls.trace_idx;
    }
    so_tls.wrlock = 0;
    pthread_rwlock_unlock(&trace_rwlock);
    if (checkpoint_mode) {
        wait_and_exit_for_checkpoint();
        return;
    }
    next_idx(&dlip->funcs[idx].cnt);
    const so_func_flags_t flags = dlip->funcs[idx].flags;
    if (flags.notrace && (so_tls.invoke_idx != 1)) {
        return;
    }
    if (flags.async || !flags.known) {
        so_tls.async = 1;
        sync_set(&checkpoint_thread_bits, (1 << so_tls.thread_idx));
    }
    uint8_t dli_idx = dlip->dli_idx;
    if (so_tls.async) {
        dli_idx |= 0x80;
    }
    else if (so_tls.event) {
        dli_idx |= 0x40;
    }
    uint8_t thread_idx = so_tls.thread_idx;
    if (!checkpoint_thread_bits) {
        thread_idx |= 0x80; // global_checkpoint
    }
    else if (!(checkpoint_thread_bits & (1 << so_tls.thread_idx))) {
        thread_idx |= 0x40; // thread_checkpoint
    }
    if (recover_mode && (so_tls.trace_idx < recover_trace_idx)) {
        trace_invoke_t * p = &ti_invokes[so_tls.trace_idx];
        if (p->invoke_idx == (uint16_t)so_tls.invoke_idx &&
                p->thread_idx == thread_idx &&
                p->dli_idx == dli_idx &&
                p->func_idx == idx) {
            sprintf(so_tls.sbuf, "Identical");
            trace_print_invoke(so_tls.trace_idx, p, 0);
            so_tls.sbuf[0] = 0;
            so_tls.recover_idx = so_tls.trace_idx;
            return;
        }
        for (uint32_t k = so_tls.recover_idx; k < recover_trace_idx; k++) {
            trace_invoke_t * p = &ti_invokes[k];
            if (p->thread_idx == thread_idx)
                continue;
            if (p->invoke_idx == (uint16_t)so_tls.invoke_idx &&
                    p->dli_idx == dli_idx &&
                    p->func_idx == idx) {
                sprintf(so_tls.sbuf, "Offset=%d", k - so_tls.trace_idx);
                trace_print_invoke(so_tls.trace_idx, p, 0);
                so_tls.sbuf[0] = 0;
                so_tls.recover_idx = k;
                return;
            }
            sprintf(so_tls.sbuf, "Miss match! invoke: %u,%lu dli: %u,%u func: %u,%u",
                p->invoke_idx, so_tls.invoke_idx, p->dli_idx, dli_idx, p->func_idx, idx);
            trace_print_invoke(so_tls.trace_idx, p, 0);
            so_tls.sbuf[0] = 0;
            so_tls.recover_idx = k;
            return;
        }
        sprintf(so_tls.sbuf, "Not found! invoke: %lu dli: %u func: %u",
            so_tls.invoke_idx, dli_idx, idx);
        trace_print_invoke(so_tls.trace_idx, p, 0);
        so_tls.sbuf[0] = 0;
        return;
    }
    if (recover_mode) {
        rename(fn_recover_dir, fn_trace_dir);
        recover_mode = 0;
    }
    if (so_tls.trace_idx >= ti_max) {
        munmap(ti_invokes, ti_size);
        ti_size *= 2;
        ti_max *= 2;
        ti_invokes = trace_mmap(fd_invoke, ti_size);
    }
    trace_invoke_t * p = &ti_invokes[so_tls.trace_idx];
    p->invoke_idx = so_tls.invoke_idx;
    p->milliseconds = trace_msec(&so_tls.timestamp);
    p->thread_idx = thread_idx;
    p->dli_idx = dli_idx;
    p->func_idx = idx;
    trace_print_invoke(so_tls.trace_idx, p, 0);
    so_tls.sbuf[0] = 0;
    if (0) {
        #define __step 60000
        static uint64_t next_checkpoint_idx = __step;
        if (so_tls.trace_idx == next_checkpoint_idx) {
            next_checkpoint_idx += __step;
            kill(getpid(), SIGPAUSE);
        }
        else if (so_tls.trace_idx > next_checkpoint_idx) {
            next_checkpoint_idx = so_tls.trace_idx + __step;
        }
        #undef __step
    }
    if (0) {
        #define __step 80000
        static uint64_t next_checkpoint_idx = __step;
        if (so_tls.trace_idx == next_checkpoint_idx) {
            next_checkpoint_idx += __step;
            kill(getpid(), SIGMIGRATE);
        }
        else if (so_tls.trace_idx > next_checkpoint_idx) {
            next_checkpoint_idx = so_tls.trace_idx + __step;
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
    update_func_info(dlip);
}

void cudawrt_so_func_copy(void *funcs[]) {
    int i = 0;
    do {
        FCOPY(cudaGetLastError);
        FCOPY(cudaMemGetInfo)
    } while(0);
};

void cudawrt_so_func_swap(void *pfuncs[]) {
    int i = 0;
    do {
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
    } while(0);
};

void cudawblas_so_func_swap(void *pfuncs[]) {
    int i = 0;
    do {
        FSWAP(cublasSgemm_v2)
        FSWAP(cublasSgemv_v2)
    } while(0);
};

__attribute ((constructor)) void cudaw_trace_init(void) {
    printf("cudaw_trace_init\n");
    int r = pthread_rwlock_init(&trace_rwlock, NULL);
    if (r != 0) {
        errmsg("init(&trace_rwlock)\n");
        exit(1);
    }
	r = sem_init(&sem_signal, 0, 0);
    if (r != 0) {
        errmsg("sem_init(&sem_signal)\n");
        exit(1);
    }
	r = sem_init(&sem_pause, 0, 0);
    if (r != 0) {
        errmsg("sem_init(&sem_pause)\n");
        exit(1);
    }
	r = sem_init(&sem_checkpoint, 0, 0);
    if (r != 0) {
        errmsg("sem_init(&sem_checkpoint)\n");
        exit(1);
    }
    init_base_usec();
    fd_recover_dir = open(fn_recover_dir, O_DIRECTORY);
    if (fd_recover_dir != -1) {
        fd_trace_dir = fd_recover_dir;
        recover_mode = 1;
        do_recover_load_info();
    }
    else {
        mkdir(fn_trace_dir, 0777);
        fd_trace_dir = open(fn_trace_dir, O_DIRECTORY);
        if (fd_trace_dir == -1) {
            errmsg("%s\n", fn_trace_dir);
            exit(1);
        }
    }
    // open and mmap for invoke.trace
    fd_invoke = trace_open(fd_trace_dir, fn_invoke);
    ti_invokes = trace_mmap(fd_invoke, ti_size);
    // open and mmap for alloc.trace
    fd_alloc = trace_open(fd_trace_dir, fn_alloc);
    ta_data = trace_mmap(fd_alloc, ta_size);
    // open and mmap for memory.data
    fd_devmem = trace_open(fd_trace_dir, fn_devmem);
    dm_data = trace_mmap(fd_devmem, dm_size);
    // start all deamons
    start_checkpoint_deamon();
    start_signal_deamon();
    // prepare signals
    signal(SIGUSR1, signal_notifier_func);
    signal(SIGUSR2, signal_notifier_func);
    signal(SIGCONT, signal_notifier_func);
}

__attribute ((destructor)) void cudaw_trace_fini(void) {
    printf("cudaw_trace_fini\n");
    for (uint32_t i = 0; i < next_trace_idx; i++) {
        uint32_t cnt = 1;
        /*
        trace_invoke_t * p = &ti_invokes[i];
        for (uint32_t j = i+cnt; j < next_invoke_idx; j++) {
            trace_invoke_t * q = &ti_invokes[j];
            if (p->thread_idx != q->thread_idx ||
                p->dli_idx != q->dli_idx ||
                p->func_idx != q->func_idx)
                break;
            cnt++;
            i++;
            p = q;
        }
        trace_print_invoke(i, &ti_invokes[i], cnt);
        */
    }
    munmap(dm_data, dm_size);
    close(fd_devmem);
    munmap(ta_data, ta_size);
    close(fd_alloc);
    munmap(ti_invokes, ti_size);
    close(fd_invoke);
    close(fd_trace_dir);
	sem_destroy(&sem_checkpoint);
	sem_destroy(&sem_pause);
    // Don't destroy the sem_signal used by checkpoint deamon!
	// sem_destroy(&sem_signal);
    // Don't destroy the trace_rwlock used by checkpoint deamon!
    // pthread_rwlock_destroy(&trace_rwlock); 
}

#ifdef TEST_MAIN

int main(int argc, const char * argv[]) {
    size_t x = 127;
    for (int e = 0; e < 42; e++) {
        size_t u = ta_val_to_used(x);
        size_t v = ta_used_val(x);
        printf("(%lx) = %lx - %lx\n", u, x, v);
        u = ta_val_to_used(x-1);
        v = ta_used_val(x-1);
        printf("(%lx) = %lx - %lx\n", u, x-1, v);
        u = ta_val_to_used(x+1);
        v = ta_used_val(x+1);
        printf("(%lx) = %lx - %lx\n", u, x+1, v);
        x <<= 1;
    }
    return 0;
}

#endif
