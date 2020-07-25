#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <string.h>
#include <pthread.h>
#include <cuda_runtime.h>
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


#include "cudaw.h"
#include "vaddr.h"

#define errmsg(format...) do { \
    const char* msg = strerror(errno); \
    fprintf(stderr, "[FAIL] %s:%d (errno=%d): %s -- ", __FILE__, __LINE__, errno, msg); \
    fprintf(stderr, format); \
} while (0)

typedef struct so_invoke_t {
	int32_t milliseconds;
	int8_t  thread_idx;
	int8_t  dli_idx;
	int16_t func_idx;
} so_invoke_t;

typedef struct so_tls_t {
	int8_t         thread_idx;
    int32_t        invoke_idx;
    struct timeval timestamp;
} so_tls_t;

static time_t   base_sec;
static int32_t  next_thread_idx = 0;
static int32_t  next_invoke_idx = 0;
static int8_t   next_dli_idx = 0;

#define next_idx(pidx) __sync_add_and_fetch(pidx, 1)
#define idx_next(pidx) __sync_fetch_and_add(pidx, 1)


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
static int32_t      invoke_size = 10 * 1024 * 1024;


static __thread so_tls_t tls = {0};
static so_dl_info_t *dlips[256] = {0};

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

void cudaw_so_begin_func(so_dl_info_t *dlip, int idx) {
    if (tls.thread_idx == 0) {
         tls.thread_idx = next_idx(&next_thread_idx);
    }
    tls.invoke_idx = idx_next(&next_invoke_idx);
    so_time(&tls.timestamp);
    dlip->funcs[idx].cnt++;
    cudawMemLock();
    if (tls.invoke_idx >= invoke_size) {
        size_t size = invoke_size * sizeof(so_invoke_t);
        munmap(so_invokes, size);
        invoke_size *= 2;
        size *= 2;
        so_invokes = so_mmap(fd_invoke, size);
    }
}

void cudaw_so_end_func(so_dl_info_t *dlip, int idx) {
    cudawMemUnlock();
    so_invoke_t * p = &so_invokes[tls.invoke_idx];
    p->milliseconds = so_msec(&tls.timestamp);
    p->thread_idx = tls.thread_idx;
    p->dli_idx = dlip->dli_idx;
    p->func_idx = idx;
}

void cudaw_rigister_dli(so_dl_info_t *dlip) {
    dlip->dli_idx = next_idx(&next_dli_idx);
    dlips[dlip->dli_idx] = dlip;
}

__attribute ((constructor)) void cudaw_trace_init(void) {
    printf("cudaw_trace_init\n");
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
}

__attribute ((destructor)) void cudaw_trace_fini(void) {
    printf("cudaw_trace_fini\n");
    for (uint32_t i = 0; i < next_invoke_idx; i++) {
        so_invoke_t * p = &so_invokes[i];
        printf("%8u %8u %u %u:%s", i, p->milliseconds, 
                    p->thread_idx, p->dli_idx,
                    dlips[p->dli_idx]->funcs[p->func_idx].func_name);
        uint32_t cnt = 1;
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
        if (cnt > 1)
            printf(" +%u\n", cnt-1);
        else
            printf("\n");
    }
    size_t size = invoke_size * sizeof(so_invoke_t);
    munmap(so_invokes, size);
    close(fd_invoke);
    close(fd_trace_dir);
}
