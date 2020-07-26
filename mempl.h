#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <cuda.h>
#include <string.h>
#include <pthread.h>

#include <sys/types.h>
#include <unistd.h>
#include <time.h>
#include <unistd.h>
#include <semaphore.h>
#include <errno.h>
#include <assert.h>



static void * min_devptr = ~1;

struct cudaw_memrec {
    void * devptr;
    size_t size;

    // 0: valid
    int invalid;
    void * hostptr;
};

typedef struct cudaw_memrec cudaw_memrec;

struct cudaw_mempl {
    size_t size;
    size_t num;
    cudaw_memrec * mems;
};

typedef struct cudaw_mempl cudaw_mempl;

static void mempl_init(cudaw_mempl * mempl) {
    mempl->size = 1023;
    mempl->num = 0;
    mempl->mems = malloc(sizeof(cudaw_memrec) * mempl->size);
}

static void mempl_trace_malloc(cudaw_mempl * mempl,
                                void * devptr, size_t size) {
    if (mempl->num == mempl->size) {
        mempl->size = mempl->size * 2 + 1;
        mempl->mems = realloc(mempl->mems, 
                                sizeof(cudaw_memrec) * mempl->size);
        // TODO: save to file if too large
    }
    if (devptr < min_devptr) {
        min_devptr = devptr;
    }
    mempl->mems[mempl->num].devptr = devptr;
    mempl->mems[mempl->num].size = size;
    mempl->num++;
}

static void mempl_trace_free(cudaw_mempl * mempl, void * devptr) {
    mempl_trace_malloc(mempl, devptr, 0);
}





static void mempl_mark(cudaw_mempl * mempl) {
    for (int k = 0; k < mempl->num; k++) {
        if (mempl->mems[k].size != 0) {
            continue;
        }
        void * devptr = mempl->mems[k].devptr;
        for (int i = 0; i < k; i++) {
            if (mempl->mems[k].size == 0) {
                continue;
            }
            if (mempl->mems[i].invalid == 1) {
                continue;
            }
            else if (devptr == mempl->mems[i].devptr) {
                mempl->mems[i].invalid = 1;
                break;
            }
        }
    }
}