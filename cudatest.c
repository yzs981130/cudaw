#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <cuda.h>
#include <string.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>
#include <pthread.h>
#include <assert.h>
#include <nvml.h>
#include <signal.h>

#include "cudamaster.h"

#define MAX_GPUS  32
#define MAX_PROC  32

static int N = 0; // count of GPU devices
static pthread_t master = 0;
static pthread_t killer = 0;

struct worker {
    pthread_t worker;
    CUdevice device;
    CUcontext context;
    nvmlDevice_t nvdev;
    nvmlProcessInfo_t procs[MAX_PROC];
    nvmlProcessInfo_t proc;
    int pidx;
};

static struct worker ws[MAX_GPUS] = {0};


#define MIN_OVERHEAD    (128 * 0x100000llu)     // 128MB < 135MB

#define CR_SLEEP    1000        // 1ms
#define TIP_SLEEP   50000       // 50ms
#define BLK_SLEEP   20000       // 20ms
#define WATCH_SLEEP 50000       // 500ms
#define WATCH_COUNT 10          

struct dev_info {
    int i;
    int state;
    int no;
    int exclusive;
    int shared;
    int target;
    int request;
    int response;
    int timeout_target;
    int timeout_response;
    size_t init_free;
    size_t tip_free;
    size_t reserved;
    size_t overhead;
    size_t free;
    size_t total;
    int free_unchanged;
    int blk_size;
    int blk_num;
    int tip_size;
    int tip_num;
    int tip_response;
    int pg_size;
    int pg_num;
    CUdeviceptr * blks;
    CUdeviceptr * tips;
    CUdeviceptr * pgs;
};

typedef struct dev_info dev_info;

enum {
    INIT, 
    RESERVING, 
    RELEASING, 
    TIPTARGET,
    TIPRESPONSE,
    WATCHING,
    FEEDING,
    GRABBING,
};

enum {
    START=0xff,
    CLEAN=0x00,
    LOREQ=0x01,
    ACKLO=0x02,
    REFIN=0x03,
    HIREQ=0x04,
    ACKHI=0x05,
    SYNHI=0x06,
    RESRV=0x07,
};

static int compare_ptr(const void * pa, const void * pb) {
    if (*(CUdeviceptr*)pb < *(CUdeviceptr*)pa)
        return 1;
    else if (*(CUdeviceptr*)pb > *(CUdeviceptr*)pa)
        return -1;
    else
        return 0;
}

static int worker_mem_get_info(dev_info * dip) {
    CUresult cr;
    size_t last_free = dip->free;
    for (int i = 0; i < 3; i++) {
        cr = cuMemGetInfo(&dip->free, &dip->total);
        if (cr != CUDA_SUCCESS) {
            usleep(CR_SLEEP);
            continue;
        }
        if (last_free != dip->free) {
            dip->free_unchanged = 0;
        }
        return 1;
    }
    fprintf(stderr, "cuMemGetInfo return %d three times\n", cr);
    return 0;
}

static void worker_release_one_blk(dev_info * dip) {
    dip->blk_num--;
    cuMemFree(dip->blks[dip->blk_num]);
    dip->blks[dip->blk_num] = 0;
    dip->reserved -= BLK_SIZE;
}

static void worker_reserve_one_blk(dev_info * dip) {
    for (;;) {
        CUdeviceptr devptr = 0;
        cuMemAlloc(&devptr, BLK_SIZE);
        if (devptr == 0) {
            continue;
        }
        dip->blks[dip->blk_num++] = devptr;
        dip->reserved = dip->blk_num * BLK_SIZE;
        return;
    }
}

static int worker_dec_pgs(dev_info * dip, int n) {
    if (n > dip->pg_num) {
        n = dip->pg_num;
    }
    for (int i = 0; i < n; ++i) {
        dip->pg_num--;
        cuMemFree(dip->pgs[dip->pg_num]);
        dip->pgs[dip->pg_num] = 0;
        dip->free += TPG_SIZE;
    }
    return n;
}

static int worker_inc_pgs(dev_info * dip, int n) {
    for (int i = 0; i < n; i++) {
        if (dip->pg_num >= dip->pg_size) {
            return i;
        }
        CUdeviceptr devptr = 0;
        cuMemAlloc(&devptr, TPG_SIZE);
        if (devptr == 0) {
            return i;
        }
        dip->pgs[dip->pg_num++] = devptr;
        dip->free -= TPG_SIZE;
    }
    return n;
}

static int worker_dec_tips(dev_info * dip, int n) {
    if (n > dip->tip_num) {
        n = dip->tip_num;
    }
    for (int i = 0; i < n; ++i) {
        dip->tip_num--;
        CUresult cr = cuMemFree(dip->tips[dip->tip_num]);
        dip->tips[dip->tip_num] = 0;
        if (cr == CUDA_ERROR_INVALID_VALUE) {
            printf("cuMemFree: tips[%d] bad ptr\n", dip->tip_num);
            i--;
        }
        //worker_mem_get_info(dip);
        //printf("dec_tips: target: %d free: %d\n", dip->target, mb(dip->free));
    }
    return n;
}

static int worker_inc_tips(dev_info * dip, int n) {
    for (int i = 0; i < n; i++) {
        if (dip->tip_num >= dip->tip_size) {
            return i;
        }
        CUdeviceptr devptr = 0;
        CUresult cr = cuMemAlloc(&devptr, TIP_SIZE);
        if (devptr == 0) {
            return i;
        }
        if (cr == CUDA_ERROR_OUT_OF_MEMORY) {
            return i;
            printf("cuMemAlloc: tips[%d] out of memory (%d:%d)\n", 
                        dip->tip_num, i, n);
        }
        dip->tips[dip->tip_num++] = devptr;
        //worker_mem_get_info(dip);
        //printf("inc_tips: target: %d free: %d\n", dip->target, mb(dip->free));
    }
    return n;
}

static void worker_wait_for_target(dev_info * dip) {
    int total_msec = 0; 
    for (int i = 0; i < TIMEOUT / WAITMS; i++) {
        worker_mem_get_info(dip);
        int mb_free = mb(dip->free);
        if (dip->free > MAX_TIPS * TIP_SIZE) {
            dip->state = RESERVING;
            printf("worker_wait_for_target: failover to reserving: %d\n", mb_free);
            return;
        }
        if (mb_free == dip->target) {
            total_msec += WAITMS;
            if (total_msec > REQWAIT) {
                printf("worker_wait_for_target: %d free: %d\n", dip->target, mb_free);
                dip->state = WATCHING;
                return;
            }
        }
        else {
            total_msec = 0;
        }
        usleep(WAITMS * MS);
    }
    dip->timeout_target = 0;
    dip->state = WATCHING;
}

static void worker_tip_response(dev_info * dip) {
    int total_msec = 0; 
    int last_free = 0;
    int ret = 0;
    for (int i = 0; i < TIMEOUT / WAITMS; i++) {
        worker_mem_get_info(dip);
        int mb_free = mb(dip->free);
        if (dip->free > MIN_OVERHEAD) {
            dip->state = RESERVING;
            printf("worker_tip_response: failover to reserving: %d\n", mb_free);
            return;
        }
        if (mb_free > dip->response && dip->tip_response > 0) {
            printf("worker_tip_response: take back 1 (%d)\n", mb_free);
            dip->tip_response -= worker_inc_tips(dip, 1);
        }
        else if (dip->free == last_free && dip->tip_response <= 1) {
            total_msec += WAITMS;
            if (total_msec > REQWAIT) {
                dip->state = WATCHING;
                dip->tip_response = 0;
                printf("worker_tip_response: master's response: %d\n", mb_free);
                return;
            }
        }
        else {
            total_msec = 0;
        }
        last_free = dip->free;
        usleep(WAITMS * MS);
    }
    dip->timeout_response = 1;
    dip->state = WATCHING;
}

static void tip_response(dev_info * dip, int response) {
    dip->timeout_response = 0;
    dip->response = response;
    dip->state = TIPRESPONSE;
}

static void tip_target(dev_info * dip, int target) {
    dip->timeout_response = 0;
    dip->timeout_target = 0;
    dip->request = 0;
    dip->response = 0;
    dip->target = target;
    dip->state = TIPTARGET;
    switch (target) {
    case POLLREQ:
        dip->no++;
        dip->shared = ws[dip->i].proc.pid;
        break;
    case WORKING:
        break;
    case IDLE:
        ws[dip->i].proc.pid = 0;
        break;
    case RESET:
        ws[dip->i].proc.pid = getpid();
        break;
    }
}

#define blkmb(mb)   ((mb)/(BLK_SIZE/TIP_SIZE))

static void worker_tip_target(dev_info * dip) {
    int mb_free = mb(dip->free);
    int n = 0;
    while (dip->target - mb_free > (int)(BLK_SIZE/TIP_SIZE)) {
        worker_release_one_blk(dip);
        worker_mem_get_info(dip);
        mb_free = mb(dip->free);
        printf("tip_tick target: %d free: %d\n", dip->target, mb_free);
    }
    if (dip->target < mb_free) {
        n = worker_inc_tips(dip, mb_free - dip->target); 
    }
    else if (dip->target > mb_free) {
        n = worker_dec_tips(dip, dip->target - mb_free);
    }
    worker_wait_for_target(dip);
}

static void worker_watching(dev_info * dip) {
    int i = dip->i;
    size_t last_free = dip->free;
    size_t mb_free = 0;
    for (int i = 0; i < WATCH_COUNT; i++) {
        usleep(WATCH_SLEEP);
        dip->free_unchanged++;
        while (!worker_mem_get_info(dip)) {
            usleep(WATCH_SLEEP); // must get a valid free value
        }
        mb_free = mb(dip->free);
        if (!dip->free_unchanged) {
            i = 0;
            if (dip->target == POLLREQ) {
                if (ws[dip->i].proc.pid == 0) {
                    break;
                }
                if (mb_free > MIN_OVERHEAD) {
                    worker_reserve_one_blk(dip);
                }
                else if (mb_free < _B) {
                    worker_release_one_blk(dip);
                }
            }
            else if (dip->free > MAX_TIPS * TIP_SIZE) {
                for (int i = blk(MAX_TIPS * TIP_SIZE); i < blk(dip->free); i++) {
                    worker_reserve_one_blk(dip);
                }
            }
        }
    }
    if (dip->free_unchanged <= 2 * WATCH_COUNT) {
        printf("watching: no:%d blk:%d tip:%d [%d:%d] (%d,%d)\n",
                    dip->no, dip->blk_num, dip->tip_num, 
                    dip->target, mb_free, dip->request, dip->response);
    }
    switch (dip->target) {
    case POLLREQ:
        switch (mb_free) {
        case POLLREQ:
            if (dip->free_unchanged > 10 * WATCH_COUNT) {
                tip_target(dip, IDLE);
            }
            break;
        case REQUEST:
            if (dip->response == 0 && dip->request == 0) {
                dip->request = REQUEST;
                dip->shared++;
                dip->tip_response = (POLLREQ - OK);
                tip_response(dip, OK);
            }
            break;
        case OK:
            if (dip->request == REQUEST && dip->response == OK) {
                dip->request == OK;
                dip->state = FEEDING;
            }
            break;
        case RETURN:
            if (dip->request == RETURN && dip->response == OK) {
                dip->tip_response = (RETURN - RETGO);
                tip_response(dip, RETGO);
            }
            break;
        case RETGO:
            if (dip->request == RETURN && dip->response == RETGO) {
                dip->request = RETGO;
                dip->state = GRABBING;
            }
            break;
        case ENDING:
            dip->tip_response = (dip->response - RETGO);
            tip_response(dip, ENDED);
            break;
        case ENDED:
            if (dip->response == ENDED) {
                if (ws[i].proc.pid != 0 && !dip->exclusive) {
                    for (int k = 0; k < MAX_PROC; k++) {
                        int j = ws[i].pidx + k;
                        if (ws[i].procs[j].pid == 0) {
                            ws[i].procs[j] = ws[i].proc;
                            tip_target(dip, WORKING);
                            ws[i].pidx++;
                            break;
                        }
                    }
                }
                tip_target(dip, WORKING);
            }
            break;
        default:
            if (ws[i].proc.pid == 0) {
                dip->state = RESERVING;
            }
            else if (dip->request == RETURN && dip->response == RETGO) {
                dip->request = RETGO;
                dip->state = GRABBING;
            }
            break;
        }
        break;
    case RESET:
        if (dip->free_unchanged < WATCH_COUNT) {
            break;
        }
        if (mb_free == RESET) {
            tip_target(dip, IDLE);
            break;
        }
        tip_target(dip, RESET);
        break;
    case IDLE: // target == IDLE
        if (mb_free <= PROCESS && ws[i].proc.pid != 0) {
            tip_target(dip, POLLREQ);
            break;
        }
        if (mb_free == NOTIFY && ws[i].proc.pid == 0) {
            tip_target(dip, POLLREQ);
            break;
        }
        // fall through
    case WORKING:
        if (mb_free > IDLE) {
            worker_inc_tips(dip, mb_free - IDLE);
            break;
        }
        if (mb_free < IDLE) {
            if (dip->tip_num > IDLE) {
                worker_dec_tips(dip, IDLE - mb_free);
            }
            else for (int i = 0; i <= blk(IDLE*MB - dip->free); i++) {
                worker_release_one_blk(dip);
            }
        }
        if ((dip->reserved + dip->free) * 10 > dip->init_free * 9) {
            if (dip->target != IDLE) {
                tip_target(dip, IDLE);
            }
        }
        break;
    }
}

static void worker_release_all_memory(dev_info * dip) {
    for (int i = 0; i < dip->blk_num; ++i) {
        cuMemFree(dip->blks[i]);
        dip->blks[i] = 0;
    }
    dip->blk_num = 0;
    dip->reserved = 0;
    dip->state = WATCHING;
    worker_mem_get_info(dip);
    printf("release all: (%d,%d) (%d) %d\n",
        dip->init_free>>20, dip->total>>20, dip->free>>20, dip->reserved>>20);
}

static void worker_reserve_all_memory(dev_info * dip) {
    worker_dec_pgs(dip, dip->pg_num);
    worker_dec_tips(dip, dip->tip_num);
    while (!worker_mem_get_info(dip)) {
        usleep(BLK_SLEEP);
    }
    size_t last_free = dip->free;
    size_t last_blk_num = dip->blk_num;
    for (;;) {
        size_t tip_free = 3 * MIN_TIPS * TIP_SIZE;
        int n = (dip->free - tip_free + BLK_SIZE - 1) / BLK_SIZE;
        for (int i = 0; i < n; i++) {
            CUdeviceptr devptr = 0;
            cuMemAlloc(&devptr, BLK_SIZE);
            if (devptr == 0) {
                break;
            }
            dip->blks[dip->blk_num++] = devptr;
            dip->reserved = dip->blk_num * BLK_SIZE;
        }
        do {
            usleep(BLK_SLEEP);
        }
        while (!worker_mem_get_info(dip));

        printf("reserve all: (%d,%d) (%d) %d\n",
                dip->init_free>>20, dip->total>>20, dip->free>>20, dip->reserved>>20);
        if (last_free == dip->free && last_blk_num == dip->blk_num) {
            if (dip->free + dip->reserved > dip->init_free) {
                dip->init_free = dip->free + dip->reserved;
                dip->overhead = dip->total - dip->init_free;
            }
            while (worker_inc_tips(dip, mb(dip->free)) != 0) {
                worker_mem_get_info(dip);
            }
            //while (worker_inc_pgs(dip, pg(dip->free)) != 0) {
            //    worker_mem_get_info(dip);
            //}
            tip_target(dip, RESET);
            return;
        }
        last_free = dip->free;
        last_blk_num = dip->blk_num;
    }
}

static void worker_feeding_memory(dev_info * dip) {
    int oom_num = 9;
    if (dip->blk_num < oom_num) {
        int cnt = 0;
        while (cnt < (OK-RETURN)) {
            cnt += worker_inc_tips(dip, 1);
        }
    }
    for (int i = 0; i < _B; i++) {
        while (!worker_inc_tips(dip, 1));
    }
    for (;;) {
        if (!worker_mem_get_info(dip)) {
            usleep(MS);
            continue;
        }
        if (mb(dip->free) == RETURN) {
            dip->tip_response = (OK - RETURN); 
            dip->request = RETURN;
            break;
        }
        if (blk(dip->free) != 0) {
            continue;
        }
        if (dip->blk_num >= oom_num) {
            worker_release_one_blk(dip); 
            while (blk(dip->free) != 1) {
                while (!worker_mem_get_info(dip)) {
                    usleep(MS);
                    continue;
                }
            }
            if (dip->blk_num <= oom_num) {
                int cnt = 0;
                while (cnt < (OK-RETURN)) {
                    cnt += worker_inc_tips(dip, 1);
                }
                dip->request = RETURN;
            }
            worker_release_one_blk(dip);
        }
        if (dip->blk_num < oom_num) {
            dip->state = WATCHING;
            break;
        }
    }
    for (int i = 0; i < _B; i++) {
        while (!worker_dec_tips(dip, 1));
    }
}

static void worker_grabbing_memory(dev_info * dip) {
    for (int i = 0; i < _B; i++) {
        while (!worker_inc_tips(dip, 1));
    }
    for (;;) {
        if (!worker_mem_get_info(dip)) {
            usleep(MS);
            continue;
        }
        if (mb(dip->free) == ENDING) {
            tip_response(dip, ENDED);
            if (dip->response == ENDED) {
                int i = dip->i;
                if (ws[i].proc.pid != 0 && !dip->exclusive) {
                    for (int k = 0; k < MAX_PROC; k++) {
                        int j = ws[i].pidx + k;
                        if (ws[i].procs[j].pid == 0) {
                            ws[i].procs[j] = ws[i].proc;
                            ws[i].pidx++;
                            break;
                        }
                    }
                }
            }
            tip_target(dip, WORKING);
            dip->state = WATCHING;
            break;
        }
        if (blk(dip->free) != 2) {
            continue;
        }
        worker_reserve_one_blk(dip); 
        worker_reserve_one_blk(dip);
    }
    for (int i = 0; i < _B; i++) {
        while (!worker_dec_tips(dip, 1));
    }
}

static void worker_init(dev_info * dip) {
    dip->no = 0;
    dip->exclusive = 0;
    dip->shared = 0;
    dip->target = 0;
    dip->request = 0;
    dip->response = 0;
    if  (!worker_mem_get_info(dip))
        return;
    dip->init_free = dip->free;
    dip->overhead = dip->total - dip->free;
    dip->reserved = 0;
    dip->tip_free = 0;
    // prepare blks
    int n = (int)(dip->total / BLK_SIZE) + 1;
    dip->blks = malloc(sizeof(CUdeviceptr) * n);
    if (dip->blks == NULL) {
        return;
    }
    dip->blk_size = n;
    dip->blk_num = 0;
    // prepare tips
    n = MAX_TIPS * 2;
    dip->tips = malloc(sizeof(CUdeviceptr) * n);
    if (dip->tips == NULL) {
        free(dip->blks);
        dip->blks = NULL;
        return;
    }
    dip->tip_size = n;
    dip->tip_num = 0;
    // prepare pgs
    n = ((TIP_SIZE / TPG_SIZE) * MAX_TIPS);
    dip->pgs = malloc(sizeof(CUdeviceptr) * n);
    if (dip->pgs == NULL) {
        free(dip->tips);
        dip->tips = NULL;
        free(dip->blks);
        dip->blks = NULL;
        return;
    }
    dip->pg_size = n;
    dip->pg_num = 0;
    // prepare pgs
    // go to reserving memory
    dip->state = RESERVING;
    worker_reserve_all_memory(dip);
}

static void * clear_worker_and_take_a_break(int i) {
    memset(ws+i, 0, sizeof(ws[i]));
    sleep(1);
    return ws+i;
}

struct cudaw_memrec {
    CUdeviceptr devptr;
    size_t size;
};

typedef struct cudaw_memrec cudaw_memrec;

struct cudaw_mempl {
    size_t size;
    size_t num;
    cudaw_memrec * mems;
};

typedef struct cudaw_mempl cudaw_mempl;

static CUdeviceptr min_devptr = ~1;

static void mempl_init(cudaw_mempl * mempl) {
    mempl->size = 1023;
    mempl->num = 0;
    mempl->mems = malloc(sizeof(cudaw_memrec) * mempl->size);
}

static void mempl_trace_malloc(cudaw_mempl * mempl,
                                CUdeviceptr devptr, size_t size) {
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

static void mempl_trace_free(cudaw_mempl * mempl, CUdeviceptr devptr) {
    mempl_trace_malloc(mempl, devptr, 0);
}

static void mempl_free_all(cudaw_mempl * mempl) {
    for (int k = 0; k < mempl->num; k++) {
        if (mempl->mems[k].size != 0) {
            continue;
        }
        CUdeviceptr devptr = mempl->mems[k].devptr;
        for (int i = 0; i < k; i++) {
            if (mempl->mems[k].size == 0) {
                continue;
            }
            if (mempl->mems[i].devptr & 1) {
                continue;
            }
            else if (devptr == mempl->mems[i].devptr) {
                mempl->mems[i].devptr |= 1;
                break;
            }
        }
    }
    for (int i = 0; i < mempl->num; i++) {
        if (mempl->mems[i].devptr & 1) {
            mempl->mems[i].devptr &= ~(typeof(mempl->mems[i].devptr))(1);
        }
        else {
            cuMemFree(mempl->mems[i].devptr);
        }
    }   
}

static void mempl_replay(cudaw_mempl * mempl) {
    for (int i = 0; i < mempl->num; i++) {
        if (mempl->mems[i].size > 0) {
            CUdeviceptr devptr = 0;
            cuMemAlloc(&devptr, mempl->mems[i].size);
            if (devptr != mempl->mems[i].devptr) {
                printf("%4d %p %p\n", i, mempl->mems[i].devptr, devptr);
            }
        }
        else {
            cuMemFree(mempl->mems[i].devptr);
        }
    }
}

static void mempl_rand_play(cudaw_mempl * mempl, int count) {
    srand(getpid());
    while (count--) {
        size_t size = rand() % (64 * 1024);
        if (size < 1024) {
            continue;
        }
        if (0 && (size & 0x800)) {
            CUdeviceptr devptr = 0;
            cuMemAlloc(&devptr, size);
            cuMemFree(devptr);
        }
        else if (size & 1 & mempl->num > 0) {
            int k = size % mempl->num;
            cuMemFree(mempl->mems[k].devptr);
            mempl_trace_free(mempl, mempl->mems[k].devptr);
        }
        else {
            CUdeviceptr devptr = 0;
            cuMemAlloc(&devptr, size);
            if (devptr != 0) {
                mempl_trace_malloc(mempl, devptr, size);
            }
        }
    }
}



static void * worker_thread(void * data) {
    int i = (size_t)data;
    CUresult cr;
    cr = cuDeviceGet(&ws[i].device, i);
    if (cr != CUDA_SUCCESS) {
        fprintf(stderr, "FAIL: cuDeviceGet(%d) return %d\n", i, cr);
        return clear_worker_and_take_a_break(i);
    }
    cr = cuCtxCreate(&ws[i].context, CU_CTX_SCHED_BLOCKING_SYNC, ws[i].device);
    if (cr != CUDA_SUCCESS) {
        fprintf(stderr, "FAIL: cuCtxCreate(%d) return %d\n", i, cr);
        return clear_worker_and_take_a_break(i);
    }
    cr = cuCtxSetCurrent(ws[i].context);
    if (cr != CUDA_SUCCESS) {
        fprintf(stderr, "FAIL: cuCtxSetCurrent(%d) return %d\n", i, cr);
        cuCtxDestroy(ws[i].context);
        return clear_worker_and_take_a_break(i);
    }
    cudaw_mempl mempl;
    mempl_init(&mempl);
    mempl_rand_play(&mempl, 10240);
    for (int i = 0; i < 100; i++) {
        usleep(rand() % 10000);
        mempl_free_all(&mempl);
        usleep(rand() % 10000);
        mempl_replay(&mempl);
        printf("round %d of %d\n", i, getpid());
    }
    mempl_free_all(&mempl);
    return NULL;

    // test
    #define MM 64
    #define NN 1024
    size_t free, total;
    cuMemGetInfo(&free, &total);
    for (int x = 1; x <= 1024; x<<=1) {
    CUdeviceptr dps[MM][NN] = {0};
    int min_i = MM;
    int max_i = 0;
    for (int k = 0; k < mb(4 * GB); k+=x) {
        CUdeviceptr devptr = 0;
        CUresult cr = cuMemAlloc(&devptr, x * MB);
        if (cr == CUDA_ERROR_OUT_OF_MEMORY) {
                size_t free, total;
                cuMemGetInfo(&free, &total);
                printf("cuMemAlloc: out of memory %lu %lu\n", 
                            mb(free), mb(total));
                break;
        }
        else if (cr != CUDA_SUCCESS) {
            printf("cuMemAlloc return %d\n", cr);
        }
        else if (devptr != 0) {
            int i = gb(devptr) % MM;
            int j = mb(devptr) % NN;
            if (dps[i][j] != 0) {
                printf("dps[%d][%d] = %p, dp = %p\n", 
                            i, j, (void *)dps[i][j], (void *)devptr);
                cuMemFree(devptr);
            }
            else {
                dps[i][j] = devptr;
            }
            if (i < min_i) min_i = i;
            if (i > max_i) max_i = i;
        }
        usleep(1000);
    }
    if (x == 0) {
        int Y = 32;
        CUdeviceptr ys[Y];
        for (int y = 1; y < Y; y++) {
            CUdeviceptr devptr = 0;
            cuMemAlloc(&devptr, y * MB);
            ys[y] = devptr;
        }
        for (int y = 1; y < Y; y++) {
            cuMemFree(ys[y]);
        }
    }
    for (int i = 0; i < MM; i++) {
        int nn = NN;
        for (int j = 0; j < NN; j++) {
            if (dps[i][j]) {
                cuMemFree(dps[i][j]);
            }
            else {
                nn--;
            }
        }
        if (nn > 0) {
            printf("%3d %d\n", i, nn);
        }
    }
    if (1) {
        CUdeviceptr devptr = 0;
        CUresult cr = cuMemAlloc(&devptr, 2 * GB);
        if (cr == CUDA_ERROR_OUT_OF_MEMORY) {
                size_t free, total;
                cuMemGetInfo(&free, &total);
                printf("cuMemAlloc: out of memory %lu %lu\n", 
                            mb(free), mb(total));
        }
        else if (cr != CUDA_SUCCESS) {
            printf("cuMemAlloc return %d\n", cr);
        }
        else if (devptr != 0) {
            int i = gb(devptr) % MM;
            int j = mb(devptr) % NN;
            if (dps[i][j] != 0) {
                printf("dps[%d][%d] = %p, dp = %p\n", 
                            (void *)dps[i][j], (void *)devptr);
                cuMemFree(devptr);
            }
            else {
                dps[i][j] = devptr;

                printf("dps[%d][%d] dp = %p (min_i=%d)\n", 
                            i, j, (void *)devptr, min_i);
                cuMemFree(devptr);
                cuMemFree(dps[min_i][NN-1]);
            }
        }
    }
}
    return NULL;



    dev_info di = {i, INIT, };
    worker_init(&di);
    for (;;) {
        if (!worker_mem_get_info(&di)) {
            continue;
        }
        switch (di.state) {
            case INIT:
                worker_init(&di);
                break;
            case WATCHING:
                worker_watching(&di);
                break;
            case RESERVING:
                worker_reserve_all_memory(&di);
                break;
            case RELEASING:
                worker_release_all_memory(&di);
                break;
            case TIPTARGET:
                worker_tip_target(&di);
                break;
            case TIPRESPONSE:
                worker_tip_response(&di);
                break;
            case FEEDING:
                worker_feeding_memory(&di);
                break;
            case GRABBING:
                worker_grabbing_memory(&di);
                break;
            default:
                break;
        }
    }
}

void * killer_thread(void * data) {
    nvmlReturn_t nr = nvmlInit();
    if (NVML_SUCCESS != nr) {
        fprintf(stderr, "FAIL: nvmlInit() - %d\n", nr);
        exit(0);
    }
    int deviceCount;
    nr = nvmlDeviceGetCount(&deviceCount);
    if (NVML_SUCCESS != nr) {
        fprintf(stderr, "FAIL: nvmlInit() - %d\n", nr);
        exit(0);
    }
    if (deviceCount != N) {
        fprintf(stderr, "FAIL: deviceCount != N %d:%d - %d\n", deviceCount, N, nr);
        exit(0);
    }
    for  (int i = 0; i < N; i++) {
        nr = nvmlDeviceGetHandleByIndex(i, &ws[i].nvdev);
        if (NVML_SUCCESS != nr) {
            fprintf(stderr, "FAIL: nvmlDeviceGetHandleByIndex(%d) - %d\n", i, nr);
            exit(0);
        }
    }
    for (;;usleep(100)) {
      for (int i = 0; i < N; i++) {
        unsigned int infoCount = 0;
        nr = nvmlDeviceGetComputeRunningProcesses(ws[i].nvdev, &infoCount, NULL);
        if (NVML_SUCCESS != nr && nr != NVML_ERROR_INSUFFICIENT_SIZE) {
            continue;
        }
        nvmlProcessInfo_t infos[infoCount];
        nr = nvmlDeviceGetComputeRunningProcesses(ws[i].nvdev, &infoCount, infos);
        if (NVML_SUCCESS != nr && nr != NVML_ERROR_INSUFFICIENT_SIZE) {
            continue;
        }
        for (int k = 0; k < infoCount; k++) {
            unsigned int pid = infos[k].pid;
            int j = 0;
            for (; j < MAX_PROC; j++) {
                if (pid == ws[i].procs[j].pid) {
                    ws[i].procs[j].usedGpuMemory = infos[k].usedGpuMemory;
                    break;
                }
            }
            if (j == MAX_PROC) {
                if (pid == getpid()) {
                    for (int x = 0; x < MAX_PROC; x++) {
                        if (ws[i].procs[x].pid == 0 && ws[i].pidx != x) {
                            ws[i].procs[x] = infos[k];
                           printf("save me %d ws[%d][%d]\n", pid, i, j);
                           break;
                        }
                    }
                }
                else if (ws[i].proc.pid == 0) {
                    ws[i].proc = infos[k];
                }
                else if (pid == ws[i].proc.pid) {
                    ws[i].proc.usedGpuMemory = infos[k].usedGpuMemory;
                }
                else {
                    char name[1024] = {0};
                    nvmlSystemGetProcessName(pid, name, sizeof(name));
                    printf("**** kill %d %s\n", pid, name);
                    kill(pid, 9);
                }
            }
        }
        for (int j = 0; j < MAX_PROC; j++) {
            unsigned int pid = ws[i].procs[j].pid;
            if (pid == 0 || pid == getpid()) {
                continue;
            }
            int k = 0;
            for (; k < infoCount; k++) {
                if (pid == infos[k].pid) {
                    break;
                }
            }
            if (k == infoCount) {
                ws[i].procs[j].pid = 0;
                printf("**** process %d exits\n", pid);
            }
        }
        if (ws[i].proc.pid != 0 && ws[i].proc.pid != getpid()) {
            unsigned int pid = ws[i].proc.pid;
            int k = 0;
            for (; k < infoCount; k++) {
                if (pid == infos[k].pid) {
                    break;
                }
            }
            if (k == infoCount) {
                ws[i].proc.pid = 0;
                printf("**** process %d exits\n", pid);
            }
        }
      }
    }
    nvmlShutdown();
}

static void * master_thread(void * data) {
    CUresult cr;
    cr = cuDeviceGetCount(&N);
    if (cr != CUDA_SUCCESS) {
		fprintf(stderr, "FAIL: unable to get CUDA device count! (%d)\n", cr);
        exit(0);
    }
    if (N == 0) {
		fprintf(stderr, "FAIL: no CUDA device found! (%d)\n", cr);
        exit(0);
    }
    worker_thread(NULL);
}

//__attribute ((constructor)) void cudaw_master_init(void) {
void main(int argc, const char * argv[]) {
    printf("cudaw_master_init\n");
    fork();
    fork();
    fork();
    fork();
    CUresult cr = cuInit(0);
    if (cr != CUDA_SUCCESS) {
		fprintf(stderr, "FAIL: unable to init CUDA dirver... (%d)\n", cr);
        exit(0);
    }
    master_thread(NULL);
}
