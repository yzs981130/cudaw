#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <cuda.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>
#include <pthread.h>
#include <assert.h>

#include "cudamaster.h"

#define MAX_GPUS  32

static int N = 0; // count of GPU devices
static pthread_t master = 0;
static pthread_t workers[MAX_GPUS] = {0};

static CUdevice devices[MAX_GPUS] = {0};
static CUcontext contexts[MAX_GPUS] = {0};


#define MIN_OVERHEAD    (128 * 0x100000llu)     // 128MB < 135MB

#define CR_SLEEP    1000        // 1ms
#define TIP_SLEEP   50000       // 50ms
#define BLK_SLEEP   20000       // 20ms
#define WATCH_SLEEP 50000      // 500ms
#define WATCH_COUNT 10          

struct dev_info {
    int i;
    int state;
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
}

static void worker_tip_target(dev_info * dip) {
    int mb_free = mb(dip->free);
    int n = 0;
    printf("tip_tick target: %d free: %d\n", dip->target, mb_free);
    if (dip->target < mb_free) {
        n = worker_inc_tips(dip, mb_free - dip->target); 
    }
    else if (dip->target > mb_free) {
        n = worker_dec_tips(dip, dip->target - mb_free);
    }
    worker_wait_for_target(dip);
}

static void worker_watching(dev_info * dip) {
    size_t last_free = dip->free;
    size_t mb_free = 0;
    for (int i = 0; i < WATCH_COUNT; i++) {
        usleep(WATCH_SLEEP);
        dip->free_unchanged++;
        while (!worker_mem_get_info(dip)) {
            usleep(WATCH_SLEEP); // must get a valid free value
        }
        if (!dip->free_unchanged) {
            // any change should be handle immidiately
            if (dip->free > MAX_TIPS * TIP_SIZE) {
                dip->state = RESERVING;
                return;
            }
            if (dip->target == POLLREQ && dip->free > MIN_OVERHEAD) {
                dip->state = RESERVING;
                return;
            }
            i = 0;
        }
    }
    mb_free = mb(dip->free);
    if (dip->free_unchanged <= 2 * WATCH_COUNT) {
        printf("watching: target %d mb_free %d request %d response %d\n",
                    dip->target, mb_free, dip->request, dip->response);
    }
    switch (dip->target) {
    case POLLREQ:
        switch (mb_free) {
        case POLLREQ:
            if (dip->free_unchanged > 10 * WATCH_COUNT) {
                dip->state = RESERVING;
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
                dip->tip_response = (OK - RETGO);
                tip_response(dip, RETGO);
            }
            break;
        case RETGO:
            if (dip->request == RETURN && dip->response == RETGO) {
                if (dip->timeout_response) {
                    tip_response(dip, RETGO);
                }
                else {
                    dip->request = RETGO;
                    dip->state = GRABBING;
                }
            }
            break;
        case ENDING:
            dip->tip_response = (dip->response - RETGO);
            tip_response(dip, ENDED);
            break;
        case ENDED:
            if (dip->response == ENDED) {
                if (dip->exclusive) {
                    dip->exclusive = 0;
                }
                else {
                    dip->shared--;
                }
                tip_target(dip, IDLE);
            }
            break;
        default:
            if (dip->free > MIN_OVERHEAD) {
                dip->state = RESERVING;
            }
            break;
        }
        break;
    case FORCERET:
        if (dip->free_unchanged < WATCH_COUNT) {
            break;
        }
        if (mb_free == FORCERET) {
            tip_target(dip, IDLE);
            break;
        }
        break;
    case IDLE:
        if (dip->free_unchanged < WATCH_COUNT) {
            break;
        }
        if (mb_free <= PROCESS) {
            tip_target(dip, POLLREQ);
            break;
        }
        if (mb_free == NOTIFY) {
            tip_target(dip, POLLREQ);
            break;
        }
        if (mb_free > IDLE) {
            worker_inc_tips(dip, mb_free - IDLE);
            break;
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
    printf("dip->free(%d,%d) (%d) %d\n",
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
        size_t tip_free = MAX_TIPS * TIP_SIZE;
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

        printf("dip->free(%d,%d) (%d) %d\n",
                dip->init_free>>20, dip->total>>20, dip->free>>20, dip->reserved>>20);
        if (last_free == dip->free && last_blk_num == dip->blk_num) {
            if (dip->free + dip->reserved > dip->init_free) {
                dip->init_free = dip->free + dip->reserved;
                dip->overhead = dip->total - dip->init_free;
            }
            while (worker_inc_tips(dip, mb(dip->free)) != 0) {
                worker_mem_get_info(dip);
            }
            while (worker_inc_pgs(dip, pg(dip->free)) != 0) {
                worker_mem_get_info(dip);
            }
            dip->target = FORCERET;
            dip->state = TIPTARGET;
            return;
        }
        last_free = dip->free;
        last_blk_num = dip->blk_num;
    }
}

static void worker_feeding_memory(dev_info * dip) {
    int oom_num = 5;
    if (dip->blk_num < oom_num) {
        int cnt = 0;
        while (cnt < 2) {
            cnt += worker_inc_tips(dip, 1);
        }
    }
    for (;;) {
        if (!worker_mem_get_info(dip)) {
            usleep(MS);
            continue;
        }
        if (mb(dip->free) == RETURN) {
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
                while (cnt < 2) {
                    cnt += worker_inc_tips(dip, 1);
                }
                dip->request = RETURN;
            }
            worker_release_one_blk(dip);
        }
        if (dip->blk_num < oom_num) {
            dip->state = WATCHING;
            return;
        }
    }
}

static void worker_grabbing_memory(dev_info * dip) {
    for (;;) {
        if (!worker_mem_get_info(dip)) {
            usleep(MS);
            continue;
        }
        if (mb(dip->free) == ENDING) {
            dip->state = WATCHING;
            break;
        }
        if (blk(dip->free) != 2) {
            continue;
        }
        worker_reserve_one_blk(dip); 
        worker_reserve_one_blk(dip);
    }
}

static void worker_init(dev_info * dip) {
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
    contexts[i] = 0;
    devices[i] = 0;
    sleep(1);
    workers[i] = 0;
    sleep(1);
    return workers+i;
}

static void * worker_thread(void * data) {
    int i = (pthread_t *)data - workers;
    CUresult cr;
    cr = cuDeviceGet(devices+i, i);
    if (cr != CUDA_SUCCESS) {
        fprintf(stderr, "FAIL: cuDeviceGet(%d) return %d\n", i, cr);
        return clear_worker_and_take_a_break(i);
    }
    cr = cuCtxCreate(contexts+i, CU_CTX_SCHED_BLOCKING_SYNC, devices[i]);
    if (cr != CUDA_SUCCESS) {
        fprintf(stderr, "FAIL: cuCtxCreate(%d) return %d\n", i, cr);
        return clear_worker_and_take_a_break(i);
    }
    cr = cuCtxSetCurrent(contexts[i]);
    if (cr != CUDA_SUCCESS) {
        fprintf(stderr, "FAIL: cuCtxSetCurrent(%d) return %d\n", i, cr);
        cuCtxDestroy(contexts[i]);
        return clear_worker_and_take_a_break(i);
    }
    dev_info di = {i, INIT};
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
    for (int i = 0; i < N; i++) {
        int r = pthread_create(workers+i, NULL, worker_thread, workers+i);
	    if (r != 0) {
		    fprintf(stderr, "FAIL: to launch the worker thread! (%d:%d)\n", i, r);
            exit(0);
	    }
    }
    for (int i = 0; ; i = (i + 1) % N) {
        if (workers[i] == 0) {
            sleep(1);
        }
        if (workers[i] != 0) {
            sleep(1);
            continue;
        }
        int r = pthread_create(workers+i, NULL, worker_thread, workers+i);
	    if (r != 0) {
		    fprintf(stderr, "WARNING: restart worker thread fail! (%d:%d)\n", i, r);
            workers[i] = 0;
	    }
        sleep(1);
    }
    return 0;
}

//__attribute ((constructor)) void cudaw_master_init(void) {
void main(int argc, const char * argv[]) {
    printf("cudaw_master_init\n");
    CUresult cr = cuInit(0);
    if (cr != CUDA_SUCCESS) {
		fprintf(stderr, "FAIL: unable to init CUDA dirver... (%d)\n", cr);
        exit(0);
    }
	int r = pthread_create(&master, NULL, master_thread, NULL);
	if (r != 0) {
		fprintf(stderr, "FAIL: unable to launch the master thread (%d)\n", r);
        exit(0);
	}
    pthread_join(master, NULL);
}
