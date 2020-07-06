#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <cuda.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>
#include <pthread.h>

#define MAX_GPUS  32

static int N = 0; // count of GPU devices
static pthread_t master = 0;
static pthread_t workers[MAX_GPUS] = {0};

static CUdevice devices[MAX_GPUS] = {0};
static CUcontext contexts[MAX_GPUS] = {0};

#define BLK_SIZE    0x2000000llu        // 32MB
#define TIP_SIZE    0x100000llu         // 1MB
#define TIP_TAIL    (TIP_SIZE<<1)       // TIP_SIZE * 2
#define TPG_SIZE    0x1000llu           // 4KB
#define TPG_HALF    (TPG_SIZE>>1)

#define MIN_OVERHEAD    0x1000000llu    // 32MB

#define NUM_MASK    0xf8
#define TIP_MASK    7
#define MAX_TIPS    248
#define MIN_TIPS    32

#define CR_SLEEP    1000        // 1ms
#define TIP_SLEEP   5000        // 5ms
#define BLK_SLEEP   20000       // 20ms
#define STATE_SLEEP 1000000     // 1s
#define WATCH_SLEEP 10000       // 10ms
#define WATCH_COUNT 10          


struct dev_info {
    int i;
    int state;
    int tip_state;
    int tip_sn;
    size_t init_free;
    size_t tip_free;
    size_t reserved;
    size_t overhead;
    size_t free;
    size_t total;
    int blk_size;
    int blk_num;
    int tip_size;
    int tip_num;
    CUdeviceptr * blks;
    CUdeviceptr * tips;
};

typedef struct dev_info dev_info;

enum {INIT, RESERVING, RELEASING, TIPTICK, WATCHING};

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
    for (int i = 0; i < 3; i++) {
        cr = cuMemGetInfo(&dip->free, &dip->total);
        if (cr != CUDA_SUCCESS) {
            usleep(CR_SLEEP);
            continue;
        }
        return 1;
    }
    fprintf(stderr, "cuMemGetInfo return %d three times\n", cr);
    return 0;
}

static int make_tip_state(dev_info * dip, int new_state) {
    new_state |= ((dip->tip_num + TIP_MASK) & ~TIP_MASK);
    new_state |= (dip->tip_sn << 8);
    return new_state;
}

static void write_tips(dev_info * dip, int new_state) {
    new_state = make_tip_state(dip, new_state);
    for (int i = 0; i < dip->tip_num; ++i) {
        cuMemsetD32(dip->tips[i], new_state, TIP_SIZE);
    }
}

static int read_tips_page(dev_info * dip, int * tips) {
    int n = TPG_SIZE / sizeof(int);
    int h = TPG_HALF / sizeof(int);
    int val = tips[0];
    for (int i = 0; i < h; i++) {
        if (val != tips[i]) {
            return 0;
        }
    }
    int tip_state = make_tip_state(dip, dip->tip_state);
    for (int i = h; i < n; i++) {
        if (tip_state != tips[i]) {
            return 0;
        }
    }
    int num = (val & NUM_MASK);
    if (num * 2 < dip->tip_num ||
        num > ((dip->tip_num + TIP_MASK) & ~TIP_MASK))
        return 0;
    return (val & TIP_MASK);
}

static int read_tips(dev_info * dip) {
    size_t size = TIP_SIZE;
    void * buf = malloc(size);
    while (buf == NULL && size > TPG_SIZE) {
        size /= 2;
        buf = malloc(size);
    }
    if (buf == NULL) {
        return dip->tip_state;
    }
    CUresult cr;
    int loreq = 0, hireq = 0, refin = 0, other = 0;
    for (int i = 0; i < dip->tip_num; i++) {
        for (int offset = 0; offset < TIP_SIZE; offset += size) {
            cr = cuMemcpyDtoH(buf, dip->tips[i] + offset, size);
            if (cr == CUDA_SUCCESS) {
                for (int pos = 0; pos < size; pos += TPG_SIZE) {
                    switch (read_tips_page(dip, buf + pos)) {
                        case LOREQ:
                            loreq++;
                            break;
                        case HIREQ:
                            hireq++;
                            break;
                        case REFIN:
                            refin++;
                            break;
                        default:
                            other++;
                            break;
                    }
                }
            }
        }
    }
    if (loreq > hireq + refin + other)
        return LOREQ;
    if (hireq > loreq + refin + other)
        return HIREQ;
    if (refin > loreq + hireq + other)
        return REFIN;
    return dip->tip_state;
}

static void worker_rw_black_board(dev_info * dip) {
    switch (dip->tip_state) {
        case START:
            write_tips(dip, CLEAN);
            dip->tip_state = CLEAN;
            break;
        case CLEAN:
            switch (read_tips(dip)) {
                case LOREQ:
                    write_tips(dip, ACKLO);
                    dip->tip_state = ACKLO;
                    dip->state = RELEASING;
                    break;
                case HIREQ:
                    write_tips(dip, ACKHI);
                    dip->tip_state = ACKHI;
                    dip->state = RELEASING;
                    break;
                default:
                    break;
            }
            break;
        case SYNHI:
            switch (read_tips(dip)) {
                case HIREQ:
                    write_tips(dip, ACKHI);
                    dip->tip_state = ACKHI;
                    dip->state = RELEASING;
                    break;
                default:
                    break;
            }
            break;
        case ACKLO:
            switch (read_tips(dip)) {
                case REFIN:
                    write_tips(dip, RESRV);
                    dip->tip_state = RESRV;
                    dip->state = RESERVING;
                    break;
                case HIREQ:
                    write_tips(dip, RESRV);
                    dip->tip_state = RESRV;
                    dip->state = RESERVING;
                    break;
                default:
                    break;
            }
            break;
        case ACKHI:
            switch (read_tips(dip)) {
                case REFIN:
                    write_tips(dip, RESRV);
                    dip->tip_state = RESRV;
                    dip->state = RESERVING;
                    break;
                default:
                    break;
            }
            break;
        case RESRV:
            write_tips(dip, RESRV);
            dip->state = RESERVING;
            break;
        default:
            break;
    }
}

static void worker_release_one_blk(dev_info * dip) {
    dip->blk_num--;
    cuMemFree(dip->blks[dip->blk_num]);
    dip->blks[dip->blk_num] = 0;
    dip->reserved -= BLK_SIZE;
}

static void worker_dec_tips(dev_info * dip) {
    for (int i = 0; i < dip->tip_num; ++i) {
        cuMemFree(dip->tips[i]);
        dip->tips[i] = 0;
    }
    dip->tip_num = 0;
}

static void worker_tip_tick(dev_info * dip) {
    size_t last_free = dip->free;
    size_t last_tip_num = dip->tip_num;
    for (;;) {
        int n = (dip->free / TIP_SIZE) + 1;
        for (int i = 0; i < n; i++) {
            if (dip->tip_num >= MAX_TIPS) {
                break;
            }
            CUdeviceptr devptr = 0;
            cuMemAlloc(&devptr, TIP_SIZE);
            if (devptr == 0) {
                break;
            }
            dip->tips[dip->tip_num++] = devptr;
        }
        usleep(TIP_SLEEP);
        if (!worker_mem_get_info(dip)) {
            return;
        }
        printf("tip_tick(%d,%d) (%d) %d\n",
                dip->init_free>>20, dip->total>>20, dip->free>>20, dip->tip_num);
        if (last_free == dip->free && last_tip_num == dip->tip_num) {
            if (dip->tip_num < MIN_TIPS) {
                worker_release_one_blk(dip);
                dip->tip_free += BLK_SIZE;
                usleep(TIP_SLEEP);
                continue;
            }
            worker_rw_black_board(dip);
            worker_dec_tips(dip);
            return;
        }
        last_free = dip->free;
        last_tip_num = dip->tip_num;
    }
}

static void worker_watching(dev_info * dip) {
    size_t frees[WATCH_COUNT];
    for (int i = 0; i < WATCH_COUNT; i++) {
        if (!worker_mem_get_info(dip)) {
            return;
        }
        frees[i] = dip->free;
        usleep(WATCH_SLEEP);
    }
    size_t max_free = frees[0];
    size_t min_free = frees[1];
    if (max_free < min_free) {
        max_free = frees[1];
        min_free = frees[0];
    }
    size_t average, sum = 0;
    for (int i = 0; i < WATCH_COUNT; i++) {
        sum += frees[i];
        if (frees[i] > max_free) {
            max_free = frees[i];
        }
        else if (frees[i] < min_free) {
            min_free = frees[i];
        }
    }
    average = (sum - max_free - min_free) / (WATCH_COUNT - 2);
    switch (dip->tip_state) {
        case RESRV:
            if (average > dip->tip_free + TIP_SIZE) {
                dip->state = RESERVING;
            }
            else if (average < dip->tip_free - MIN_OVERHEAD) {
                dip->tip_state = START;
                dip->tip_sn++;
                dip->state = TIPTICK;
            }
            break;
        case CLEAN:
            break;
        case ACKLO:
        case ACKHI:
            if (max_free == min_free) {
                if (average > dip->init_free - TIP_TAIL) {
                    dip->tip_state = RESRV;
                    dip->tip_sn++;
                    dip->state = TIPTICK;
                }
            }
            break;
        default:
            dip->state = TIPTICK;
            break;
    }
}

static void worker_release_all_memory(dev_info * dip) {
    int tip_state = dip->tip_state;
    tip_state |= (dip->tip_sn << 8);
    for (int i = 0; i < dip->blk_num; ++i) {
        cuMemsetD32(dip->blks[i], tip_state, BLK_SIZE);
    }
    for (int i = 0; i < dip->blk_num; ++i) {
        cuMemFree(dip->blks[i]);
        dip->blks[i] = 0;
    }
    dip->blk_num = 0;
    dip->reserved = 0;
    dip->state = WATCHING;
}

static void worker_reserve_all_memory(dev_info * dip) {
    size_t last_free = dip->free;
    size_t last_blk_num = dip->blk_num;
    for (;;) {
        int n = (dip->free - MAX_TIPS * TIP_SIZE + BLK_SIZE - 1) / BLK_SIZE;
        for (int i = 0; i < n; i++) {
            CUdeviceptr devptr = 0;
            cuMemAlloc(&devptr, BLK_SIZE);
            if (devptr == 0) {
                break;
            }
            dip->blks[dip->blk_num++] = devptr;
            dip->reserved = dip->blk_num * BLK_SIZE;
        }
        usleep(BLK_SLEEP);
        if (!worker_mem_get_info(dip)) {
            return;
        }
        printf("dip->free(%d,%d) (%d) %d\n",
                dip->init_free>>20, dip->total>>20, dip->free>>20, dip->reserved>>20);
        if (last_free == dip->free && last_blk_num == dip->blk_num) {
            if (dip->free + dip->reserved > dip->init_free) {
                dip->init_free = dip->free + dip->reserved;
                dip->overhead = dip->total - dip->init_free;
            }
            if (dip->init_free - dip->free - dip->reserved < TIP_TAIL) {
                dip->tip_free = dip->free;
                dip->state = WATCHING;
            }
            else {
                if (dip->tip_free < dip->free) {
                    dip->tip_free = dip->free;
                }
                dip->tip_state = RESRV;
                dip->tip_sn++;
                dip->state = TIPTICK;
            }
            return;
        }
        last_free = dip->free;
        last_blk_num = dip->blk_num;
    }
}

static void worker_init(dev_info * dip) {
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
    n = MAX_TIPS;
    dip->tips = malloc(sizeof(CUdeviceptr) * n);
    if (dip->tips == NULL) {
        free(dip->blks);
        dip->blks = NULL;
        return;
    }
    dip->tip_size = n;
    dip->tip_num = 0;
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
    dev_info di = {i, INIT, RESRV, 0};
    worker_init(&di);
    for (;;) {
        usleep(STATE_SLEEP);
        if (!worker_mem_get_info(&di)) {
            continue;
        }
        switch (di.state) {
            case INIT:
                worker_init(&di);
                break;
            case RESERVING:
                worker_reserve_all_memory(&di);
                break;
            case RELEASING:
                worker_release_all_memory(&di);
                break;
            case TIPTICK:
                worker_tip_tick(&di);
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
