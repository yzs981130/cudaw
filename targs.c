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
#include <assert.h>

#include "cudawrt.h"
#include "vaddr.h"

//#define printf(...) do { } while (0)

//
// kernel functions info
//

#define KI_EMPTY_KERNEL_INFO {0}

#define KI_LOOKUP_SIZE      0x10000
#define KI_MAX_ARGC         40

#ifndef KI_PADDING
  #define KI_PADDING        -1
#endif

#define KIS_NORMAL          0x0
#define KIS_TEST            0x1
#define KIS_BYPASS          0x2
#define KIS_PRINT_ARGS      0x4

#define KI_PRINT_CNT_MAX    10
#define KI_PRINT_CNT_MASK   0xfff

struct kernel_info {
    const void *   func;
    unsigned short tail;   // tail_addr: offset in 4K page;
    unsigned short status; // KIS_XXX
    unsigned int   crc;    // func_crc32: crc32 value of first 128 bytes of a kernel func
    unsigned int   cnt;    // counts for being called
    unsigned short argc;   // number of args
    unsigned short size;   // size of memory for copy all arguments with devptr
    unsigned short addc;   // count of device address args
    unsigned short addv[9]; // holds the index of device address args and size of copied args
    unsigned short objc;   // count of device address args
    unsigned short objv[9]; // holds the index of device address args and size of copied args
                            // index=(objv[i] & 0x1f) size=max{((objv[i]>>1) & 0xfff0), 8}
};

// bits for struct kernel_info.status


static struct kernel_info kernel_infos[1024*16] = {
    KI_EMPTY_KERNEL_INFO,
    // TARGS_KI_AUTO_GENERATED_FUNC_INSERT_BELOW
    {0}
};

static unsigned short ki_lookup_table[KI_LOOKUP_SIZE] = {0};
static sem_t ki_sem = {0};

//
// mblk_alloc: alloc buffer for copy args when translate args
//

#define MBLK_PAD_SIZE    32
#define MBLK_MAX_GROUP   64
#define MBLK_GROUP_STEP  16
#define MBLK_UNIT_SIZE   256
#define MBLK_STAMP_DIFF  (4096*256)
#define MBLK_ALIGNMENT   64

struct mblk_item {
    struct mblk_item * next;
    unsigned int     stamp;
    unsigned int     __r1;
    long long        __pad[MBLK_PAD_SIZE];
    void * argv[];
};

struct mblk_group {
    struct mblk_item * head;
    struct mblk_item * tail;
    unsigned int       count;
    unsigned int       total;
    sem_t              sem;
};

static struct mblk_group mblk_groups[MBLK_MAX_GROUP] = {0};

//
// crc32 table
//

static const unsigned int crc32_table[] =
{
  0x00000000, 0x04c11db7, 0x09823b6e, 0x0d4326d9,
  0x130476dc, 0x17c56b6b, 0x1a864db2, 0x1e475005,
  0x2608edb8, 0x22c9f00f, 0x2f8ad6d6, 0x2b4bcb61,
  0x350c9b64, 0x31cd86d3, 0x3c8ea00a, 0x384fbdbd,
  0x4c11db70, 0x48d0c6c7, 0x4593e01e, 0x4152fda9,
  0x5f15adac, 0x5bd4b01b, 0x569796c2, 0x52568b75,
  0x6a1936c8, 0x6ed82b7f, 0x639b0da6, 0x675a1011,
  0x791d4014, 0x7ddc5da3, 0x709f7b7a, 0x745e66cd,
  0x9823b6e0, 0x9ce2ab57, 0x91a18d8e, 0x95609039,
  0x8b27c03c, 0x8fe6dd8b, 0x82a5fb52, 0x8664e6e5,
  0xbe2b5b58, 0xbaea46ef, 0xb7a96036, 0xb3687d81,
  0xad2f2d84, 0xa9ee3033, 0xa4ad16ea, 0xa06c0b5d,
  0xd4326d90, 0xd0f37027, 0xddb056fe, 0xd9714b49,
  0xc7361b4c, 0xc3f706fb, 0xceb42022, 0xca753d95,
  0xf23a8028, 0xf6fb9d9f, 0xfbb8bb46, 0xff79a6f1,
  0xe13ef6f4, 0xe5ffeb43, 0xe8bccd9a, 0xec7dd02d,
  0x34867077, 0x30476dc0, 0x3d044b19, 0x39c556ae,
  0x278206ab, 0x23431b1c, 0x2e003dc5, 0x2ac12072,
  0x128e9dcf, 0x164f8078, 0x1b0ca6a1, 0x1fcdbb16,
  0x018aeb13, 0x054bf6a4, 0x0808d07d, 0x0cc9cdca,
  0x7897ab07, 0x7c56b6b0, 0x71159069, 0x75d48dde,
  0x6b93dddb, 0x6f52c06c, 0x6211e6b5, 0x66d0fb02,
  0x5e9f46bf, 0x5a5e5b08, 0x571d7dd1, 0x53dc6066,
  0x4d9b3063, 0x495a2dd4, 0x44190b0d, 0x40d816ba,
  0xaca5c697, 0xa864db20, 0xa527fdf9, 0xa1e6e04e,
  0xbfa1b04b, 0xbb60adfc, 0xb6238b25, 0xb2e29692,
  0x8aad2b2f, 0x8e6c3698, 0x832f1041, 0x87ee0df6,
  0x99a95df3, 0x9d684044, 0x902b669d, 0x94ea7b2a,
  0xe0b41de7, 0xe4750050, 0xe9362689, 0xedf73b3e,
  0xf3b06b3b, 0xf771768c, 0xfa325055, 0xfef34de2,
  0xc6bcf05f, 0xc27dede8, 0xcf3ecb31, 0xcbffd686,
  0xd5b88683, 0xd1799b34, 0xdc3abded, 0xd8fba05a,
  0x690ce0ee, 0x6dcdfd59, 0x608edb80, 0x644fc637,
  0x7a089632, 0x7ec98b85, 0x738aad5c, 0x774bb0eb,
  0x4f040d56, 0x4bc510e1, 0x46863638, 0x42472b8f,
  0x5c007b8a, 0x58c1663d, 0x558240e4, 0x51435d53,
  0x251d3b9e, 0x21dc2629, 0x2c9f00f0, 0x285e1d47,
  0x36194d42, 0x32d850f5, 0x3f9b762c, 0x3b5a6b9b,
  0x0315d626, 0x07d4cb91, 0x0a97ed48, 0x0e56f0ff,
  0x1011a0fa, 0x14d0bd4d, 0x19939b94, 0x1d528623,
  0xf12f560e, 0xf5ee4bb9, 0xf8ad6d60, 0xfc6c70d7,
  0xe22b20d2, 0xe6ea3d65, 0xeba91bbc, 0xef68060b,
  0xd727bbb6, 0xd3e6a601, 0xdea580d8, 0xda649d6f,
  0xc423cd6a, 0xc0e2d0dd, 0xcda1f604, 0xc960ebb3,
  0xbd3e8d7e, 0xb9ff90c9, 0xb4bcb610, 0xb07daba7,
  0xae3afba2, 0xaafbe615, 0xa7b8c0cc, 0xa379dd7b,
  0x9b3660c6, 0x9ff77d71, 0x92b45ba8, 0x9675461f,
  0x8832161a, 0x8cf30bad, 0x81b02d74, 0x857130c3,
  0x5d8a9099, 0x594b8d2e, 0x5408abf7, 0x50c9b640,
  0x4e8ee645, 0x4a4ffbf2, 0x470cdd2b, 0x43cdc09c,
  0x7b827d21, 0x7f436096, 0x7200464f, 0x76c15bf8,
  0x68860bfd, 0x6c47164a, 0x61043093, 0x65c52d24,
  0x119b4be9, 0x155a565e, 0x18197087, 0x1cd86d30,
  0x029f3d35, 0x065e2082, 0x0b1d065b, 0x0fdc1bec,
  0x3793a651, 0x3352bbe6, 0x3e119d3f, 0x3ad08088,
  0x2497d08d, 0x2056cd3a, 0x2d15ebe3, 0x29d4f654,
  0xc5a92679, 0xc1683bce, 0xcc2b1d17, 0xc8ea00a0,
  0xd6ad50a5, 0xd26c4d12, 0xdf2f6bcb, 0xdbee767c,
  0xe3a1cbc1, 0xe760d676, 0xea23f0af, 0xeee2ed18,
  0xf0a5bd1d, 0xf464a0aa, 0xf9278673, 0xfde69bc4,
  0x89b8fd09, 0x8d79e0be, 0x803ac667, 0x84fbdbd0,
  0x9abc8bd5, 0x9e7d9662, 0x933eb0bb, 0x97ffad0c,
  0xafb010b1, 0xab710d06, 0xa6322bdf, 0xa2f33668,
  0xbcb4666d, 0xb8757bda, 0xb5365d03, 0xb1f740b4
};

static unsigned int xcrc32(const unsigned char *buf, int len) {
    unsigned int crc = 0xFFFFFFFF;
    while (len--) {
        crc = (crc << 8) ^ crc32_table[((crc >> 24) ^ *buf) & 255];
        buf++;
    }
    return crc;
}

static void printerr() {
    char *errstr = dlerror();
    if (errstr != NULL) {
        printf ("A dynamic linking error occurred: (%s)\n", errstr);
    }
}

//
// cuda APIs that be wrapped by this file
//

#define DEFSO(func)  static cudaError_t (*so_##func)

#define LDSYM(func)  do { \
    so_##func = dlsym(so_handle, #func); \
    printerr(); \
} while(0)

DEFSO(cudaLaunchKernel)(const void * func, dim3 gridDim, dim3 blockDim, void ** args, size_t sharedMem, cudaStream_t stream);

static void * so_handle = NULL;

__attribute ((constructor)) void cudaw_targs_init(const void * funcs[]) {
    printf("cudaw_targs_init\n");
    // load cudart API funcs
    so_handle = dlopen(LIB_STRING_RT, RTLD_NOW);
    if (!so_handle) {
        fprintf (stderr, "FAIL: %s\n", dlerror());
        exit(1);
    }
    printerr();
    LDSYM(cudaLaunchKernel);
    // sem_init for ki_sem
    sem_init(&ki_sem, 0, 1);
    // sem_init for all mblk group
    for (int i = 0; i < MBLK_MAX_GROUP; ++i) {
        sem_init(&mblk_groups[i].sem, 0, 1);
    }
}

__attribute ((destructor)) void cudawtargs_fini(void) {
    printf("cudaw_targs_fini\n");
    sem_destroy(&ki_sem);
    for (int i = 0; i < MBLK_MAX_GROUP; ++i) {
        sem_destroy(&mblk_groups[i].sem);
    }
    if (so_handle) {
        dlclose(so_handle);
    }
}

static void * mblk_group_alloc(struct mblk_group * g,  size_t size) {
    struct mblk_item * tmp;
    if (g->head != NULL) {
        sem_wait(&g->sem);
        if (g->head->stamp + MBLK_STAMP_DIFF < g->count) {
            if (g->head != g->tail) {
                tmp = g->head;
                g->tail->next = tmp;
                g->tail = tmp;
                g->head = tmp->next;
                tmp->stamp = ++(g->count);
                sem_post(&g->sem);
                return (void *)(tmp->argv);
            }
        }
        sem_post(&g->sem);
    }
    void * p = malloc(size);
    memset(p, KI_PADDING, size);
    tmp = (struct mblk_item *)p;
    sem_wait(&g->sem);
    tmp->stamp = ++(g->count);
    if (g->head == NULL) {
        g->head = tmp;
        g->tail = tmp;
    }
    g->tail->next = tmp;
    g->tail = tmp;
    g->total += size;
    sem_post(&g->sem);
#ifdef PRINT_MBLK_TOTAL
    printf("mblk_malloc: (%lu) count: %u total: %u\n", size, g->count, g->total);
#endif
    return (void *)(tmp->argv);
}

static void * mblk_alloc(size_t size, void * p) {
    size = size < MBLK_ALIGNMENT ? MBLK_ALIGNMENT : size;
    size += sizeof(struct mblk_item) * 2;
    size_t unit_size = MBLK_UNIT_SIZE;
    for (int i = 0; i < MBLK_MAX_GROUP / MBLK_GROUP_STEP; ++i) {
        int k = (size + unit_size - 1) / unit_size;
        if (k <= MBLK_GROUP_STEP) {
            size = k * unit_size;
            void * blk = mblk_group_alloc(&mblk_groups[i * MBLK_GROUP_STEP + k], size);
            blk = (void *)((unsigned long long)blk | ((MBLK_ALIGNMENT-1) & (unsigned long long)p));
            return blk;
        }
        unit_size *= MBLK_GROUP_STEP;
    }
    printf("FAIL: %lu is larger than the max size of mblk (%lu)\n", size, unit_size);
    exit(1);
    return NULL;
}

static void print_all_func_cnt(void) {
    int n = sizeof(kernel_infos) / sizeof(struct kernel_info);
    for (int k = 0; k < n; k++) {
        struct kernel_info * kip = &kernel_infos[k];
        printf("tail: %u crc: %u cnt: %u\n", kip->tail, kip->crc, kip->cnt);
    }
}

static struct kernel_info * ki_lookup(const void * func) {
    int hash = (int)((unsigned long long)func & 0xffff0) >> 4;
    for (int i = hash; i < KI_LOOKUP_SIZE; i = (i + 1) % KI_LOOKUP_SIZE) {
        int k = ki_lookup_table[i];
        if (func == kernel_infos[k].func) {
            return kernel_infos + k;
        }
        if (k == 0) {
            sem_wait(&ki_sem);
            assert(KI_LOOKUP_SIZE > 0xffff);
            unsigned short tail = (unsigned short)((unsigned long long)func & 0xfff);
            int n = sizeof(kernel_infos) / sizeof(struct kernel_info);
            const int nbytes = 128;
            unsigned int crc = xcrc32(func, nbytes);
            for (k = 1; kernel_infos[k].argc ; ++k) { // 0 for empty func
                if (crc == kernel_infos[k].crc && tail == kernel_infos[k].tail) {
                    kernel_infos[k].func = func;
                    ki_lookup_table[i] = k;
                    sem_post(&ki_sem);
                    return kernel_infos + k;
                }
            }
            assert(kernel_infos[k+1].argc == 0);
            struct kernel_info * new_func = &kernel_infos[k];
            new_func->func = func;
            new_func->tail = tail;
            new_func->crc = crc;
            new_func->status = KIS_TEST;
            printf("new-func: %p tail: 0x%x crc: %u cnt: %u\n", func, tail, crc, new_func->cnt);
            return new_func;
        }
    }
    assert(0);
}

// Translate pointer args[i]
static void trans_args_addv(struct kernel_info * kip, void ** args, void ** pargs) {
    void * base = mblk_alloc(kip->size, NULL);
    int copied = 0;
    for (int k = 0; k < kip->objc; ++k) {
        int i = kip->objv[k] & 0x1f;
        int s = (kip->objv[k] > 1) & 0xfff0;
        pargs[i] = base + copied;
        if (s > 0) {
            memcpy(pargs[i], args[i], s);
            copied += s;
        }
        else {
            memcpy(pargs[i], args[i], 8);
            copied += 16;
        }
    }
#ifdef VA_ENABLE_VIR_ADDR
    void ** addv = (void **)base;
    for (int k = 0; k < kip->addc; ++k) {
        unsigned short idx = kip->addv[k];
        VtoR1(addv[idx]);
    }
#endif
}

// Copy values args[i] to pargs[i]
static void trans_cp_args(struct kernel_info * kip, void ** args, void ** pargs) {
    for (int i = 0; i < kip->argc; ++i) {
        pargs[i] = args[i];
    }
}

static void ki_print_func(struct kernel_info * kip) {
    if (kip->cnt <= KI_PRINT_CNT_MAX || (kip->cnt & KI_PRINT_CNT_MASK) == 1) {
        printf("tail: 0x%x crc: %u argc: %d status: %d cnt: %u\n",
                kip->tail, kip->crc, kip->argc, kip->status, kip->cnt);
    }
}

// TT

#define BUF_SIZE 1024
#define LL long long

struct addr_range{
    void ** s;
    void ** e;
};

struct mem_maps {
    union {
        struct {
            unsigned int num;        // the total number of addr range with rw
            unsigned int heap;       // the index of the heap range
            unsigned int stack;      // the start index of the first stack range
            unsigned int stack_num;  // the total number of stack ranges
        };
        struct addr_range ranges[1]; // NOTE: the first range starts from 1
    };
};

struct mem_maps * load_text_maps() {
    int cnt=0,i;
    pid_t pid = getpid();
    char proc_pid_path[BUF_SIZE];
    char buf[BUF_SIZE];
    unsigned int size = 255;
    sprintf(proc_pid_path, "/proc/%d/maps", pid);
    FILE* fp = fopen(proc_pid_path, "r");
    struct mem_maps* res=malloc(sizeof(struct addr_range) * (size+1));
    memset(res, 0, sizeof(struct mem_maps));
    if (NULL != fp){
        while( fgets(buf, BUF_SIZE-1, fp)!= NULL ){
            if (strstr(buf, " r-xp ") == NULL) {
                continue;
            }
            if (strstr(buf, "(deleted)") != NULL) {
                continue;
            }
            for (int k=0; k < 27; ++k) {
                if (buf[k] == '-') {
                    buf[k] = ' ';
                    break;
                }
            }
            res->num++;
            printf("%d %s", res->num, buf);
            char add_s[20],add_e[20];
            sscanf(buf,"%s %s",add_s,add_e);
            void ** s = (void **)strtoull(add_s,NULL,16);
            void ** e = (void **)strtoull(add_e,NULL,16);
            i = res->num;
            if (i >= size) {
                size = size * 2 + 1;
                res = realloc(res, sizeof(struct addr_range) * (size+1));
            }
            res->ranges[i].s=s;
            res->ranges[i].e=e;
        }
        fclose(fp);
    }
    return res;
}

struct mem_maps * load_mem_maps() {
    int cnt=0,i;
    pid_t pid = getpid();
    //printf("%d\n",pid);
    char proc_pid_path[BUF_SIZE];
    char buf[BUF_SIZE];
    unsigned int size = 511;
    sprintf(proc_pid_path, "/proc/%d/maps", pid);
    FILE* fp = fopen(proc_pid_path, "r");
    struct mem_maps* res=malloc(sizeof(struct addr_range) * (size+1));
    memset(res, 0, sizeof(struct mem_maps));
    res->heap = 1;
    res->stack = 2;
    if (NULL != fp){
        while( fgets(buf, BUF_SIZE-1, fp)!= NULL ){
            if (strstr(buf, " rw-p ") == NULL) {
                continue;
            }
            if (strstr(buf, "(deleted)") != NULL) {
                continue;
            }
            for (int k=0; k < 27; ++k) {
                if (buf[k] == '-') {
                    buf[k] = ' ';
                    break;
                }
            }
            res->num++;
            int flag=0;
            if (strstr(buf, "[heap]") != NULL) {
                flag = 3;
            }
            else if (strstr(buf, "[stack]") != NULL) {
                res->stack_num++;
                flag = 4;
            }
            //printf("%s", buf);
            char add_s[20],add_e[20];
            sscanf(buf,"%s %s",add_s,add_e);
            //printf("%s %s\n",add_s,add_e);
            void ** s = (void **)strtoull(add_s,NULL,16);
            void ** e = (void **)strtoull(add_e,NULL,16);
            i = res->num;
            if (i >= (size - 2)) { // for empty heap and stack
                size = size * 2 + 1;
                res = realloc(res, sizeof(struct addr_range) * (size+1));
            }
            if(flag==3) { // heap = 1
                if (res->num > 1) {
                    res->ranges[i] = res->ranges[1];
                }
                res->ranges[1].s=s;
                res->ranges[1].e=e;
            } else if (flag==4) { // stack = 2
                if (res->num > (1+res->stack_num)) {
                    res->ranges[i] = res->ranges[1+res->stack_num];
                }
                res->ranges[1+res->stack_num].s=s;
                res->ranges[1+res->stack_num].e=e;
            } else {
                res->ranges[i].s=s;
                res->ranges[i].e=e;
            }
        }
        fclose(fp);
    }
    return res;
}

struct addr_val {
    unsigned LL val;
};

//int cnt=0;
//void func() {
// search for the addr and val that 'minVal <= val < maxVal'
// the return list ends with (nil, 0)

struct addr_val * search_dev_addr_vals(struct mem_maps * maps) {
    unsigned int cnt = 0, size = 1024;
    struct addr_val* devals = malloc(sizeof(struct addr_val) * (size+1));
    for(int i=1; i <= maps->num; ++i) {
        void ** pp;
        for(pp = maps->ranges[i].s; pp<maps->ranges[i].e; ++pp) {
            if(cudawIsDevAddr(*pp)) {
                if(cnt >= size) {
                    size <<= 1;
                    devals = realloc(devals, (size+1)*sizeof(struct addr_val));
                }
                devals[cnt++].val = ~(unsigned LL)*pp;
            }
        }
    }
    devals[cnt].val=0llu;
    return devals;
}

// count the number of appearances of a give val in the addr_vals
int count_value(struct addr_val * addr_vals, void* val) {
    int cnt = 0;
    struct addr_val * vp = addr_vals;
    for(; vp->val; ++vp) {
        if((void *)~vp->val == val) {
            ++cnt;
        }
    }
    return cnt;
}

unsigned short guess_argc(struct mem_maps * maps,void** args) {
    int i, k;
    for (i=0; ; ++i) {
        void ** val = (void **)args[i];
        printf("guess_argc argi: %d = %p\n", i, args[i]);
        for (k=1; k<=maps->num; ++k) {
            if (maps->ranges[k].s <= val && val < maps->ranges[k].e) {
                break;
            }
        }
        if (k > maps->num) {
            break;
        }
        printf("guess_argc argi: %d in %d (%p)\n", i, k, val);
        fflush(stdout);
    }
    return i;
}

#define MAX_BOUNDRY_DIFF 4096llu

void * find_boundry_in_args(void ** args, unsigned short argc, void * ptr) {
    void * boundry = ptr + MAX_BOUNDRY_DIFF;
    for(int i=0; i<argc; ++i) {
        void * val = args[i];
        if (ptr < val && val < boundry) {
            boundry = val;
        }
    }
    return boundry;
}

void * find_boundry_in_stack(struct mem_maps * maps, void * sp, void * ptr) {
    void * boundry = ptr + MAX_BOUNDRY_DIFF;
    for(int i=1; i<=maps->num; ++i) {
        if ((void **)sp < maps->ranges[i].s || (void **)sp >= maps->ranges[i].e)
            continue;
        if ((void **)prt < maps->ranges[i].s || (void **)prt >= maps->ranges[i].e)
            continue; // TODO
        void **pp = (void **)sp;
        for(; pp < maps->ranges[i].e; ++pp) {
            void * val = *pp;
            if ((ptr + 8) <= val && val < boundry) {
                printf("boundry: %d %p %p %p\n", i, sp, pp, val);
                boundry = val;
            }
        }
        struct mem_maps * texts = load_text_maps();
        for (pp = (void**)ptr; ptr < boundry; ++pp) {
            int i = texts->num;
            for (; i > 0; --i) {
                if ((void **)*pp >= texts->ranges[i].s &&
                    (void **)*pp < texts->ranges[i].e) {
                    if ((void **)*(pp-1) > pp
                    printf("stack: %p %p - func %d\n", pp, *pp, i);
                    break;
                }
            }
            if (i == 0) {
                printf("stack: %p %p\n", pp, *pp);
            }
        }
            
    pp = (void **)sp + 0x1000;
    if (pp >= maps->ranges[i].e) {
        pp = maps->ranges[i].e - 1;
    }
    for (; pp >= (void **)sp; pp--) {
        int i = texts->num;
        for (; i > 0; --i) {
            if ((void **)*pp >= texts->ranges[i].s &&
                (void **)*pp < texts->ranges[i].e) {
                printf("stack: %p %p - func %d\n", pp, *pp, i);
                break;
            }
        }
        if (i == 0) {
            printf("stack: %p %p\n", pp, *pp);
        }
    }
    free(texts);
        break;
    }
    static int cc = 0;
    if (++cc == 17) {
        //do { sleep(1); } while (1);
    }
    return boundry;
}

void * find_boundry(struct mem_maps * maps, void * ptr) {
    void * boundry = ptr + MAX_BOUNDRY_DIFF;
    for(int i=0; i<=maps->num; ++i) {
        void **pp;
        for(pp = maps->ranges[i].s; pp < maps->ranges[i].e; ++pp) {
            if (i == maps->stack) { // TODO
                if (pp < &boundry) {
                    pp = &boundry;
                }
            }
            void * val = *pp;
            if ((ptr + 8) < val && val < boundry) {
if (val - ptr < 2048) {
    printf("boundry: %d %p %p\n", i, pp, val);
}
                boundry = val;
            }
        }
    }
if (boundry - ptr < 16) {
    for (void ** p = &boundry; p < (&boundry) + 0x1000; p++) {
        printf("stack: %p %p\n", p, *p);
    }
    exit(0);
}
    return boundry;
}

enum {
    USE_PARGS, USE_ARGS, BYPASS_FUNC
};

static int trans_args(const void * func, void ** args, void ** pargs) {
    struct kernel_info * kip = ki_lookup(func);

    if (kip->status & KIS_TEST) {
        if (kip->argc > 0) // new func found
            return USE_ARGS;
        // Guass args
        struct mem_maps * maps = load_mem_maps();
        printf("maps: num: %d heap: %d stack: %d stack_num: %d\n",
            maps->num, maps->heap, maps->stack, maps->stack_num);
        //for (int i = 1; i <= maps->num; ++i) {
        //    struct addr_range * r = maps->ranges + i; 
        //    printf("maps.ranges[%d] = ( %llx , %llx )\n", i, r->s, r->e);
        //}
        fflush(stdout);

        struct addr_val * vals = search_dev_addr_vals(maps);
        //for (int i = 0; vals[i].ptr != NULL && vals[i].val != 0ull; ++i) {
        //    printf("vals: %p %llx\n", vals[i].ptr, vals[i].val);
        //}
        fflush(stdout);

        kip->argc = guess_argc(maps, args);
        printf("(0x%x, %u) == argc: %d (args=%p)\n", kip->tail, kip->crc, kip->argc, args);
        for (int i = 0; i < kip->argc; ++i) {
            fflush(stdout);
            if (((unsigned long long )args[i] & 0x7) != 0) {
                printf("argi: %d of %p not aligned\n", i, args[i]);
                continue;
            }
            void * boundry = find_boundry_in_args(args, kip->argc, args[i]);
            if (boundry - args[i] < 8) {
                printf("argi: %d of %p boundry %p\n", i, args[i], boundry);
                continue;
            }
            else if (boundry - args[i] == 8) {
                void * devptr = *(void **)args[i];
                if (cudawIsDevAddr(devptr)) {
                    int c = count_value(vals, devptr);
                    printf("(0x%x, %u) == argi: %d(%p) count_value: %d\n", 
                                kip->tail, kip->crc, i, args[i], c);
                    kip->addv[kip->addc] = kip->size / sizeof(void*);
                    kip->size += 16;
                    kip->objv[kip->objc] = i;
                    kip->addc++;
                    kip->objc++;
                }
                else {
                    printf("argi: %d(%p) not devptr %p\n", i, args[i], devptr);
                }
                continue;
            }
            else if (boundry - args[i] >= MAX_BOUNDRY_DIFF) {
                boundry = find_boundry_in_stack(maps, pargs, args[i]);
            } else {
                void * stack_boundry = find_boundry_in_stack(maps, pargs, args[i]);
                if (stack_boundry < boundry) {
                    boundry = stack_boundry;
                }
            }
            void ** pv = (void **)args[i];
            size_t n = (boundry - args[i]) / sizeof(void*);
            unsigned short old_addc = kip->addc;
            for (int k = 0; k < n; ++k) {
                if (cudawIsDevAddr(pv[k])) {
                    int c = count_value(vals, pv[k]);
                    printf("(0x%x, %u) == argi: %d(%p)[%d] count_value: %d\n", 
                                kip->tail, kip->crc, i, args[i], k, c);
                    if (c > 2) {
                        kip->addv[kip->addc] = kip->size / sizeof(void*) + k;
                        kip->addc++;
                    }
                }
            }
            if (kip->addc > old_addc) {
                unsigned short size = (boundry - args[i] + 15) & 0xfff0;
                kip->size += size;
                kip->objv[kip->objc] = i + (size << 1);
                kip->objc++;
            }
            printf("(0x%x, %u) == argi: %d(%p) size: %lu\n",
                        kip->tail, kip->crc, i, args[i], n * sizeof(void*));
        }
        fflush(stdout);
        if (kip->addc == 0) {
            *kip = kernel_infos[0];
        }
        sem_post(&ki_sem);
        if (kip->argc > 0) {
        printf("{NULL, 0x%x, 0, %uu, 0, %u, %u, %u, {%u", kip->tail, kip->crc, 
                            kip->argc, kip->size, kip->addc, kip->addv[0]);
        for (int k = 1; k < kip->addc; ++k) {
            printf(",%u", kip->addv[k]);
        }
        printf("}, %u, {", kip->objc);
        for (int k = 0; k < kip->objc; ++k) {
            unsigned short v = kip->objv[k];
            unsigned short s = (v >> 1) & 0xfff0u;
            unsigned short i = v & 0x1fu;
            if (k > 0)
                printf(",");
            if (s > 0) 
                printf("%d*2u+%d", s, i);
            else
                printf("%d", i);
        }
        printf("},\n");
        }
        fflush(stdout);
        free(vals);
        free(maps);
        return USE_ARGS;
    }

    ++kip->cnt;
    ki_print_func(kip);


#ifdef KI_TEST_FUNC
    if (!(kip->status & KIS_TEST)) {
        return USE_ARGS;
    }

 #ifdef KI_TEST_ARGC
    trans_cp_args(kip, args, pargs);
    return USE_PARGS;
 #endif

  #ifdef KI_TEST_DEVPTR
    ki_test_devptr(kip, args);
    return USE_ARGS;
  #endif
#endif // KI_TEST_FUNC

    int use = USE_PARGS;
    if (kip->size > 0) {
        trans_cp_args(kip, args, pargs);
        trans_args_addv(kip, args, pargs);
    }
    else {
        use = USE_ARGS;
    }
    if (kip->status & KIS_BYPASS) {
        use = BYPASS_FUNC;
    }
    return use;
}

cudaError_t cudawLaunchKernel(const void * func, dim3 gridDim, dim3 blockDim, void ** args, size_t sharedMem, cudaStream_t stream) {
    cudaError_t r = 0;

#ifdef KI_BYPASS_ALL_FUNC
    return r;
#endif

#ifdef KI_DISABLE_TRANS_ARGS
    r = so_cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);
    return r;
#endif

    void * pargs[KI_MAX_ARGC] = {0};
    switch (trans_args(func, args, pargs)) {
    case USE_PARGS:
        r = so_cudaLaunchKernel(func, gridDim, blockDim, pargs, sharedMem, stream);
        break;
    case USE_ARGS:
        r = so_cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);
        break;
    case BYPASS_FUNC:
        break;
    default:
        assert(0);
        break;
    }
    return r;
}
