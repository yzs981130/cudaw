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
#include "targs.h"

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
#define KIS_PARGS_ARGC      0x2
#define KIS_PARGS_CPALL     0x3
#define KIS_ACTION_MASK     0xf

#define KIS_NEW_FUNC        0x10
#define KIS_BYPASS          0x20
#define KIS_PRINT_ARGS      0x40

#define KI_PRINT_CNT_MAX    10
#define KI_PRINT_CNT_MASK   0xfff

struct kernel_info {
    const void *   func;    // point to the func when in use
    unsigned int   offset;  // the offset of a kernel func in it's lib.so
    unsigned char  lib;     // index of the lib.so
    unsigned char  status;  // KIS_XXX
    unsigned short argc;    // number of args
    unsigned short size;    // size of memory for copy all arguments with devptr
    unsigned char  objc;    // count of device address (devptr) args
    unsigned char  addc;    // count of devptr args
    unsigned short objv[10];// holds the index of devptr args and size of copied args
                            // index=(objv[i] & 0x000f) size=max{objv[i] & 0x0ff0, 8}
                            // 0x00ii the index of devptr args (all size is 8)
                            // 0x4ssi size & index of obj args (unknowd i/o)
                            // 0x5ssi size & index of obj args (as input)
                            // 0x6ssi size & index of obj args (as output)
                            // 0x7ssi size & index of obj args (as input/output)
                            // 0x8tti type & index of obj args
                            // when addc == 0, objv[10..19] locates in addv[]
    unsigned short addv[10];// holds the index of devptr args and size of copied args
    unsigned int   cnt;     // counts for being called. objv[20..21] locates in cnt
};

typedef struct kernel_info kernel_info;

static kernel_info kernel_infos[1024*16] = {
    KI_EMPTY_KERNEL_INFO,
    // TARGS_KI_AUTO_GENERATED_FUNC_INSERT_BELOW
    {0}
};

struct kernel_lib {
    void * start;
    void * end;
};

typedef struct kernel_lib kernel_lib;

static const char * ki_lib_names[] = {
    "/libtorch.so",
};

static kernel_lib kernel_libs[256] = {
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

static void * so_handle = NULL;

#define LL long long

struct addr_range {
    void * s;
    void * e;
};

struct mem_maps {
    union {
        struct {
            unsigned int num;        // the total number of addr range with rw
            unsigned int heap;       // the index of the heap range
            unsigned int stack;      // the start index of the first stack range
            unsigned int thread;     // the stack of the current thread
        };
        struct {
            unsigned int tnum;       // the total number of text range with xp
            unsigned int cudaw;      // the index of the cudawrt text
            unsigned int torch;      // the index of the libtorch text
            unsigned int __pad;
        };
        struct addr_range ranges[1]; // NOTE: the first range starts from 1
    };
};

struct addr_val {
    unsigned LL val;
};


typedef struct mem_maps mem_maps;
typedef struct addr_range addr_range;
typedef struct addr_val addr_val;

typedef unsigned char       BYTE;
typedef unsigned short      WORD;
typedef unsigned int        DWORD;
typedef unsigned long long  QWORD;

static mem_maps * ki_maps = NULL;
static mem_maps * ki_text = NULL;
static void * ki_bottom = NULL;
static void * ki_top = NULL;

static FILE * open_proc_maps() {
    pid_t pid = getpid();
    char proc_pid_path[64];
    sprintf(proc_pid_path, "/proc/%d/maps", pid);
    FILE * fp = fopen(proc_pid_path, "r");
    if (fp == NULL) {
        fprintf(stderr, "FAIL: Unable to open %s\n", proc_pid_path);
    }
    return fp;
}

static void load_lib_maps(const void * func) {
    FILE* fp = open_proc_maps();
    if (NULL != fp) {
        const int BUF_SIZE = 2048;
        int cnt=0, i;
        char buf[BUF_SIZE];
        while( fgets(buf, BUF_SIZE-1, fp)!= NULL ){
            if (strstr(buf, " r-xp ") == NULL) {
                continue;
            }
            if (strstr(buf, "(deleted)") != NULL) {
                continue;
            }
            for (int k = 0; k < 27; ++k) {
                if (buf[k] == '-') {
                    buf[k] = ' ';
                    break;
                }
            }
            //printf("%d %s", maps->num, buf);
            char add_s[20], add_e[20];
            sscanf(buf,"%s %s",add_s,add_e);
            void * s = (void *)strtoull(add_s,NULL,16);
            void * e = (void *)strtoull(add_e,NULL,16);
            if (s <= func && func < e) {
                printf("WARNING: new-lib: func %p in %s\n", func, buf);
                int j = 0;
                while (kernel_libs[j].start != NULL)
                    j++;
                kernel_libs[j].start = s;
                kernel_libs[j].end = e;
            }
            else for (int j=0; j < sizeof(ki_lib_names)/sizeof(void*); ++j) {
                if(strstr(buf, ki_lib_names[j]) != NULL) {
                    kernel_libs[j].start = s;
                    kernel_libs[j].end = e;
                    break;
                }
            }
        }
        fclose(fp);
    }
}

static mem_maps * load_text_maps() {
    FILE* fp = open_proc_maps();
    if (NULL != fp) {
        const int BUF_SIZE = 2048;
        int cnt=0, i;
        char buf[BUF_SIZE];
        unsigned int size = 254;
        mem_maps * maps = malloc(sizeof(addr_range) * (size+1));
        memset(maps, 0, sizeof(mem_maps));
        while( fgets(buf, BUF_SIZE-1, fp)!= NULL ){
            if (strstr(buf, " r-xp ") == NULL) {
                continue;
            }
            if (strstr(buf, "(deleted)") != NULL) {
                continue;
            }
            for (int k = 0; k < 27; ++k) {
                if (buf[k] == '-') {
                    buf[k] = ' ';
                    break;
                }
            }
            maps->num++;
            //printf("%d %s", maps->num, buf);
            char add_s[20], add_e[20];
            sscanf(buf,"%s %s",add_s,add_e);
            void * s = (void *)strtoull(add_s,NULL,16);
            void * e = (void *)strtoull(add_e,NULL,16);
            i = maps->num;
            if (s <= (void *)load_text_maps &&
                     (void *)load_text_maps < e) {
                maps->cudaw = i;
            }
            else if (strstr(buf, "/libtorch.so") != NULL) {
                maps->torch = i;
            }
            if (i >= size) {
                size = (size + 1) * 2;
                maps = realloc(maps, sizeof(addr_range) * (size+1));
            }
            maps->ranges[i].s = s;
            maps->ranges[i].e = e;
        }
        fclose(fp);
        ki_text = maps;
    }
    return ki_text;
}

static mem_maps * load_mem_maps() {
    FILE* fp = open_proc_maps();
    if (NULL != fp) {
        const int BUF_SIZE = 2048;
        int cnt=0, i;
        char buf[BUF_SIZE];
        unsigned int size = 254;
        mem_maps * maps = malloc(sizeof(addr_range) * (size+1));
        memset(maps, 0, sizeof(mem_maps));
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
            maps->num++;
            char add_s[20],add_e[20];
            sscanf(buf,"%s %s",add_s,add_e);
            void * s = (void *)strtoull(add_s,NULL,16);
            void * e = (void *)strtoull(add_e,NULL,16);
            i = maps->num;
            if (s <= (void *)&s && (void *)&s < e) {
                maps->thread = i;
            }
            if (strstr(buf, "[stack]") != NULL) {
                maps->stack = i;
            }
            else if (strstr(buf, "[heap]") != NULL) {
                maps->heap = i;
            }
            if (i >= size) {
                size = (size + 1) * 2;
                maps = realloc(maps, sizeof(addr_range) * (size+1));
            }
            maps->ranges[i].s = s;
            maps->ranges[i].e = e;
        }
        fclose(fp);
		ki_maps = maps;
    }
    return ki_maps;
}

struct args_type {
    char type[256];
};
struct func_args {
    union {
        unsigned int num;
        struct args_type types[1]; //reverse
    };
};


struct func_args* get_func_args(kernel_info * kip) {
    unsigned int size = 255;
    char nm_res[size+1];

    memset(nm_res, 0, sizeof(char));
    FILE* fp = open_proc_maps();
    const int BUF_SIZE = 2048;
    int cnt=0, i;
    char buf[BUF_SIZE];
    
    struct func_args* res=malloc(sizeof(struct args_type) * (size+1));
    memset(res, 0, sizeof(struct func_args));

    if (NULL != fp) {
        while( fgets(buf, BUF_SIZE-1, fp)!= NULL ){
            if (strstr(buf, " r-xp ") == NULL) {
                continue;
            }
            if (strstr(buf, "(deleted)") != NULL) {
                continue;
            }
            int k;
            for ( k=0; buf[k]!='\0'; ++k) {
                if (buf[k] == '-') {
                    buf[k] = ' ';
                    break;

                }
            }
            for ( ; buf[k]!='\0'; ++k) {
                if(buf[k] == '\n') {
                    buf[k] = '\0';
                    break;
                }
            }
            char add_s[20],add_e[20];
            sscanf(buf,"%s %s",add_s,add_e);
            void ** s = (void **)strtoull(add_s,NULL,16);
            void ** e = (void **)strtoull(add_e,NULL,16);

            if(kernel_libs[kip->lib].start==s && kernel_libs[kip->lib].end == e) {
                strcpy(nm_res,strstr(buf,"/"));
                break;
            }

        }
        fclose(fp);
    }
    if(nm_res[0]==0) {
        printf("not find func\n");
        return res;
    }
    
    char nm_string[BUF_SIZE];
    sprintf(nm_string,"nm %s",nm_res);
    //printf("find:nm_string:%s",nm_string);
    FILE* nm_fp = popen(nm_string, "r");
    
    if (NULL != nm_fp){
        printf("find:nm_string:%s\n",nm_string);
        while( fgets(buf, BUF_SIZE-1, nm_fp)!= NULL ){
            if(strstr(buf,"no symbols") != NULL) {
                continue;
            }
            char add_pos[20];
            sscanf(buf,"%s",add_pos);
            unsigned LL tmp = strtoull(add_pos,NULL,16);
            if (kip->offset != tmp) {
                continue;
            }
            
            char t;
            char func_find[512];
            sscanf(buf,"%s %c %s",add_pos,&t,func_find);
            char filt_string[BUF_SIZE];
            sprintf(filt_string,"c++filt %s",func_find);
            printf("filt_string:%s\n",filt_string);
            FILE *filt_fp = NULL;
            char filt_buf[1<<16];
            filt_fp = popen(filt_string, "r");
            if (NULL == filt_fp) {
                printf("c++filt popen fail\n");
                continue;
            }
            if ( fgets(filt_buf, BUF_SIZE-1, filt_fp)!= NULL ) {
                int filt_len=0,last_parenth=0;

                printf("filt_buf:%s",filt_buf);
                for(; filt_buf[filt_len] != '\0'; ++filt_len) {
                    if(filt_buf[filt_len]==')') {
                        last_parenth = filt_len;
                    }
                }
                int s_p = 0, m_p = 0, b_p = 0, a_p = 0, com_pos = last_parenth;
                for(int p = last_parenth; p >= 0; --p) {
                    if(filt_buf[p] == ')') {
                        --s_p;
                    } else if (filt_buf[p] == ']') {
                        --m_p;
                    } else if (filt_buf[p] == '}') {
                        --b_p;
                    } else if (filt_buf[p] == '>') {
                        --a_p;
                    } else if (filt_buf[p] == '(') {
                        ++s_p;
                    } else if (filt_buf[p] == '[') {
                        ++m_p;
                    } else if (filt_buf[p] == '{') {
                        ++b_p;
                    } else if (filt_buf[p] == '<') {
                        ++a_p;
                    }
                    if ((filt_buf[p] == ','&&(a_p==0 && s_p == -1 && m_p == 0 && b_p == 0))||(a_p==0 && s_p == 0 && m_p == 0 && b_p == 0)) {
                        res->num++;
                        i = res->num;

                        if (i >= size) {
                            size = size * 2 + 1;
                            res = realloc(res, sizeof(char*) * (size+1));
                        }
                        
                        if ((filt_buf[p] == ','&&(a_p==0 && s_p == -1 && m_p == 0 && b_p == 0))) {

                            //strncpy(res->ranges[i], filt_buf+p+2, com_pos-p-2);
                            strncpy(res->types[i].type, filt_buf+p+2, com_pos-p-2);
                            res->types[i].type[com_pos-p-2]='\0';
                            //printf("args:%s\n", res->types[i].type);
                            com_pos = p;
                        } else {
                            //strncpy(res->ranges[i], filt_buf+p+1, com_pos-p-1);
                            strncpy(res->types[i].type, filt_buf+p+1, com_pos-p-1);
                            res->types[i].type[com_pos-p-1]='\0';
                            //printf("args:%s\n", res->types[i].type);
                            
                            break;
                        }
                    }
                }
                
            }
            pclose(filt_fp);

        }
        pclose(nm_fp);
    }

    return res;
}

//int cnt=0;
//void func() {
// search for the addr and val that 'minVal <= val < maxVal'
// the return list ends with (nil, 0)

addr_val * search_devptr_vals(mem_maps * maps) {
    unsigned int cnt = 0, size = 1022;
    addr_val* devals = malloc(sizeof(addr_val) * (size+1));
    for(int i=1; i <= maps->num; ++i) {
        void ** pp;
        for(pp = maps->ranges[i].s; pp < (void **)maps->ranges[i].e; ++pp) {
            if(cudawIsDevAddr(*pp)) {
                if(cnt >= size) {
                    size = (size + 1) * 2;
                    devals = realloc(devals, (size+1)*sizeof(addr_val));
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

static addr_range * lookup_text_range(void *p) {
	for (int k = ki_text->num; k > 0; --k) {
        addr_range range = ki_text->ranges[k];
        if (range.s <= p && p < range.e) {
            return ki_text->ranges + k;
        }
	}
	return NULL;
}

static addr_range * lookup_addr_range(void *p) {
	for (int k = ki_maps->num; k > 0; --k) {
        addr_range range = ki_maps->ranges[k];
        if (range.s <= p && p < range.e) {
            return ki_maps->ranges + k;
        }
	}
	return NULL;
}

static int is_alive_heap_ptr(void * ptr) {
    addr_range * hrp = &ki_maps->ranges[ki_maps->heap];
    if ((ptr-16) < hrp->s || ptr >= hrp->e) {
        printf("is_alive_heap_ptr %p is NOT in heap range!\n", ptr);
        return 0;
    }
    QWORD * headp = (QWORD*)(ptr - sizeof(QWORD));
    if ((*headp & 3) == 0) { // P=0 and M=0
        QWORD prev_size = *(headp - 1);
        if (prev_size > 0) {
            QWORD * prev_headp = (ptr - prev_size - sizeof(QWORD));
            if ((void *)prev_headp < hrp->s ||
                    prev_size == (*prev_headp & ~(QWORD)(sizeof(QWORD)-1))) {
                printf("is_alive_heap_ptr %p is NOT a valid chunk!\n", ptr);
                return 0;
            }
        }
    }
    else if ((*headp & 2) == 1) {
        printf("is_alive_heap_ptr %p should NOT be a mmap chunk (head=%llx)!\n", 
                    ptr, *headp);
        return 0;
    }
    QWORD size = (*headp & ~(QWORD)(sizeof(QWORD)-1));
    QWORD * nextp = (QWORD*)(ptr +size - sizeof(QWORD));
    printf("is_alive_heap_ptr %p head=%llx next=%llx\n", ptr, *headp, *nextp);
    if (size < sizeof(QWORD) * 2) {
        printf("is_alive_heap_ptr %p is NOT a valid chunk!\n", ptr);
        return 0;
    }
    return (*nextp & 3);
}

#define OP_CODE(pc, pos) (((BYTE*)(pc))[pos])
#define OP_VAL(T, pc, pos) (*(T*)((BYTE*)(pc)+(pos)))

static int is_ret_of_call_func(void * retpc, void * func, const addr_range * trp) {
    if (retpc - 5 >= trp->s && 0xe8 == OP_CODE(retpc, -5)) {
        int offset = OP_VAL(int, retpc, -4);
        if ((retpc + offset) == func) {
            return -5;
        }
    }
    if (retpc - 6 >= trp->s && 0xff == OP_CODE(retpc, -6)) {
        if (0x15 == OP_CODE(retpc, -5)) {
            int offset = OP_VAL(int, retpc, -4);
            void ** funcp = (void **)(retpc + offset);
            if (*funcp == func) {
                return -6;
            }
        }
    }
    return 0;
}

static int is_op_ret_in_front(void * pc, const addr_range * trp) {
    BYTE op;
    if (pc - 1 >= trp->s) {
        op = OP_CODE(pc, -1);
        if (op == 0xc3 || op == 0xcb) {
            return -1;
        }
    }
    if (pc - 3 >= trp->s) {
        op = OP_CODE(pc, -3);
        if (op == 0xc2 || op == 0xca) {
            return -3;
        }
    }
    return 0;
}

static int is_nop_in_front(void * pc, const addr_range * trp) {
    int nop_in_front = 0;
    if (pc - 1 >= trp->s && 0x90 == OP_CODE(pc, -1)) {
        nop_in_front = -1;
    }
    else if (pc - 2 >= trp->s && 0x9066 == OP_VAL(WORD, pc, -2)) {
        nop_in_front = -2;
    } 
    else if (pc - 3 >= trp->s && 
             0x1f0f == (0xffffff & OP_VAL(DWORD, pc, -3))) {
        nop_in_front = -3;
    }
    else if (pc - 4 >= trp->s && 0x401f0f == OP_VAL(DWORD, pc, -4)) {
        nop_in_front = -4;
    }
    else if (pc - 5 >= trp->s && 0x0f == OP_CODE(pc, -5) && 
             0x441f == OP_VAL(DWORD, pc, -4)) {
        nop_in_front = -5;
    }
    else if (pc - 6 >= trp->s && 
             0x0f66 == OP_VAL(WORD, pc, -6) &&
             0x441f == OP_VAL(DWORD, pc, -4)) {
        nop_in_front = -6;
    }
    else if (pc - 7 >= trp->s && 
             0x801f0f == OP_VAL(DWORD, pc, -7) &&
             0 == OP_VAL(DWORD, pc, -4)) {
        nop_in_front = -7;
    }
    else if (pc - 8 >= trp->s &&
             0x841f0f == OP_VAL(DWORD, pc, -8) &&
             0 == OP_VAL(DWORD, pc, -4)) {
        if (pc - 10 >= trp->s && 0x2e66 == OP_VAL(WORD, pc, -10)) 
            nop_in_front = -10;
        else if (pc - 9 >= trp->s && 0x66 == OP_CODE(pc, -9))
            nop_in_front = -9;
        else
            nop_in_front = -8;
    }
    return nop_in_front;
}

static int jmp_back_after_ret(void * retpc, const addr_range * trp) {
    void * end = retpc + 127;
    if (end + 1 >= (void *)trp->e) {
        end = (void *)trp->e - 1;
    }
    for(void * pc = retpc + 1; pc < end; ++pc) {
        if (OP_CODE(pc, 0) == 0xeb && 
            (pc + OP_VAL(char, pc, 1)) <= retpc) {
            return 1;
        }
    }
    return 0;
}

static int is_func_entry(void * pc, const addr_range * trp) {
    if (pc == trp->s && pc < (void *)trp->e) {
        printf("is_func: entry at the start of addr_range(%p-%p)\n", 
                trp->s, trp->e);
        return 1;
    }
    int ret_in_front = is_op_ret_in_front(pc, trp);
    BYTE op = OP_CODE(pc, 0);
    int entry_is_push_ebp = (0x55 == op);
    int entry_is_push = (0x50 <= op && op <= 0x57);
    if (ret_in_front && entry_is_push_ebp) {
        printf("is_func: entry: %p ret_in_front(%d) and entry_is_push_ebp\n",
                pc, ret_in_front);
        return 1;
    }
    int nop_in_front = is_nop_in_front(pc, trp);
    while (nop_in_front && !ret_in_front) {
        int offset = is_nop_in_front(pc + nop_in_front, trp);
        nop_in_front += offset;
        ret_in_front = is_op_ret_in_front(pc + nop_in_front, trp);
        if (offset == 0) 
            break;
    }
    if (nop_in_front && ret_in_front) {
        printf("is_func: entry: %p nop_in_front(%d) and ret_in_front(%d)\n",
                pc, nop_in_front, ret_in_front);
        return 1;
    }
    if (ret_in_front) {
        if (jmp_back_after_ret(pc + ret_in_front, trp)) {
            printf("is_func: NOT a entry: %p has jmp_back_after_ret(%d)\n",
                pc, ret_in_front);
            return 0;
        }
        printf("is_func: entry: %p no jmp_back_after_ret(%d)\n",
                pc, ret_in_front);
        return 1;
    }
    printf("WARNING: %p is NOT a function pc?\n", pc);
    return 0;
}

static int is_ret_of_call(void * retpc, const addr_range * trp) {
    if (retpc - 5 >= trp->s && 0xe8 == OP_CODE(retpc, -5)) {
        int offset = OP_VAL(int, retpc, -4);
        if (lookup_text_range(retpc + offset) != NULL) {
            return -5;
        }
    }
    if (retpc - 6 >= trp->s && 0xff == OP_CODE(retpc, -6)) {
        if (0x15 == OP_CODE(retpc, -5)) {
            int offset = OP_VAL(int, retpc, -4);
            void ** funcp = (void **)(retpc + offset);
            if (lookup_addr_range(funcp) != NULL) {
                if (lookup_text_range(*funcp) != NULL) {
                    return -6;
                }
            }
        }
    }
    if (retpc - 2 >= trp->s && 0xff == OP_CODE(retpc, -2)) {
        BYTE op = OP_CODE(retpc, -1);
        if (0xd0 == (0xf8 & op)) {
            return -2;
        }
        if (0x10 == (0xf8 & op) && 4 != (6 & op)) { 
            // op [0..2] != 4 and 5
            return -2;
        }
    }
    if (retpc - 3 >= trp->s && 0xff == OP_CODE(retpc, -3)) {
        BYTE op = OP_CODE(retpc, -2);
        if (0x50 == (0xf8 & op) && 0x4 != (0x7 & op)) {
            return -3;
        }
        if (0x14 == op) {
            return -3;
        }
    }
    if (retpc - 4 >= trp->s && 0xff == OP_CODE(retpc, -4)) {
        BYTE op = OP_CODE(retpc, -3);
        if (0x54 == op) {
            return -4;
        }
    }
    if (is_func_entry(retpc, trp)) {
        return 0;
    }
    printf("WARNING: %p is NOT a retpc?\n", retpc);
    return 0;
}

static void * find_call_in_stack(void * func, void ** pp) {
    addr_range * trp = lookup_text_range(func);
    if (trp == NULL) {
        return NULL;
    }
    for (void ** np = pp - 8; np > (void **)ki_bottom; np--) {
        if (trp->s <= *np && *np < trp->e) {
            if (is_ret_of_call(*np, trp))
                return np;
        }
    }
    return NULL;
}

static void * find_stack_of_func(void * funcs[], void * bottom, void * top) {
	void ** func_ret = (void **)bottom;
    for (int k = 0; funcs[k] != NULL; ++k) {
    	while (func_ret < (void **)top) {
	    	addr_range * trp = lookup_text_range(*func_ret);
		    if (trp != NULL) {
                if (is_ret_of_call_func(*func_ret, funcs[k], trp)) {
		            func_ret++;
                    bottom = func_ret;
                    break;
                }
                else if (is_ret_of_call(*func_ret, trp)) {
		            func_ret++;
                    bottom = func_ret;
                    break;
                }
		    }
		    func_ret++;
	    }
    }
    return bottom;
}

static unsigned short guess_argc(kernel_info * kip, void** args) {
    int i = 0, k;
    int n = (kip->argc > 0) ? kip->argc : 256;
	for (i = 0; i < n; i++) {
		addr_range * rp = lookup_addr_range(args[i]);
		if (rp == NULL) {
			printf("guess_argc: args[%d]=%p is not a valid ptr!\n", i, args[i]);
			break;
		}
        if (rp->s <= ki_bottom && ki_bottom < rp->e) {
            if (args[i] < ki_bottom) {
				printf("guess_argc: args[%d]=%p is less than bottom %p!\n", 
						i, args[i], ki_bottom);
                break;
            }
			printf("guess_argc: args[%d]=%p in stack!\n", i, args[i]);
        }
        else {
	        void ** pp;
	        for (pp = (void **)ki_bottom; pp < (void **)ki_top; ++pp) {
            	if ( *pp == args[i] && pp != args+i) {
					break;
				}
            }
            if (pp >= (void **)ki_top) {
				printf("guess_argc: args[%d]=%p value not found in stack!\n", 
						i, args[i]);
                break;
            }
            if (is_alive_heap_ptr(args[i])) {
    			printf("guess_argc: args[%d]=%p in heap!\n", i, args[i]);
            }
            else {
    			printf("guess_argc: args[%d]=%p is NOT alive in heap!\n", i, args[i]);
                break;
            }
        }
    }
    //sleep(10);
    return i;
}

#define MAX_BOUNDRY_DIFF (2048llu - 16)

size_t find_size_in_args(void ** args, unsigned short argc, void * ptr) {
    void * boundry = NULL;
    for(int i=0; i<argc; ++i) {
        void * val = args[i];
        if (ptr < val) {
            if (boundry == NULL) {
                boundry = val;
            }
            else if (val < boundry) {
                boundry = val;
            }
        }
    }
    if (boundry != NULL) {
        return (boundry - ptr);
    }
    return -1;
}

static int has_sibling_args(void ** args, unsigned short argc, void * ptr) {
    for (int i = 0; i < argc; i++) {
        if (args[i] + 4 == ptr || args[i] + 8 == ptr) {
            return 1;
        }
    }
    return 0;
}

int valid_func(mem_maps * texts,void** pp,void * boundry) {
    int i=texts->num;
    for (; i > 0; --i) {
        if (*pp >= texts->ranges[i].s &&
                *pp < texts->ranges[i].e) {
            if (*(pp-1) < boundry) {
                printf("stack: %p %p - func %d\n", pp, *pp, i);
                return i;
            }
            break;
        }
    }
    return 0;
}

int find_func_chain(mem_maps * texts, void** pp, void * boundry, int cnt) {
    if(cnt<=0) {
        return 1;
    }
    int maps=valid_func(texts,pp,boundry);
    if (maps!=0&&(void **)*(pp-1) > pp) {
        return find_func_chain(texts,(void **)*(pp-1)+1,boundry,cnt-1);
    }
    return 0;
}

size_t find_size_by_loop(void * ptr, size_t size) {
    assert(ki_bottom <= ptr && ptr < ki_top);
    assert(((unsigned long long)ptr & 7) == 0);
    int loop_cnt = 0;
    int min_len = 5;
    void ** stop = ptr + size;
    void ** end = ptr + size + 1024;
    void ** loop_start = NULL;
    void ** loop_end = NULL;
    void ** loop_pos = NULL;
    void ** loop_mark = NULL;
    void ** current = NULL;
    if (end > (void **)ki_top) {
        end = ki_top;
    }
    for (loop_start = ptr; loop_start < stop; loop_start++) {
        for (loop_mark = loop_start + min_len; loop_mark < end; loop_mark++) {
            if (*loop_start == *loop_mark) {
                loop_end = loop_mark;
                loop_pos = loop_start + 1;
                loop_cnt = 0;
                for (current = loop_mark + 1; current < end; ++current) {
                    if (*current != *loop_pos) {
                        break;
                    }
                    loop_pos++;
                    if (loop_pos == loop_end) {
                        loop_cnt++;
                        if (loop_cnt >= 1) {
                            for (void ** pp = loop_start; pp < loop_end; ++pp) {
                                if (cudawIsDevAddr(*pp)) {
                                    printf("find_boundry_by_loop for %p ", ptr);
                                    if (loop_start == ptr) {
                                        printf("at start %p and end %p\n", 
                                                loop_start, loop_end);
                                      return (void *)loop_end - ptr;
                                    }
                                    printf("at start %p\n", loop_start);
                                    return (void *)loop_start - ptr;
                                }
                            }
                        }
                        loop_pos = loop_start + 1;
                        loop_mark += (loop_end - loop_start);
                    }
                }
            }
        }
    }
    return size;
}

size_t find_size_by_call_stack(void * ptr, size_t size) {
    void ** top = ki_top;
    if (ki_top - ptr > size) {
        top = ptr + size;
    }
    void ** pp = ptr;
    for (; pp < top; ++pp) {
        addr_range * tp = lookup_text_range(*pp);
        if (tp == NULL)
            continue;
        int ret = is_ret_of_call(*pp, tp);
        void * func = NULL;
        addr_range * ftp = NULL;
        switch (ret) {
            case 0:
                printf("find_boundry met %p:%p as a func ptr for %p\n", 
                            pp, *pp, ptr);
                continue;
            case -6:
                func = *(void **)(*pp + OP_VAL(int, *pp, -4));
            case -5:
                if (func == NULL)
                    func = *pp + OP_VAL(int, *pp, -4);
                if (find_call_in_stack(func, ptr) == NULL) {
                    printf("find_boundry met %p:%p a expired retpc for %p\n", 
                                pp, *pp, ptr);
                    continue;
                }
            default:
                printf("find_boundry for %p at %p:%p with ret=%d\n", 
                                ptr, pp, *pp, ret);
                return (void *)pp - ptr;
        }
    }
    printf("find_boundry met no func ptr for %p\n", ptr);
    return size;
}

size_t find_size_in_heap(void * ptr) {
    QWORD head = *(QWORD*)(ptr - 8);
    QWORD next = *(QWORD*)(ptr - 8 + (head & ~0x7llu));
    assert((next & 3) == 1);
    head = head >> 3 << 3;
    printf("find_boundry for %p with head %lld\n", ptr, head);
    return head;
}

enum {
    USE_PARGS, USE_ARGS, BYPASS_FUNC
};


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


__attribute ((constructor)) void cudaw_targs_init(void) {
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
    // load lib maps
    load_lib_maps(NULL);
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
        kernel_info * kip = &kernel_infos[k];
        printf("offset: %x lib: %u cnt: %u\n", kip->offset, kip->lib, kip->cnt);
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
            unsigned char lib = 0;
            int j, maxj = sizeof(kernel_libs) / sizeof(kernel_lib);
            for (j = 0; j < maxj; ++j) {
                if (kernel_libs[j].start <= func && 
                        func < kernel_libs[j].end) {
                    lib = (unsigned char)j;
                    break;
                }
            }
            if (j == maxj) {
                load_lib_maps(func);
                lib = j;
            }
            int n = sizeof(kernel_infos) / sizeof(struct kernel_info);
            unsigned int offset = (unsigned int)(func - kernel_libs[j].start);
            for (k = 1; kernel_infos[k].argc ; ++k) { // 0 for empty func
                if (offset == kernel_infos[k].offset && 
                        lib == kernel_infos[k].lib) {
                    kernel_infos[k].func = func;
                    ki_lookup_table[i] = k;
                    sem_post(&ki_sem);
                    return kernel_infos + k;
                }
            }
            assert(kernel_infos[k+1].argc == 0);
            struct kernel_info * new_func = &kernel_infos[k];
            new_func->func = func;
            new_func->lib = lib;
            new_func->offset = offset;
            new_func->status = KIS_NEW_FUNC;
            printf("new-func: %p offset: 0x%x lib: %u\n", func, offset, lib);
            return new_func;
        }
    }
    assert(0);
}

// Translate pointer args[i]
static void trans_args_addv(kernel_info * kip, void ** args, void ** pargs) {
    void * base = mblk_alloc(kip->size, NULL);
    int copied = 0, i, s;
    for (int k = 0; k < kip->objc; ++k) {
        if (kip->objv[k] < 0x4000) {
            i = kip->objv[k];
            s = 8;
        }
        else {
            i = kip->objv[k] & 0xf;
            s = kip->objv[k] & 0xff0;
        }
        pargs[i] = base + copied;
        memcpy(pargs[i], args[i], s);
        copied += s;
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
static void trans_cp_args(kernel_info * kip, void ** args, void ** pargs) {
    for (int i = 0; i < kip->argc; ++i) {
        pargs[i] = args[i];
    }
}

static void ki_print_func(kernel_info * kip) {
    if (kip->cnt <= KI_PRINT_CNT_MAX || (kip->cnt & KI_PRINT_CNT_MASK) == 1) {
        printf("offset: 0x%x lib: %u argc: %d status: %d cnt: %u\n",
                kip->offset, kip->lib, kip->argc, kip->status, kip->cnt);
    }
}

static void guess_type1_args(kernel_info * kip, void ** args, void ** pargs) {
    void * lop = NULL;
    void * hip = NULL;
    int devptr_cnt = 0;
    int addc = 0, objc = 0, argc = kip->argc;
    for (int i = 0; i < kip->argc; ++i) {
        if (hip == NULL) {
            hip = lop = args[i];
        } 
        else if (hip < args[i]) {
            hip = args[i];
        }
        else if (lop > args[i]) {
            lop = args[i];
        }
        if ((hip - lop) != i * sizeof(void*)) {
            argc = i;
            break;
        }
        void * devptr = *(void **)args[i];
        if (cudawIsDevAddr(devptr)) {
            devptr_cnt++;
            if (devptr_cnt < sizeof(kip->objv)/sizeof(short)) {
                kip->objv[objc++] = i;
                kip->addv[addc++] = (addc-1);
            }
        }
    }
    if (argc >= 5 && devptr_cnt >= 2 && argc >= kip->argc - 2) {
        kip->argc = argc;
        kip->objc = objc;
        kip->addc = addc;
        kip->size = sizeof(void *) * addc;
        // kip->status = 0; // TODO
        printf("guess_type1_args for (0x%x, %u)\n", kip->offset, kip->lib);
    }
}

static void guess_type2_args(kernel_info * kip, void ** args, void ** pargs) {
    void * lop = NULL;
    void * hip = NULL;
    void * devhip = NULL;
    void * devlop = NULL;
    int devptr_cnt = 0;
    int addc = 0, objc = 0, argc = kip->argc;
    for (int i = 0; i < kip->argc; ++i) {
        void * devptr = *(void **)args[i];
        if (cudawIsDevAddr(devptr)) {
            if (devhip == NULL) {
                devhip = args[i];
                devlop = args[i];
            }
            else if (devhip < args[i]) {
                devhip = args[i];
            }
            if (devptr_cnt > 0 && // must put before devptr_cnt++
                (devhip - devlop) != devptr_cnt * sizeof(void*)) {
                argc = i;
                break;
            }
            devptr_cnt++;
            if (devptr_cnt < sizeof(kip->objv)/sizeof(short)) {
                kip->objv[objc++] = i;
                kip->addv[addc++] = (addc-1);
            }
            continue;
        }
        if (hip == NULL) {
            hip = args[i];
            lop = args[i];
        }
        else if (hip < args[i]) {
            hip = args[i];
        }
        else if (lop > args[i]) {
            lop = args[i];
        }
        if ((i - devptr_cnt) > 0 &&
            (hip - lop) != (i - devptr_cnt) * 4 ) {
            argc = i;
            break;
        }
    }
    if (argc >= 5 && devptr_cnt >= 2 && argc >= kip->argc - 2) {
        kip->argc = argc;
        kip->objc = objc;
        kip->addc = addc;
        kip->size = sizeof(void *) * addc;
        // kip->status = 0; // TODO
        printf("guess_type2_args for (0x%x, %u)\n", kip->offset, kip->lib);
    }
}

static void guess_type3_args(kernel_info * kip, void ** args, void ** pargs) {
    void * lop = args[0];
    void * hip = args[0];
    void * phip = args[0];
    void * plop = args[0];
    int cnt = -1, pcnt = -1;
    int devptr_cnt = 0;
    int addc = 0, objc = 0, argc = kip->argc;
    for (int i = 0; i < kip->argc; ++i) {
        void * devptr = *(void **)args[i];
        if (cudawIsDevAddr(devptr)) {
            devptr_cnt++;
            if (devptr_cnt < sizeof(kip->objv)/sizeof(short)) {
                kip->objv[objc++] = i;
                kip->addv[addc++] = (addc-1);
            }
        }
        if (i == 0) {
            continue;
        }
        if (args[i] == lop - 4 || hip + 4 == args[i]) {
            if (phip == plop && plop == lop && lop == hip) {
                phip = plop = NULL;
                cnt = 0;
            }
            if (args[i] < lop) {
                lop = args[i];
            }
            else if (args[i] > hip) {
                hip = args[i];
            }
            cnt++;
        }
        else if (args[i] == plop - 8 || phip + 8 == args[i]) {
            if (phip == plop && plop == lop && lop == hip) {
                hip = lop = NULL;
                pcnt = 0;
            }
            if (args[i] < plop) {
                plop = args[i];
            }
            else if (args[i] > phip) {
                phip = args[i];
            }
            pcnt++;
        }
        else if (hip == NULL) {
            hip = args[i];
            lop = args[i];
            cnt = 0;
        }
        else if (phip == NULL) {
            phip = args[i];
            plop = args[i];
            pcnt = 0;
        }
        if (cnt + pcnt + 1 != i) {
            argc = i;
            break;
        }
        if (cnt > 0 && (hip - lop) != cnt * 4) {
            argc = i;
            break;
        }
        if (pcnt > 0 && (phip - plop) != pcnt * 8) {
            argc = i;
            break;
        }
    }
    if (argc >= 5 && devptr_cnt >= 2 && argc >= kip->argc - 2) {
        kip->argc = argc;
        kip->objc = objc;
        kip->addc = addc;
        kip->size = sizeof(void *) * addc;
        // kip->status = 0; // TODO
        printf("guess_type3_args for (0x%x, %u)\n", kip->offset, kip->lib);
    }
}

static void guess_type4_args(kernel_info * kip, void ** args, void ** pargs) {
    void * lo1p = NULL;
    void * hi1p = NULL;
    void * hi2p = NULL;
    void * lo2p = NULL;
    int cnt1 = -1, cnt2 = -1;
    int devptr_cnt = 0;
    int addc = 0, objc = 0, argc = kip->argc;
    for (int i = 0; i < kip->argc; ++i) {
        if (hi1p == NULL) {
            lo1p = hi1p = args[i];
            cnt1 = 0;
        }
        else if (lo1p - 8 <= args[i] && args[i] <= hi1p + 8) {
            if (args[i] < lo1p) {
                lo1p = args[i];
            }
            else if (args[i] > hi1p) {
                hi1p = args[i];
            }
            cnt1++;
        }
        else if (hi2p == NULL) {
           hi2p = lo2p = args[i];
           cnt2 = 0;
        }
        else if (lo2p - 8 <= args[i] && args[i] <= hi2p + 8) {
            if (args[i] < lo2p) {
                lo2p = args[i];
            }
            else if (args[i] > hi1p) {
                hi2p = args[i];
            }
            cnt2++;
        }
        else {
            argc = i;
            break;
        }
        void * devptr = *(void **)args[i];
        if (cudawIsDevAddr(devptr)) {
            devptr_cnt++;
            if (devptr_cnt < sizeof(kip->objv)/sizeof(short)) {
                kip->objv[objc++] = i;
                kip->addv[addc++] = (addc-1);
            }
        }
    }
    if (devptr_cnt >= 2 && argc == kip->argc && 
            (cnt2 == -1 || cnt2 >= 2) && cnt1 >= 2) {
        kip->argc = argc;
        kip->objc = objc;
        kip->addc = addc;
        kip->size = sizeof(void *) * addc;
        // kip->status = 0; // TODO
        printf("guess_type4_args for (0x%x, %u)\n", kip->offset, kip->lib);
    }
}

static void guess_type5_args(kernel_info * kip, void ** args, void ** pargs) {
    void * lop = args[0];
    void * hip = args[0];
    int devptr_cnt = 0;
    int addc = 0, objc = 0, argc = kip->argc;
    for (int i = 0; i < kip->argc; ++i) {
        if (hip < args[i]) {
            hip = args[i];
        }
        else if (lop > args[i]) {
            lop = args[i];
        }
        if ((hip - lop) > 512) {
            argc = i;
            break;
        }
        void * devptr = *(void **)args[i];
        if (cudawIsDevAddr(devptr)) {
            devptr_cnt++;
            if (devptr_cnt < sizeof(kip->objv)/sizeof(short)) {
                kip->objv[objc++] = i;
                kip->addv[addc++] = (addc-1);
            }
        }
    }
    if (argc >= 10 && devptr_cnt >= 2 && argc >= kip->argc - 2) {
        kip->argc = argc;
        kip->objc = objc;
        kip->addc = addc;
        kip->size = sizeof(void *) * addc;
        // kip->status = 0; // TODO
        printf("guess_type5_args for (0x%x, %u)\n", kip->offset, kip->lib);
    }
}

static void assume_all_type_args(kernel_info * kip, void ** args, void ** pargs) {
    int devptr_cnt = 0;
    int addc = 0, objc = 0, argc = kip->argc;
    for (int i = 0; i < kip->argc; ++i) {
        void * devptr = *(void **)args[i];
        if (cudawIsDevAddr(devptr)) {
            devptr_cnt++;
            if (devptr_cnt < sizeof(kip->objv)/sizeof(short)) {
                kip->objv[objc++] = i;
                kip->addv[addc++] = i;
            }
        }
    }
    kip->objc = objc;
    kip->addc = addc;
    kip->size = sizeof(void *) * addc;
    // kip->status = 0; // TODO
    printf("assume_all_type_args for (0x%x, %u)\n", kip->offset, kip->lib);
}

static void mark_args_ptr(kernel_info * kip, void ** args, void ** pargs) {
    // marks the stack position
    kip->cnt = (unsigned long long)pargs;
    for (int i = 0; i < kip->argc; ++i) {
        // marks the ptr of args[i]
        kip->objv[i] = (unsigned long long)args[i];
    }
    kip->objc = kip->argc;
}

static void verify_args_ptr(kernel_info * kip, void ** args, void ** pargs) {
    kip->objc = kip->argc;
    // when meats the same stack position
    unsigned int cnt = (unsigned long long)pargs;
    //printf("verify_args_ptr: cnt: %x - %x\n", kip->cnt, cnt);
    if (kip->cnt == cnt) {
        for (int i = 0; i < kip->argc; ++i) {
            // verify the ptr of args[i] is unchanged
            unsigned short val = (unsigned long long)args[i];
            //printf("verify_args_ptr: args[%d]: %x - %x\n", i, 
            //            kip->objv[i], val);
            if (kip->objv[i] != val) {
                // changed ptr must be alive heep ptr
                if (!is_alive_heap_ptr(args[i])) {
                    kip->argc = i;
                    break;
                }
            }
        }
        if (kip->argc < kip->objc) {
            printf("BETTER: argc: %d -> %d same stack!\n", 
                        kip->objc, kip->argc);
            kip->objc = kip->argc;
        }
    }
    else {
        unsigned short objv0 = kip->objv[0];
        unsigned short args0 = (unsigned long long)args[0];
        for (int i = 1; i < kip->argc; ++i) {
            // verify the offset of args[i] is unchanged
            unsigned short argsi = (unsigned long long)args[i];
            if (kip->objv[i] - objv0 != argsi - args0) {
                // changed ptr must be alive heep ptr
                if (!is_alive_heap_ptr(args[i])) {
                    kip->argc = i;
                    break;
                }
            }
        }
        if (kip->argc < kip->objc) {
            printf("BETTER: argc: %d -> %d diff stack!\n", 
                        kip->objc, kip->argc);
            kip->objc = kip->argc;
        }
    }
}

static int trans_args(kernel_info * kip, void ** args, void ** pargs, char * buf) {
    if (kip->status & KIS_NEW_FUNC) {
        if (kip->objc > 0 || kip->addc > 0)
            return USE_ARGS;
/*
        if (kip->cnt > 20)
            return USE_ARGS;
        static const void * old_func = NULL;
        static void ** old_pargs = NULL;
        if (old_pargs == pargs && old_fun == func)
            return USE_ARGS;
        old_pargs = pargs;
        old_func = func;
*/
        // Load maps for guassing args
        mem_maps * maps = load_mem_maps();
        mem_maps * text = load_text_maps();
        addr_val * vals = search_devptr_vals(maps);
        printf("maps: num: %d heap: %d stack: %d thread: %d\n",
                    maps->num, maps->heap, maps->stack, maps->thread);
        fflush(stdout);
    
        // Find the bottom of stack in cudaLaunchKernel
		void * bottom = (void*)pargs;
        addr_range * rp = lookup_addr_range(bottom);
        void * funcs[] = {
            cudawLaunchKernel, 
            cudaLaunchKernel,
            NULL,
        };
        ki_bottom = bottom = find_stack_of_func(funcs, bottom, rp->e);
        ki_top = rp->e;
        printf("current_thread_stack: bottom: %p top: %p", ki_bottom, ki_top);

        if (kip->argc == 0) {
            kip->argc = guess_argc(kip, args);
            if (kip->argc >= 5) {
                guess_type1_args(kip, args, pargs);
            }
            if (kip->addc == 0 && kip->argc >= 5) {
                guess_type2_args(kip, args, pargs);
            }
            if (kip->addc == 0 && kip->argc >= 5) {
                guess_type3_args(kip, args, pargs);
            }
            if (kip->addc == 0 && kip->argc >= 5) {
                guess_type4_args(kip, args, pargs);
            }
            if (kip->addc == 0 && kip->argc > 16) {
                assume_all_type_args(kip, args, pargs);
            }
            if (kip->addc == 0 && kip->argc >= 5) {
                //mark_args_ptr(kip, args, pargs);
            }
        }
        /*
        else if (kip->argc == kip->objc && kip->addc == 0) {
            unsigned short argc = guess_argc(kip, args);
            if (argc < kip->argc) {
                printf("BETTER: argc: %d -> %d\n", kip->argc, argc);
                kip->argc = argc;
            }
            verify_args_ptr(kip, args, pargs);
            if (argc == kip->argc) {
                kip->size++;
                if (kip->size > 4) {
                    kip->objc = 0;
                    kip->size = 0;
                }
            }
        }
        */
        printf("(0x%x, %u) == argc: %d (args=%p)\n", 
                    kip->offset, kip->lib, kip->argc, args);
        if ((kip->addc == 0) && (kip->objc == 0)) 
        for (int i = 0; i < kip->argc; ++i) {
            fflush(stdout);
            if (((unsigned long long )args[i] & 0x7) != 0) {
                printf("argi: %d of %p not aligned\n", i, args[i]);
                continue;
            }
            size_t argi_size = find_size_in_args(args, kip->argc, args[i]);
            if (argi_size < 8) {
                printf("argi: %d of %p size %ld\n", i, args[i], argi_size);
                continue;
            }
            else if (argi_size == 8 || 
                has_sibling_args(args, kip->argc, args[i])) {
                void * devptr = *(void **)args[i];
                if (cudawIsDevAddr(devptr)) {
                    int c = count_value(vals, devptr);
                    kip->objv[kip->objc++] = 0x4010 + i;
                    kip->addv[kip->addc++] = kip->size / sizeof(void*);
                    kip->size += 16;
                    printf("(0x%x, %u) == argi: %d(%p) count_value: %d\n",
                                kip->offset, kip->lib, i, args[i], c);
                }
                else {
                    printf("argi: %d(%p) not devptr %p\n", i, args[i], devptr);
                }
                continue;
            }
            if (ki_bottom <= args[i] && args[i] < ki_top) {
                argi_size = find_size_by_call_stack(args[i], argi_size);
                argi_size = find_size_by_loop(args[i], argi_size);
            }
            else {
                argi_size = find_size_in_heap(args[i]);  
            }
            void ** pv = (void **)args[i];
            size_t n = argi_size / sizeof(void*);
            unsigned short old_addc = kip->addc;
            for (int k = 0; k < n; ++k) {
                printf("(0x%x, %u) == argi: %d - %3d: %p\n", 
                            kip->offset, kip->lib, i, k, pv[k]);
                if (cudawIsDevAddr(pv[k])) {
                    int c = count_value(vals, pv[k]);
                    printf("(0x%x, %u) == argi: %d(%p)[%d] count_value: %d\n",
                                kip->offset, kip->lib, i, args[i], k, c);
                    if (c > 2 //&& (kip->addc - old_addc) < 4 &&
                            //kip->addc + 1 < sizeof(kip->addv) / sizeof(short)
                            ) {
                        kip->addv[kip->addc] = kip->size / sizeof(void*) + k;
                        kip->addc++;
                    }
                }
            }
            if (kip->addc > old_addc) {
                unsigned short size = (argi_size + 15) & 0xfff0;
                kip->size += size;
                kip->objv[kip->objc++] = 0x4000 + i + size;
            }
            printf("(0x%x, %u) == argi: %d(%p) size: %lu\n",
                        kip->offset, kip->lib, i, args[i], n * sizeof(void*));
        }
        fflush(stdout);
        static int cs = 0;
        if (cs++ < 10) {
            kip->status = 0;
            printf("{NULL, 0x%x, %u ...\n", kip->offset, kip->lib);
        }
        sem_post(&ki_sem);
        if (kip->argc > 0) {
            printf("{NULL, 0x%x, %u, 0, %2u, %4u, %u, %u, {", kip->offset, kip->lib,
                            kip->argc, kip->size, kip->objc, kip->addc);
            for (int k = 0; k < kip->objc; ++k) {
                unsigned short v = kip->objv[k];
                unsigned short t = v & 0xf00fu;
                unsigned short s = v & 0x0ff0u;
                if (k > 0) 
                    printf(",");
                if (v >= 0x4000) 
                    printf("0x%x+%d", t, s);
                else
                    printf("%d", v);
            }
            printf("}, {");
            for (int k = 0; k < kip->addc; ++k) {
                if (k > 0) 
                    printf(",");
                printf("%u", kip->addv[k]);
            }
            printf("}, 0},\n");
        }
        fflush(stdout);
        free(vals);
        free(maps);
        free(text);
        if (kip->status != 0) {
            return USE_ARGS;
        }
    }

    ++kip->cnt;
    ki_print_func(kip);

    int use = USE_PARGS;
    switch (kip->status & KIS_ACTION_MASK) {
        case KIS_PARGS_ARGC:
            trans_cp_args(kip, args, pargs);
            break;
        case KIS_PARGS_CPALL:
            //trans_args_cpall(kip, args, pargs);
            break;
        case 0:
            trans_cp_args(kip, args, pargs);
            trans_args_addv(kip, args, pargs);
            break;
    }
    struct func_args* tmp = get_func_args(kip);
    for(int i=1;i<=tmp->num;++i) {
        printf("args:%s\n", tmp->types[i].type);
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

    kernel_info * kip = ki_lookup(func);
    void * pargs[KI_MAX_ARGC] = {0};
    char buf[kip->size];
    int use = trans_args(kip, args, pargs, buf);
    switch (use) {
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
