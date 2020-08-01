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
#include <ctype.h>

#include "cudaw.h"

struct addr_range {
    void * s;
    void * e;
};

struct mem_maps {
    union {
        struct {
            uint32_t num;        // the total number of addr range with rw
            uint32_t heap;       // the index of the heap range
            uint32_t stack;      // the start index of the first stack range
            uint32_t thread;     // the stack of the current thread
        };
        struct addr_range ranges[1]; // NOTE: the first range starts from 1
    };
};

typedef struct mem_maps mem_maps;
typedef struct addr_range addr_range;

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

static mem_maps * load_mem_maps() {
    FILE* fp = open_proc_maps();
    if (NULL == fp) {
        return NULL;
    }
    const int BUF_SIZE = 2048;
    int cnt=0, i;
    char buf[BUF_SIZE];
    uint32_t size = 254;
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
    return maps;
}

int realloc_trans_addr(void * oldptr, void * newptr, size_t size) {
    mem_maps * maps = load_mem_maps();
    if (maps == NULL) {
        return 1;
    }
    void * oldend = oldptr + size;
    for(int i=1; i <= maps->num; ++i) {
        if (maps->thread == i) {
            continue;
        }
        void ** ps = maps->ranges[i].s;
        void ** pe = maps->ranges[i].e;
        for (void ** pp = ps; pp < pe; ++pp) {
            if (oldptr <= *pp && *pp < oldend) {
                *pp = *pp - oldptr + newptr;
            }
        }
    }
    return 0;
}
