#ifndef __CUDAW_H__
#define __CUDAW_H__

#define BLK_SIZE (32 * 1024 * 1024lu)

union so_func_flags_t {
    uint64_t    known;
    struct {
        int8_t  notrace;
        int8_t  sync;
        int8_t  async;
        int8_t  wrlock;
        int8_t  event;
        int8_t  checkpoint;
    };
};

typedef union so_func_flags_t so_func_flags_t;

struct so_func_info_t {
    const char      *func_name;
    void            *func;
    uint64_t         cnt;
    so_func_flags_t  flags;
};

typedef struct so_func_info_t so_func_info_t;

struct so_dl_info_t {
    const char     *dli_fname;
    void           *dli_fbase;
    int             dli_idx;
    int             func_num;
    so_func_info_t *funcs;
};

typedef struct so_dl_info_t so_dl_info_t;

extern void cudaw_so_begin_func(so_dl_info_t *dlip, int idx);
extern void cudaw_so_end_func(so_dl_info_t *dlip, int idx);
extern void cudaw_so_register_dli(so_dl_info_t *dlip);
extern void cudawrt_so_func_copy(void *funcs[]);
extern void cudawrt_so_func_swap(void *pfuncs[]);
extern void cudawblas_so_func_swap(void *pfuncs[]);
extern void cudawrt_blkcpy_func_swap(void *pfuncs[]);

#define begin_func(func) cudaw_so_begin_func(&so_dli, idx_##func)
#define end_func(func)   cudaw_so_end_func(&so_dli, idx_##func)

#define errmsg(format...) do { \
    const char* msg = strerror(errno); \
    fprintf(stderr, "FAIL: in %s of %s at line %d\n", __func__, __FILE__, __LINE__); \
    fprintf(stderr, "\t(errno=%d): %s\n\t", errno, msg); \
    fprintf(stderr, format); \
} while (0)

#define dlerrmsg(format...) do { \
    const char* msg = dlerror(); \
    fprintf(stderr, "FAIL: in %s of %s at line %d\n", __func__, __FILE__, __LINE__); \
    fprintf(stderr, "\tdlerror: %s\n\t", msg); \
    fprintf(stderr, format); \
} while (0)

#endif // __CUDAW_H__
