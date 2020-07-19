#ifndef __CUDAW_H__
#define __CUDAW_H__

struct so_func_info_t {
    const char *func_name;
    void       *func_addr;
    int         cnt;
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

#endif // __CUDAW_H__
