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

extern void cudaw_so_begin_func(so_dl_info_t *dlip, int idx);
extern void cudaw_so_end_func(so_dl_info_t *dlip, int idx);
extern void cudaw_rigister_dli(so_dl_info_t *dlip);

#define begin_func(func) cudaw_so_begin_func(&so_dli, idx_##func)
#define end_func(func)   cudaw_so_end_func(&so_dli, idx_##func)

#endif // __CUDAW_H__
