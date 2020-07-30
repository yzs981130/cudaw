// DEFSO & LDSYM

static so_func_info_t so_funcs[MAX_FUNC] = {0};
static so_dl_info_t   so_dli = {LIB_STRING, NULL, 0, 0, so_funcs};
static void          *so_handle = NULL;

static void so_update_func(int idx, void * so_func, void * func, const char * func_name) {
    if (so_func == NULL) {
        char *errstr = dlerror();
        fprintf(stderr, "FAIL: dlsym(%s) return(%s)\n", func_name, errstr);
        exit(1);
    }
    Dl_info dli;
    if (dladdr(so_func, &dli) != 0) {
        assert(strcmp(so_dli.dli_fname, dli.dli_fname) == 0);
        if (so_dli.dli_fbase == NULL) {
            so_dli.dli_fbase = dli.dli_fbase;
        }
        assert(strcmp(func_name, dli.dli_sname) == 0);
        so_funcs[idx].func_name = func_name;
        so_funcs[idx].func = func;
    }
    else {
        fprintf(stderr, "FAIL: dladdr(%s) return 0\n", func_name);
        exit(1);
    }
}

#define LDSYM(func)  do { \
    idx_##func = ++so_dli.func_num; \
    so_##func = dlsym(so_handle, #func); \
    so_update_func(idx_##func, so_##func, func, #func); \
} while(0)

