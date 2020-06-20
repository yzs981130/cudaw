#ifndef __CUDAWVADDR_H__
#define __CUDAWVADDR_H__

// API functions 

extern cudaError_t cudawMalloc(void ** devPtr, size_t bytesize);
extern cudaError_t cudawFree(void * devPtr);

#ifndef VA_DISABLE_VIR_ADDR
#define VA_ENABLE_VIR_ADDR
#endif

#ifdef VA_ENABLE_VIR_ADDR
extern void * cudawVirAddrToDev(void * virAddr);
extern void * cudawDevAddrToVir(void * devAddr);
extern int    cudawIsVirAddr(void * virAddr);
#endif


#ifdef VA_TEST_DEV_ADDR
extern int     cudawIsDevAddr(const void * devAddr);
#endif

#ifdef VA_VTOR_PRINT
  #define cudaw_printf(s, a, b) printf(s, a, b)
#else
  #define cudaw_printf(s, a, b) do { } while(0)
#endif

#ifdef VA_ENABLE_VIR_ADDR

#define VtoR(r, v) do { \
	    cudaw_printf("%s [%p] ", #v, v); \
	    r = (typeof(r))cudawVirAddrToDev((void *)(v)); \
	    cudaw_printf("%s (%p)\n", #r, r); \
} while (0)

#define RtoV(v, r) do { \
	    cudaw_printf("%s (%p) ", #r, r); \
	    v = (typeof(v))cudawDevAddrToVir((void *)(r)); \
	    cudaw_printf("%s [%p]\n", #v, v); \
} while (0)

#else 
#define VtoR(r, v) do { } while (0)
#define RtoV(r, v) do { } while (0)
#endif // VA_ENABLE_VIR_ADDR


#define VtoR1(x) VtoR(x, x)
#define VtoR2(x, y) do { VtoR(x, x); VtoR(y, y); } while (0)
#define VtoR3(x, y, z) do { VtoR(x, x); VtoR(y, y); VtoR(z, z); } while (0)
#define VtoR4(x, y, z, a) do { VtoR3(x, y, z); VtoR(a, a); } while (0)
#define VtoR5(x, y, z, a, b) do { VtoR3(x, y, z); VtoR2(a, b); } while (0)
#define VtoR6(x, y, z, a, b, c) do { VtoR3(x, y, z); VtoR3(a, b, c); } while (0)

#define RtoV1(x) RotV(x, x)


#endif // __CUDAWVADDR_H__