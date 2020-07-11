#ifndef __CUDA_MASTER_H__
#define __CUDA_MASTER_H__

#define BLK_SIZE    0x2000000llu        // 32MB
#define TIP_SIZE    0x100000llu         // 1MB
#define TIP_TAIL    (TIP_SIZE<<1)       // TIP_SIZE * 2
#define TPG_SIZE    0x1000llu           // 4KB
#define TPG_HALF    (TPG_SIZE>>1)

#define MAX_TIPS    256
#define MIN_TIPS    32


#define GB              0x40000000llu
#define MB              0x100000llu
#define PG              0x1000lu

#define UNIT_SIZE       (256 * MB)

#define gb(x)           (((int)((x)>>30)))
#define mb(x)           (((int)((x)>>20)))
//#define gb(x)           (((int)((x)/GB)+1)&~1)
//#define mb(x)           (((int)((x)/MB)+1)&~1)
#define pg(x)           ((int)((x)/PG))
#define blk(x)          ((int)((x)/BLK_SIZE))

#define REQWAIT         2000    // 2s
#define TIMEOUT         10000   // 10s
#define WAITMS          100     // 100ms
#define MS              1000    // 1000us

// -- the master sets the free memory to the target MB value
// ++ a process sets the free memory to the target MB value

#define WORKING         128 // -- working with proc for a while
#define IDLE            192 // -- waiting for request
#define NOTIFY          98  // ++ nofity for return memory

#define PROCESS         64  // ++ free < 64 when there is a cuda process.

// +++++++++

#define BZ              mb(BLK_SIZE)
#define _B              0

#define POLLREQ         (_B+30)  // -- poll for memory request
#define REQUEST         (_B+26)  // ++ request memory inexclusive

// +++++++++

#define OOM             (2*BZ+(OK-2)%BZ)   // -- out of memory / no more memory
#define OK1             (BZ+(OK%BZ))     // -- OK for process to return memory 

// +++++++++

#define OK              (_B+22)  // -- accept the request, go on...
#define DENY            (_B+6)  // -- deny the request, end.

#define RETURN          (_B+18)  // ++ return memory -- out of memory
#define RETGO           (_B+14)  // -- let's go

#define ENDING          (_B+10)  // ++ ending the session
#define ENDED           (_B+6)  // -- the session end.

// +++++++++

#define RESET           4   // -- force all process return memory

#endif // __CUDA_MASTER_H__
