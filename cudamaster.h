#ifndef __CUDA_MASTER_H__
#define __CUDA_MASTER_H__

#define BLK_SIZE    0x2000000llu        // 32MB
#define TIP_SIZE    0x100000llu         // 1MB
#define TIP_TAIL    (TIP_SIZE<<1)       // TIP_SIZE * 2
#define TPG_SIZE    0x1000llu           // 4KB
#define TPG_HALF    (TPG_SIZE>>1)

#define MAX_TIPS    256
#define MIN_TIPS    32


#define MB              0x100000llu
#define PG              0x1000lu

#define UNIT_SIZE       (256 * MB)

// -- the master sets the free memory to the target MB value
// ++ a process sets the free memory to the target MB value

#define IDLE            192 // -- waiting for request
#define NOTIFY          98  // ++ nofity for return memory

#define PROCESS         64  // ++ free < 64 when there is a cuda process.

// +++++++++

#define POLLREQ         28  // -- poll for memory request
#define REQUEST         26  // ++ request memory inexclusive

// +++++++++

#define OOM             (64+OK-2)   // -- out of memory / no more memory
#define OK1             (32+OK)     // -- OK for process to return memory 

// +++++++++

#define OK              24  // -- accept the request, go on...
#define DENY            16  // -- deny the request, end.

#define RETURN          22  // ++ return memory -- out of memory
#define RETGO           20  // -- let's go

#define ENDING          18  // ++ ending the session
#define ENDED           16  // -- the session end.

#define REQWAIT         2000    // 2s
#define TIMEOUT         10000   // 10s
#define WAITMS          100     // 100ms
#define MS              1000    // 1000us

// +++++++++

#define FORCERET        4       // -- force all process return memory

#define mb(x)           (((int)((x)/MB)+1)&~1)
#define pg(x)           ((int)((x)/PG))
#define blk(x)          ((int)((x)/BLK_SIZE))

#endif // __CUDA_MASTER_H__
