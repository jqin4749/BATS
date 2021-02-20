
#define PKT_SIZE 1024 // pkt size M
#define PKT_NUM 64
#define BATCH_SIZE 4 // batch size N
#define MAX_DEGREE 16 // K   MAX
#define N_BATCH 20
#define MAX_NUM_BATCH 20 // used only at buffer creation
#define FILE_SIZE PKT_SIZE*PKT_NUM

#ifndef GCC
#define uint16_t unsigned short
#define uint8_t unsigned char
#endif

#define TSM 128                // The tile-size in dimension M
#define TSN 128                // The tile-size in dimension N
#define TSK 16                 // The tile-size in dimension K
#define WPTM 8                 // The work-per-thread in dimension M
#define WPTN 8                 // The work-per-thread in dimension N
#define RTSM (TSM/WPTM)        // The reduced tile-size in dimension M
#define RTSN (TSN/WPTN)        // The reduced tile-size in dimension N
#define LPTA ((TSK*TSM)/(RTSM*RTSN)) // Loads-per-thread for A
#define LPTB ((TSK*TSN)/(RTSM*RTSN)) // Loads-per-thread for B

#define SIMD_TS 2
#define CMP_UNIT 2
#define SEED 2021

#define MOD2(x,y) ((x) % (y))
#define DIV2(x,y) ((x) / (y))
