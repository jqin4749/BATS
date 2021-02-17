
#define PKT_SIZE 1024 // pkt size M
#define PKT_NUM 64
#define BATCH_SIZE 4 // batch size N
#define MAX_DEGREE 64 // K   MAX
#define N_BATCH 20
#define MAX_NUM_BATCH 30 // used only at buffer creation
#define FILE_SIZE PKT_SIZE*PKT_NUM

#ifndef GCC
#define uint16_t unsigned short
#define uint8_t unsigned char
#endif

#define TS 4
#define SIMD_TS 2
#define CMP_UNIT 2

#define SEED 2021



#define WPT 8                        // The amount of work-per-thread, i.e. the thread-coarsening factor
#define RTS (TS/WPT)                 // The reduced tile-size in one dimension
