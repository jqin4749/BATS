
#define PKT_SIZE 1024 // pkt size M
#define BATCH_SIZE 4 // batch size N
#define DEGREE 16 // K   MAX
#define N_BATCH 20

#ifndef GCC
#define uint16_t unsigned short
#define uint8_t unsigned char
#endif

#define TS 4
#define SIMD_TS 2
#define CMP_UNIT 3
#define SEED 2021
#define INPUT_SIZE_A PKT_SIZE*N_BATCH
#define INPUT_SIZE_B BATCH_SIZE*N_BATCH
#define OUTPUT_SIZE PKT_SIZE*BATCH_SIZE*N_BATCH 

