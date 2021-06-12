
// matrix parameters
#define PKT_SIZE 1024 // pkt size M
#define PKT_NUM 64
#define BATCH_SIZE 16 // batch size N
#define MAX_DEGREE 20 // K   MAX
#define N_BATCH 20
#define MAX_NUM_BATCH N_BATCH // used only at buffer creation
#define FILE_SIZE PKT_SIZE*PKT_NUM
#define RECODER_ENABLE 1
#define DECODER_ENABLE 2

// BATS parameters
#define COEFF_SIZE BATCH_SIZE
#define PKT_WITH_COEFF (COEFF_SIZE+PKT_SIZE)

#ifndef DEBUG
#define BATS_HEADER 0 // sizeof(struct BatsHeader)
#else
#define BATS_HEADER 40 // sizeof(struct BatsHeader)
#endif

#ifndef GCC
#define uint16_t unsigned short
#define uint8_t unsigned char
#endif

// tiling parameters
#define TSM 128                // The tile-size in dimension M
#define TSN 4                // The tile-size in dimension N
#define TSK 4                 // The tile-size in dimension K
#define WPTM 8                 // The work-per-thread in dimension M
#define WPTN 4                 // The work-per-thread in dimension N
#define RTSM (TSM/WPTM)        // The reduced tile-size in dimension M 16
#define RTSN (TSN/WPTN)        // The reduced tile-size in dimension N
#define LPTA ((TSK*TSM)/(RTSM*RTSN)) // Loads-per-thread for A 32
#define LPTB ((TSK*TSN)/(RTSM*RTSN)) // Loads-per-thread for B 2
#define PADDING_ID -1

#define TS_COF 4 // tile size for coeff multiplication
// const size_t global[3] = {  PKT_SIZE/WPTM, BATCH_SIZE/WPTN, N_BATCH };
// const size_t local[3] = { TSM/WPTM, TSN/WPTN, 1 };
#define SIMD_TS 2
#define CMP_UNIT 1
#define SEED 2021

#define MOD2(x,y) ((x) % (y))
#define DIV2(x,y) ((x) / (y))

#define UNROLL_FACTOR 2