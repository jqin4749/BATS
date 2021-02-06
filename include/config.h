
#define PKT_SIZE 256 // pkt size M
#define BATCH_SIZE 16 // batch size N
#define DEGREE 16 // K

#ifndef GCC
#define uint16_t unsigned short
#define uint8_t unsigned char
#endif

#define TS 8
#define SIMD_TS 2
#define CMP_UNIT 2
#define SEED 2021
