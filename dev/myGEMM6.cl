#include "config.h"

uint8_t gf_mu_x86(uint8_t a, uint8_t b) {
	uint8_t p = 0; /* the product of the multiplication */
    #pragma unroll
	for (int i=0;i<8;i++){
            // if (!(a && b)){
            //         break;
            //     }
            if (b & 1) /* if b is odd, then add the corresponding a to p (final product = sum of all a's corresponding to odd b's) */
                p ^= a; /* since we're in GF(2^m), addition is an XOR */

            if (a & 0x80) /* GF modulo: if a >= 128, then it will overflow when shifted left, so reduce */
                a = (a << 1) ^ 0x11D; /* XOR with the primitive polynomial x^8 + x^4 + x^3 + x + 1 (0b1_0001_1011) – you can change it but it must be irreducible */
            else
                a <<= 1; /* equivalent to a*2 */
            b >>= 1; /* equivalent to b // 2 */
            
	}
	return p;
}

int address_interpretor(int x, int y, int offset, __global const uint8_t* restrict sample_idx){
    // use x to find index of required packet (file space) in sample_idx    
    uint8_t file_pkt_idx = sample_idx[offset+x];
    // calculate idx of required data in file space
    return file_pkt_idx*PKT_SIZE + y;
}

// Use 2D register blocking (further increase in work per thread)
__kernel 
__attribute__((num_compute_units(CMP_UNIT)))
__attribute__((reqd_work_group_size(TSM/WPTM, TSN/WPTN, 1)))  // 8, 1, 1
void myGEMM6(
            __global const uint8_t* restrict A,
            __global const uint8_t* restrict B,
            __global uint8_t* restrict C,
            __global volatile uint8_t* restrict DEGREE_,
            __global const uint8_t* restrict sample_idx, // cached
            __global volatile int* restrict DEGREE_OFF
                      ) {
                

    // Thread identifiers
    const int tidm = get_local_id(0); // Local row ID (max: TSM/WPTM == RTSM)
    const int tidn = get_local_id(1); // Local col ID (max: TSN/WPTN == RTSN)
    const int offsetM = TSM*get_group_id(0); // Work-group offset
    const int offsetN = TSN*get_group_id(1); // Work-group offset
    const int batch_id = get_global_id(2); // max: N_BATCH

    // Local memory to fit a tile of A and B
    __local uint8_t Asub[TSK][TSM];
    __local uint8_t Bsub[TSM][TSK];
    int deg_offset; // private
    uint8_t my_deg; // private
    // Allocate register space
    uint8_t Areg;
    uint8_t Breg[WPTN];
    uint8_t acc[WPTM][WPTN];
    

    // Initialise the accumulation registers
    #pragma unroll
    for (int wm=0; wm<WPTM; wm++) {
        #pragma unroll
        for (int wn=0; wn<WPTN; wn++) {
            acc[wm][wn] = 0;
        }
    }
    
    // load degrees and calculate offsets    
    my_deg = DEGREE_[batch_id];                                                                                          
    deg_offset = DEGREE_OFF[batch_id]; 
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
    // Loop over all tiles
    const int numTiles = my_deg/TSK;
    for(int t=0;t<numTiles;t++){

        // Load one tile of A and B into local memory
    
        #pragma unroll 3
        for (int la=0; la<LPTA; la++) {
            int tid = tidn*RTSM + tidm;
            int id = la*RTSN*RTSM + tid;
            int row = MOD2(id,TSM);
            int col = DIV2(id,TSM);
            int tiledIndex = TSK*t + col;
            int A_vec = address_interpretor(tiledIndex, offsetM + row, deg_offset,sample_idx);

            Asub[col][row] = A[A_vec];
            Bsub[row][col]= B[tiledIndex*BATCH_SIZE + offsetN + row + deg_offset*BATCH_SIZE];
        }

        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        // Loop over the values of a single tile
        // #pragma unroll 2
        for (int k=0; k<TSK; k++) {
            // Cache the values of Bsub in registers
            #pragma unroll
            for (int wn=0; wn<WPTN; wn++) {
                int col = tidn + wn*RTSN;
                Breg[wn] = Bsub[col][k];
            }

            // Perform the computation
            #pragma unroll
            for (int wm=0; wm<WPTM; wm++) {
                int row = tidm + wm*RTSM;

                Areg = Asub[k][row];
                #pragma unroll
                for (int wn=0; wn<WPTN; wn++) {
                    acc[wm][wn] ^= gf_mu_x86(Areg , Breg[wn]);
                }
            }
        }

        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store the final results in C
    // #pragma unroll 2
    for (int wm=0; wm<WPTM; wm++) {
        int globalRow = offsetM + tidm + wm*RTSM;
        #pragma unroll
        for (int wn=0; wn<WPTN; wn++) {
            int globalCol = offsetN + tidn + wn*RTSN; 
            C[globalCol*PKT_SIZE + globalRow + batch_id*PKT_SIZE*BATCH_SIZE] = acc[wm][wn];
        }
    }
    
}