#include "config.h"

uint8_t gf_mu_x86(uint8_t a, uint8_t b) {
	uint8_t p = 0; /* the product of the multiplication */
    #pragma unroll
	for (int i=0;i<8;i++){
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


int address_interpretor(int x, int y, int offset, __global const int* restrict sample_idx){
    // use x to find index of required packet (file space) in sample_idx    
    int file_pkt_idx = sample_idx[offset+x];
    if(file_pkt_idx == PADDING_ID){
        return PADDING_ID;
    }
    // calculate idx of required data in file space
    return file_pkt_idx*(PKT_SIZE) + y;
}

__attribute__((num_simd_work_items(TS_COF)))
__attribute__((reqd_work_group_size(TS_COF, 1, 1))) 
__kernel void vec_mul(__global const uint8_t* restrict x, 
                         __global const uint8_t* restrict y, 
                         __global uint8_t *restrict z)
{
    // get index of the work item
    int index = get_global_id(0);

    z[index] ^= gf_mu_x86(x[index] , y[index]);
}

__kernel
__attribute__((reqd_work_group_size(TS_COF, TS_COF, 1))) 
void recoder_cof(__global volatile uint8_t* restrict A, // 1028 by 4 (only calculate 4 by 4)
                    __global volatile uint8_t* restrict B, // 4 by 4
                    __global uint8_t* restrict C){

     // Thread identifiers
    const int row = get_local_id(0); // Local row ID (max: TS)
    const int col = get_local_id(1); // Local col ID (max: TS)
    const int globalRow = TS_COF*get_group_id(0) + row; // Row ID of C (0..M)
    const int globalCol = TS_COF*get_group_id(1) + col; // Col ID of C (0..N)
    const int batch_id_glb = get_global_id(2);
    // Local memory to fit a tile of TS*TS elements of A and B
    __local uint8_t Asub[TS_COF][TS_COF];
    __local uint8_t Bsub[TS_COF][TS_COF];
 
    // Initialise the accumulation register
    uint8_t acc = 0;
    
    // Loop over all tiles
    const int numTiles = COEFF_SIZE/TS_COF;
    #pragma unroll
    for (int t=0; t<numTiles; t++) {
 
        // Load one tile of A and B into local memory
        const int tiledRow = TS_COF*t + row;
        const int tiledCol = TS_COF*t + col;
        Asub[col][row] = A[tiledCol*PKT_WITH_COEFF + globalRow + batch_id_glb*PKT_WITH_COEFF*BATCH_SIZE];
        Bsub[row][col] = B[tiledRow*COEFF_SIZE + globalCol + batch_id_glb*COEFF_SIZE*COEFF_SIZE];
 
        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);
 
        // Perform the computation for a single tile
        #pragma unroll
        for (int k=0; k<TS_COF; k++) {
            acc ^= gf_mu_x86(Asub[k][row] , Bsub[k][col]) ;
        }
 
        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }
 
    // Store the final result in C
    C[BATS_HEADER+globalCol*(PKT_WITH_COEFF + BATS_HEADER)+ globalRow + batch_id_glb*(PKT_WITH_COEFF + BATS_HEADER)*BATCH_SIZE] = acc;
    
}


// Use 2D register blocking (further increase in work per thread)
__kernel 
__attribute__((reqd_work_group_size(TSM/WPTM, TSN/WPTN, 1)))  // 8, 1, 1
__attribute__((num_compute_units(CMP_UNIT)))
void coder(

            __global const uint8_t* restrict A,
            __global const uint8_t* restrict B,
            __global uint8_t* restrict C,
            __global volatile uint8_t* restrict DEGREE_,
            __global const int* restrict sample_idx, // cached
            __global volatile int* restrict DEGREE_OFF,
            const uint8_t recoder_enable
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
    if(recoder_enable == 1){
        my_deg = BATCH_SIZE;
        deg_offset = BATCH_SIZE*batch_id;
    }    
    else{
        my_deg = DEGREE_[batch_id];                                                                                          
        deg_offset = DEGREE_OFF[batch_id];
    }
     
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
    // Loop over all tiles
    const int numTiles = my_deg/TSK;
    for(int t=0;t<numTiles;t++){

        // Load one tile of A and B into local memory
    
        #pragma unroll 2
        for (int la=0; la<LPTA; la++) {
            int tid = tidn*RTSM + tidm;
            int id = la*RTSN*RTSM + tid;
            int row = MOD2(id,TSM);
            int col = DIV2(id,TSM);
            int tiledIndex = TSK*t + col;
            int A_vec = 0;
            if(recoder_enable == 1){
                A_vec = COEFF_SIZE + tiledIndex*PKT_WITH_COEFF + offsetM + row + batch_id*PKT_WITH_COEFF*BATCH_SIZE;
            }
            else{
                A_vec = address_interpretor(tiledIndex, offsetM + row, deg_offset,sample_idx);
            }
            
            if(A_vec == PADDING_ID){
                Asub[col][row] = 0;
            }
            else{
                Asub[col][row] = A[A_vec];
            }
            Bsub[row][col]= B[tiledIndex*BATCH_SIZE + offsetN + row + deg_offset*BATCH_SIZE];
        }

        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        // Loop over the values of a single tile
        // #pragma unroll
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
    // #pragma unroll
    for (int wm=0; wm<WPTM; wm++) {
        int globalRow = offsetM + tidm + wm*RTSM;
        #pragma unroll
        for (int wn=0; wn<WPTN; wn++) {
            int globalCol = offsetN + tidn + wn*RTSN; 
            C[BATS_HEADER + COEFF_SIZE + globalCol*(PKT_WITH_COEFF+BATS_HEADER) + globalRow + batch_id*(PKT_WITH_COEFF+BATS_HEADER)*BATCH_SIZE] = acc[wm][wn];
        }
    }
    
}


