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



int address_interpretor(int x, int y, int offset, __global volatile int* restrict sample_idx){
    // use x to find index of required packet (file space) in sample_idx    
    int file_pkt_idx = sample_idx[offset+x];
    if(file_pkt_idx == PADDING_ID){
        return PADDING_ID;
    }
    // calculate idx of required data in file space
    return file_pkt_idx*(PKT_SIZE) + y;
}

__kernel
__attribute__((reqd_work_group_size(TS_COF, TS_COF, 1))) 
void recoder_cof(__global volatile uint8_t* restrict A, // 1040 by 16 (only calculate 16 by 16)
                    __global volatile uint8_t* restrict B, // 16 by 16
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
    // test[globalRow] = 1;
    // Loop over all tiles
    const int numTiles = COEFF_SIZE/TS_COF;
    // #pragma unroll 2
    for (int t=0; t<numTiles; t++) {
 
        // Load one tile of A and B into local memory
        const int tiledRow = TS_COF*t + row;
        const int tiledCol = TS_COF*t + col;
        Asub[col][row] = A[tiledCol*PKT_WITH_COEFF + globalRow + batch_id_glb*PKT_WITH_COEFF*BATCH_SIZE];
        Bsub[col][row] = B[tiledRow*COEFF_SIZE + globalCol + batch_id_glb*COEFF_SIZE*COEFF_SIZE];
 
        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);
 
        // Perform the computation for a single tile
        #pragma unroll
        for (int k=0; k<TS_COF; k++) {
            acc ^= gf_mu_x86(Asub[k][row] , Bsub[col][k]) ;
        }
 
        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }
 
    // Store the final result in C
    C[BATS_HEADER+globalCol*(PKT_WITH_COEFF + BATS_HEADER)+ globalRow + batch_id_glb*(PKT_WITH_COEFF + BATS_HEADER)*BATCH_SIZE] = acc;
    
}

// Use 2D register blocking (further increase in work per thread)
__kernel 
__attribute__((reqd_work_group_size(TSM/WPTM, TSN/WPTN, 1)))  // 10, 1, 1
__kernel void coder( __global volatile uint8_t* restrict A,
            __global volatile uint8_t* restrict B,
            __global uint8_t* restrict C,
            __global volatile uint8_t* restrict common_dim, // multiple of 4
            __global volatile int* restrict sample_idx,
            __global volatile int* restrict common_dim_offsets, 
            __global volatile int* restrict output_sample_idx,
            const uint8_t outer_dim, // multiple of 4
            const uint8_t mode,
            const uint8_t add_to_enable) {
    
    // Thread identifiers
    const int local_m = get_local_id(0); // Local row ID (max: TSM/WPTM)
    const int local_n = get_local_id(1); // Local col ID (max: TSN/WPTN)
    const int global_m = get_group_id(0); // Work-group offset
    const int global_n = get_group_id(1); // Work-group offset
    const int batch_id = get_global_id(2); // max: N_BATCH
 
    // Local memory to fit a tile of A and B
    __local uint8_t Asub[TSK][TSM];
    __local uint8_t Bsub[TSN][TSK];
 
    // Allocate register space
    uint8_t Areg = 0;
    uint8_t Breg[WPTN];
    uint8_t acc[WPTM][WPTN];
    int deg_offset = 0; // private
    uint8_t my_deg = 0; // private
    uint8_t out_dim = 0;
    int out_dim_offset = 0;

    // Initialise the accumulation registers
    #pragma unroll
    for (int wm=0; wm<WPTM; wm++) {
        #pragma unroll
        for (int wn=0; wn<WPTN; wn++) {
            acc[wm][wn] = 0;
        }
    }

    // load degrees and calculate offsets
    if(mode == RECODER_ENABLE){
        my_deg = BATCH_SIZE;
        deg_offset = BATCH_SIZE*batch_id;
    }  
    else if (mode == DECODER_ENABLE){
        my_deg = common_dim[batch_id];   
        deg_offset = common_dim_offsets[batch_id];
        out_dim = outer_dim ; 
        out_dim_offset = out_dim*batch_id;                                                       
    }  
    else{
        my_deg = common_dim[batch_id];                                                                                          
        deg_offset = common_dim_offsets[batch_id];
    }


    
    // Loop over all tiles
    const int numTiles = my_deg/TSK;
    for (int t=0; t<numTiles; t++) {
        // Load one tile of A and B into local memory
        // Load tile A
        for(int j=0;j<TSK;j++){
            for(int i=0;i<WPTM;i++){
                int row_tile = i + local_m*WPTM;
                int col_tile = j;
                int row_global = row_tile + global_m*TSM;
                int col_global = col_tile + t*TSK;
                int A_vec = 0;
                if(mode == RECODER_ENABLE){
                    A_vec = COEFF_SIZE + col_global*PKT_WITH_COEFF +  row_global + batch_id*PKT_WITH_COEFF*BATCH_SIZE;
                }
                else{
                    A_vec = sample_idx[deg_offset + col_global] * PKT_SIZE + row_global;
                    // A_vec = address_interpretor(col_global, row_global, deg_offset,sample_idx);
                }
                if(A_vec == PADDING_ID){
                    Asub[col_tile][row_tile] = 0;
                }
                else{
                    Asub[col_tile][row_tile] = A[A_vec];
                }
            }
        }
        // Load title B
        for(int j=0;j<WPTN;j++){
            for(int i=0;i<TSK;i++){
                int row_tile = i;
                int col_tile = j + local_n*WPTN;
                int row_global = row_tile + t*TSK;
                int col_global = col_tile + global_n*TSN;
                int B_vec = col_global*out_dim + row_global + deg_offset*out_dim;

                Bsub[col_tile][row_tile] = B[B_vec];
            }
        }
        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);
 
        // Loop over the values of a single tile
        for (int k=0; k<TSK; k++) {
 
            // Cache the values of Bsub in registers
            for (int wn=0; wn<WPTN; wn++) {
                int col = wn + local_n*WPTN;
                Breg[wn] = Bsub[col][k];
            }
 
            // Perform the computation
            for (int wm=0; wm<WPTM; wm++) {
                int row = wm + local_m*WPTM;
                Areg = Asub[k][row];
                for (int wn=0; wn<WPTN; wn++) {
                    acc[wm][wn] ^= gf_mu_x86(Areg , Breg[wn]);
                }
            }
        }
        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }
 
    // Store the final results in C
    for(int i=0;i<WPTN;i++){
        for(int j=0;j<WPTM;j++){
            int row_global = j + local_m*WPTM + global_m*TSM;
            int col_global = i + local_n*WPTN + global_n*TSN;
            int C_vec = 0;

            if(mode == DECODER_ENABLE){
                C_vec = output_sample_idx[out_dim_offset + col_global] * PKT_SIZE + row_global;
                // C_vec = address_interpretor(col_global, row_global, out_dim_offset,output_sample_idx);
                if(C_vec != PADDING_ID){
                    if(add_to_enable){
                        C[C_vec] ^= acc[j][i];
                    }
                    else{
                        C[C_vec] = acc[j][i];
                    }
                }
            }
            else{
                C_vec = BATS_HEADER + COEFF_SIZE + col_global*(PKT_WITH_COEFF+BATS_HEADER) 
                            + row_global + batch_id*(PKT_WITH_COEFF+BATS_HEADER)*BATCH_SIZE;
                C[C_vec] = acc[j][i];
            }
          
        }
    }
}
