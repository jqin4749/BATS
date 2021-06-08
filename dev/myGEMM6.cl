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
void coder(
            __global const uint8_t* restrict A,
            __global const uint8_t* restrict B,
            __global uint8_t* restrict C,
            __global volatile uint8_t* restrict common_dim, // multiple of 4
            __global volatile int* restrict sample_idx,
            __global volatile int* restrict common_dim_offsets, 
            __global volatile int* restrict output_sample_idx,
            const uint8_t outer_dim, // multiple of 4
            const uint8_t mode,
            const uint8_t add_to_enable

                      ) {
    // Thread identifiers
    const int tidm = get_local_id(0); // Local row ID (max: TSM/WPTM == RTSM)
    const int tidn = get_local_id(1); // Local col ID (max: TSN/WPTN == RTSN)
    const int offsetM = TSM*get_group_id(0); // Work-group offset
    const int offsetN = TSN*get_group_id(1); // Work-group offset
    const int batch_id = get_global_id(2); // max: N_BATCH

    // Local memory to fit a tile of A and B
    __local uint8_t Asub[TSK][TSM];
    __local uint8_t Bsub[TSK][TSM];
    int deg_offset = 0; // private
    uint8_t my_deg = 0; // private
    uint8_t out_dim = 0;
    int out_dim_offset = 0;
    // Allocate register space
    uint8_t Areg = 0;
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
            int B_vec = tiledIndex*BATCH_SIZE + offsetN + row + deg_offset*BATCH_SIZE;
            if(mode == RECODER_ENABLE){
                A_vec = COEFF_SIZE + tiledIndex*PKT_WITH_COEFF + offsetM + row + batch_id*PKT_WITH_COEFF*BATCH_SIZE;
            }
            else if(mode == DECODER_ENABLE){
                // A_vec = tiledIndex*PKT_SIZE + offsetM + row + deg_offset*PKT_SIZE;
                // B_vec = tiledIndex*var_batch_size + offsetN + row + deg_offset*var_batch_size;
                A_vec = address_interpretor(tiledIndex, offsetM + row, deg_offset,sample_idx);
                B_vec = tiledIndex*out_dim + offsetN + row + deg_offset*out_dim;
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
            
            Bsub[col][row] = B[B_vec];
            
        }

        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        // Loop over the values of a single tile
        #pragma unroll 2
        for (int k=0; k<TSK; k++) {
            // Cache the values of Bsub in registers
            #pragma unroll
            for (int wn=0; wn<WPTN; wn++) {
                int col = tidn + wn*RTSN;
                Breg[wn] = Bsub[k][col];
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
            int C_vec = BATS_HEADER + COEFF_SIZE + globalCol*(PKT_WITH_COEFF+BATS_HEADER) 
                            + globalRow + batch_id*(PKT_WITH_COEFF+BATS_HEADER)*BATCH_SIZE;
            if(mode == DECODER_ENABLE){
                // C_vec = globalCol*PKT_SIZE + globalRow + batch_id * var_batch_size * PKT_SIZE; 
                C_vec = address_interpretor(globalCol, globalRow, out_dim_offset,output_sample_idx);
            }
            if(C_vec != PADDING_ID){
                if(add_to_enable){
                    C[C_vec] ^= acc[wm][wn];
                }
                else{
                    C[C_vec] = acc[wm][wn];
                }
            }
        }
    }
    
}


