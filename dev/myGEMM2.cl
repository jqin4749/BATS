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

// inferred target matrix from offset,degree and sample idx
// given requied position at target matix (x,y), return mapped position in file A 
int address_interpretor(int x, int y, int offset, __global const uint8_t* restrict sample_idx){
    // use x to find index of required packet (file space) in sample_idx    
    uint8_t file_pkt_idx = sample_idx[offset+x];
    // calculate idx of required data in file space
    return file_pkt_idx*PKT_SIZE + y;
}

// Tiled and coalesced version
__kernel
// __attribute__((num_simd_work_items(SIMD_TS)))
__attribute__((num_compute_units(CMP_UNIT)))
// __attribute__((max_work_group_size(TS*TS*MAX_NUM_BATCH))) 
__attribute__((reqd_work_group_size(TS,TS,1))) 
void myGEMM2(
            __global const uint8_t* restrict  A, // file to be encoded cached
            __global const uint8_t* restrict  B, // Generator matrix cached
            __global uint8_t* restrict C,
            __global volatile uint8_t* restrict DEGREE_,
            __global const uint8_t* restrict sample_idx // cached
            ) {
    
    // Thread identifiers
    const int row = get_local_id(0); // Local row ID (max: TS) (tile space)
    const int col = get_local_id(1); // Local col ID (max: TS)
    const int globalRow = TS*get_group_id(0) + row; // Row ID of C (0..M)
    const int globalCol = TS*get_group_id(1) + col; // Col ID of C (0..N)
    const int batch_id = get_global_id(2); // max: N_BATCH

    // Local memory to fit a tile of TS*TS elements of A and B
    __local uint8_t Asub[TS][TS][MAX_NUM_BATCH];
    __local uint8_t Bsub[TS][TS][MAX_NUM_BATCH];
    __local uint8_t degrees[MAX_NUM_BATCH];
    int deg_offset = 0;
    uint8_t acc = 0;
    // load degrees and calculate offsets
    // printf("Tile no: %d Group:[%d,%d] Global:(%d,%d)\n",0,get_group_id(0),get_group_id(1),get_global_id(0),get_global_id(1));
  
    degrees[batch_id] = DEGREE_[batch_id];
    barrier(CLK_LOCAL_MEM_FENCE);
    
    uint8_t my_deg = degrees[batch_id];
    
    #pragma ii 1
    for(int i=0;i<batch_id;i++){
        deg_offset += degrees[i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Loop over all tiles
    int numTiles = my_deg/TS;
    int A_vec;
    #pragma ii 1
    for (int t=0; t<numTiles; t++) {
 
        // Load one tile of A and B into local memory
        int tiledRow = TS*t + row; // tile space
        int tiledCol = TS*t + col; // tile space
        int A_x = tiledCol; // A space
        int A_y = globalRow;// A space

        A_vec = address_interpretor(A_x,A_y,deg_offset,sample_idx); // get vectorized position idx in file

        Asub[row][col][batch_id] = A[A_vec];         
        Bsub[col][row][batch_id] = B[globalCol*my_deg + tiledRow + deg_offset*BATCH_SIZE];

        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // Perform the computation for a single tile
        #pragma unroll
        for (int k=0; k<TS; k++) {
            acc ^= gf_mu_x86(Asub[row][k][batch_id] , Bsub[col][k][batch_id]); // now we can access Asub and Bsun in consecutive order

        }

        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // Store the final result in C
    C[globalCol*PKT_SIZE + globalRow + batch_id*PKT_SIZE*BATCH_SIZE] = acc;

}