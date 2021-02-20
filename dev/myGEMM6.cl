// Use 2D register blocking (further increase in work per thread)
__kernel void myGEMM6(const int M, const int N, const int K,
                      __global const uint8_t* restrict A,
                      __global const uint8_t* restrict B,
                      __global uint8_t* restrict C
                      ) {
                

    // Thread identifiers
    const int tidm = get_local_id(0); // Local row ID (max: TSM/WPTM == RTSM)
    const int tidn = get_local_id(1); // Local col ID (max: TSN/WPTN == RTSN)
    const int offsetM = TSM*get_group_id(0); // Work-group offset
    const int offsetN = TSN*get_group_id(1); // Work-group offset
    const int batch_id = get_global_id(2); // max: N_BATCH

    // Local memory to fit a tile of A and B
    __local uint8_t Asub[TSK][TSM][MAX_NUM_BATCH];
    __local uint8_t Bsub[TSN][TSK+2][MAX_NUM_BATCH];
    __local uint8_t degrees[MAX_NUM_BATCH];
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
    
    // Load degrees
    if(tidm == 0 && tidn == 0){
        degrees[batch_id] = DEGREE_[batch_id];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    uint8_t my_deg = degrees[batch_id];

    // Loop over all tiles
    const int numTiles = K/TSK;
    int t=0;
    do {

        // Load one tile of A and B into local memory
        #pragma unroll
        for (int la=0; la<LPTA; la++) {
            int tid = tidn*RTSM + tidm;
            int id = la*RTSN*RTSM + tid;
            int row = MOD2(id,TSM);
            int col = DIV2(id,TSM);
            int tiledIndex = TSK*t + col;
            Asub[col][row] = A[tiledIndex*M + offsetM + row];
            Bsub[row][col] = B[tiledIndex*N + offsetN + row];
        }

        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        // Loop over the values of a single tile
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
                    acc[wm][wn] += Areg * Breg[wn];
                }
            }
        }

        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);

        // Next tile
        t++;
    } while (t<numTiles);

    // Store the final results in C
    #pragma unroll
    for (int wm=0; wm<WPTM; wm++) {
        int globalRow = offsetM + tidm + wm*RTSM;
        #pragma unroll
        for (int wn=0; wn<WPTN; wn++) {
            int globalCol = offsetN + tidn + wn*RTSN;
            C[globalCol*M + globalRow] = acc[wm][wn];
        }
    }
}