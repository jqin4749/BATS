

// column major
template<int row, int col>
void matrix_flatten(uint8_t A[row][col],uint8_t res[row*col]){
    int count=0;
    for(int i=0;i<col;i++){
        for(int j=0;j<row;j++){
            res[count] = A[j][i];
            count++;
        }
    }
}
template<int row, int col>
void matrix_reform(uint8_t A[row*col], uint8_t res[row][col]){
    int count=0;
    for(int i=0;i<col;i++){
        for(int j=0;j<row;j++){
            res[j][i] = A[count];
            count++;
        }
    }
}

uint8_t gf_mu_x86(uint8_t a, uint8_t b) {
	int p = 0; /* the product of the multiplication */

	while (a && b) {
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

// assume input array is 1D
void matrix_multi(uint8_t* A, uint8_t* B, uint8_t* C, int M, int K, int N){
    for (int m=0; m<M; m++) {
        for (int n=0; n<N; n++) {
            int acc = 0;
            for (int k=0; k<K; k++) {
                acc ^= gf_mu_x86(A[k*M + m] , B[n*K + k]);
            }
            C[n*M + m] = acc;
        }
    }
}

