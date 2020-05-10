#include "conv_net.h"
#include <stdint.h>

#define max(a,b) \
({ __typeof__ (a) _a = (a); \
   __typeof__ (b) _b = (b); \
 _a > _b ? _a : _b; })

DTYPE maxFour(DTYPE a, DTYPE b, DTYPE c, DTYPE d) {
    return max(max(a,b), max(c,d));
}

void pooling_p1(DTYPE in[P1_SIZE][P1_SIZE][C2_N_FILTERS], DTYPE out[P1_DOWNSIZE][P1_DOWNSIZE][C2_N_FILTERS]) {

    uint8_t i, j, m;

    for (m = 0; m < C1_N_FILTERS; ++m) {
        for (i = 0; i < P1_DOWNSIZE; i++) {
            for (j = 0; j < P1_DOWNSIZE; j++) {
                #pragma HLS UNROLL
            	out[i][j][m] = maxFour(
            	        in[i << 1][j << 1][m],
            	        in[(i << 1) + 1][j << 1][m],
            	        in[i << 1][(j << 1) + 1][m],
            	        in[(i << 1) + 1][(j << 1) + 1][m]

                );
            }
        }
    }
}

void pooling_p2(DTYPE in[P2_SIZE][P2_SIZE][C4_N_FILTERS], DTYPE out[P2_DOWNSIZE][P2_DOWNSIZE][C4_N_FILTERS]) {

    uint8_t i, j, m;

    for (m = 0; m < C2_N_FILTERS; ++m) {
        for (i = 0; i < P2_DOWNSIZE; i++) {
            for (j = 0; j < P2_DOWNSIZE; j++) {
                #pragma HLS UNROLL
                out[i][j][m] = maxFour(
                        in[i << 1][j << 1][m],
                        in[(i << 1) + 1][j << 1][m],
                        in[i << 1][(j << 1) + 1][m],
                        in[(i << 1) + 1][(j << 1) + 1][m]
                );
            }
        }
    }
}
