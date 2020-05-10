#include "conv_net.h"
#include <stdint.h>

void flatten(
        DTYPE IN[P2_DOWNSIZE][P2_DOWNSIZE][C4_N_FILTERS],
        DTYPE OUT[FLAT_VEC_SZ]) {

    uint16_t i,j,k;
    uint16_t t = 0;

    for (j = 0; j < P2_DOWNSIZE; ++j) {
    	for (k = 0; k < P2_DOWNSIZE; ++k) {
    	    flatten:for (i = 0; i < C4_N_FILTERS; ++i) {
                OUT[t++] = IN[j][k][i];
            }
        }
    }
}

void vec_mat_mul_f1(
          DTYPE X[FLAT_VEC_SZ],
    const DTYPE W[F1_COLS][F1_ROWS],
    const DTYPE bias[F1_ROWS],
          DTYPE Z[F1_ROWS]) {

    uint16_t c;
    uint16_t r;

    for (r = 0; r < F1_ROWS; ++r) {
        Z[r] = bias[r];
        for (c = 0; c < F1_COLS; ++c) {
            Z[r] += W[c][r] * X[c];
        }
    }
}

void vec_mat_mul_f2(
          DTYPE X[F1_ROWS],
    const DTYPE W[F2_COLS][F2_ROWS],
    const DTYPE bias[F2_ROWS],
          DTYPE Z[F2_ROWS]) {

    uint16_t r, c;

    for (r = 0; r < F2_ROWS; ++r) {
        Z[r] = bias[r];
        for (c = 0; c < F2_COLS; ++c) {
            Z[r] += W[c][r] * X[c];
        }
    }
}

void vec_mat_mul_f3(
          DTYPE X[F2_ROWS],
    const DTYPE W[F3_COLS][F3_ROWS],
    const DTYPE bias[F3_ROWS],
          DTYPE Z[F3_ROWS]) {

    uint8_t r, c;

    for (r = 0; r < F3_ROWS; ++r) {
        Z[r] = bias[r];
        for (c = 0; c < F3_COLS; ++c) {
            Z[r] += W[c][r] * X[c];
        }
    }
}
