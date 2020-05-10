#include "conv_net.h"
#include <stdint.h>

void relu_a1(DTYPE in[A1_ROWS][A1_COLS][C1_N_FILTERS], DTYPE out[A1_ROWS][A1_COLS][C1_N_FILTERS]) {

    uint16_t r;
    uint8_t c, m;

    for (r = 0; r < A1_ROWS; ++r) {
    	for (c = 0; c < A1_COLS; ++c) {
    		relu_a1:for (m = 0; m < C1_N_FILTERS; ++m) {
    			out[r][c][m] = (in[r][c][m] > 0) ? (in[r][c][m]>>10) : 0;
            }
        }
    }
}

void relu_a2(DTYPE in[A2_ROWS][A2_COLS][C2_N_FILTERS], DTYPE out[A2_ROWS][A2_COLS][C2_N_FILTERS]) {

    uint16_t r;
    uint8_t c, m;

    for (r = 0; r < A2_ROWS; ++r) {
    	for (c = 0; c < A2_COLS; ++c) {
    		relu_a2:for (m = 0; m < C2_N_FILTERS; ++m) {
                out[r][c][m] = (in[r][c][m] > 0) ? (in[r][c][m]>>10) : 0;
            }
        }
    }
}

void relu_a3(DTYPE in[A3_ROWS][A3_COLS][C3_N_FILTERS], DTYPE out[A3_ROWS][A3_COLS][C3_N_FILTERS]) {

    uint16_t r;
    uint8_t c, m;

    for (r = 0; r < A3_ROWS; ++r) {
    	for (c = 0; c < A3_COLS; ++c) {
    		relu_a3:for (m = 0; m < C3_N_FILTERS; ++m) {
                out[r][c][m] = (in[r][c][m] > 0) ? (in[r][c][m]>>10) : 0;
            }
        }
    }
}

void relu_a4(DTYPE in[A4_ROWS][A4_COLS][C4_N_FILTERS], DTYPE out[A4_ROWS][A4_COLS][C4_N_FILTERS]) {

    uint16_t r;
    uint8_t c, m;

    for (r = 0; r < A4_ROWS; ++r) {
    	for (c = 0; c < A4_COLS; ++c) {
    		relu_a4:for (m = 0; m < C4_N_FILTERS; ++m) {
                out[r][c][m] = (in[r][c][m] > 0) ? (in[r][c][m]>>10) : 0;
            }
        }
    }
}

void relu_a5(DTYPE in[F1_ROWS], DTYPE out[F1_ROWS]) {

    uint16_t i;

    relu_a3:for (i = 0; i < F1_ROWS; ++i) {
        out[i] = (in[i] > 0) ? (in[i]>>10) : 0;
    }
}

void relu_a6(DTYPE in[F2_ROWS], DTYPE out[F2_ROWS]) {

    uint16_t i;

    relu_a4:for (i = 0; i < F2_ROWS; ++i) {
        out[i] = (in[i] > 0) ? (in[i]>>10) : 0;
    }
}
