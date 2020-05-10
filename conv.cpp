#include "conv_net.h"
#include <stdint.h>

void convolution_c1 (
		      DTYPE   X[C1_X_DMNIN][C1_X_DMNIN][C1_N_CHAN],
        const DTYPE   W[C1_W_DMNIN][C1_W_DMNIN][C1_N_CHAN][C1_N_FILTERS],
              DTYPE out[C1_OUT_DMNIN][C1_OUT_DMNIN][C1_N_FILTERS],
        const DTYPE bias[C1_N_FILTERS]) {

    uint8_t ch, f, i, j, r, c;

    for (f = 0; f < C1_N_FILTERS; ++f) {
            for (r = 0; r < C1_OUT_DMNIN; ++r) {
                for (c = 0; c < C1_OUT_DMNIN; ++c) {
					#pragma HLS PIPELINE
                    out[r][c][f] = bias[f];
                }
            }
        }
      for (f = 0; f < C1_N_FILTERS; ++f) {
        for (ch = 0; ch < C1_N_CHAN; ++ch) {
            for (r = 0; r < C1_X_DMNIN - C1_W_DMNIN + 1; r += STRIDE) {
                for (c = 0, i = 0, j = 0; c < C1_X_DMNIN - C1_W_DMNIN + 1; c += STRIDE) {
                	for (i = 0; i < C1_W_DMNIN; ++i) {
                        for (j = 0; j < C1_W_DMNIN; ++j) {
                        	out[r][c][f] += W[i][j][ch][f] * X[r + i][j + c][ch];
                        }
                    }
                }
            }
        }
    }

}

void convolution_c2 (
              DTYPE   X[C2_X_DMNIN][C2_X_DMNIN][C2_N_CHAN],
        const DTYPE   W[C2_W_DMNIN][C2_W_DMNIN][C2_N_CHAN][C2_N_FILTERS],
              DTYPE out[C2_OUT_DMNIN][C2_OUT_DMNIN][C2_N_FILTERS],
        const DTYPE bias[C2_N_FILTERS]) {

    uint8_t ch, f, i, j, r, c;

    for (f = 0; f < C2_N_FILTERS; ++f) {
        for (r = 0; r < C2_OUT_DMNIN; ++r) {
            for (c = 0; c < C2_OUT_DMNIN; ++c) {
                #pragma HLS PIPELINE
                out[r][c][f] = bias[f];
            }
        }
    }
    for (f = 0; f < C2_N_FILTERS; ++f) {
        for (ch = 0; ch < C2_N_CHAN; ++ch) {
            for (r = 0; r < C2_X_DMNIN - C2_W_DMNIN + 1; r += STRIDE) {
                for (c = 0, i = 0, j = 0; c < C2_X_DMNIN - C2_W_DMNIN + 1; c += STRIDE) {
                    for (i = 0; i < C2_W_DMNIN; ++i) {
                        for (j = 0; j < C2_W_DMNIN; ++j) {
                        	out[r][c][f] += W[i][j][ch][f] * X[r + i][j + c][ch];
                        }
                    }
                }
            }
        }
    }
}

void convolution_c3 (
              DTYPE   X[C3_X_DMNIN][C3_X_DMNIN][C3_N_CHAN],
        const DTYPE   W[C3_W_DMNIN][C3_W_DMNIN][C3_N_CHAN][C3_N_FILTERS],
              DTYPE out[C3_OUT_DMNIN][C3_OUT_DMNIN][C3_N_FILTERS],
        const DTYPE bias[C3_N_FILTERS]) {

    uint8_t ch, f, i, j, r, c;

    for (f = 0; f < C3_N_FILTERS; ++f) {
        for (r = 0; r < C3_OUT_DMNIN; ++r) {
            for (c = 0; c < C3_OUT_DMNIN; ++c) {
                #pragma HLS PIPELINE
                out[r][c][f] = bias[f];
            }
        }
    }
    for (f = 0; f < C3_N_FILTERS; ++f) {
        for (ch = 0; ch < C3_N_CHAN; ++ch) {
            for (r = 0; r < C3_X_DMNIN - C3_W_DMNIN + 1; r += STRIDE) {
                for (c = 0, i = 0, j = 0; c < C3_X_DMNIN - C3_W_DMNIN + 1; c += STRIDE) {
                    for (i = 0; i < C3_W_DMNIN; ++i) {
                        for (j = 0; j < C3_W_DMNIN; ++j) {
                        	out[r][c][f] += W[i][j][ch][f] * X[r + i][j + c][ch];
                        }
                    }
                }
            }
        }
    }
}

void convolution_c4 (
              DTYPE   X[C4_X_DMNIN][C4_X_DMNIN][C4_N_CHAN],
        const DTYPE   W[C4_W_DMNIN][C4_W_DMNIN][C4_N_CHAN][C4_N_FILTERS],
              DTYPE out[C4_OUT_DMNIN][C4_OUT_DMNIN][C4_N_FILTERS],
        const DTYPE bias[C4_N_FILTERS]) {

    uint8_t ch, f, i, j, r, c;

    for (f = 0; f < C4_N_FILTERS; ++f) {
        for (r = 0; r < C4_OUT_DMNIN; ++r) {
            for (c = 0; c < C4_OUT_DMNIN; ++c) {
                #pragma HLS PIPELINE
                out[r][c][f] = bias[f];
            }
        }
    }
    for (f = 0; f < C4_N_FILTERS; ++f) {
        for (ch = 0; ch < C4_N_CHAN; ++ch) {
            for (r = 0; r < C4_X_DMNIN - C4_W_DMNIN + 1; r += STRIDE) {
                for (c = 0, i = 0, j = 0; c < C4_X_DMNIN - C4_W_DMNIN + 1; c += STRIDE) {
                    for (i = 0; i < C4_W_DMNIN; ++i) {
                        for (j = 0; j < C4_W_DMNIN; ++j) {
                        	out[r][c][f] += W[i][j][ch][f] * X[r + i][j + c][ch];
                        }
                    }
                }
            }
        }
    }
}
