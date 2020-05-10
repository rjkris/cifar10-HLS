#ifndef CONV_NET_H
#define CONV_NET_H


#include <stdint.h>

typedef int32_t DTYPE;


const uint8_t IMG_DMNIN = 32;
const uint8_t IMG_CHANNELS = 3;
const uint8_t STRIDE = 1;

// conv1
const uint8_t C1_N_CHAN = 3;//输入通道数
const uint8_t C1_X_DMNIN = 32;//图片大小
const uint8_t C1_W_DMNIN = 5;//5*5卷积核
const uint8_t C1_OUT_DMNIN = 28;
const uint8_t C1_N_FILTERS = 32;//卷积层filter

// relu1
const uint16_t A1_ROWS = 28;
const uint8_t A1_COLS = 28;

// conv2
const uint8_t C2_N_CHAN = 32;
const uint8_t C2_X_DMNIN = 28;
const uint8_t C2_W_DMNIN = 5;
const uint8_t C2_OUT_DMNIN = 24;
const uint8_t C2_N_FILTERS = 32;

// relu2
const uint16_t A2_ROWS = 24;
const uint8_t A2_COLS = 24;

// maxpool1
const uint8_t P1_SIZE = 24;
const uint8_t P1_DOWNSIZE = 12;

// conv3
const uint8_t C3_N_CHAN = 32;//输入通道数
const uint8_t C3_X_DMNIN = 12;//图片大小
const uint8_t C3_W_DMNIN = 3;//5*5卷积核
const uint8_t C3_OUT_DMNIN = 10;
const uint8_t C3_N_FILTERS = 32;//卷积层filter

// relu3
const uint16_t A3_ROWS = 10;
const uint8_t A3_COLS = 10;

// conv4
const uint8_t C4_N_CHAN = 32;
const uint8_t C4_X_DMNIN = 10;
const uint8_t C4_W_DMNIN = 3;
const uint8_t C4_OUT_DMNIN = 8;
const uint8_t C4_N_FILTERS = 32;

// relu4
const uint16_t A4_ROWS = 8;
const uint8_t A4_COLS = 8;

// maxpool2
const uint8_t P2_SIZE = 8;
const uint8_t P2_DOWNSIZE = 4;

// flatten
const uint16_t FLAT_VEC_SZ = 512;

//dense1
const uint16_t F1_ROWS = 256;
const uint16_t F1_COLS = 512;

//dense2
const uint8_t F2_ROWS = 128;
const uint16_t F2_COLS = 256;

//dense3
const uint8_t F3_ROWS = 10;
const uint8_t F3_COLS = 128;

//softmax
const uint8_t SFMX_SIZE = 10;
//const uint16_t SFMX_RES = 400;
//const uint32_t total = 921600;

int CNN(int in_r);

void xillybus_wrapper(uint8_t *in_b, uint8_t *in_g, uint8_t *in_r,DTYPE out_t[1]);

DTYPE maxFour(DTYPE a, DTYPE b, DTYPE c, DTYPE d);
uint16_t averFour(uint16_t a, uint16_t b, uint16_t c, uint16_t d);

//void pooling_1(uint16_t in[224][224][3], uint16_t out[112][112][3]);
void pooling_1(uint16_t in[112][112][3], uint16_t out[56][56][3]);
void pooling_2(uint16_t in[56][56][3], DTYPE out[28][28][3]);

void predict(DTYPE img[IMG_DMNIN][IMG_DMNIN][IMG_CHANNELS], DTYPE p[SFMX_SIZE]);


void convolution_c1 (
		      DTYPE    X[C1_X_DMNIN][C1_X_DMNIN][C1_N_CHAN],
        const DTYPE    W[C1_W_DMNIN][C1_W_DMNIN][C1_N_CHAN][C1_N_FILTERS],
              DTYPE  out[C1_OUT_DMNIN][C1_OUT_DMNIN][C1_N_FILTERS],
        const DTYPE bias[C1_N_FILTERS]);

void relu_a1(
        DTYPE in[A1_ROWS][A1_COLS][C1_N_FILTERS],
        DTYPE out[A1_ROWS][A1_COLS][C1_N_FILTERS]);

void convolution_c2 (
		      DTYPE    X[C2_X_DMNIN][C2_X_DMNIN][C2_N_CHAN],
        const DTYPE    W[C2_W_DMNIN][C2_W_DMNIN][C2_N_CHAN][C2_N_FILTERS],
              DTYPE  out[C2_OUT_DMNIN][C2_OUT_DMNIN][C2_N_FILTERS],
        const DTYPE bias[C2_N_FILTERS]);

void relu_a2(
        DTYPE in[A2_ROWS][A2_COLS][C2_N_FILTERS],
        DTYPE out[A2_ROWS][A2_COLS][C2_N_FILTERS]);

void pooling_p1(DTYPE in[P1_SIZE][P1_SIZE][C2_N_FILTERS],
        DTYPE out[P1_DOWNSIZE][P1_DOWNSIZE][C2_N_FILTERS]);

void convolution_c3 (
		      DTYPE    X[C3_X_DMNIN][C3_X_DMNIN][C3_N_CHAN],
        const DTYPE    W[C3_W_DMNIN][C3_W_DMNIN][C3_N_CHAN][C3_N_FILTERS],
              DTYPE  out[C3_OUT_DMNIN][C3_OUT_DMNIN][C3_N_FILTERS],
        const DTYPE bias[C3_N_FILTERS]);

void relu_a3(
        DTYPE in[A3_ROWS][A3_COLS][C3_N_FILTERS],
        DTYPE out[A3_ROWS][A3_COLS][C3_N_FILTERS]);

void convolution_c4 (
		      DTYPE    X[C4_X_DMNIN][C4_X_DMNIN][C4_N_CHAN],
        const DTYPE    W[C4_W_DMNIN][C4_W_DMNIN][C4_N_CHAN][C4_N_FILTERS],
              DTYPE  out[C4_OUT_DMNIN][C4_OUT_DMNIN][C4_N_FILTERS],
        const DTYPE bias[C4_N_FILTERS]);

void relu_a4(
        DTYPE in[A4_ROWS][A4_COLS][C4_N_FILTERS],
        DTYPE out[A4_ROWS][A4_COLS][C4_N_FILTERS]);

void pooling_p2(DTYPE in[P2_SIZE][P2_SIZE][C4_N_FILTERS],
        DTYPE out[P2_DOWNSIZE][P2_DOWNSIZE][C4_N_FILTERS]);

void flatten(
        DTYPE IN[P2_DOWNSIZE][P2_DOWNSIZE][C4_N_FILTERS],
        DTYPE OUT[FLAT_VEC_SZ]);

void vec_mat_mul_f1(
              DTYPE X[FLAT_VEC_SZ],
        const DTYPE W[F1_COLS][F1_ROWS],
        const DTYPE bias[F1_ROWS],
              DTYPE Z[F1_ROWS]);

void relu_a5(DTYPE in[F1_ROWS], DTYPE out[F1_ROWS]);

void vec_mat_mul_f2(
          DTYPE X[F1_ROWS],
    const DTYPE W[F2_COLS][F2_ROWS],
    const DTYPE bias[F2_ROWS],
          DTYPE Z[F2_ROWS]);

void relu_a6(DTYPE in[F2_ROWS], DTYPE out[F2_ROWS]);

void vec_mat_mul_f3(
          DTYPE X[F2_ROWS],
    const DTYPE W[F3_COLS][F3_ROWS],
    const DTYPE bias[F3_ROWS],
          DTYPE Z[F3_ROWS]);

void softmax(DTYPE Z[SFMX_SIZE], DTYPE p[1]);

#endif
