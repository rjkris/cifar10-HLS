#include "conv_net.h"
#include "weights.h"
#include "biases.h"
#include <stdio.h>
#include <string.h>


void xillybus_wrapper(uint8_t *in_b, uint8_t *in_g, uint8_t *in_r,DTYPE *out_t) {
#pragma HLS INTERFACE s_axilite register port=in_b bundle=ctl
#pragma HLS INTERFACE m_axi depth=512 port=in_b offset=slave bundle=b
#pragma HLS INTERFACE s_axilite register port=in_g bundle=ctl
#pragma HLS INTERFACE m_axi depth=512 port=in_g offset=slave bundle=g
#pragma HLS INTERFACE s_axilite register port=in_r bundle=ctl
#pragma HLS INTERFACE m_axi depth=512 port=in_r offset=slave bundle=r
#pragma HLS INTERFACE s_axilite register port=out_t bundle=ctl
#pragma HLS INTERFACE m_axi depth=512 port=out_t offset=slave bundle=out
#pragma HLS INTERFACE s_axilite register port=return bundle=ctl

       uint16_t c,r,x,i;
       DTYPE img[32][32][3];
       uint8_t in_b_buffer[1024];//in_g_buffer[200704],in_r_buffer[200704];
       DTYPE p[1];

       for (i=0;i<3;i++)
       {
    	   if(i==0){
    	x=0;
       	memcpy(in_b_buffer,in_b,1024*sizeof(uint8_t));
       //////////////////////////////////////////////
		for (r = 0; r < 32; r++) {
	                   for (c = 0; c < 32; c++) {
						   img[r][c][2] = in_b_buffer[x];
						   x++;

	                   }
	               }}

    	   if(i==1){
    		   x=0;
       	memcpy(in_b_buffer,in_g,1024*sizeof(uint8_t));
       //////////////////////////////////////////////
		for (r = 0; r < 32; r++) {
	                   for (c = 0; c < 32; c++) {
						   img[r][c][1] = in_b_buffer[x];
						 x++;

	                   }
	               }}
    	   if(i==2){
    		   x=0;
       	memcpy(in_b_buffer,in_r,1024*sizeof(uint8_t));
       //////////////////////////////////////////////
		for (r = 0; r < 32; r++) {
	                   for (c = 0; c < 32; c++) {
						   img[r][c][0] = in_b_buffer[x];
						  x++;

	                   }
	               }}
       }
     predict(img,p);
     out_t[0]=p[0];

}

void predict(DTYPE img[IMG_DMNIN][IMG_DMNIN][IMG_CHANNELS],DTYPE p[1]) {

    int r, c;
    DTYPE layer1_out[C1_OUT_DMNIN][C1_OUT_DMNIN][C1_N_FILTERS];
    DTYPE layer2_out[A1_ROWS][A1_COLS][C1_N_FILTERS];
    DTYPE layer3_out[C2_OUT_DMNIN][C2_OUT_DMNIN][C2_N_FILTERS];
    DTYPE layer4_out[A2_ROWS][A2_COLS][C2_N_FILTERS];
    DTYPE layer5_out[P1_DOWNSIZE][P1_DOWNSIZE][C2_N_FILTERS];

    DTYPE layer6_out[C3_OUT_DMNIN][C3_OUT_DMNIN][C3_N_FILTERS];
    DTYPE layer7_out[A3_ROWS][A3_COLS][C3_N_FILTERS];
    DTYPE layer8_out[C4_OUT_DMNIN][C4_OUT_DMNIN][C4_N_FILTERS];
    DTYPE layer9_out[A4_ROWS][A4_COLS][C4_N_FILTERS];
    DTYPE layer10_out[P2_DOWNSIZE][P2_DOWNSIZE][C4_N_FILTERS];

    DTYPE layer11_out[FLAT_VEC_SZ];
    DTYPE layer12_out[F1_ROWS];
    DTYPE layer13_out[F1_ROWS];
    DTYPE layer14_out[F2_ROWS];
    DTYPE layer15_out[F2_ROWS];
    DTYPE layer16_out[F3_ROWS];


    convolution_c1(img, weights_C1, layer1_out, biases_C1);
    relu_a1(layer1_out, layer2_out);

    convolution_c2(layer2_out, weights_C2, layer3_out, biases_C2);
    relu_a2(layer3_out, layer4_out);
    pooling_p1(layer4_out, layer5_out);

    convolution_c3(layer5_out, weights_C3, layer6_out, biases_C3);
    relu_a3(layer6_out, layer7_out);

    convolution_c4(layer7_out, weights_C4, layer8_out, biases_C4);
    relu_a4(layer8_out, layer9_out);
    pooling_p2(layer9_out, layer10_out);

    flatten(layer10_out, layer11_out);

    vec_mat_mul_f1(layer11_out, weights_F1, biases_F1, layer12_out);
    relu_a5(layer12_out, layer13_out);

    vec_mat_mul_f2(layer13_out, weights_F2, biases_F2, layer14_out);
    relu_a6(layer14_out, layer15_out);

    vec_mat_mul_f3(layer15_out, weights_F3, biases_F3, layer16_out);

    softmax(layer16_out, p);
}


