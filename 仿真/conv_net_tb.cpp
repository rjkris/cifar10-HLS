#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include<iostream>
#include "conv_net.h"
using namespace std;



int main (){
	uint8_t *in_b;
	uint8_t *in_g;
	uint8_t *in_r;

	int r,g,b;
	uint8_t inb[32*32];
	uint8_t ing[32*32];
	uint8_t inr[32*32];

	uint8_t tempb,tempg,tempr;
	FILE *fpb;
	FILE *fpg;
	FILE *fpr;
	DTYPE out_t[1];
	//DTYPE  test[1];
	fpb = fopen("truck_b.txt", "r");
	fpg = fopen("truck_g.txt", "r");
	fpr = fopen("truck_r.txt", "r");
    for(b=0; b <32*32; b++) {
        fscanf(fpb, "%d\n", &tempb);
        inb[b] = tempb;
    }
    fclose(fpb);
    for(g=0; g < 32*32; g++) {
        fscanf(fpg, "%d\n", &tempg);
        ing[g] = tempg;
    }
    fclose(fpg);
    for(r=0; r < 32*32; r++) {
        fscanf(fpr, "%d\n", &tempr);
        inr[r] = tempr;
    }
    fclose(fpr);
    in_b=inb;
    in_r=inr;
    in_g=ing;

    xillybus_wrapper(in_b,in_g,in_r,out_t);
    printf ("%d",out_t[0]);


    return 0 ;





}
  

