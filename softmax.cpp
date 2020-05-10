#include "conv_net.h"
#include <stdint.h>

void softmax(DTYPE Z[SFMX_SIZE], DTYPE p[1]) {

    uint8_t i,k=0;
    int32_t idx[SFMX_SIZE];
    DTYPE max=Z[0];

    for (i = 0; i < SFMX_SIZE; ++i) {
    	if(Z[i]>max) { k = i; max=Z[i];}
    }
/*    DTYPE denom = 0;
    for (i = 0; i < SFMX_SIZE; ++i) {
        idx[i] = (SFMX_RES>>1) + (int)(Z[i] * 10);
        denom += expZ[idx[i]];
    }*/

    p[0] = k;
}
