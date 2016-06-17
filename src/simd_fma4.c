
#include "simd_x86.h"


// With -ffp-contract=fast the compiler (gcc, clang) will turn eligible
// pairs of mul+add into fused multiply-add instructions. No extra code
// needed.


void nnedi3_computeNetwork0_FMA4(const float *input, const float *weights, uint8_t *d) {
    nnedi3_computeNetwork0(input, weights, d);
}


void nnedi3_e0_m16_FMA4(float *s, const intptr_t n) {
    nnedi3_e0_m16(s, n);
}


void nnedi3_dotProd_FMA4(const float *data, const float *weights, float *vals, const intptr_t n, const intptr_t len, const float *istd) {
    nnedi3_dotProd(data, weights, vals, n, len, istd);
}
