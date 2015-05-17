#include <stdint.h>
#include <arm_neon.h>


static const uint32x4_t sign_bits_f = { 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff };
static const uint32x4_t sign_bits_f_zero_l = { 0, 0x7fffffff, 0x7fffffff, 0x7fffffff };
static const float32x4_t ones_f = { 1.0f, 1.0f, 1.0f, 1.0f };
static const float32x4_t zeroes_f = { 0.0f, 0.0f, 0.0f, 0.0f };


// http://stackoverflow.com/questions/6759897/
static inline __attribute__((always_inline)) float32x4_t reciprocal(float32x4_t x) {
    float32x4_t recip = vrecpeq_f32(x);
    recip = vmulq_f32(vrecpsq_f32(x, recip), recip);
    recip = vmulq_f32(vrecpsq_f32(x, recip), recip);
    return recip;
}


void byte2word48_neon(const uint8_t *t, const int pitch, float *pf) {
    uint16_t *p = (uint16_t *)pf;

    uint8x8_t m0, m1, m2, m3, m4, m5;

    m0 = vld1_u8(t);
    m1 = vreinterpret_u8_u32(vld1_lane_u32((const uint32_t *)(t + 8), vreinterpret_u32_u8(m1), 0));
    m1 = vreinterpret_u8_u32(vld1_lane_u32((const uint32_t *)(t + pitch * 2), vreinterpret_u32_u8(m1), 1));
    m2 = vld1_u8(t + pitch * 2 + 4);

    t += pitch * 4;

    m3 = vld1_u8(t);
    m4 = vreinterpret_u8_u32(vld1_lane_u32((const uint32_t *)(t + 8), vreinterpret_u32_u8(m4), 0));
    m4 = vreinterpret_u8_u32(vld1_lane_u32((const uint32_t *)(t + pitch * 2), vreinterpret_u32_u8(m4), 1));
    m5 = vld1_u8(t + pitch * 2 + 4);

    vst1q_u16(p, vmovl_u8(m0));
    vst1q_u16(p + 8, vmovl_u8(m1));
    vst1q_u16(p + 16, vmovl_u8(m2));
    vst1q_u16(p + 24, vmovl_u8(m3));
    vst1q_u16(p + 32, vmovl_u8(m4));
    vst1q_u16(p + 40, vmovl_u8(m5));
}


void byte2word64_neon(const uint8_t *t, const int pitch, float *pf) {
    uint16_t *p = (uint16_t *)pf;

    vst1q_u16(p, vmovl_u8(vld1_u8(t)));
    vst1q_u16(p + 8, vmovl_u8(vld1_u8(t + 8)));
    vst1q_u16(p + 16, vmovl_u8(vld1_u8(t + pitch * 2)));
    vst1q_u16(p + 24, vmovl_u8(vld1_u8(t + pitch * 2 + 8)));
    vst1q_u16(p + 32, vmovl_u8(vld1_u8(t + pitch * 4)));
    vst1q_u16(p + 40, vmovl_u8(vld1_u8(t + pitch * 4 + 8)));
    vst1q_u16(p + 48, vmovl_u8(vld1_u8(t + pitch * 6)));
    vst1q_u16(p + 56, vmovl_u8(vld1_u8(t + pitch * 6 + 8)));
}


void byte2float48_neon(const uint8_t *t, const int pitch, float *p) {
    uint16x8_t m0, m1, m2, m3, m4, m5;
    uint32x2_t temp1, temp4;

    m0 = vmovl_u8(vld1_u8(t));
    temp1 = vld1_lane_u32((const uint32_t *)(t + 8), temp1, 0);
    temp1 = vld1_lane_u32((const uint32_t *)(t + pitch * 2), temp1, 1);
    m1 = vmovl_u8(vreinterpret_u8_u32(temp1));
    m2 = vmovl_u8(vld1_u8(t + pitch * 2 + 4));

    t += pitch * 4;

    m3 = vmovl_u8(vld1_u8(t));
    temp4 = vld1_lane_u32((const uint32_t *)(t + 8), temp4, 0);
    temp4 = vld1_lane_u32((const uint32_t *)(t + pitch * 2), temp4, 1);
    m4 = vmovl_u8(vreinterpret_u8_u32(temp4));
    m5 = vmovl_u8(vld1_u8(t + pitch * 2 + 4));

    vst1q_f32(p, vcvtq_f32_u32(vmovl_u16(vget_low_u16(m0))));
    vst1q_f32(p + 4, vcvtq_f32_u32(vmovl_u16(vget_high_u16(m0))));
    vst1q_f32(p + 8, vcvtq_f32_u32(vmovl_u16(vget_low_u16(m1))));
    vst1q_f32(p + 12, vcvtq_f32_u32(vmovl_u16(vget_high_u16(m1))));
    vst1q_f32(p + 16, vcvtq_f32_u32(vmovl_u16(vget_low_u16(m2))));
    vst1q_f32(p + 20, vcvtq_f32_u32(vmovl_u16(vget_high_u16(m2))));
    vst1q_f32(p + 24, vcvtq_f32_u32(vmovl_u16(vget_low_u16(m3))));
    vst1q_f32(p + 28, vcvtq_f32_u32(vmovl_u16(vget_high_u16(m3))));
    vst1q_f32(p + 32, vcvtq_f32_u32(vmovl_u16(vget_low_u16(m4))));
    vst1q_f32(p + 36, vcvtq_f32_u32(vmovl_u16(vget_high_u16(m4))));
    vst1q_f32(p + 40, vcvtq_f32_u32(vmovl_u16(vget_low_u16(m5))));
    vst1q_f32(p + 44, vcvtq_f32_u32(vmovl_u16(vget_high_u16(m5))));
}


void word2float48_neon(const uint8_t *t8, const int pitch, float *p) {
    const uint16_t *t = (const uint16_t *)t8;

    for (int i = 0; i < 4; i++) {
        uint32x4_t u1 = vmovl_u16(vld1_u16(t));
        uint32x4_t u2 = vmovl_u16(vld1_u16(t + 4));
        uint32x4_t u3 = vmovl_u16(vld1_u16(t + 8));

        float32x4_t f1 = vcvtq_f32_u32(u1);
        float32x4_t f2 = vcvtq_f32_u32(u2);
        float32x4_t f3 = vcvtq_f32_u32(u3);

        vst1q_f32(p, f1);
        vst1q_f32(p + 4, f2);
        vst1q_f32(p + 8, f3);

        t += pitch * 2; // it was already halved
        p += 12;
    }
}


void computeNetwork0_neon(const float *input, const float *weights, uint8_t *d) {
    float32x4_t m0 = { 0.0f, 0.0f, 0.0f, 0.0f };
    float32x4_t m1 = m0;
    float32x4_t m2 = m0;
    float32x4_t m3 = m0;

    float32x4_t m4, m5, m6, m7;

    for (int i = 0; i < 192/4; i += 4) {
        m4 = vld1q_f32(input + i);
        m5 = m4;
        m6 = m4;
        m7 = m4;

        m4 = vmulq_f32(m4, vld1q_f32(weights + i * 4));
        m5 = vmulq_f32(m5, vld1q_f32(weights + i * 4 + 4));
        m6 = vmulq_f32(m6, vld1q_f32(weights + i * 4 + 8));
        m7 = vmulq_f32(m7, vld1q_f32(weights + i * 4 + 12));

        m0 = vaddq_f32(m0, m4);
        m1 = vaddq_f32(m1, m5);
        m2 = vaddq_f32(m2, m6);
        m3 = vaddq_f32(m3, m7);
    }

    float32x2_t sum0 = vpadd_f32(vget_low_f32(m0), vget_high_f32(m0));
    float32x2_t sum1 = vpadd_f32(vget_low_f32(m1), vget_high_f32(m1));
    float32x2_t sum2 = vpadd_f32(vget_low_f32(m2), vget_high_f32(m2));
    float32x2_t sum3 = vpadd_f32(vget_low_f32(m3), vget_high_f32(m3));
    sum0 = vpadd_f32(sum0, sum1);
    sum1 = vpadd_f32(sum2, sum3);
    m0 = vcombine_f32(sum0, sum1);

    m0 = vaddq_f32(m0, vld1q_f32(weights + 768/4));

    m1 = m0;
    m0 = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(m0), sign_bits_f_zero_l));
    m0 = vaddq_f32(m0, ones_f);
    m0 = vmulq_f32(reciprocal(m0), m1);

    m1 = vdupq_lane_f32(vget_low_f32(m0), 0);
    m2 = vdupq_lane_f32(vget_low_f32(m0), 1);
    m3 = vdupq_lane_f32(vget_high_f32(m0), 0);
    m4 = vdupq_lane_f32(vget_high_f32(m0), 1);

    m1 = vmulq_f32(m1, vld1q_f32(weights + 784/4));
    m2 = vmulq_f32(m2, vld1q_f32(weights + (784+16)/4));
    m3 = vmulq_f32(m3, vld1q_f32(weights + (784+32)/4));
    m4 = vmulq_f32(m4, vld1q_f32(weights + (784+48)/4));

    m1 = vaddq_f32(m1, m2);
    m3 = vaddq_f32(m3, m4);
    m1 = vaddq_f32(m1, m3);
    m1 = vaddq_f32(m1, vld1q_f32(weights + (784+64)/4));

    m7 = m1;
    m1 = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(m1), sign_bits_f));
    m1 = vaddq_f32(m1, ones_f);
    m7 = vmulq_f32(reciprocal(m1), m7);

    m3 = m0;

    m0 = vdupq_lane_f32(vget_low_f32(m0), 0);
    m1 = vdupq_lane_f32(vget_low_f32(m3), 1);
    m2 = vdupq_lane_f32(vget_high_f32(m3), 0);
    m3 = vdupq_lane_f32(vget_high_f32(m3), 1);

    m0 = vmulq_f32(m0, vld1q_f32(weights + 864/4));
    m1 = vmulq_f32(m1, vld1q_f32(weights + (864+16)/4));
    m2 = vmulq_f32(m2, vld1q_f32(weights + (864+32)/4));
    m3 = vmulq_f32(m3, vld1q_f32(weights + (864+48)/4));

    m4 = vdupq_lane_f32(vget_low_f32(m7), 0);
    m5 = vdupq_lane_f32(vget_low_f32(m7), 1);
    m6 = vdupq_lane_f32(vget_high_f32(m7), 0);
    m7 = vdupq_lane_f32(vget_high_f32(m7), 1);

    m4 = vmulq_f32(m4, vld1q_f32(weights + (864+64)/4));
    m5 = vmulq_f32(m5, vld1q_f32(weights + (864+80)/4));
    m6 = vmulq_f32(m6, vld1q_f32(weights + (864+96)/4));
    m7 = vmulq_f32(m7, vld1q_f32(weights + (864+112)/4));

    m0 = vaddq_f32(m0, m1);
    m2 = vaddq_f32(m2, m3);
    m4 = vaddq_f32(m4, m5);
    m6 = vaddq_f32(m6, m7);

    m0 = vaddq_f32(m0, m2);
    m4 = vaddq_f32(m4, m6);
    m0 = vaddq_f32(m0, m4);

    m0 = vaddq_f32(m0, vld1q_f32(weights + (864+128)/4));

    float32x2_t maximum = vmax_f32(vget_low_f32(m0), vget_high_f32(m0));
    d[0] = (vget_lane_f32(maximum, 1) <= vget_lane_f32(maximum, 0));
}


void computeNetwork0_i16_neon(const float *inputf, const float *weightsf, uint8_t *d) {
    const int16_t *input = (const int16_t *)inputf;
    const int16_t *weights = (const int16_t *)weightsf;

    int32x4_t accum0 = { 0, 0, 0, 0 };
    int32x4_t accum1 = accum0;
    int32x4_t accum2 = accum0;
    int32x4_t accum3 = accum0;

    for (int i = 0; i < 96/2; i += 8) {
        int16x4x2_t d0 = vld2_s16(input + i);

        int16x4x2_t w0 = vld2_s16(weights + i * 4);
        int16x4x2_t w1 = vld2_s16(weights + i * 4 + 8);
        int16x4x2_t w2 = vld2_s16(weights + i * 4 + 16);
        int16x4x2_t w3 = vld2_s16(weights + i * 4 + 24);

        accum0 = vmlal_s16(accum0, d0.val[0], w0.val[0]);
        accum0 = vmlal_s16(accum0, d0.val[1], w0.val[1]);

        accum1 = vmlal_s16(accum1, d0.val[0], w1.val[0]);
        accum1 = vmlal_s16(accum1, d0.val[1], w1.val[1]);

        accum2 = vmlal_s16(accum2, d0.val[0], w2.val[0]);
        accum2 = vmlal_s16(accum2, d0.val[1], w2.val[1]);

        accum3 = vmlal_s16(accum3, d0.val[0], w3.val[0]);
        accum3 = vmlal_s16(accum3, d0.val[1], w3.val[1]);
    }

    int32x2_t sum0 = vpadd_s32(vget_low_s32(accum0), vget_high_s32(accum0));
    int32x2_t sum1 = vpadd_s32(vget_low_s32(accum1), vget_high_s32(accum1));
    int32x2_t sum2 = vpadd_s32(vget_low_s32(accum2), vget_high_s32(accum2));
    int32x2_t sum3 = vpadd_s32(vget_low_s32(accum3), vget_high_s32(accum3));
    sum0 = vpadd_s32(sum0, sum1);
    sum1 = vpadd_s32(sum2, sum3);
    int32x4_t sum = vcombine_s32(sum0, sum1);

    float32x4_t m0 = vcvtq_f32_s32(sum);

    m0 = vmulq_f32(m0, vld1q_f32(weightsf + 384/4));
    m0 = vaddq_f32(m0, vld1q_f32(weightsf + 400/4));

    float32x4_t m1, m2, m3, m4, m5, m6, m7;

    m1 = m0;

    m0 = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(m0), sign_bits_f_zero_l));
    m0 = vaddq_f32(m0, ones_f);
    m0 = vmulq_f32(reciprocal(m0), m1);

    m1 = vdupq_lane_f32(vget_low_f32(m0), 0);
    m2 = vdupq_lane_f32(vget_low_f32(m0), 1);
    m3 = vdupq_lane_f32(vget_high_f32(m0), 0);
    m4 = vdupq_lane_f32(vget_high_f32(m0), 1);

    m1 = vmulq_f32(m1, vld1q_f32(weightsf + 416/4));
    m2 = vmulq_f32(m2, vld1q_f32(weightsf + (416+16)/4));
    m3 = vmulq_f32(m3, vld1q_f32(weightsf + (416+32)/4));
    m4 = vmulq_f32(m4, vld1q_f32(weightsf + (416+48)/4));

    m1 = vaddq_f32(m1, m2);
    m3 = vaddq_f32(m3, m4);
    m1 = vaddq_f32(m1, m3);
    m1 = vaddq_f32(m1, vld1q_f32(weightsf + (416+64)/4));

    m7 = m1;
    m1 = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(m1), sign_bits_f));
    m1 = vaddq_f32(m1, ones_f);
    m7 = vmulq_f32(reciprocal(m1), m7);

    m3 = m0;

    m0 = vdupq_lane_f32(vget_low_f32(m0), 0);
    m1 = vdupq_lane_f32(vget_low_f32(m3), 1);
    m2 = vdupq_lane_f32(vget_high_f32(m3), 0);
    m3 = vdupq_lane_f32(vget_high_f32(m3), 1);

    m0 = vmulq_f32(m0, vld1q_f32(weightsf + 496/4));
    m1 = vmulq_f32(m1, vld1q_f32(weightsf + (496+16)/4));
    m2 = vmulq_f32(m2, vld1q_f32(weightsf + (496+32)/4));
    m3 = vmulq_f32(m3, vld1q_f32(weightsf + (496+48)/4));

    m4 = vdupq_lane_f32(vget_low_f32(m7), 0);
    m5 = vdupq_lane_f32(vget_low_f32(m7), 1);
    m6 = vdupq_lane_f32(vget_high_f32(m7), 0);
    m7 = vdupq_lane_f32(vget_high_f32(m7), 1);

    m4 = vmulq_f32(m4, vld1q_f32(weightsf + (496+64)/4));
    m5 = vmulq_f32(m5, vld1q_f32(weightsf + (496+80)/4));
    m6 = vmulq_f32(m6, vld1q_f32(weightsf + (496+96)/4));
    m7 = vmulq_f32(m7, vld1q_f32(weightsf + (496+112)/4));

    m0 = vaddq_f32(m0, m1);
    m2 = vaddq_f32(m2, m3);
    m4 = vaddq_f32(m4, m5);
    m6 = vaddq_f32(m6, m7);

    m0 = vaddq_f32(m0, m2);
    m4 = vaddq_f32(m4, m6);
    m0 = vaddq_f32(m0, m4);

    m0 = vaddq_f32(m0, vld1q_f32(weightsf + (496+128)/4));

    float32x2_t maximum = vmax_f32(vget_low_f32(m0), vget_high_f32(m0));
    d[0] = (vget_lane_f32(maximum, 1) <= vget_lane_f32(maximum, 0));
}


void computeNetwork0new_neon(const float *dataf, const float *weightsf, uint8_t *d) {
    const int16_t *data = (const int16_t *)dataf;
    const int16_t *weights = (const int16_t *)weightsf;

    int32x4_t accum0 = { 0, 0, 0, 0 };
    int32x4_t accum1 = accum0;
    int32x4_t accum2 = accum0;
    int32x4_t accum3 = accum0;

    for (int i = 0; i < 128/2; i += 8) {
        int16x4x2_t d0 = vld2_s16(data + i);

        int16x4x2_t w0 = vld2_s16(weights + i * 4);
        int16x4x2_t w1 = vld2_s16(weights + i * 4 + 8);
        int16x4x2_t w2 = vld2_s16(weights + i * 4 + 16);
        int16x4x2_t w3 = vld2_s16(weights + i * 4 + 24);

        accum0 = vmlal_s16(accum0, d0.val[0], w0.val[0]);
        accum0 = vmlal_s16(accum0, d0.val[1], w0.val[1]);

        accum1 = vmlal_s16(accum1, d0.val[0], w1.val[0]);
        accum1 = vmlal_s16(accum1, d0.val[1], w1.val[1]);

        accum2 = vmlal_s16(accum2, d0.val[0], w2.val[0]);
        accum2 = vmlal_s16(accum2, d0.val[1], w2.val[1]);

        accum3 = vmlal_s16(accum3, d0.val[0], w3.val[0]);
        accum3 = vmlal_s16(accum3, d0.val[1], w3.val[1]);
    }

    int32x2_t sum0 = vpadd_s32(vget_low_s32(accum0), vget_high_s32(accum0));
    int32x2_t sum1 = vpadd_s32(vget_low_s32(accum1), vget_high_s32(accum1));
    int32x2_t sum2 = vpadd_s32(vget_low_s32(accum2), vget_high_s32(accum2));
    int32x2_t sum3 = vpadd_s32(vget_low_s32(accum3), vget_high_s32(accum3));
    sum0 = vpadd_s32(sum0, sum1);
    sum1 = vpadd_s32(sum2, sum3);
    int32x4_t sum = vcombine_s32(sum0, sum1);

    float32x4_t m0 = vcvtq_f32_s32(sum);

    m0 = vmulq_f32(m0, vld1q_f32(weightsf + 512/4));
    m0 = vaddq_f32(m0, vld1q_f32(weightsf + 528/4));

    float32x4_t m1, m2, m3, m4;

    m1 = m0;

    m0 = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(m0), sign_bits_f));
    m0 = vaddq_f32(m0, ones_f);
    m0 = vmulq_f32(reciprocal(m0), m1);

    m1 = vdupq_lane_f32(vget_low_f32(m0), 0);
    m2 = vdupq_lane_f32(vget_low_f32(m0), 1);
    m3 = vdupq_lane_f32(vget_high_f32(m0), 0);
    m4 = vdupq_lane_f32(vget_high_f32(m0), 1);

    m1 = vmulq_f32(m1, vld1q_f32(weightsf + 544/4));
    m2 = vmulq_f32(m2, vld1q_f32(weightsf + 560/4));
    m3 = vmulq_f32(m3, vld1q_f32(weightsf + 576/4));
    m4 = vmulq_f32(m4, vld1q_f32(weightsf + 592/4));

    m1 = vaddq_f32(m1, m2);
    m3 = vaddq_f32(m3, m4);
    m1 = vaddq_f32(m1, m3);
    m1 = vaddq_f32(m1, vld1q_f32(weightsf + 608/4));

    uint32x4_t gte = vcgeq_f32(m1, zeroes_f);
    uint16x4_t gte_u16 = vmovn_u32(gte);
    uint8x8_t gte_u8 = vmovn_u16(vcombine_u16(gte_u16, vget_low_u16(vreinterpretq_u16_u32(sign_bits_f))));
    gte_u8 = vshr_n_u8(gte_u8, 7);
    vst1_lane_u32((uint32_t *)d, vreinterpret_u32_u8(gte_u8), 0);
}


void dotProd_neon(const float *data, const float *weights, float *vals, const int n, const int len, const float *istd) {
    for (int i = 0; i < n; i += 4) {
        float32x4_t accum0 = { 0.0f, 0.0f, 0.0f, 0.0f };
        float32x4_t accum1 = accum0;
        float32x4_t accum2 = accum0;
        float32x4_t accum3 = accum0;

        for (int j = 0; j < len; j += 4) {
            float32x4_t d0 = vld1q_f32(data + j);
            float32x4_t d1 = d0;
            float32x4_t d2 = d0;
            float32x4_t d3 = d0;

            float32x4_t w0 = vld1q_f32(weights);
            float32x4_t w1 = vld1q_f32(weights + 4);
            float32x4_t w2 = vld1q_f32(weights + 8);
            float32x4_t w3 = vld1q_f32(weights + 12);

            accum0 = vaddq_f32(accum0, vmulq_f32(d0, w0));
            accum1 = vaddq_f32(accum1, vmulq_f32(d1, w1));
            accum2 = vaddq_f32(accum2, vmulq_f32(d2, w2));
            accum3 = vaddq_f32(accum3, vmulq_f32(d3, w3));

            weights += 16;
        }

        float32x2_t sum0 = vpadd_f32(vget_low_f32(accum0), vget_high_f32(accum0));
        float32x2_t sum1 = vpadd_f32(vget_low_f32(accum1), vget_high_f32(accum1));
        float32x2_t sum2 = vpadd_f32(vget_low_f32(accum2), vget_high_f32(accum2));
        float32x2_t sum3 = vpadd_f32(vget_low_f32(accum3), vget_high_f32(accum3));
        sum0 = vpadd_f32(sum0, sum1);
        sum1 = vpadd_f32(sum2, sum3);
        float32x4_t sum = vcombine_f32(sum0, sum1);
        
        sum = vmulq_n_f32(sum, istd[0]);
        sum = vaddq_f32(sum, vld1q_f32(weights + n*len + i));
        vst1q_f32(vals + i, sum);
    }
}


void dotProd_i16_neon(const float *dataf, const float *weightsf, float *vals, const int n, const int len, const float *istd) {
    const int16_t *data = (const int16_t *)dataf;
    const int16_t *weights = (const int16_t *)weightsf;
    weightsf += n * len / 2; // sizeof(float) / sizeof(int16_t)

    for (int i = 0; i < n; i += 4) {
        int32x4_t accum0 = { 0, 0, 0, 0 };
        int32x4_t accum1 = accum0;
        int32x4_t accum2 = accum0;
        int32x4_t accum3 = accum0;

        for (int j = 0; j < len; j += 8) {
            int16x4x2_t d0 = vld2_s16(data + j);

            int16x4x2_t w0 = vld2_s16(weights);
            int16x4x2_t w1 = vld2_s16(weights + 8);
            int16x4x2_t w2 = vld2_s16(weights + 16);
            int16x4x2_t w3 = vld2_s16(weights + 24);

            accum0 = vmlal_s16(accum0, d0.val[0], w0.val[0]);
            accum0 = vmlal_s16(accum0, d0.val[1], w0.val[1]);

            accum1 = vmlal_s16(accum1, d0.val[0], w1.val[0]);
            accum1 = vmlal_s16(accum1, d0.val[1], w1.val[1]);

            accum2 = vmlal_s16(accum2, d0.val[0], w2.val[0]);
            accum2 = vmlal_s16(accum2, d0.val[1], w2.val[1]);

            accum3 = vmlal_s16(accum3, d0.val[0], w3.val[0]);
            accum3 = vmlal_s16(accum3, d0.val[1], w3.val[1]);

            weights += 32;
        }

        int32x2_t sum0 = vpadd_s32(vget_low_s32(accum0), vget_high_s32(accum0));
        int32x2_t sum1 = vpadd_s32(vget_low_s32(accum1), vget_high_s32(accum1));
        int32x2_t sum2 = vpadd_s32(vget_low_s32(accum2), vget_high_s32(accum2));
        int32x2_t sum3 = vpadd_s32(vget_low_s32(accum3), vget_high_s32(accum3));
        sum0 = vpadd_s32(sum0, sum1);
        sum1 = vpadd_s32(sum2, sum3);
        int32x4_t sum = vcombine_s32(sum0, sum1);

        float32x4_t val = vcvtq_f32_s32(sum);
        val = vmulq_f32(val, vld1q_f32(weightsf + i*2));
        val = vmulq_n_f32(val, istd[0]);
        val = vaddq_f32(val, vld1q_f32(weightsf + i*2 + 4));
        vst1q_f32(vals + i, val);
    }
}
