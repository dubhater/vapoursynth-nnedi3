#include <float.h>
#include <stdint.h>
#include <string.h>
#include <emmintrin.h>

#include "simd_x86.h"


void nnedi3_byte2float48_SSE2(const uint8_t *t, const intptr_t pitch, float *p) {
    __m128i zero = _mm_setzero_si128();

    __m128i m0, m1, m2, m3, m4, m5;

    m0 = _mm_loadl_epi64((const __m128i *)t);
    m4 = _mm_cvtsi32_si128(*(const int *)(t + 8));
    m2 = _mm_loadl_epi64((const __m128i *)(t + pitch * 2));
    m5 = _mm_cvtsi32_si128(*(const int *)(t + pitch * 2 + 8));

    m0 = _mm_unpacklo_epi8(m0, zero);
    m4 = _mm_unpacklo_epi8(m4, zero);
    m2 = _mm_unpacklo_epi8(m2, zero);
    m5 = _mm_unpacklo_epi8(m5, zero);

    m1 = m0;
    m3 = m2;

    m4 = _mm_unpacklo_epi8(m4, zero);
    m5 = _mm_unpacklo_epi8(m5, zero);

    m0 = _mm_unpacklo_epi8(m0, zero);
    m1 = _mm_unpackhi_epi8(m1, zero);
    m2 = _mm_unpacklo_epi8(m2, zero);
    m3 = _mm_unpackhi_epi8(m3, zero);

    _mm_store_ps(p, _mm_cvtepi32_ps(m0));
    _mm_store_ps(p + 4, _mm_cvtepi32_ps(m1));
    _mm_store_ps(p + 8, _mm_cvtepi32_ps(m4));
    _mm_store_ps(p + 12, _mm_cvtepi32_ps(m2));
    _mm_store_ps(p + 16, _mm_cvtepi32_ps(m3));
    _mm_store_ps(p + 20, _mm_cvtepi32_ps(m5));

    t += pitch * 4;
    p += 24;

    m0 = _mm_loadl_epi64((const __m128i *)t);
    m4 = _mm_cvtsi32_si128(*(const int *)(t + 8));
    m2 = _mm_loadl_epi64((const __m128i *)(t + pitch * 2));
    m5 = _mm_cvtsi32_si128(*(const int *)(t + pitch * 2 + 8));

    m0 = _mm_unpacklo_epi8(m0, zero);
    m4 = _mm_unpacklo_epi8(m4, zero);
    m2 = _mm_unpacklo_epi8(m2, zero);
    m5 = _mm_unpacklo_epi8(m5, zero);

    m1 = m0;
    m3 = m2;

    m4 = _mm_unpacklo_epi8(m4, zero);
    m5 = _mm_unpacklo_epi8(m5, zero);

    m0 = _mm_unpacklo_epi8(m0, zero);
    m1 = _mm_unpackhi_epi8(m1, zero);
    m2 = _mm_unpacklo_epi8(m2, zero);
    m3 = _mm_unpackhi_epi8(m3, zero);

    _mm_store_ps(p, _mm_cvtepi32_ps(m0));
    _mm_store_ps(p + 4, _mm_cvtepi32_ps(m1));
    _mm_store_ps(p + 8, _mm_cvtepi32_ps(m4));
    _mm_store_ps(p + 12, _mm_cvtepi32_ps(m2));
    _mm_store_ps(p + 16, _mm_cvtepi32_ps(m3));
    _mm_store_ps(p + 20, _mm_cvtepi32_ps(m5));
}


void nnedi3_word2float48_SSE2(const uint8_t *t, const intptr_t pitch, float *pf) {
    __m128i zero = _mm_setzero_si128();

    for (int i = 0; i < 4; i++) {
        __m128i m1 = _mm_loadl_epi64((const __m128i *)t);
        __m128i m2 = _mm_loadl_epi64((const __m128i *)(t + 8));
        __m128i m3 = _mm_loadl_epi64((const __m128i *)(t + 16));

        m1 = _mm_unpacklo_epi16(m1, zero);
        m2 = _mm_unpacklo_epi16(m2, zero);
        m3 = _mm_unpacklo_epi16(m3, zero);

        _mm_store_ps(pf, _mm_cvtepi32_ps(m1));
        _mm_store_ps(pf + 4, _mm_cvtepi32_ps(m2));
        _mm_store_ps(pf + 8, _mm_cvtepi32_ps(m3));

        pf += 12;
        t += pitch * 4;
    }
}


void nnedi3_byte2word48_SSE2(const uint8_t *t, const intptr_t pitch, float *pf) {
    uint8_t *p = (uint8_t *)pf;

    __m128i zero = _mm_setzero_si128();

    __m128i m0 = _mm_loadl_epi64((const __m128i *)t);
    __m128i m1 = _mm_cvtsi32_si128(*(const int *)(t + 8));
    __m128i m2 = _mm_cvtsi32_si128(*(const int *)(t + pitch * 2));
    __m128i m3 = _mm_loadl_epi64((const __m128i *)(t + pitch * 2 + 4));
    __m128i m4 = _mm_loadl_epi64((const __m128i *)(t + pitch * 4));
    __m128i m5 = _mm_cvtsi32_si128(*(const int *)(t + pitch * 4 + 8));
    __m128i m6 = _mm_cvtsi32_si128(*(const int *)(t + pitch * 6));
    __m128i m7 = _mm_loadl_epi64((const __m128i *)(t + pitch * 6 + 4));

    m1 = _mm_unpacklo_epi32(m1, m2);
    m5 = _mm_unpacklo_epi32(m5, m6);

    m0 = _mm_unpacklo_epi8(m0, zero);
    m1 = _mm_unpacklo_epi8(m1, zero);
    m3 = _mm_unpacklo_epi8(m3, zero);
    m4 = _mm_unpacklo_epi8(m4, zero);
    m5 = _mm_unpacklo_epi8(m5, zero);
    m7 = _mm_unpacklo_epi8(m7, zero);

    _mm_store_si128((__m128i *)p, m0);
    _mm_store_si128((__m128i *)(p + 16), m1);
    _mm_store_si128((__m128i *)(p + 32), m3);
    _mm_store_si128((__m128i *)(p + 48), m4);
    _mm_store_si128((__m128i *)(p + 64), m5);
    _mm_store_si128((__m128i *)(p + 80), m7);
}


void nnedi3_byte2word64_SSE2(const uint8_t *t, const intptr_t pitch, float *pf) {
    uint8_t *p = (uint8_t *)pf;

    __m128i zero = _mm_setzero_si128();

    __m128i m0 = _mm_loadl_epi64((const __m128i *)t);
    __m128i m1 = _mm_loadl_epi64((const __m128i *)(t + 8));
    __m128i m2 = _mm_loadl_epi64((const __m128i *)(t + pitch * 2));
    __m128i m3 = _mm_loadl_epi64((const __m128i *)(t + pitch * 2 + 8));
    __m128i m4 = _mm_loadl_epi64((const __m128i *)(t + pitch * 4));
    __m128i m5 = _mm_loadl_epi64((const __m128i *)(t + pitch * 4 + 8));
    __m128i m6 = _mm_loadl_epi64((const __m128i *)(t + pitch * 6));
    __m128i m7 = _mm_loadl_epi64((const __m128i *)(t + pitch * 6 + 8));

    m0 = _mm_unpacklo_epi8(m0, zero);
    m1 = _mm_unpacklo_epi8(m1, zero);
    m2 = _mm_unpacklo_epi8(m2, zero);
    m3 = _mm_unpacklo_epi8(m3, zero);
    m4 = _mm_unpacklo_epi8(m4, zero);
    m5 = _mm_unpacklo_epi8(m5, zero);
    m6 = _mm_unpacklo_epi8(m6, zero);
    m7 = _mm_unpacklo_epi8(m7, zero);

    _mm_store_si128((__m128i *)p, m0);
    _mm_store_si128((__m128i *)(p + 16), m1);
    _mm_store_si128((__m128i *)(p + 32), m2);
    _mm_store_si128((__m128i *)(p + 48), m3);
    _mm_store_si128((__m128i *)(p + 64), m4);
    _mm_store_si128((__m128i *)(p + 80), m5);
    _mm_store_si128((__m128i *)(p + 96), m6);
    _mm_store_si128((__m128i *)(p + 112), m7);
}


int32_t nnedi3_processLine0_SSE2(const uint8_t *tempu, intptr_t width, uint8_t *dstp, const uint8_t *src3p, const intptr_t src_pitch) {
    __m128i zero = _mm_setzero_si128();

    __m128i word_19 = _mm_set1_epi16(19);
    __m128i word_3 = _mm_set1_epi16(3);
    __m128i byte_1 = _mm_set1_epi8(1);
    __m128i word_16 = _mm_set1_epi16(16);
    __m128i word_254 = _mm_set1_epi16(254);
    __m128i byte_255 = _mm_set1_epi8(255);

    __m128i accum = _mm_setzero_si128();

    for (int i = 0; i < width; i += 16) {
        __m128i m0 = _mm_load_si128((const __m128i *)(src3p + src_pitch * 2));
        __m128i m1 = _mm_load_si128((const __m128i *)(src3p + src_pitch * 4));
        __m128i m2 = m0;
        __m128i m3 = m1;

        m0 = _mm_unpacklo_epi8(m0, zero);
        m2 = _mm_unpackhi_epi8(m2, zero);

        m1 = _mm_unpacklo_epi8(m1, zero);
        m3 = _mm_unpackhi_epi8(m3, zero);

        m0 = _mm_add_epi16(m0, m1);
        m2 = _mm_add_epi16(m2, m3);

        m0 = _mm_mullo_epi16(m0, word_19);
        m2 = _mm_mullo_epi16(m2, word_19);

        __m128i m4 = _mm_load_si128((const __m128i *)(src3p));
        __m128i m5 = _mm_load_si128((const __m128i *)(src3p + src_pitch * 6));
        __m128i m6 = m4;
        __m128i m7 = m5;

        m4 = _mm_unpacklo_epi8(m4, zero);
        m6 = _mm_unpackhi_epi8(m6, zero);

        m5 = _mm_unpacklo_epi8(m5, zero);
        m7 = _mm_unpackhi_epi8(m7, zero);

        m4 = _mm_add_epi16(m4, m5);
        m6 = _mm_add_epi16(m6, m7);

        m4 = _mm_mullo_epi16(m4, word_3);
        m6 = _mm_mullo_epi16(m6, word_3);

        m0 = _mm_subs_epu16(m0, m4);
        m2 = _mm_subs_epu16(m2, m6);

        m0 = _mm_adds_epu16(m0, word_16);
        m2 = _mm_adds_epu16(m2, word_16);

        m0 = _mm_srli_epi16(m0, 5);
        m2 = _mm_srli_epi16(m2, 5);

        m0 = _mm_min_epi16(m0, word_254);
        m2 = _mm_min_epi16(m2, word_254);

        m0 = _mm_packus_epi16(m0, m2);

        __m128i m8 = _mm_load_si128((const __m128i *)tempu);
        m8 = _mm_cmpeq_epi8(m8, byte_1);

        __m128i m9 = _mm_xor_si128(m8, byte_255);

        m0 = _mm_and_si128(m0, m8);
        m0 = _mm_or_si128(m0, m9);
        _mm_store_si128((__m128i *)dstp, m0);

        __m128i m10 = _mm_and_si128(m9, byte_1);
        m10 = _mm_sad_epu8(m10, zero);

        __m128i m11 = m10;

        m10 = _mm_srli_si128(m10, 8);

        m10 = _mm_adds_epu16(m10, m11);
        accum = _mm_adds_epu16(accum, m10);

        src3p += 16;
        tempu += 16;
        dstp += 16;
    }

    return _mm_cvtsi128_si32(accum);
}


void nnedi3_extract_m8_SSE2(const uint8_t *srcp, const intptr_t stride, const intptr_t xdia, const intptr_t ydia, float *mstd, float *input) {
    __m128 sum = _mm_setzero_ps();
    __m128 sumsq = _mm_setzero_ps();

    __m128i zero = _mm_setzero_si128();

    for (int y = 0; y < ydia; y += 2) {
        for (int x = 0; x < xdia; x += 8) {
            __m128i i0 = _mm_loadl_epi64((const __m128i *)(srcp + x));
            __m128i i2 = _mm_loadl_epi64((const __m128i *)(srcp + stride * 2 + x));

            i0 = _mm_unpacklo_epi8(i0, zero);
            i2 = _mm_unpacklo_epi8(i2, zero);
            __m128i i1 = i0;
            __m128i i3 = i2;
            i0 = _mm_unpacklo_epi16(i0, zero);
            i1 = _mm_unpackhi_epi16(i1, zero);
            i2 = _mm_unpacklo_epi16(i2, zero);
            i3 = _mm_unpackhi_epi16(i3, zero);

            __m128 m0 = _mm_cvtepi32_ps(i0);
            __m128 m1 = _mm_cvtepi32_ps(i1);
            __m128 m2 = _mm_cvtepi32_ps(i2);
            __m128 m3 = _mm_cvtepi32_ps(i3);

            _mm_store_ps(input + x, m0);
            _mm_store_ps(input + x + 4, m1);
            _mm_store_ps(input + xdia + x, m2);
            _mm_store_ps(input + xdia + x + 4, m3);

            sum = _mm_add_ps(sum, m0);
            sum = _mm_add_ps(sum, m1);
            sum = _mm_add_ps(sum, m2);
            sum = _mm_add_ps(sum, m3);

            m0 = _mm_mul_ps(m0, m0);
            m1 = _mm_mul_ps(m1, m1);
            m2 = _mm_mul_ps(m2, m2);
            m3 = _mm_mul_ps(m3, m3);

            m0 = _mm_add_ps(m0, m1);
            m2 = _mm_add_ps(m2, m3);
            sumsq = _mm_add_ps(sumsq, m0);
            sumsq = _mm_add_ps(sumsq, m2);
        }

        srcp += stride * 4;
        input += xdia * 2;
    }

    __m128 m0 = _mm_setzero_ps();
    __m128 m1 = _mm_setzero_ps();

    m0 = _mm_movehl_ps(m0, sum);
    m1 = _mm_movehl_ps(m1, sumsq);

    sum = _mm_add_ps(sum, m0);
    sumsq = _mm_add_ps(sumsq, m1);

    m0 = _mm_castsi128_ps(_mm_shufflelo_epi16(_mm_castps_si128(sum), 14));
    m1 = _mm_castsi128_ps(_mm_shufflelo_epi16(_mm_castps_si128(sumsq), 14));

    sum = _mm_add_ss(sum, m0);
    sumsq = _mm_add_ss(sumsq, m1);

    __m128 m7 = _mm_set_ss(xdia * ydia);
    m7 = _mm_rcp_ss(m7);

    sum = _mm_mul_ss(sum, m7);
    sumsq = _mm_mul_ss(sumsq, m7);

    _mm_store_ss(&mstd[0], sum);

    sum = _mm_mul_ss(sum, sum);
    sumsq = _mm_sub_ss(sumsq, sum);

    mstd[3] = 0.0f;

    if (_mm_comile_ss(sumsq, _mm_set_ss(FLT_EPSILON))) {
        mstd[1] = mstd[2] = 0.0f;
    } else {
        sumsq = _mm_rsqrt_ss(sumsq);
        sum = _mm_rcp_ss(sumsq);
        _mm_store_ss(&mstd[1], sum);
        _mm_store_ss(&mstd[2], sumsq);
    }
}


void nnedi3_extract_m8_i16_SSE2(const uint8_t *srcp, const intptr_t stride, const intptr_t xdia, const intptr_t ydia, float *mstd, float *inputf) {
    uint8_t *input = (uint8_t *)inputf;

    __m128i zero = _mm_setzero_si128();

    __m128i sum = _mm_setzero_si128();
    __m128i sumsq = _mm_setzero_si128();

    for (int y = 0; y < ydia; y += 2) {
        for (int x = 0; x < xdia; x += 8) {
            __m128i m0 = _mm_loadl_epi64((const __m128i *)(srcp + x));
            __m128i m1 = _mm_loadl_epi64((const __m128i *)(srcp + stride * 2 + x));

            __m128i m2 = m0;
            __m128i m3 = m1;

            m0 = _mm_unpacklo_epi8(m0, zero);
            m1 = _mm_unpacklo_epi8(m1, zero);

            m2 = _mm_sad_epu8(m2, zero);
            m3 = _mm_sad_epu8(m3, zero);

            _mm_store_si128((__m128i *)(input + x * 2), m0);
            _mm_store_si128((__m128i *)(input + xdia * 2 + x * 2), m1);

            m0 = _mm_madd_epi16(m0, m0);
            m1 = _mm_madd_epi16(m1, m1);

            sum = _mm_add_epi32(sum, m2);
            sum = _mm_add_epi32(sum, m3);

            sumsq = _mm_add_epi32(sumsq, m0);
            sumsq = _mm_add_epi32(sumsq, m1);
        }

        srcp += stride * 4;
        input += xdia * 4;
    }

    __m128i m4 = _mm_setzero_si128();
    m4 = _mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(m4), _mm_castsi128_ps(sumsq)));
    sumsq = _mm_add_epi32(sumsq, m4);
    m4 = _mm_shufflelo_epi16(sumsq, 14);
    sumsq = _mm_add_epi32(sumsq, m4);

    __m128 sumf = _mm_cvtepi32_ps(sum);
    __m128 sumsqf = _mm_cvtepi32_ps(sumsq);

    __m128 m7 = _mm_set_ss(xdia * ydia);
    m7 = _mm_rcp_ss(m7);

    sumf = _mm_mul_ss(sumf, m7);
    sumsqf = _mm_mul_ss(sumsqf, m7);

    _mm_store_ss(&mstd[0], sumf);

    sumf = _mm_mul_ss(sumf, sumf);
    sumsqf = _mm_sub_ss(sumsqf, sumf);

    mstd[3] = 0.0f;

    if (_mm_comile_ss(sumsqf, _mm_set_ss(FLT_EPSILON))) {
        mstd[1] = mstd[2] = 0.0f;
    } else {
        __m128 m5 = _mm_rsqrt_ss(sumsqf);
        __m128 m6 = _mm_rcp_ss(m5);
        _mm_store_ss(&mstd[1], m6);
        _mm_store_ss(&mstd[2], m5);
    }
}


void nnedi3_computeNetwork0_SSE2(const float *input, const float *weights, uint8_t *d) {
    nnedi3_computeNetwork0(input, weights, d);
}


void nnedi3_computeNetwork0_i16_SSE2(const float *inputf, const float *weightsf, uint8_t *d) {
    const uint8_t *input = (const uint8_t *)inputf;
    const uint8_t *weights = (const uint8_t *)weightsf;

    __m128i m0 = _mm_setzero_si128();
    __m128i m1 = _mm_setzero_si128();
    __m128i m2 = _mm_setzero_si128();
    __m128i m3 = _mm_setzero_si128();

    for (int i = 0; i < 96; i += 16) {
        __m128i m4, m5, m6, m7;

        m4 = m5 = m6 = m7 = _mm_load_si128((const __m128i *)(input + i));

        m4 = _mm_madd_epi16(m4, _mm_load_si128((const __m128i *)(weights + i * 4)));
        m5 = _mm_madd_epi16(m5, _mm_load_si128((const __m128i *)(weights + i * 4 + 16)));
        m6 = _mm_madd_epi16(m6, _mm_load_si128((const __m128i *)(weights + i * 4 + 32)));
        m7 = _mm_madd_epi16(m7, _mm_load_si128((const __m128i *)(weights + i * 4 + 48)));

        m0 = _mm_add_epi32(m0, m4);
        m1 = _mm_add_epi32(m1, m5);
        m2 = _mm_add_epi32(m2, m6);
        m3 = _mm_add_epi32(m3, m7);
    }

    __m128i m4 = m0;
    __m128i m5 = m2;

    m0 = _mm_unpacklo_epi64(m0, m1);
    m2 = _mm_unpacklo_epi64(m2, m3);
    m4 = _mm_unpackhi_epi64(m4, m1);
    m5 = _mm_unpackhi_epi64(m5, m3);

    m0 = _mm_add_epi32(m0, m4);
    m2 = _mm_add_epi32(m2, m5);

    __m128i m6 = m0;
    m0 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(m0), _mm_castsi128_ps(m2), 136));
    m6 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(m6), _mm_castsi128_ps(m2), 221));
    m0 = _mm_add_epi32(m0, m6);

    __m128 m8 = _mm_cvtepi32_ps(m0);
    m8 = _mm_mul_ps(m8, _mm_load_ps((const float *)(weights + 384)));
    m8 = _mm_add_ps(m8, _mm_load_ps((const float *)(weights + 400)));

    __m128 m9 = m8;

    m8 = _mm_and_ps(m8, sign_bits_f_zero_l);
    m8 = _mm_add_ps(m8, ones_f);
    m8 = _mm_rcp_ps(m8);
    m8 = _mm_mul_ps(m8, m9);

    __m128 m10, m11, m12;

    m9 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(m8), 0));
    m10 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(m8), 85));
    m11 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(m8), 170));
    m12 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(m8), 255));

    m9 = _mm_mul_ps(m9, _mm_load_ps((const float *)(weights + 416)));
    m10 = _mm_mul_ps(m10, _mm_load_ps((const float *)(weights + 416 + 16)));
    m11 = _mm_mul_ps(m11, _mm_load_ps((const float *)(weights + 416 + 32)));
    m12 = _mm_mul_ps(m12, _mm_load_ps((const float *)(weights + 416 + 48)));

    m9 = _mm_add_ps(m9, m10);
    m11 = _mm_add_ps(m11, m12);
    m9 = _mm_add_ps(m9, m11);
    m9 = _mm_add_ps(m9, _mm_load_ps((const float *)(weights + 416 + 64)));

    __m128 m13 = m9;
    m9 = _mm_and_ps(m9, sign_bits_f);
    m9 = _mm_add_ps(m9, ones_f);
    m9 = _mm_rcp_ps(m9);
    m13 = _mm_mul_ps(m13, m9);

    __m128 m14 = m8;
    m8 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(m14), 0));
    m9 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(m14), 85));
    m10 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(m14), 170));
    m11 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(m14), 255));

    m8 = _mm_mul_ps(m8, _mm_load_ps((const float *)(weights + 496)));
    m9 = _mm_mul_ps(m9, _mm_load_ps((const float *)(weights + 496 + 16)));
    m10 = _mm_mul_ps(m10, _mm_load_ps((const float *)(weights + 496 + 32)));
    m11 = _mm_mul_ps(m11, _mm_load_ps((const float *)(weights + 496 + 48)));

    __m128 m15, m16, m17;
    m14 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(m13), 0));
    m15 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(m13), 85));
    m16 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(m13), 170));
    m17 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(m13), 255));

    m14 = _mm_mul_ps(m14, _mm_load_ps((const float *)(weights + 496 + 64)));
    m15 = _mm_mul_ps(m15, _mm_load_ps((const float *)(weights + 496 + 80)));
    m16 = _mm_mul_ps(m16, _mm_load_ps((const float *)(weights + 496 + 96)));
    m17 = _mm_mul_ps(m17, _mm_load_ps((const float *)(weights + 496 + 112)));

    m8 = _mm_add_ps(m8, m9);
    m10 = _mm_add_ps(m10, m11);
    m14 = _mm_add_ps(m14, m15);
    m16 = _mm_add_ps(m16, m17);

    m8 = _mm_add_ps(m8, m10);
    m14 = _mm_add_ps(m14, m16);
    m8 = _mm_add_ps(m8, m14);
    m8 = _mm_add_ps(m8, _mm_load_ps((const float *)(weights + 496 + 128)));

    m9 = _mm_movehl_ps(m9, m8);
    m8 = _mm_max_ps(m8, m9);
    m9 = _mm_castsi128_ps(_mm_shufflelo_epi16(_mm_castps_si128(m8), 14));
    d[0] = _mm_comile_ss(m9, m8);
}


void nnedi3_computeNetwork0new_SSE2(const float *dataf, const float *weightsf, uint8_t *d) {
    const uint8_t *data = (const uint8_t *)dataf;
    const uint8_t *weights = (const uint8_t *)weightsf;

    __m128i m0 = _mm_setzero_si128();
    __m128i m1 = _mm_setzero_si128();
    __m128i m2 = _mm_setzero_si128();
    __m128i m3 = _mm_setzero_si128();

    for (int i = 0; i < 128; i += 16) {
        __m128i m4, m5, m6, m7;

        m4 = m5 = m6 = m7 = _mm_load_si128((const __m128i *)(data + i));

        m4 = _mm_madd_epi16(m4, _mm_load_si128((const __m128i *)(weights + i * 4)));
        m5 = _mm_madd_epi16(m5, _mm_load_si128((const __m128i *)(weights + i * 4 + 16)));
        m6 = _mm_madd_epi16(m6, _mm_load_si128((const __m128i *)(weights + i * 4 + 32)));
        m7 = _mm_madd_epi16(m7, _mm_load_si128((const __m128i *)(weights + i * 4 + 48)));

        m0 = _mm_add_epi32(m0, m4);
        m1 = _mm_add_epi32(m1, m5);
        m2 = _mm_add_epi32(m2, m6);
        m3 = _mm_add_epi32(m3, m7);
    }

    __m128i m4 = m0;
    __m128i m5 = m2;

    m0 = _mm_unpacklo_epi64(m0, m1);
    m2 = _mm_unpacklo_epi64(m2, m3);

    m4 = _mm_unpackhi_epi64(m4, m1);
    m5 = _mm_unpackhi_epi64(m5, m3);

    m0 = _mm_add_epi32(m0, m4);
    m2 = _mm_add_epi32(m2, m5);

    __m128i m6 = m0;

    m0 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(m0), _mm_castsi128_ps(m2), 136));
    m6 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(m6), _mm_castsi128_ps(m2), 221));
    m0 = _mm_add_epi32(m0, m6);

    __m128 m7 = _mm_cvtepi32_ps(m0);
    m7 = _mm_mul_ps(m7, _mm_load_ps((const float *)(weights + 512)));
    m7 = _mm_add_ps(m7, _mm_load_ps((const float *)(weights + 528)));

    __m128 m8 = m7;

    m7 = _mm_and_ps(m7, sign_bits_f);
    m7 = _mm_add_ps(m7, ones_f);

    m7 = _mm_rcp_ps(m7);
    m7 = _mm_mul_ps(m7, m8);

    __m128 m9 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(m7), 0));
    __m128 m10 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(m7), 85));
    __m128 m11 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(m7), 170));
    __m128 m12 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(m7), 255));

    m9 = _mm_mul_ps(m9, _mm_load_ps((const float *)(weights + 544)));
    m10 = _mm_mul_ps(m10, _mm_load_ps((const float *)(weights + 560)));
    m11 = _mm_mul_ps(m11, _mm_load_ps((const float *)(weights + 576)));
    m12 = _mm_mul_ps(m12, _mm_load_ps((const float *)(weights + 592)));

    m9 = _mm_add_ps(m9, m10);
    m11 = _mm_add_ps(m11, m12);
    m9 = _mm_add_ps(m9, m11);

    m9 = _mm_add_ps(m9, _mm_load_ps((const float *)(weights + 608)));

    __m128i m13 = _mm_castps_si128(_mm_cmplt_ps(m9, _mm_setzero_ps()));

    __m128i zero = _mm_setzero_si128();

    m13 = _mm_packs_epi32(m13, zero);
    m13 = _mm_packs_epi16(m13, zero);

    int result = _mm_cvtsi128_si32(m13);
    result = result ^ 0xffffffff;
    result = result & 0x01010101;
    memcpy(d, &result, 4);
}


void nnedi3_weightedAvgElliottMul5_m16_SSE2(const float *w, const intptr_t n, float *mstd) {
    __m128 wsum = _mm_setzero_ps();
    __m128 vsum = _mm_setzero_ps();

    for (int i = 0; i < n; i += 16) {
        __m128 m0 = _mm_load_ps(w + i);
        __m128 m1 = _mm_load_ps(w + i + 4);
        __m128 m2 = _mm_load_ps(w + n + i);
        __m128 m3 = _mm_load_ps(w + n + i + 4);

        wsum = _mm_add_ps(wsum, m0);
        wsum = _mm_add_ps(wsum, m1);

        __m128 m4 = m2;
        __m128 m5 = m3;

        m2 = _mm_and_ps(m2, sign_bits_f);
        m3 = _mm_and_ps(m3, sign_bits_f);

        m2 = _mm_add_ps(m2, ones_f);
        m3 = _mm_add_ps(m3, ones_f);

        m2 = _mm_rcp_ps(m2);
        m3 = _mm_rcp_ps(m3);

        m4 = _mm_mul_ps(m4, m2);
        m5 = _mm_mul_ps(m5, m3);

        m4 = _mm_mul_ps(m4, m0);
        m5 = _mm_mul_ps(m5, m1);

        vsum = _mm_add_ps(vsum, m4);
        vsum = _mm_add_ps(vsum, m5);


        m0 = _mm_load_ps(w + i + 8);
        m1 = _mm_load_ps(w + i + 12);
        m2 = _mm_load_ps(w + n + i + 8);
        m3 = _mm_load_ps(w + n + i + 12);

        wsum = _mm_add_ps(wsum, m0);
        wsum = _mm_add_ps(wsum, m1);

        m4 = m2;
        m5 = m3;

        m2 = _mm_and_ps(m2, sign_bits_f);
        m3 = _mm_and_ps(m3, sign_bits_f);

        m2 = _mm_add_ps(m2, ones_f);
        m3 = _mm_add_ps(m3, ones_f);

        m2 = _mm_rcp_ps(m2);
        m3 = _mm_rcp_ps(m3);

        m4 = _mm_mul_ps(m4, m2);
        m5 = _mm_mul_ps(m5, m3);

        m4 = _mm_mul_ps(m4, m0);
        m5 = _mm_mul_ps(m5, m1);

        vsum = _mm_add_ps(vsum, m4);
        vsum = _mm_add_ps(vsum, m5);
    }

    __m128 wsum_high = _mm_setzero_ps();
    __m128 vsum_high = _mm_setzero_ps();
    wsum_high = _mm_movehl_ps(wsum_high, wsum);
    vsum_high = _mm_movehl_ps(vsum_high, vsum);

    wsum = _mm_add_ps(wsum, wsum_high);
    vsum = _mm_add_ps(vsum, vsum_high);

    __m128 wsum_shuffled = _mm_castsi128_ps(_mm_shufflelo_epi16(_mm_castps_si128(wsum), 14));
    __m128 vsum_shuffled = _mm_castsi128_ps(_mm_shufflelo_epi16(_mm_castps_si128(vsum), 14));

    wsum = _mm_add_ss(wsum, wsum_shuffled);
    vsum = _mm_add_ss(vsum, vsum_shuffled);

    __m128 min_weight_sum = _mm_set_ss(1.0e-10f);

    if (_mm_comile_ss(wsum, min_weight_sum)) {
        mstd[3] += mstd[0];
    } else {
        vsum = _mm_mul_ss(vsum, _mm_set_ss(5.0f));
        wsum = _mm_rcp_ss(wsum);
        vsum = _mm_mul_ss(vsum, wsum);
        vsum = _mm_mul_ss(vsum, _mm_load_ss(&mstd[1]));
        vsum = _mm_add_ss(vsum, _mm_load_ss(&mstd[0]));
        vsum = _mm_add_ss(vsum, _mm_load_ss(&mstd[3]));
        _mm_store_ss(&mstd[3], vsum);
    }
}


void nnedi3_e0_m16_SSE2(float *s, const intptr_t n) {
    nnedi3_e0_m16(s, n);
}


void nnedi3_e1_m16_SSE2(float *s, const intptr_t n) {
    __m128 e1_scale = _mm_set1_ps(1.4426950409f); // 1/ln(s)
    __m128 e1_bias = _mm_set1_ps(12582912.0f); // 3 << 22
    __m128 e1_c0 = _mm_set1_ps(1.00035f);
    __m128 e1_c1 = _mm_set1_ps(0.701277797f);
    __m128 e1_c2 = _mm_set1_ps(0.237348593f);

    for (int i = 0; i < n; i += 8) {
        __m128 m0 = _mm_load_ps(s + i);
        __m128 m3 = _mm_load_ps(s + i + 4);

        m0 = _mm_min_ps(m0, exp_hi);
        m3 = _mm_min_ps(m3, exp_hi);
        m0 = _mm_max_ps(m0, exp_lo);
        m3 = _mm_max_ps(m3, exp_lo);
        m0 = _mm_mul_ps(m0, e1_scale);
        m3 = _mm_mul_ps(m3, e1_scale);

        __m128 m1 = m0;
        __m128 m4 = m3;

        m0 = _mm_add_ps(m0, e1_bias);
        m3 = _mm_add_ps(m3, e1_bias);

        __m128 m2 = m0;
        __m128 m5 = m3;

        m0 = _mm_sub_ps(m0, e1_bias);
        m3 = _mm_sub_ps(m3, e1_bias);

        m2 = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(m2), 23));
        m5 = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(m5), 23));

        m1 = _mm_sub_ps(m1, m0);
        m4 = _mm_sub_ps(m4, m3);

        m0 = m1;
        m3 = m4;

        m1 = _mm_mul_ps(m1, m1);
        m4 = _mm_mul_ps(m4, m4);

        m0 = _mm_mul_ps(m0, e1_c1);
        m3 = _mm_mul_ps(m3, e1_c1);

        m1 = _mm_mul_ps(m1, e1_c2);
        m4 = _mm_mul_ps(m4, e1_c2);

        m0 = _mm_add_ps(m0, e1_c0);
        m3 = _mm_add_ps(m3, e1_c0);

        m0 = _mm_add_ps(m0, m1);
        m3 = _mm_add_ps(m3, m4);

        m0 = _mm_castsi128_ps(_mm_add_epi32(_mm_castps_si128(m0), _mm_castps_si128(m2)));
        m3 = _mm_castsi128_ps(_mm_add_epi32(_mm_castps_si128(m3), _mm_castps_si128(m5)));

        _mm_store_ps(s + i, m0);
        _mm_store_ps(s + i + 4, m3);
    }
}


void nnedi3_e2_m16_SSE2(float *s, const intptr_t n) {
    __m128 am_0p5 = _mm_set1_ps(0.5f);
    __m128 am_1 = _mm_set1_ps(1.0f);
    __m128 exp_rln2 = _mm_set1_ps(1.442695041f);
    __m128 exp_p0 = _mm_set1_ps(1.261771931e-4f);
    __m128 exp_p1 = _mm_set1_ps(3.029944077e-2f);
    __m128 exp_q0 = _mm_set1_ps(3.001985051e-6f);
    __m128 exp_q1 = _mm_set1_ps(2.524483403e-3f);
    __m128 exp_q2 = _mm_set1_ps(2.272655482e-1f);
    __m128 exp_q3 = _mm_set1_ps(2.0f);
    __m128 exp_c1 = _mm_set1_ps(6.931457520e-1f);
    __m128 exp_c2 = _mm_set1_ps(1.428606820e-6f);
    __m128i epi32_1 = _mm_set1_epi32(1);
    __m128i epi32_0x7f = _mm_set1_epi32(0x7f);

    for (int i = 0; i < n; i += 4) {
        __m128 m0 = _mm_load_ps(s + i);
        m0 = _mm_min_ps(m0, exp_hi);
        m0 = _mm_max_ps(m0, exp_lo);

        __m128 m1 = exp_rln2;
        m1 = _mm_mul_ps(m1, m0);
        m1 = _mm_add_ps(m1, am_0p5);

        __m128 m2 = _mm_setzero_ps();
        m2 = _mm_cmpnlt_ps(m2, m1);
        m2 = _mm_castsi128_ps(_mm_and_si128(_mm_castps_si128(m2), epi32_1));

        m1 = _mm_castsi128_ps(_mm_cvttps_epi32(m1));
        m1 = _mm_castsi128_ps(_mm_sub_epi32(_mm_castps_si128(m1), _mm_castps_si128(m2)));

        __m128 m3 = _mm_cvtepi32_ps(_mm_castps_si128(m1));

        __m128 m4 = exp_c2;
        __m128 m5 = exp_c1;

        m4 = _mm_mul_ps(m4, m3);
        m5 = _mm_mul_ps(m5, m3);

        m0 = _mm_sub_ps(m0, m4);
        m0 = _mm_sub_ps(m0, m5);

        __m128 m6 = exp_q0;
        m4 = exp_p0;

        m1 = _mm_castsi128_ps(_mm_add_epi32(_mm_castps_si128(m1), epi32_0x7f));

        m2 = m0;

        m0 = _mm_mul_ps(m0, m0);
        m6 = _mm_mul_ps(m6, m0);
        m4 = _mm_mul_ps(m4, m0);

        m6 = _mm_add_ps(m6, exp_q1);
        m4 = _mm_add_ps(m4, exp_p1);

        m6 = _mm_mul_ps(m6, m0);
        m4 = _mm_mul_ps(m4, m0);

        m6 = _mm_add_ps(m6, exp_q2);

        m4 = _mm_mul_ps(m4, m2);
        m6 = _mm_mul_ps(m6, m0);

        m0 = am_1;

        m2 = _mm_add_ps(m2, m4);
        m6 = _mm_add_ps(m6, exp_q3);

        m1 = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(m1), 23));

        m6 = _mm_sub_ps(m6, m2);
        m6 = _mm_rcp_ps(m6);

        m2 = _mm_mul_ps(m2, m6);
        m2 = _mm_add_ps(m2, m2);

        m0 = _mm_add_ps(m0, m2);
        m0 = _mm_mul_ps(m0, m1);

        _mm_store_ps(s + i, m0);
    }
}


void nnedi3_dotProd_SSE2(const float *data, const float *weights, float *vals, const intptr_t n, const intptr_t len, const float *istd) {
    nnedi3_dotProd(data, weights, vals, n, len, istd);
}


void nnedi3_dotProd_i16_SSE2(const float *dataf, const float *weightsf, float *vals, const intptr_t n, const intptr_t len, const float *istd) {
    const uint8_t *data = (const uint8_t *)dataf;
    const uint8_t *weights = (const uint8_t *)weightsf;

    const uint8_t *orig_weights = weights;

    __m128 m8 = _mm_set1_ps(istd[0]);

    for (int i = 0; i < n; i += 4) {
        __m128i m0, m1, m2, m3;
        m0 = m1 = m2 = m3 = _mm_setzero_si128();

        for (int j = 0; j < len; j += 8) {
            __m128i m4, m5, m6, m7;

            m4 = m5 = m6 = m7 = _mm_load_si128((const __m128i *)(data + j * 2));

            m4 = _mm_madd_epi16(m4, _mm_load_si128((const __m128i *)weights));
            m5 = _mm_madd_epi16(m5, _mm_load_si128((const __m128i *)(weights + 16)));
            m6 = _mm_madd_epi16(m6, _mm_load_si128((const __m128i *)(weights + 32)));
            m7 = _mm_madd_epi16(m7, _mm_load_si128((const __m128i *)(weights + 48)));

            m0 = _mm_add_epi32(m0, m4);
            m1 = _mm_add_epi32(m1, m5);
            m2 = _mm_add_epi32(m2, m6);
            m3 = _mm_add_epi32(m3, m7);

            weights += 64;
        }

        __m128i m4 = m0;
        __m128i m5 = m2;

        m0 = _mm_unpacklo_epi64(m0, m1);
        m2 = _mm_unpacklo_epi64(m2, m3);
        m4 = _mm_unpackhi_epi64(m4, m1);
        m5 = _mm_unpackhi_epi64(m5, m3);

        m0 = _mm_add_epi32(m0, m4);
        m2 = _mm_add_epi32(m2, m5);

        __m128i m6 = m0;
        m0 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(m0), _mm_castsi128_ps(m2), 136));
        m6 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(m6), _mm_castsi128_ps(m2), 221));
        m6 = _mm_add_epi32(m6, m0);

        __m128 m9 = _mm_cvtepi32_ps(m6);
        m9 = _mm_mul_ps(m9, _mm_load_ps((const float *)(orig_weights + n * len * 2 + i * 8)));
        m9 = _mm_mul_ps(m9, m8);
        m9 = _mm_add_ps(m9, _mm_load_ps((const float *)(orig_weights + n * len * 2 + i * 8 + 16)));

        _mm_store_ps(vals + i, m9);
    }
}
