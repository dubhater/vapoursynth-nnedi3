#ifndef SIMD_X86_H
#define SIMD_X86_H

#include <stdint.h>
#include <emmintrin.h>


#define exp_hi _mm_set1_ps(80.0f)
#define exp_lo _mm_set1_ps(-80.0f)

#define sign_bits_f_zero_l _mm_castsi128_ps(_mm_set_epi64x(0x7fffffff7fffffff, 0x7fffffff00000000))
#define ones_f _mm_set1_ps(1.0f)
#define sign_bits_f _mm_castsi128_ps(_mm_set1_epi32(0x7fffffff))


static void nnedi3_e0_m16(float *s, const intptr_t n) {
    __m128 e0_mult = _mm_set1_ps(12102203.161561486f); // (1.0/ln(s))*(2^23)
    __m128 e0_bias = _mm_set1_ps(1064866805.0f); // (2^23)*127.0-486411.0

    for (int i = 0; i < n; i += 4) {
        __m128 m0 = _mm_load_ps(s + i);
        m0 = _mm_min_ps(m0, exp_hi);
        m0 = _mm_max_ps(m0, exp_lo);
        m0 = _mm_mul_ps(m0, e0_mult);
        m0 = _mm_add_ps(m0, e0_bias);

        __m128i m1 = _mm_cvtps_epi32(m0);
        _mm_store_si128((__m128i *)(s + i), m1);
    }
}


static void nnedi3_computeNetwork0(const float *input, const float *weights, uint8_t *d) {
    __m128 m0, m1, m2, m3;
    m0 = m1 = m2 = m3 = _mm_setzero_ps();

    for (int i = 0; i < 192 / 4; i += 4) {
        __m128 m4, m5, m6, m7;
        m4 = m5 = m6 = m7 = _mm_load_ps(input + i);

        m4 = _mm_mul_ps(m4, _mm_load_ps(weights + i * 4));
        m5 = _mm_mul_ps(m5, _mm_load_ps(weights + i * 4 + 4));
        m6 = _mm_mul_ps(m6, _mm_load_ps(weights + i * 4 + 8));
        m7 = _mm_mul_ps(m7, _mm_load_ps(weights + i * 4 + 12));

        m0 = _mm_add_ps(m0, m4);
        m1 = _mm_add_ps(m1, m5);
        m2 = _mm_add_ps(m2, m6);
        m3 = _mm_add_ps(m3, m7);
    }

    // Equivalent to
    // haddps m0, m1
    // haddps m2, m3
    // haddps m0, m2
    __m128 m4 = m0;
    __m128 m5 = m2;
    m0 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(m0), _mm_castps_pd(m1)));
    m2 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(m2), _mm_castps_pd(m3)));
    m4 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(m4), _mm_castps_pd(m1)));
    m5 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(m5), _mm_castps_pd(m3)));
    m0 = _mm_add_ps(m0, m4);
    m2 = _mm_add_ps(m2, m5);
    __m128 m6 = m0;
    m0 = _mm_shuffle_ps(m0, m2, 136);
    m6 = _mm_shuffle_ps(m6, m2, 221);
    m0 = _mm_add_ps(m0, m6);

    m0 = _mm_add_ps(m0, _mm_load_ps(weights + 768 / 4));
    m1 = m0;

    m0 = _mm_and_ps(m0, sign_bits_f_zero_l);
    m0 = _mm_add_ps(m0, ones_f);
    m0 = _mm_rcp_ps(m0);
    m0 = _mm_mul_ps(m0, m1);

    m1 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(m0), 0));
    m2 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(m0), 85));
    m3 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(m0), 170));
    m4 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(m0), 255));

    m1 = _mm_mul_ps(m1, _mm_load_ps(weights + 784 / 4));
    m2 = _mm_mul_ps(m2, _mm_load_ps(weights + 784 / 4 + 4));
    m3 = _mm_mul_ps(m3, _mm_load_ps(weights + 784 / 4 + 8));
    m4 = _mm_mul_ps(m4, _mm_load_ps(weights + 784 / 4 + 12));

    m1 = _mm_add_ps(m1, m2);
    m3 = _mm_add_ps(m3, m4);
    m1 = _mm_add_ps(m1, m3);
    m1 = _mm_add_ps(m1, _mm_load_ps(weights + 784 / 4 + 16));

    __m128 m7 = m1;

    m1 = _mm_and_ps(m1, sign_bits_f);
    m1 = _mm_add_ps(m1, ones_f);
    m1 = _mm_rcp_ps(m1);
    m7 = _mm_mul_ps(m7, m1);

    m3 = m0;
    m0 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(m3), 0));
    m1 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(m3), 85));
    m2 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(m3), 170));
    m3 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(m3), 255));

    m0 = _mm_mul_ps(m0, _mm_load_ps(weights + 864 / 4));
    m1 = _mm_mul_ps(m1, _mm_load_ps(weights + 864 / 4 + 4));
    m2 = _mm_mul_ps(m2, _mm_load_ps(weights + 864 / 4 + 8));
    m3 = _mm_mul_ps(m3, _mm_load_ps(weights + 864 / 4 + 12));

    m4 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(m7), 0));
    m5 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(m7), 85));
    m6 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(m7), 170));
    m7 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(m7), 255));

    m4 = _mm_mul_ps(m4, _mm_load_ps(weights + 864 / 4 + 16));
    m5 = _mm_mul_ps(m5, _mm_load_ps(weights + 864 / 4 + 20));
    m6 = _mm_mul_ps(m6, _mm_load_ps(weights + 864 / 4 + 24));
    m7 = _mm_mul_ps(m7, _mm_load_ps(weights + 864 / 4 + 28));

    m0 = _mm_add_ps(m0, m1);
    m2 = _mm_add_ps(m2, m3);
    m4 = _mm_add_ps(m4, m5);
    m6 = _mm_add_ps(m6, m7);

    m0 = _mm_add_ps(m0, m2);
    m4 = _mm_add_ps(m4, m6);
    m0 = _mm_add_ps(m0, m4);
    m0 = _mm_add_ps(m0, _mm_load_ps(weights + 864 / 4 + 32));

    m1 = _mm_movehl_ps(m1, m0);
    m0 = _mm_max_ps(m0, m1);
    m1 = _mm_castsi128_ps(_mm_shufflelo_epi16(_mm_castps_si128(m0), 14));
    d[0] = _mm_comile_ss(m1, m0);
}


static void nnedi3_dotProd(const float *data, const float *weights, float *vals, const intptr_t n, const intptr_t len, const float *istd) {
    const float *orig_weights = weights;

    __m128 m8 = _mm_set1_ps(istd[0]);

    for (int i = 0; i < n; i += 4) {
        __m128 m0, m1, m2, m3;
        m0 = m1 = m2 = m3 = _mm_setzero_ps();

        for (int j = 0; j < len; j += 4) {
            __m128 m4, m5, m6, m7;
            m4 = m5 = m6 = m7 = _mm_load_ps(data + j);

            m4 = _mm_mul_ps(m4, _mm_load_ps(weights));
            m5 = _mm_mul_ps(m5, _mm_load_ps(weights + 4));
            m6 = _mm_mul_ps(m6, _mm_load_ps(weights + 8));
            m7 = _mm_mul_ps(m7, _mm_load_ps(weights + 12));

            m0 = _mm_add_ps(m0, m4);
            m1 = _mm_add_ps(m1, m5);
            m2 = _mm_add_ps(m2, m6);
            m3 = _mm_add_ps(m3, m7);

            weights += 16;
        }

        __m128 m4 = m0;
        __m128 m5 = m2;

        m0 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(m0), _mm_castps_pd(m1)));
        m2 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(m2), _mm_castps_pd(m3)));
        m4 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(m4), _mm_castps_pd(m1)));
        m5 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(m5), _mm_castps_pd(m3)));

        m0 = _mm_add_ps(m0, m4);
        m2 = _mm_add_ps(m2, m5);

        __m128 m6 = m0;
        m0 = _mm_shuffle_ps(m0, m2, 136);
        m6 = _mm_shuffle_ps(m6, m2, 221);
        m6 = _mm_add_ps(m6, m0);
        m6 = _mm_mul_ps(m6, m8);
        m6 = _mm_add_ps(m6, _mm_load_ps(orig_weights + n * len + i));
        _mm_store_ps(vals + i, m6);
    }
}

#endif // SIMD_X86_H
