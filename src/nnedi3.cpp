/*
**   Copyright (C) 2010-2011 Kevin Stone
**
**   VapourSynth port by dubhater.
**
**   This program is free software; you can redistribute it and/or modify
**   it under the terms of the GNU General Public License as published by
**   the Free Software Foundation; either version 2 of the License, or
**   (at your option) any later version.
**
**   This program is distributed in the hope that it will be useful,
**   but WITHOUT ANY WARRANTY; without even the implied warranty of
**   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
**   GNU General Public License for more details.
**
**   You should have received a copy of the GNU General Public License
**   along with this program; if not, write to the Free Software
**   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/

#include <cerrno>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <algorithm>
#include <string>
#include <type_traits>

#include <VapourSynth.h>
#include <VSHelper.h>

#include "cpufeatures.h"

#ifdef _WIN32
#include <codecvt>
#include <locale>
#endif


#if defined(NNEDI3_X86)
// Functions implemented in simd_x86.h, simd_sse2.c, simd_fma3.c, simd_fma4.c.
extern "C" {
    extern void nnedi3_byte2float48_SSE2(const uint8_t *t, const intptr_t pitch, float *p);
    extern void nnedi3_word2float48_SSE2(const uint8_t *t, const intptr_t pitch, float *pf);
    extern void nnedi3_byte2word48_SSE2(const uint8_t *t, const intptr_t pitch, float *pf);
    extern void nnedi3_byte2word64_SSE2(const uint8_t *t, const intptr_t pitch, float *p);

    extern int32_t nnedi3_processLine0_SSE2(const uint8_t *tempu, intptr_t width, uint8_t *dstp, const uint8_t *src3p, const intptr_t src_pitch);

    extern void nnedi3_extract_m8_SSE2(const uint8_t *srcp, const intptr_t stride, const intptr_t xdia, const intptr_t ydia, float *mstd, float *input);
    extern void nnedi3_extract_m8_i16_SSE2(const uint8_t *srcp, const intptr_t stride, const intptr_t xdia, const intptr_t ydia, float *mstd, float *inputf);

    extern void nnedi3_computeNetwork0_SSE2(const float *input, const float *weights, uint8_t *d);
    extern void nnedi3_computeNetwork0_i16_SSE2(const float *inputf, const float *weightsf, uint8_t *d);
    extern void nnedi3_computeNetwork0new_SSE2(const float *datai, const float *weights, uint8_t *d);

    extern void nnedi3_weightedAvgElliottMul5_m16_SSE2(const float *w, const intptr_t n, float *mstd);

    extern void nnedi3_e0_m16_SSE2(float *s, const intptr_t n);
    extern void nnedi3_e1_m16_SSE2(float *s, const intptr_t n);
    extern void nnedi3_e2_m16_SSE2(float *s, const intptr_t n);

    extern void nnedi3_dotProd_SSE2(const float *data, const float *weights, float *vals, const intptr_t n, const intptr_t len, const float *istd);
    extern void nnedi3_dotProd_i16_SSE2(const float *dataf, const float *weightsf, float *vals, const intptr_t n, const intptr_t len, const float *istd);

    extern void nnedi3_computeNetwork0_FMA3(const float *input, const float *weights, uint8_t *d);
    extern void nnedi3_e0_m16_FMA3(float *s, const intptr_t n);
    extern void nnedi3_dotProd_FMA3(const float *data, const float *weights, float *vals, const intptr_t n, const intptr_t len, const float *istd);

    extern void nnedi3_computeNetwork0_FMA4(const float *input, const float *weights, uint8_t *d);
    extern void nnedi3_e0_m16_FMA4(float *s, const intptr_t n);
    extern void nnedi3_dotProd_FMA4(const float *data, const float *weights, float *vals, const intptr_t n, const intptr_t len, const float *istd);
}
#elif defined(NNEDI3_ARM)
// Functions implemented in simd_neon.c
extern "C" {
    extern void byte2word48_neon(const uint8_t *t, const intptr_t pitch, float *pf);
    extern void byte2word64_neon(const uint8_t *t, const intptr_t pitch, float *pf);
    extern void byte2float48_neon(const uint8_t *t, const intptr_t pitch, float *p);
    extern void word2float48_neon(const uint8_t *t8, const intptr_t pitch, float *p);

    extern void computeNetwork0_neon(const float *input, const float *weights, uint8_t *d);
    extern void computeNetwork0_i16_neon(const float *inputf, const float *weightsf, uint8_t *d);
    extern void computeNetwork0new_neon(const float *dataf, const float *weightsf, uint8_t *d);

    extern void dotProd_neon(const float *data, const float *weights, float *vals, const intptr_t n, const intptr_t len, const float *istd);
    extern void dotProd_i16_neon(const float *dataf, const float *weightsf, float *vals, const intptr_t n, const intptr_t len, const float *istd);
}
#endif


// Things that mustn't be shared between threads.
typedef struct {
    uint8_t *paddedp[3];
    int padded_stride[3];
    int padded_width[3];
    int padded_height[3];

    uint8_t *dstp[3];
    int dst_stride[3];

    int field[3];

    int32_t *lcount[3];
    float *input;
    float *temp;
} FrameData;


typedef struct nnedi3Data nnedi3Data;


struct nnedi3Data {
    VSNodeRef *node;
    VSVideoInfo vi;

    float *weights0;
    float *weights1[2];
    int asize;
    int nns;
    int xdia;
    int ydia;

    // Parameters.
    int field;
    int dh; // double height
    int process[3];
    int nsize;
    int nnsparam;
    int qual;
    int etype;
    int pscrn;
    int opt;
    int int16_prescreener;
    int int16_predictor;
    int exp;
    int show_mask;

    int max_value;

    void (*copyPad)(const VSFrameRef *, FrameData *, const nnedi3Data *, int, const VSAPI *);
    void (*evalFunc_0)(const nnedi3Data *, FrameData *);
    void (*evalFunc_1)(const nnedi3Data *, FrameData *);

    // Functions used in evalFunc_0
    void (*readPixels)(const uint8_t *, const intptr_t, float *);
    void (*computeNetwork0)(const float *, const float *, uint8_t *);
    int32_t (*processLine0)(const uint8_t *, int, uint8_t *, const uint8_t *, const int, const int);

    // Functions used in evalFunc_1
    void (*extract)(const uint8_t *, const intptr_t, const intptr_t, const intptr_t, float *, float *);
    void (*dotProd)(const float *, const float *, float *, const intptr_t, const intptr_t, const float *);
    void (*expfunc)(float *, const intptr_t);
    void (*wae5)(const float *, const intptr_t, float *);
};


template <typename PixelType>
static void copyPad(const VSFrameRef *src, FrameData *frameData, const nnedi3Data *d, int fn, const VSAPI *vsapi) {
    const int off = 1 - fn;

    for (int plane = 0; plane < d->vi.format->numPlanes; ++plane) {
        if (!d->process[plane])
            continue;

        const PixelType *srcp = (const PixelType *)vsapi->getReadPtr(src, plane);
        PixelType *dstp = (PixelType *)frameData->paddedp[plane];

        const int src_stride = vsapi->getStride(src, plane) / sizeof(PixelType);
        const int dst_stride = frameData->padded_stride[plane] / sizeof(PixelType);

        const int src_height = vsapi->getFrameHeight(src, plane);
        const int dst_height = frameData->padded_height[plane];

        const int src_width = vsapi->getFrameWidth(src, plane);
        const int dst_width = frameData->padded_width[plane];

        // Copy.
        if (!d->dh) {
            for (int y = off; y < src_height; y += 2)
                memcpy(dstp + 32 + (6 + y) * dst_stride,
                       srcp + y * src_stride,
                       src_width * sizeof(PixelType));
        } else {
            for (int y = 0; y < src_height; y++)
                memcpy(dstp + 32 + (6 + y * 2 + off) * dst_stride,
                       srcp + y * src_stride,
                       src_width * sizeof(PixelType));
        }

        // And pad.
        dstp += (6 + off) * dst_stride;
        for (int y = 6 + off; y < dst_height - 6; y += 2) {
            for (int x = 0; x < 32; ++x)
                dstp[x] = dstp[64 - x];

            int c = 2;
            for (int x = dst_width - 32; x < dst_width; ++x, c += 2)
                dstp[x] = dstp[x - c];

            dstp += dst_stride * 2;
        }

        dstp = (PixelType *)frameData->paddedp[plane];
        for (int y = off; y < 6; y += 2)
            memcpy(dstp + y * dst_stride,
                   dstp + (12 + 2 * off - y) * dst_stride,
                   dst_width * sizeof(PixelType));

        int c = 4;
        for (int y = dst_height - 6 + off; y < dst_height; y += 2, c += 4)
            memcpy(dstp + y * dst_stride,
                   dstp + (y - c) * dst_stride,
                   dst_width * sizeof(PixelType));
    }
}


static void elliott_C(float *data, const int n) {
    for (int i = 0; i < n; ++i)
        data[i] = data[i] / (1.0f + std::fabs(data[i]));
}


static void dotProd_C(const float *data, const float *weights, float *vals, const intptr_t n, const intptr_t len, const float *scale) {
    for (int i = 0; i < n; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < len; ++j)
            sum += data[j] * weights[i * len + j];

        vals[i] = sum * scale[0] + weights[n * len + i];
    }
}


static void dotProdS_C(const float *dataf, const float *weightsf, float *vals, const intptr_t n, const intptr_t len, const float *scale) {
    const int16_t *data = (int16_t *)dataf;
    const int16_t *weights = (int16_t *)weightsf;
    const float *wf = (float *)&weights[n * len];

    for (int i = 0; i < n; ++i) {
        int sum = 0, off = ((i >> 2) << 3) + (i & 3);
        for (int j = 0; j < len; ++j)
            sum += data[j] * weights[i * len + j];

        vals[i] = sum * wf[off] * scale[0] + wf[off + 4];
    }
}


static void computeNetwork0_C(const float *input, const float *weights, uint8_t *d) {
    float temp[12], scale = 1.0f;
    dotProd_C(input, weights, temp, 4, 48, &scale);
    const float t = temp[0];
    elliott_C(temp, 4);
    temp[0] = t;
    dotProd_C(temp, weights + 4 * 49, temp + 4, 4, 4, &scale);
    elliott_C(temp + 4, 4);
    dotProd_C(temp, weights + 4 * 49 + 4 * 5, temp + 8, 4, 8, &scale);
    if (std::max(temp[10], temp[11]) <= std::max(temp[8], temp[9]))
        d[0] = 1;
    else
        d[0] = 0;
}


static void computeNetwork0_i16_C(const float *inputf, const float *weightsf, uint8_t *d) {
    const float *wf = weightsf + 2 * 48;
    float temp[12], scale = 1.0f;
    dotProdS_C(inputf, weightsf, temp, 4, 48, &scale);
    const float t = temp[0];
    elliott_C(temp, 4);
    temp[0] = t;
    dotProd_C(temp, wf + 8, temp + 4, 4, 4, &scale);
    elliott_C(temp + 4, 4);
    dotProd_C(temp, wf + 8 + 4 * 5, temp + 8, 4, 8, &scale);
    if (std::max(temp[10], temp[11]) <= std::max(temp[8], temp[9]))
        d[0] = 1;
    else
        d[0] = 0;
}


template <typename PixelType>
static void pixel2float48_C(const uint8_t *t8, const intptr_t pitch, float *p) {
    const PixelType *t = (const PixelType *)t8;

    for (int y = 0; y < 4; ++y)
        for (int x = 0; x < 12; ++x)
            p[y * 12 + x] = t[y * pitch * 2 + x];
}


static void byte2word48_C(const uint8_t *t, const intptr_t pitch, float *pf) {
    int16_t *p = (int16_t *)pf;
    for (int y = 0; y < 4; ++y)
        for (int x = 0; x < 12; ++x)
            p[y * 12 + x] = t[y * pitch * 2 + x];
}


#ifdef NNEDI3_X86
#define CB2(n) std::max(std::min((n), 254), 0)
static int32_t processLine0_maybeSSE2(const uint8_t *tempu, int width, uint8_t *dstp, const uint8_t *src3p, const int src_pitch, const int max_value) {
    int32_t count = 0;
    const int remain = width & 15;
    width -= remain;
    if (width)
        count = nnedi3_processLine0_SSE2(tempu, width, dstp, src3p, src_pitch);

    for (int x = width; x < width + remain; ++x) {
        if (tempu[x]) {
            dstp[x] = CB2((19 * (src3p[x + src_pitch * 2] + src3p[x + src_pitch * 4]) - 3 * (src3p[x] + src3p[x + src_pitch * 6]) + 16) / 32);
        } else {
            dstp[x] = 255;
            ++count;
        }
    }
    return count;
}
#undef CB2
#endif


// PixelType can be uint8_t, uint16_t, or float.
// TempType can be int or float.
template <typename PixelType, typename TempType>
static int32_t processLine0_C(const uint8_t *tempu, int width, uint8_t *dstp8, const uint8_t *src3p8, const int src_pitch, const int max_value) {
    PixelType *dstp = (PixelType *)dstp8;
    const PixelType *src3p = (const PixelType *)src3p8;

    TempType minimum = 0;
    TempType maximum = max_value - 1;
    // Technically the -1 is only needed for 8 and 16 bit input.

    int count = 0;
    for (int x = 0; x < width; ++x) {
        if (tempu[x]) {
            TempType tmp = 19 * (src3p[x + src_pitch * 2] + src3p[x + src_pitch * 4]) - 3 * (src3p[x] + src3p[x + src_pitch * 6]);
            if (std::is_same<TempType, float>::value) {
                dstp[x] = tmp / 32;
            } else {
                tmp = (tmp + 16) / 32;
                dstp[x] = std::max(std::min(tmp, maximum), minimum);
            }
        } else {
            memset(dstp + x, 255, sizeof(PixelType));
            ++count;
        }
    }
    return count;
}

// new prescreener functions
static void byte2word64_C(const uint8_t *t, const intptr_t pitch, float *p) {
    int16_t *ps = (int16_t *)p;
    for (int y = 0; y < 4; ++y)
        for (int x = 0; x < 16; ++x)
            ps[y * 16 + x] = t[y * pitch * 2 + x];
}


template <int size, bool shift>
static void word2word_C(const uint8_t *t, const intptr_t pitch, float *p) {
    int16_t *ps = (int16_t *)p;
    const uint16_t *ts = (const uint16_t *)t;

    const int height = 4;
    const int width = size / height;

    for (int y = 0; y < height; ++y) {
        if (shift) {
            for (int x = 0; x < width; ++x)
                ps[y * width + x] = ts[y * pitch * 2 + x] >> (int)shift;
        } else {
            memcpy(&ps[y * width], &ts[y * pitch * 2], width * sizeof(uint16_t));
        }
    }
}


static void computeNetwork0new_C(const float *datai, const float *weights, uint8_t *d) {
    int16_t *data = (int16_t *)datai;
    int16_t *ws = (int16_t *)weights;
    float *wf = (float *)&ws[4 * 64];
    float vals[8];
    for (int i = 0; i < 4; ++i) {
        int sum = 0;
        for (int j = 0; j < 64; ++j)
            sum += data[j] * ws[(i << 3) + ((j >> 3) << 5) + (j & 7)];
        const float t = sum * wf[i] + wf[4 + i];
        vals[i] = t / (1.0f + std::fabs(t));
    }
    for (int i = 0; i < 4; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < 4; ++j)
            sum += vals[j] * wf[8 + i + (j << 2)];
        vals[4 + i] = sum + wf[8 + 16 + i];
    }
    int mask = 0;
    for (int i = 0; i < 4; ++i) {
        if (vals[4 + i] > 0.0f)
            mask |= (0x1 << (i << 3));
    }
    ((int *)d)[0] = mask;
}


template <typename PixelType>
static void evalFunc_0(const nnedi3Data *d, FrameData *frameData) {
    float *input = frameData->input;
    const float *weights0 = d->weights0;
    float *temp = frameData->temp;
    uint8_t *tempu = (uint8_t *)temp;

    // And now the actual work.
    for (int plane = 0; plane < d->vi.format->numPlanes; ++plane) {
        if (!d->process[plane])
            continue;

        const PixelType *srcp = (const PixelType *)frameData->paddedp[plane];
        const int src_stride = frameData->padded_stride[plane] / sizeof(PixelType);

        const int width = frameData->padded_width[plane];
        const int height = frameData->padded_height[plane];

        PixelType *dstp = (PixelType *)frameData->dstp[plane];
        const int dst_stride = frameData->dst_stride[plane] / sizeof(PixelType);

        for (int y = 1 - frameData->field[plane]; y < height - 12; y += 2)
            memcpy(dstp + y * dst_stride,
                   srcp + 32 + (6 + y) * src_stride,
                   (width - 64) * sizeof(PixelType));

        const int ystart = 6 + frameData->field[plane];
        const int ystop = height - 6;
        srcp += ystart * src_stride;
        dstp += (ystart - 6) * dst_stride - 32;
        const PixelType *src3p = srcp - src_stride * 3;
        int32_t *lcount = frameData->lcount[plane] - 6;
        if (d->pscrn == 1) {// original
            for (int y = ystart; y < ystop; y += 2) {
                for (int x = 32; x < width - 32; ++x) {
                    d->readPixels((const uint8_t *)(src3p + x - 5), src_stride, input);
                    d->computeNetwork0(input, weights0, tempu+x);
                }
                lcount[y] += d->processLine0(tempu + 32, width - 64, (uint8_t *)(dstp + 32), (const uint8_t *)(src3p + 32), src_stride, d->max_value);
                src3p += src_stride * 2;
                dstp += dst_stride * 2;
            }
        } else if (d->pscrn >= 2) {// new
            for (int y = ystart; y < ystop; y += 2) {
                for (int x = 32; x < width - 32; x += 4) {
                    d->readPixels((const uint8_t *)(src3p + x - 6), src_stride, input);
                    d->computeNetwork0(input, weights0, tempu + x);
                }
                lcount[y] += d->processLine0(tempu + 32, width - 64, (uint8_t *)(dstp + 32), (const uint8_t *)(src3p + 32), src_stride, d->max_value);
                src3p += src_stride * 2;
                dstp += dst_stride * 2;
            }
        } else {// no prescreening
            for (int y = ystart; y < ystop; y += 2) {
                memset(dstp + 32, 255, (width - 64) * sizeof(PixelType));
                lcount[y] += width - 64;
                dstp += dst_stride * 2;
            }
        }
    }
}


template <typename PixelType, typename AccumType, typename FloatType>
static void extract_m8_C(const uint8_t *srcp8, const intptr_t stride, const intptr_t xdia, const intptr_t ydia, float *mstd, float *input) {
    // uint8_t or uint16_t or float
    const PixelType *srcp = (const PixelType *)srcp8;

    // int32_t or int64_t or double
    AccumType sum = 0, sumsq = 0;
    for (int y = 0; y < ydia; ++y) {
        const PixelType *srcpT = srcp + y * stride * 2;
        for (int x = 0; x < xdia; ++x) {
            sum += srcpT[x];
            if (std::is_same<PixelType, float>::value)
                sumsq += (double)srcpT[x] * srcpT[x];
            else
                sumsq += (uint32_t)srcpT[x] * (uint32_t)srcpT[x];
            input[x] = srcpT[x];
        }
        input += xdia;
    }
    const float scale = 1.0f / (xdia * ydia);
    mstd[0] = sum * scale;
    // float or double or double
    const FloatType tmp = (FloatType)sumsq * scale - (FloatType)mstd[0] * mstd[0];
    mstd[3] = 0.0f;
    if (tmp <= FLT_EPSILON)
        mstd[1] = mstd[2] = 0.0f;
    else {
        mstd[1] = (float)std::sqrt(tmp);
        mstd[2] = 1.0f / mstd[1];
    }
}


template <typename PixelType>
static void extract_m8_i16_C(const uint8_t *srcp8, const intptr_t stride, const intptr_t xdia, const intptr_t ydia, float *mstd, float *inputf) {
    const PixelType *srcp = (const PixelType *)srcp8;
    int16_t *input = (int16_t *)inputf;
    int64_t sum = 0, sumsq = 0;
    for (int y = 0; y < ydia; ++y) {
        const PixelType *srcpT = srcp + y * stride * 2;
        for (int x = 0; x < xdia; ++x) {
            sum += srcpT[x];
            sumsq += srcpT[x] * srcpT[x];
            input[x] = srcpT[x];
        }
        input += xdia;
    }
    const float scale = 1.0f / (float)(xdia * ydia);
    mstd[0] = sum * scale;
    mstd[1] = (float)((double)sumsq * scale - mstd[0] * mstd[0]);
    mstd[3] = 0.0f;
    if (mstd[1] <= FLT_EPSILON)
        mstd[1] = mstd[2] = 0.0f;
    else {
        mstd[1] = std::sqrt(mstd[1]);
        mstd[2] = 1.0f / mstd[1];
    }
}


static const float exp_lo = -80.0f;
static const float exp_hi = +80.0f;


// exp from:  A Fast, Compact Approximation of the Exponential Function (1998)
//            Nicol N. Schraudolph


static void e0_m16_C(float *s, const intptr_t n) {
    const float e0_mult = 12102203.161561486f; // (1.0/ln(2))*(2^23)
    const float e0_bias = 1064866805.0f; // (2^23)*127.0-486411.0

    for (int i = 0; i < n; ++i) {
        const int t = (int)(std::max(std::min(s[i], exp_hi), exp_lo) * e0_mult + e0_bias);
        memcpy(&s[i], &t, sizeof(float));
    }
}


// exp from Loren Merritt


static void e1_m16_C(float *s, const intptr_t n) {
    const float e1_scale = 1.4426950409f; // 1/ln(2)
//    const float e1_bias = 12582912.0f; // 3<<22
    const float e1_c0 = 1.00035f;
    const float e1_c1 = 0.701277797f;
    const float e1_c2 = 0.237348593f;

    for (int q = 0; q < n; ++q) {
        float x = std::max(std::min(s[q], exp_hi), exp_lo) * e1_scale;
        int i = (int)(x + 128.5f) - 128;
        x -= i;
        x = e1_c0 + e1_c1 * x + e1_c2 * x * x;
        i = (i + 127) << 23;
        float i_f;
        memcpy(&i_f, &i, sizeof(float));
        s[q] = x * i_f;
    }
}


static void e2_m16_C(float *s, const intptr_t n) {
    for (int i = 0; i < n; ++i)
        s[i] = std::exp(std::max(std::min(s[i], exp_hi), exp_lo));
}

// exp from Intel Approximate Math (AM) Library


static void weightedAvgElliottMul5_m16_C(const float *w, const intptr_t n, float *mstd) {
    float vsum = 0.0f, wsum = 0.0f;
    for (int i = 0; i < n; ++i) {
        vsum += w[i] * (w[n + i] / (1.0f + std::fabs(w[n + i])));
        wsum += w[i];
    }

    const float min_weight_sum = 1e-10f;

    if (wsum > min_weight_sum)
        mstd[3] += ((5.0f * vsum) / wsum) * mstd[1] + mstd[0];
    else
        mstd[3] += mstd[0];
}


template <typename PixelType>
static void evalFunc_1(const nnedi3Data *d, FrameData *frameData) {
    float *input = frameData->input;
    float *temp = frameData->temp;
    const float * const *weights1 = d->weights1;
    const int qual = d->qual;
    const int asize = d->asize;
    const int nns = d->nns;
    const int xdia = d->xdia;
    const int xdiad2m1 = (xdia / 2) - 1;
    const int ydia = d->ydia;
    const float scale = 1.0f / (float)qual;

    for (int plane = 0; plane < d->vi.format->numPlanes; ++plane) {
        if (!d->process[plane])
            continue;

        const PixelType *srcp = (const PixelType *)frameData->paddedp[plane];
        const int src_stride = frameData->padded_stride[plane] / sizeof(PixelType);

        const int width = frameData->padded_width[plane];
        const int height = frameData->padded_height[plane];

        PixelType *dstp = (PixelType *)frameData->dstp[plane];
        const int dst_stride = frameData->dst_stride[plane] / sizeof(PixelType);

        const int ystart = frameData->field[plane];
        const int ystop = height - 12;

        srcp += (ystart + 6) * src_stride;
        dstp += ystart * dst_stride - 32;
        const PixelType *srcpp = srcp - (ydia - 1) * src_stride - xdiad2m1;

        for (int y = ystart; y < ystop; y += 2) {
            for (int x = 32; x < width - 32; ++x) {
                uint32_t pixel = 0;
                memcpy(&pixel, dstp + x, sizeof(PixelType));

                uint32_t all_ones = 0;
                memset(&all_ones, 255, sizeof(PixelType));

                if (pixel != all_ones)
                    continue;

                float mstd[4];
                d->extract((const uint8_t *)(srcpp + x), src_stride, xdia, ydia, mstd, input);
                for (int i = 0; i < qual; ++i) {
                    d->dotProd(input, weights1[i], temp, nns * 2, asize, mstd + 2);
                    d->expfunc(temp, nns);
                    d->wae5(temp, nns, mstd);
                }

                if (std::is_same<PixelType, float>::value)
                    dstp[x] = mstd[3] * scale;
                else
                    dstp[x] = std::min(std::max((int)(mstd[3] * scale + 0.5f), 0), d->max_value);
            }
            srcpp += src_stride * 2;
            dstp += dst_stride * 2;
        }
    }
}


#define NUM_NSIZE 7
#define NUM_NNS 5


static int roundds(const double f) {
    if (f - std::floor(f) >= 0.5)
        return std::min((int)std::ceil(f), 32767);
    return std::max((int)std::floor(f), -32768);
}


static void shufflePreScrnL2L3(float *wf, float *rf) {
    for (int j = 0; j < 4; ++j)
        for (int k = 0; k < 4; ++k)
            wf[k * 4 + j] = rf[j * 4 + k];
    rf += 4 * 5;
    wf += 4 * 5;
    const int jtable[4] = { 0, 2, 1, 3 };
    for (int j = 0; j < 4; ++j) {
        for (int k = 0; k < 8; ++k)
            wf[k * 4 + j] = rf[jtable[j] * 8 + k];
        wf[4 * 8 + j] = rf[4 * 8 + jtable[j]];
    }
}


static void selectFunctions(nnedi3Data *d) {
#if defined(NNEDI3_X86) || defined(NNEDI3_ARM)
    CPUFeatures cpu;
    getCPUFeatures(&cpu);
#endif

#if defined(NNEDI3_ARM)
    if (!cpu.neon)
        // Must set opt to 0 so the weights don't get shuffled.
        d->opt = 0;
#endif

    if (d->vi.format->sampleType == stInteger && d->vi.format->bitsPerSample == 8) {
        d->copyPad = copyPad<uint8_t>;
        d->evalFunc_0 = evalFunc_0<uint8_t>;
        d->evalFunc_1 = evalFunc_1<uint8_t>;

        // evalFunc_0
        d->processLine0 = processLine0_C<uint8_t, int>;

        if (d->pscrn < 2) { // original prescreener
            if (d->int16_prescreener) { // int16 dot products
                d->readPixels = byte2word48_C;
                d->computeNetwork0 = computeNetwork0_i16_C;
            } else {
                d->readPixels = pixel2float48_C<uint8_t>;
                d->computeNetwork0 = computeNetwork0_C;
            }
        } else { // new prescreener
            // only int16 dot products
            d->readPixels = byte2word64_C;
            d->computeNetwork0 = computeNetwork0new_C;
        }

        // evalFunc_1
        d->wae5 = weightedAvgElliottMul5_m16_C;

        if (d->int16_predictor) { // use int16 dot products
            d->extract = extract_m8_i16_C<uint8_t>;
            d->dotProd = dotProdS_C;
        } else { // use float dot products
            d->extract = extract_m8_C<uint8_t, int32_t, float>;
            d->dotProd = dotProd_C;
        }

        if (d->exp == 2) // use slow exp
            d->expfunc = e2_m16_C;
        else if (d->exp == 1) // use faster exp
            d->expfunc = e1_m16_C;
        else // use fastest exp
            d->expfunc = e0_m16_C;

#if defined(NNEDI3_X86)
        if (d->opt) {
            // evalFunc_0
            d->processLine0 = processLine0_maybeSSE2;

            if (d->pscrn < 2) { // original prescreener
                if (d->int16_prescreener) { // int16 dot products
                    d->readPixels = nnedi3_byte2word48_SSE2;
                    d->computeNetwork0 = nnedi3_computeNetwork0_i16_SSE2;
                } else {
                    d->readPixels = nnedi3_byte2float48_SSE2;
                    d->computeNetwork0 = nnedi3_computeNetwork0_SSE2;
                    if (cpu.fma3)
                        d->computeNetwork0 = nnedi3_computeNetwork0_FMA3;
                    if (cpu.fma4)
                        d->computeNetwork0 = nnedi3_computeNetwork0_FMA4;
                }
            } else { // new prescreener
                // only int16 dot products
                d->readPixels = nnedi3_byte2word64_SSE2;
                d->computeNetwork0 = nnedi3_computeNetwork0new_SSE2;
            }

            // evalFunc_1
            d->wae5 = nnedi3_weightedAvgElliottMul5_m16_SSE2;

            if (d->int16_predictor) { // use int16 dot products
                d->extract = nnedi3_extract_m8_i16_SSE2;
                d->dotProd = nnedi3_dotProd_i16_SSE2;
            } else { // use float dot products
                d->extract = nnedi3_extract_m8_SSE2;
                d->dotProd = nnedi3_dotProd_SSE2;
                if (cpu.fma3)
                    d->dotProd = nnedi3_dotProd_FMA3;
                if (cpu.fma4)
                    d->dotProd = nnedi3_dotProd_FMA4;
            }

            if (d->exp == 2) { // use slow exp
                d->expfunc = nnedi3_e2_m16_SSE2;
            } else if (d->exp == 1) { // use faster exp
                d->expfunc = nnedi3_e1_m16_SSE2;
            } else { // use fastest exp
                d->expfunc = nnedi3_e0_m16_SSE2;
                if (cpu.fma3)
                    d->expfunc = nnedi3_e0_m16_FMA3;
                if (cpu.fma4)
                    d->expfunc = nnedi3_e0_m16_FMA4;
            }
        }
#elif defined(NNEDI3_ARM)
        if (d->opt && cpu.neon) {
            if (d->pscrn < 2) { // original prescreener
                if (d->int16_prescreener) { // int16 dot products
                    d->readPixels = byte2word48_neon;
                    d->computeNetwork0 = computeNetwork0_i16_neon;
                } else {
                    d->readPixels = byte2float48_neon;
                    d->computeNetwork0 = computeNetwork0_neon;
                }
            } else { // new prescreener
                // only int16 dot products
                d->readPixels = byte2word64_neon;
                d->computeNetwork0 = computeNetwork0new_neon;
            }

            // evalFunc_1
            if (d->int16_predictor) // use int16 dot products
                d->dotProd = dotProd_i16_neon;
            else // use float dot products
                d->dotProd = dotProd_neon;
        }
#endif
    } else if (d->vi.format->sampleType == stInteger && d->vi.format->bitsPerSample <= 16) {
        d->copyPad = copyPad<uint16_t>;
        d->evalFunc_0 = evalFunc_0<uint16_t>;
        d->evalFunc_1 = evalFunc_1<uint16_t>;

        // evalFunc_0
        d->processLine0 = processLine0_C<uint16_t, int>;

        if (d->pscrn < 2) {
            if (d->int16_prescreener) {
                d->readPixels = d->vi.format->bitsPerSample == 16 ? word2word_C<48, true> : word2word_C<48, false>;
                d->computeNetwork0 = computeNetwork0_i16_C;
            } else {
                d->readPixels = pixel2float48_C<uint16_t>;
                d->computeNetwork0 = computeNetwork0_C;
            }
        } else {
            d->readPixels = d->vi.format->bitsPerSample == 16 ? word2word_C<64, true> : word2word_C<64, false>;
            d->computeNetwork0 = computeNetwork0new_C;
        }

        // evalFunc_1
        d->wae5 = weightedAvgElliottMul5_m16_C;

        if (d->int16_predictor) { // only used for 9..15 bits
            d->extract = extract_m8_i16_C<uint16_t>;
            d->dotProd = dotProdS_C;
        } else {
            d->extract = extract_m8_C<uint16_t, int64_t, double>;
            d->dotProd = dotProd_C;
        }

        if (d->exp == 2) // use slow exp
            d->expfunc = e2_m16_C;
        else if (d->exp == 1) // use faster exp
            d->expfunc = e1_m16_C;
        else // use fastest exp
            d->expfunc = e0_m16_C;

#if defined(NNEDI3_X86)
        if (d->opt) {
            // evalFunc_0
            if (d->pscrn < 2) {
                if (d->int16_prescreener) {
                    d->computeNetwork0 = nnedi3_computeNetwork0_i16_SSE2;
                } else {
                    d->readPixels = nnedi3_word2float48_SSE2;
                    d->computeNetwork0 = nnedi3_computeNetwork0_SSE2;
                    if (cpu.fma3)
                        d->computeNetwork0 = nnedi3_computeNetwork0_FMA3;
                    if (cpu.fma4)
                        d->computeNetwork0 = nnedi3_computeNetwork0_FMA4;
                }
            } else {
                d->computeNetwork0 = nnedi3_computeNetwork0new_SSE2;
            }

            // evalFunc_1
            d->wae5 = nnedi3_weightedAvgElliottMul5_m16_SSE2;

            if (d->int16_predictor) {
                d->dotProd = nnedi3_dotProd_i16_SSE2;
            } else {
                d->dotProd = nnedi3_dotProd_SSE2;
                if (cpu.fma3)
                    d->dotProd = nnedi3_dotProd_FMA3;
                if (cpu.fma4)
                    d->dotProd = nnedi3_dotProd_FMA4;
            }

            if (d->exp == 2) { // use slow exp
                d->expfunc = nnedi3_e2_m16_SSE2;
            } else if (d->exp == 1) { // use faster exp
                d->expfunc = nnedi3_e1_m16_SSE2;
            } else { // use fastest exp
                d->expfunc = nnedi3_e0_m16_SSE2;
                if (cpu.fma3)
                    d->expfunc = nnedi3_e0_m16_FMA3;
                if (cpu.fma4)
                    d->expfunc = nnedi3_e0_m16_FMA4;
            }
        }
#elif defined(NNEDI3_ARM)
        if (d->opt && cpu.neon) {
            if (d->pscrn < 2) {
                if (d->int16_prescreener) {
                    d->computeNetwork0 = computeNetwork0_i16_neon;
                } else {
                    d->readPixels = word2float48_neon;
                    d->computeNetwork0 = computeNetwork0_neon;
                }
            } else {
                d->computeNetwork0 = computeNetwork0new_neon;
            }

            if (d->int16_predictor) {
                d->dotProd = dotProd_i16_neon;
            } else {
                d->dotProd = dotProd_neon;
            }
        }
#endif
    } else if (d->vi.format->sampleType == stFloat && d->vi.format->bitsPerSample == 32) {
        d->copyPad = copyPad<float>;
        d->evalFunc_0 = evalFunc_0<float>;
        d->evalFunc_1 = evalFunc_1<float>;

        // evalFunc_0
        d->processLine0 = processLine0_C<float, float>;

        d->readPixels = pixel2float48_C<float>;
        d->computeNetwork0 = computeNetwork0_C;

        // evalFunc_1
        d->wae5 = weightedAvgElliottMul5_m16_C;

        d->extract = extract_m8_C<float, double, double>;
        d->dotProd = dotProd_C;

        if (d->exp == 2) // use slow exp
            d->expfunc = e2_m16_C;
        else if (d->exp == 1) // use faster exp
            d->expfunc = e1_m16_C;
        else // use fastest exp
            d->expfunc = e0_m16_C;

#if defined(NNEDI3_X86)
        if (d->opt) {
            // evalFunc_0
            d->computeNetwork0 = nnedi3_computeNetwork0_SSE2;
            if (cpu.fma3)
                d->computeNetwork0 = nnedi3_computeNetwork0_FMA3;
            if (cpu.fma4)
                d->computeNetwork0 = nnedi3_computeNetwork0_FMA4;

            // evalFunc_1
            d->wae5 = nnedi3_weightedAvgElliottMul5_m16_SSE2;

            d->dotProd = nnedi3_dotProd_SSE2;
            if (cpu.fma3)
                d->dotProd = nnedi3_dotProd_FMA3;
            if (cpu.fma4)
                d->dotProd = nnedi3_dotProd_FMA4;

            if (d->exp == 2) { // use slow exp
                d->expfunc = nnedi3_e2_m16_SSE2;
            } else if (d->exp == 1) { // use faster exp
                d->expfunc = nnedi3_e1_m16_SSE2;
            } else { // use fastest exp
                d->expfunc = nnedi3_e0_m16_SSE2;
                if (cpu.fma3)
                    d->expfunc = nnedi3_e0_m16_FMA3;
                if (cpu.fma4)
                    d->expfunc = nnedi3_e0_m16_FMA4;
            }
        }
#elif defined(NNEDI3_ARM)
        if (d->opt && cpu.neon) {
            d->computeNetwork0 = computeNetwork0_neon;
            d->dotProd = dotProd_neon;
        }
#endif
    }
}


static void VS_CC nnedi3Init(VSMap *in, VSMap *out, void **instanceData, VSNode *node, VSCore *core, const VSAPI *vsapi) {
    nnedi3Data *d = (nnedi3Data *) * instanceData;
    vsapi->setVideoInfo(&d->vi, 1, node);
}


static int modnpf(const int m, const int n) {
    if ((m % n) == 0)
        return m;
    return m + n - (m % n);
}


typedef enum VSFieldBased {
    VSFieldBasedProgressive = 0,
    VSFieldBasedBFF,
    VSFieldBasedTFF
} VSFieldBased;


typedef enum VSField {
    VSFieldBottom = 0,
    VSFieldTop
} VSField;


static const VSFrameRef *VS_CC nnedi3GetFrame(int n, int activationReason, void **instanceData, void **fData, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) {
    const nnedi3Data *d = (const nnedi3Data *) * instanceData;

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(d->field > 1 ? n / 2 : n, d->node, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        const VSFrameRef *src = vsapi->getFrameFilter(d->field > 1 ? n / 2 : n, d->node, frameCtx);

        int err;
        const VSMap *src_props = vsapi->getFramePropsRO(src);

        int effective_field = d->field;
        if (effective_field > 1)
            effective_field -= 2;

        if (d->dh) {
            int field = int64ToIntS(vsapi->propGetInt(src_props, "_Field", 0, &err));
            if (!err) {
                if (field == VSFieldBottom)
                    effective_field = 0;
                else if (field == VSFieldTop)
                    effective_field = 1;
            }
        } else {
            int fieldbased = int64ToIntS(vsapi->propGetInt(src_props, "_FieldBased", 0, &err));

            if (fieldbased == VSFieldBasedBFF)
                effective_field = 0;
            else if (fieldbased == VSFieldBasedTFF)
                effective_field = 1;
        }

        int field_n;
        if (d->field > 1) {
            if (n & 1) {
                field_n = (effective_field == 0);
            } else {
                field_n = (effective_field == 1);
            }
        } else {
            field_n = effective_field;
        }

        VSFrameRef *dst = vsapi->newVideoFrame(d->vi.format, d->vi.width, d->vi.height, src, core);


        FrameData *frameData = (FrameData *)malloc(sizeof(FrameData));
        memset(frameData, 0, sizeof(FrameData));

        for (int plane = 0; plane < d->vi.format->numPlanes; plane++) {
            if (!d->process[plane])
                continue;

            const int min_pad = 10;
            const int min_alignment = 16;

            int dst_width = vsapi->getFrameWidth(dst, plane);
            int dst_height = vsapi->getFrameHeight(dst, plane);

            frameData->padded_width[plane]  = dst_width + 64;
            frameData->padded_height[plane] = dst_height + 12;
            frameData->padded_stride[plane] = modnpf(frameData->padded_width[plane] * d->vi.format->bytesPerSample + min_pad, min_alignment); // TODO: maybe min_pad is in pixels too?
            frameData->paddedp[plane] = vs_aligned_malloc<uint8_t>((size_t)frameData->padded_stride[plane] * (size_t)frameData->padded_height[plane], min_alignment);

            frameData->dstp[plane] = vsapi->getWritePtr(dst, plane);
            frameData->dst_stride[plane] = vsapi->getStride(dst, plane);

            frameData->lcount[plane] = vs_aligned_malloc<int32_t>(dst_height * sizeof(int32_t), 16);
            memset(frameData->lcount[plane], 0, dst_height * sizeof(int32_t));

            frameData->field[plane] = field_n;
        }

        frameData->input = vs_aligned_malloc<float>(512 * sizeof(float), 16);
        // evalFunc_0 requires at least padded_width[0] bytes.
        // evalFunc_1 requires at least 512 floats.
        size_t temp_size = std::max((size_t)frameData->padded_width[0], 512 * sizeof(float));
        frameData->temp = vs_aligned_malloc<float>(temp_size, 16);

        // Copy src to a padded "frame" in frameData and mirror the edges.
        d->copyPad(src, frameData, d, field_n, vsapi);


        // Handles prescreening and the cubic interpolation.
        d->evalFunc_0(d, frameData);

        // The rest.
        if (!d->show_mask)
            d->evalFunc_1(d, frameData);


        // Clean up.
        for (int plane = 0; plane < d->vi.format->numPlanes; plane++) {
            if (!d->process[plane])
                continue;

            vs_aligned_free(frameData->paddedp[plane]);
            vs_aligned_free(frameData->lcount[plane]);
        }
        vs_aligned_free(frameData->input);
        vs_aligned_free(frameData->temp);

        free(frameData);

        vsapi->freeFrame(src);

        VSMap *dst_props = vsapi->getFramePropsRW(dst);

        if (d->field > 1) {
            int err_num, err_den;
            int64_t duration_num = vsapi->propGetInt(dst_props, "_DurationNum", 0, &err_num);
            int64_t duration_den = vsapi->propGetInt(dst_props, "_DurationDen", 0, &err_den);
            if (!err_num && !err_den) {
                muldivRational(&duration_num, &duration_den, 1, 2); // Divide duration by 2.
                vsapi->propSetInt(dst_props, "_DurationNum", duration_num, paReplace);
                vsapi->propSetInt(dst_props, "_DurationDen", duration_den, paReplace);
            }
        }

        vsapi->propSetInt(dst_props, "_FieldBased", VSFieldBasedProgressive, paReplace);

        return dst;
    }

    return 0;
}


static void VS_CC nnedi3Free(void *instanceData, VSCore *core, const VSAPI *vsapi) {
    nnedi3Data *d = (nnedi3Data *)instanceData;
    vsapi->freeNode(d->node);

    vs_aligned_free(d->weights0);

    for (int i = 0; i < 2; i++)
        vs_aligned_free(d->weights1[i]);

    free(d);
}


static void VS_CC nnedi3Create(const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi) {
    nnedi3Data d;
    nnedi3Data *data;
    int err;

    // Get a clip reference from the input arguments. This must be freed later.
    d.node = vsapi->propGetNode(in, "clip", 0, 0);
    d.vi = *vsapi->getVideoInfo(d.node);

    if (!d.vi.format ||
        (d.vi.format->sampleType == stInteger && d.vi.format->bitsPerSample > 16) ||
        (d.vi.format->sampleType == stFloat && d.vi.format->bitsPerSample != 32)) {
        vsapi->setError(out, "nnedi3: only constant format 8..16 bit integer or 32 bit float input supported");
        vsapi->freeNode(d.node);
        return;
    }

    // Get the parameters.
    d.field = int64ToIntS(vsapi->propGetInt(in, "field", 0, 0));

    // Defaults to 0.
    d.dh = int64ToIntS(vsapi->propGetInt(in, "dh", 0, &err));

    int n = d.vi.format->numPlanes;
    int m = vsapi->propNumElements(in, "planes");

    for (int i = 0; i < 3; i++)
        d.process[i] = (m <= 0);

    for (int i = 0; i < m; i++) {
        int o = int64ToIntS(vsapi->propGetInt(in, "planes", i, 0));

        if (o < 0 || o >= n) {
            vsapi->setError(out, "nnedi3: plane index out of range");
            vsapi->freeNode(d.node);
            return;
        }

        if (d.process[o]) {
            vsapi->setError(out, "nnedi3: plane specified twice");
            vsapi->freeNode(d.node);
            return;
        }

        d.process[o] = 1;
    }

    d.nsize = int64ToIntS(vsapi->propGetInt(in, "nsize", 0, &err));
    if (err)
        d.nsize = 6;

    d.nnsparam = int64ToIntS(vsapi->propGetInt(in, "nns", 0, &err));
    if (err)
        d.nnsparam = 1;

    d.qual = int64ToIntS(vsapi->propGetInt(in, "qual", 0, &err));
    if (err)
        d.qual = 1;

    d.etype = int64ToIntS(vsapi->propGetInt(in, "etype", 0, &err));

    d.pscrn = int64ToIntS(vsapi->propGetInt(in, "pscrn", 0, &err));
    if (err) {
        if (d.vi.format->sampleType == stInteger)
            d.pscrn = 2;
        else
            d.pscrn = 1;
    }

    d.opt = !!vsapi->propGetInt(in, "opt", 0, &err);
#if defined(NNEDI3_X86) || defined(NNEDI3_ARM)
    if (err)
        d.opt = 1;
#else
    d.opt = 0;
#endif

    d.int16_prescreener = !!vsapi->propGetInt(in, "int16_prescreener", 0, &err);
    if (err)
        d.int16_prescreener = 1;

    d.int16_predictor = !!vsapi->propGetInt(in, "int16_predictor", 0, &err);
    if (err)
        d.int16_predictor = 1;

    d.exp = int64ToIntS(vsapi->propGetInt(in, "exp", 0, &err));

    d.show_mask = !!vsapi->propGetInt(in, "show_mask", 0, &err);

    // Check the values.
    if (d.field < 0 || d.field > 3) {
        vsapi->setError(out, "nnedi3: field must be between 0 and 3 (inclusive)");
        vsapi->freeNode(d.node);
        return;
    }

    d.dh = !!d.dh; // Just consider any nonzero value true.

    if (d.dh && d.field > 1) {
        vsapi->setError(out, "nnedi3: field must be 0 or 1 when dh is true");
        vsapi->freeNode(d.node);
        return;
    }

    if (d.nsize < 0 || d.nsize >= NUM_NSIZE) {
        vsapi->setError(out, "nnedi3: nsize must be between 0 and 6 (inclusive)");
        vsapi->freeNode(d.node);
        return;
    }

    if (d.nnsparam < 0 || d.nnsparam >= NUM_NNS) {
        vsapi->setError(out, "nnedi3: nns must be between 0 and 4 (inclusive)");
        vsapi->freeNode(d.node);
        return;
    }

    if (d.qual < 1 || d.qual > 2) {
        vsapi->setError(out, "nnedi3: qual must be between 1 and 2 (inclusive)");
        vsapi->freeNode(d.node);
        return;
    }

    if (d.etype < 0 || d.etype > 1) {
        vsapi->setError(out, "nnedi3: etype must be between 0 and 1 (inclusive)");
        vsapi->freeNode(d.node);
        return;
    }

    if (d.vi.format->sampleType == stInteger) {
        if (d.pscrn < 0 || d.pscrn > 4) {
            vsapi->setError(out, "nnedi3: pscrn must be between 0 and 4 (inclusive)");
            vsapi->freeNode(d.node);
            return;
        }
    } else {
        if (d.pscrn < 0 || d.pscrn > 1) {
            vsapi->setError(out, "nnedi3: pscrn must be between 0 and 1 (inclusive)");
            vsapi->freeNode(d.node);
            return;
        }
    }

    if (d.exp < 0 || d.exp > 2) {
        vsapi->setError(out, "nnedi3: exp must be between 0 and 2 (inclusive)");
        vsapi->freeNode(d.node);
        return;
    }

    // Changing the video info probably has to be done before createFilter.
    if (d.field > 1) {
        if (d.vi.numFrames > INT_MAX / 2) {
            vsapi->setError(out, "nnedi3: output clip would be too long");
            vsapi->freeNode(d.node);
            return;
        }

        d.vi.numFrames *= 2;
        muldivRational(&d.vi.fpsNum, &d.vi.fpsDen, 2, 1);
    }

    if (d.dh)
        d.vi.height *= 2;

    d.max_value = 65535 >> (16 - d.vi.format->bitsPerSample);

    if (d.vi.format->sampleType == stFloat)
        d.int16_prescreener = 0;

    // int16 dotProd can be used with up to 15 bits input
    if (d.vi.format->bitsPerSample > 15)
        d.int16_predictor = 0;

    selectFunctions(&d);


    std::string weights_name("nnedi3_weights.bin");

    VSPlugin *nnedi3Plugin = vsapi->getPluginById("com.deinterlace.nnedi3", core);
    std::string plugin_path(vsapi->getPluginPath(nnedi3Plugin));
    std::string weights_path(plugin_path.substr(0, plugin_path.find_last_of('/')) + "/" + weights_name);

    FILE *weights_file = NULL;

#ifdef _WIN32
    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>, wchar_t> utf16;

    weights_file = _wfopen(utf16.from_bytes(weights_path).c_str(), L"rb");
#else
    weights_file = fopen(weights_path.c_str(), "rb");
#endif


#if ! defined(_WIN32) && defined(NNEDI3_DATADIR)
    if (!weights_file) {
        weights_path = std::string(NNEDI3_DATADIR) + "/" + weights_name;
        weights_file = fopen(weights_path.c_str(), "rb");
    }
#endif
    if (!weights_file) {
        vsapi->setError(out, ("nnedi3: Couldn't open file '" + weights_path + "'. Error message: " + strerror(errno)).c_str());
        vsapi->freeNode(d.node);
        return;
    }

    if (fseek(weights_file, 0, SEEK_END)) {
        vsapi->setError(out, ("nnedi3: Failed to seek to the end of '" + weights_path + "'. Error message: " + strerror(errno)).c_str());
        fclose(weights_file);
        vsapi->freeNode(d.node);
        return;
    }

    long expected_size = 13574928; // Version 0.9.4 of the Avisynth plugin.
    long weights_size = ftell(weights_file);
    if (weights_size == -1) {
        vsapi->setError(out, ("nnedi3: Failed to determine the size of '" + weights_path + "'. Error message: " + strerror(errno)).c_str());
        fclose(weights_file);
        vsapi->freeNode(d.node);
        return;
    } else if (weights_size != expected_size) {
        vsapi->setError(out, ("nnedi3: '" + weights_path + "' has the wrong size. Expected " + std::to_string(expected_size) + " bytes, got " + std::to_string(weights_size) + " bytes.").c_str());
        fclose(weights_file);
        vsapi->freeNode(d.node);
        return;
    }

    if (fseek(weights_file, 0, SEEK_SET)) {
        vsapi->setError(out, ("nnedi3: Failed to seek back to the beginning of '" + weights_path + "'. Error message: " + strerror(errno)).c_str());
        fclose(weights_file);
        vsapi->freeNode(d.node);
        return;
    }

    float *bdata = (float *)malloc(expected_size);
    size_t bytes_read = fread(bdata, 1, expected_size, weights_file);

    if (bytes_read != (size_t)expected_size) {
        vsapi->setError(out, ("nnedi3: Expected to read " + std::to_string(expected_size) + " bytes from '" + weights_path + "', read " + std::to_string(bytes_read) + " bytes instead.").c_str());
        fclose(weights_file);
        free(bdata);
        vsapi->freeNode(d.node);
        return;
    }

    fclose(weights_file);

    const int xdiaTable[NUM_NSIZE] = { 8, 16, 32, 48, 8, 16, 32 };
    const int ydiaTable[NUM_NSIZE] = { 6, 6, 6, 6, 4, 4, 4 };
    const int nnsTable[NUM_NNS] = { 16, 32, 64, 128, 256 };

    const int dims0 = 49 * 4 + 5 * 4 + 9 * 4;
    const int dims0new = 4 * 65 + 4 * 5;
    const int dims1 = nnsTable[d.nnsparam] * 2 * (xdiaTable[d.nsize] * ydiaTable[d.nsize] + 1);
    int dims1tsize = 0;
    int dims1offset = 0;

    for (int j = 0; j < NUM_NNS; ++j) {
        for (int i = 0; i < NUM_NSIZE; ++i) {
            if (i == d.nsize && j == d.nnsparam)
                dims1offset = dims1tsize;
            dims1tsize += nnsTable[j] * 2 * (xdiaTable[i] * ydiaTable[i] + 1) * 2;
        }
    }

    d.weights0 = vs_aligned_malloc<float>(std::max(dims0, dims0new) * sizeof(float), 16);

    for (int i = 0; i < 2; ++i)
        d.weights1[i] = vs_aligned_malloc<float>(dims1 * sizeof(float), 16);


    // Adjust prescreener weights
    if (d.pscrn >= 2) {// using new prescreener
        int *offt = (int *)calloc(4 * 64, sizeof(int));
        for (int j = 0; j < 4; ++j)
            for (int k = 0; k < 64; ++k)
                offt[j * 64 + k] = ((k >> 3) << 5) + ((j & 3) << 3) + (k & 7);
        const float *bdw = bdata + dims0 + dims0new * (d.pscrn - 2);
        int16_t *ws = (int16_t *)d.weights0;
        float *wf = (float *)&ws[4 * 64];
        double mean[4] = { 0.0, 0.0, 0.0, 0.0 };
        // Calculate mean weight of each first layer neuron
        for (int j = 0; j < 4; ++j) {
            double cmean = 0.0;
            for (int k = 0; k < 64; ++k)
                cmean += bdw[offt[j * 64 + k]];
            mean[j] = cmean / 64.0;
        }

        // 16 bit pixels will be shifted by 1 for the prescreener.
        const int prescreener_bits = std::min(d.vi.format->bitsPerSample, 15);
        const double half = ((1 << prescreener_bits) - 1) / 2.0;

        // Factor mean removal and 1.0/half scaling
        // into first layer weights. scale to int16 range
        for (int j = 0; j < 4; ++j) {
            double mval = 0.0;
            for (int k = 0; k < 64; ++k)
                mval = std::max(mval, std::fabs((bdw[offt[j * 64 + k]] - mean[j]) / half));
            const double scale = 32767.0 / mval;
            for (int k = 0; k < 64; ++k)
                ws[offt[j * 64 + k]] = roundds(((bdw[offt[j * 64 + k]] - mean[j]) / half) * scale);
            wf[j] = (float)(mval / 32767.0);
        }
        memcpy(wf + 4, bdw + 4 * 64, (dims0new - 4 * 64) * sizeof(float));
        free(offt);
    } else {// using old prescreener
        double mean[4] = { 0.0, 0.0, 0.0, 0.0 };
        // Calculate mean weight of each first layer neuron
        for (int j = 0; j < 4; ++j) {
            double cmean = 0.0;
            for (int k = 0; k < 48; ++k)
                cmean += bdata[j * 48 + k];
            mean[j] = cmean / 48.0;
        }
        if (d.int16_prescreener) {// use int16 dot products in first layer
            int16_t *ws = (int16_t *)d.weights0;
            float *wf = (float *)&ws[4 * 48];

            // 16 bit pixels will be shifted by 1 for the prescreener.
            const int prescreener_bits = std::min(d.vi.format->bitsPerSample, 15);
            const double half = ((1 << prescreener_bits) - 1) / 2.0;

            // Factor mean removal and 1.0/half scaling
            // into first layer weights. scale to int16 range
            for (int j = 0; j < 4; ++j) {
                double mval = 0.0;
                for (int k = 0; k < 48; ++k)
                    mval = std::max(mval, std::fabs((bdata[j * 48 + k] - mean[j]) / half));
                const double scale = 32767.0 / mval;
                for (int k = 0; k < 48; ++k)
                    ws[j * 48 + k] = roundds(((bdata[j * 48 + k] - mean[j]) / half) * scale);
                wf[j] = (float)(mval / 32767.0);
            }
            memcpy(wf + 4, bdata + 4 * 48, (dims0 - 4 * 48) * sizeof(float));
            if (d.opt) {// shuffle weight order for asm
                int16_t *rs = (int16_t *)malloc(dims0 * sizeof(float));
                memcpy(rs, d.weights0, dims0 * sizeof(float));
                for (int j = 0; j < 4; ++j)
                    for (int k = 0; k < 48; ++k)
                        ws[(k >> 3) * 32 + j * 8 + (k & 7)] = rs[j * 48 + k];
                shufflePreScrnL2L3(wf + 8, ((float *)&rs[4 * 48]) + 8);
                free(rs);
            }
        } else {// use float dot products in first layer
            double half = (1 << d.vi.format->bitsPerSample) - 1;
            if (d.vi.format->sampleType == stFloat)
                half = 1.0;
            half /= 2;

            // Factor mean removal and 1.0/half scaling
            // into first layer weights.
            for (int j = 0; j < 4; ++j)
                for (int k = 0; k < 48; ++k)
                    d.weights0[j * 48 + k] = (float)((bdata[j * 48 + k] - mean[j]) / half);
            memcpy(d.weights0 + 4 * 48, bdata + 4 * 48, (dims0 - 4 * 48) * sizeof(float));
            if (d.opt) {// shuffle weight order for asm
                float *wf = d.weights0;
                float *rf = (float *)malloc(dims0 * sizeof(float));
                memcpy(rf, d.weights0, dims0 * sizeof(float));
                for (int j = 0; j < 4; ++j)
                    for (int k = 0; k < 48; ++k)
                        wf[(k >> 2) * 16 + j * 4 + (k & 3)] = rf[j * 48 + k];
                shufflePreScrnL2L3(wf + 4 * 49, rf + 4 * 49);
                free(rf);
            }
        }
    }

    // Adjust prediction weights
    for (int i = 0; i < 2; ++i) {
        const float *bdataT = bdata + dims0 + dims0new * 3 + dims1tsize * d.etype + dims1offset + i * dims1;
        const int nnst = nnsTable[d.nnsparam];
        const int asize = xdiaTable[d.nsize] * ydiaTable[d.nsize];
        const int boff = nnst * 2 * asize;
        double *mean = (double *)calloc(asize + 1 + nnst * 2, sizeof(double));
        // Calculate mean weight of each neuron (ignore bias)
        for (int j = 0; j < nnst * 2; ++j) {
            double cmean = 0.0;
            for (int k = 0; k < asize; ++k)
                cmean += bdataT[j * asize + k];
            mean[asize + 1 + j] = cmean / (double)asize;
        }
        // Calculate mean softmax neuron
        for (int j = 0; j < nnst; ++j) {
            for (int k = 0; k < asize; ++k)
                mean[k] += bdataT[j * asize + k] - mean[asize + 1 + j];
            mean[asize] += bdataT[boff + j];
        }
        for (int j = 0; j < asize + 1; ++j)
            mean[j] /= (double)(nnst);

        if (d.int16_predictor) {// use int16 dot products
            int16_t *ws = (int16_t *)d.weights1[i];
            float *wf = (float *)&ws[nnst * 2 * asize];
            // Factor mean removal into weights, remove global offset from
            // softmax neurons, and scale weights to int16 range.
            for (int j = 0; j < nnst; ++j) {// softmax neurons
                double mval = 0.0;
                for (int k = 0; k < asize; ++k)
                    mval = std::max(mval, std::fabs(bdataT[j * asize + k] - mean[asize + 1 + j] - mean[k]));
                const double scale = 32767.0 / mval;
                for (int k = 0; k < asize; ++k)
                    ws[j * asize + k] = roundds((bdataT[j * asize + k] - mean[asize + 1 + j] - mean[k]) * scale);
                wf[(j >> 2) * 8 + (j & 3)] = (float)(mval / 32767.0);
                wf[(j >> 2) * 8 + (j & 3) + 4] = (float)(bdataT[boff + j] - mean[asize]);
            }
            for (int j = nnst; j < nnst * 2; ++j) {// elliott neurons
                double mval = 0.0;
                for (int k = 0; k < asize; ++k)
                    mval = std::max(mval, std::fabs(bdataT[j * asize + k] - mean[asize + 1 + j]));
                const double scale = 32767.0 / mval;
                for (int k = 0; k < asize; ++k)
                    ws[j * asize + k] = roundds((bdataT[j * asize + k] - mean[asize + 1 + j]) * scale);
                wf[(j >> 2) * 8 + (j & 3)] = (float)(mval / 32767.0);
                wf[(j >> 2) * 8 + (j & 3) + 4] = bdataT[boff + j];
            }
            if (d.opt) {// shuffle weight order for asm
                int16_t *rs = (int16_t *)malloc(nnst * 2 * asize * sizeof(int16_t));
                memcpy(rs, ws, nnst * 2 * asize * sizeof(int16_t));
                for (int j = 0; j < nnst * 2; ++j)
                    for (int k = 0; k < asize; ++k)
                        ws[(j >> 2) * asize * 4 + (k >> 3) * 32 + (j & 3) * 8 + (k & 7)] = rs[j * asize + k];
                free(rs);
            }
        } else {// use float dot products
            // Factor mean removal into weights, and remove global
            // offset from softmax neurons.
            for (int j = 0; j < nnst * 2; ++j) {
                for (int k = 0; k < asize; ++k) {
                    const double q = j < nnst ? mean[k] : 0.0;
                    if (d.opt) // shuffle weight order for asm
                        d.weights1[i][(j >> 2) * asize * 4 + (k >> 2) * 16 + (j & 3) * 4 + (k & 3)] =
                            (float)(bdataT[j * asize + k] - mean[asize + 1 + j] - q);
                    else
                        d.weights1[i][j * asize + k] = (float)(bdataT[j * asize + k] - mean[asize + 1 + j] - q);
                }
                d.weights1[i][boff + j] = (float)(bdataT[boff + j] - (j < nnst ? mean[asize] : 0.0));
            }
        }
        free(mean);
    }

    d.nns = nnsTable[d.nnsparam];
    d.xdia = xdiaTable[d.nsize];
    d.ydia = ydiaTable[d.nsize];
    d.asize = xdiaTable[d.nsize] * ydiaTable[d.nsize];

    free(bdata);


    data = (nnedi3Data *)malloc(sizeof(d));
    *data = d;

    vsapi->createFilter(in, out, "nnedi3", nnedi3Init, nnedi3GetFrame, nnedi3Free, fmParallel, 0, data, core);
}


VS_EXTERNAL_API(void) VapourSynthPluginInit(VSConfigPlugin configFunc, VSRegisterFunction registerFunc, VSPlugin *plugin) {
    configFunc("com.deinterlace.nnedi3", "nnedi3", "Neural network edge directed interpolation (3rd gen.), v" PACKAGE_VERSION, VAPOURSYNTH_API_VERSION, 1, plugin);
    registerFunc("nnedi3",
            "clip:clip;"
            "field:int;"
            "dh:int:opt;"
            "planes:int[]:opt;"
            "nsize:int:opt;"
            "nns:int:opt;"
            "qual:int:opt;"
            "etype:int:opt;"
            "pscrn:int:opt;"
            "opt:int:opt;"
            "int16_prescreener:int:opt;"
            "int16_predictor:int:opt;"
            "exp:int:opt;"
            "show_mask:int:opt;"
            , nnedi3Create, 0, plugin);
}

