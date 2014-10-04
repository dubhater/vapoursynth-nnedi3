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

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <VapourSynth.h>
#include <VSHelper.h>


#define min(a, b)  (((a) < (b)) ? (a) : (b))
#define max(a, b)  (((a) > (b)) ? (a) : (b))


// Functions implemented in nnedi3.asm
extern void nnedi3_uc2s48_SSE2(const uint8_t *t, const int pitch, float *pf);
extern void nnedi3_uc2s64_SSE2(const uint8_t *t, const int pitch, float *p);
extern void nnedi3_computeNetwork0new_SSE2(const float *datai, const float *weights, uint8_t *d);
extern int32_t nnedi3_processLine0_SSE2(const uint8_t *tempu, int width, uint8_t *dstp, const uint8_t *src3p, const int src_pitch);
extern void nnedi3_weightedAvgElliottMul5_m16_SSE2(const float *w, const int n, float *mstd);
extern void nnedi3_extract_m8_i16_SSE2(const uint8_t *srcp, const int stride, const int xdia, const int ydia, float *mstd, float *inputf);
extern void nnedi3_dotProd_m32_m16_i16_SSE2(const float *dataf, const float *weightsf, float *vals, const int n, const int len, const float *istd);
extern void nnedi3_e0_m16_SSE2(float *s, const int n);
extern void nnedi3_castScale_SSE(const float *val, const float *scale, uint8_t *dstp);
extern void nnedi3_computeNetwork0_SSE2(const float *input, const float *weights, uint8_t *d);
extern void nnedi3_uc2f48_SSE2(const uint8_t *t, const int pitch, float *p);
extern void nnedi3_computeNetwork0_i16_SSE2(const float *inputf, const float *weightsf, uint8_t *d);
extern void nnedi3_e1_m16_SSE2(float *s, const int n);
extern void nnedi3_e2_m16_SSE2(float *s, const int n);
extern void nnedi3_extract_m8_SSE2(const uint8_t *srcp, const int stride, const int xdia, const int ydia, float *mstd, float *input);
extern void nnedi3_dotProd_m32_m16_SSE2(const float *data, const float *weights, float *vals, const int n, const int len, const float *istd);
extern void nnedi3_dotProd_m48_m16_i16_SSE2(const float *dataf, const float *weightsf, float *vals, const int n, const int len, const float *istd);
extern void nnedi3_dotProd_m48_m16_SSE2(const float *data, const float *weights, float *vals, const int n, const int len, const float *istd);



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


typedef struct {
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
    int Y, U, V; // used as bool
    int nsize;
    int nnsparam;
    int qual;
    int etype;
    int pscrn;
    int opt;
    int fapprox;

    int max_value;

    void (*copyPad)(const VSFrameRef *, FrameData *, void **, int, const VSAPI *);
    void (*evalFunc_0)(void **, FrameData *);
    void (*evalFunc_1)(void **, FrameData *);

    // Functions used in evalFunc_0
    void (*uc2s)(const uint8_t*, const int, float*);
    void (*computeNetwork0)(const float*, const float*, uint8_t *);
    int32_t (*processLine0)(const uint8_t*, int, uint8_t*, const uint8_t*, const int, const int);

    // Functions used in evalFunc_1
    void (*extract)(const uint8_t*, const int, const int, const int, float*, float*);
    void (*dotProd)(const float*, const float*, float*, const int, const int, const float*);
    void (*expfunc)(float *, const int);
    void (*wae5)(const float*, const int, float*);
} nnedi3Data;



static void copyPad(const VSFrameRef *src, FrameData *frameData, void **instanceData, int fn, const VSAPI *vsapi) {
    const int off = 1 - fn;

    nnedi3Data *d = (nnedi3Data *) * instanceData;

    for (int plane = 0; plane < d->vi.format->numPlanes; ++plane) {
        const uint8_t *srcp = vsapi->getReadPtr(src, plane);
        uint8_t *dstp = frameData->paddedp[plane];

        const int src_stride = vsapi->getStride(src, plane);
        const int dst_stride = frameData->padded_stride[plane];

        const int src_height = vsapi->getFrameHeight(src, plane);
        const int dst_height = frameData->padded_height[plane];

        const int src_width = vsapi->getFrameWidth(src, plane);
        const int dst_width = frameData->padded_width[plane];

        // Copy.
        if (!d->dh) {
            for (int y = off; y < src_height; y += 2) {
                memcpy(dstp + 32 + (6+y)*dst_stride,
                        srcp + y*src_stride,
                        src_width);
            }
        } else {
            for (int y = 0; y < src_height; y++) {
                memcpy(dstp + 32 + (6+y*2+off)*dst_stride,
                        srcp + y*src_stride,
                        src_width);
            }
        }

        // And pad.
        dstp += (6+off)*dst_stride;
        for (int y = 6 + off; y < dst_height - 6; y += 2) {
            for (int x = 0; x < 32; ++x) {
                dstp[x] = dstp[64-x];
            }
            int c = 2;
            for (int x = dst_width - 32; x < dst_width; ++x, c += 2) {
                dstp[x] = dstp[x-c];
            }
            dstp += dst_stride*2;
        }

        dstp = frameData->paddedp[plane];
        for (int y = off; y < 6; y += 2) {
            memcpy(dstp + y*dst_stride,
                    dstp + (12+2*off-y)*dst_stride,
                    dst_width);
        }
        int c = 4;
        for (int y = dst_height - 6 + off; y < dst_height; y += 2, c += 4) {
            memcpy(dstp + y*dst_stride,
                    dstp + (y-c)*dst_stride,
                    dst_width);
        }
    }
}


static void copyPad_uint16(const VSFrameRef *src, FrameData *frameData, void **instanceData, int fn, const VSAPI *vsapi) {
    const int off = 1 - fn;

    nnedi3Data *d = (nnedi3Data *) * instanceData;

    for (int plane = 0; plane < d->vi.format->numPlanes; ++plane) {
        const uint16_t *srcp16 = (const uint16_t *)vsapi->getReadPtr(src, plane);
        uint16_t *dstp16 = (uint16_t *)frameData->paddedp[plane];

        const int src_stride = vsapi->getStride(src, plane) / 2;
        const int dst_stride = frameData->padded_stride[plane] / 2;

        const int src_height = vsapi->getFrameHeight(src, plane);
        const int dst_height = frameData->padded_height[plane];

        const int src_width = vsapi->getFrameWidth(src, plane);
        const int dst_width = frameData->padded_width[plane];

        // Copy.
        if (!d->dh) {
            for (int y = off; y < src_height; y += 2) {
                memcpy(dstp16 + 32 + (6+y)*dst_stride,
                        srcp16 + y*src_stride,
                        src_width * 2);
            }
        } else {
            for (int y = 0; y < src_height; y++) {
                memcpy(dstp16 + 32 + (6+y*2+off)*dst_stride,
                        srcp16 + y*src_stride,
                        src_width * 2);
            }
        }

        // And pad.
        dstp16 += (6+off)*dst_stride;
        for (int y = 6 + off; y < dst_height - 6; y += 2) {
            for (int x = 0; x < 32; ++x) {
                dstp16[x] = dstp16[64-x];
            }
            int c = 2;
            for (int x = dst_width - 32; x < dst_width; ++x, c += 2) {
                dstp16[x] = dstp16[x-c];
            }
            dstp16 += dst_stride*2;
        }

        dstp16 = (uint16_t *)frameData->paddedp[plane];
        for (int y = off; y < 6; y += 2) {
            memcpy(dstp16 + y*dst_stride,
                    dstp16 + (12+2*off-y)*dst_stride,
                    dst_width * 2);
        }
        int c = 4;
        for (int y = dst_height - 6 + off; y < dst_height; y += 2, c += 4) {
            memcpy(dstp16 + y*dst_stride,
                    dstp16 + (y-c)*dst_stride,
                    dst_width * 2);
        }
    }
}


void elliott_C(float *data, const int n)
{
    for (int i=0; i<n; ++i)
        data[i] = data[i]/(1.0f+fabsf(data[i]));
}


void dotProd_C(const float *data, const float *weights, 
        float *vals, const int n, const int len, const float *scale)
{
    for (int i=0; i<n; ++i)
    {
        float sum = 0.0f;
        for (int j=0; j<len; ++j)
            sum += data[j]*weights[i*len+j];
        vals[i] = sum*scale[0]+weights[n*len+i];
    }
}


void dotProdS_C(const float *dataf, const float *weightsf, 
        float *vals, const int n, const int len, const float *scale)
{
    const int16_t *data = (int16_t*)dataf;
    const int16_t *weights = (int16_t*)weightsf;
    const float *wf = (float*)&weights[n*len];
    for (int i=0; i<n; ++i)
    {
        int sum = 0, off = ((i>>2)<<3)+(i&3);
        for (int j=0; j<len; ++j)
            sum += data[j]*weights[i*len+j];
        vals[i] = sum*wf[off]*scale[0]+wf[off+4];
    }
}


void computeNetwork0_C(const float *input, const float *weights, uint8_t *d)
{
    float temp[12], scale = 1.0f;
    dotProd_C(input,weights,temp,4,48,&scale);
    const float t = temp[0];
    elliott_C(temp,4);
    temp[0] = t;
    dotProd_C(temp,weights+4*49,temp+4,4,4,&scale);
    elliott_C(temp+4,4);
    dotProd_C(temp,weights+4*49+4*5,temp+8,4,8,&scale);
    if (max(temp[10],temp[11]) <= max(temp[8],temp[9]))
        d[0] = 1;
    else
        d[0] = 0;
}


void computeNetwork0_i16_C(const float *inputf, const float *weightsf, uint8_t *d)
{
    const float *wf = weightsf+2*48;
    float temp[12], scale = 1.0f;
    dotProdS_C(inputf,weightsf,temp,4,48,&scale);
    const float t = temp[0];
    elliott_C(temp,4);
    temp[0] = t;
    dotProd_C(temp,wf+8,temp+4,4,4,&scale);
    elliott_C(temp+4,4);
    dotProd_C(temp,wf+8+4*5,temp+8,4,8,&scale);
    if (max(temp[10],temp[11]) <= max(temp[8],temp[9]))
        d[0] = 1;
    else
        d[0] = 0;
}


void uc2f48_C(const uint8_t *t, const int pitch, float *p)
{
    for (int y=0; y<4; ++y)
        for (int x=0; x<12; ++x)
            p[y*12+x] = t[y*pitch*2+x];
}


// Name's wrong now. The input is no longer unsigned char.
void uc2f48_uint16_C(const uint8_t *t8, const int pitch, float *p)
{
    const uint16_t *t = (const uint16_t *)t8;

    for (int y=0; y<4; ++y)
        for (int x=0; x<12; ++x)
            p[y*12+x] = t[y*pitch*2+x];
}


void uc2s48_C(const uint8_t *t, const int pitch, float *pf)
{
    int16_t *p = (int16_t*)pf;
    for (int y=0; y<4; ++y)
        for (int x=0; x<12; ++x)
            p[y*12+x] = t[y*pitch*2+x];
}

#define CB2(n) max(min((n),254),0)

int32_t processLine0_maybeSSE2(const uint8_t *tempu, int width, uint8_t *dstp,
        const uint8_t *src3p, const int src_pitch, const int max_value) {
    int32_t count = 0;
    const int remain = width & 15;
    width -= remain;
    if (width)
        count = nnedi3_processLine0_SSE2(tempu, width, dstp, src3p, src_pitch);
    for (int x = width; x < width + remain; ++x) {
        if (tempu[x]) {
            dstp[x] = CB2((19*(src3p[x+src_pitch*2]+src3p[x+src_pitch*4])-
                        3*(src3p[x]+src3p[x+src_pitch*6])+16)>>5);
        } else {
            dstp[x] = 255;
            ++count;
        }
    }
    return count;
}


int32_t processLine0_C(const uint8_t *tempu, int width, uint8_t *dstp,
        const uint8_t *src3p, const int src_pitch, const int max_value)
{
    int count = 0;
    for (int x=0; x<width; ++x)
    {
        if (tempu[x])
            dstp[x] = CB2((19 * (src3p[x+src_pitch*2] + src3p[x+src_pitch*4]) - 3 * (src3p[x] + src3p[x+src_pitch*6]) + 16) / 32);
        else
        {
            dstp[x] = 255;
            ++count;
        }
    }
    return count;
}

#undef CB2


int32_t processLine0_uint16_C(const uint8_t *tempu, int width, uint8_t *dstp8,
        const uint8_t *src3p8, const int src_pitch, const int max_value)
{
    uint16_t *dstp = (uint16_t *)dstp8;
    const uint16_t *src3p = (const uint16_t *)src3p8;

    int count = 0;
    for (int x=0; x<width; ++x)
    {
        if (tempu[x])
        {
            int tmp = (19 * (src3p[x+src_pitch*2] + src3p[x+src_pitch*4]) - 3 * (src3p[x] + src3p[x+src_pitch*6]) + 16) / 32;
            dstp[x] = VSMAX(VSMIN(tmp, max_value - 1), 0);
            // Technically the -1 is only needed for 16 bit input.
        }
        else
        {
            dstp[x] = 65535;
            ++count;
        }
    }
    return count;
}

// new prescreener functions
void uc2s64_C(const uint8_t *t, const int pitch, float *p)
{
    int16_t *ps = (int16_t*)p;
    for (int y=0; y<4; ++y)
        for (int x=0; x<16; ++x)
            ps[y*16+x] = t[y*pitch*2+x];
}


void computeNetwork0new_C(const float *datai, const float *weights, uint8_t *d)
{
    int16_t *data = (int16_t*)datai;
    int16_t *ws = (int16_t*)weights;
    float *wf = (float*)&ws[4*64];
    float vals[8];
    for (int i=0; i<4; ++i)
    {
        int sum = 0;
        for (int j=0; j<64; ++j)
            sum += data[j]*ws[(i<<3)+((j>>3)<<5)+(j&7)];
        const float t = sum*wf[i]+wf[4+i];
        vals[i] = t/(1.0f+fabsf(t));
    }
    for (int i=0; i<4; ++i)
    {
        float sum = 0.0f;
        for (int j=0; j<4; ++j)
            sum += vals[j]*wf[8+i+(j<<2)];
        vals[4+i] = sum+wf[8+16+i];
    }
    int mask = 0;
    for (int i=0; i<4; ++i)
    {
        if (vals[4+i]>0.0f)
            mask |= (0x1<<(i<<3));
    }
    ((int*)d)[0] = mask;
}


void evalFunc_0(void **instanceData, FrameData *frameData)
{
    nnedi3Data *d = (nnedi3Data*) * instanceData;

    float *input = frameData->input;
    const float *weights0 = d->weights0;
    float *temp = frameData->temp;
    uint8_t *tempu = (uint8_t*)temp;

    // And now the actual work.
    for (int b = 0; b < d->vi.format->numPlanes; ++b)
    {
        if ((b == 0 && !d->Y) || 
                (b == 1 && !d->U) ||
                (b == 2 && !d->V))
            continue;

        const uint8_t *srcp = frameData->paddedp[b];
        const int src_stride = frameData->padded_stride[b];
        const int width = frameData->padded_width[b];
        const int height = frameData->padded_height[b];
        uint8_t *dstp = frameData->dstp[b];
        const int dst_stride = frameData->dst_stride[b];

        for (int y = 1 - frameData->field[b]; y < height - 12; y += 2) {
            memcpy(dstp + y*dst_stride,
                    srcp + 32 + (6+y)*src_stride,
                    width - 64);
        }

        const int ystart = 6 + frameData->field[b];
        const int ystop = height - 6;
        srcp += ystart*src_stride;
        dstp += (ystart-6)*dst_stride-32;
        const uint8_t *src3p = srcp-src_stride*3;
        int32_t *lcount = frameData->lcount[b]-6;
        if (d->pscrn == 1) // original
        {
            for (int y=ystart; y<ystop; y+=2)
            {
                for (int x=32; x<width-32; ++x)
                {
                    d->uc2s(src3p+x-5,src_stride,input);
                    d->computeNetwork0(input,weights0,tempu+x);
                }
                lcount[y] += d->processLine0(tempu+32,width-64,dstp+32,src3p+32,src_stride, 42); // 42 is the answer
                src3p += src_stride*2;
                dstp += dst_stride*2;
            }
        }
        else if (d->pscrn >= 2) // new
        {
            for (int y=ystart; y<ystop; y+=2)
            {
                for (int x=32; x<width-32; x+=4)
                {
                    d->uc2s(src3p+x-6,src_stride,input);
                    d->computeNetwork0(input,weights0,tempu+x);
                }
                lcount[y] += d->processLine0(tempu+32,width-64,dstp+32,src3p+32,src_stride, 42);
                src3p += src_stride*2;
                dstp += dst_stride*2;
            }
        }
        else // no prescreening
        {
            for (int y=ystart; y<ystop; y+=2)
            {
                memset(dstp+32,255,width-64);
                lcount[y] += width-64;
                dstp += dst_stride*2;
            }
        }
    }
}


void evalFunc_0_uint16(void **instanceData, FrameData *frameData)
{
    nnedi3Data *d = (nnedi3Data*) * instanceData;

    float *input = frameData->input;
    const float *weights0 = d->weights0;
    float *temp = frameData->temp;
    uint8_t *tempu = (uint8_t*)temp;

    // And now the actual work.
    for (int b = 0; b < d->vi.format->numPlanes; ++b)
    {
        if ((b == 0 && !d->Y) || 
                (b == 1 && !d->U) ||
                (b == 2 && !d->V))
            continue;

        const uint16_t *srcp16 = (const uint16_t *)frameData->paddedp[b];
        const int src_stride = frameData->padded_stride[b] / 2;

        const int width = frameData->padded_width[b];
        const int height = frameData->padded_height[b];

        uint16_t *dstp16 = (uint16_t *)frameData->dstp[b];
        const int dst_stride = frameData->dst_stride[b] / 2;

        for (int y = 1 - frameData->field[b]; y < height - 12; y += 2) {
            memcpy(dstp16 + y*dst_stride,
                    srcp16 + 32 + (6+y)*src_stride,
                    (width - 64) * 2);
        }

        const int ystart = 6 + frameData->field[b];
        const int ystop = height - 6;
        srcp16 += ystart*src_stride;
        dstp16 += (ystart-6)*dst_stride-32;
        const uint16_t *src3p16 = srcp16 - src_stride*3;
        int32_t *lcount = frameData->lcount[b]-6;
        if (d->pscrn == 1) // original
        {
            for (int y=ystart; y<ystop; y+=2)
            {
                for (int x=32; x<width-32; ++x)
                {
                    d->uc2s((const uint8_t *)(src3p16 + x - 5), src_stride, input);
                    d->computeNetwork0(input, weights0, tempu+x);
                }
                lcount[y] += d->processLine0(tempu+32, width-64, (uint8_t *)(dstp16 + 32), (const uint8_t *)(src3p16 + 32), src_stride, d->max_value);
                src3p16 += src_stride*2;
                dstp16 += dst_stride*2;
            }
        }
        else // no prescreening
        {
            for (int y=ystart; y<ystop; y+=2)
            {
                memset(dstp16 + 32, 255, (width - 64) * 2);
                lcount[y] += width-64;
                dstp16 += dst_stride*2;
            }
        }
    }
}


void extract_m8_C(const uint8_t *srcp, const int stride, 
        const int xdia, const int ydia, float *mstd, float *input)
{
    int sum = 0, sumsq = 0;
    for (int y=0; y<ydia; ++y)
    {
        const uint8_t *srcpT = srcp+y*stride*2;
        for (int x=0; x<xdia; ++x, ++input)
        {
            sum += srcpT[x];
            sumsq += srcpT[x]*srcpT[x];
            input[0] = srcpT[x];
        }
    }
    const float scale = 1.0f/(float)(xdia*ydia);
    mstd[0] = sum*scale;
    mstd[1] = sumsq*scale-mstd[0]*mstd[0];
    mstd[3] = 0.0f;
    if (mstd[1] <= FLT_EPSILON)
        mstd[1] = mstd[2] = 0.0f;
    else
    {
        mstd[1] = sqrtf(mstd[1]);
        mstd[2] = 1.0f/mstd[1];
    }
}


void extract_m8_uint16_C(const uint8_t *srcp8, const int stride, 
        const int xdia, const int ydia, float *mstd, float *input)
{
    const uint16_t *srcp = (const uint16_t *)srcp8;

    int64_t sum = 0, sumsq = 0;
    for (int y=0; y<ydia; ++y)
    {
        const uint16_t *srcpT = srcp+y*stride*2;
        for (int x=0; x<xdia; ++x, ++input)
        {
            sum += srcpT[x];
            sumsq += (uint32_t)srcpT[x]*(uint32_t)srcpT[x];
            input[0] = srcpT[x];
        }
    }
    const float scale = 1.0f/(xdia*ydia);
    mstd[0] = sum*scale;
    const double tmp = (double)sumsq*scale - (double)mstd[0]*mstd[0];
    mstd[3] = 0.0f;
    if (tmp <= FLT_EPSILON)
        mstd[1] = mstd[2] = 0.0f;
    else
    {
        mstd[1] = (float)sqrt(tmp);
        mstd[2] = 1.0f/mstd[1];
    }
}


void extract_m8_i16_C(const uint8_t *srcp, const int stride, 
        const int xdia, const int ydia, float *mstd, float *inputf)
{
    int16_t *input = (int16_t*)inputf;
    int sum = 0, sumsq = 0;
    for (int y=0; y<ydia; ++y)
    {
        const uint8_t *srcpT = srcp+y*stride*2;
        for (int x=0; x<xdia; ++x, ++input)
        {
            sum += srcpT[x];
            sumsq += srcpT[x]*srcpT[x];
            input[0] = srcpT[x];
        }
    }
    const float scale = 1.0f/(float)(xdia*ydia);
    mstd[0] = sum*scale;
    mstd[1] = sumsq*scale-mstd[0]*mstd[0];
    mstd[3] = 0.0f;
    if (mstd[1] <= FLT_EPSILON)
        mstd[1] = mstd[2] = 0.0f;
    else
    {
        mstd[1] = sqrtf(mstd[1]);
        mstd[2] = 1.0f/mstd[1];
    }
}


const float exp_lo[4] = { -80.0f, -80.0f, -80.0f, -80.0f };
const float exp_hi[4] = { +80.0f, +80.0f, +80.0f, +80.0f };


// exp from:  A Fast, Compact Approximation of the Exponential Function (1998)
//            Nicol N. Schraudolph


const float e0_mult[4] = { // (1.0/ln(2))*(2^23)
    12102203.161561486f, 12102203.161561486f, 12102203.161561486f, 12102203.161561486f };
const float e0_bias[4] = { // (2^23)*127.0-486411.0
    1064866805.0f, 1064866805.0f, 1064866805.0f, 1064866805.0f };


void e0_m16_C(float *s, const int n)
{
    for (int i=0; i<n; ++i)
    {
        const int t = (int)(max(min(s[i],exp_hi[0]),exp_lo[0])*e0_mult[0]+e0_bias[0]);
        s[i] = (*((float*)&t));
    }
}


// exp from Loren Merritt


const float e1_scale[4] = { // 1/ln(2)
    1.4426950409f, 1.4426950409f, 1.4426950409f, 1.4426950409f };
const float e1_bias[4] = { // 3<<22
    12582912.0f, 12582912.0f, 12582912.0f, 12582912.0f };
const float e1_c0[4] = { 1.00035f, 1.00035f, 1.00035f, 1.00035f };
const float e1_c1[4] = { 0.701277797f, 0.701277797f, 0.701277797f, 0.701277797f };
const float e1_c2[4] = { 0.237348593f, 0.237348593f, 0.237348593f, 0.237348593f };


void e1_m16_C(float *s, const int n)
{
    for (int q=0; q<n; ++q)
    {
        float x = max(min(s[q],exp_hi[0]),exp_lo[0])*e1_scale[0];
        int i = (int)(x + 128.5f) - 128;
        x -= i;
        x = e1_c0[0] + e1_c1[0]*x + e1_c2[0]*x*x;
        i = (i+127)<<23;
        s[q] = x * *((float*)&i);
    }
}


void e2_m16_C(float *s, const int n)
{
    for (int i=0; i<n; ++i)
        s[i] = expf(max(min(s[i],exp_hi[0]),exp_lo[0]));
}

// exp from Intel Approximate Math (AM) Library


const float min_weight_sum[4] = { 1e-10f, 1e-10f, 1e-10f, 1e-10f };


void weightedAvgElliottMul5_m16_C(const float *w, const int n, float *mstd)
{
    float vsum = 0.0f, wsum = 0.0f;
    for (int i=0; i<n; ++i)
    {
        vsum += w[i]*(w[n+i]/(1.0f+fabsf(w[n+i])));
        wsum += w[i];
    }
    if (wsum > min_weight_sum[0])
        mstd[3] += ((5.0f*vsum)/wsum)*mstd[1]+mstd[0];
    else
        mstd[3] += mstd[0];
}


void evalFunc_1(void **instanceData, FrameData *frameData)
{
    nnedi3Data *d = (nnedi3Data*) * instanceData;

    float *input = frameData->input;
    float *temp = frameData->temp;
    float **weights1 = d->weights1;
    const int opt = d->opt;
    const int qual = d->qual;
    const int asize = d->asize;
    const int nns = d->nns;
    const int xdia = d->xdia;
    const int xdiad2m1 = (xdia / 2) - 1;
    const int ydia = d->ydia;
    const float scale = 1.0f/(float)qual;

    for (int b = 0; b < d->vi.format->numPlanes; ++b)
    {
        if ((b == 0 && !d->Y) || 
                (b == 1 && !d->U) ||
                (b == 2 && !d->V))
            continue;

        const uint8_t *srcp = frameData->paddedp[b];
        const int src_stride = frameData->padded_stride[b];
        const int width = frameData->padded_width[b];
        const int height = frameData->padded_height[b];
        uint8_t *dstp = frameData->dstp[b];
        const int dst_stride = frameData->dst_stride[b];
        const int ystart = frameData->field[b];
        const int ystop = height - 12;
        srcp += (ystart+6)*src_stride;
        dstp += ystart*dst_stride-32;
        const uint8_t *srcpp = srcp-(ydia-1)*src_stride-xdiad2m1;
        for (int y=ystart; y<ystop; y+=2)
        {
            for (int x=32; x<width-32; ++x)
            {
                if (dstp[x] != 255)
                    continue;

                float mstd[4];
                d->extract(srcpp+x,src_stride,xdia,ydia,mstd,input);
                for (int i=0; i<qual; ++i)
                {
                    d->dotProd(input,weights1[i],temp,nns*2,asize,mstd+2);
                    d->expfunc(temp,nns);
                    d->wae5(temp,nns,mstd);
                }
                if (opt > 1)
                    nnedi3_castScale_SSE(mstd,&scale,dstp+x);
                else
                    dstp[x] = min(max((int)(mstd[3]*scale+0.5f),0),255);
            }
            srcpp += src_stride*2;
            dstp += dst_stride*2;
        }
    }
}


void evalFunc_1_uint16(void **instanceData, FrameData *frameData)
{
    nnedi3Data *d = (nnedi3Data*) * instanceData;

    float *input = frameData->input;
    float *temp = frameData->temp;
    float **weights1 = d->weights1;
    const int qual = d->qual;
    const int asize = d->asize;
    const int nns = d->nns;
    const int xdia = d->xdia;
    const int xdiad2m1 = (xdia / 2) - 1;
    const int ydia = d->ydia;
    const float scale = 1.0f/(float)qual;

    for (int b = 0; b < d->vi.format->numPlanes; ++b)
    {
        if ((b == 0 && !d->Y) || 
                (b == 1 && !d->U) ||
                (b == 2 && !d->V))
            continue;

        const uint16_t *srcp16 = (const uint16_t *)frameData->paddedp[b];
        const int src_stride = frameData->padded_stride[b] / 2;

        const int width = frameData->padded_width[b];
        const int height = frameData->padded_height[b];

        uint16_t *dstp16 = (uint16_t *)frameData->dstp[b];
        const int dst_stride = frameData->dst_stride[b] / 2;

        const int ystart = frameData->field[b];
        const int ystop = height - 12;

        srcp16 += (ystart+6)*src_stride;
        dstp16 += ystart*dst_stride-32;
        const uint16_t *srcpp16 = srcp16 - (ydia-1)*src_stride - xdiad2m1;

        for (int y=ystart; y<ystop; y+=2)
        {
            for (int x=32; x<width-32; ++x)
            {
                if (dstp16[x] != 65535)
                    continue;

                float mstd[4];
                d->extract((const uint8_t *)(srcpp16 + x), src_stride, xdia, ydia, mstd, input);
                for (int i=0; i<qual; ++i)
                {
                    d->dotProd(input, weights1[i], temp, nns*2, asize, mstd+2);
                    d->expfunc(temp, nns);
                    d->wae5(temp, nns, mstd);
                }
                dstp16[x] = VSMIN(VSMAX((int)(mstd[3]*scale+0.5f), 0), d->max_value);
            }
            srcpp16 += src_stride*2;
            dstp16 += dst_stride*2;
        }
    }
}


#define NUM_NSIZE 7
#define NUM_NNS 5


int roundds(const double f)
{
    if (f-floor(f) >= 0.5)
        return min((int)ceil(f),32767);
    return max((int)floor(f),-32768);
}


void shufflePreScrnL2L3(float *wf, float *rf, const int opt)
{
    for (int j=0; j<4; ++j)
        for (int k=0; k<4; ++k)
            wf[k*4+j] = rf[j*4+k];
    rf += 4*5;
    wf += 4*5;
    const int jtable[4] = { 0, 2, 1, 3 };
    for (int j=0; j<4; ++j)
    {
        for (int k=0; k<8; ++k)
            wf[k*4+j] = rf[jtable[j]*8+k];
        wf[4*8+j] = rf[4*8+jtable[j]];
    }
}


static void selectFunctions(nnedi3Data *d) {
    d->copyPad = copyPad;
    d->evalFunc_0 = evalFunc_0;
    d->evalFunc_1 = evalFunc_1;

    // evalFunc_0
    if (d->opt == 1)
        d->processLine0 = processLine0_C;
    else
        d->processLine0 = processLine0_maybeSSE2;

    if (d->pscrn < 2) { // original prescreener
        if (d->fapprox & 1) { // int16 dot products
            if (d->opt == 1) {
                d->uc2s = uc2s48_C;
                d->computeNetwork0 = computeNetwork0_i16_C;
            } else {
                d->uc2s = nnedi3_uc2s48_SSE2;
                d->computeNetwork0 = nnedi3_computeNetwork0_i16_SSE2;
            }
        } else {
            if (d->opt == 1) {
                d->uc2s = uc2f48_C;
                d->computeNetwork0 = computeNetwork0_C;
            } else {
                d->uc2s = nnedi3_uc2f48_SSE2;
                d->computeNetwork0 = nnedi3_computeNetwork0_SSE2;
            }
        }
    } else { // new prescreener
        // only int16 dot products
        if (d->opt == 1) {
            d->uc2s = uc2s64_C;
            d->computeNetwork0 = computeNetwork0new_C;
        } else {
            d->uc2s = nnedi3_uc2s64_SSE2;
            d->computeNetwork0 = nnedi3_computeNetwork0new_SSE2;
        }
    }

    // evalFunc_1
    if (d->opt == 1)
        d->wae5 = weightedAvgElliottMul5_m16_C;
    else
        d->wae5 = nnedi3_weightedAvgElliottMul5_m16_SSE2;

    if (d->fapprox & 2) { // use int16 dot products
        if (d->opt == 1) {
            d->extract = extract_m8_i16_C;
            d->dotProd = dotProdS_C;
        } else {
            d->extract = nnedi3_extract_m8_i16_SSE2;
            d->dotProd = (d->asize % 48) ? nnedi3_dotProd_m32_m16_i16_SSE2 : nnedi3_dotProd_m48_m16_i16_SSE2;
        }
    } else { // use float dot products
        if (d->opt == 1) {
            d->extract = extract_m8_C;
            d->dotProd = dotProd_C;
        } else {
            d->extract = nnedi3_extract_m8_SSE2;
            d->dotProd = (d->asize % 48) ? nnedi3_dotProd_m32_m16_SSE2 : nnedi3_dotProd_m48_m16_SSE2;
        }
    }

    if ((d->fapprox & 12) == 0) { // use slow exp
        if (d->opt == 1)
            d->expfunc = e2_m16_C;
        else
            d->expfunc = nnedi3_e2_m16_SSE2;
    } else if ((d->fapprox & 12) == 4) { // use faster exp
        if (d->opt == 1)
            d->expfunc = e1_m16_C;
        else
            d->expfunc = nnedi3_e1_m16_SSE2;
    } else { // use fastest exp
        if (d->opt == 1)
            d->expfunc = e0_m16_C;
        else
            d->expfunc = nnedi3_e0_m16_SSE2;
    }
}


static void selectFunctions_uint16(nnedi3Data *d) {
    d->copyPad = copyPad_uint16;
    d->evalFunc_0 = evalFunc_0_uint16;
    d->evalFunc_1 = evalFunc_1_uint16;

    if (d->opt == 1) {
        // evalFunc_0
        d->processLine0 = processLine0_uint16_C;

        d->uc2s = uc2f48_uint16_C;
        d->computeNetwork0 = computeNetwork0_C;

        // evalFunc_1
        d->wae5 = weightedAvgElliottMul5_m16_C;

        d->extract = extract_m8_uint16_C;
        d->dotProd = dotProd_C;

        if ((d->fapprox & 12) == 0) { // use slow exp
            d->expfunc = e2_m16_C;
        } else if ((d->fapprox & 12) == 4) { // use faster exp
            d->expfunc = e1_m16_C;
        } else { // use fastest exp
            d->expfunc = e0_m16_C;
        }
    } else { // opt == 2
        // evalFunc_0
        d->processLine0 = processLine0_uint16_C; // C works too

        d->uc2s = uc2f48_uint16_C; // C works too
        d->computeNetwork0 = nnedi3_computeNetwork0_SSE2;

        // evalFunc_1
        d->wae5 = nnedi3_weightedAvgElliottMul5_m16_SSE2;

        d->extract = extract_m8_uint16_C; // C works too
        d->dotProd = (d->asize % 48) ? nnedi3_dotProd_m32_m16_SSE2 : nnedi3_dotProd_m48_m16_SSE2;

        if ((d->fapprox & 12) == 0) { // use slow exp
            d->expfunc = nnedi3_e2_m16_SSE2;
        } else if ((d->fapprox & 12) == 4) { // use faster exp
            d->expfunc = nnedi3_e1_m16_SSE2;
        } else { // use fastest exp
            d->expfunc = nnedi3_e0_m16_SSE2;
        }
    }
}


// From binary1.asm
extern uint8_t binary1;


int dims1offset;
static void VS_CC nnedi3Init(VSMap *in, VSMap *out, void **instanceData, VSNode *node, VSCore *core, const VSAPI *vsapi) {
    nnedi3Data *d = (nnedi3Data *) * instanceData;
    vsapi->setVideoInfo(&d->vi, 1, node);

    const float* bdata = (const float*)&binary1;

    const int xdiaTable[NUM_NSIZE] = { 8, 16, 32, 48, 8, 16, 32 };
    const int ydiaTable[NUM_NSIZE] = { 6, 6, 6, 6, 4, 4, 4 };
    const int nnsTable[NUM_NNS] = { 16, 32, 64, 128, 256 };

    const int dims0 = 49*4+5*4+9*4;
    const int dims0new = 4*65+4*5;
    const int dims1 = nnsTable[d->nnsparam]*2*(xdiaTable[d->nsize]*ydiaTable[d->nsize]+1);
    int dims1tsize = 0;

    for (int j=0; j<NUM_NNS; ++j)
    {
        for (int i=0; i<NUM_NSIZE; ++i)
        {
            if (i == d->nsize && j == d->nnsparam) {
                dims1offset = dims1tsize;
            }
            dims1tsize += nnsTable[j]*2*(xdiaTable[i]*ydiaTable[i]+1)*2;
        }
    }

    VS_ALIGNED_MALLOC(&d->weights0, max(dims0, dims0new) * sizeof(float), 16);

    for (int i = 0; i < 2; ++i)
    {
        VS_ALIGNED_MALLOC(&d->weights1[i], dims1 * sizeof(float), 16);
    }


    // Adjust prescreener weights
    if (d->pscrn >= 2) // using new prescreener
    {
        int *offt = (int*)calloc(4*64,sizeof(int));
        for (int j=0; j<4; ++j)
            for (int k=0; k<64; ++k)
                offt[j*64+k] = ((k>>3)<<5)+((j&3)<<3)+(k&7);
        const float *bdw = bdata+dims0+dims0new*(d->pscrn-2);
        int16_t *ws = (int16_t*)d->weights0;
        float *wf = (float*)&ws[4*64];
        double mean[4] = { 0.0, 0.0, 0.0, 0.0 };
        // Calculate mean weight of each first layer neuron
        for (int j=0; j<4; ++j)
        {
            double cmean = 0.0;
            for (int k=0; k<64; ++k)
                cmean += bdw[offt[j*64+k]];
            mean[j] = cmean/64.0;
        }
        // Factor mean removal and 1.0/127.5 scaling 
        // into first layer weights. scale to int16 range
        for (int j=0; j<4; ++j)
        {
            double mval = 0.0;
            for (int k=0; k<64; ++k)
                mval = max(mval,fabs((bdw[offt[j*64+k]]-mean[j])/127.5));
            const double scale = 32767.0/mval;
            for (int k=0; k<64; ++k)
                ws[offt[j*64+k]] = roundds(((bdw[offt[j*64+k]]-mean[j])/127.5)*scale);
            wf[j] = (float)(mval/32767.0);
        }
        memcpy(wf+4,bdw+4*64,(dims0new-4*64)*sizeof(float));
        free(offt);
    }
    else // using old prescreener
    {
        double mean[4] = { 0.0, 0.0, 0.0, 0.0 };
        // Calculate mean weight of each first layer neuron
        for (int j=0; j<4; ++j)
        {
            double cmean = 0.0;
            for (int k=0; k<48; ++k)
                cmean += bdata[j*48+k];
            mean[j] = cmean/48.0;
        }
        if (d->fapprox & 1) // use int16 dot products in first layer
        {
            int16_t *ws = (int16_t*)d->weights0;
            float *wf = (float*)&ws[4*48];
            // Factor mean removal and 1.0/127.5 scaling 
            // into first layer weights. scale to int16 range
            for (int j=0; j<4; ++j)
            {
                double mval = 0.0;
                for (int k=0; k<48; ++k)
                    mval = max(mval,fabs((bdata[j*48+k]-mean[j])/127.5));
                const double scale = 32767.0/mval;
                for (int k=0; k<48; ++k)
                    ws[j*48+k] = roundds(((bdata[j*48+k]-mean[j])/127.5)*scale);
                wf[j] = (float)(mval/32767.0);
            }
            memcpy(wf+4,bdata+4*48,(dims0-4*48)*sizeof(float));
            if (d->opt > 1) // shuffle weight order for asm
            {
                int16_t *rs = (int16_t*)malloc(dims0*sizeof(float));
                memcpy(rs,d->weights0,dims0*sizeof(float));
                for (int j=0; j<4; ++j)
                    for (int k=0; k<48; ++k)
                        ws[(k>>3)*32+j*8+(k&7)] = rs[j*48+k];
                shufflePreScrnL2L3(wf+8,((float*)&rs[4*48])+8,d->opt);
                free(rs);
            }
        }
        else // use float dot products in first layer
        {
            // Factor mean removal and 1.0/127.5 scaling 
            // into first layer weights.
            for (int j=0; j<4; ++j)
                for (int k=0; k<48; ++k)
                    d->weights0[j*48+k] = (bdata[j*48+k]-mean[j])/ (127.5 * (1 << (d->vi.format->bitsPerSample - 8)));
            memcpy(d->weights0+4*48,bdata+4*48,(dims0-4*48)*sizeof(float));
            if (d->opt > 1) // shuffle weight order for asm
            {
                float *wf = d->weights0;
                float *rf = (float*)malloc(dims0*sizeof(float));
                memcpy(rf,d->weights0,dims0*sizeof(float));
                for (int j=0; j<4; ++j)
                    for (int k=0; k<48; ++k)
                        wf[(k>>2)*16+j*4+(k&3)] = rf[j*48+k];
                shufflePreScrnL2L3(wf+4*49,rf+4*49,d->opt);
                free(rf);
            }
        }
    }

    // Adjust prediction weights
    for (int i=0; i<2; ++i)
    {
        const float *bdataT = bdata+dims0+dims0new*3+dims1tsize*d->etype+dims1offset+i*dims1;
        const int nnst = nnsTable[d->nnsparam];
        const int asize = xdiaTable[d->nsize]*ydiaTable[d->nsize];
        const int boff = nnst*2*asize;
        double *mean = (double*)calloc(asize+1+nnst*2,sizeof(double));
        // Calculate mean weight of each neuron (ignore bias)
        for (int j=0; j<nnst*2; ++j)
        {
            double cmean = 0.0;
            for (int k=0; k<asize; ++k)
                cmean += bdataT[j*asize+k];
            mean[asize+1+j] = cmean/(double)asize;
        }
        // Calculate mean softmax neuron
        for (int j=0; j<nnst; ++j)
        {
            for (int k=0; k<asize; ++k)
                mean[k] += bdataT[j*asize+k]-mean[asize+1+j];
            mean[asize] += bdataT[boff+j];
        }
        for (int j=0; j<asize+1; ++j)
            mean[j] /= (double)(nnst);
        if (d->fapprox&2) // use int16 dot products
        {
            int16_t *ws = (int16_t*)d->weights1[i];
            float *wf = (float*)&ws[nnst*2*asize];
            // Factor mean removal into weights, remove global offset from
            // softmax neurons, and scale weights to int16 range.
            for (int j=0; j<nnst; ++j) // softmax neurons
            {
                double mval = 0.0;
                for (int k=0; k<asize; ++k)
                    mval = max(mval,fabs(bdataT[j*asize+k]-mean[asize+1+j]-mean[k]));
                const double scale = 32767.0/mval;
                for (int k=0; k<asize; ++k)
                    ws[j*asize+k] = roundds((bdataT[j*asize+k]-mean[asize+1+j]-mean[k])*scale);
                wf[(j>>2)*8+(j&3)] = (float)(mval/32767.0);
                wf[(j>>2)*8+(j&3)+4] = bdataT[boff+j]-mean[asize];
            }
            for (int j=nnst; j<nnst*2; ++j) // elliott neurons
            {
                double mval = 0.0;
                for (int k=0; k<asize; ++k)
                    mval = max(mval,fabs(bdataT[j*asize+k]-mean[asize+1+j]));
                const double scale = 32767.0/mval;
                for (int k=0; k<asize; ++k)
                    ws[j*asize+k] = roundds((bdataT[j*asize+k]-mean[asize+1+j])*scale);
                wf[(j>>2)*8+(j&3)] = (float)(mval/32767.0);
                wf[(j>>2)*8+(j&3)+4] = bdataT[boff+j];
            }
            if (d->opt > 1) // shuffle weight order for asm
            {
                int16_t *rs = (int16_t*)malloc(nnst*2*asize*sizeof(int16_t));
                memcpy(rs,ws,nnst*2*asize*sizeof(int16_t));
                for (int j=0; j<nnst*2; ++j)
                    for (int k=0; k<asize; ++k)
                        ws[(j>>2)*asize*4+(k>>3)*32+(j&3)*8+(k&7)] = rs[j*asize+k];
                free(rs);
            }
        }
        else // use float dot products
        {
            // Factor mean removal into weights, and remove global
            // offset from softmax neurons.
            for (int j=0; j<nnst*2; ++j)
            {
                for (int k=0; k<asize; ++k)
                {
                    const double q = j < nnst ? mean[k] : 0.0;
                    if (d->opt > 1) // shuffle weight order for asm
                        d->weights1[i][(j>>2)*asize*4+(k>>2)*16+(j&3)*4+(k&3)] = 
                            bdataT[j*asize+k]-mean[asize+1+j]-q;
                    else
                        d->weights1[i][j*asize+k] = bdataT[j*asize+k]-mean[asize+1+j]-q;
                }
                d->weights1[i][boff+j] = bdataT[boff+j]-(j<nnst?mean[asize]:0.0);
            }
        }
        free(mean);
    }

    d->nns = nnsTable[d->nnsparam];
    d->xdia = xdiaTable[d->nsize];
    d->ydia = ydiaTable[d->nsize];
    d->asize = xdiaTable[d->nsize] * ydiaTable[d->nsize];

    if (d->vi.format->bitsPerSample == 8)
        selectFunctions(d);
    else
        selectFunctions_uint16(d);
}


int modnpf(const int m, const int n)
{
    if ((m%n) == 0)
        return m;
    return m+n-(m%n);
}


static const VSFrameRef *VS_CC nnedi3GetFrame(int n, int activationReason, void **instanceData, void **frameData, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) {
    nnedi3Data *d = (nnedi3Data *) * instanceData;

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(d->field > 1 ? n / 2 : n, d->node, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        int field_n;
        if (d->field > 1) {
            if (n & 1) {
                field_n = d->field == 3 ? 0 : 1;
            } else {
                field_n = d->field == 3 ? 1 : 0;
            }
        } else {
            field_n = d->field;
        }

        const VSFrameRef *src = vsapi->getFrameFilter(d->field > 1 ? n / 2 : n, d->node, frameCtx);
        VSFrameRef *dst = vsapi->newVideoFrame(d->vi.format, d->vi.width, d->vi.height, src, core);


        FrameData *frameData = malloc(sizeof(FrameData));

        for (int plane = 0; plane < d->vi.format->numPlanes; plane++) {
            const int min_pad = 10;
            const int min_alignment = 16;

            int dst_width = vsapi->getFrameWidth(dst, plane);
            int dst_height = vsapi->getFrameHeight(dst, plane);

            frameData->padded_width[plane]  = dst_width + 64;
            frameData->padded_height[plane] = dst_height + 12;
            frameData->padded_stride[plane] = modnpf(frameData->padded_width[plane] * d->vi.format->bytesPerSample + min_pad, min_alignment); // TODO: maybe min_pad is in pixels too?
            VS_ALIGNED_MALLOC(&frameData->paddedp[plane], frameData->padded_stride[plane] * frameData->padded_height[plane], min_alignment);

            frameData->dstp[plane] = vsapi->getWritePtr(dst, plane);
            frameData->dst_stride[plane] = vsapi->getStride(dst, plane);

            VS_ALIGNED_MALLOC(&frameData->lcount[plane], dst_height * sizeof(int32_t), 16);
            memset(frameData->lcount[plane], 0, dst_height * sizeof(int32_t));

            frameData->field[plane] = field_n;
        }

        VS_ALIGNED_MALLOC(&frameData->input, 512 * sizeof(float), 16);
        VS_ALIGNED_MALLOC(&frameData->temp, 2048 * sizeof(float), 16);

        // Copy src to a padded "frame" in frameData and mirror the edges.
        d->copyPad(src, frameData, instanceData, field_n, vsapi);


        // Handles prescreening and the cubic interpolation.
        d->evalFunc_0(instanceData, frameData);

        // The rest.
        d->evalFunc_1(instanceData, frameData);


        // Clean up.
        for (int plane = 0; plane < d->vi.format->numPlanes; plane++) {
            VS_ALIGNED_FREE(frameData->paddedp[plane]);
            VS_ALIGNED_FREE(frameData->lcount[plane]);
        }
        VS_ALIGNED_FREE(frameData->input);
        VS_ALIGNED_FREE(frameData->temp);

        free(frameData);

        vsapi->freeFrame(src);

        // And then return dst.
        return dst;
    }

    return 0;
}


static void VS_CC nnedi3Free(void *instanceData, VSCore *core, const VSAPI *vsapi) {
    nnedi3Data *d = (nnedi3Data *)instanceData;
    vsapi->freeNode(d->node);

    VS_ALIGNED_FREE(d->weights0);

    for (int i = 0; i < 2; i++) {
        VS_ALIGNED_FREE(d->weights1[i]);
    }

    free(d);
}


static void VS_CC nnedi3Create(const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi) {
    nnedi3Data d;
    nnedi3Data *data;
    int err;

    // Get a clip reference from the input arguments. This must be freed later.
    d.node = vsapi->propGetNode(in, "clip", 0, 0);
    d.vi = *vsapi->getVideoInfo(d.node);

    if (!d.vi.format || d.vi.format->sampleType != stInteger
            || d.vi.format->bitsPerSample > 16) {
        vsapi->setError(out, "nnedi3: only constant format 8..16 bit integer input supported");
        vsapi->freeNode(d.node);
        return;
    }

    // Get the parameters.
    d.field = vsapi->propGetInt(in, "field", 0, 0);

    // Defaults to 0.
    d.dh = vsapi->propGetInt(in, "dh", 0, &err);

    d.Y = vsapi->propGetInt(in, "Y", 0, &err);
    if (err) {
        d.Y = 1;
    }
    d.U = vsapi->propGetInt(in, "U", 0, &err);
    if (err) {
        d.U = 1;
    }
    d.V = vsapi->propGetInt(in, "V", 0, &err);
    if (err) {
        d.V = 1;
    }

    d.nsize = vsapi->propGetInt(in, "nsize", 0, &err);
    if (err) {
        d.nsize = 6;
    }

    d.nnsparam = vsapi->propGetInt(in, "nns", 0, &err);
    if (err) {
        d.nnsparam = 1;
    }

    d.qual = vsapi->propGetInt(in, "qual", 0, &err);
    if (err) {
        d.qual = 1;
    }

    d.etype = vsapi->propGetInt(in, "etype", 0, &err);

    d.pscrn = vsapi->propGetInt(in, "pscrn", 0, &err);
    if (err) {
        if (d.vi.format->bitsPerSample == 8)
            d.pscrn = 2;
        else
            d.pscrn = 1;
    }

    d.opt = vsapi->propGetInt(in, "opt", 0, &err);
    if (err) {
        d.opt = 2;
    }

    d.fapprox = vsapi->propGetInt(in, "fapprox", 0, &err);
    if (err) {
        if (d.vi.format->bitsPerSample == 8)
            d.fapprox = 15;
        else
            d.fapprox = 12;
    }

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

    d.Y = !!d.Y;
    d.U = !!d.U;
    d.V = !!d.V;

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

    if (d.vi.format->bitsPerSample == 8) {
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

    if (d.opt < 1 || d.opt > 2) {
        vsapi->setError(out, "nnedi3: opt must be 1 or 2");
        vsapi->freeNode(d.node);
        return;
    }

    if (d.vi.format->bitsPerSample == 8) {
        if (d.fapprox < 0 || d.fapprox > 15) {
            vsapi->setError(out, "nnedi3: fapprox must be between 0 and 15 (inclusive)");
            vsapi->freeNode(d.node);
            return;
        }
    } else {
        if (d.fapprox != 0 && d.fapprox != 4 && d.fapprox != 8 && d.fapprox != 12) {
            vsapi->setError(out, "nnedi3: fapprox must be 4, 8, or 12");
            vsapi->freeNode(d.node);
            return;
        }
    }

    // Changing the video info probably has to be done before createFilter.
    if (d.field > 1) {
        d.vi.numFrames *= 2;
        d.vi.fpsNum *= 2;
    }

    if (d.dh) {
        d.vi.height *= 2;
    }

    d.max_value = 65535 >> (16 - d.vi.format->bitsPerSample);


    data = malloc(sizeof(d));
    *data = d;

    vsapi->createFilter(in, out, "nnedi3", nnedi3Init, nnedi3GetFrame, nnedi3Free, fmParallel, 0, data, core);
    return;
}


static void VS_CC nnedi3_rpow2Create(const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi) {
    int rfactor, width, height, correct_shift;
    int err, tmp, times, i;
    VSNodeRef *node;
    const VSVideoInfo *vi;
    VSMap *nnedi3Args, *ret, *args;
    VSPlugin *stdPlugin, *nnedi3Plugin, *fmtcPlugin;
    const char *error;
    double hshift = 0;
    double vshift = -0.5;
    const char kernel[] = "spline36";
    const char *keysToCopy[] = { "nsize", "nns", "qual", "etype", "pscrn", "opt", "fapprox", NULL };

    rfactor = vsapi->propGetInt(in, "rfactor", 0, NULL);
    if (rfactor < 2 || rfactor > 1024) {
        vsapi->setError(out, "nnedi3_rpow2: rfactor must be between 2 and 1024");
        return;
    }

    tmp = 1;
    times = 0;
    while (tmp < rfactor) {
        tmp *= 2;
        times++;
    }
    if (tmp != rfactor) {
        vsapi->setError(out, "nnedi3_rpow2: rfactor must be a power of 2");
        return;
    }

    node = vsapi->propGetNode(in, "clip", 0, NULL);
    vi = vsapi->getVideoInfo(node);

    if (vi->format->subSamplingW > 1 || vi->format->subSamplingH > 1) {
        vsapi->setError(out, "nnedi3_rpow2: I don't know what to do with so much subsampling. Send patches.");
        vsapi->freeNode(node);
        return;
    }

    width = vsapi->propGetInt(in, "width", 0, &err);

    height = vsapi->propGetInt(in, "height", 0, &err);

    correct_shift = !!vsapi->propGetInt(in, "correct_shift", 0, &err);
    if (err) {
        correct_shift = 1;
    }


    stdPlugin = vsapi->getPluginByNs("std", core);
    if (!stdPlugin) {
        vsapi->setError(out, "nnedi3_rpow2: std plugin isn't loaded... wtf?");
        vsapi->freeNode(node);
        return;
    }
    nnedi3Plugin = vsapi->getPluginByNs("nnedi3", core);
    if (!nnedi3Plugin) {
        vsapi->setError(out, "nnedi3_rpow2: nnedi3 plugin isn't loaded... wtf?");
        vsapi->freeNode(node);
        return;
    }
    fmtcPlugin = vsapi->getPluginByNs("fmtc", core);
    if (!fmtcPlugin && (correct_shift || vi->format->subSamplingH)) {
        vsapi->setError(out, "nnedi3_rpow2: the fmtconv plugin is required");
        vsapi->freeNode(node);
        return;
    }


    nnedi3Args = vsapi->createMap();

    i = 0;
    while (keysToCopy[i]) {
        int64_t value = vsapi->propGetInt(in, keysToCopy[i], 0, &err);
        if (!err) {
            vsapi->propSetInt(nnedi3Args, keysToCopy[i], value, paReplace);
        }
        i++;
    }

    vsapi->propGetInt(in, "nsize", 0, &err);
    if (err) {
        vsapi->propSetInt(nnedi3Args, "nsize", 0, paReplace);
    }

    vsapi->propGetInt(in, "nns", 0, &err);
    if (err) {
        vsapi->propSetInt(nnedi3Args, "nns", 3, paReplace);
    }

    vsapi->propSetInt(nnedi3Args, "dh", 1, paReplace);
    vsapi->propSetInt(nnedi3Args, "Y", 1, paReplace);
    vsapi->propSetInt(nnedi3Args, "U", 1, paReplace);
    vsapi->propSetInt(nnedi3Args, "V", 1, paReplace);

    args = vsapi->createMap();
    for (i = 0; i < times; i++) {
        int field;

        vsapi->propSetNode(nnedi3Args, "clip", node, paReplace);
        vsapi->freeNode(node);
        vsapi->propSetInt(nnedi3Args, "field", i == 0 ? 1 : 0 /*hurr durr*/, paReplace);
        ret = vsapi->invoke(nnedi3Plugin, "nnedi3", nnedi3Args);
        error = vsapi->getError(ret);
        if (error) {
            vsapi->setError(out, error);
            vsapi->freeMap(ret);
            return;
        }

        node = vsapi->propGetNode(ret, "clip", 0, NULL);
        vsapi->freeMap(ret);
        vsapi->propSetNode(args, "clip", node, paReplace);
        vsapi->freeNode(node);
        ret = vsapi->invoke(stdPlugin, "Transpose", args);
        error = vsapi->getError(ret);
        if (error) {
            vsapi->setError(out, error);
            vsapi->freeMap(ret);
            return;
        }

        if (vi->format->subSamplingW) {
            // Apparently always using field=1 for the horizontal pass somehow
            // keeps luma/chroma alignment.
            field = 1;
            hshift = hshift*2 - 0.5;
        } else {
            field = i == 0 ? 1 : 0;
            hshift = -0.5;
        }

        node = vsapi->propGetNode(ret, "clip", 0, NULL);
        vsapi->freeMap(ret);
        vsapi->propSetNode(nnedi3Args, "clip", node, paReplace);
        vsapi->freeNode(node);
        vsapi->propSetInt(nnedi3Args, "field", field, paReplace);
        ret = vsapi->invoke(nnedi3Plugin, "nnedi3", nnedi3Args);
        error = vsapi->getError(ret);
        if (error) {
            vsapi->setError(out, error);
            vsapi->freeMap(ret);
            return;
        }

        node = vsapi->propGetNode(ret, "clip", 0, NULL);
        vsapi->freeMap(ret);
        vsapi->propSetNode(args, "clip", node, paReplace);
        vsapi->freeNode(node);
        ret = vsapi->invoke(stdPlugin, "Transpose", args);
        error = vsapi->getError(ret);
        if (error) {
            vsapi->setError(out, error);
            vsapi->freeMap(ret);
            return;
        }

        node = vsapi->propGetNode(ret, "clip", 0, NULL);
        vsapi->freeMap(ret);
    }
    vsapi->freeMap(nnedi3Args);

    // Correct vertical shift of the chroma.
    if (vi->format->subSamplingH) {
        vsapi->clearMap(args);
        vsapi->propSetNode(args, "clip", node, paReplace);
        vsapi->freeNode(node);
        vsapi->propSetData(args, "kernel", kernel, strlen(kernel), paReplace);
        vsapi->propSetFloat(args, "sy", -0.5, paReplace);
        vsapi->propSetFloat(args, "planes", 2, paReplace); // copy the luma
        vsapi->propSetFloat(args, "planes", 3, paAppend); // process the chroma
        vsapi->propSetFloat(args, "planes", 3, paAppend); // process the chroma

        ret = vsapi->invoke(fmtcPlugin, "resample", args);
        error = vsapi->getError(ret);
        if (error) {
            vsapi->setError(out, error);
            vsapi->freeMap(ret);
            return;
        }

        node = vsapi->propGetNode(ret, "clip", 0, NULL);
        vsapi->freeMap(ret);
    }

    if (correct_shift) {
        vsapi->clearMap(args);

        vsapi->propSetNode(args, "clip", node, paReplace);
        vsapi->freeNode(node);
        if (width) {
            vsapi->propSetInt(args, "w", width, paReplace);
        }
        if (height) {
            vsapi->propSetInt(args, "h", height, paReplace);
        }
        vsapi->propSetData(args, "kernel", kernel, strlen(kernel), paReplace);
        vsapi->propSetFloat(args, "sx", hshift, paReplace);
        vsapi->propSetFloat(args, "sy", vshift, paReplace);

        ret = vsapi->invoke(fmtcPlugin, "resample", args);
        error = vsapi->getError(ret);
        if (error) {
            vsapi->setError(out, error);
            vsapi->freeMap(ret);
            return;
        }

        node = vsapi->propGetNode(ret, "clip", 0, NULL);
        vsapi->freeMap(ret);
    }

    vsapi->clearMap(args);
    vsapi->propSetNode(args, "clip", node, paReplace);
    vsapi->freeNode(node);
    vsapi->propSetInt(args, "csp", vi->format->id, paReplace);

    ret = vsapi->invoke(fmtcPlugin, "bitdepth", args);
    error = vsapi->getError(ret);
    if (error) {
        vsapi->setError(out, error);
        vsapi->freeMap(ret);
        return;
    }

    node = vsapi->propGetNode(ret, "clip", 0, NULL);
    vsapi->freeMap(ret);
    vsapi->freeMap(args);

    vsapi->propSetNode(out, "clip", node, paReplace);
    vsapi->freeNode(node);
}


VS_EXTERNAL_API(void) VapourSynthPluginInit(VSConfigPlugin configFunc, VSRegisterFunction registerFunc, VSPlugin *plugin) {
    configFunc("com.deinterlace.nnedi3", "nnedi3", "Neural network edge directed interpolation", VAPOURSYNTH_API_VERSION, 1, plugin);
    registerFunc("nnedi3", "clip:clip;"
                           "field:int;"
                           "dh:int:opt;"
                           "Y:int:opt;"
                           "U:int:opt;"
                           "V:int:opt;"
                           "nsize:int:opt;"
                           "nns:int:opt;"
                           "qual:int:opt;"
                           "etype:int:opt;"
                           "pscrn:int:opt;"
                           "opt:int:opt;"
                           "fapprox:int:opt;", nnedi3Create, 0, plugin);
    registerFunc("nnedi3_rpow2", "clip:clip;"
                                 "rfactor:int;"
                                 "width:int:opt;"
                                 "height:int:opt;"
                                 "correct_shift:int:opt;"
                                 "nsize:int:opt;"
                                 "nns:int:opt;"
                                 "qual:int:opt;"
                                 "etype:int:opt;"
                                 "pscrn:int:opt;"
                                 "opt:int:opt;"
                                 "fapprox:int:opt;", nnedi3_rpow2Create, 0, plugin);
}

