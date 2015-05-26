%include "include/x86inc.asm"

SECTION_RODATA
sign_bits_f times 4 dd 0x7FFFFFFF
sign_bits_f_zero_l  dq 0x7FFFFFFF00000000, 0x7FFFFFFF7FFFFFFF
ones_f      times 4 dd 1.0

ub_1     times 16 db 1
w_19     times 8  dw 19
w_3      times 8  dw 3
w_254    times 8  dw 254
uw_16    times 8  dw 16

; If FLT_EPSILON is 1.0e-7, how does this work?
min_weight_sum times 4 dd 1.0e-10
five_f         times 4 dd 5.0

; This seems to be the value of FLT_EPSILON, according to clang.
flt_epsilon_sse times 4 dd 1.0e-7

exp_lo   times 4 dd -80.0
exp_hi   times 4 dd  80.0
e0_mult  times 4 dd 12102203.161561486 ; (1.0/ln(2))*(2^23)
e0_bias  times 4 dd 1064866805.0 ; (2^23)*127.0-486411.0

e1_scale times 4 dd 1.4426950409 ; 1/ln(2)
e1_bias  times 4 dd 12582912.0 ; 3 << 22
e1_c0    times 4 dd 1.00035
e1_c1    times 4 dd 0.701277797
e1_c2    times 4 dd 0.237348593

am_0p5      times 4 dd 0.5
am_1        times 4 dd 1.0 ; this is the same as ones_f... why duplicate?
exp_rln2    times 4 dd 1.442695041 ; e1_scale...
exp_p0      times 4 dd 1.261771931e-4
exp_p1      times 4 dd 3.029944077e-2
exp_q0      times 4 dd 3.001985051e-6
exp_q1      times 4 dd 2.524483403e-3
exp_q2      times 4 dd 2.272655482e-1
exp_q3      times 4 dd 2.000000000 ; seriously...
exp_c1      times 4 dd 6.931457520e-1
exp_c2      times 4 dd 1.428606820e-6
epi32_1     times 4 dd 1
epi32_0x7f  times 4 dd 0x7F

SECTION_TEXT

; parameters:
; const uint8_t *t, const int pitch, float *pf
; r0 - t
; r1 - pitch
; r2 - pf
INIT_XMM
cglobal word2float48_SSE2, 3, 4, 4, srcp, src_pitch, dstp
    pxor m0, m0

    shl src_pitchq, 2 ; stride is halved

    mov r3, 4
.loop:
    movq m1, [srcpq]
    movq m2, [srcpq + 8]
    movq m3, [srcpq + 16]

    punpcklwd m1, m0
    punpcklwd m2, m0
    punpcklwd m3, m0

    cvtdq2ps m1, m1
    cvtdq2ps m2, m2
    cvtdq2ps m3, m3

    movaps [dstpq], m1
    movaps [dstpq + 16], m2
    movaps [dstpq + 32], m3

    add dstpq, 48
    add srcpq, src_pitchq
    sub r3, 1
    jnz .loop

    RET


; parameters:
; const uint8_t *t, const int pitch, float *pf
; r0 - t
; r1 - pitch
; r2 - pf
INIT_XMM
cglobal byte2word48_SSE2, 3, 3, 8, srcp, src_pitch, dstp
   movq m0, [srcpq]
   movd m1, [srcpq + 8]
   movd m2, [srcpq + src_pitchq * 2]
   movq m3, [srcpq + src_pitchq * 2 + 4]

   lea srcpq, [srcpq + src_pitchq * 4]

   movq m4, [srcpq]
   movd m5, [srcpq + 8]
   movd m6, [srcpq + src_pitchq * 2]
   movq m7, [srcpq + src_pitchq * 2 + 4]

   punpckldq m1, m2
   pxor m2, m2
   punpckldq m5, m6

   punpcklbw m0, m2
   punpcklbw m3, m2
   punpcklbw m1, m2
   punpcklbw m4, m2
   punpcklbw m5, m2
   punpcklbw m7, m2

   mova [dstpq], m0
   mova [dstpq + 16], m1
   mova [dstpq + 32], m3
   mova [dstpq + 48], m4
   mova [dstpq + 64], m5
   mova [dstpq + 80], m7

   RET

INIT_XMM
cglobal byte2word64_SSE2, 3, 3, 8, srcp, src_stride, dstp
   pxor m7, m7
   
   movq m0, [srcpq]
   movq m1, [srcpq + 8]
   movq m2, [srcpq + src_strideq*2]
   movq m3, [srcpq + src_strideq*2 + 8]
   
   punpcklbw m0, m7
   punpcklbw m1, m7
   punpcklbw m2, m7
   punpcklbw m3, m7
   
   mova [dstpq], m0
   mova [dstpq + 16], m1
   mova [dstpq + 32], m2
   mova [dstpq + 48], m3
   
   lea srcpq, [srcpq + src_strideq*4]
   
   movq m4, [srcpq]
   movq m5, [srcpq + 8]
   movq m6, [srcpq + src_strideq*2]
   movq m0, [srcpq + src_strideq*2 + 8]
   
   punpcklbw m4, m7
   punpcklbw m5, m7
   punpcklbw m6, m7
   punpcklbw m0, m7
   
   mova [dstpq + 64], m4
   mova [dstpq + 80], m5
   mova [dstpq + 96], m6
   mova [dstpq + 112], m0

   RET

; parameters:
; const float *datai, const float *weights, uint8_t *d
INIT_XMM
cglobal computeNetwork0new_SSE2, 3, 4, 8, datai, weights, d
   pxor m0, m0
   pxor m1, m1
   pxor m2, m2
   pxor m3, m3

   xor r3, r3
.loop:
   mova m4, [dataiq + r3]
   mova m5, m4
   mova m6, m4
   mova m7, m4

   pmaddwd m4, [weightsq + r3 * 4]
   pmaddwd m5, [weightsq + r3 * 4 + 16]
   pmaddwd m6, [weightsq + r3 * 4 + 32]
   pmaddwd m7, [weightsq + r3 * 4 + 48]

   paddd m0, m4
   paddd m1, m5
   paddd m2, m6
   paddd m3, m7

   add r3, 16
   cmp r3, 128
   jl .loop

   mova m4,m0
   mova m5,m2
   
   punpcklqdq m0,m1 ; m0 = m1[1] m1[0] m0[1] m0[0]
   punpcklqdq m2,m3 ; m2 = m3[1] m3[0] m2[1] m2[0]
   
   punpckhqdq m4,m1 ; m4 = m1[3] m1[2] m0[3] m0[2]
   punpckhqdq m5,m3 ; m5 = m3[3] m3[2] m2[3] m2[2]
   
   paddd m0,m4 ; m0 = m1[1]+m1[3] m1[0]+m1[2] m0[1]+m0[3] m0[0]+m0[2]
   paddd m2,m5 ; m2 = m3[1]+m3[3] m3[0]+m3[2] m2[1]+m2[3] m2[0]+m2[2]
   
   mova m6,m0
   
   shufps m0,m2,136 ; m0 = m3[0]+m3[2] m2[0]+m2[2] m1[0]+m1[2] m0[0]+m0[2]
   shufps m6,m2,221 ; m6 = m3[1]+m3[3] m2[1]+m2[3] m1[1]+m1[3] m0[1]+m0[3]
   
   paddd m0,m6 ; m0 = sum(m3) sum(m2) sum(m1) sum(m0)
   cvtdq2ps m0,m0
   mulps m0,[weightsq+512]
   addps m0,[weightsq+528]
   movaps m1,m0
   andps m0,[sign_bits_f]
   addps m0,[ones_f]
   rcpps m0,m0
   mulps m0,m1
   
   pshufd m1,m0,0
   pshufd m2,m0,85
   pshufd m3,m0,170
   pshufd m4,m0,255

   mulps m1,[weightsq+544]
   mulps m2,[weightsq+560]
   mulps m3,[weightsq+576]
   mulps m4,[weightsq+592]
   
   pxor m0,m0
   
   addps m1,m2
   addps m3,m4
   addps m1,m3
   
   addps m1,[weightsq+608]
   ; yasm wouldn't take this cmpps
   ;cmpps m1,m0,1
   cmpltps m1,m0
   packssdw m1,m0
   packsswb m1,m0
   movd r1d,m1
   xor r1d,0xFFFFFFFF
   and r1d,0x01010101
   mov [dq],r1d
   RET

; parameters:
; const uint8_t *tempu, int width, uint8_t *dstp, const uint8_t *src3p, const int src_pitch
; r0 - tempu
; r1 - width
; r2 - dstp
; r3 - src3p
; r4 - src_pitch
INIT_XMM
cglobal processLine0_SSE2, 5, 6, 8, tempu, width, dstp, src3p, src_pitch
%if WIN64
   ; The parameter is 32 bit. Make sure the high 32 bits are cleared.
   shl src_pitchq, 32
   shr src_pitchq, 32
%endif
   lea r5,[src3pq+src_pitchq*4]
   pxor m6,m6
   pxor m7,m7
.xloop:
   mova m0,[src3pq+src_pitchq*2]
   mova m1,[r5]
   mova m2,m0
   mova m3,m1
   punpcklbw m0,m7
   punpckhbw m2,m7
   punpcklbw m1,m7
   punpckhbw m3,m7
   paddw m0,m1
   paddw m2,m3
   pmullw m0,[w_19]
   pmullw m2,[w_19]
   mova m1,[src3pq]
   mova m3,[r5+src_pitchq*2]
   mova m4,m1
   mova m5,m3
   punpcklbw m1,m7
   punpckhbw m4,m7
   punpcklbw m3,m7
   punpckhbw m5,m7
   paddw m1,m3
   paddw m4,m5
   pmullw m1,[w_3]
   pmullw m4,[w_3]
   mova m3,[tempuq]
   psubusw m0,m1
   psubusw m2,m4
   pcmpeqb m3,[ub_1]
   paddusw m0,[uw_16]
   paddusw m2,[uw_16]
   mova m1,m3
   pcmpeqb m4,m4
   psrlw m0,5
   psrlw m2,5
   pxor m1,m4
   pminsw m0,[w_254]
   pminsw m2,[w_254]
   mova m5,m1
   packuswb m0,m2
   pand m5,[ub_1]
   pand m0,m3
   psadbw m5,m7
   por m0,m1
   mova m2,m5
   psrldq m5,8
   mova [dstpq],m0
   paddusw m5,m2
   paddusw m6,m5
   add src3pq,16
   add r5,16
   add tempuq,16
   add dstpq,16
   sub widthq,16
   jnz .xloop
   movd eax,m6
   RET


; parameters:
; const float *w, const int n, float *mstd
INIT_XMM
cglobal weightedAvgElliottMul5_m16_SSE2, 3, 5, 8, w, n, mstd
   lea r3,[wq+nq*4]
   xor r4,r4
   xorps m0,m0 ; sum w
   xorps m1,m1 ; sum w*v
.nloop:
   movaps m2,[wq+r4*4]
   movaps m3,[wq+r4*4+16]
   movaps m4,[r3+r4*4]
   movaps m5,[r3+r4*4+16]
   addps m0,m2
   movaps m6,m4
   movaps m7,m5
   addps m0,m3
   andps m4,[sign_bits_f]
   andps m5,[sign_bits_f]
   addps m4,[ones_f]
   addps m5,[ones_f]
   rcpps m4,m4
   rcpps m5,m5
   mulps m6,m4
   mulps m7,m5
   mulps m6,m2
   mulps m7,m3
   addps m1,m6
   addps m1,m7
   movaps m2,[wq+r4*4+32]
   movaps m3,[wq+r4*4+48]
   movaps m4,[r3+r4*4+32]
   movaps m5,[r3+r4*4+48]
   addps m0,m2
   movaps m6,m4
   movaps m7,m5
   addps m0,m3
   andps m4,[sign_bits_f]
   andps m5,[sign_bits_f]
   addps m4,[ones_f]
   addps m5,[ones_f]
   rcpps m4,m4
   rcpps m5,m5
   mulps m6,m4
   mulps m7,m5
   mulps m6,m2
   mulps m7,m3
   addps m1,m6
   addps m1,m7
   add r4,16
   sub nq,16
   jnz .nloop
   movhlps m2,m0
   movhlps m3,m1
   addps m0,m2
   addps m1,m3
   pshuflw m2,m0,14
   pshuflw m3,m1,14
   addss m0,m2
   addss m1,m3
   comiss m0,[min_weight_sum]
   jbe .nodiv
   mulss m1,[five_f]
   rcpss m0,m0
   mulss m1,m0
   jmp .finish
.nodiv:
   xorps m1,m1
.finish:
   mulss m1,[mstdq+4]
   addss m1,[mstdq]
   addss m1,[mstdq+12]
   movss [mstdq+12],m1
   RET


; parameters:
; const uint8_t *srcp, const int stride, const int xdia, const int ydia, float *mstd, float *inputf
INIT_XMM
cglobal extract_m8_i16_SSE2, 6, 7, 8, srcp, stride, xdia, ydia, mstd, inputf
   lea r6,[srcpq+strideq*2]
   pxor m4,m4 ; sum
   pxor m5,m5 ; sumsq
   pxor m6,m6
   PUSH ydiaq
   PUSH mstdq ; r4
.yloop:
   xor r4,r4
.xloop:
   movq m0,[srcpq+r4]
   movq m1,[r6+r4]
   mova m2,m0
   mova m3,m1
   punpcklbw m0,m6
   punpcklbw m1,m6
   psadbw m2,m6
   psadbw m3,m6
   mova [inputfq],m0
   mova [inputfq+xdiaq*2],m1
   pmaddwd m0,m0
   pmaddwd m1,m1
   paddd m4,m2
   paddd m5,m0
   paddd m4,m3
   paddd m5,m1
   add r4,8
   add inputfq,16
   cmp r4,xdiaq
   jl .xloop

   lea srcpq,[srcpq+strideq*4]
   lea r6,[r6+strideq*4]
   lea inputfq,[inputfq+xdiaq*2]
   sub ydiaq,2
   jnz .yloop
   POP mstdq ; r4
   POP ydiaq

   movhlps m1,m5
   paddd m5,m1
   movd m2,xdiad
   movd m3,ydiad
   pmuludq m2,m3
   cvtdq2ps m7,m2

   pshuflw m1,m5,14
   paddd m5,m1
   rcpss m7,m7 ; scale
   cvtdq2ps m4,m4
   cvtdq2ps m5,m5
   mulss m4,m7 ; mean
   mulss m5,m7
   movss [mstdq],m4
   mulss m4,m4
   subss m5,m4 ; var
   comiss m5,[flt_epsilon_sse]
   jbe .novarjmp
   rsqrtss m5,m5 ; 1.0/std
   rcpss m4,m5 ; std
   movss [mstdq+4],m4
   movss [mstdq+8],m5
   jmp .finish
.novarjmp:
   movss [mstdq+4],m6
   movss [mstdq+8],m6
.finish:
   movss [mstdq+12],m6
   RET




; Used by default with 16 bit input
;
; parameters:
;  float *s,
;  const int n
INIT_XMM
cglobal e0_m16_SSE2, 2, 2, 4
.eloop16:
   movaps m0,[r0]
   minps m0,[exp_hi]
   maxps m0,[exp_lo]
   mulps m0,[e0_mult]
   addps m0,[e0_bias]
   cvtps2dq m0,m0
   movaps [r0],m0

   add r0,16
   sub r1,4
   jnz .eloop16
   RET


; parameters:
;  float *s,
;  const int n
INIT_XMM
cglobal e0_m16_FMA3, 2, 2, 4
   movaps m1, [e0_mult]
.eloop16:
   movaps m0,[r0]
   minps m0,[exp_hi]
   maxps m0,[exp_lo]
   vfmadd213ps m0, m1, [e0_bias]
   cvtps2dq m0,m0
   movaps [r0],m0

   add r0,16
   sub r1,4
   jnz .eloop16
   RET


; parameters:
;  float *s,
;  const int n
INIT_XMM
cglobal e0_m16_FMA4, 2, 2, 4
   movaps m1, [e0_mult]
.eloop16:
   movaps m0,[r0]
   minps m0,[exp_hi]
   maxps m0,[exp_lo]
   vfmaddps m0, m0, m1, [e0_bias]
   cvtps2dq m0,m0
   movaps [r0],m0

   add r0,16
   sub r1,4
   jnz .eloop16
   RET


; Used by default with 16 bit input
;
; parameters:
;  const float *input,
;  const float *weights,
;  uint8_t *d
INIT_XMM
cglobal computeNetwork0_SSE2, 3, 5, 8, input, weights, d
   ;//    dotProd48_m4_SSE(input,weights,temp,4);
   mov r3,1

   xorps m0, m0
   xorps m1, m1
   xorps m2, m2
   xorps m3, m3

   xor r4, r4
.loop:
   movaps m4, [inputq + r4]
   movaps m5, m4
   movaps m6, m4
   movaps m7, m4

   mulps m4, [weightsq + r4 * 4]
   mulps m5, [weightsq + r4 * 4 + 16]
   mulps m6, [weightsq + r4 * 4 + 32]
   mulps m7, [weightsq + r4 * 4 + 48]

   addps m0, m4
   addps m1, m5
   addps m2, m6
   addps m3, m7

   add r4, 16
   cmp r4, 192
   jl .loop

   ; This block performs a horizontal sum of each accumulator (m0..m3) and packs the results in m0 (sum(m3) sum(m2) sum(m1) sum(m0)).
   ; Sadly replacing the twelve instructions with three haddps makes no difference whatsoever on this Core 2 Duo.
   movaps m4,m0
   movaps m5,m2
   unpcklpd m0,m1
   unpcklpd m2,m3
   unpckhpd m4,m1
   unpckhpd m5,m3
   addps m0,m4
   addps m2,m5
   movaps m6,m0
   shufps m0,m2,136 ; 10001000b
   shufps m6,m2,221 ; 11011101b
   addps m0,m6

   addps m0,[weightsq+768]
   ;// const float t = temp[0];
   ;// elliott4_SSE(temp);
   ;// temp[0] = t;
   movaps m1,m0
   andps m0,[sign_bits_f_zero_l]
   addps m0,[ones_f]
   rcpps m0,m0
   mulps m0,m1
   ;//    dotProd4_m4_SSE2(temp,weights+4*49,temp+4,4);
   pshufd m1,m0,0
   pshufd m2,m0,85
   pshufd m3,m0,170
   pshufd m4,m0,255
   mulps m1,[weightsq+784]
   mulps m2,[weightsq+784+16]
   mulps m3,[weightsq+784+32]
   mulps m4,[weightsq+784+48]
   addps m1,m2
   addps m3,m4
   addps m1,m3
   addps m1,[weightsq+784+64]
   ;// elliott4_SSE(temp+4);
   movaps m7,m1
   andps m1,[sign_bits_f]
   movaps m3,m0
   addps m1,[ones_f]
   rcpps m1,m1
   mulps m7,m1
   ;//    dotProd8_m4_SSE2(temp,weights+4*49+4*5,temp+32,4);
   pshufd m0,m0,0
   pshufd m1,m3,85
   pshufd m2,m3,170
   pshufd m3,m3,255
   mulps m0,[weightsq+864]
   mulps m1,[weightsq+864+16]
   mulps m2,[weightsq+864+32]
   mulps m3,[weightsq+864+48]
   pshufd m4,m7,0
   pshufd m5,m7,85
   pshufd m6,m7,170
   pshufd m7,m7,255
   mulps m4,[weightsq+864+64]
   mulps m5,[weightsq+864+80]
   mulps m6,[weightsq+864+96]
   mulps m7,[weightsq+864+112]
   addps m0,m1
   addps m2,m3
   addps m4,m5
   addps m6,m7
   addps m0,m2
   addps m4,m6
   addps m0,m4
   addps m0,[weightsq+864+128]
   movhlps m1,m0
   maxps m0,m1
   pshuflw m1,m0,14
   comiss m1,m0
   jbe .finish
   xor r3,r3
.finish:
   mov [dq],r3b
   RET


; parameters:
;  const float *input,
;  const float *weights,
;  uint8_t *d
INIT_XMM
cglobal computeNetwork0_FMA3, 3, 5, 8, input, weights, d
   ;//    dotProd48_m4_SSE(input,weights,temp,4);
   mov r3,1

   xorps m0, m0
   xorps m1, m1
   xorps m2, m2
   xorps m3, m3

   xor r4, r4
.loop:
   movaps m4, [inputq + r4]
   movaps m5, m4
   movaps m6, m4
   movaps m7, m4

   vfmadd231ps m0, m4, [weightsq + r4 * 4]
   vfmadd231ps m1, m5, [weightsq + r4 * 4 + 16]
   vfmadd231ps m2, m6, [weightsq + r4 * 4 + 32]
   vfmadd231ps m3, m7, [weightsq + r4 * 4 + 48]

   add r4, 16
   cmp r4, 192
   jl .loop

   haddps m0, m1
   haddps m2, m3
   haddps m0, m2

   addps m0,[weightsq+768]
   ;// const float t = temp[0];
   ;// elliott4_SSE(temp);
   ;// temp[0] = t;
   movaps m1,m0
   andps m0,[sign_bits_f_zero_l]
   addps m0,[ones_f]
   rcpps m0,m0
   mulps m0,m1
   ;//    dotProd4_m4_SSE2(temp,weights+4*49,temp+4,4);
   pshufd m1,m0,0
   pshufd m2,m0,85
   pshufd m3,m0,170
   pshufd m4,m0,255
   mulps m1,[weightsq+784]
   vfmadd231ps m1, m2, [weightsq+784+16]
   mulps m3,[weightsq+784+32]
   vfmadd231ps m3, m4, [weightsq+784+48]
   addps m1,m3
   addps m1,[weightsq+784+64]
   ;// elliott4_SSE(temp+4);
   movaps m7,m1
   andps m1,[sign_bits_f]
   movaps m3,m0
   addps m1,[ones_f]
   rcpps m1,m1
   mulps m7,m1
   ;//    dotProd8_m4_SSE2(temp,weights+4*49+4*5,temp+32,4);
   pshufd m0,m0,0
   pshufd m1,m3,85
   pshufd m2,m3,170
   pshufd m3,m3,255
   mulps m0,[weightsq+864]
   vfmadd231ps m0, m1, [weightsq+864+16]
   mulps m2,[weightsq+864+32]
   vfmadd231ps m2, m3, [weightsq+864+48]
   pshufd m4,m7,0
   pshufd m5,m7,85
   pshufd m6,m7,170
   pshufd m7,m7,255
   mulps m4,[weightsq+864+64]
   vfmadd231ps m4, m5, [weightsq+864+80]
   mulps m6,[weightsq+864+96]
   vfmadd231ps m6, m7, [weightsq+864+112]
   addps m0,m2
   addps m4,m6
   addps m0,m4
   addps m0,[weightsq+864+128]
   movhlps m1,m0
   maxps m0,m1
   pshuflw m1,m0,14
   comiss m1,m0
   jbe .finish
   xor r3,r3
.finish:
   mov [dq],r3b
   RET


; parameters:
;  const float *input,
;  const float *weights,
;  uint8_t *d
INIT_XMM
cglobal computeNetwork0_FMA4, 3, 5, 8, input, weights, d
   ;//    dotProd48_m4_SSE(input,weights,temp,4);
   mov r3,1

   xorps m0, m0
   xorps m1, m1
   xorps m2, m2
   xorps m3, m3

   xor r4, r4
.loop:
   movaps m4, [inputq + r4]
   movaps m5, m4
   movaps m6, m4
   movaps m7, m4

   vfmaddps m0, m4, [weightsq + r4 * 4], m0
   vfmaddps m1, m5, [weightsq + r4 * 4 + 16], m1
   vfmaddps m2, m6, [weightsq + r4 * 4 + 32], m2
   vfmaddps m3, m7, [weightsq + r4 * 4 + 48], m3

   add r4, 16
   cmp r4, 192
   jl .loop

   haddps m0, m1
   haddps m2, m3
   haddps m0, m2

   addps m0,[weightsq+768]
   ;// const float t = temp[0];
   ;// elliott4_SSE(temp);
   ;// temp[0] = t;
   movaps m1,m0
   andps m0,[sign_bits_f_zero_l]
   addps m0,[ones_f]
   rcpps m0,m0
   mulps m0,m1
   ;//    dotProd4_m4_SSE2(temp,weights+4*49,temp+4,4);
   pshufd m1,m0,0
   pshufd m2,m0,85
   pshufd m3,m0,170
   pshufd m4,m0,255
   mulps m1,[weightsq+784]
   vfmaddps m1, m2, [weightsq+784+16], m1
   mulps m3,[weightsq+784+32]
   vfmaddps m3, m4, [weightsq+784+48], m3
   addps m1,m3
   addps m1,[weightsq+784+64]
   ;// elliott4_SSE(temp+4);
   movaps m7,m1
   andps m1,[sign_bits_f]
   movaps m3,m0
   addps m1,[ones_f]
   rcpps m1,m1
   mulps m7,m1
   ;//    dotProd8_m4_SSE2(temp,weights+4*49+4*5,temp+32,4);
   pshufd m0,m0,0
   pshufd m1,m3,85
   pshufd m2,m3,170
   pshufd m3,m3,255
   mulps m0,[weightsq+864]
   vfmaddps m0, m1, [weightsq+864+16], m0
   mulps m2,[weightsq+864+32]
   vfmaddps m2, m3, [weightsq+864+48], m2
   pshufd m4,m7,0
   pshufd m5,m7,85
   pshufd m6,m7,170
   pshufd m7,m7,255
   mulps m4,[weightsq+864+64]
   vfmaddps m4, m5, [weightsq+864+80], m4
   mulps m6,[weightsq+864+96]
   vfmaddps m6, m7, [weightsq+864+112], m6
   addps m0,m2
   addps m4,m6
   addps m0,m4
   addps m0,[weightsq+864+128]
   movhlps m1,m0
   maxps m0,m1
   pshuflw m1,m0,14
   comiss m1,m0
   jbe .finish
   xor r3,r3
.finish:
   mov [dq],r3b
   RET


; parameters:
;  const uint8_t *t,
;  const int pitch,
;  float *p
INIT_XMM
cglobal byte2float48_SSE2, 3, 3, 7, srcp, src_pitch, dstp
   pxor m6,m6
   movq m0,[srcpq]
   movd m4,[srcpq+8]
   movq m2,[srcpq+src_pitchq*2]
   movd m5,[srcpq+src_pitchq*2+8]
   punpcklbw m0,m6
   punpcklbw m4,m6
   punpcklbw m2,m6
   punpcklbw m5,m6
   movdqa m1,m0
   punpcklbw m4,m6
   movdqa m3,m2
   punpcklbw m5,m6
   punpcklbw m0,m6
   punpckhbw m1,m6
   punpcklbw m2,m6
   punpckhbw m3,m6
   lea srcpq,[srcpq+src_pitchq*4]
   cvtdq2ps m4,m4
   cvtdq2ps m5,m5
   cvtdq2ps m0,m0
   cvtdq2ps m1,m1
   cvtdq2ps m2,m2
   cvtdq2ps m3,m3
   movaps [dstpq],m0
   movaps [dstpq+16],m1
   movaps [dstpq+32],m4
   movaps [dstpq+48],m2
   movaps [dstpq+64],m3
   movaps [dstpq+80],m5
   movq m0,[srcpq]
   movd m4,[srcpq+8]
   movq m2,[srcpq+src_pitchq*2]
   movd m5,[srcpq+src_pitchq*2+8]
   punpcklbw m0,m6
   punpcklbw m4,m6
   punpcklbw m2,m6
   punpcklbw m5,m6
   movdqa m1,m0
   punpcklbw m4,m6
   movdqa m3,m2
   punpcklbw m5,m6
   punpcklbw m0,m6
   punpckhbw m1,m6
   punpcklbw m2,m6
   punpckhbw m3,m6
   cvtdq2ps m4,m4
   cvtdq2ps m5,m5
   cvtdq2ps m0,m0
   cvtdq2ps m1,m1
   cvtdq2ps m2,m2
   cvtdq2ps m3,m3
   movaps [dstpq+96],m0
   movaps [dstpq+112],m1
   movaps [dstpq+128],m4
   movaps [dstpq+144],m2
   movaps [dstpq+160],m3
   movaps [dstpq+176],m5
   RET


; parameters:
;  const float *inputf,
;  const float *weightsf,
;  uint8_t *d
INIT_XMM
cglobal computeNetwork0_i16_SSE2, 3, 5, 8, input, weights, d
   mov r3,1

   pxor m0, m0
   pxor m1, m1
   pxor m2, m2
   pxor m3, m3

   xor r4, r4
.loop:
   movdqa m4, [inputq + r4]
   movdqa m5, m4
   movdqa m6, m4
   movdqa m7, m4
   pmaddwd m4, [weightsq + r4 * 4]
   pmaddwd m5, [weightsq + r4 * 4 + 16]
   pmaddwd m6, [weightsq + r4 * 4 + 32]
   pmaddwd m7, [weightsq + r4 * 4 + 48]
   paddd m0, m4
   paddd m1, m5
   paddd m2, m6
   paddd m3, m7

   add r4, 16
   cmp r4, 96
   jl .loop

   movdqa m4,m0
   movdqa m5,m2
   punpcklqdq m0,m1
   punpcklqdq m2,m3
   punpckhqdq m4,m1
   punpckhqdq m5,m3
   paddd m0,m4
   paddd m2,m5
   movdqa m6,m0
   shufps m0,m2,136
   shufps m6,m2,221
   paddd m0,m6
   cvtdq2ps m0,m0
   mulps m0,[weightsq+384]
   addps m0,[weightsq+400]
   ;// const float t = temp[0];
   ;// elliott4_SSE(temp);
   ;// temp[0] = t;
   movaps m1,m0
   andps m0,[sign_bits_f_zero_l]
   addps m0,[ones_f]
   rcpps m0,m0
   mulps m0,m1
   ;//    dotProd4_m4_SSE2(temp,weights+4*49,temp+4,4);
   pshufd m1,m0,0
   pshufd m2,m0,85
   pshufd m3,m0,170
   pshufd m4,m0,255
   mulps m1,[weightsq+416]
   mulps m2,[weightsq+416+16]
   mulps m3,[weightsq+416+32]
   mulps m4,[weightsq+416+48]
   addps m1,m2
   addps m3,m4
   addps m1,m3
   addps m1,[weightsq+416+64]
   ;// elliott4_SSE(temp+4);
   movaps m7,m1
   andps m1,[sign_bits_f]
   movaps m3,m0
   addps m1,[ones_f]
   rcpps m1,m1
   mulps m7,m1
   ;//    dotProd8_m4_SSE2(temp,weights+4*49+4*5,temp+32,4);
   pshufd m0,m0,0
   pshufd m1,m3,85
   pshufd m2,m3,170
   pshufd m3,m3,255
   mulps m0,[weightsq+496]
   mulps m1,[weightsq+496+16]
   mulps m2,[weightsq+496+32]
   mulps m3,[weightsq+496+48]
   pshufd m4,m7,0
   pshufd m5,m7,85
   pshufd m6,m7,170
   pshufd m7,m7,255
   mulps m4,[weightsq+496+64]
   mulps m5,[weightsq+496+80]
   mulps m6,[weightsq+496+96]
   mulps m7,[weightsq+496+112]
   addps m0,m1
   addps m2,m3
   addps m4,m5
   addps m6,m7
   addps m0,m2
   addps m4,m6
   addps m0,m4

   addps m0,[weightsq+496+128]
   movhlps m1,m0
   maxps m0,m1
   pshuflw m1,m0,14
   comiss m1,m0
   jbe .finish
   xor r3,r3
.finish:
   mov [dq],r3b
   RET


; parameters:
;  float *s,
;  const int n
INIT_XMM
cglobal e1_m16_SSE2, 2, 2, 6
.eloop8:
   movaps m0,[r0]
   movaps m3,[r0+16]
   minps m0,[exp_hi]
   minps m3,[exp_hi]
   maxps m0,[exp_lo]
   maxps m3,[exp_lo]
   mulps m0,[e1_scale]
   mulps m3,[e1_scale]
   movaps m1,m0
   movaps m4,m3
   addps m0,[e1_bias]
   addps m3,[e1_bias]
   movaps m2,m0
   movaps m5,m3
   subps m0,[e1_bias]
   subps m3,[e1_bias]
   pslld m2,23
   pslld m5,23
   subps m1,m0
   subps m4,m3
   movaps m0,m1
   movaps m3,m4
   mulps m1,m1
   mulps m4,m4
   mulps m0,[e1_c1]
   mulps m3,[e1_c1]
   mulps m1,[e1_c2]
   mulps m4,[e1_c2]
   addps m0,[e1_c0]
   addps m3,[e1_c0]
   addps m0,m1
   addps m3,m4
   paddd m0,m2
   paddd m3,m5
   movaps [r0],m0
   movaps [r0+16],m3
   add r0,32
   sub r1,8
   jnz .eloop8
   RET


; parameters:
;  float *s,
;  const int n
INIT_XMM
cglobal e2_m16_SSE2, 2, 2, 7
.eloop4:
   movaps m0,[r0]
   minps m0,[exp_hi]
   maxps m0,[exp_lo]
   movaps m1,[exp_rln2]
   mulps m1,m0
   xorps m2,m2
   addps m1,[am_0p5]
   cmpnltps m2,m1
   pand m2,[epi32_1]
   cvttps2dq m1,m1
   movaps m4,[exp_c2]
   psubd m1,m2
   movaps m5,[exp_c1]
   cvtdq2ps m3,m1
   mulps m4,m3
   mulps m5,m3
   movaps m6,[exp_q0]
   subps m0,m4
   movaps m4,[exp_p0]
   subps m0,m5
   paddd m1,[epi32_0x7f]
   movaps m2,m0
   mulps m0,m0
   mulps m6,m0
   mulps m4,m0
   addps m6,[exp_q1]
   addps m4,[exp_p1]
   mulps m6,m0
   mulps m4,m0
   addps m6,[exp_q2]
   mulps m4,m2
   mulps m6,m0
   movaps m0,[am_1]
   addps m2,m4
   addps m6,[exp_q3]
   pslld m1,23
   subps m6,m2
   rcpps m6,m6
   mulps m2,m6
   addps m2,m2
   addps m0,m2
   mulps m0,m1
   movaps [r0],m0
   add r0,16
   sub r1,4
   jnz .eloop4
   RET


; parameters:
;  const uint8_t *srcp,
;  const int stride,
;  const int xdia,
;  const int ydia,
;  float *mstd,
;  float *input
INIT_XMM
cglobal extract_m8_SSE2, 6, 7, 8, srcp, stride, xdia, ydia, mstd, input
   lea r6,[srcpq+strideq*2]
   pxor m5,m5 ;// sum
   pxor m6,m6 ;// sumsq
   pxor m3,m3
   PUSH ydiaq
   PUSH mstdq ; r4
.yloop2:
   xor r4,r4
.xloop2:
   movq m0,[srcpq+r4]
   movq m2,[r6+r4]
   punpcklbw m0,m3
   punpcklbw m2,m3
   movdqa m1,m0
   movdqa m4,m2
   punpcklwd m0,m3
   punpckhwd m1,m3
   punpcklwd m2,m3
   punpckhwd m4,m3
   cvtdq2ps m0,m0
   cvtdq2ps m1,m1
   cvtdq2ps m2,m2
   cvtdq2ps m4,m4
   movaps [inputq],m0
   movaps [inputq+16],m1
   movaps [inputq+xdiaq*4],m2
   movaps [inputq+xdiaq*4+16],m4
   addps m5,m0
   addps m5,m1
   addps m5,m2
   addps m5,m4
   mulps m0,m0
   mulps m1,m1
   mulps m2,m2
   mulps m4,m4
   addps m0,m1
   addps m2,m4
   addps m6,m0
   addps m6,m2
   add r4,8
   add inputq,32
   cmp r4,xdiaq
   jl .xloop2
   lea srcpq,[srcpq+strideq*4]
   lea r6,[r6+strideq*4]
   lea inputq,[inputq+xdiaq*4]
   sub ydiaq,2
   jnz .yloop2
   POP mstdq
   POP ydiaq

   movhlps m0,m5
   movhlps m1,m6
   movd m2,xdiad
   movd m4,ydiad
   pmuludq m2,m4
   addps m5,m0
   addps m6,m1
   cvtdq2ps m7,m2
   pshuflw m0,m5,14
   pshuflw m1,m6,14
   rcpss m7,m7 ;// scale
   addss m5,m0
   addss m6,m1
   mulss m5,m7 ;// mean
   mulss m6,m7
   movss [mstdq],m5
   mulss m5,m5
   subss m6,m5 ;// var
   comiss m6,[flt_epsilon_sse]
   jbe .novarjmp
   rsqrtss m6,m6 ;// 1.0/std
   rcpss m5,m6 ;// std
   movss [mstdq+4],m5
   movss [mstdq+8],m6
   jmp .finish
.novarjmp:
   movss [mstdq+4],m3
   movss [mstdq+8],m3
.finish:
   movss [mstdq+12],m3
   RET


; Used by default with 16 bit input
;
; parameters:
;  const float *data,
;  const float *weights,
;  float *vals,
;  const int n,
;  const int len,
;  const float *istd
INIT_XMM
cglobal dotProd_SSE2, 6, 7, 8, data, weights, vals, n, len, istd
   PUSH valsq
   PUSH nq
   PUSH istdq ; r5
   mov r5,dataq
   mov r6d,lend
.nloop:
   mov dataq,r5
   xorps m0,m0
   xorps m1,m1
   xorps m2,m2
   xorps m3,m3
   mov lend,r6d
.lloop:
   movaps m4,[dataq]
   movaps m5,m4
   movaps m6,m4
   movaps m7,m4
   mulps m4,[weightsq]
   mulps m5,[weightsq+16]
   mulps m6,[weightsq+32]
   mulps m7,[weightsq+48]
   addps m0,m4
   addps m1,m5
   addps m2,m6
   addps m3,m7

   add dataq,16
   add weightsq,64
   sub lend,4
   jnz .lloop

   ; This block performs a horizontal sum of each accumulator (m0..m3) and packs the results in m6 (sum(m3) sum(m2) sum(m1) sum(m0)).
   ; Sadly replacing the twelve instructions with three haddps makes no difference whatsoever on this Core 2 Duo.
   movaps m4,m0
   movaps m5,m2
   unpcklpd m0,m1
   unpcklpd m2,m3
   unpckhpd m4,m1
   unpckhpd m5,m3
   addps m0,m4
   addps m2,m5
   movaps m6,m0
   shufps m0,m2,136
   shufps m6,m2,221
   addps m6,m0

   movaps [valsq],m6
   add valsq,16
   sub nq,4
   jnz .nloop
   POP istdq
   POP nq
   POP valsq

   movss m7,[istdq]
   shufps m7,m7,0
   xor r5,r5
.aloop:
   movaps m0,[valsq+r5]
   mulps m0,m7
   addps m0,[weightsq+r5]
   movaps [valsq+r5],m0
   add r5,16
   sub nq,4
   jnz .aloop
   RET


; parameters:
;  const float *data,
;  const float *weights,
;  float *vals,
;  const int n,
;  const int len,
;  const float *istd
INIT_XMM
cglobal dotProd_FMA3, 6, 7, 8, data, weights, vals, n, len, istd
   PUSH valsq
   PUSH nq
   PUSH istdq ; r5
   mov r5,dataq
   mov r6d,lend
.nloop:
   mov dataq,r5
   xorps m0,m0
   xorps m1,m1
   xorps m2,m2
   xorps m3,m3
   mov lend,r6d
.lloop:
   movaps m4,[dataq]
   movaps m5,m4
   movaps m6,m4
   movaps m7,m4

   vfmadd231ps m0, m4, [weightsq]
   vfmadd231ps m1, m5, [weightsq + 16]
   vfmadd231ps m2, m6, [weightsq + 32]
   vfmadd231ps m3, m7, [weightsq + 48]

   add dataq,16
   add weightsq,64
   sub lend,4
   jnz .lloop

   haddps m0, m1
   haddps m2, m3
   haddps m0, m2

   movaps [valsq],m0
   add valsq,16
   sub nq,4
   jnz .nloop
   POP istdq
   POP nq
   POP valsq

   movss m7,[istdq]
   shufps m7,m7,0
   xor r5,r5
.aloop:
   movaps m0,[valsq+r5]
   vfmadd213ps m0, m7, [weightsq+r5]
   movaps [valsq+r5],m0
   add r5,16
   sub nq,4
   jnz .aloop
   RET


; parameters:
;  const float *data,
;  const float *weights,
;  float *vals,
;  const int n,
;  const int len,
;  const float *istd
INIT_XMM
cglobal dotProd_FMA4, 6, 7, 8, data, weights, vals, n, len, istd
   PUSH valsq
   PUSH nq
   PUSH istdq ; r5
   mov r5,dataq
   mov r6d,lend
.nloop:
   mov dataq,r5
   xorps m0,m0
   xorps m1,m1
   xorps m2,m2
   xorps m3,m3
   mov lend,r6d
.lloop:
   movaps m4,[dataq]
   movaps m5,m4
   movaps m6,m4
   movaps m7,m4

   vfmaddps m0, m4, [weightsq], m0
   vfmaddps m1, m5, [weightsq + 16], m1
   vfmaddps m2, m6, [weightsq + 32], m2
   vfmaddps m3, m7, [weightsq + 48], m3

   add dataq,16
   add weightsq,64
   sub lend,4
   jnz .lloop

   haddps m0, m1
   haddps m2, m3
   haddps m0, m2

   movaps [valsq],m0
   add valsq,16
   sub nq,4
   jnz .nloop
   POP istdq
   POP nq
   POP valsq

   movss m7,[istdq]
   shufps m7,m7,0
   xor r5,r5
.aloop:
   movaps m0,[valsq+r5]
   vfmaddps m0, m0, m7, [weightsq+r5]
   movaps [valsq+r5],m0
   add r5,16
   sub nq,4
   jnz .aloop
   RET


; parameters:
;  const float *dataf,
;  const float *weightsf,
;  float *vals,
;  const int n,
;  const int len,
;  const float *istd
INIT_XMM
cglobal dotProd_i16_SSE2, 6, 7, 8, data, weights, vals, n, len, istd
   PUSH valsq
   PUSH nq
   PUSH istdq ; r5
   mov r5,dataq
   mov r6d,lend
.nloop:
   mov dataq,r5
   pxor m0,m0
   pxor m1,m1
   pxor m2,m2
   pxor m3,m3
   mov lend,r6d
.lloop:
   movdqa m4,[dataq]
   movdqa m5,m4
   movdqa m6,m4
   movdqa m7,m4
   pmaddwd m4,[weightsq]
   pmaddwd m5,[weightsq+16]
   pmaddwd m6,[weightsq+32]
   pmaddwd m7,[weightsq+48]
   paddd m0,m4
   paddd m1,m5
   paddd m2,m6
   paddd m3,m7

   add dataq,16
   add weightsq,64
   sub lend,8
   jnz .lloop
   movdqa m4,m0
   movdqa m5,m2
   punpcklqdq m0,m1
   punpcklqdq m2,m3
   punpckhqdq m4,m1
   punpckhqdq m5,m3
   paddd m0,m4
   paddd m2,m5
   movdqa m6,m0
   shufps m0,m2,136
   shufps m6,m2,221
   paddd m6,m0
   movdqa [valsq],m6
   add valsq,16
   sub nq,4
   jnz .nloop
   POP istdq
   POP nq
   POP valsq

   movss m7,[istdq]
   pshufd m7,m7,0
   xor r5,r5
.aloop:
   movdqa m0,[valsq+r5]
   cvtdq2ps m0,m0
   mulps m0,[weightsq+r5*2]
   mulps m0,m7
   addps m0,[weightsq+r5*2+16]
   movaps [valsq+r5],m0
   add r5,16
   sub nq,4
   jnz .aloop
   RET


