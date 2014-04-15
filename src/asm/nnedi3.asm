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

sse_half times 4 dd 0.5

e1_scale times 4 dd 1.4426950409 ; 1/ln(2)
e1_bias  times 4 dd 12582912.0 ; 3 << 22
e1_c0    times 4 dd 1.00035
e1_c1    times 4 dd 0.701277797
e1_c2    times 4 dd 0.237348593

am_0p5      times 4 dd 0.5 ; this is the same as sse_half...
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

SECTION .text

; parameters:
; const uint8_t *t, const int pitch, float *pf
; r0 - t
; r1 - pitch
; r2 - pf
INIT_XMM
cglobal uc2s48_SSE2, 3, 4, 8
   lea r3, [r0 + r1 * 4]
   movq m0, [r0]
   movd m1, [r0 + 8]
   movd m2, [r0 + r1 * 2]
   movq m3, [r0 + r1 * 2 + 4]

   movq m4, [r3]
   movd m5, [r3 + 8]
   movd m6, [r3 + r1 * 2]
   movq m7, [r3 + r1 * 2 + 4]

   punpckldq m1, m2
   pxor m2, m2
   punpckldq m5, m6

   punpcklbw m0, m2
   punpcklbw m3, m2
   punpcklbw m1, m2
   punpcklbw m4, m2
   punpcklbw m5, m2
   punpcklbw m7, m2

   mova [r2], m0
   mova [r2 + 16], m1
   mova [r2 + 32], m3
   mova [r2 + 48], m4
   mova [r2 + 64], m5
   mova [r2 + 80], m7

   RET

INIT_XMM
cglobal uc2s64_SSE2, 3, 3, 8
   pxor m7, m7
   
   movq m0, [r0]
   movq m1, [r0 + 8]
   movq m2, [r0 + r1*2]
   movq m3, [r0 + r1*2 + 8]
   
   punpcklbw m0, m7
   punpcklbw m1, m7
   punpcklbw m2, m7
   punpcklbw m3, m7
   
   mova [r2], m0
   mova [r2 + 16], m1
   mova [r2 + 32], m2
   mova [r2 + 48], m3
   
   lea r0, [r0 + r1*4]
   
   movq m4, [r0]
   movq m5, [r0 + 8]
   movq m6, [r0 + r1*2]
   movq m0, [r0 + r1*2 + 8]
   
   punpcklbw m4, m7
   punpcklbw m5, m7
   punpcklbw m6, m7
   punpcklbw m0, m7
   
   mova [r2 + 64], m4
   mova [r2 + 80], m5
   mova [r2 + 96], m6
   mova [r2 + 112], m0

   RET

; parameters:
; const float *datai, const float *weights, uint8_t *d
INIT_XMM
cglobal computeNetwork0new_SSE2, 3, 3, 8
   mova m0,[r0]
   mova m1,m0
   mova m2,m0
   mova m3,m0

   pmaddwd m0,[r1]
   pmaddwd m1,[r1+16]
   pmaddwd m2,[r1+32]
   pmaddwd m3,[r1+48]

   mova m4,[r0+16]
   mova m5,m4
   mova m6,m4
   mova m7,m4

   pmaddwd m4,[r1+64]
   pmaddwd m5,[r1+80]
   pmaddwd m6,[r1+96]
   pmaddwd m7,[r1+112]

   paddd m0,m4
   paddd m1,m5
   paddd m2,m6
   paddd m3,m7

   mova m4,[r0+32]
   mova m5,m4
   mova m6,m4
   mova m7,m4

   pmaddwd m4,[r1+128]
   pmaddwd m5,[r1+144]
   pmaddwd m6,[r1+160]
   pmaddwd m7,[r1+176]

   paddd m0,m4
   paddd m1,m5
   paddd m2,m6
   paddd m3,m7

   mova m4,[r0+48]
   mova m5,m4
   mova m6,m4
   mova m7,m4
   
   pmaddwd m4,[r1+192]
   pmaddwd m5,[r1+208]
   pmaddwd m6,[r1+224]
   pmaddwd m7,[r1+240]
   
   paddd m0,m4
   paddd m1,m5
   paddd m2,m6
   paddd m3,m7
   
   mova m4,[r0+64]
   mova m5,m4
   mova m6,m4
   mova m7,m4
   
   pmaddwd m4,[r1+256]
   pmaddwd m5,[r1+272]
   pmaddwd m6,[r1+288]
   pmaddwd m7,[r1+304]
   
   paddd m0,m4
   paddd m1,m5
   paddd m2,m6
   paddd m3,m7
   
   mova m4,[r0+80]
   mova m5,m4
   mova m6,m4
   mova m7,m4
   
   pmaddwd m4,[r1+320]
   pmaddwd m5,[r1+336]
   pmaddwd m6,[r1+352]
   pmaddwd m7,[r1+368]
   
   paddd m0,m4
   paddd m1,m5
   paddd m2,m6
   paddd m3,m7
   
   mova m4,[r0+96]
   mova m5,m4
   mova m6,m4
   mova m7,m4
   
   pmaddwd m4,[r1+384]
   pmaddwd m5,[r1+400]
   pmaddwd m6,[r1+416]
   pmaddwd m7,[r1+432]
   
   paddd m0,m4
   paddd m1,m5
   paddd m2,m6
   paddd m3,m7
   
   mova m4,[r0+112]
   mova m5,m4
   mova m6,m4
   mova m7,m4
   
   pmaddwd m4,[r1+448]
   pmaddwd m5,[r1+464]
   pmaddwd m6,[r1+480]
   pmaddwd m7,[r1+496]
   
   paddd m0,m4
   paddd m1,m5
   paddd m2,m6
   paddd m3,m7
   
   mova m4,m0
   mova m5,m2
   
   punpcklqdq m0,m1
   punpcklqdq m2,m3
   
   punpckhqdq m4,m1
   punpckhqdq m5,m3
   
   paddd m0,m4
   paddd m2,m5
   
   mova m6,m0
   
   shufps m0,m2,136
   shufps m6,m2,221
   
   paddd m0,m6
   cvtdq2ps m0,m0
   mulps m0,[r1+512]
   addps m0,[r1+528]
   movaps m1,m0
   andps m0,[sign_bits_f]
   addps m0,[ones_f]
   rcpps m0,m0
   mulps m0,m1
   
   pshufd m1,m0,0
   pshufd m2,m0,85
   pshufd m3,m0,170
   pshufd m4,m0,255

   mulps m1,[r1+544]
   mulps m2,[r1+560]
   mulps m3,[r1+576]
   mulps m4,[r1+592]
   
   pxor m0,m0
   
   addps m1,m2
   addps m3,m4
   addps m1,m3
   
   addps m1,[r1+608]
   ; yasm wouldn't take this cmpps
   ;cmpps m1,m0,1
   cmpltps m1,m0
   packssdw m1,m0
   packsswb m1,m0
   movd r1d,m1
   xor r1d,0xFFFFFFFF
   and r1d,0x01010101
   mov [r2],r1d
   RET

; parameters:
; const uint8_t *tempu, int width, uint8_t *dstp, const uint8_t *src3p, const int src_pitch
; r0 - tempu
; r1 - width
; r2 - dstp
; r3 - src3p
; r4 - src_pitch
INIT_XMM
cglobal processLine0_SSE2, 5, 6, 8
%if WIN64
   ; The parameter is 32 bit. Make sure the high 32 bits are cleared.
   shl r4, 32
   shr r4, 32
%endif
   lea r5,[r3+r4*4]
   pxor m6,m6
   pxor m7,m7
.xloop:
   mova m0,[r3+r4*2]
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
   mova m1,[r3]
   mova m3,[r5+r4*2]
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
   mova m3,[r0]
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
   mova [r2],m0
   paddusw m5,m2
   paddusw m6,m5
   add r3,16
   add r5,16
   add r0,16
   add r2,16
   sub r1,16
   jnz .xloop
   movd eax,m6
   RET


; parameters:
; const float *w, const int n, float *mstd
INIT_XMM
cglobal weightedAvgElliottMul5_m16_SSE2, 3, 5, 8
   ;push edi ; why?
   lea r3,[r0+r1*4]
   xor r4,r4
   xorps m0,m0 ; sum w
   xorps m1,m1 ; sum w*v
.nloop:
   movaps m2,[r0+r4*4]
   movaps m3,[r0+r4*4+16]
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
   movaps m2,[r0+r4*4+32]
   movaps m3,[r0+r4*4+48]
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
   sub r1,16
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
   mulss m1,[r2+4]
   addss m1,[r2]
   addss m1,[r2+12]
   movss [r2+12],m1
   ;pop edi
   RET


; parameters:
; const uint8_t *srcp, const int stride, const int xdia, const int ydia, float *mstd, float *inputf
INIT_XMM
cglobal extract_m8_i16_SSE2, 6, 7, 8
   lea r6,[r0+r1*2]
   pxor m4,m4 ; sum
   pxor m5,m5 ; sumsq
   pxor m6,m6
   PUSH r3
   PUSH r4
.yloop:
   xor r4,r4
.xloop:
   movq m0,[r0+r4]
   movq m1,[r6+r4]
   mova m2,m0
   mova m3,m1
   punpcklbw m0,m6
   punpcklbw m1,m6
   psadbw m2,m6
   psadbw m3,m6
   mova [r5],m0
   mova [r5+r2*2],m1
   pmaddwd m0,m0
   pmaddwd m1,m1
   paddd m4,m2
   paddd m5,m0
   paddd m4,m3
   paddd m5,m1
   add r4,8
   add r5,16
   cmp r4,r2
   jl .xloop

   lea r0,[r0+r1*4]
   lea r6,[r6+r1*4]
   lea r5,[r5+r2*2]
   sub r3,2
   jnz .yloop
   POP r4
   POP r3

   movhlps m1,m5
   paddd m5,m1
   ; multiply r3d by r2d
   movd m2,r2d
   movd m3,r3d
   pmuludq m2,m3
   cvtdq2ps m7,m2

   pshuflw m1,m5,14
   paddd m5,m1
   rcpss m7,m7 ; scale
   cvtdq2ps m4,m4
   cvtdq2ps m5,m5
   mulss m4,m7 ; mean
   mulss m5,m7
   movss [r4],m4
   mulss m4,m4
   subss m5,m4 ; var
   comiss m5,[flt_epsilon_sse]
   jbe .novarjmp
   rsqrtss m5,m5 ; 1.0/std
   rcpss m4,m5 ; std
   movss [r4+4],m4
   movss [r4+8],m5
   jmp .finish
.novarjmp:
   movss [r4+4],m6
   movss [r4+8],m6
.finish:
   movss [r4+12],m6
   RET


; parameters:
;  const float *dataf,
;  const float *weightsf,
;  float *vals,
;  const int n,
;  const int len,
;  const float *istd
INIT_XMM
cglobal dotProd_m32_m16_i16_SSE2, 6, 7, 8
   mov r6,r0 ; r6, r0 - dataf
   PUSH r2   ; r2 - vals
   PUSH r3   ; r3 - n
.nloop:
   mov r0,r6 ; r0, r6 - dataf
   pxor m0,m0
   pxor m1,m1
   pxor m2,m2
   pxor m3,m3
   ; original value of r4 needs to be restored here
   PUSH r4   ; r4 - len
.lloop:
   mova m4,[r0] ; r0 - dataf
   mova m5,m4
   mova m6,m4
   mova m7,m4
   pmaddwd m4,[r1] ; r1 - weightsf
   pmaddwd m5,[r1+16]
   pmaddwd m6,[r1+32]
   pmaddwd m7,[r1+48]
   paddd m0,m4
   paddd m1,m5
   paddd m2,m6
   paddd m3,m7
   mova m4,[r0+16] ; r0 - dataf
   mova m5,m4
   mova m6,m4
   mova m7,m4
   pmaddwd m4,[r1+64] ; r1 - weightsf
   pmaddwd m5,[r1+80]
   pmaddwd m6,[r1+96]
   pmaddwd m7,[r1+112]
   paddd m0,m4
   paddd m1,m5
   paddd m2,m6
   paddd m3,m7
   mova m4,[r0+32] ; r0 - dataf
   mova m5,m4
   mova m6,m4
   mova m7,m4
   pmaddwd m4,[r1+128] ; r1 - weightsf
   pmaddwd m5,[r1+144]
   pmaddwd m6,[r1+160]
   pmaddwd m7,[r1+176]
   paddd m0,m4
   paddd m1,m5
   paddd m2,m6
   paddd m3,m7
   mova m4,[r0+48] ; r0 - dataf
   mova m5,m4
   mova m6,m4
   mova m7,m4
   pmaddwd m4,[r1+192] ; r1 - weightsf
   pmaddwd m5,[r1+208]
   pmaddwd m6,[r1+224]
   pmaddwd m7,[r1+240]
   paddd m0,m4
   paddd m1,m5
   paddd m2,m6
   paddd m3,m7
   add r0,64 ; r0 - dataf
   add r1,256 ; r1 - weightsf
   sub r4d,32 ; r4 - len
   jnz .lloop
   POP r4 ; r4 - len (original)
   mova m4,m0
   mova m5,m2
   punpcklqdq m0,m1
   punpcklqdq m2,m3
   punpckhqdq m4,m1
   punpckhqdq m5,m3
   paddd m0,m4
   paddd m2,m5
   mova m6,m0
   shufps m0,m2,136
   shufps m6,m2,221
   paddd m6,m0
   mova [r2],m6 ; r2 - vals
   add r2,16 ; r2 - vals
   sub r3,4 ; r3 - n
   jnz .nloop
   POP r3 ; r3 - n (original)
   POP r2 ; r2 - vals (original)
   movss m7,[r5] ; r5 - istd
   pshufd m7,m7,0
   xor r5,r5 ; r5 - 0
.aloop:
   mova m0,[r2+r5*4] ; r2 - vals (original)
   mova m1,[r2+r5*4+16]
   mova m2,[r2+r5*4+32]
   mova m3,[r2+r5*4+48]
   cvtdq2ps m0,m0
   cvtdq2ps m1,m1
   cvtdq2ps m2,m2
   cvtdq2ps m3,m3
   mulps m0,[r1+r5*8] ; r1 - weightsf
   mulps m1,[r1+r5*8+32]
   mulps m2,[r1+r5*8+64]
   mulps m3,[r1+r5*8+96]
   mulps m0,m7
   mulps m1,m7
   mulps m2,m7
   mulps m3,m7
   addps m0,[r1+r5*8+16] ; r1 - weightsf
   addps m1,[r1+r5*8+48]
   addps m2,[r1+r5*8+80]
   addps m3,[r1+r5*8+112]
   movaps [r2+r5*4],m0 ; r2 - vals
   movaps [r2+r5*4+16],m1
   movaps [r2+r5*4+32],m2
   movaps [r2+r5*4+48],m3
   add r5,16
   sub r3,16 ; r3 - n
   jnz .aloop
   RET


; parameters:
;  float *s,
;  const int n
INIT_XMM
cglobal e0_m16_SSE2, 2, 2, 4
   ;mov r0,[esp+4]
   ;mov r1,[esp+8]
.eloop16:
   movaps m0,[r0]
   movaps m1,[r0+16]
   movaps m2,[r0+32]
   movaps m3,[r0+48]
   minps m0,[exp_hi]
   minps m1,[exp_hi]
   minps m2,[exp_hi]
   minps m3,[exp_hi]
   maxps m0,[exp_lo]
   maxps m1,[exp_lo]
   maxps m2,[exp_lo]
   maxps m3,[exp_lo]
   mulps m0,[e0_mult]
   mulps m1,[e0_mult]
   mulps m2,[e0_mult]
   mulps m3,[e0_mult]
   addps m0,[e0_bias]
   addps m1,[e0_bias]
   addps m2,[e0_bias]
   addps m3,[e0_bias]
   cvtps2dq m0,m0
   cvtps2dq m1,m1
   cvtps2dq m2,m2
   cvtps2dq m3,m3
   movaps [r0],m0
   movaps [r0+16],m1
   movaps [r0+32],m2
   movaps [r0+48],m3
   add r0,64
   sub r1,16
   jnz .eloop16
   RET


; parameters:
;  const float *val,
;  const float *scale,
;  uint8_t *dstp
INIT_XMM
cglobal castScale_SSE, 3, 3, 1
   movss m0,[r0+12]
   mulss m0,[r1]
   addss m0,[sse_half]
   cvttss2si r1,m0
   cmp r1,255
   jl .b255
   mov r1,255
   jmp .finish
.b255:
   cmp r1,0
   jge .finish
   xor r1,r1
.finish:
   mov byte [r2],r1b ; lowest 8 bits of r1
   RET


; parameters:
;  const float *input,
;  const float *weights,
;  uint8_t *d
INIT_XMM
cglobal computeNetwork0_SSE2, 3, 4, 8
   ;//    dotProd48_m4_SSE(input,weights,temp,4);
   ;mov r0,[esp+4]
   ;mov r1,[esp+8]
   mov r3,1
   movaps m0,[r0]
   movaps m1,m0
   movaps m2,m0
   movaps m3,m0
   mulps m0,[r1]
   mulps m1,[r1+16]
   mulps m2,[r1+32]
   mulps m3,[r1+48]
   movaps m4,[r0+16]
   movaps m5,m4
   movaps m6,m4
   movaps m7,m4
   mulps m4,[r1+64]
   mulps m5,[r1+80]
   mulps m6,[r1+96]
   mulps m7,[r1+112]
   addps m0,m4
   addps m1,m5
   addps m2,m6
   addps m3,m7
   movaps m4,[r0+32]
   movaps m5,m4
   movaps m6,m4
   movaps m7,m4
   mulps m4,[r1+128]
   mulps m5,[r1+144]
   mulps m6,[r1+160]
   mulps m7,[r1+176]
   addps m0,m4
   addps m1,m5
   addps m2,m6
   addps m3,m7
   movaps m4,[r0+48]
   movaps m5,m4
   movaps m6,m4
   movaps m7,m4
   mulps m4,[r1+192]
   mulps m5,[r1+208]
   mulps m6,[r1+224]
   mulps m7,[r1+240]
   addps m0,m4
   addps m1,m5
   addps m2,m6
   addps m3,m7
   movaps m4,[r0+64]
   movaps m5,m4
   movaps m6,m4
   movaps m7,m4
   mulps m4,[r1+256]
   mulps m5,[r1+272]
   mulps m6,[r1+288]
   mulps m7,[r1+304]
   addps m0,m4
   addps m1,m5
   addps m2,m6
   addps m3,m7
   movaps m4,[r0+80]
   movaps m5,m4
   movaps m6,m4
   movaps m7,m4
   mulps m4,[r1+320]
   mulps m5,[r1+336]
   mulps m6,[r1+352]
   mulps m7,[r1+368]
   addps m0,m4
   addps m1,m5
   addps m2,m6
   addps m3,m7
   movaps m4,[r0+96]
   movaps m5,m4
   movaps m6,m4
   movaps m7,m4
   mulps m4,[r1+384]
   mulps m5,[r1+400]
   mulps m6,[r1+416]
   mulps m7,[r1+432]
   addps m0,m4
   addps m1,m5
   addps m2,m6
   addps m3,m7
   movaps m4,[r0+112]
   movaps m5,m4
   movaps m6,m4
   movaps m7,m4
   mulps m4,[r1+448]
   mulps m5,[r1+464]
   mulps m6,[r1+480]
   mulps m7,[r1+496]
   addps m0,m4
   addps m1,m5
   addps m2,m6
   addps m3,m7
   movaps m4,[r0+128]
   movaps m5,m4
   movaps m6,m4
   movaps m7,m4
   mulps m4,[r1+512]
   mulps m5,[r1+528]
   mulps m6,[r1+544]
   mulps m7,[r1+560]
   addps m0,m4
   addps m1,m5
   addps m2,m6
   addps m3,m7
   movaps m4,[r0+144]
   movaps m5,m4
   movaps m6,m4
   movaps m7,m4
   mulps m4,[r1+576]
   mulps m5,[r1+592]
   mulps m6,[r1+608]
   mulps m7,[r1+624]
   addps m0,m4
   addps m1,m5
   addps m2,m6
   addps m3,m7
   movaps m4,[r0+160]
   movaps m5,m4
   movaps m6,m4
   movaps m7,m4
   mulps m4,[r1+640]
   mulps m5,[r1+656]
   mulps m6,[r1+672]
   mulps m7,[r1+688]
   addps m0,m4
   addps m1,m5
   addps m2,m6
   addps m3,m7
   movaps m4,[r0+176]
   movaps m5,m4
   movaps m6,m4
   movaps m7,m4
   mulps m4,[r1+704]
   mulps m5,[r1+720]
   mulps m6,[r1+736]
   mulps m7,[r1+752]
   addps m0,m4
   addps m1,m5
   addps m2,m6
   addps m3,m7
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
   addps m0,m6
   addps m0,[r1+768]
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
   mulps m1,[r1+784]
   mulps m2,[r1+784+16]
   mulps m3,[r1+784+32]
   mulps m4,[r1+784+48]
   addps m1,m2
   addps m3,m4
   addps m1,m3
   addps m1,[r1+784+64]
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
   mulps m0,[r1+864]
   mulps m1,[r1+864+16]
   mulps m2,[r1+864+32]
   mulps m3,[r1+864+48]
   pshufd m4,m7,0
   pshufd m5,m7,85
   pshufd m6,m7,170
   pshufd m7,m7,255
   mulps m4,[r1+864+64]
   mulps m5,[r1+864+80]
   mulps m6,[r1+864+96]
   mulps m7,[r1+864+112]
   addps m0,m1
   addps m2,m3
   addps m4,m5
   addps m6,m7
   addps m0,m2
   addps m4,m6
   addps m0,m4
   ;mov ecx/r2,[esp+12]
   addps m0,[r1+864+128]
   movhlps m1,m0
   maxps m0,m1
   pshuflw m1,m0,14
   comiss m1,m0
   jbe .finish
   xor r3,r3
.finish:
   mov [r2],r3b
   RET


; parameters:
;  const uint8_t *t,
;  const int pitch,
;  float *p
INIT_XMM
cglobal uc2f48_SSE2, 3, 3, 7
   pxor m6,m6
   movq m0,[r0]
   movd m4,[r0+8]
   movq m2,[r0+r1*2]
   movd m5,[r0+r1*2+8]
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
   lea r0,[r0+r1*4]
   cvtdq2ps m4,m4
   cvtdq2ps m5,m5
   cvtdq2ps m0,m0
   cvtdq2ps m1,m1
   cvtdq2ps m2,m2
   cvtdq2ps m3,m3
   movaps [r2],m0
   movaps [r2+16],m1
   movaps [r2+32],m4
   movaps [r2+48],m2
   movaps [r2+64],m3
   movaps [r2+80],m5
   movq m0,[r0]
   movd m4,[r0+8]
   movq m2,[r0+r1*2]
   movd m5,[r0+r1*2+8]
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
   movaps [r2+96],m0
   movaps [r2+112],m1
   movaps [r2+128],m4
   movaps [r2+144],m2
   movaps [r2+160],m3
   movaps [r2+176],m5
   RET


; parameters:
;  const float *inputf,
;  const float *weightsf,
;  uint8_t *d
INIT_XMM
cglobal computeNetwork0_i16_SSE2, 3, 4, 8
   mov r3,1
   movdqa m0,[r0]
   movdqa m1,m0
   movdqa m2,m0
   movdqa m3,m0
   pmaddwd m0,[r1]
   pmaddwd m1,[r1+16]
   pmaddwd m2,[r1+32]
   pmaddwd m3,[r1+48]
   movdqa m4,[r0+16]
   movdqa m5,m4
   movdqa m6,m4
   movdqa m7,m4
   pmaddwd m4,[r1+64]
   pmaddwd m5,[r1+80]
   pmaddwd m6,[r1+96]
   pmaddwd m7,[r1+112]
   paddd m0,m4
   paddd m1,m5
   paddd m2,m6
   paddd m3,m7
   movdqa m4,[r0+32]
   movdqa m5,m4
   movdqa m6,m4
   movdqa m7,m4
   pmaddwd m4,[r1+128]
   pmaddwd m5,[r1+144]
   pmaddwd m6,[r1+160]
   pmaddwd m7,[r1+176]
   paddd m0,m4
   paddd m1,m5
   paddd m2,m6
   paddd m3,m7
   movdqa m4,[r0+48]
   movdqa m5,m4
   movdqa m6,m4
   movdqa m7,m4
   pmaddwd m4,[r1+192]
   pmaddwd m5,[r1+208]
   pmaddwd m6,[r1+224]
   pmaddwd m7,[r1+240]
   paddd m0,m4
   paddd m1,m5
   paddd m2,m6
   paddd m3,m7
   movdqa m4,[r0+64]
   movdqa m5,m4
   movdqa m6,m4
   movdqa m7,m4
   pmaddwd m4,[r1+256]
   pmaddwd m5,[r1+272]
   pmaddwd m6,[r1+288]
   pmaddwd m7,[r1+304]
   paddd m0,m4
   paddd m1,m5
   paddd m2,m6
   paddd m3,m7
   movdqa m4,[r0+80]
   movdqa m5,m4
   movdqa m6,m4
   movdqa m7,m4
   pmaddwd m4,[r1+320]
   pmaddwd m5,[r1+336]
   pmaddwd m6,[r1+352]
   pmaddwd m7,[r1+368]
   paddd m0,m4
   paddd m1,m5
   paddd m2,m6
   paddd m3,m7
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
   mulps m0,[r1+384]
   addps m0,[r1+400]
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
   mulps m1,[r1+416]
   mulps m2,[r1+416+16]
   mulps m3,[r1+416+32]
   mulps m4,[r1+416+48]
   addps m1,m2
   addps m3,m4
   addps m1,m3
   addps m1,[r1+416+64]
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
   mulps m0,[r1+496]
   mulps m1,[r1+496+16]
   mulps m2,[r1+496+32]
   mulps m3,[r1+496+48]
   pshufd m4,m7,0
   pshufd m5,m7,85
   pshufd m6,m7,170
   pshufd m7,m7,255
   mulps m4,[r1+496+64]
   mulps m5,[r1+496+80]
   mulps m6,[r1+496+96]
   mulps m7,[r1+496+112]
   addps m0,m1
   addps m2,m3
   addps m4,m5
   addps m6,m7
   addps m0,m2
   addps m4,m6
   addps m0,m4

   addps m0,[r1+496+128]
   movhlps m1,m0
   maxps m0,m1
   pshuflw m1,m0,14
   comiss m1,m0
   jbe .finish
   xor r3,r3
.finish:
   mov [r2],r3b
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
cglobal extract_m8_SSE2, 6, 7, 8
   lea r6,[r0+r1*2]
   pxor m5,m5 ;// sum
   pxor m6,m6 ;// sumsq
   pxor m3,m3
   PUSH r3
   PUSH r4
.yloop2:
   xor r4,r4
.xloop2:
   movq m0,[r0+r4]
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
   movaps [r5],m0
   movaps [r5+16],m1
   movaps [r5+r2*4],m2
   movaps [r5+r2*4+16],m4
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
   add r5,32
   cmp r4,r2
   jl .xloop2
   lea r0,[r0+r1*4]
   lea r6,[r6+r1*4]
   lea r5,[r5+r2*4]
   sub r3,2
   jnz .yloop2
   POP r4
   POP r3

   movhlps m0,m5
   movhlps m1,m6
   movd m2,r2d
   movd m4,r3d
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
   movss [r4],m5
   mulss m5,m5
   subss m6,m5 ;// var
   comiss m6,[flt_epsilon_sse]
   jbe .novarjmp
   rsqrtss m6,m6 ;// 1.0/std
   rcpss m5,m6 ;// std
   movss [r4+4],m5
   movss [r4+8],m6
   jmp .finish
.novarjmp:
   movss [r4+4],m3
   movss [r4+8],m3
.finish:
   movss [r4+12],m3
   RET


; parameters:
;  const float *data,
;  const float *weights,
;  float *vals,
;  const int n,
;  const int len,
;  const float *istd
INIT_XMM
cglobal dotProd_m32_m16_SSE2, 6, 7, 8
   PUSH r2
   PUSH r3
   PUSH r5
   mov r5,r0
   mov r6d,r4d
.nloop:
   mov r0,r5
   xorps m0,m0
   xorps m1,m1
   xorps m2,m2
   xorps m3,m3
   mov r4d,r6d
.lloop:
   movaps m4,[r0]
   movaps m5,m4
   movaps m6,m4
   movaps m7,m4
   mulps m4,[r1]
   mulps m5,[r1+16]
   mulps m6,[r1+32]
   mulps m7,[r1+48]
   addps m0,m4
   addps m1,m5
   addps m2,m6
   addps m3,m7
   movaps m4,[r0+16]
   movaps m5,m4
   movaps m6,m4
   movaps m7,m4
   mulps m4,[r1+64]
   mulps m5,[r1+80]
   mulps m6,[r1+96]
   mulps m7,[r1+112]
   addps m0,m4
   addps m1,m5
   addps m2,m6
   addps m3,m7
   movaps m4,[r0+32]
   movaps m5,m4
   movaps m6,m4
   movaps m7,m4
   mulps m4,[r1+128]
   mulps m5,[r1+144]
   mulps m6,[r1+160]
   mulps m7,[r1+176]
   addps m0,m4
   addps m1,m5
   addps m2,m6
   addps m3,m7
   movaps m4,[r0+48]
   movaps m5,m4
   movaps m6,m4
   movaps m7,m4
   mulps m4,[r1+192]
   mulps m5,[r1+208]
   mulps m6,[r1+224]
   mulps m7,[r1+240]
   addps m0,m4
   addps m1,m5
   addps m2,m6
   addps m3,m7
   movaps m4,[r0+64]
   movaps m5,m4
   movaps m6,m4
   movaps m7,m4
   mulps m4,[r1+256]
   mulps m5,[r1+272]
   mulps m6,[r1+288]
   mulps m7,[r1+304]
   addps m0,m4
   addps m1,m5
   addps m2,m6
   addps m3,m7
   movaps m4,[r0+80]
   movaps m5,m4
   movaps m6,m4
   movaps m7,m4
   mulps m4,[r1+320]
   mulps m5,[r1+336]
   mulps m6,[r1+352]
   mulps m7,[r1+368]
   addps m0,m4
   addps m1,m5
   addps m2,m6
   addps m3,m7
   movaps m4,[r0+96]
   movaps m5,m4
   movaps m6,m4
   movaps m7,m4
   mulps m4,[r1+384]
   mulps m5,[r1+400]
   mulps m6,[r1+416]
   mulps m7,[r1+432]
   addps m0,m4
   addps m1,m5
   addps m2,m6
   addps m3,m7
   movaps m4,[r0+112]
   movaps m5,m4
   movaps m6,m4
   movaps m7,m4
   mulps m4,[r1+448]
   mulps m5,[r1+464]
   mulps m6,[r1+480]
   mulps m7,[r1+496]
   addps m0,m4
   addps m1,m5
   addps m2,m6
   addps m3,m7
   add r0,128
   add r1,512
   sub r4d,32
   jnz .lloop
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
   movaps [r2],m6
   add r2,16
   sub r3,4
   jnz .nloop
   POP r5
   POP r3
   POP r2

   movss m7,[r5]
   shufps m7,m7,0
   xor r5,r5
.aloop:
   movaps m0,[r2+r5*4]
   movaps m1,[r2+r5*4+16]
   movaps m2,[r2+r5*4+32]
   movaps m3,[r2+r5*4+48]
   mulps m0,m7
   mulps m1,m7
   mulps m2,m7
   mulps m3,m7
   addps m0,[r1+r5*4]
   addps m1,[r1+r5*4+16]
   addps m2,[r1+r5*4+32]
   addps m3,[r1+r5*4+48]
   movaps [r2+r5*4],m0
   movaps [r2+r5*4+16],m1
   movaps [r2+r5*4+32],m2
   movaps [r2+r5*4+48],m3
   add r5,16
   sub r3,16
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
cglobal dotProd_m48_m16_i16_SSE2, 6, 7, 8
   PUSH r2
   PUSH r3
   PUSH r5
   mov r5,r0
   mov r6d,r4d
.nloop:
   mov r0,r5
   pxor m0,m0
   pxor m1,m1
   pxor m2,m2
   pxor m3,m3
   mov r4d,r6d
.lloop:
   movdqa m4,[r0]
   movdqa m5,m4
   movdqa m6,m4
   movdqa m7,m4
   pmaddwd m4,[r1]
   pmaddwd m5,[r1+16]
   pmaddwd m6,[r1+32]
   pmaddwd m7,[r1+48]
   paddd m0,m4
   paddd m1,m5
   paddd m2,m6
   paddd m3,m7
   movdqa m4,[r0+16]
   movdqa m5,m4
   movdqa m6,m4
   movdqa m7,m4
   pmaddwd m4,[r1+64]
   pmaddwd m5,[r1+80]
   pmaddwd m6,[r1+96]
   pmaddwd m7,[r1+112]
   paddd m0,m4
   paddd m1,m5
   paddd m2,m6
   paddd m3,m7
   movdqa m4,[r0+32]
   movdqa m5,m4
   movdqa m6,m4
   movdqa m7,m4
   pmaddwd m4,[r1+128]
   pmaddwd m5,[r1+144]
   pmaddwd m6,[r1+160]
   pmaddwd m7,[r1+176]
   paddd m0,m4
   paddd m1,m5
   paddd m2,m6
   paddd m3,m7
   movdqa m4,[r0+48]
   movdqa m5,m4
   movdqa m6,m4
   movdqa m7,m4
   pmaddwd m4,[r1+192]
   pmaddwd m5,[r1+208]
   pmaddwd m6,[r1+224]
   pmaddwd m7,[r1+240]
   paddd m0,m4
   paddd m1,m5
   paddd m2,m6
   paddd m3,m7
   movdqa m4,[r0+64]
   movdqa m5,m4
   movdqa m6,m4
   movdqa m7,m4
   pmaddwd m4,[r1+256]
   pmaddwd m5,[r1+272]
   pmaddwd m6,[r1+288]
   pmaddwd m7,[r1+304]
   paddd m0,m4
   paddd m1,m5
   paddd m2,m6
   paddd m3,m7
   movdqa m4,[r0+80]
   movdqa m5,m4
   movdqa m6,m4
   movdqa m7,m4
   pmaddwd m4,[r1+320]
   pmaddwd m5,[r1+336]
   pmaddwd m6,[r1+352]
   pmaddwd m7,[r1+368]
   paddd m0,m4
   paddd m1,m5
   paddd m2,m6
   paddd m3,m7
   add r0,96
   add r1,384
   sub r4d,48
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
   movdqa [r2],m6
   add r2,16
   sub r3,4
   jnz .nloop
   POP r5
   POP r3
   POP r2

   movss m7,[r5]
   pshufd m7,m7,0
   xor r5,r5
.aloop:
   movdqa m0,[r2+r5*4]
   movdqa m1,[r2+r5*4+16]
   movdqa m2,[r2+r5*4+32]
   movdqa m3,[r2+r5*4+48]
   cvtdq2ps m0,m0
   cvtdq2ps m1,m1
   cvtdq2ps m2,m2
   cvtdq2ps m3,m3
   mulps m0,[r1+r5*8]
   mulps m1,[r1+r5*8+32]
   mulps m2,[r1+r5*8+64]
   mulps m3,[r1+r5*8+96]
   mulps m0,m7
   mulps m1,m7
   mulps m2,m7
   mulps m3,m7
   addps m0,[r1+r5*8+16]
   addps m1,[r1+r5*8+48]
   addps m2,[r1+r5*8+80]
   addps m3,[r1+r5*8+112]
   movaps [r2+r5*4],m0
   movaps [r2+r5*4+16],m1
   movaps [r2+r5*4+32],m2
   movaps [r2+r5*4+48],m3
   add r5,16
   sub r3,16
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
cglobal dotProd_m48_m16_SSE2, 6, 7, 8
   PUSH r2
   PUSH r3
   PUSH r5
   mov r5,r0
   mov r6d,r4d
.nloop:
   mov r0,r5
   xorps m0,m0
   xorps m1,m1
   xorps m2,m2
   xorps m3,m3
   mov r4d,r6d
.lloop:
   movaps m4,[r0]
   movaps m5,m4
   movaps m6,m4
   movaps m7,m4
   mulps m4,[r1]
   mulps m5,[r1+16]
   mulps m6,[r1+32]
   mulps m7,[r1+48]
   addps m0,m4
   addps m1,m5
   addps m2,m6
   addps m3,m7
   movaps m4,[r0+16]
   movaps m5,m4
   movaps m6,m4
   movaps m7,m4
   mulps m4,[r1+64]
   mulps m5,[r1+80]
   mulps m6,[r1+96]
   mulps m7,[r1+112]
   addps m0,m4
   addps m1,m5
   addps m2,m6
   addps m3,m7
   movaps m4,[r0+32]
   movaps m5,m4
   movaps m6,m4
   movaps m7,m4
   mulps m4,[r1+128]
   mulps m5,[r1+144]
   mulps m6,[r1+160]
   mulps m7,[r1+176]
   addps m0,m4
   addps m1,m5
   addps m2,m6
   addps m3,m7
   movaps m4,[r0+48]
   movaps m5,m4
   movaps m6,m4
   movaps m7,m4
   mulps m4,[r1+192]
   mulps m5,[r1+208]
   mulps m6,[r1+224]
   mulps m7,[r1+240]
   addps m0,m4
   addps m1,m5
   addps m2,m6
   addps m3,m7
   movaps m4,[r0+64]
   movaps m5,m4
   movaps m6,m4
   movaps m7,m4
   mulps m4,[r1+256]
   mulps m5,[r1+272]
   mulps m6,[r1+288]
   mulps m7,[r1+304]
   addps m0,m4
   addps m1,m5
   addps m2,m6
   addps m3,m7
   movaps m4,[r0+80]
   movaps m5,m4
   movaps m6,m4
   movaps m7,m4
   mulps m4,[r1+320]
   mulps m5,[r1+336]
   mulps m6,[r1+352]
   mulps m7,[r1+368]
   addps m0,m4
   addps m1,m5
   addps m2,m6
   addps m3,m7
   movaps m4,[r0+96]
   movaps m5,m4
   movaps m6,m4
   movaps m7,m4
   mulps m4,[r1+384]
   mulps m5,[r1+400]
   mulps m6,[r1+416]
   mulps m7,[r1+432]
   addps m0,m4
   addps m1,m5
   addps m2,m6
   addps m3,m7
   movaps m4,[r0+112]
   movaps m5,m4
   movaps m6,m4
   movaps m7,m4
   mulps m4,[r1+448]
   mulps m5,[r1+464]
   mulps m6,[r1+480]
   mulps m7,[r1+496]
   addps m0,m4
   addps m1,m5
   addps m2,m6
   addps m3,m7
   movaps m4,[r0+128]
   movaps m5,m4
   movaps m6,m4
   movaps m7,m4
   mulps m4,[r1+512]
   mulps m5,[r1+528]
   mulps m6,[r1+544]
   mulps m7,[r1+560]
   addps m0,m4
   addps m1,m5
   addps m2,m6
   addps m3,m7
   movaps m4,[r0+144]
   movaps m5,m4
   movaps m6,m4
   movaps m7,m4
   mulps m4,[r1+576]
   mulps m5,[r1+592]
   mulps m6,[r1+608]
   mulps m7,[r1+624]
   addps m0,m4
   addps m1,m5
   addps m2,m6
   addps m3,m7
   movaps m4,[r0+160]
   movaps m5,m4
   movaps m6,m4
   movaps m7,m4
   mulps m4,[r1+640]
   mulps m5,[r1+656]
   mulps m6,[r1+672]
   mulps m7,[r1+688]
   addps m0,m4
   addps m1,m5
   addps m2,m6
   addps m3,m7
   movaps m4,[r0+176]
   movaps m5,m4
   movaps m6,m4
   movaps m7,m4
   mulps m4,[r1+704]
   mulps m5,[r1+720]
   mulps m6,[r1+736]
   mulps m7,[r1+752]
   addps m0,m4
   addps m1,m5
   addps m2,m6
   addps m3,m7
   add r0,192
   add r1,768
   sub r4d,48
   jnz .lloop
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
   movaps [r2],m6
   add r2,16
   sub r3,4
   jnz .nloop
   POP r5
   POP r3
   POP r2

   movss m7,[r5]
   shufps m7,m7,0
   xor r5,r5
.aloop:
   movaps m0,[r2+r5*4]
   movaps m1,[r2+r5*4+16]
   movaps m2,[r2+r5*4+32]
   movaps m3,[r2+r5*4+48]
   mulps m0,m7
   mulps m1,m7
   mulps m2,m7
   mulps m3,m7
   addps m0,[r1+r5*4]
   addps m1,[r1+r5*4+16]
   addps m2,[r1+r5*4+32]
   addps m3,[r1+r5*4+48]
   movaps [r2+r5*4],m0
   movaps [r2+r5*4+16],m1
   movaps [r2+r5*4+32],m2
   movaps [r2+r5*4+48],m3
   add r5,16
   sub r3,16
   jnz .aloop
   RET

