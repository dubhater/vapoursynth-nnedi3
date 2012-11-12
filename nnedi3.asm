%include "x86inc.asm"

SECTION_RODATA
sign_bits_f times 4 dd 0x7FFFFFFF
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
   mul r2
   pshuflw m1,m5,14
   cvtsi2ss m7,r3
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


