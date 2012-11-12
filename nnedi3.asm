%include "x86inc.asm"

SECTION_RODATA
sign_bits_f times 4 dd 0x7FFFFFFF
ones_f      times 4 dd 1.0


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

