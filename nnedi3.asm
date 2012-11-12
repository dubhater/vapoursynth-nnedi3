%include "x86inc.asm"


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
