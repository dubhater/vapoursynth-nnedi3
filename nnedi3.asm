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

