%include "include/x86inc.asm"


SECTION_RODATA

global mangle(binary1)
mangle(binary1) incbin "../binary1_0.9.4.bin"
