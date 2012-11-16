Description
===========

nnedi3 filter for VapourSynth.

This is a port of tritical's nnedi3 filter.


Usage
=====

::

   nnedi3.nnedi3(clip clip, int field[, bint dh=False, bint Y=True, bint U=True, bint V=True, int nsize=6, int nns=1, int qual=1, int etype=0, int pscrn=2, int opt=1, int fapprox=15])

Allowed values (ranges are inclusive):

- field: 1..3
- nsize: 0..6
- nns: 0..4
- qual: 1..2
- etype: 0..1
- pscrn: 0..4
- opt: 1..2
- fapprox: 0..15

"opt" defaults to 1 (C routines only) because not all asm functions are ported yet.


Compilation
===========

To compile the filter in 64 bit Linux (and possibly other Unix-like systems)::

   yasm -f elf64 -o nnedi3-asm.elf64.o -DARCH_X86_64=1 -DPIC nnedi3.asm
   objcopy --input binary --output elf64-x86-64 --binary-architecture i386 binary1_0.9.4.bin binary1_0.9.4.elf64.o
   clang -O3 -Wall -Wextra -Wno-unused-parameter -shared -fPIC -o libnnedi3.so nnedi3.c binary1_0.9.4.elf64.o nnedi3-asm.elf64.o

A win32 dll can be found in the Downloads section.


Things to do
============

- The output is slightly different from the original filter's output (with opt=1). Some pixels are off by one.

- The asm probably should be converted to work with yasm and x264's x86inc.asm
