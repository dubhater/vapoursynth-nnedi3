Description
===========

nnedi3 filter for VapourSynth.

This is a port of tritical's nnedi3 filter.


Usage
=====

::

   nnedi3.nnedi3(clip clip, int field[, bint dh=False, bint Y=True, bint U=True, bint V=True, int nsize=6, int nns=1, int qual=1, int etype=0, int pscrn=2, int opt=2, int fapprox=15])

Allowed values (ranges are inclusive):

- field: 0..3
- nsize: 0..6
- nns: 0..4
- qual: 1..2
- etype: 0..1
- pscrn: 0..4
- opt: 1..2
- fapprox: 0..15


Compilation
===========

::

   ./autogen.sh
   ./configure
   make

objcopy from binutils is required.
yasm is currently not optional.

A win32 dll can be found here: http://uloz.to/xSuqyUw/nnedi3-dll. sha256sum: 45e712f038d23718b912b69fa69a8d603329a02bb19701f06873c2e84ef52dad


Things to do
============

- Support more than 8 bits/sample.
- The output is slightly different from the original filter's output (with opt=1). Some pixels are off by one.
