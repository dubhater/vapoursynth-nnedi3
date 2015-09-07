Description
===========

nnedi3 filter for VapourSynth.

This is a port of tritical's nnedi3 filter.


Usage
=====

The file ``nnedi3_weights.bin`` is required. In Windows, it must be located in the same folder as ``libnnedi3.dll``. Everywhere else it can be located either in the same folder as ``libnnedi3.so``/``libnnedi3.dylib``, or in ``$prefix/share/nnedi3/``. The build system installs it at the latter location automatically.

::

   nnedi3.nnedi3(clip clip, int field[, bint dh=False, int[] planes=[0, 1, 2], bint Y=True, bint U=True, bint V=True, int nsize=6, int nns=1, int qual=1, int etype=0, int pscrn=2, bint opt=True, int fapprox=15])

Allowed values (ranges are inclusive):

- field: 0..3
- nsize: 0..6
- nns: 0..4
- qual: 1..2
- etype: 0..1
- pscrn: 0..4
- fapprox: 0..15

When the input clip has more than 8 bits per sample, some parameters' allowed and default values change:

- pscrn: 0, 1, default 1
- fapprox: 0, 4, 8, 12, default 12

The opt parameter is now a boolean. If False, only C functions will be used. If True, the best functions that can run on your CPU will be selected automatically.

The ``_FieldBased`` frame property is now used to determine each frame's field dominance. The *field* parameter is only a fallback for frames that don't have the ``_FieldBased`` property, or where said property indicates that the frame is progressive.

The Y, U, and V parameters are deprecated. Use the planes parameter instead.

This plugin no longer provides the nnedi3_rpow2 filter.


Compilation
===========

::

   ./autogen.sh
   ./configure
   make

On x86, yasm is currently not optional.

DLLs can be found in the "releases" section.


License
=======

GPLv2.
