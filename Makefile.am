AM_CFLAGS = -Wall -Wextra -Wno-unused-parameter

AM_CPPFLAGS = $(VapourSynth_CFLAGS)

lib_LTLIBRARIES = libnnedi3.la

BUILT_SOURCES = src/asm/nnedi3.o src/asm/binary1.o
CLEANFILES = $(BUILT_SOURCES)

AM_V_YASM = $(am__v_YASM_$(V))
am__v_YASM_ = $(am__v_YASM_$(AM_DEFAULT_VERBOSITY))
am__v_YASM_0 = @echo "  YASM    " $@;
am__v_YASM_1 = 

.asm.o:
	$(AM_V_YASM)$(AS) $(ASFLAGS) -o $@ $<

libnnedi3_la_SOURCES = src/nnedi3.c src/asm/nnedi3.asm src/asm/binary1.asm

libnnedi3_la_LDFLAGS = -no-undefined -avoid-version  -lm  \
	-Xlinker src/asm/nnedi3.o -Xlinker src/asm/binary1.o

libnnedi3_la_DEPENDENCIES = src/asm/nnedi3.o src/asm/binary1.o
