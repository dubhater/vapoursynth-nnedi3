AM_CFLAGS = -O2 -msse2 -mfpmath=sse -Wall -Wextra -Wno-unused-parameter

AM_CPPFLAGS = $(VapourSynth_CFLAGS)

lib_LTLIBRARIES = libnnedi3.la

yasm_verbose = $(yasm_verbose_$(V))
yasm_verbose_ = $(yasm_verbose_$(AM_DEFAULT_VERBOSITY))
yasm_verbose_0 = @echo "  YASM    " $@;

.asm.lo:
	$(yasm_verbose)$(LIBTOOL) $(AM_V_lt) --mode=compile --tag=CC $(AS) $(ASFLAGS) -o $@ $< -prefer-non-pic

libnnedi3_la_SOURCES = src/nnedi3.c src/asm/nnedi3.asm src/asm/binary1.asm

libnnedi3_la_LDFLAGS = -no-undefined -avoid-version $(PLUGINLDFLAGS)
