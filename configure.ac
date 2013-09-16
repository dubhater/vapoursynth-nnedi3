AC_INIT([nnedi3], [1], [https://github.com/dubhater/vapoursynth-nnedi3/issues], [nnedi3], [https://github.com/dubhater/vapoursynth-nnedi3/])

: ${CFLAGS="-O2"}

AM_INIT_AUTOMAKE([foreign no-dist-gzip dist-xz subdir-objects no-define])
AM_SILENT_RULES([yes])

LT_INIT([disable-static win32-dll])

AC_CANONICAL_HOST

AC_PROG_CC_C99

AC_CHECK_PROGS([YASM], [yasm])
AS_IF(
   [test "x$YASM" = "x"],
   [AC_MSG_ERROR([yasm required but not found])],
   [AS="$YASM"]
)

AC_CHECK_TOOL([OBJCOPY], [objcopy])
AS_IF(
   [test "x$OBJCOPY" = "x"],
   [AC_MSG_ERROR([objcopy required but not found])],
)

AS_CASE(
  [$host_cpu],
  [i?86],   [BITS="32" ASFLAGS="$ASFLAGS -DARCH_X86_64=0"       BFDARCH="i386"],
  [x86_64], [BITS="64" ASFLAGS="$ASFLAGS -DARCH_X86_64=1 -DPIC" BFDARCH="x86-64"],
  [AC_MSG_ERROR([Currently nnedi3 can only be compiled for Intel CPUs.])]
)

OBJCOPY_ARGS="--input-target=binary --binary-architecture=i386"

AS_CASE(
   [$host_os],
   [darwin*],
   [
      ASFLAGS="$ASFLAGS -f macho$BITS -DPREFIX"
      BFDNAME="mach-o-$BFDARCH"
   ],
   [*linux*],
   [
      ASFLAGS="$ASFLAGS -f elf$BITS"
      BFDNAME="elf$BITS-$BFDARCH"
   ],
   [gnu*],
   [
      ASFLAGS="$ASFLAGS -f elf$BITS"
      BFDNAME="elf$BITS-$BFDARCH"
   ],
   [cygwin*],
   [
      ASFLAGS="$ASFLAGS -f win$BITS"
      BFDNAME="pe-$BFDARCH"
      AS_IF(
         [test "x$BITS" = "x32"],
         [ASFLAGS="$ASFLAGS -DPREFIX" OBJCOPY_ARGS="$OBJCOPY_ARGS --prefix-symbols=_"]
      )
   ],
   [mingw*],
   [
      ASFLAGS="$ASFLAGS -f win$BITS"
      BFDNAME="pe-$BFDARCH"
      AS_IF(
         [test "x$BITS" = "x32"],
         [ASFLAGS="$ASFLAGS -DPREFIX" OBJCOPY_ARGS="$OBJCOPY_ARGS --prefix-symbols=_"]
      )
   ],
   [AC_MSG_ERROR([Unknown host OS])]
)
OBJCOPY_ARGS="$OBJCOPY_ARGS --output-target=$BFDNAME"
AC_SUBST([ASFLAGS])
AC_SUBST([OBJCOPY_ARGS])

PKG_CHECK_MODULES([VapourSynth], [vapoursynth])

AC_CONFIG_FILES([Makefile])
AC_OUTPUT
