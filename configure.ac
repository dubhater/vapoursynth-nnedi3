AC_INIT([nnedi3], [3], [https://github.com/dubhater/vapoursynth-nnedi3/issues], [nnedi3], [https://github.com/dubhater/vapoursynth-nnedi3/])

: ${CXXFLAGS=""}

AM_INIT_AUTOMAKE([foreign no-dist-gzip dist-xz subdir-objects no-define])
AM_SILENT_RULES([yes])

LT_INIT([disable-static win32-dll])

AC_CANONICAL_HOST

AC_PROG_CXX

AC_SEARCH_LIBS([sqrt], [m], [], [
   AC_MSG_ERROR([unable to find the sqrt() function])
])


X86="false"
PPC="false"
ARM="false"

AS_CASE(
        [$host_cpu],
        [i?86], [BITS="32" ASFLAGS="$ASFLAGS -DARCH_X86_64=0" X86="true"],
        [x86_64], [BITS="64" ASFLAGS="$ASFLAGS -DARCH_X86_64=1 -DPIC -m amd64" X86="true"],
        [powerpc*], [PPC="true"],
        [arm*], [ARM="true"] # Maybe doesn't work for all arm systems?
)

AS_CASE(
        [$host_os],
        [darwin*],
        [
         ASFLAGS="$ASFLAGS -f macho$BITS -DPREFIX"
        ],
        [*linux*|gnu*|dragonfly*|*bsd*], # The BSDs are close enough, right?
        [
         ASFLAGS="$ASFLAGS -f elf"
        ],
        [cygwin*|mingw*],
        [
         ASFLAGS="$ASFLAGS -f win32"
         AS_IF(
               [test "x$BITS" = "x32"],
               [
                ASFLAGS="$ASFLAGS -DPREFIX"
                AC_SUBST([PLUGINLDFLAGS], ["-Wl,--kill-at"])
               ]
         )
        ],
        [AC_MSG_ERROR(["Unknown host OS: $host_os"])]
)

AS_IF(
      [test "x$X86" = "xtrue"],
      [
       AC_DEFINE([NNEDI3_X86])

       ASFLAGS="$ASFLAGS -Dprivate_prefix=nnedi3"

       AC_SUBST([MFLAGS], ["-mfpmath=sse -msse2"])

       AC_CHECK_PROGS([YASM], [yasm])
       AS_IF(
             [test "x$YASM" = "x"],
             [AC_MSG_ERROR([yasm required but not found])],
             [AS="$YASM"]
       )
      ]
)

AS_IF(
      [test "x$PPC" = "xtrue"],
      [AC_DEFINE([NNEDI3_PPC])]
)

AS_IF(
      [test "x$ARM" = "xtrue"],
      [AC_DEFINE([NNEDI3_ARM])]
)

AC_SUBST([ASFLAGS])


AM_CONDITIONAL([NNEDI3_X86], [test "x$X86" = "xtrue"])
AM_CONDITIONAL([NNEDI3_ARM], [test "x$ARM" = "xtrue"])
AM_CONDITIONAL([NNEDI3_PPC], [test "x$PPC" = "xtrue"])


PKG_CHECK_MODULES([VapourSynth], [vapoursynth])

AC_CONFIG_FILES([Makefile])
AC_OUTPUT
