/*
* Copyright (c) 2012-2013 Fredrik Mellbin
*
* This file is part of VapourSynth.
*
* VapourSynth is free software; you can redistribute it and/or
* modify it under the terms of the GNU Lesser General Public
* License as published by the Free Software Foundation; either
* version 2.1 of the License, or (at your option) any later version.
*
* VapourSynth is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
* Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public
* License along with VapourSynth; if not, write to the Free Software
* Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
*/

#include <stdint.h>
#include <string.h>

#include "cpufeatures.h"

#ifdef NNEDI3_X86
extern "C" {
extern void nnedi3_cpu_cpuid(uint32_t index, uint32_t *eax, uint32_t *ebx, uint32_t *ecx, uint32_t *edx);
extern void nnedi3_cpu_xgetbv(uint32_t op, uint32_t *eax, uint32_t *edx);
extern void nnedi3_cpu_cpuid_test(void);
}

void getCPUFeatures(CPUFeatures *cpuFeatures) {
    memset(cpuFeatures, 0, sizeof(CPUFeatures));

    uint32_t eax, ebx, ecx, edx;
    nnedi3_cpu_cpuid(1, &eax, &ebx, &ecx, &edx);
    cpuFeatures->can_run_vs = !!(edx & (1 << 26)); //sse2
    cpuFeatures->sse3 = !!(ecx & 1);
    cpuFeatures->ssse3 = !!(ecx & (1 << 9));
    cpuFeatures->sse4_1 = !!(ecx & (1 << 19));
    cpuFeatures->sse4_2 = !!(ecx & (1 << 20));
    cpuFeatures->fma3 = !!(ecx & (1 << 12));
    if ((ecx & (1 << 27)) && (ecx & (1 << 28))) {
        nnedi3_cpu_xgetbv(0, &eax, &edx);
        cpuFeatures->avx = ((eax & 0x6) == 0x6);
        if (cpuFeatures->avx) {
            eax = 0;
            ebx = 0;
            ecx = 0;
            edx = 0;
            nnedi3_cpu_cpuid(7, &eax, &ebx, &ecx, &edx);
            cpuFeatures->avx2 = !!(ebx & (1 << 5));
        }
    }

    nnedi3_cpu_cpuid( 0x80000000, &eax, &ebx, &ecx, &edx );

    if (eax >= 0x80000001) {
        nnedi3_cpu_cpuid(0x80000001, &eax, &ebx, &ecx, &edx);

        if (cpuFeatures->avx)
            cpuFeatures->fma4 = !!(ecx & 0x00010000);
    }

}
#else
#include <sys/auxv.h>

void getCPUFeatures(CPUFeatures *cpuFeatures) {
    memset(cpuFeatures, 0, sizeof(CPUFeatures));

    unsigned long long hwcap = getauxval(AT_HWCAP);

    cpuFeatures->can_run_vs = 1;

#if defined(NNEDI3_ARM)
    cpuFeatures->half_fp = !!(hwcap & HWCAP_ARM_HALF);
    cpuFeatures->edsp = !!(hwcap & HWCAP_ARM_EDSP);
    cpuFeatures->iwmmxt = !!(hwcap & HWCAP_ARM_IWMMXT);
    cpuFeatures->neon = !!(hwcap & HWCAP_ARM_NEON);
    cpuFeatures->fast_mult = !!(hwcap & HWCAP_ARM_FAST_MULT);
    cpuFeatures->idiv_a = !!(hwcap & HWCAP_ARM_IDIVA);
#elif defined(NNEDI3_PPC)
    cpuFeatures->altivec = !!(hwcap & PPC_FEATURE_HAS_ALTIVEC);
    cpuFeatures->spe = !!(hwcap & PPC_FEATURE_HAS_SPE);
    cpuFeatures->efp_single = !!(hwcap & PPC_FEATURE_HAS_EFP_SINGLE);
    cpuFeatures->efp_double = !!(hwcap & PPC_FEATURE_HAS_EFP_DOUBLE);
    cpuFeatures->dfp = !!(hwcap & PPC_FEATURE_HAS_DFP);
    cpuFeatures->vsx = !!(hwcap & PPC_FEATURE_HAS_VSX);
#endif
}
#endif
