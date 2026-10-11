/* Included (-include) by the Makefile for core-math/bf16/*.c only: GCC before 14
   on AArch64 with the fp16 extension (-march=native on most current cores, or
   +fp16) loads a __bf16 constant with FMOV's half-precision immediate, which
   encodes binary16, so 1.0 becomes 0x3c00 (0.0078125 as bfloat16). GCC PR111867,
   fixed in GCC 14 (r14-6421) and never backported to 13. Built that way, 17 of
   these functions were wrong (cos(0) = 0.0078125; cospi on 124,424 of 262,144
   inputs and modes), on cfarm424 with Debian's gcc 13.3.0, 2026-10-11. Without
   fp16 for these files, GCC 13 loads the constant from memory, as GCC 14 does. */
#if defined(__aarch64__) && defined(__GNUC__) && !defined(__clang__) && __GNUC__ < 14 \
    && defined(__ARM_FEATURE_FP16_SCALAR_ARITHMETIC)
#pragma GCC target("+nofp16")
#endif
