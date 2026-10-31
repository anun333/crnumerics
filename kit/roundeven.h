/* kit/roundeven.h: an exact roundeven for C libraries that have none (musl; glibc before 2.25), added 2026-10-31.
   CORE-MATH calls __builtin_roundeven, which gcc lowers to a library call where the processor has no instruction for
   it (riscv64 without Zfa, x86-64 without SSE4.1), and on musl that call does not link (cfarm94). The Makefile then
   includes this header in every file, so the builtin becomes the function below, inline. Integer operations on the
   bits, as crmvec's crmvec-roundeven.c (checked there against glibc's): C23's roundeven in every rounding mode, and no
   exception but invalid for a signalling NaN. kit/test/roundeven-check.c checks it. */
#ifndef KIT_ROUNDEVEN_H
#define KIT_ROUNDEVEN_H
#include <stdint.h>
#include <string.h>

static inline double kit_roundeven(double x)
{
  uint64_t u; memcpy(&u, &x, 8);
  int e = (int)((u >> 52) & 0x7ff) - 1023;
  if (e >= 52) return e == 1024 ? x + x : x;   /* already an integer, or Inf or NaN (x + x quiets a signalling NaN) */
  if (e < 0) {                                  /* |x| < 1: +-1 above one half, +-0 up to it */
    uint64_t a = u & 0x7fffffffffffffffULL;
    u &= 0x8000000000000000ULL;
    if (a > 0x3fe0000000000000ULL) u |= 0x3ff0000000000000ULL;
  } else {                                      /* the bit worth 1 is bit 52 - e; below it, the fraction */
    uint64_t one = 1ULL << (52 - e), f = u & (one - 1);
    u -= f;
    if (f > one >> 1 || (f == one >> 1 && (u & one))) u += one;   /* a carry into the exponent is right */
  }
  memcpy(&x, &u, 8); return x;
}

#define __builtin_roundeven kit_roundeven
#endif
