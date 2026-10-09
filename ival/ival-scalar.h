/* ival-scalar.h: ival's interval arithmetic on one interval at a time, inline (2026-10-09). For callers that compute
   interval by interval, such as a C++ interval class (IBEX's backends), where a library call per operation would cost
   more than the operation itself. Each result is the tightest enclosure, bit for bit what ival.h's array functions
   give (ival/test/arith-check.c compares them on every input it has); the conventions are ival.h's: [NaN, NaN] is
   empty, zero ends come out +0, 0 * inf at an end is 0, division by 1788's table.
     - The caller must round to nearest, with flush-to-zero and denormals-are-zero off. Each function checks that
       (x86-64: MXCSR; aarch64: FPCR; reading either is cheap) and otherwise calls the library, which works in any
       mode. Elsewhere it always calls the library.
     - Flags are raised as ordinary arithmetic raises them (inexact, overflow, ...); the library's calls, unlike
       these, give the caller's flags back.
     - Products, quotients and roots near the underflow range, where the exact residuals below are not exact, go to
       the library too.
   Each bound comes from the result rounded to nearest and an error-free transformation of it: a + b = s + e exactly
   (TwoSum), a * b = p + e and a = q * b + r and x = s * s + e exactly (fma: inline where the compiler targets an fma
   instruction, as on aarch64 or x86-64 with -mfma or -march, a call to the C library's fma otherwise). The sign of the
   residual says on which side of the rounded result the exact one lies; a bound steps one double that way if it must. */
#ifndef IVAL_SCALAR_H
#define IVAL_SCALAR_H
#include <float.h>   /* DBL_MAX; the constants below are written in decimal, exact, for C99 and C++11 */
#include <math.h>
#include <stdint.h>
#include <string.h>
#include "ival.h"
#if defined(__x86_64__)
#include <immintrin.h>
#endif
#ifdef __cplusplus
extern "C" {
#endif

/* ---- helpers, not API ---- */
static inline int ival1__ready(void)   /* to nearest, flush modes off */
{
#if defined(__x86_64__) && defined(__GNUC__)
  /* from arithmetic, not MXCSR, whose read (stmxcsr) was more than half of an add's cost on Zen 3. With t = 2^-60,
     1 + t and 1 - t are both 1 only to nearest (up moves the first, down and toward zero the second); d + d, with
     d = 2^-1074, is 0 only with denormals-are-zero (d read as 0) or flush-to-zero (2^-1073 flushed). The asm hides
     t and d from constant folding, so the operations run in the caller's mode. */
  double t = 8.6736173798840355e-19, d = 4.9406564584124654e-324;   /* 2^-60, 2^-1074 */
  __asm__("" : "+x"(t), "+x"(d));
  return (1.0 + t == 1.0) & (1.0 - t == 1.0) & (d + d != 0);
#elif defined(__x86_64__)
  return (_mm_getcsr() & 0xe040u) == 0;
#elif defined(__aarch64__)
  uint64_t r;
  __asm__ volatile("mrs %0, fpcr" : "=r"(r));
  return (r & 0x1c00000ull) == 0;
#else
  return 0;
#endif
}
static inline int ival1__empty(double lo, double hi) { return !(lo <= hi) || lo == INFINITY || hi == -INFINITY; }
static inline double ival1__canon(double x) { return x == 0 ? 0.0 : x; }
static inline double ival1__step(double x, int dir)   /* x moved |dir| <= 1 doubles; x finite, nonzero when dir != 0 */
{
  uint64_t b;
  memcpy(&b, &x, 8);
  b += (uint64_t)(int64_t)(x > 0 ? dir : -dir);
  memcpy(&x, &b, 8);
  return x;
}
static inline double ival1__fma(double a, double b, double c) { return __builtin_fma(a, b, c); }
/* a + b rounded down (up = 0) or up; an overflow rounds to DBL_MAX one way */
static inline double ival1__add_r(double a, double b, int up)
{
  double s = a + b;
  if (isinf(s)) {
    if (isinf(a) || isinf(b)) return s;
    return up ? (s > 0 ? s : -DBL_MAX) : (s > 0 ? DBL_MAX : s);
  }
  double bb = s - a, e = (a - (s - bb)) + (b - bb);
  return ival1__step(s, up ? (e > 0) : -(e < 0));
}
/* a * b rounded down (*d) and up (*u), 0 * inf = 0; 1 when the product is near underflow (the library decides) */
static inline int ival1__mul2(double a, double b, double *d, double *u)
{
  if (a == 0 || b == 0) { *d = *u = 0.0; return 0; }
  double p = a * b;
  if (isinf(p)) {
    if (isinf(a) || isinf(b)) { *d = *u = p; return 0; }
    *u = p > 0 ? p : -DBL_MAX; *d = p > 0 ? DBL_MAX : p;
    return 0;
  }
  if (!(fabs(p) >= 2.004168360008973e-292)) return 1;   /* 2^-969 */
  double e = ival1__fma(a, b, -p);
  *d = ival1__step(p, -(e < 0));
  *u = ival1__step(p, e > 0);
  return 0;
}
/* a / b rounded down or up into *r, b != 0; 1 near underflow or overflow of the remainder (the library decides) */
static inline int ival1__div_r(double a, double b, int up, double *r)
{
  if (a == 0 || isinf(b)) { *r = 0.0; return 0; }
  double q = a / b;
  if (isinf(a)) { *r = q; return 0; }
  if (isinf(q)) { *r = up ? (q > 0 ? q : -DBL_MAX) : (q > 0 ? DBL_MAX : q); return 0; }
  if (!(fabs(a) >= 1.0261342003245941e-289 && fabs(b) >= 1.0261342003245941e-289 && fabs(q) >= 1.0261342003245941e-289 &&
        fabs(b) <= 9.7453140114e+288))   /* 2^-960, 2^960 */
    return 1;
  double rem = ival1__fma(-q, b, a);
  if (rem == 0) { *r = q; return 0; }
  int above = (rem > 0) == (b > 0);
  *r = ival1__step(q, up ? above : -!above);
  return 0;
}
/* A / B by 1788's table, as ival-arith.c's div1; 1 if a quotient needs the library */
static inline int ival1__div1(double al, double ah, double bl, double bh, double *zl, double *zh)
{
  const double I = INFINITY;
  int f = 0;
  if (bl == 0 && bh == 0) { *zl = *zh = NAN; return 0; }
  if (al == 0 && ah == 0) { *zl = *zh = 0.0; return 0; }
  if (bl > 0) {
    if (al >= 0) { f |= ival1__div_r(al, bh, 0, zl); f |= ival1__div_r(ah, bl, 1, zh); }
    else if (ah <= 0) { f |= ival1__div_r(al, bl, 0, zl); f |= ival1__div_r(ah, bh, 1, zh); }
    else { f |= ival1__div_r(al, bl, 0, zl); f |= ival1__div_r(ah, bl, 1, zh); }
  } else if (bh < 0) {
    if (al >= 0) { f |= ival1__div_r(ah, bh, 0, zl); f |= ival1__div_r(al, bl, 1, zh); }
    else if (ah <= 0) { f |= ival1__div_r(ah, bl, 0, zl); f |= ival1__div_r(al, bh, 1, zh); }
    else { f |= ival1__div_r(ah, bh, 0, zl); f |= ival1__div_r(al, bh, 1, zh); }
  } else if (bl == 0) {
    if (ah < 0) { *zl = -I; f |= ival1__div_r(ah, bh, 1, zh); }
    else if (al > 0) { f |= ival1__div_r(al, bh, 0, zl); *zh = I; }
    else if (ah == 0) { *zl = -I; *zh = 0.0; }
    else if (al == 0) { *zl = 0.0; *zh = I; }
    else { *zl = -I; *zh = I; }
  } else if (bh == 0) {
    if (ah < 0) { f |= ival1__div_r(ah, bl, 0, zl); *zh = I; }
    else if (al > 0) { *zl = -I; f |= ival1__div_r(al, bl, 1, zh); }
    else if (ah == 0) { *zl = 0.0; *zh = I; }
    else if (al == 0) { *zl = -I; *zh = 0.0; }
    else { *zl = -I; *zh = I; }
  } else { *zl = -I; *zh = I; }
  return f;
}

/* ---- the operations: Z = A op B (or op A), one interval ---- */
static inline void ival1_add(double al, double ah, double bl, double bh, double *zl, double *zh)
{
  if (!ival1__ready()) { ival_add(&al, &ah, &bl, &bh, zl, zh, 1); return; }
  if (ival1__empty(al, ah) || ival1__empty(bl, bh)) { *zl = *zh = NAN; return; }
  *zl = ival1__canon(ival1__add_r(al, bl, 0)); *zh = ival1__canon(ival1__add_r(ah, bh, 1));
}
static inline void ival1_sub(double al, double ah, double bl, double bh, double *zl, double *zh)
{
  if (!ival1__ready()) { ival_sub(&al, &ah, &bl, &bh, zl, zh, 1); return; }
  if (ival1__empty(al, ah) || ival1__empty(bl, bh)) { *zl = *zh = NAN; return; }
  *zl = ival1__canon(ival1__add_r(al, -bh, 0)); *zh = ival1__canon(ival1__add_r(ah, -bl, 1));
}
static inline void ival1_neg(double al, double ah, double *zl, double *zh)
{
  /* exact, but its tests compare, and denormals-are-zero would read a subnormal end as 0 (arith-check found it) */
  if (!ival1__ready()) { ival_neg(&al, &ah, zl, zh, 1); return; }
  if (ival1__empty(al, ah)) { *zl = *zh = NAN; return; }
  *zl = ival1__canon(-ah); *zh = ival1__canon(-al);
}
static inline void ival1_mul(double al, double ah, double bl, double bh, double *zl, double *zh)
{
  if (!ival1__ready()) { ival_mul(&al, &ah, &bl, &bh, zl, zh, 1); return; }
  if (ival1__empty(al, ah) || ival1__empty(bl, bh)) { *zl = *zh = NAN; return; }
  double d0 = 0, u0 = 0, d1 = 0, u1 = 0, d2 = 0, u2 = 0, d3 = 0, u3 = 0;   /* set before use; 0 quiets -Wmaybe-uninitialized */
  if (al >= 0 && bl >= 0) {   /* both nonnegative, the common case: two corners */
    if (ival1__mul2(al, bl, &d0, &u0) | ival1__mul2(ah, bh, &d1, &u1)) { ival_mul(&al, &ah, &bl, &bh, zl, zh, 1); return; }
    *zl = ival1__canon(d0); *zh = ival1__canon(u1);
    return;
  }
  if (ival1__mul2(al, bl, &d0, &u0) | ival1__mul2(al, bh, &d1, &u1) | ival1__mul2(ah, bl, &d2, &u2) |
      ival1__mul2(ah, bh, &d3, &u3)) {
    ival_mul(&al, &ah, &bl, &bh, zl, zh, 1);
    return;
  }
  double x = d0 < d1 ? d0 : d1, y = d2 < d3 ? d2 : d3, s = u0 > u1 ? u0 : u1, t = u2 > u3 ? u2 : u3;
  *zl = ival1__canon(x < y ? x : y); *zh = ival1__canon(s > t ? s : t);
}
static inline void ival1_sqr(double al, double ah, double *zl, double *zh)
{
  if (!ival1__ready()) { ival_sqr(&al, &ah, zl, zh, 1); return; }
  if (ival1__empty(al, ah)) { *zl = *zh = NAN; return; }
  double d0 = 0, u0 = 0, d1 = 0, u1 = 0;
  if (ival1__mul2(al, al, &d0, &u0) | ival1__mul2(ah, ah, &d1, &u1)) { ival_sqr(&al, &ah, zl, zh, 1); return; }
  if (al >= 0) { *zl = d0; *zh = u1; }
  else if (ah <= 0) { *zl = d1; *zh = u0; }
  else { *zl = 0.0; *zh = u0 > u1 ? u0 : u1; }
  *zl = ival1__canon(*zl); *zh = ival1__canon(*zh);
}
static inline void ival1_div(double al, double ah, double bl, double bh, double *zl, double *zh)
{
  if (!ival1__ready()) { ival_div(&al, &ah, &bl, &bh, zl, zh, 1); return; }
  if (ival1__empty(al, ah) || ival1__empty(bl, bh)) { *zl = *zh = NAN; return; }
  double l, h;
  if (ival1__div1(al, ah, bl, bh, &l, &h)) { ival_div(&al, &ah, &bl, &bh, zl, zh, 1); return; }
  *zl = ival1__canon(l); *zh = ival1__canon(h);
}
static inline void ival1_recip(double al, double ah, double *zl, double *zh)
{
  if (!ival1__ready()) { ival_recip(&al, &ah, zl, zh, 1); return; }
  if (ival1__empty(al, ah)) { *zl = *zh = NAN; return; }
  double l, h;
  if (ival1__div1(1.0, 1.0, al, ah, &l, &h)) { ival_recip(&al, &ah, zl, zh, 1); return; }
  *zl = ival1__canon(l); *zh = ival1__canon(h);
}
static inline void ival1_sqrt(double al, double ah, double *zl, double *zh)
{
  if (!ival1__ready()) { ival_sqrt(&al, &ah, zl, zh, 1); return; }
  double a = al > 0 ? al : 0.0;
  if (al != al || !(a <= ah)) { *zl = *zh = NAN; return; }   /* nothing of [0, inf] left, or a NaN end */
  double r[2], x[2] = { a, ah };
  for (int k = 0; k < 2; k++) {
    double s = sqrt(x[k]);
    if (x[k] == 0 || isinf(x[k])) { r[k] = s; continue; }
    if (!(x[k] >= 1.1830521861667747e-271)) {   /* 2^-900 */
      ival_sqrt(&al, &ah, zl, zh, 1);
      return;
    }
    double e = ival1__fma(-s, s, x[k]);   /* x - s^2: positive when the root is above s */
    r[k] = ival1__step(s, k ? (e > 0) : -(e < 0));
  }
  *zl = ival1__canon(r[0]); *zh = ival1__canon(r[1]);
}

#ifdef __cplusplus
}
#endif
#endif
