/* ival-scalar.h: ival's interval arithmetic on one interval at a time, inline (2026-10-09). For callers that compute
   interval by interval, such as a C++ interval class (IBEX's backends), where a library call per operation would cost
   more than the operation itself. Each result is the tightest enclosure, bit for bit what ival.h's array functions
   give (ival/test/arith-check.c compares them on every input it has); the conventions are ival.h's: [NaN, NaN] is
   empty, zero ends come out +0, 0 * inf at an end is 0, division by 1788's table.
     - Two rounding modes run inline. Rounding upward, as Gaol-style interval classes keep it, each upper bound is
       the operation itself and each lower bound the negation of one on negated operands, -((-a) - b): one rounding
       each, which is the tight bound. To nearest, each bound comes from the result to nearest and an error-free
       transformation of it: a + b = s + e (TwoSum), a * b = p + e, a = q * b + r and x = s * s + e (fma: inline
       where the compiler targets an fma instruction, a call to the C library's fma otherwise); the residual's sign
       says on which side of s the exact value lies, and the bound steps one double that way if it must.
     - Any other mode, or flush-to-zero or denormals-are-zero on, and products, quotients and roots near the
       underflow range to nearest (where the residuals are not exact): the library call, which works in any mode.
       Off x86-64 and aarch64, always the library.
     - Flags are raised as ordinary arithmetic raises them (inexact, overflow, ...); the library's calls, unlike
       these, give the caller's flags back. */
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
/* 0: to nearest; 1: upward; -1: another mode, or a flush mode on (the library then) */
static inline int ival1__mode(void)
{
#if defined(__x86_64__) && defined(__GNUC__)
  /* from arithmetic, not MXCSR, whose read (stmxcsr) was more than half of an add's cost on Zen 3. With t = 2^-60,
     1 - t is 1 only to nearest and upward, and 1 + t is 1 only to nearest; d + d, with d = 2^-1074, is 0 only with
     denormals-are-zero (d read as 0) or flush-to-zero (2^-1073 flushed). The asm hides t and d from constant
     folding, so the operations run in the caller's mode; it is volatile so that a compiler without
     -frounding-math, to which these operations are pure, cannot reuse one check across a change of mode or move
     it out of a loop (ival/test/mode-change-check.c; about 3% on a scale-and-add chain, 2026-10-10). */
  double t = 8.6736173798840355e-19, d = 4.9406564584124654e-324;   /* 2^-60, 2^-1074 */
  __asm__ volatile("" : "+x"(t), "+x"(d));
#if IVAL_PLANT_ARITH == 47   /* 47: the answer of a check made upward and reused */
  if (t == t) return 1;
#endif
  if (!(d + d != 0) || 1.0 - t != 1.0) return -1;
  return 1.0 + t == 1.0 ? 0 : 1;
#elif defined(__x86_64__)
  unsigned c = _mm_getcsr();
  if (c & 0x8040u) return -1;
  c &= 0x6000u;
  return c == 0 ? 0 : c == 0x4000u ? 1 : -1;
#elif defined(__aarch64__)
  uint64_t r;
  __asm__ volatile("mrs %0, fpcr" : "=r"(r));
  if (r & 0x1000000ull) return -1;
  r &= 0xc00000ull;
  return r == 0 ? 0 : r == 0x400000ull ? 1 : -1;
#else
  return -1;
#endif
}
/* x, hidden from the compiler: -((-a) - b) must not be folded into a + b, which is the same only to nearest */
static inline double ival1__opaque(double x)
{
#if defined(__GNUC__) && defined(__x86_64__)
  __asm__("" : "+x"(x));
#elif defined(__GNUC__) && defined(__aarch64__)
  __asm__("" : "+w"(x));
#else
  volatile double v = x;
  x = v;
#endif
  return x;
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
/* a + b rounded down (up = 0) or up; an overflow rounds to DBL_MAX one way. m: the mode, 0 or 1 */
static inline double ival1__add_r(double a, double b, int up, int m)
{
  if (m) return up ? a + b : -ival1__opaque(ival1__opaque(-a) - b);   /* upward: the hardware's rounding */
  double s = a + b;
  if (isinf(s)) {
    if (isinf(a) || isinf(b)) return s;
    return up ? (s > 0 ? s : -DBL_MAX) : (s > 0 ? DBL_MAX : s);
  }
  double bb = s - a, e = (a - (s - bb)) + (b - bb);
  return ival1__step(s, up ? (e > 0) : -(e < 0));
}
/* a * b rounded down (*d) and up (*u), 0 * inf = 0; 1 when the product is near underflow (the library decides) */
static inline int ival1__mul2(double a, double b, double *d, double *u, int m)
{
  if (a == 0 || b == 0) { *d = *u = 0.0; return 0; }
  if (m) { *u = a * b; *d = -ival1__opaque(ival1__opaque(-a) * b); return 0; }   /* upward: exact near underflow too */
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
static inline int ival1__div_r(double a, double b, int up, double *r, int m)
{
  if (a == 0 || isinf(b)) { *r = 0.0; return 0; }
  if (m) { *r = up || isinf(a) ? a / b : -ival1__opaque(ival1__opaque(-a) / b); return 0; }
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
static inline int ival1__div1(double al, double ah, double bl, double bh, double *zl, double *zh, int m)
{
  const double I = INFINITY;
  int f = 0;
  if (bl == 0 && bh == 0) { *zl = *zh = NAN; return 0; }
  if (al == 0 && ah == 0) { *zl = *zh = 0.0; return 0; }
  if (bl > 0) {
    if (al >= 0) { f |= ival1__div_r(al, bh, 0, zl, m); f |= ival1__div_r(ah, bl, 1, zh, m); }
    else if (ah <= 0) { f |= ival1__div_r(al, bl, 0, zl, m); f |= ival1__div_r(ah, bh, 1, zh, m); }
    else { f |= ival1__div_r(al, bl, 0, zl, m); f |= ival1__div_r(ah, bl, 1, zh, m); }
  } else if (bh < 0) {
    if (al >= 0) { f |= ival1__div_r(ah, bh, 0, zl, m); f |= ival1__div_r(al, bl, 1, zh, m); }
    else if (ah <= 0) { f |= ival1__div_r(ah, bl, 0, zl, m); f |= ival1__div_r(al, bh, 1, zh, m); }
    else { f |= ival1__div_r(ah, bh, 0, zl, m); f |= ival1__div_r(al, bh, 1, zh, m); }
  } else if (bl == 0) {
    if (ah < 0) { *zl = -I; f |= ival1__div_r(ah, bh, 1, zh, m); }
    else if (al > 0) { f |= ival1__div_r(al, bh, 0, zl, m); *zh = I; }
    else if (ah == 0) { *zl = -I; *zh = 0.0; }
    else if (al == 0) { *zl = 0.0; *zh = I; }
    else { *zl = -I; *zh = I; }
  } else if (bh == 0) {
    if (ah < 0) { f |= ival1__div_r(ah, bl, 0, zl, m); *zh = I; }
    else if (al > 0) { *zl = -I; f |= ival1__div_r(al, bl, 1, zh, m); }
    else if (ah == 0) { *zl = 0.0; *zh = I; }
    else if (al == 0) { *zl = -I; *zh = 0.0; }
    else { *zl = -I; *zh = I; }
  } else { *zl = -I; *zh = I; }
  return f;
}

/* ---- the operations: Z = A op B (or op A), one interval ---- */
static inline void ival1_add(double al, double ah, double bl, double bh, double *zl, double *zh)
{
  int m = ival1__mode();
  if (m < 0) { ival_add(&al, &ah, &bl, &bh, zl, zh, 1); return; }
  if (ival1__empty(al, ah) || ival1__empty(bl, bh)) { *zl = *zh = NAN; return; }
  *zl = ival1__canon(ival1__add_r(al, bl, 0, m)); *zh = ival1__canon(ival1__add_r(ah, bh, 1, m));
}
static inline void ival1_sub(double al, double ah, double bl, double bh, double *zl, double *zh)
{
  int m = ival1__mode();
  if (m < 0) { ival_sub(&al, &ah, &bl, &bh, zl, zh, 1); return; }
  if (ival1__empty(al, ah) || ival1__empty(bl, bh)) { *zl = *zh = NAN; return; }
  *zl = ival1__canon(ival1__add_r(al, -bh, 0, m)); *zh = ival1__canon(ival1__add_r(ah, -bl, 1, m));
}
static inline void ival1_neg(double al, double ah, double *zl, double *zh)
{
  /* exact, but its tests compare, and denormals-are-zero would read a subnormal end as 0 (arith-check found it) */
  if (ival1__mode() < 0) { ival_neg(&al, &ah, zl, zh, 1); return; }
  if (ival1__empty(al, ah)) { *zl = *zh = NAN; return; }
  *zl = ival1__canon(-ah); *zh = ival1__canon(-al);
}
static inline void ival1_mul(double al, double ah, double bl, double bh, double *zl, double *zh)
{
  int m = ival1__mode();
  if (m < 0) { ival_mul(&al, &ah, &bl, &bh, zl, zh, 1); return; }
  if (ival1__empty(al, ah) || ival1__empty(bl, bh)) { *zl = *zh = NAN; return; }
  double d0 = 0, u0 = 0, d1 = 0, u1 = 0, d2 = 0, u2 = 0, d3 = 0, u3 = 0;   /* set before use; 0 quiets -Wmaybe-uninitialized */
  if (al >= 0 && bl >= 0) {   /* both nonnegative, the common case: two corners */
    if (ival1__mul2(al, bl, &d0, &u0, m) | ival1__mul2(ah, bh, &d1, &u1, m)) { ival_mul(&al, &ah, &bl, &bh, zl, zh, 1); return; }
    *zl = ival1__canon(d0); *zh = ival1__canon(u1);
    return;
  }
  if (ival1__mul2(al, bl, &d0, &u0, m) | ival1__mul2(al, bh, &d1, &u1, m) | ival1__mul2(ah, bl, &d2, &u2, m) |
      ival1__mul2(ah, bh, &d3, &u3, m)) {
    ival_mul(&al, &ah, &bl, &bh, zl, zh, 1);
    return;
  }
  double x = d0 < d1 ? d0 : d1, y = d2 < d3 ? d2 : d3, s = u0 > u1 ? u0 : u1, t = u2 > u3 ? u2 : u3;
  *zl = ival1__canon(x < y ? x : y); *zh = ival1__canon(s > t ? s : t);
}
/* [d, d] * [bl, bh]: what ival1_mul(d, d, bl, bh, ...) gives, in two products where a mixed-sign product takes
   four. A matrix of doubles times an interval matrix (IBEX's Newton preconditioning) is made of these. d infinite
   or NaN is not a point interval: the result is empty, as ival1_mul's. */
static inline void ival1_scale(double d, double bl, double bh, double *zl, double *zh)
{
  int m = ival1__mode();
  if (m < 0) { ival_mul(&d, &d, &bl, &bh, zl, zh, 1); return; }
  if (ival1__empty(d, d) || ival1__empty(bl, bh)) { *zl = *zh = NAN; return; }
#if IVAL_PLANT_ARITH == 46   /* 46: the ends of a negative scale in the order of a positive one */
  double x = bl, y = bh;
#else
  double x = d >= 0 ? bl : bh, y = d >= 0 ? bh : bl;   /* d x <= d y: the products at the two ends, in order */
#endif
  double d0 = 0, u0 = 0, d1 = 0, u1 = 0;
  if (ival1__mul2(d, x, &d0, &u0, m) | ival1__mul2(d, y, &d1, &u1, m)) { ival_mul(&d, &d, &bl, &bh, zl, zh, 1); return; }
  *zl = ival1__canon(d0); *zh = ival1__canon(u1);
}
static inline void ival1_sqr(double al, double ah, double *zl, double *zh)
{
  int m = ival1__mode();
  if (m < 0) { ival_sqr(&al, &ah, zl, zh, 1); return; }
  if (ival1__empty(al, ah)) { *zl = *zh = NAN; return; }
  double d0 = 0, u0 = 0, d1 = 0, u1 = 0;
  if (ival1__mul2(al, al, &d0, &u0, m) | ival1__mul2(ah, ah, &d1, &u1, m)) { ival_sqr(&al, &ah, zl, zh, 1); return; }
  if (al >= 0) { *zl = d0; *zh = u1; }
  else if (ah <= 0) { *zl = d1; *zh = u0; }
  else { *zl = 0.0; *zh = u0 > u1 ? u0 : u1; }
  *zl = ival1__canon(*zl); *zh = ival1__canon(*zh);
}
static inline void ival1_div(double al, double ah, double bl, double bh, double *zl, double *zh)
{
  int m = ival1__mode();
  if (m < 0) { ival_div(&al, &ah, &bl, &bh, zl, zh, 1); return; }
  if (ival1__empty(al, ah) || ival1__empty(bl, bh)) { *zl = *zh = NAN; return; }
  double l, h;
  if (ival1__div1(al, ah, bl, bh, &l, &h, m)) { ival_div(&al, &ah, &bl, &bh, zl, zh, 1); return; }
  *zl = ival1__canon(l); *zh = ival1__canon(h);
}
static inline void ival1_recip(double al, double ah, double *zl, double *zh)
{
  int m = ival1__mode();
  if (m < 0) { ival_recip(&al, &ah, zl, zh, 1); return; }
  if (ival1__empty(al, ah)) { *zl = *zh = NAN; return; }
  double l, h;
  if (ival1__div1(1.0, 1.0, al, ah, &l, &h, m)) { ival_recip(&al, &ah, zl, zh, 1); return; }
  *zl = ival1__canon(l); *zh = ival1__canon(h);
}
static inline void ival1_sqrt(double al, double ah, double *zl, double *zh)
{
  int m = ival1__mode();
  if (m < 0) { ival_sqrt(&al, &ah, zl, zh, 1); return; }
  double a = al > 0 ? al : 0.0;
  if (al != al || !(a <= ah)) { *zl = *zh = NAN; return; }   /* nothing of [0, inf] left, or a NaN end */
  double r[2], x[2] = { a, ah };
  for (int k = 0; k < 2; k++) {
    double s = sqrt(x[k]);
    if (x[k] == 0 || isinf(x[k])) { r[k] = s; continue; }
    if (m) {   /* upward: s is the root rounded up; it is the root itself exactly when s * s - x, never negative, is 0 */
      r[k] = k || ival1__fma(s, s, -x[k]) == 0 ? s : ival1__step(s, -1);
      continue;
    }
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
