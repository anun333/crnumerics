/* ival-eft.h: the rounding primitives ival-arith.c and ival-1788.c share (not API).

   add_r, mul_r and div_r give a + b, a * b and a / b rounded down (up = 0) or up (up = 1) while the rounding mode is to
   nearest: an error-free transformation gives the exact error, which says whether the result moves one ulp; near
   underflow, where the error stops being exact, they switch the mode for the one operation. cmp_diff compares two
   differences exactly. The callers run with the mode to nearest and the flush modes off (flush_off). */
#ifndef IVAL_EFT_H
#define IVAL_EFT_H
#include <fenv.h>
#include <float.h>
#include <math.h>
#if defined(__x86_64__)
#include <immintrin.h>
#endif
#ifndef IVAL_PLANT_ARITH
#define IVAL_PLANT_ARITH 0
#endif

static inline int empty(double lo, double hi) { return !(lo <= hi) || lo == INFINITY || hi == -INFINITY; }
static inline double pred(double x) { return nextafter(x, -INFINITY); }
static inline double succ(double x) { return nextafter(x, INFINITY); }

/* one operation rounding down or up, for the cases the error-free transformations do not cover */
static inline double directed(int op, double a, double b, int up)
{
  int m = fegetround();
  fesetround(up ? FE_UPWARD : FE_DOWNWARD);
  volatile double x = a, y = b, r;
  r = op == 0 ? x + y : op == 1 ? x * y : x / y;
  fesetround(m);
  return r;
}

/* a + b rounded down (up = 0) or up (up = 1) */
static inline double add_r(double a, double b, int up)
{
  double s = a + b;
  if (isinf(s)) {
    if (isinf(a) || isinf(b)) return s;                    /* exact */
#if IVAL_PLANT_ARITH == 6   /* overflow left at infinity both ways */
    return s;
#else
    return up ? (s > 0 ? s : -DBL_MAX) : (s > 0 ? DBL_MAX : s);   /* overflow */
#endif
  }
  double bb = s - a, e = (a - (s - bb)) + (b - bb);          /* TwoSum: a + b = s + e exactly */
#if IVAL_PLANT_ARITH == 1   /* the check's control: step up even when exact */
  if (up) return e >= 0 ? succ(s) : s;
#else
  if (up) return e > 0 ? succ(s) : s;
#endif
  return e < 0 ? pred(s) : s;
}

/* a * b rounded down or up, with 0 * inf = 0 */
static inline double mul_r(double a, double b, int up)
{
#if IVAL_PLANT_ARITH != 5   /* 5: 0 * inf left to IEEE (NaN) */
  if (a == 0 || b == 0) return 0.0;
#endif
  double p = a * b;
  if (isinf(p)) {
    if (isinf(a) || isinf(b)) return p;
    return up ? (p > 0 ? p : -DBL_MAX) : (p > 0 ? DBL_MAX : p);
  }
#if IVAL_PLANT_ARITH == 2   /* the residual trusted near underflow */
  if (1) {
#else
  if (fabs(p) >= 0x1p-969) {
#endif
    double e = __builtin_fma(a, b, -p);                      /* a * b = p + e exactly */
    if (up) return e > 0 ? succ(p) : p;
    return e < 0 ? pred(p) : p;
  }
  return directed(1, a, b, up);
}

/* a / b rounded down or up, b != 0; at the ends, finite / inf is 0 and inf / finite is inf */
static inline double div_r(double a, double b, int up)
{
  if (a == 0) return 0.0;
  if (isinf(b)) return 0.0;
  double q = a / b;
  if (isinf(a)) return q;
  if (isinf(q)) return up ? (q > 0 ? q : -DBL_MAX) : (q > 0 ? DBL_MAX : q);
  if (fabs(a) >= 0x1p-960 && fabs(b) >= 0x1p-960 && fabs(q) >= 0x1p-960 && fabs(b) <= 0x1p+960) {
    double r = __builtin_fma(-q, b, a);                      /* a = q * b + r exactly; a / b - q = r / b */
    if (r == 0) return q;
#if IVAL_PLANT_ARITH == 4   /* the remainder's sign read the wrong way */
    int above = (r > 0) != (b > 0);
#else
    int above = (r > 0) == (b > 0);                          /* the true quotient is above q */
#endif
    if (up) return above ? succ(q) : q;
    return above ? q : pred(q);
  }
  return directed(2, a, b, up);
}

static inline double canon(double x) { return x == 0 ? 0.0 : x; }

/* the sign of (a - b) - (c - d), exactly, for finite a, b, c, d. Each difference is s + e exactly by TwoSum, and
   rounding to nearest is monotone, so different s order the exact values; equal s leave it to e1 - e2, whose rounded
   value has its sign. When a difference overflows, its operands are at least 2^970 in magnitude, so if both overflow
   all four are and halving them is exact; if one does, it is the larger. */
static inline int cmp_diff(double a, double b, double c, double d)
{
  double s1 = a - b, s2 = c - d;
  if (isinf(s1) || isinf(s2)) {
    if (!isinf(s2)) return s1 > 0 ? 1 : -1;
    if (!isinf(s1)) return s2 > 0 ? -1 : 1;
    a *= 0.5; b *= 0.5; c *= 0.5; d *= 0.5; s1 = a - b; s2 = c - d;
  }
  if (s1 != s2) return s1 > s2 ? 1 : -1;
  double t1 = s1 - a, e1 = (a - (s1 - t1)) + (-b - t1);
  double t2 = s2 - c, e2 = (c - (s2 - t2)) + (-d - t2);
  double g = e1 - e2;
#if IVAL_PLANT_ARITH == 16   /* 16: equal rounded differences taken as equal */
  g = 0;
#endif
  return g > 0 ? 1 : g < 0 ? -1 : 0;
}

/* Flush-to-zero and denormals-are-zero off for the call (x86-64: MXCSR's FZ and DAZ; aarch64: FPCR.FZ); fesetenv
   gives them back. A program built with -ffast-math starts with them on, and either breaks the enclosures. */
static inline void flush_off(void)
{
#if IVAL_PLANT_ARITH != 10   /* 10: the flush modes left as the caller set them */
#if defined(__x86_64__)
  _mm_setcsr(_mm_getcsr() & ~0x8040u);
#elif defined(__aarch64__)
  unsigned long r;
  __asm__ volatile("mrs %0, fpcr" : "=r"(r));
  __asm__ volatile("msr fpcr, %0" : : "r"(r & ~(1ul << 24)));
#endif
#endif
}

#endif
