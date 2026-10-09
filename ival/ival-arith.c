/* ival-arith.c: interval arithmetic, each result the tightest enclosure (ival.h).

   Each bound is computed rounding to nearest, and the exact rounding error, from an error-free transformation, says
   whether the bound moves one ulp (down for a lower bound, up for an upper one):
     - a + b: TwoSum's error term, exact for any operands when the sum does not overflow;
     - a * b: fma(a, b, -p), exact when |p| >= 2^-969 (the error then is a binary64 value);
     - a / b: the remainder fma(-q, b, a), exact when a, b and q are at least 2^-960 in magnitude; the sign of r / b is
       the sign of the true quotient minus q.
   Where those conditions fail (results near the underflow range), the bound is computed rounding down or up instead.
   An overflow rounds to DBL_MAX one way and to the infinity the other. So no rounding-mode switch is needed in the
   common case, and the same steps vectorize.

   Intervals as IEEE 1788 has them: [NaN, NaN] is empty, as is any input with lo > hi, a NaN end, lo = +inf or
   hi = -inf; [-inf, +inf] is the whole line; zeros are unsigned. At interval ends, 0 * inf is 0 (the end stands for a
   limit). Division follows 1788's case table, by where 0 lies in the divisor (a divisor that is exactly [0, 0] gives the
   empty interval). The C rounding mode and flags are left as they were. */
#include <fenv.h>
#include <float.h>
#include <math.h>
#include "ival.h"
#ifndef IVAL_PLANT_ARITH
#define IVAL_PLANT_ARITH 0
#endif

static int empty(double lo, double hi) { return !(lo <= hi) || lo == INFINITY || hi == -INFINITY; }
static double pred(double x) { return nextafter(x, -INFINITY); }
static double succ(double x) { return nextafter(x, INFINITY); }

/* one operation rounding down or up, for the cases the error-free transformations do not cover */
static double directed(int op, double a, double b, int up)
{
  int m = fegetround();
  fesetround(up ? FE_UPWARD : FE_DOWNWARD);
  volatile double x = a, y = b, r;
  r = op == 0 ? x + y : op == 1 ? x * y : x / y;
  fesetround(m);
  return r;
}

/* a + b rounded down (up = 0) or up (up = 1) */
static double add_r(double a, double b, int up)
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
static double mul_r(double a, double b, int up)
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
static double div_r(double a, double b, int up)
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

#define LOOP2(...)                                                                      \
  fenv_t env;                                                                           \
  fegetenv(&env);                                                                       \
  fesetround(FE_TONEAREST);                                                             \
  for (size_t i = 0; i < n; i++) {                                                      \
    double al = alo[i], ah = ahi[i], bl = blo[i], bh = bhi[i], zl, zh;                  \
    if (empty(al, ah) || empty(bl, bh)) { zlo[i] = zhi[i] = NAN; continue; }           \
    __VA_ARGS__                                                                         \
    zlo[i] = zl; zhi[i] = zh;                                                           \
  }                                                                                     \
  fesetenv(&env);

void ival_add(const double *alo, const double *ahi, const double *blo, const double *bhi, double *zlo, double *zhi,
              size_t n)
{
  LOOP2(zl = add_r(al, bl, 0); zh = add_r(ah, bh, 1);)
}

void ival_sub(const double *alo, const double *ahi, const double *blo, const double *bhi, double *zlo, double *zhi,
              size_t n)
{
  LOOP2(zl = add_r(al, -bh, 0); zh = add_r(ah, -bl, 1);)
}

static double min4(double a, double b, double c, double d) { double x = a < b ? a : b, y = c < d ? c : d; return x < y ? x : y; }
static double max4(double a, double b, double c, double d) { double x = a > b ? a : b, y = c > d ? c : d; return x > y ? x : y; }

void ival_mul(const double *alo, const double *ahi, const double *blo, const double *bhi, double *zlo, double *zhi,
              size_t n)
{
  LOOP2(zl = min4(mul_r(al, bl, 0), mul_r(al, bh, 0), mul_r(ah, bl, 0), mul_r(ah, bh, 0));
        zh = max4(mul_r(al, bl, 1), mul_r(al, bh, 1), mul_r(ah, bl, 1), mul_r(ah, bh, 1));)
}

/* A / B by 1788's table: where 0 lies in B (outside, at its lower end, at its upper end, inside), and A's sign */
static void div1(double al, double ah, double bl, double bh, double *zl, double *zh)
{
  const double I = INFINITY;
  if (bl == 0 && bh == 0) { *zl = *zh = NAN; return; }               /* B = [0, 0]: empty */
  if (al == 0 && ah == 0) { *zl = *zh = 0.0; return; }               /* A = [0, 0] */
  if (bl > 0) {                                                      /* B > 0 */
    if (al >= 0) { *zl = div_r(al, bh, 0); *zh = div_r(ah, bl, 1); }
    else if (ah <= 0) { *zl = div_r(al, bl, 0); *zh = div_r(ah, bh, 1); }
    else { *zl = div_r(al, bl, 0); *zh = div_r(ah, bl, 1); }
  } else if (bh < 0) {                                               /* B < 0 */
    if (al >= 0) { *zl = div_r(ah, bh, 0); *zh = div_r(al, bl, 1); }
    else if (ah <= 0) { *zl = div_r(ah, bl, 0); *zh = div_r(al, bh, 1); }
    else { *zl = div_r(ah, bh, 0); *zh = div_r(al, bh, 1); }
  } else if (bl == 0) {                                              /* B = [0, b], b > 0 */
    if (ah < 0) { *zl = -I; *zh = div_r(ah, bh, 1); }
    else if (al > 0) { *zl = div_r(al, bh, 0); *zh = I; }
#if IVAL_PLANT_ARITH == 3   /* A = [a, 0] over B = [0, b] taken as the whole line */
    else if (ah == 0) { *zl = -I; *zh = I; }
#else
    else if (ah == 0) { *zl = -I; *zh = 0.0; }                       /* A = [a, 0], a < 0 */
#endif
    else if (al == 0) { *zl = 0.0; *zh = I; }                        /* A = [0, a], a > 0 */
    else { *zl = -I; *zh = I; }
  } else if (bh == 0) {                                              /* B = [b, 0], b < 0 */
    if (ah < 0) { *zl = div_r(ah, bl, 0); *zh = I; }
    else if (al > 0) { *zl = -I; *zh = div_r(al, bl, 1); }
    else if (ah == 0) { *zl = 0.0; *zh = I; }
    else if (al == 0) { *zl = -I; *zh = 0.0; }
    else { *zl = -I; *zh = I; }
  } else { *zl = -I; *zh = I; }                                      /* 0 inside B, A != [0, 0] */
}

void ival_div(const double *alo, const double *ahi, const double *blo, const double *bhi, double *zlo, double *zhi,
              size_t n)
{
  LOOP2(div1(al, ah, bl, bh, &zl, &zh);)
}

#define LOOP1(...)                                                                      \
  fenv_t env;                                                                           \
  fegetenv(&env);                                                                       \
  fesetround(FE_TONEAREST);                                                             \
  for (size_t i = 0; i < n; i++) {                                                      \
    double al = lo[i], ah = hi[i], zl, zh;                                              \
    if (empty(al, ah)) { ylo[i] = yhi[i] = NAN; continue; }                            \
    __VA_ARGS__                                                                         \
    ylo[i] = zl; yhi[i] = zh;                                                           \
  }                                                                                     \
  fesetenv(&env);

void ival_neg(const double *lo, const double *hi, double *ylo, double *yhi, size_t n)
{
  LOOP1(zl = -ah; zh = -al;)
}

void ival_sqr(const double *lo, const double *hi, double *ylo, double *yhi, size_t n)
{
  LOOP1(if (al >= 0) { zl = mul_r(al, al, 0); zh = mul_r(ah, ah, 1); }
        else if (ah <= 0) { zl = mul_r(ah, ah, 0); zh = mul_r(al, al, 1); }
        else { double u = mul_r(al, al, 1), v = mul_r(ah, ah, 1); zl = 0.0; zh = u > v ? u : v; })
}

void ival_recip(const double *lo, const double *hi, double *ylo, double *yhi, size_t n)
{
  LOOP1(div1(1.0, 1.0, al, ah, &zl, &zh);)
}
