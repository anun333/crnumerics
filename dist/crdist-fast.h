/* crdist-fast.h: the binary64 fast paths of crdist's functions, shared by
   the library (crdist.c) and the table generator (gen-exceptions.c). Each
   returns a binary64 value within a relative error of 2^-51 of the exact
   result, for an input inside the domain where no special answer applies,
   in the default floating-point environment. crdist_round32 turns it into
   the correctly rounded binary32 result when the error interval can't
   round two ways, and says so; otherwise the table has the answer.

   Two inverses in binary64, each from a start and Newton's method on a
   CORE-MATH function (correctly rounded), and nothing else but IEEE
   operations, so every machine computes the same bits:
   - ierf(x) = erfinv(x) for 0 < x <= 1/2 (the root y of erf(y) = x, with
     0 < y < 0.477). Below 2^-14 the series (sqrt pi/2)(x + (pi/12) x^3):
     the next term is below 0.144 x^5, under 2^-58 relative, and the three
     roundings that matter (x + c x^3, the product, the constant) give at
     most 3u. Otherwise Winitzki's approximation (a = 0.147, relative error
     below 2e-3), then four Newton steps y -= (erf(y) - x)/((2/sqrt pi)
     e^-y^2).
   - ierfc(q) = erfcinv(q) for 0 < q <= 1/2 (the root t of erfc(t) = q, with
     0.477 < t < 10.1, q down to 2^-149). Winitzki's approximation with
     log(q (2 - q)), relative error below 2e-3 here too, then five Newton
     steps t += (erfc(t) - q)/((2/sqrt pi) e^-t^2).
   Newton's error: with e the relative error of the iterate, the next is
   about y^2 e^2 (erf) or t^2 e^2 (erfc), since f''/f' = -2y; from 2e-3
   that is below 1e-20 after three steps for erf (y^2 < 0.23), and below
   1e-23 after five for erfc (t^2 < 103). The last step adds its rounding
   errors: the residual erf(y) - x (or erfc(t) - q) is exact by Sterbenz
   but erf(y) carries u of x, which moves the root by u x/(y erf'(y)), at
   most 1.17u of y (at x = 1/2; the same for erfc at q = 1/2, and less as
   t grows); the correction's own few u apply to a correction below 2^-60
   of y; the final addition adds u. So at most 2.2u, plus terms below
   2^-60.
   The functions, from those, with every argument formed exactly
   (Sterbenz, or a multiplication by 2):
   - erfinv(x): ierf(|x|) for |x| <= 1/2, else ierfc(1 - |x|); the sign of x.
   - erfcinv(x): ierfc(x) for x <= 1/2; ierf(1 - x) up to 1; -ierf(x - 1)
     up to 3/2; -ierfc(2 - x) below 2.
   - probit(p) = sqrt 2 erfinv(2p - 1): -sqrt 2 ierfc(2p) below 1/4;
     -sqrt 2 ierf(1 - 2p) below 1/2; sqrt 2 ierf(2p - 1) up to 3/4; sqrt 2
     ierfc(2 - 2p) below 1. sqrt 2 as a double-double (hi + lo to about
     2^-107) and the product's error from fma, so the product adds u and
     about 2^-104: at most 3.2u.
   All below 2^-51 = 4u; the test uses 2^-50. The two roundings in the test
   itself (r - |r| 2^-50 and r + |r| 2^-50) move each end by at most u
   relative, which the margin covers. crdist-check measures the error on
   every input (make crdist-check-all) and fails above 2^-51. */
#ifndef CRDIST_FAST_H
#define CRDIST_FAST_H
#include <math.h>
#include <stdint.h>
#include <string.h>

double cr_erf(double), cr_erfc(double), cr_exp(double), cr_log(double), cr_log1p(double);

#define CRDIST_EPS 0x1p-50
#define CRDIST_SQRTPI_2 0x1.c5bf891b4ef6bp-1       /* sqrt pi / 2, rounded */
#define CRDIST_TWO_RSQRTPI 0x1.20dd750429b6dp+0    /* 2 / sqrt pi, rounded */
#define CRDIST_PI_12 0x1.0c152382d7366p-2          /* pi / 12, rounded */
#define CRDIST_SQRT2_HI 0x1.6a09e667f3bcdp+0       /* sqrt 2 = hi + lo to about 2^-107 */
#define CRDIST_SQRT2_LO (-0x1.bdd3413b26456p-54)
#define CRDIST_W_T 0x1.152af46b7c65ep+2            /* Winitzki: 2/(pi a), a = 0.147 */
#define CRDIST_W_A 0x1.b35fc845a8ecep+2            /* 1/a */

/* the correctly rounded binary32 value of anything within |r| 2^-50 of r,
   in *y, and 1; or 0 when that interval holds two roundings */
static inline int crdist_round32(double r, float *y)
{
  double e = fabs(r) * CRDIST_EPS;
  float lo = (float)(r - e), hi = (float)(r + e);
  if (lo != hi) return 0;
  *y = lo;
  return 1;
}

/* the generator's look at the iterations: the start, the result and the
   last correction (gen-exceptions.c); nothing in the library */
#ifndef CRDIST_NOTE
#define CRDIST_NOTE(which, start, result, last) ((void)0)
#endif

/* Winitzki's erfinv, from L = log(1 - x^2) (L < 0) */
static inline double crdist_winitzki(double L)
{
  double t = CRDIST_W_T + 0.5 * L;
  return __builtin_sqrt(__builtin_sqrt(t * t - L * CRDIST_W_A) - t);
}

static inline double crdist_ierf(double x)       /* 0 < x <= 1/2 */
{
  if (x < 0x1p-14) return CRDIST_SQRTPI_2 * (x + CRDIST_PI_12 * (x * x * x));
  double y0 = crdist_winitzki(cr_log1p(-(x * x))), y = y0, c = 0;   /* x^2 exact: x has 24 bits */
  for (int i = 0; i < 3; i++) { c = (cr_erf(y) - x) / (CRDIST_TWO_RSQRTPI * cr_exp(-(y * y))); y -= c; }
  CRDIST_NOTE(0, y0, y, c);
  return y;
}

static inline double crdist_ierfc(double q)      /* 0 < q <= 1/2 */
{
  double t0 = crdist_winitzki(cr_log(q * (2 - q))), t = t0, c = 0;
  for (int i = 0; i < 5; i++) { c = (cr_erfc(t) - q) / (CRDIST_TWO_RSQRTPI * cr_exp(-(t * t))); t += c; }
  CRDIST_NOTE(1, t0, t, c);
  return t;
}

static inline double crdist_mul_sqrt2(double t)
{
  double h = t * CRDIST_SQRT2_HI, l = __builtin_fma(t, CRDIST_SQRT2_HI, -h) + t * CRDIST_SQRT2_LO;
  return h + l;
}

/* the functions by number, for the generator and the check */
enum { CRDIST_ERFINV, CRDIST_ERFCINV, CRDIST_PROBIT, CRDIST_NFN };
static const char *const crdist_fn_name[CRDIST_NFN] = {"erfinv", "erfcinv", "probit"};

/* the value at an input outside the domain, at its ends, or where the
   answer is exact; 1 if x is one of those */
static inline int crdist_special(int f, float x, float *y)
{
  if (x != x) { *y = x + x; return 1; }
  switch (f) {
    case CRDIST_ERFINV:
      if (x == 0) { *y = x; return 1; }
      if (fabsf(x) >= 1) { *y = fabsf(x) == 1 ? copysignf(INFINITY, x) : NAN; return 1; }
      return 0;
    case CRDIST_ERFCINV:
      if (x <= 0 || x >= 2) { *y = x == 0 ? INFINITY : x == 2 ? -INFINITY : NAN; return 1; }
      if (x == 1) { *y = 0; return 1; }
      return 0;
    default:
      if (x <= 0 || x >= 1) { *y = x == 0 ? -INFINITY : x == 1 ? INFINITY : NAN; return 1; }
      if (x == 0.5f) { *y = 0; return 1; }
      return 0;
  }
}

/* the binary64 value at an x that crdist_special leaves */
static inline double crdist_fast(int f, double x)
{
  switch (f) {
    case CRDIST_ERFINV: {
      double a = fabs(x), y = a <= 0.5 ? crdist_ierf(a) : crdist_ierfc(1 - a);
      return copysign(y, x);
    }
    case CRDIST_ERFCINV:
      if (x <= 0.5) return crdist_ierfc(x);
      if (x <= 1) return crdist_ierf(1 - x);
      if (x <= 1.5) return -crdist_ierf(x - 1);
      return -crdist_ierfc(2 - x);
    default:
      if (x < 0.25) return -crdist_mul_sqrt2(crdist_ierfc(2 * x));
      if (x < 0.5) return -crdist_mul_sqrt2(crdist_ierf(1 - 2 * x));
      if (x <= 0.75) return crdist_mul_sqrt2(crdist_ierf(2 * x - 1));
      return crdist_mul_sqrt2(crdist_ierfc(2 - 2 * x));
  }
}
#endif
