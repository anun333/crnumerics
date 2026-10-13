/* crdist-fast.h: the binary64 fast paths of crdist's functions, shared by
   the library (crdist.c) and the table generator (gen-exceptions.c). Each
   returns a binary64 value within a relative error of 2^-50 of the exact
   result, for an input inside the domain where no special answer applies,
   in the default floating-point environment. crdist_round32 turns it into
   the correctly rounded binary32 result when the error interval can't
   round two ways, and says so; otherwise the table has the answer.

   Two inverses in binary64, from polynomials (crdist-poly.h, made by
   fit.py) and CORE-MATH's correctly rounded log, with nothing else but
   IEEE operations, so every machine computes the same bits. u = 2^-53:
   - ierf(x) = erfinv(x) for 0 < x <= 1/2: x P(x^2), P of degree 13. x^2 is
     exact (x has 24 bits). P interpolates erfinv(x)/x within 2^-57.8
     (fit.py's grid), and Horner's rule in binary64 adds about u (the last
     addition, to sqrt pi/2, dominates: z P'(z)/P(z) < 0.1), 2^-53.0
     measured on the grid; the product with x adds u. About 2.1u.
   - ierfc(q) = erfcinv(q) for 0 < q <= 1/2 (q down to 2^-149): the root
     as a function of s = sqrt(-log q), 0.83 <= s <= 10.17, in six pieces,
     each a polynomial of degree 14 in s - m (m the piece's middle; s - m
     is exact by Sterbenz). log is correctly rounded (u), and the square
     root halves that and adds u: s carries 1.5u. The root's relative
     change is s y'(s)/y times s's, at most 1.62 (at q = 1/2, where
     y' = sqrt pi s e^(y^2 - s^2); it falls toward 1 as s grows): 2.43u.
     The pieces interpolate within 2^-54.1, and in binary64 within 2^-51.9
     (fit.py's grids). About 4.7u.
   The functions, from those, with every argument formed exactly
   (Sterbenz, or a multiplication by 2):
   - erfinv(x): ierf(|x|) for |x| <= 1/2, else ierfc(1 - |x|); the sign of x.
   - erfcinv(x): ierfc(x) for x <= 1/2; ierf(1 - x) up to 1; -ierf(x - 1)
     up to 3/2; -ierfc(2 - x) below 2.
   - probit(p) = sqrt 2 erfinv(2p - 1): -sqrt 2 ierfc(2p) below 1/4;
     -sqrt 2 ierf(1 - 2p) below 1/2; sqrt 2 ierf(2p - 1) up to 3/4; sqrt 2
     ierfc(2 - 2p) below 1. sqrt 2 as a double-double (hi + lo to about
     2^-107) and the product's error from fma, so the product adds u and
     about 2^-104: at most 5.7u.
   All below 2^-50 = 8u (CRDIST_BOUND); the test uses 2^-49. The two
   roundings in the test itself (r - |r| 2^-49 and r + |r| 2^-49) move each
   end by at most u relative, which the margin covers. The grid figures are
   design figures, not a proof: crdist-check checks the bound against MPFR
   on 2^16 encodings a function, and crdist-check bound-all on every input.
   (The first version, 2026-10-12, started from Winitzki's approximation
   and took three to five Newton steps through CORE-MATH's erf or erfc and
   exp: within 2^-51, but 290 to 880 ns an element against this one's 40
   to 55. Its table had 51 entries.) */
#ifndef CRDIST_FAST_H
#define CRDIST_FAST_H
#include <math.h>
#include <stdint.h>
#include <string.h>

#include "crdist-poly.h"

double cr_log(double);

#define CRDIST_BOUND 50                             /* the fast path is within 2^-CRDIST_BOUND, relative */
#define CRDIST_EPS 0x1p-49                          /* the test's margin, 2^(1 - CRDIST_BOUND) */
#define CRDIST_SQRT2_HI 0x1.6a09e667f3bcdp+0       /* sqrt 2 = hi + lo to about 2^-107 */
#define CRDIST_SQRT2_LO (-0x1.bdd3413b26456p-54)

/* the correctly rounded binary32 value of anything within |r| 2^-49 of r,
   in *y, and 1; or 0 when that interval holds two roundings */
static inline int crdist_round32(double r, float *y)
{
  double e = fabs(r) * CRDIST_EPS;
  float lo = (float)(r - e), hi = (float)(r + e);
  if (lo != hi) return 0;
  *y = lo;
  return 1;
}

static inline double crdist_ierf(double x)       /* 0 < x <= 1/2 */
{
  double z = x * x, h = crdist_c[CRDIST_DEG_C];   /* x^2 exact: x has 24 bits */
  for (int j = CRDIST_DEG_C - 1; j >= 0; j--) h = h * z + crdist_c[j];
  return x * h;
}

static inline double crdist_ierfc(double q)      /* 0 < q <= 1/2 */
{
  double s = __builtin_sqrt(-cr_log(q));
  int k = 0;
  while (k < CRDIST_NPIECE - 1 && s >= crdist_t_end[k]) k++;
  const double *c = crdist_t[k];
  double d = s - crdist_t_mid[k], h = c[CRDIST_DEG_T];   /* d exact (Sterbenz) */
  for (int j = CRDIST_DEG_T - 1; j >= 0; j--) h = h * d + c[j];
  return h;
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
