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

double cr_log(double), cr_log1p(double), cr_exp(double), cr_erfc(double);

#define CRDIST_BOUND 50                             /* the fast path is within 2^-CRDIST_BOUND, relative */
#define CRDIST_EPS 0x1p-49                          /* the test's margin, 2^(1 - CRDIST_BOUND) */
#define CRDIST_SQRT2_HI 0x1.6a09e667f3bcdp+0       /* sqrt 2 = hi + lo to about 2^-107 */
#define CRDIST_SQRT2_LO (-0x1.bdd3413b26456p-54)
#define CRDIST_RSQRT2_HI 0x1.6a09e667f3bcdp-1      /* 1/sqrt 2 = hi + lo to about 2^-107 */
#define CRDIST_RSQRT2_LO (-0x1.bdd3413b26456p-55)
#define CRDIST_TWO_RSQRTPI 0x1.20dd750429b6dp+0     /* 2/sqrt pi, rounded */
#define CRDIST_RSQRT2PI 0x1.9884533d43651p-2        /* 1/sqrt(2 pi), rounded */
#define CRDIST_LOG_SQRT2PI 0x1.d67f1c864beb5p-1     /* log sqrt(2 pi), rounded */

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

/* the rounding test for a value hi + lo whose error lies in lo alone,
   below 2^-51 |lo|: the binary32 candidate c = RN(hi + lo), then V against
   the midpoints on either side of c, each difference (hi - m) + lo exact
   but for its last rounding (hi - m is exact by Sterbenz: m is within |lo|
   plus an ulp of hi, and |lo| < |hi|/5 here). Past the largest finite
   value the next one is taken as 2^128, so the midpoint there is the
   overflow threshold. 1 and *y when both sides are clear by the margin
   |lo| 2^-49 (and the sum's own rounding); 0 otherwise. */
static inline double crdist_mid32(float y, float toward)
{
  float n = nextafterf(y, toward);
  return ((double)y + (isinf(n) ? copysign(0x1p128, n) : (double)n)) * 0.5;   /* exact */
}
static inline int crdist_round32_dd(double hi, double lo, float *y)
{
  double e = fabs(lo) * CRDIST_EPS;
  float c = (float)(hi + lo);   /* the candidate: when hi + lo drops lo and hi is a midpoint, the tie's even side */
  for (int step = 0; step < 2; step++) {   /* c, and once its neighbour on the side V lies */
    if (isinf(c)) {   /* beyond the threshold, or not */
      double d = (hi - copysign(0x1p128 - 0x1p103, c)) + lo, m = e + fabs(d) * 0x1p-52;
      if ((c > 0 ? d : -d) > m) { *y = c; return 1; }   /* V past the threshold, on c's side */
      if ((c > 0 ? d : -d) < -m) { c = nextafterf(c, 0); continue; }
      return 0;
    }
    double du = (hi - crdist_mid32(c, INFINITY)) + lo, dd = (hi - crdist_mid32(c, -INFINITY)) + lo;
    double mu = e + fabs(du) * 0x1p-52, md = e + fabs(dd) * 0x1p-52;
    if (du < -mu && dd > md) { *y = c; return 1; }
    if (du > mu) c = nextafterf(c, INFINITY);
    else if (dd < -md) c = nextafterf(c, -INFINITY);
    else return 0;   /* within the margin of a midpoint: the table answers */
  }
  return 0;
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

/* The normal distribution's functions of x (added 2026-10-13), each
   within 2^-50 as well, with u = 2^-53:
   - ndtr(x) = Phi(x) = erfc(t)/2, t = -x/sqrt 2: t kept to about 2^-104 as
     th + tl (an exact product with fma, 1/sqrt 2 as a double-double), and
     erfc(th + tl) = erfc(th) - (2/sqrt pi) e^-th^2 tl, the correction below
     2 t^2 u of erfc(th) (t < 10.7 here, as x > -15) and its own error
     and the dropped second-order term below 2^-80: erfc(th) u, the
     subtraction u, the halving exact. At most 2u plus 2^-80. (crnn's GELU
     does the same.)
   - log_ndtr(x) = log Phi(x): for x >= 0, log1p(-Phi(-x)), whose condition
     number q/((1 - q) |log(1 - q)|) is at most 1.45 for q = Phi(-x) <= 1/2:
     1.45 x 2u + u. For -30 <= x < 0, log(Phi(x)), whose condition number
     1/|log Phi| is at most 1/log 2: 2u/log 2 + u. Both below 4u. Below -30
     Phi(x) heads out of binary64's range (erfc(t) is subnormal past t =
     26.5), so the asymptotic series: log Phi(x) = -x^2/2 - log(-x) -
     log sqrt(2 pi) + log1p(s), s = -1/x^2 + 3/x^4 - 15/x^6 + 105/x^8 -
     945/x^10, whose next term, 10395/x^12, is below 2^-105 here and, the
     series enveloping, bounds what is dropped. x^2 is exact (x has 24 bits),
     and the four terms are all negative, so the three additions and the
     log and the constant add at most 4u.
   - normpdf(x) = e^(-x^2/2)/sqrt(2 pi): x^2/2 exact, exp u, the constant
     u, the product u: 3u.
   - logit(p) = log(p/(1 - p)): for p < 1/4, log of the quotient (1 - p
     carries u, the quotient u; log's condition number 1/|log y| is below
     0.92 there): 2.9u. For 1/4 <= p <= 3/4, log1p((2p - 1)/(1 - p)), both
     exact (Sterbenz), the quotient u, and log1p's condition number below
     1: 2u. Above 3/4, -logit(1 - p), 1 - p exact.
   The specials: ndtr is +0 below -15 (Phi(-15) < 2^-150, half the
   smallest subnormal) and 1 above 6 (1 - Phi(6) < 2^-25); log_ndtr is -0
   above 15; normpdf is +0 beyond 15. */
static inline double crdist_ndtr(double x)
{
  double th = -x * CRDIST_RSQRT2_HI, tl = __builtin_fma(-x, CRDIST_RSQRT2_HI, -th) + -x * CRDIST_RSQRT2_LO;
  double corr = CRDIST_TWO_RSQRTPI * cr_exp(-(th * th)) * tl;
  return 0.5 * (cr_erfc(th) - corr);
}

/* log_ndtr below -30, as hi + lo: hi = -x^2/2 exactly, lo the rest, within
   2.5u |lo| (see above). Kept apart because -x^2/2 is often a binary32
   midpoint (x^2 has 48 bits), and then lo decides the rounding: a single
   binary64 near 2^56 and beyond has no room for it. */
static inline void crdist_log_ndtr_far(double x, double *hi, double *lo)
{
  double r = 1 / (x * x);   /* the series in r = 1/x^2 */
  double s = r * (-1 + r * (3 + r * (-15 + r * (105 + r * -945))));
  *hi = -(x * x) * 0.5;
  *lo = (cr_log1p(s) - CRDIST_LOG_SQRT2PI) - cr_log(-x);
}
static inline double crdist_log_ndtr(double x)
{
  if (x >= 0) return cr_log1p(-crdist_ndtr(-x));
  if (x >= -30) return cr_log(crdist_ndtr(x));
  double hi, lo;
  crdist_log_ndtr_far(x, &hi, &lo);
  return hi + lo;
}

static inline double crdist_normpdf(double x)
{
  return cr_exp(-(x * x) * 0.5) * CRDIST_RSQRT2PI;
}

static inline double crdist_logit(double p)
{
  if (p > 0.75) return -crdist_logit(1 - p);
  if (p < 0.25) return cr_log(p / (1 - p));
  return cr_log1p((2 * p - 1) / (1 - p));
}

/* the functions by number, for the generator and the check */
enum { CRDIST_ERFINV, CRDIST_ERFCINV, CRDIST_PROBIT, CRDIST_NDTR, CRDIST_LOG_NDTR, CRDIST_NORMPDF, CRDIST_LOGIT, CRDIST_NFN };
static const char *const crdist_fn_name[CRDIST_NFN] = {"erfinv", "erfcinv", "probit", "ndtr", "log_ndtr", "normpdf", "logit"};

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
    case CRDIST_PROBIT:
    case CRDIST_LOGIT:
      if (x <= 0 || x >= 1) { *y = x == 0 ? -INFINITY : x == 1 ? INFINITY : NAN; return 1; }
      if (x == 0.5f) { *y = 0; return 1; }
      return 0;
    case CRDIST_NDTR:
      if (x <= -15 || x >= 6) { *y = x >= 6 ? 1 : 0; return 1; }
      if (x == 0) { *y = 0.5f; return 1; }
      return 0;
    case CRDIST_LOG_NDTR:
      if (isinf(x)) { *y = x > 0 ? 0 : -INFINITY; return 1; }
      if (x >= 15) { *y = -0.0f; return 1; }
      return 0;
    default:   /* CRDIST_NORMPDF */
      if (fabsf(x) >= 15) { *y = 0; return 1; }
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
    case CRDIST_NDTR: return crdist_ndtr(x);
    case CRDIST_LOG_NDTR: return crdist_log_ndtr(x);
    case CRDIST_NORMPDF: return crdist_normpdf(x);
    case CRDIST_LOGIT: return crdist_logit(x);
    default:
      if (x < 0.25) return -crdist_mul_sqrt2(crdist_ierfc(2 * x));
      if (x < 0.5) return -crdist_mul_sqrt2(crdist_ierf(1 - 2 * x));
      if (x <= 0.75) return crdist_mul_sqrt2(crdist_ierf(2 * x - 1));
      return crdist_mul_sqrt2(crdist_ierfc(2 - 2 * x));
  }
}
/* the whole decision for one binary32 input, shared by the library and the
   generator so that the two can't disagree: the special values, then the
   fast path with its rounding test (log_ndtr below -30 with the test for
   hi + lo); 1 and *y when decided, 0 when the table must answer */
static inline int crdist_decide32(int f, float x, float *y)
{
  if (crdist_special(f, x, y)) return 1;
  if (f == CRDIST_LOG_NDTR && x < -30) {
    double hi, lo;
    crdist_log_ndtr_far(x, &hi, &lo);
    return crdist_round32_dd(hi, lo, y);
  }
  return crdist_round32(crdist_fast(f, x), y);
}
#endif
