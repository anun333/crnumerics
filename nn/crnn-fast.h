/* crnn-fast.h: the binary64 fast paths of crnn's one-argument functions,
   shared by the library (crnn.c) and the table generator
   (gen-exceptions.c). Each returns a binary64 value within a relative
   error of 2^-51 of the exact result, for a finite input, in the default
   floating-point environment. crnn_round32 turns it into the correctly
   rounded binary32 result when the error interval can't round two ways,
   and says so; otherwise the table has the answer.

   The bounds, with u = 2^-53 the unit roundoff and every CORE-MATH
   function correctly rounded (error at most u, relative):
   - sigmoid = 1/(1 + e), e = exp(-x): e carries u; 1 + e divides it by
     (1 + e)/e > 1 and adds u; the division adds u. At most 3u, plus terms
     in u^2.
   - silu = x/(1 + e): the same three roundings, 3u.
   - softplus: for x > 0, x + log1p(exp(-x)); otherwise log1p(exp(x)).
     log1p's condition number, e/((1 + e) log1p(e)), is below 1 for e >= 0,
     so exp's u stays u; log1p adds u; the addition of x, which only
     shrinks the relative error of the smaller term, adds u. At most 3u.
   - gelu = x/2 erfc(t), t = -x/sqrt 2. erfc's condition number grows like
     2t^2, so t is kept to about 2^-104 as th + tl (an exact product with
     fma, and 1/sqrt 2 as a double-double), and erfc(th + tl) is erfc(th)
     - (2/sqrt pi) exp(-th^2) tl. The correction is below 2t^2 u of
     erfc(th) (t below 27 wherever erfc(th) is a normal double), so its own
     error of about t^2 u, and the dropped second-order term, 2t^4 u^2, are
     below 2^-80. erfc(th) carries u, the subtraction u, the product with
     x/2 (exact) u. At most 3u plus 2^-80.
   - rsqrt: CORE-MATH's binary64 rsqrt, u.
   All below 2^-51; the test uses 2^-50. The two roundings in the test
   itself (r - |r| 2^-50 and r + |r| 2^-50) move each end by at most u
   relative, which the margin covers. */
#ifndef CRNN_FAST_H
#define CRNN_FAST_H
#include <math.h>
#include <stdint.h>
#include <string.h>

double cr_exp(double), cr_log1p(double), cr_erfc(double), cr_rsqrt(double);

#define CRNN_EPS 0x1p-50
#define CRNN_RSQRT2_HI 0x1.6a09e667f3bcdp-1      /* 1/sqrt 2 = hi + lo to about 2^-107 */
#define CRNN_RSQRT2_LO (-0x1.bdd3413b26456p-55)
#define CRNN_TWO_RSQRTPI 0x1.20dd750429b6dp+0    /* 2/sqrt pi, rounded */

/* the correctly rounded binary32 value of anything within |r| 2^-50 of r,
   in *y, and 1; or 0 when that interval holds two roundings */
static inline int crnn_round32(double r, float *y)
{
  double e = fabs(r) * CRNN_EPS;
  float lo = (float)(r - e), hi = (float)(r + e);
  if (lo != hi) return 0;
  *y = lo;
  return 1;
}

static inline double crnn_fast_sigmoid(double x) { return 1.0 / (1.0 + cr_exp(-x)); }
static inline double crnn_fast_silu(double x) { return x / (1.0 + cr_exp(-x)); }
static inline double crnn_fast_softplus(double x) { return x > 0 ? x + cr_log1p(cr_exp(-x)) : cr_log1p(cr_exp(x)); }
static inline double crnn_fast_gelu(double x)
{
  double th = -x * CRNN_RSQRT2_HI, tl = __builtin_fma(-x, CRNN_RSQRT2_HI, -th) + -x * CRNN_RSQRT2_LO;
  double corr = CRNN_TWO_RSQRTPI * cr_exp(-(th * th)) * tl;
  return 0.5 * x * (cr_erfc(th) - corr);
}
static inline double crnn_fast_rsqrt(double x) { return cr_rsqrt(x); }

/* the one-argument functions by number, for the generator and the check */
enum { CRNN_SIGMOID, CRNN_SILU, CRNN_GELU, CRNN_SOFTPLUS, CRNN_RSQRT, CRNN_NFN1 };
static const char *const crnn_fn1_name[CRNN_NFN1] = {"sigmoid", "silu", "gelu", "softplus", "rsqrt"};

/* the value at a non-finite input, or at an input where no rounding is
   needed; 1 if x is one of those */
static inline int crnn_special(int f, float x, float *y)
{
  if (x != x) { *y = x + x; return 1; }
  if (f == CRNN_RSQRT) { if (x == 0 || isinf(x) || x < 0) { *y = (float)cr_rsqrt(x); return 1; } return 0; }
  if (x == INFINITY) { *y = f == CRNN_SIGMOID ? 1.0f : INFINITY; return 1; }
  if (x == -INFINITY) { *y = f == CRNN_SIGMOID || f == CRNN_SOFTPLUS ? 0.0f : -0.0f; return 1; }
  if (x == 0 && f != CRNN_SOFTPLUS) { *y = f == CRNN_SIGMOID ? 0.5f : x; return 1; }   /* sigmoid(+-0) = 1/2 exactly */
  /* silu and gelu at |x| < 2^-120: x/2 plus a positive term below x^2/2
     (silu - x/2 = x tanh(x/2)/2, gelu - x/2 = x/2 erf(x/sqrt 2), both > 0
     for x != 0), far below binary64's reach. x/2 is exact in binary64 and
     often a binary32 midpoint (x a subnormal, or in the lowest normal
     binade, with an odd last bit), where the exact value is just above it.
     Adding |x/2| 2^-40 breaks the tie the same way: it survives in
     binary64 (2^-40 is above binary64's 2^-53), and stays far below half a
     binary32 quantum, so it moves nothing else. (A fixed 2^-300, tried
     first, vanished in binary64 next to 2^-150: crnn-check caught it.) */
  if ((f == CRNN_SILU || f == CRNN_GELU) && fabsf(x) < 0x1p-120f) { double t = (double)x * 0.5; *y = (float)(t + fabs(t) * 0x1p-40); return 1; }
  return 0;
}

static inline double crnn_fast(int f, double x)
{
  switch (f) {
    case CRNN_SIGMOID: return crnn_fast_sigmoid(x);
    case CRNN_SILU: return crnn_fast_silu(x);
    case CRNN_GELU: return crnn_fast_gelu(x);
    case CRNN_SOFTPLUS: return crnn_fast_softplus(x);
    default: return crnn_fast_rsqrt(x);
  }
}
#endif
