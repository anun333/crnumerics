/* mx.c: correctly rounded functions on MX blocks (lowp.h; the definition
   is lowp/MX.md).

   Each element's result is taken from CORE-MATH's binary64 function
   rounded down (lo) and up (hi). They are equal when the result is exact,
   and otherwise adjacent, on either side of it. Of the two, the one with
   the odd last bit is the result rounded to odd, and rounding that again
   to any format at least 2 bits narrower, in any direction, is correct
   (Boldo and Melquiond, 2008). The one nearer zero keeps the result's
   exponent, since rounding toward zero never passes a power of two: the
   block's scale comes from those. So every block is correct by
   construction, given CORE-MATH's correct rounding downward and upward,
   with no case left to exhaustive checks (lowp/test/mx-check.c checks
   anyway). */
#include <fenv.h>
#include <math.h>
#include <string.h>
#include "lowp.h"

typedef struct { int ebits, mbits, bias, has_inf, has_nan, emax; } mxfmt;
static const mxfmt T_e5m2 = {5, 2, 15, 1, 1, 15}, T_e4m3 = {4, 3, 7, 0, 1, 8}, T_e3m2 = {3, 2, 3, 0, 0, 4},
                   T_e2m3 = {2, 3, 1, 0, 0, 2}, T_e2m1 = {2, 1, 1, 0, 0, 2};

/* the element's value into *v; 0 for a NaN or an infinity (FP8) */
static int decode(const mxfmt *f, uint8_t x, double *v)
{
  int nb = 1 + f->ebits + f->mbits, eall = (1 << f->ebits) - 1, mmask = (1 << f->mbits) - 1;
  int e = x >> f->mbits & eall, m = x & mmask;
  if (f->has_nan && e == eall && (f->has_inf || m == mmask)) return 0;
  double a = e ? ldexp(m | 1 << f->mbits, e - f->bias - f->mbits) : ldexp(m, 1 - f->bias - f->mbits);
  *v = x >> (nb - 1) & 1 ? -a : a;
  return 1;
}

/* o * 2^-s rounded to the element type in mode rnd, saturating: integer
   arithmetic on o's significand and exponent, so nothing is rounded on the
   way (o * 2^-s itself may lie far outside binary64's range) */
static uint8_t element(const mxfmt *f, double o, int s, int rnd)
{
  uint64_t b;
  memcpy(&b, &o, 8);
  int nb = 1 + f->ebits + f->mbits;
  uint8_t sg = (uint8_t)(b >> 63 << (nb - 1));
  uint32_t top = (uint32_t)(((1 << f->ebits) - 1 - f->has_inf) << f->mbits | ((1 << f->mbits) - 1 - (f->has_nan && !f->has_inf)));
  uint64_t a = b & 0x7fffffffffffffffULL;
  if (a == 0) return sg;
  int e = (int)(a >> 52);
  uint64_t m = a & ((1ULL << 52) - 1);
  if (e) m |= 1ULL << 52;
  else e = 1;
  int e2 = e - 1075 - s;   /* |o| * 2^-s = m * 2^e2 */
  int ev = e2 + 63 - __builtin_clzll(m), emin = 1 - f->bias;
  int qe = (ev > emin ? ev : emin) - f->mbits, shift = qe - e2;
  int away = rnd == LOWP_NEAREST || (rnd == LOWP_UP && !sg) || (rnd == LOWP_DOWN && sg);
  uint64_t q;
  if (shift <= 0) q = m << -shift;
  else {
    uint64_t rem, half;
    if (shift >= 64) { q = 0; rem = 1; half = ~0ULL; }
    else { q = m >> shift; rem = m & ((1ULL << shift) - 1); half = 1ULL << (shift - 1); }
    if (rem && (rnd == LOWP_NEAREST ? rem > half || (rem == half && (q & 1)) : away)) q++;
  }
  if (!q) return sg;
  if (q >> (f->mbits + 1)) { q >>= 1; qe++; }
  uint32_t biased = q >> f->mbits ? (uint32_t)(qe + f->mbits + f->bias) : 0;
  uint32_t enc = biased << f->mbits | (uint32_t)(q & ((1u << f->mbits) - 1));
  return sg | (uint8_t)(enc > top ? top : enc);   /* MX elements saturate */
}

/* one block from the results rounded down (lo) and up (hi) */
static void finish(const mxfmt *f, const double *lo, const double *hi, int nan, uint8_t *q, uint8_t *y, int rnd)
{
  for (int i = 0; i < LOWP_MX_K && !nan; i++) nan = isnan(lo[i]) || isnan(hi[i]) || isinf(lo[i]) || isinf(hi[i]);
  int E = -126, any = 0;
  for (int i = 0; i < LOWP_MX_K && !nan; i++) {
    double z = fabs(lo[i]) < fabs(hi[i]) ? lo[i] : hi[i];   /* toward zero */
    if (z == 0) continue;
    int e;
    frexp(z, &e);
    if (!any || e - 1 > E) E = e - 1;
    any = 1;
  }
  int s = E - f->emax;
  if (nan || s > 127) {
    memset(q, 0, LOWP_MX_K);
    *y = 0xff;
    return;
  }
  if (s < -127) s = -127;
  for (int i = 0; i < LOWP_MX_K; i++) {
    uint64_t a, b;
    memcpy(&a, &lo[i], 8);
    memcpy(&b, &hi[i], 8);
    double o = a == b || (a & 1) ? lo[i] : hi[i];   /* exact, or rounded to odd */
    q[i] = element(f, o, s, rnd);
  }
  *y = (uint8_t)(s + 127);
}

/* the blocks, with the caller's floating-point environment restored; every
   input is read before an output is written, so q may be p and y x */
static int one(const mxfmt *f, double (*g)(double), const uint8_t *p, const uint8_t *x, uint8_t *q, uint8_t *y,
               size_t nblocks, int mode)
{
  if (mode < 0 || mode > 3) return -1;
  fenv_t env;
  fegetenv(&env);
  for (size_t k = 0; k < nblocks; k++) {
    const uint8_t *pb = p + k * LOWP_MX_K;
    double v[LOWP_MX_K], lo[LOWP_MX_K], hi[LOWP_MX_K];
    int nan = x[k] == 0xff;
    for (int i = 0; i < LOWP_MX_K && !nan; i++) {
      if (!decode(f, pb[i], &v[i])) nan = 1;
      else v[i] = ldexp(v[i], x[k] - 127);   /* exact: within 2^-143 and 2^143 */
    }
    if (!nan) {
      fesetround(FE_DOWNWARD);
      for (int i = 0; i < LOWP_MX_K; i++) lo[i] = g(v[i]);
      fesetround(FE_UPWARD);
      for (int i = 0; i < LOWP_MX_K; i++) hi[i] = g(v[i]);
    }
    finish(f, lo, hi, nan, q + k * LOWP_MX_K, y + k, mode);
  }
  fesetenv(&env);
  return 0;
}

static int two(const mxfmt *f, double (*g)(double, double), const uint8_t *p, const uint8_t *x, const uint8_t *p2,
               const uint8_t *x2, uint8_t *q, uint8_t *y, size_t nblocks, int mode)
{
  if (mode < 0 || mode > 3) return -1;
  fenv_t env;
  fegetenv(&env);
  for (size_t k = 0; k < nblocks; k++) {
    const uint8_t *pb = p + k * LOWP_MX_K, *pb2 = p2 + k * LOWP_MX_K;
    double v[LOWP_MX_K], w[LOWP_MX_K], lo[LOWP_MX_K], hi[LOWP_MX_K];
    int nan = x[k] == 0xff || x2[k] == 0xff;
    for (int i = 0; i < LOWP_MX_K && !nan; i++) {
      if (!decode(f, pb[i], &v[i]) || !decode(f, pb2[i], &w[i])) nan = 1;
      else { v[i] = ldexp(v[i], x[k] - 127); w[i] = ldexp(w[i], x2[k] - 127); }
    }
    if (!nan) {
      fesetround(FE_DOWNWARD);
      for (int i = 0; i < LOWP_MX_K; i++) lo[i] = g(v[i], w[i]);
      fesetround(FE_UPWARD);
      for (int i = 0; i < LOWP_MX_K; i++) hi[i] = g(v[i], w[i]);
    }
    finish(f, lo, hi, nan, q + k * LOWP_MX_K, y + k, mode);
  }
  fesetenv(&env);
  return 0;
}

double cr_acos(double), cr_acosh(double), cr_acospi(double), cr_asin(double), cr_asinh(double), cr_asinpi(double),
  cr_atan(double), cr_atanh(double), cr_atanpi(double), cr_cbrt(double), cr_cos(double), cr_cosh(double),
  cr_cospi(double), cr_erf(double), cr_erfc(double), cr_exp(double), cr_exp10(double), cr_exp2(double),
  cr_expm1(double), cr_lgamma(double), cr_log(double), cr_log10(double), cr_log1p(double), cr_log2(double),
  cr_rsqrt(double), cr_sin(double), cr_sinh(double), cr_sinpi(double), cr_tan(double), cr_tanh(double),
  cr_tanpi(double), cr_tgamma(double), cr_atan2(double, double), cr_atan2pi(double, double),
  cr_hypot(double, double), cr_pow(double, double);
/* IEEE 754 requires square root correctly rounded in every mode, and the
   C library's is the hardware's instruction */
static double cr_sqrt(double x) { return sqrt(x); }

#define LOWP_MX1(t, f)                                                                                   \
  int lowp_mx_##t##_##f(const uint8_t *p, const uint8_t *x, uint8_t *q, uint8_t *y, size_t nblocks, int mode) \
  { return one(&T_##t, cr_##f, p, x, q, y, nblocks, mode); }
#define LOWP_MX2(t, f)                                                                                   \
  int lowp_mx_##t##_##f(const uint8_t *p, const uint8_t *x, const uint8_t *p2, const uint8_t *x2, uint8_t *q, \
                        uint8_t *y, size_t nblocks, int mode)                                             \
  { return two(&T_##t, cr_##f, p, x, p2, x2, q, y, nblocks, mode); }
#include "lowp-mx-list.h"
