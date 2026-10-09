/* ival-arith.c: interval arithmetic, each result the tightest enclosure (ival.h).

   Each bound is computed rounding to nearest, and the exact rounding error, from an error-free transformation, says
   whether the bound moves one ulp (down for a lower bound, up for an upper one):
     - a + b: TwoSum's error term, exact for any operands when the sum does not overflow;
     - a * b: fma(a, b, -p), exact when |p| >= 2^-969 (the error then is a binary64 value);
     - a / b: the remainder fma(-q, b, a), exact when a, b and q are at least 2^-960 in magnitude; the sign of r / b is
       the sign of the true quotient minus q.
   Where those conditions fail (results near the underflow range), the bound is computed rounding down or up instead.
   An overflow rounds to DBL_MAX one way and to the infinity the other. So no rounding-mode switch is needed in the
   common case, and the same steps vectorize: on x86-64 with AVX2 and FMA, four intervals at a time, the lanes the
   transformations do not cover redone by the scalar code, which the vector results equal bit for bit.
   Flush-to-zero and denormals-are-zero are turned off for the call: either would break the transformations.

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

/* ---- four intervals at a time (AVX2 and FMA, x86-64), bit for bit the scalar results ----
   The same steps with masks. Lanes the transformations do not cover (near underflow; division by an interval that
   touches or holds 0, or of [0, 0]) are recomputed by the scalar functions above, which define the result. pred and
   succ step the bit pattern as nextafter does (0 to the smallest subnormal of the other sign, +inf to DBL_MAX). */
#if defined(__x86_64__)
#include <immintrin.h>
#define TGT __attribute__((target("avx2,fma")))
static int vec_ok = -1;
int ival__arith_scalar;   /* not API: the check sets it to run the scalar code alone */
static int have_vec(void)
{
  if (ival__arith_scalar) return 0;
  if (vec_ok < 0) { __builtin_cpu_init(); vec_ok = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma"); }
  return vec_ok;
}
TGT static inline __m256d vpred(__m256d x)
{
  const __m256d z = _mm256_setzero_pd();
  __m256i u = _mm256_castpd_si256(x), one = _mm256_set1_epi64x(1);
  __m256d pos = _mm256_cmp_pd(x, z, _CMP_GT_OQ), neg = _mm256_cmp_pd(x, z, _CMP_LT_OQ);
  __m256d isz = _mm256_cmp_pd(x, z, _CMP_EQ_OQ), ninf = _mm256_cmp_pd(x, _mm256_set1_pd(-INFINITY), _CMP_EQ_OQ);
  __m256d r = _mm256_blendv_pd(x, _mm256_castsi256_pd(_mm256_sub_epi64(u, one)), pos);
#if IVAL_PLANT_ARITH == 7   /* a negative number stepped toward 0 */
  r = _mm256_blendv_pd(r, _mm256_castsi256_pd(_mm256_sub_epi64(u, one)), _mm256_andnot_pd(ninf, neg));
#else
  r = _mm256_blendv_pd(r, _mm256_castsi256_pd(_mm256_add_epi64(u, one)), _mm256_andnot_pd(ninf, neg));
#endif
  return _mm256_blendv_pd(r, _mm256_set1_pd(-0x1p-1074), isz);
}
TGT static inline __m256d vsucc(__m256d x) { const __m256d m = _mm256_set1_pd(-0.0); return _mm256_xor_pd(vpred(_mm256_xor_pd(x, m)), m); }
TGT static inline __m256d vabs(__m256d x) { return _mm256_andnot_pd(_mm256_set1_pd(-0.0), x); }
TGT static inline __m256d visinf(__m256d x) { return _mm256_cmp_pd(vabs(x), _mm256_set1_pd(INFINITY), _CMP_EQ_OQ); }
TGT static inline __m256d vempty(__m256d lo, __m256d hi)
{
  __m256d bad = _mm256_cmp_pd(lo, hi, _CMP_NLE_UQ);   /* !(lo <= hi), NaN included */
  bad = _mm256_or_pd(bad, _mm256_cmp_pd(lo, _mm256_set1_pd(INFINITY), _CMP_EQ_OQ));
  return _mm256_or_pd(bad, _mm256_cmp_pd(hi, _mm256_set1_pd(-INFINITY), _CMP_EQ_OQ));
}
/* overflow to infinity rounding to nearest: rounding down gives DBL_MAX for +inf, rounding up -DBL_MAX for -inf */
TGT static inline __m256d vovf(__m256d s, int up)
{
  __m256d posinf = _mm256_cmp_pd(s, _mm256_set1_pd(INFINITY), _CMP_EQ_OQ), neginf = _mm256_cmp_pd(s, _mm256_set1_pd(-INFINITY), _CMP_EQ_OQ);
  return up ? _mm256_blendv_pd(s, _mm256_set1_pd(-DBL_MAX), neginf) : _mm256_blendv_pd(s, _mm256_set1_pd(DBL_MAX), posinf);
}
TGT static inline __m256d vadd_r(__m256d a, __m256d b, int up)
{
  __m256d s = _mm256_add_pd(a, b);
  __m256d bb = _mm256_sub_pd(s, a), e = _mm256_add_pd(_mm256_sub_pd(a, _mm256_sub_pd(s, bb)), _mm256_sub_pd(b, bb));
  const __m256d z = _mm256_setzero_pd();
  __m256d r = up ? _mm256_blendv_pd(s, vsucc(s), _mm256_cmp_pd(e, z, _CMP_GT_OQ))
                 : _mm256_blendv_pd(s, vpred(s), _mm256_cmp_pd(e, z, _CMP_LT_OQ));
  __m256d sinf = visinf(s), opinf = _mm256_or_pd(visinf(a), visinf(b));
  r = _mm256_blendv_pd(r, vovf(s, up), _mm256_andnot_pd(opinf, sinf));   /* overflow */
  return _mm256_blendv_pd(r, s, _mm256_and_pd(opinf, sinf));             /* an infinite operand: exact */
}
/* a * b rounded down or up; lanes needing the scalar path are set in *fb */
TGT static inline __m256d vmul_r(__m256d a, __m256d b, int up, __m256d *fb)
{
  const __m256d z = _mm256_setzero_pd();
  __m256d p = _mm256_mul_pd(a, b), e = _mm256_fmsub_pd(a, b, p);
  __m256d r = up ? _mm256_blendv_pd(p, vsucc(p), _mm256_cmp_pd(e, z, _CMP_GT_OQ))
                 : _mm256_blendv_pd(p, vpred(p), _mm256_cmp_pd(e, z, _CMP_LT_OQ));
  __m256d zero = _mm256_or_pd(_mm256_cmp_pd(a, z, _CMP_EQ_OQ), _mm256_cmp_pd(b, z, _CMP_EQ_OQ));
  __m256d pinf = visinf(p), opinf = _mm256_or_pd(visinf(a), visinf(b));
  r = _mm256_blendv_pd(r, vovf(p, up), _mm256_andnot_pd(opinf, pinf));
  r = _mm256_blendv_pd(r, p, _mm256_and_pd(opinf, pinf));
  __m256d tiny = _mm256_andnot_pd(_mm256_or_pd(zero, pinf), _mm256_cmp_pd(vabs(p), _mm256_set1_pd(0x1p-969), _CMP_LT_OQ));
  *fb = _mm256_or_pd(*fb, tiny);
  return _mm256_blendv_pd(r, z, zero);
}
/* a / b rounded down or up, b != 0; fallback lanes as for vmul_r */
TGT static inline __m256d vdiv_r(__m256d a, __m256d b, int up, __m256d *fb)
{
  const __m256d z = _mm256_setzero_pd();
  __m256d q = _mm256_div_pd(a, b), r = _mm256_fnmadd_pd(q, b, a);   /* a - q * b, exact where used */
  __m256d above = _mm256_xor_pd(_mm256_cmp_pd(r, z, _CMP_GT_OQ), _mm256_cmp_pd(b, z, _CMP_LT_OQ));   /* (r > 0) == (b > 0) */
  above = _mm256_andnot_pd(_mm256_cmp_pd(r, z, _CMP_EQ_OQ), above);
  __m256d below = _mm256_andnot_pd(_mm256_or_pd(above, _mm256_cmp_pd(r, z, _CMP_EQ_OQ)), _mm256_castsi256_pd(_mm256_set1_epi64x(-1)));
  __m256d res = up ? _mm256_blendv_pd(q, vsucc(q), above) : _mm256_blendv_pd(q, vpred(q), below);
  __m256d azero = _mm256_cmp_pd(a, z, _CMP_EQ_OQ), binf = visinf(b), ainf = visinf(a), qinf = visinf(q);
  res = _mm256_blendv_pd(res, vovf(q, up), _mm256_andnot_pd(_mm256_or_pd(ainf, binf), qinf));   /* overflow */
  res = _mm256_blendv_pd(res, q, ainf);                                                          /* inf / finite */
  res = _mm256_blendv_pd(res, z, _mm256_or_pd(azero, binf));                                     /* 0 / b, a / inf */
  const __m256d lo = _mm256_set1_pd(0x1p-960), hi = _mm256_set1_pd(0x1p+960);
  __m256d ok = _mm256_and_pd(_mm256_cmp_pd(vabs(a), lo, _CMP_GE_OQ), _mm256_cmp_pd(vabs(b), lo, _CMP_GE_OQ));
  ok = _mm256_and_pd(ok, _mm256_and_pd(_mm256_cmp_pd(vabs(q), lo, _CMP_GE_OQ), _mm256_cmp_pd(vabs(b), hi, _CMP_LE_OQ)));
  __m256d special = _mm256_or_pd(_mm256_or_pd(azero, binf), _mm256_or_pd(ainf, qinf));
#if IVAL_PLANT_ARITH != 8   /* 8: the vector quotient trusted near underflow and overflow */
  *fb = _mm256_or_pd(*fb, _mm256_andnot_pd(_mm256_or_pd(ok, special), _mm256_castsi256_pd(_mm256_set1_epi64x(-1))));
#endif
  return res;
}
#endif

/* ---- one interval: the scalar results, which define what the vector code must give ---- */
static void s_add(double al, double ah, double bl, double bh, double *zl, double *zh)
{
  if (empty(al, ah) || empty(bl, bh)) { *zl = *zh = NAN; return; }
  *zl = add_r(al, bl, 0); *zh = add_r(ah, bh, 1);
}
static void s_sub(double al, double ah, double bl, double bh, double *zl, double *zh)
{
  if (empty(al, ah) || empty(bl, bh)) { *zl = *zh = NAN; return; }
  *zl = add_r(al, -bh, 0); *zh = add_r(ah, -bl, 1);
}
static double min4(double a, double b, double c, double d) { double x = a < b ? a : b, y = c < d ? c : d; return x < y ? x : y; }
static double max4(double a, double b, double c, double d) { double x = a > b ? a : b, y = c > d ? c : d; return x > y ? x : y; }
static void s_mul(double al, double ah, double bl, double bh, double *zl, double *zh)
{
  if (empty(al, ah) || empty(bl, bh)) { *zl = *zh = NAN; return; }
  *zl = min4(mul_r(al, bl, 0), mul_r(al, bh, 0), mul_r(ah, bl, 0), mul_r(ah, bh, 0));
  *zh = max4(mul_r(al, bl, 1), mul_r(al, bh, 1), mul_r(ah, bl, 1), mul_r(ah, bh, 1));
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
static void s_div(double al, double ah, double bl, double bh, double *zl, double *zh)
{
  if (empty(al, ah) || empty(bl, bh)) { *zl = *zh = NAN; return; }
  div1(al, ah, bl, bh, zl, zh);
}
static void s_neg(double al, double ah, double *zl, double *zh)
{
  if (empty(al, ah)) { *zl = *zh = NAN; return; }
  *zl = -ah; *zh = -al;
}
static void s_sqr(double al, double ah, double *zl, double *zh)
{
  if (empty(al, ah)) { *zl = *zh = NAN; return; }
  if (al >= 0) { *zl = mul_r(al, al, 0); *zh = mul_r(ah, ah, 1); }
  else if (ah <= 0) { *zl = mul_r(ah, ah, 0); *zh = mul_r(al, al, 1); }
  else { double u = mul_r(al, al, 1), v = mul_r(ah, ah, 1); *zl = 0.0; *zh = u > v ? u : v; }
}
static void s_recip(double al, double ah, double *zl, double *zh)
{
  if (empty(al, ah)) { *zl = *zh = NAN; return; }
  div1(1.0, 1.0, al, ah, zl, zh);
}

/* ---- four at a time: each loop returns how many it did (a multiple of 4); the lanes marked in fb, empties among
   them, are redone by the scalar function from the loaded values (so zlo may be alo) ---- */
#if defined(__x86_64__)
#define VLOOP2(SC, ...)                                                                                       \
  size_t i = 0;                                                                                               \
  for (; i + 4 <= n; i += 4) {                                                                                \
    __m256d al = _mm256_loadu_pd(alo + i), ah = _mm256_loadu_pd(ahi + i);                                     \
    __m256d bl = _mm256_loadu_pd(blo + i), bh = _mm256_loadu_pd(bhi + i), zl, zh;                             \
    __m256d fb = _mm256_or_pd(vempty(al, ah), vempty(bl, bh));                                                \
    __VA_ARGS__                                                                                               \
    int m = _mm256_movemask_pd(fb);                                                                           \
    if (m) {                                                                                                  \
      double x[4][4], y[2][4];                                                                                \
      _mm256_storeu_pd(x[0], al); _mm256_storeu_pd(x[1], ah); _mm256_storeu_pd(x[2], bl); _mm256_storeu_pd(x[3], bh); \
      _mm256_storeu_pd(y[0], zl); _mm256_storeu_pd(y[1], zh);                                                 \
      for (; m; m &= m - 1) { int k = __builtin_ctz(m); SC(x[0][k], x[1][k], x[2][k], x[3][k], &y[0][k], &y[1][k]); } \
      zl = _mm256_loadu_pd(y[0]); zh = _mm256_loadu_pd(y[1]);                                                 \
    }                                                                                                         \
    _mm256_storeu_pd(zlo + i, zl); _mm256_storeu_pd(zhi + i, zh);                                             \
  }                                                                                                           \
  return i;
#define VLOOP1(SC, ...)                                                                                       \
  size_t i = 0;                                                                                               \
  for (; i + 4 <= n; i += 4) {                                                                                \
    __m256d al = _mm256_loadu_pd(lo + i), ah = _mm256_loadu_pd(hi + i), zl, zh, fb = vempty(al, ah);         \
    __VA_ARGS__                                                                                               \
    int m = _mm256_movemask_pd(fb);                                                                           \
    if (m) {                                                                                                  \
      double x[2][4], y[2][4];                                                                                \
      _mm256_storeu_pd(x[0], al); _mm256_storeu_pd(x[1], ah); _mm256_storeu_pd(y[0], zl); _mm256_storeu_pd(y[1], zh); \
      for (; m; m &= m - 1) { int k = __builtin_ctz(m); SC(x[0][k], x[1][k], &y[0][k], &y[1][k]); }          \
      zl = _mm256_loadu_pd(y[0]); zh = _mm256_loadu_pd(y[1]);                                                 \
    }                                                                                                         \
    _mm256_storeu_pd(ylo + i, zl); _mm256_storeu_pd(yhi + i, zh);                                             \
  }                                                                                                           \
  return i;
#define ARGS2 const double *alo, const double *ahi, const double *blo, const double *bhi, double *zlo, double *zhi, size_t n
#define ARGS1 const double *lo, const double *hi, double *ylo, double *yhi, size_t n
TGT static inline __m256d vneg(__m256d x) { return _mm256_xor_pd(x, _mm256_set1_pd(-0.0)); }
TGT static size_t v_add(ARGS2) { VLOOP2(s_add, zl = vadd_r(al, bl, 0); zh = vadd_r(ah, bh, 1);) }
TGT static size_t v_sub(ARGS2) { VLOOP2(s_sub, zl = vadd_r(al, vneg(bh), 0); zh = vadd_r(ah, vneg(bl), 1);) }
TGT static size_t v_mul(ARGS2)
{
  VLOOP2(s_mul,
         zl = _mm256_min_pd(_mm256_min_pd(vmul_r(al, bl, 0, &fb), vmul_r(al, bh, 0, &fb)),
                            _mm256_min_pd(vmul_r(ah, bl, 0, &fb), vmul_r(ah, bh, 0, &fb)));
         zh = _mm256_max_pd(_mm256_max_pd(vmul_r(al, bl, 1, &fb), vmul_r(al, bh, 1, &fb)),
                            _mm256_max_pd(vmul_r(ah, bl, 1, &fb), vmul_r(ah, bh, 1, &fb)));)
}
/* the table for B > 0 or B < 0, where each end is one quotient: for B > 0 the lower end is al over bh when A >= 0,
   else over bl, and the upper end ah over bh when A <= 0 (and not A >= 0), else over bl; for B < 0 the lower end is
   ah over bl when A <= 0, else over bh, and the upper al over bl when A >= 0, else over bh. B touching or holding 0
   goes to the scalar table. */
TGT static inline void vdiv1(__m256d al, __m256d ah, __m256d bl, __m256d bh, __m256d *zl, __m256d *zh, __m256d *fb)
{
  const __m256d z = _mm256_setzero_pd();
  __m256d P = _mm256_cmp_pd(bl, z, _CMP_GT_OQ), N = _mm256_cmp_pd(bh, z, _CMP_LT_OQ);
  __m256d ge = _mm256_cmp_pd(al, z, _CMP_GE_OQ), le = _mm256_andnot_pd(ge, _mm256_cmp_pd(ah, z, _CMP_LE_OQ));
  *fb = _mm256_or_pd(*fb, _mm256_andnot_pd(_mm256_or_pd(P, N), _mm256_castsi256_pd(_mm256_set1_epi64x(-1))));
  __m256d nl = _mm256_blendv_pd(ah, al, P), nh = _mm256_blendv_pd(al, ah, P);
#if IVAL_PLANT_ARITH == 9   /* B > 0, A >= 0: the lower end over bl */
  __m256d dl = _mm256_blendv_pd(_mm256_blendv_pd(bh, bl, le), bl, P);
#else
  __m256d dl = _mm256_blendv_pd(_mm256_blendv_pd(bh, bl, le), _mm256_blendv_pd(bl, bh, ge), P);
#endif
  __m256d dh = _mm256_blendv_pd(_mm256_blendv_pd(bh, bl, ge), _mm256_blendv_pd(bl, bh, le), P);
  dl = _mm256_blendv_pd(_mm256_set1_pd(1.0), dl, _mm256_or_pd(P, N));   /* lanes for the scalar table: no 0 divisor */
  dh = _mm256_blendv_pd(_mm256_set1_pd(1.0), dh, _mm256_or_pd(P, N));
  *zl = vdiv_r(nl, dl, 0, fb); *zh = vdiv_r(nh, dh, 1, fb);
}
TGT static size_t v_div(ARGS2) { VLOOP2(s_div, vdiv1(al, ah, bl, bh, &zl, &zh, &fb);) }
TGT static size_t v_neg(ARGS1) { VLOOP1(s_neg, zl = vneg(ah); zh = vneg(al);) }
TGT static size_t v_sqr(ARGS1)
{
  VLOOP1(s_sqr,
         const __m256d z = _mm256_setzero_pd();
         __m256d ge = _mm256_cmp_pd(al, z, _CMP_GE_OQ), le = _mm256_andnot_pd(ge, _mm256_cmp_pd(ah, z, _CMP_LE_OQ));
         __m256d ld = vmul_r(al, al, 0, &fb), hd = vmul_r(ah, ah, 0, &fb), lu = vmul_r(al, al, 1, &fb), hu = vmul_r(ah, ah, 1, &fb);
         zl = _mm256_blendv_pd(_mm256_blendv_pd(z, hd, le), ld, ge);
         zh = _mm256_blendv_pd(_mm256_blendv_pd(_mm256_max_pd(lu, hu), lu, le), hu, ge);)
}
TGT static size_t v_recip(ARGS1)
{
  VLOOP1(s_recip, const __m256d one = _mm256_set1_pd(1.0); vdiv1(one, one, al, ah, &zl, &zh, &fb);)
}
#define VEC(f, ...) (have_vec() ? f(__VA_ARGS__) : 0)
#else
#define VEC(f, ...) 0
#endif

/* Every call runs rounding to nearest with flush-to-zero and denormals-are-zero off (a program built with
   -ffast-math starts with both on, and either breaks the transformations), and gives the caller's state back. */
static void enter(fenv_t *env)
{
  fegetenv(env);
  fesetround(FE_TONEAREST);
#if defined(__x86_64__) && IVAL_PLANT_ARITH != 10   /* 10: the flush modes left as the caller set them */
  _mm_setcsr(_mm_getcsr() & ~0x8040u);
#endif
}

void ival_add(const double *alo, const double *ahi, const double *blo, const double *bhi, double *zlo, double *zhi,
              size_t n)
{
  fenv_t env; enter(&env);
  for (size_t i = VEC(v_add, alo, ahi, blo, bhi, zlo, zhi, n); i < n; i++) s_add(alo[i], ahi[i], blo[i], bhi[i], zlo + i, zhi + i);
  fesetenv(&env);
}

void ival_sub(const double *alo, const double *ahi, const double *blo, const double *bhi, double *zlo, double *zhi,
              size_t n)
{
  fenv_t env; enter(&env);
  for (size_t i = VEC(v_sub, alo, ahi, blo, bhi, zlo, zhi, n); i < n; i++) s_sub(alo[i], ahi[i], blo[i], bhi[i], zlo + i, zhi + i);
  fesetenv(&env);
}

void ival_mul(const double *alo, const double *ahi, const double *blo, const double *bhi, double *zlo, double *zhi,
              size_t n)
{
  fenv_t env; enter(&env);
  for (size_t i = VEC(v_mul, alo, ahi, blo, bhi, zlo, zhi, n); i < n; i++) s_mul(alo[i], ahi[i], blo[i], bhi[i], zlo + i, zhi + i);
  fesetenv(&env);
}

void ival_div(const double *alo, const double *ahi, const double *blo, const double *bhi, double *zlo, double *zhi,
              size_t n)
{
  fenv_t env; enter(&env);
  for (size_t i = VEC(v_div, alo, ahi, blo, bhi, zlo, zhi, n); i < n; i++) s_div(alo[i], ahi[i], blo[i], bhi[i], zlo + i, zhi + i);
  fesetenv(&env);
}

void ival_neg(const double *lo, const double *hi, double *ylo, double *yhi, size_t n)
{
  fenv_t env; enter(&env);
  for (size_t i = VEC(v_neg, lo, hi, ylo, yhi, n); i < n; i++) s_neg(lo[i], hi[i], ylo + i, yhi + i);
  fesetenv(&env);
}

void ival_sqr(const double *lo, const double *hi, double *ylo, double *yhi, size_t n)
{
  fenv_t env; enter(&env);
  for (size_t i = VEC(v_sqr, lo, hi, ylo, yhi, n); i < n; i++) s_sqr(lo[i], hi[i], ylo + i, yhi + i);
  fesetenv(&env);
}

void ival_recip(const double *lo, const double *hi, double *ylo, double *yhi, size_t n)
{
  fenv_t env; enter(&env);
  for (size_t i = VEC(v_recip, lo, hi, ylo, yhi, n); i < n; i++) s_recip(lo[i], hi[i], ylo + i, yhi + i);
  fesetenv(&env);
}
