/* ival-arith.c: interval arithmetic, each result the tightest enclosure (ival.h).

   Arrays go in blocks of 256 intervals: every lower end of the block computed rounding down, then every upper end
   rounding up, so the rounding mode changes twice a block, not per operation (on x86-64 with AVX2 and FMA four lanes
   at a time, setting only the SSE unit's mode; elsewhere plain C and fesetround). Rounding down and up are monotone,
   so a product's lower end is the least of the four corner products rounded down, and so on.

   The scalar code is a second algorithm, the reference the passes are checked against and the one they hand the
   awkward lanes to (empty operands, divisors touching or holding 0). It computes each bound rounding to nearest, and
   the exact rounding error, from an error-free transformation, says whether the bound moves one ulp:
     - a + b: TwoSum's error term, exact for any operands when the sum does not overflow;
     - a * b: fma(a, b, -p), exact when |p| >= 2^-969 (the error then is a binary64 value);
     - a / b: the remainder fma(-q, b, a), exact when a, b and q are at least 2^-960 in magnitude; the sign of r / b is
       the sign of the true quotient minus q.
   Where those conditions fail (results near the underflow range), it rounds down or up instead. An overflow rounds to
   DBL_MAX one way and to the infinity the other. It was the vector code first (2026-10-08), until arith-bench showed
   the mode set per block 3 to 13 times faster.

   Intervals as IEEE 1788 has them: [NaN, NaN] is empty, as is any input with lo > hi, a NaN end, lo = +inf or
   hi = -inf; [-inf, +inf] is the whole line; zeros are unsigned, and a zero end comes out as +0. At interval ends,
   0 * inf is 0 (the end stands for a limit). Division follows 1788's case table, by where 0 lies in the divisor (a
   divisor that is exactly [0, 0] gives the empty interval). Flush-to-zero and denormals-are-zero are off for the call;
   the C rounding mode, those modes and the flags are left as they were. */
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

/* ---- the arrays: blocks of 256 intervals, every lower end computed rounding down, then every upper end rounding up
   (the rounding mode set twice a block, not per operation). The passes write to a buffer, so the result may overwrite
   an operand; a last pass, rounding to nearest, copies it out, sets zero ends to +0, and redoes by the scalar code
   above the lanes the passes do not cover: empty operands, and for division a divisor touching or holding 0.
   0 * inf, which IEEE makes NaN, is 0 at interval ends: a product with a zero factor is set to 0. ---- */
#define BLK 256
typedef void (*pass_fn)(const double *, const double *, const double *, const double *, double *, size_t);
typedef void (*fix_fn)(const double *, const double *, const double *, const double *, const double *, const double *,
                       double *, double *, size_t);
struct passes { pass_fn lo, hi; fix_fn fix; };
static double canon(double x) { return x == 0 ? 0.0 : x; }
static double mn(double a, double b) { return a < b ? a : b; }
static double mx(double a, double b) { return a > b ? a : b; }
#if IVAL_PLANT_ARITH == 12   /* 0 * inf left to IEEE in the portable passes */
static double pz(double a, double b) { return a * b; }
#else
static double pz(double a, double b) { double p = a * b; return ((a == 0) | (b == 0)) ? 0.0 : p; }   /* p first: vectorizes */
#endif

/* one lane of each pass, in whatever rounding mode is set: the portable passes and the vector passes' tails */
#define LO_add(al, ah, bl, bh) ((al) + (bl))
#define HI_add(al, ah, bl, bh) ((ah) + (bh))
#define LO_sub(al, ah, bl, bh) ((al) - (bh))
#define HI_sub(al, ah, bl, bh) ((ah) - (bl))
#define LO_mul(al, ah, bl, bh) mn(mn(pz(al, bl), pz(al, bh)), mn(pz(ah, bl), pz(ah, bh)))
#define HI_mul(al, ah, bl, bh) mx(mx(pz(al, bl), pz(al, bh)), mx(pz(ah, bl), pz(ah, bh)))
/* 1788's table for a divisor without 0: for B > 0 the lower end is al over bh when A >= 0, else over bl, and the upper
   end ah over bh when A <= 0 (and not A >= 0), else over bl; for B < 0 the lower end is ah over bl when A <= 0, else
   over bh, and the upper al over bl when A >= 0, else over bh. Other divisors are left to the fix pass (a 1 in their
   place keeps the pass free of 0 / 0). */
static double lo_div(double al, double ah, double bl, double bh)
{
  if (bl > 0) return al / (al >= 0 ? bh : bl);
#if IVAL_PLANT_ARITH == 9   /* B < 0, A <= 0: the lower end over bh */
  if (bh < 0) return ah / bh;
#else
  if (bh < 0) return ah / (al < 0 && ah <= 0 ? bl : bh);
#endif
  return 1.0;
}
static double hi_div(double al, double ah, double bl, double bh)
{
  if (bl > 0) return ah / (al < 0 && ah <= 0 ? bh : bl);
  if (bh < 0) return al / (al >= 0 ? bl : bh);
  return 1.0;
}
#define LO_div(al, ah, bl, bh) lo_div(al, ah, bl, bh)
#define HI_div(al, ah, bl, bh) hi_div(al, ah, bl, bh)
#define LO_recip(al, ah, bl, bh) lo_div(1.0, 1.0, al, ah)
#define HI_recip(al, ah, bl, bh) hi_div(1.0, 1.0, al, ah)
#define LO_sqr(al, ah, bl, bh) ((al) >= 0 ? (al) * (al) : (ah) <= 0 ? (ah) * (ah) : 0.0)
#define HI_sqr(al, ah, bl, bh) ((al) >= 0 ? (ah) * (ah) : (ah) <= 0 ? (al) * (al) : mx((al) * (al), (ah) * (ah)))
#define LO_neg(al, ah, bl, bh) (-(ah))
#define HI_neg(al, ah, bl, bh) (-(al))
/* the lanes for the scalar code, without branches (| not ||), so a block's scan vectorizes */
#define EMPTYV(lo, hi) (!((lo) <= (hi)) | ((lo) == INFINITY) | ((hi) == -INFINITY))
#define FB_add(al, ah, bl, bh) (EMPTYV(al, ah) | EMPTYV(bl, bh))
#define FB_sub FB_add
#define FB_mul FB_add
#define FB_div(al, ah, bl, bh) (FB_add(al, ah, bl, bh) | !(((bl) > 0) | ((bh) < 0)))
#define FB_neg(al, ah, bl, bh) EMPTYV(al, ah)
#define FB_sqr FB_neg
#define FB_recip(al, ah, bl, bh) (EMPTYV(al, ah) | !(((al) > 0) | ((ah) < 0)))
/* the one-interval scalar functions in the two-interval form the fix pass calls */
static void s_neg2(double al, double ah, double bl, double bh, double *zl, double *zh) { (void)bl; (void)bh; s_neg(al, ah, zl, zh); }
static void s_sqr2(double al, double ah, double bl, double bh, double *zl, double *zh) { (void)bl; (void)bh; s_sqr(al, ah, zl, zh); }
static void s_recip2(double al, double ah, double bl, double bh, double *zl, double *zh) { (void)bl; (void)bh; s_recip(al, ah, zl, zh); }

#define NOI __attribute__((noinline))
static void copy_out(const double *restrict tl, const double *restrict th, double *restrict zl, double *restrict zh, size_t m)
{
  size_t i = 0;
  for (; i + 4 <= m; i += 4) for (int k = 0; k < 4; k++) { zl[i + k] = canon(tl[i + k]); zh[i + k] = canon(th[i + k]); }
  for (; i < m; i++) { zl[i] = canon(tl[i]); zh[i] = canon(th[i]); }
}
#define C_PASSES(op, sfn)                                                                                     \
  NOI static void clo_##op(const double *al, const double *ah, const double *bl, const double *bh, double *restrict t, size_t m) \
  { (void)al; (void)ah; (void)bl; (void)bh; size_t i = 0;                                                     \
    for (; i + 4 <= m; i += 4) for (int k = 0; k < 4; k++) t[i + k] = LO_##op(al[i + k], ah[i + k], bl[i + k], bh[i + k]); \
    for (; i < m; i++) t[i] = LO_##op(al[i], ah[i], bl[i], bh[i]); }                                          \
  NOI static void chi_##op(const double *al, const double *ah, const double *bl, const double *bh, double *restrict t, size_t m) \
  { (void)al; (void)ah; (void)bl; (void)bh; size_t i = 0;                                                     \
    for (; i + 4 <= m; i += 4) for (int k = 0; k < 4; k++) t[i + k] = HI_##op(al[i + k], ah[i + k], bl[i + k], bh[i + k]); \
    for (; i < m; i++) t[i] = HI_##op(al[i], ah[i], bl[i], bh[i]); }                                          \
  static void cfix_##op(const double *al, const double *ah, const double *bl, const double *bh, const double *tl, \
                        const double *th, double *zl, double *zh, size_t m)                                   \
  {                                                                                                           \
    double *wl = (double *)tl, *wh = (double *)th;   /* the block buffer, the fix's to write */             \
    int a0 = 0, a1 = 0, a2 = 0, a3 = 0;   /* four accumulators, so the scan vectorizes at -O2 */             \
    size_t j = 0;                                                                                             \
    for (; j + 4 <= m; j += 4) {                                                                              \
      a0 |= FB_##op(al[j], ah[j], bl[j], bh[j]); a1 |= FB_##op(al[j + 1], ah[j + 1], bl[j + 1], bh[j + 1]);    \
      a2 |= FB_##op(al[j + 2], ah[j + 2], bl[j + 2], bh[j + 2]); a3 |= FB_##op(al[j + 3], ah[j + 3], bl[j + 3], bh[j + 3]); \
    }                                                                                                         \
    for (; j < m; j++) a0 |= FB_##op(al[j], ah[j], bl[j], bh[j]);                                             \
    if (a0 | a1 | a2 | a3)                                                                                    \
      for (size_t i = 0; i < m; i++)                                                                          \
        if (FB_##op(al[i], ah[i], bl[i], bh[i])) sfn(al[i], ah[i], bl[i], bh[i], wl + i, wh + i);             \
    copy_out(wl, wh, zl, zh, m);                                                                              \
  }                                                                                                           \
  static const struct passes c_##op = { clo_##op, chi_##op, cfix_##op };
C_PASSES(add, s_add) C_PASSES(sub, s_sub) C_PASSES(mul, s_mul) C_PASSES(div, s_div)
C_PASSES(neg, s_neg2) C_PASSES(sqr, s_sqr2) C_PASSES(recip, s_recip2)

#if defined(__x86_64__)
#include <immintrin.h>
#define TGT __attribute__((target("avx2,fma")))
static int vec_ok = -1;
static int have_vec(void)
{
  if (vec_ok < 0) { __builtin_cpu_init(); vec_ok = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma"); }
  return vec_ok;
}
TGT static inline __m256d vempty(__m256d lo, __m256d hi)
{
  __m256d bad = _mm256_cmp_pd(lo, hi, _CMP_NLE_UQ);   /* !(lo <= hi), NaN included */
  bad = _mm256_or_pd(bad, _mm256_cmp_pd(lo, _mm256_set1_pd(INFINITY), _CMP_EQ_OQ));
  return _mm256_or_pd(bad, _mm256_cmp_pd(hi, _mm256_set1_pd(-INFINITY), _CMP_EQ_OQ));
}
#define Z _mm256_setzero_pd()
#define GT0(x) _mm256_cmp_pd(x, Z, _CMP_GT_OQ)
#define LT0(x) _mm256_cmp_pd(x, Z, _CMP_LT_OQ)
#define GE0(x) _mm256_cmp_pd(x, Z, _CMP_GE_OQ)
#define LE0(x) _mm256_cmp_pd(x, Z, _CMP_LE_OQ)
#define EQ0(x) _mm256_cmp_pd(x, Z, _CMP_EQ_OQ)
#define SEL(m, x, y) _mm256_blendv_pd(y, x, m)   /* m ? x : y */
TGT static inline __m256d vpz(__m256d a, __m256d b, __m256d az, __m256d bz)
{
#if IVAL_PLANT_ARITH == 7   /* 0 * inf left to IEEE in the vector passes */
  (void)az; (void)bz; return _mm256_mul_pd(a, b);
#else
  return _mm256_andnot_pd(_mm256_or_pd(az, bz), _mm256_mul_pd(a, b));
#endif
}
TGT static inline __m256d vdlo(__m256d al, __m256d ah, __m256d bl, __m256d bh)
{
  __m256d P = GT0(bl), N = LT0(bh), ge = GE0(al), le = _mm256_andnot_pd(ge, LE0(ah));
#if IVAL_PLANT_ARITH == 9
  __m256d d = SEL(P, SEL(ge, bh, bl), bh);
#else
  __m256d d = SEL(P, SEL(ge, bh, bl), SEL(le, bl, bh));
#endif
  return _mm256_div_pd(SEL(P, al, ah), SEL(_mm256_or_pd(P, N), d, _mm256_set1_pd(1.0)));
}
TGT static inline __m256d vdhi(__m256d al, __m256d ah, __m256d bl, __m256d bh)
{
  __m256d P = GT0(bl), N = LT0(bh), ge = GE0(al), le = _mm256_andnot_pd(ge, LE0(ah));
  __m256d d = SEL(P, SEL(le, bh, bl), SEL(ge, bl, bh));
  return _mm256_div_pd(SEL(P, ah, al), SEL(_mm256_or_pd(P, N), d, _mm256_set1_pd(1.0)));
}
#define V_LO_add(al, ah, bl, bh) _mm256_add_pd(al, bl)
#define V_HI_add(al, ah, bl, bh) _mm256_add_pd(ah, bh)
#define V_LO_sub(al, ah, bl, bh) _mm256_sub_pd(al, bh)
#define V_HI_sub(al, ah, bl, bh) _mm256_sub_pd(ah, bl)
#define V_PRODS                                                                                               \
  __m256d azl = EQ0(al), azh = EQ0(ah), bzl = EQ0(bl), bzh = EQ0(bh);                                         \
  __m256d p0 = vpz(al, bl, azl, bzl), p1 = vpz(al, bh, azl, bzh), p2 = vpz(ah, bl, azh, bzl), p3 = vpz(ah, bh, azh, bzh);
#define V_LO_mul(al, ah, bl, bh) _mm256_min_pd(_mm256_min_pd(p0, p1), _mm256_min_pd(p2, p3))
#define V_HI_mul(al, ah, bl, bh) _mm256_max_pd(_mm256_max_pd(p0, p1), _mm256_max_pd(p2, p3))
#define V_LO_div(al, ah, bl, bh) vdlo(al, ah, bl, bh)
#define V_HI_div(al, ah, bl, bh) vdhi(al, ah, bl, bh)
#define V_LO_recip(al, ah, bl, bh) vdlo(_mm256_set1_pd(1.0), _mm256_set1_pd(1.0), al, ah)
#define V_HI_recip(al, ah, bl, bh) vdhi(_mm256_set1_pd(1.0), _mm256_set1_pd(1.0), al, ah)
#define V_LO_sqr(al, ah, bl, bh) SEL(GE0(al), _mm256_mul_pd(al, al), SEL(LE0(ah), _mm256_mul_pd(ah, ah), Z))
#define V_HI_sqr(al, ah, bl, bh)                                                                              \
  SEL(GE0(al), _mm256_mul_pd(ah, ah), SEL(LE0(ah), _mm256_mul_pd(al, al), _mm256_max_pd(_mm256_mul_pd(al, al), _mm256_mul_pd(ah, ah))))
#define V_LO_neg(al, ah, bl, bh) _mm256_xor_pd(ah, _mm256_set1_pd(-0.0))
#define V_HI_neg(al, ah, bl, bh) _mm256_xor_pd(al, _mm256_set1_pd(-0.0))
#define V_FB_add(al, ah, bl, bh) _mm256_or_pd(vempty(al, ah), vempty(bl, bh))
#define V_FB_sub V_FB_add
#define V_FB_mul V_FB_add
#define V_FB_div(al, ah, bl, bh) _mm256_or_pd(V_FB_add(al, ah, bl, bh), _mm256_xor_pd(_mm256_or_pd(GT0(bl), LT0(bh)), _mm256_castsi256_pd(_mm256_set1_epi64x(-1))))
#define V_FB_neg(al, ah, bl, bh) vempty(al, ah)
#define V_FB_sqr V_FB_neg
#define V_FB_recip(al, ah, bl, bh) _mm256_or_pd(vempty(al, ah), _mm256_xor_pd(_mm256_or_pd(GT0(al), LT0(ah)), _mm256_castsi256_pd(_mm256_set1_epi64x(-1))))
#define LOAD4 __m256d al = _mm256_loadu_pd(a0 + i), ah = _mm256_loadu_pd(a1 + i), bl = _mm256_loadu_pd(b0 + i), bh = _mm256_loadu_pd(b1 + i); \
  (void)al; (void)ah; (void)bl; (void)bh;
#define V_PASSES(op, sfn, PRE)                                                                                \
  TGT NOI static void vlo_##op(const double *a0, const double *a1, const double *b0, const double *b1, double *t, size_t m) \
  {                                                                                                           \
    size_t i = 0;                                                                                             \
    for (; i + 4 <= m; i += 4) { LOAD4 PRE _mm256_storeu_pd(t + i, V_LO_##op(al, ah, bl, bh)); } \
    for (; i < m; i++) t[i] = LO_##op(a0[i], a1[i], b0[i], b1[i]);                                            \
  }                                                                                                           \
  TGT NOI static void vhi_##op(const double *a0, const double *a1, const double *b0, const double *b1, double *t, size_t m) \
  {                                                                                                           \
    size_t i = 0;                                                                                             \
    for (; i + 4 <= m; i += 4) { LOAD4 PRE _mm256_storeu_pd(t + i, V_HI_##op(al, ah, bl, bh)); } \
    for (; i < m; i++) t[i] = HI_##op(a0[i], a1[i], b0[i], b1[i]);                                            \
  }                                                                                                           \
  TGT static void vfix_##op(const double *a0, const double *a1, const double *b0, const double *b1, const double *tl, \
                            const double *th, double *zl, double *zh, size_t m)                               \
  {                                                                                                           \
    const __m256d sg = _mm256_set1_pd(-0.0);                                                                  \
    size_t i = 0;                                                                                             \
    for (; i + 4 <= m; i += 4) {                                                                              \
      LOAD4                                                                                                   \
      __m256d x = _mm256_loadu_pd(tl + i), y = _mm256_loadu_pd(th + i);                                       \
      int f = _mm256_movemask_pd(V_FB_##op(al, ah, bl, bh));                                                  \
      if (f) {                                                                                                \
        double u[4], v[4];                                                                                    \
        _mm256_storeu_pd(u, x); _mm256_storeu_pd(v, y);                                                       \
        for (; f; f &= f - 1) { int k = __builtin_ctz(f); sfn(a0[i + k], a1[i + k], b0[i + k], b1[i + k], &u[k], &v[k]); } \
        x = _mm256_loadu_pd(u); y = _mm256_loadu_pd(v);                                                       \
      }                                                                                                       \
      if (IVAL_PLANT_ARITH != 8) {   /* 8: zero ends left as the passes give them */                         \
        x = _mm256_andnot_pd(_mm256_and_pd(EQ0(x), sg), x); y = _mm256_andnot_pd(_mm256_and_pd(EQ0(y), sg), y); \
      }                                                                                                       \
      _mm256_storeu_pd(zl + i, x); _mm256_storeu_pd(zh + i, y);                                               \
    }                                                                                                         \
    cfix_##op(a0 + i, a1 + i, b0 + i, b1 + i, tl + i, th + i, zl + i, zh + i, m - i);                         \
  }                                                                                                           \
  static const struct passes v_##op = { vlo_##op, vhi_##op, vfix_##op };
V_PASSES(add, s_add, ) V_PASSES(sub, s_sub, ) V_PASSES(mul, s_mul, V_PRODS) V_PASSES(div, s_div, )
V_PASSES(neg, s_neg2, ) V_PASSES(sqr, s_sqr2, ) V_PASSES(recip, s_recip2, )
#endif

/* Flush-to-zero and denormals-are-zero off for the call (x86-64: MXCSR's FZ and DAZ; aarch64: FPCR.FZ); fesetenv
   gives them back. A program built with -ffast-math starts with them on, and either breaks the enclosures. */
static void flush_off(void)
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

/* Which code runs: 0 the best this CPU has, 1 the portable passes, 2 the scalar code alone (the reference). Not API:
   arith-check sets it to compare them. */
int ival__arith_path;

static void run(const struct passes *c, const struct passes *v, void (*sfn)(double, double, double, double, double *, double *),
                const double *a0, const double *a1, const double *b0, const double *b1, double *zl, double *zh, size_t n)
{
  fenv_t env;
  fegetenv(&env);
  fesetround(FE_TONEAREST);
  const struct passes *p = c;
  flush_off();
#if defined(__x86_64__)
  unsigned rn = _mm_getcsr();
  if (ival__arith_path == 0 && have_vec()) p = v;
#else
  (void)v;
#endif
  if (ival__arith_path == 2) {
    for (size_t i = 0; i < n; i++) { double x, y; sfn(a0[i], a1[i], b0[i], b1[i], &x, &y); zl[i] = canon(x); zh[i] = canon(y); }
    fesetenv(&env);
    return;
  }
  double tl[BLK], th[BLK];
  for (size_t i = 0; i < n; i += BLK) {
    size_t m = n - i < BLK ? n - i : BLK;
#if defined(__x86_64__)
    if (p == v) {   /* only the SSE unit's mode: the vector passes use nothing else */
      _mm_setcsr((rn & ~0x6000u) | 0x2000u); p->lo(a0 + i, a1 + i, b0 + i, b1 + i, tl, m);
#if IVAL_PLANT_ARITH == 11   /* the upper ends rounded down too */
      _mm_setcsr((rn & ~0x6000u) | 0x2000u); p->hi(a0 + i, a1 + i, b0 + i, b1 + i, th, m);
#else
      _mm_setcsr((rn & ~0x6000u) | 0x4000u); p->hi(a0 + i, a1 + i, b0 + i, b1 + i, th, m);
#endif
      _mm_setcsr(rn);
    } else
#endif
    {
      fesetround(FE_DOWNWARD); p->lo(a0 + i, a1 + i, b0 + i, b1 + i, tl, m);
      fesetround(FE_UPWARD); p->hi(a0 + i, a1 + i, b0 + i, b1 + i, th, m);
      fesetround(FE_TONEAREST);
    }
    p->fix(a0 + i, a1 + i, b0 + i, b1 + i, tl, th, zl + i, zh + i, m);
  }
  fesetenv(&env);
}
#if !defined(__x86_64__)
#define V(op) 0
#else
#define V(op) &v_##op
#endif

void ival_add(const double *alo, const double *ahi, const double *blo, const double *bhi, double *zlo, double *zhi,
              size_t n)
{ run(&c_add, V(add), s_add, alo, ahi, blo, bhi, zlo, zhi, n); }
void ival_sub(const double *alo, const double *ahi, const double *blo, const double *bhi, double *zlo, double *zhi,
              size_t n)
{ run(&c_sub, V(sub), s_sub, alo, ahi, blo, bhi, zlo, zhi, n); }
void ival_mul(const double *alo, const double *ahi, const double *blo, const double *bhi, double *zlo, double *zhi,
              size_t n)
{ run(&c_mul, V(mul), s_mul, alo, ahi, blo, bhi, zlo, zhi, n); }
void ival_div(const double *alo, const double *ahi, const double *blo, const double *bhi, double *zlo, double *zhi,
              size_t n)
{ run(&c_div, V(div), s_div, alo, ahi, blo, bhi, zlo, zhi, n); }
void ival_neg(const double *lo, const double *hi, double *ylo, double *yhi, size_t n)
{ run(&c_neg, V(neg), s_neg2, lo, hi, lo, hi, ylo, yhi, n); }
void ival_sqr(const double *lo, const double *hi, double *ylo, double *yhi, size_t n)
{ run(&c_sqr, V(sqr), s_sqr2, lo, hi, lo, hi, ylo, yhi, n); }
void ival_recip(const double *lo, const double *hi, double *ylo, double *yhi, size_t n)
{ run(&c_recip, V(recip), s_recip2, lo, hi, lo, hi, ylo, yhi, n); }

/* ---- fma(A, B, C): the least and greatest of a * b + c over the box. The product's extremes are at corners and c's
   least goes with the product's least, so the lower end is the least of the four corner fmas with clo, each rounded
   down (one rounding: rounding is monotone, so the least rounded is the least rounded down), and the upper end the
   greatest of those with chi, rounded up. At ends a zero factor makes the product 0, so the term is c. A corner whose
   infinite product meets an infinite c of the other sign (inf - inf, NaN) is never the extreme on that side, since
   the least product is then finite or -inf: the lower end takes such a term as +inf, the upper end as -inf. Empty
   operands give the empty interval. The same passes, per block, as the arithmetic above. ---- */
static double fterm(double a, double b, double c, int up)
{
  double t = fma(a, b, c);
#if IVAL_PLANT_ARITH != 13   /* 13: 0 * inf left to IEEE in fma */
  t = ((a == 0) | (b == 0)) ? c : t;
#endif
#if IVAL_PLANT_ARITH == 14   /* 14: inf - inf left as NaN */
  return t;
#else
  return t != t ? (up ? -INFINITY : INFINITY) : t;
#endif
}
#define FMA_ARGS const double *a0, const double *a1, const double *b0, const double *b1, const double *c0, const double *c1
NOI static void clo_fma(FMA_ARGS, double *restrict t, size_t m)
{
  (void)c1;
  for (size_t i = 0; i < m; i++)
    t[i] = mn(mn(fterm(a0[i], b0[i], c0[i], 0), fterm(a0[i], b1[i], c0[i], 0)),
              mn(fterm(a1[i], b0[i], c0[i], 0), fterm(a1[i], b1[i], c0[i], 0)));
}
NOI static void chi_fma(FMA_ARGS, double *restrict t, size_t m)
{
  (void)c0;
#if IVAL_PLANT_ARITH == 15   /* 15: the upper end with clo */
  c1 = c0;
#endif
  for (size_t i = 0; i < m; i++)
    t[i] = mx(mx(fterm(a0[i], b0[i], c1[i], 1), fterm(a0[i], b1[i], c1[i], 1)),
              mx(fterm(a1[i], b0[i], c1[i], 1), fterm(a1[i], b1[i], c1[i], 1)));
}
#if defined(__x86_64__)
TGT static inline __m256d vfterm(__m256d a, __m256d b, __m256d c, __m256d zf, __m256d nanv)
{
  __m256d t = _mm256_fmadd_pd(a, b, c);
#if IVAL_PLANT_ARITH != 13
  t = SEL(zf, c, t);
#else
  (void)zf;
#endif
#if IVAL_PLANT_ARITH == 14
  (void)nanv; return t;
#else
  return SEL(_mm256_cmp_pd(t, t, _CMP_UNORD_Q), nanv, t);
#endif
}
#define LOAD6                                                                                                 \
  __m256d al = _mm256_loadu_pd(a0 + i), ah = _mm256_loadu_pd(a1 + i), bl = _mm256_loadu_pd(b0 + i);           \
  __m256d bh = _mm256_loadu_pd(b1 + i), azl = EQ0(al), azh = EQ0(ah), bzl = EQ0(bl), bzh = EQ0(bh);
TGT NOI static void vlo_fma(FMA_ARGS, double *restrict t, size_t m)
{
  const __m256d inf = _mm256_set1_pd(INFINITY);
  size_t i = 0;
  for (; i + 4 <= m; i += 4) {
    LOAD6 __m256d c = _mm256_loadu_pd(c0 + i);
    __m256d x = _mm256_min_pd(vfterm(al, bl, c, _mm256_or_pd(azl, bzl), inf), vfterm(al, bh, c, _mm256_or_pd(azl, bzh), inf));
    __m256d y = _mm256_min_pd(vfterm(ah, bl, c, _mm256_or_pd(azh, bzl), inf), vfterm(ah, bh, c, _mm256_or_pd(azh, bzh), inf));
    _mm256_storeu_pd(t + i, _mm256_min_pd(x, y));
  }
  clo_fma(a0 + i, a1 + i, b0 + i, b1 + i, c0 + i, c1 + i, t + i, m - i);
}
TGT NOI static void vhi_fma(FMA_ARGS, double *restrict t, size_t m)
{
#if IVAL_PLANT_ARITH == 15
  c1 = c0;
#endif
  const __m256d ninf = _mm256_set1_pd(-INFINITY);
  size_t i = 0;
  for (; i + 4 <= m; i += 4) {
    LOAD6 __m256d c = _mm256_loadu_pd(c1 + i);
    __m256d x = _mm256_max_pd(vfterm(al, bl, c, _mm256_or_pd(azl, bzl), ninf), vfterm(al, bh, c, _mm256_or_pd(azl, bzh), ninf));
    __m256d y = _mm256_max_pd(vfterm(ah, bl, c, _mm256_or_pd(azh, bzl), ninf), vfterm(ah, bh, c, _mm256_or_pd(azh, bzh), ninf));
    _mm256_storeu_pd(t + i, _mm256_max_pd(x, y));
  }
  chi_fma(a0 + i, a1 + i, b0 + i, b1 + i, c0 + i, c1 + i, t + i, m - i);
}
#endif

void ival_fma(const double *alo, const double *ahi, const double *blo, const double *bhi, const double *clo,
              const double *chi, double *zlo, double *zhi, size_t n)
{
  fenv_t env;
  fegetenv(&env);
  fesetround(FE_TONEAREST);
  void (*lo)(FMA_ARGS, double *restrict, size_t) = clo_fma, (*hi)(FMA_ARGS, double *restrict, size_t) = chi_fma;
  flush_off();
#if defined(__x86_64__)
  unsigned rn = _mm_getcsr();
  int v = ival__arith_path == 0 && have_vec();
  if (v) { lo = vlo_fma; hi = vhi_fma; }
#endif
  double tl[BLK], th[BLK];
  for (size_t i = 0; i < n; i += BLK) {
    size_t m = n - i < BLK ? n - i : BLK;
    const double *a0 = alo + i, *a1 = ahi + i, *b0 = blo + i, *b1 = bhi + i, *c0 = clo + i, *c1 = chi + i;
#if defined(__x86_64__)
    if (v) {
      _mm_setcsr((rn & ~0x6000u) | 0x2000u); lo(a0, a1, b0, b1, c0, c1, tl, m);
      _mm_setcsr((rn & ~0x6000u) | 0x4000u); hi(a0, a1, b0, b1, c0, c1, th, m);
      _mm_setcsr(rn);
    } else
#endif
    {
      fesetround(FE_DOWNWARD); lo(a0, a1, b0, b1, c0, c1, tl, m);
      fesetround(FE_UPWARD); hi(a0, a1, b0, b1, c0, c1, th, m);
      fesetround(FE_TONEAREST);
    }
    for (size_t k = 0; k < m; k++)
      if (empty(a0[k], a1[k]) || empty(b0[k], b1[k]) || empty(c0[k], c1[k])) tl[k] = th[k] = NAN;
    copy_out(tl, th, zlo + i, zhi + i, m);
  }
  fesetenv(&env);
}
