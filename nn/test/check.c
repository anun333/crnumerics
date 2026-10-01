/* check: crnn (crnn.h).

     1  the constants in crnn-fast.h, against MPFR
     2  the one-argument functions, to nearest, against the MPFR references
        (crnn-ref.c): the edge values and 2^18 inputs drawn at every
        exponent (kit_sample1), or with "all" every one of the 2^32
        (kit_exhaust1); each run with the kit's control
     3  the table: every entry is an input the fast path can't decide, and
        its result is MPFR's
     4  a negative control: the naive binary32 formulas must differ from
        the references on the same inputs
     5  the environment: under round-upward with flush-to-zero set, every
        function gives the bits it gives in the default environment, and
        the caller's environment is back afterwards. The control: the fast
        paths, run bare in that environment, give other bits
     6  the composites, bit for bit against the specification in crnn.h
        computed independently (MPFR's exp, log1p and rsqrt, and mpfr_sum,
        through the kit): vectors of 1 to 4097 elements, normal, wide,
        cancelling, huge, tiny, subnormal, swamped (one term of 1 and
        many below half its ulp, which a plain sum drops), and with NaN or
        infinities
     7  the same composites under shuffling (softmax and the norms
        permuted with their inputs) and in place (y = x): the same bits
     8  how close the composites come to the exact mathematics (MPFR at
        600 bits, rounded to binary32): the share correctly rounded, and
        the largest error, which must stay below 1 ulp
     9  negative control for 6 and 7: a naive binary32 logsumexp (a plain
        loop, the C library's expf and logf) must differ from the
        specification, and from itself when its input is shuffled
    10  the binary16 and bfloat16 functions (added 2026-10-01): every one of
        the 2^16 inputs of each, against MPFR (kit_ref1 in the format), in
        the default environment and under round-upward with flush-to-zero;
        control: sigmoid's results judged against softplus's references
   The last line is the verdict.

     crnn-check        sections 1-9, sampled (seconds)
     crnn-check all    section 2 on every input (hours of CPU; use many cores) */
#include <fenv.h>
#include <float.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "kit.h"
#include "crnn.h"
#include "crnn-fast.h"
#include "crnn-ref.h"
#include "crnn-exceptions.h"

static const kit_mpfr1 REF[CRNN_NFN1] = {crnn_mpfr_sigmoid, crnn_mpfr_silu, crnn_mpfr_gelu, crnn_mpfr_softplus, kit_mpfr_rsqrt};
typedef void (*arr1)(float *, const float *, size_t);
static const arr1 FN[CRNN_NFN1] = {crnn_sigmoidf, crnn_siluf, crnn_geluf, crnn_softplusf, crnn_rsqrtf};

static uint64_t rng = 0x243f6a8885a308d3ULL;
static uint64_t next(void)
{
  uint64_t z = (rng += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}
static double unit(void) { return (next() >> 11) * 0x1p-53; }
static uint32_t bits(float x) { uint32_t u; memcpy(&u, &x, 4); return u; }
static float flt(uint64_t u) { uint32_t v = (uint32_t)u; float x; memcpy(&x, &v, 4); return x; }
static int same32(float a, float b) { return (a != a && b != b) || bits(a) == bits(b); }

/* 1 */
static int sec1(void)
{
  mpfr_t a, b; mpfr_inits2(300, a, b, (mpfr_ptr)0);
  mpfr_sqrt_ui(a, 2, MPFR_RNDN); mpfr_ui_div(a, 1, a, MPFR_RNDN);            /* 1/sqrt 2 */
  mpfr_set_d(b, CRNN_RSQRT2_HI, MPFR_RNDN); mpfr_add_d(b, b, CRNN_RSQRT2_LO, MPFR_RNDN);
  mpfr_sub(b, b, a, MPFR_RNDN); mpfr_div(b, b, a, MPFR_RNDN);
  int ok1 = mpfr_zero_p(b) || mpfr_get_exp(b) <= -106;
  mpfr_set_d(b, CRNN_RSQRT2_HI, MPFR_RNDN);
  int ok2 = mpfr_get_d(a, MPFR_RNDN) == CRNN_RSQRT2_HI;
  mpfr_const_pi(a, MPFR_RNDN); mpfr_sqrt(a, a, MPFR_RNDN); mpfr_ui_div(a, 2, a, MPFR_RNDN);   /* 2/sqrt pi */
  int ok3 = mpfr_get_d(a, MPFR_RNDN) == CRNN_TWO_RSQRTPI;
  mpfr_clears(a, b, (mpfr_ptr)0);
  printf("constants: 1/sqrt 2 as hi + lo within 2^-106 %s, hi rounded %s, 2/sqrt pi rounded %s\n",
         ok1 ? "yes" : "NO", ok2 ? "yes" : "NO", ok3 ? "yes" : "NO");
  return ok1 && ok2 && ok3 ? 0 : 1;
}

/* 2 and 4: the candidates, on encodings */
static int cur;
static uint64_t cand(uint64_t x) { float a = flt(x), y; FN[cur](&y, &a, 1); return bits(y); }
static uint64_t naive(uint64_t x)
{
  float a = flt(x), e;
  switch (cur) {
    case CRNN_SIGMOID: return bits(1.0f / (1.0f + expf(-a)));
    case CRNN_SILU: return bits(a / (1.0f + expf(-a)));
    case CRNN_GELU: return bits(0.5f * a * (1.0f + erff(a * 0.70710678f)));
    case CRNN_SOFTPLUS: e = expf(a); return bits(log1pf(e));
    default: return bits(1.0f / sqrtf(a));
  }
}

static int sec2(int all)
{
  int r = 0;
  for (cur = 0; cur < CRNN_NFN1; cur++) {
    kit_tally c, t = all ? kit_exhaust1(KIT_B32, cand, REF[cur], MPFR_RNDN, &c)
                         : kit_sample1(KIT_B32, cand, REF[cur], MPFR_RNDN, 1 << 18, 0x5eed + cur, &c);
    char what[64]; snprintf(what, sizeof what, "%s%s", crnn_fn1_name[cur], all ? ", every input" : "");
    r = kit_worst(r, kit_report(what, KIT_B32, t, c));
  }
  if (crnn_ref_long) printf("(%lu inputs needed more than 512 bits of MPFR)\n", crnn_ref_long);
  return r;
}

/* 3 */
static int sec3(void)
{
  int r = 0;
  for (int f = 0; f < CRNN_NFN1; f++) {
    size_t n = crnn_exc[f].n, decided = 0, wrong = 0, naive_wrong = 0;
    for (size_t i = 0; i < n; i++) {
      float x = flt(crnn_exc[f].t[i][0]), y, z, fast;
      decided += crnn_round32(crnn_fast(f, x), &z);
      uint32_t want = (uint32_t)kit_ref1(KIT_B32, REF[f], crnn_exc[f].t[i][0], MPFR_RNDN);
      FN[f](&y, &x, 1);
      wrong += bits(y) != want || crnn_exc[f].t[i][1] != want;
      fast = (float)crnn_fast(f, x);
      naive_wrong += bits(fast) != want;
    }
    printf("table, %-8s %6zu entries: %zu decided by the fast path (must be 0), %zu differ from MPFR; "
           "the fast path's plain rounding is wrong on %zu of them\n", crnn_fn1_name[f], n, decided, wrong, naive_wrong);
    if (decided || wrong) r = 1;
  }
  return r;
}

/* 4 */
static int sec4(void)
{
  int r = 0;
  for (cur = 0; cur < CRNN_NFN1; cur++) {
    kit_tally c, t = kit_sample1(KIT_B32, naive, REF[cur], MPFR_RNDN, 1 << 16, 0xbad + cur, &c);
    printf("negative control, naive binary32 %-8s %llu of %llu differ (must be > 0)\n", crnn_fn1_name[cur], t.differ, t.tested);
    if (!t.differ) r = 2;
  }
  return r;
}

/* 5: flush-to-zero and denormals-are-zero, on x86 (MXCSR) and aarch64 (FPCR.FZ) */
static void ftz(int on)
{
#if defined(__x86_64__)
  unsigned m; __asm__ volatile("stmxcsr %0" : "=m"(m)); m = on ? m | 0x8040 : m & ~0x8040u; __asm__ volatile("ldmxcsr %0" :: "m"(m));
#elif defined(__aarch64__)
  uint64_t c; __asm__ volatile("mrs %0, fpcr" : "=r"(c)); c = on ? c | (1u << 24) : c & ~(uint64_t)(1u << 24); __asm__ volatile("msr fpcr, %0" :: "r"(c));
#else
  (void)on;
#endif
}
static int ftz_on(void)
{
#if defined(__x86_64__)
  unsigned m; __asm__ volatile("stmxcsr %0" : "=m"(m)); return (m & 0x8040) == 0x8040;
#elif defined(__aarch64__)
  uint64_t c; __asm__ volatile("mrs %0, fpcr" : "=r"(c)); return (c >> 24) & 1;
#else
  return 1;
#endif
}

#define NV 4097
static float X[NV], G[NV], B[NV];
static int sec5(void)
{
  enum { N = 1 << 16 };
  static float x[N], y0[N], y1[N];
  for (int i = 0; i < N; i++) x[i] = i % 4 ? (float)((unit() - 0.5) * 60) : flt(next());   /* moderate, and raw bits (subnormals too) */
  long diff = 0, ctl = 0, restored = 1;
  for (int f = 0; f < CRNN_NFN1; f++) {
    FN[f](y0, x, N);
    fesetround(FE_UPWARD); ftz(1);
    FN[f](y1, x, N);
    restored &= fegetround() == FE_UPWARD && ftz_on();
    for (int i = 0; i < N; i++) {
      diff += !same32(y0[i], y1[i]);
      float z; if (!crnn_special(f, x[i], &z)) ctl += !same32(y0[i], (float)crnn_fast(f, x[i]));
    }
    ftz(0); fesetround(FE_TONEAREST);
  }
  /* the composites too */
  for (int i = 0; i < 1000; i++) { X[i] = (float)((unit() - 0.5) * 100) * (i % 7 ? 1 : 0x1p-130f); G[i] = (float)unit(); B[i] = (float)(unit() - 0.5); }
  float a[4][1000], b[4][1000], s0, s1;
  s0 = crnn_logsumexpf(X, 1000); crnn_softmaxf(a[0], X, 1000); crnn_layernormf(a[1], X, 1000, G, B, 1e-5f); crnn_rmsnormf(a[2], X, 1000, G, 1e-6f);
  fesetround(FE_UPWARD); ftz(1);
  s1 = crnn_logsumexpf(X, 1000); crnn_softmaxf(b[0], X, 1000); crnn_layernormf(b[1], X, 1000, G, B, 1e-5f); crnn_rmsnormf(b[2], X, 1000, G, 1e-6f);
  restored &= fegetround() == FE_UPWARD && ftz_on();
  ftz(0); fesetround(FE_TONEAREST);
  diff += !same32(s0, s1);
  for (int k = 0; k < 3; k++) for (int i = 0; i < 1000; i++) diff += !same32(a[k][i], b[k][i]);
  printf("environment: round-upward with flush-to-zero: %ld results differ from the default environment's "
         "(control, the bare fast paths there: %ld differ, must be > 0); caller's environment restored: %s\n",
         diff, ctl, restored ? "yes" : "NO");
  return diff || !restored ? 1 : !ctl ? 2 : 0;
}

/* 6: the specification, independently: MPFR's correctly rounded exp, log,
   rsqrt at binary64 (kit_ref1), mpfr_sum (kit_sum_ref); the basic
   operations in C, which round to nearest in binary64 here, as specified */
static double mexp(double x) { uint64_t u; memcpy(&u, &x, 8); u = kit_ref1(KIT_B64, mpfr_exp, u, MPFR_RNDN); double y; memcpy(&y, &u, 8); return y; }
static double mlog1p(double x) { uint64_t u; memcpy(&u, &x, 8); u = kit_ref1(KIT_B64, mpfr_log1p, u, MPFR_RNDN); double y; memcpy(&y, &u, 8); return y; }
static double mrsqrt(double x) { uint64_t u; memcpy(&u, &x, 8); u = kit_ref1(KIT_B64, kit_mpfr_rsqrt, u, MPFR_RNDN); double y; memcpy(&y, &u, 8); return y; }
static double D1[NV + 1];

static float spec_lse(const float *x, size_t n, double *S_out, double *m_out)
{
  if (!n) return -INFINITY;
  float m = -INFINITY; int nan = 0;
  for (size_t i = 0; i < n; i++) { if (x[i] != x[i]) nan = 1; else if (x[i] > m) m = x[i]; }
  for (size_t i = 0; i < n; i++) D1[i] = mexp((double)x[i] - m);
  double S = kit_sum_ref(KIT_B64, D1, NULL, n, MPFR_RNDN);
  D1[n] = -1;
  double T = kit_sum_ref(KIT_B64, D1, NULL, n + 1, MPFR_RNDN);
  *S_out = S; *m_out = m;
  if (nan) return NAN;
  if (isinf(m)) return m;
  return (float)((double)m + mlog1p(T));
}
static void spec_softmax(float *y, const float *x, size_t n)
{
  double S, m; spec_lse(x, n, &S, &m);
  for (size_t i = 0; i < n; i++) y[i] = (float)(mexp((double)x[i] - m) / S);
}
static void spec_layernorm(float *y, const float *x, size_t n, const float *g, const float *b, float eps)
{
  for (size_t i = 0; i < n; i++) D1[i] = x[i];
  double s1 = kit_sum_ref(KIT_B64, D1, NULL, n, MPFR_RNDN);
  D1[n] = -s1;
  double s2 = kit_sum_ref(KIT_B64, D1, NULL, n + 1, MPFR_RNDN), dn = (double)n;
  double m1 = s1 / dn, m2 = (fma(-m1, dn, s1) + s2) / dn;
  for (size_t i = 0; i < n; i++) D1[i] = ((double)x[i] - m1) - m2;
  double r = mrsqrt(kit_sum_ref(KIT_B64, D1, D1, n, MPFR_RNDN) / (double)n + (double)eps);
  for (size_t i = 0; i < n; i++) { double t = D1[i] * r; if (g) t = t * g[i]; if (b) t = t + b[i]; y[i] = (float)t; }
}
static void spec_rmsnorm(float *y, const float *x, size_t n, const float *g, float eps)
{
  for (size_t i = 0; i < n; i++) D1[i] = x[i];
  double r = mrsqrt(kit_sum_ref(KIT_B64, D1, D1, n, MPFR_RNDN) / (double)n + (double)eps);
  for (size_t i = 0; i < n; i++) { double t = (double)x[i] * r; if (g) t = t * g[i]; y[i] = (float)t; }
}

/* the vectors: kind k, length n */
static const size_t LEN[] = {1, 2, 3, 7, 64, 255, 256, 257, 1000, 4097};
enum { NK = 10 };
/* normal, wide, cancel, huge, tiny, offset, equal, special, subnormal, swamped */
static void fill(int k, size_t n)
{
  for (size_t i = 0; i < n; i++) {
    double u = unit() - 0.5;
    switch (k) {
      case 0: X[i] = (float)(u * 8); break;                                         /* logits-like */
      case 1: X[i] = (float)ldexp(u, (int)(next() % 60) - 30); break;               /* every scale */
      case 2: X[i] = i % 2 ? -X[i - 1] + (float)(u * 1e-3) : (float)(u * 1e4); break;   /* cancelling pairs */
      case 3: X[i] = (float)(u * 1e38); break;                                      /* exp overflows without the max */
      case 4: X[i] = (float)(u * 1e-30); break;
      case 5: X[i] = (float)(1e4 + u * 1e-2); break;                                /* a large mean, a small spread */
      case 6: X[i] = 3.25f; break;                                                   /* variance exactly 0 */
      case 7: X[i] = (float)(u * 10); if (next() % 5 == 0) X[i] = (float[]){NAN, INFINITY, -INFINITY, -0.0f}[next() % 4]; break;
      case 8: X[i] = flt(next() & 0x807fffffu); break;                               /* subnormals */
      default: X[i] = i ? (float)(-38 + u * 1e-3) : 0; break;   /* swamped: e^-38 < 2^-54, so a plain
                    binary64 sum that starts at the max (0) drops every other term, and gets 0 for
                    logsumexp where the exact sum gives about n e^-38 */
    }
    G[i] = (float)(0.5 + unit()); B[i] = (float)(unit() - 0.5);
  }
}

static float Y0[NV], Y1[NV], XS[NV], GS[NV], BS[NV];
static size_t IDX[NV];
static int sec67(void)
{
  long cases = 0, results = 0, diff = 0, order = 0, inplace = 0;
  for (int k = 0; k < NK; k++)
    for (unsigned li = 0; li < sizeof LEN / sizeof LEN[0]; li++)
      for (int rep = 0; rep < 3; rep++) {
        size_t n = LEN[li]; fill(k, n); cases++;
        double S, m;
        float a = crnn_logsumexpf(X, n), e = spec_lse(X, n, &S, &m);
        results++; diff += !same32(a, e);
        for (int w = 0; w < 5; w++) {                     /* softmax, layernorm, layernorm bare, rmsnorm, rmsnorm bare */
          const float *g = w == 2 || w == 4 ? NULL : G, *b = w == 2 ? NULL : B;
          switch (w) {
            case 0: crnn_softmaxf(Y0, X, n); spec_softmax(Y1, X, n); break;
            case 1: case 2: crnn_layernormf(Y0, X, n, g, b, 1e-5f); spec_layernorm(Y1, X, n, g, b, 1e-5f); break;
            default: crnn_rmsnormf(Y0, X, n, g, 1e-6f); spec_rmsnorm(Y1, X, n, g, 1e-6f); break;
          }
          for (size_t i = 0; i < n; i++) { results++; diff += !same32(Y0[i], Y1[i]); }
          /* 7: shuffled, and in place */
          for (size_t i = 0; i < n; i++) IDX[i] = i;
          kit_shuffle(next(), IDX, n);
          for (size_t i = 0; i < n; i++) { XS[i] = X[IDX[i]]; GS[i] = G[IDX[i]]; BS[i] = B[IDX[i]]; }
          const float *gs = g ? GS : NULL, *bs = b ? BS : NULL;
          switch (w) {
            case 0: crnn_softmaxf(Y1, XS, n); break;
            case 1: case 2: crnn_layernormf(Y1, XS, n, gs, bs, 1e-5f); break;
            default: crnn_rmsnormf(Y1, XS, n, gs, 1e-6f); break;
          }
          for (size_t i = 0; i < n; i++) order += !same32(Y1[i], Y0[IDX[i]]);
          switch (w) {                                     /* in place, on the shuffled copy */
            case 0: crnn_softmaxf(XS, XS, n); break;
            case 1: case 2: crnn_layernormf(XS, XS, n, gs, bs, 1e-5f); break;
            default: crnn_rmsnormf(XS, XS, n, gs, 1e-6f); break;
          }
          for (size_t i = 0; i < n; i++) inplace += !same32(XS[i], Y1[i]);
        }
        for (size_t i = 0; i < n; i++) XS[i] = X[IDX[i]];
        order += !same32(crnn_logsumexpf(XS, n), a);
      }
  printf("composites against the specification: %ld cases (%d kinds x %zu lengths x 3), %ld results, %ld differ\n",
         cases, NK, sizeof LEN / sizeof LEN[0], results, diff);
  printf("the same composites shuffled: %ld differ; in place: %ld differ\n", order, inplace);
  return diff || order || inplace;
}

/* 8: against the exact mathematics, at 600 bits */
static double ulps(float got, mpfr_srcptr exact)   /* |got - exact| in ulps of the binary32 nearest exact */
{
  if (got != got || isinf(got) || !mpfr_number_p(exact)) return 0;
  uint32_t near = (uint32_t)kit_round(KIT_B32, exact, MPFR_RNDN);
  float f = flt(near);
  if (isinf(f)) return 0;
  float nb = nextafterf(fabsf(f), INFINITY);
  double ulp = (double)nb - (double)fabsf(f);
  mpfr_t d; mpfr_init2(d, 600); mpfr_sub_d(d, exact, got, MPFR_RNDN);
  double e = fabs(mpfr_get_d(d, MPFR_RNDN)) / ulp; mpfr_clear(d);
  return e;
}
enum { A_LSE, A_SOFTMAX, A_LAYERNORM, A_RMSNORM, NA };
static const char *ANAME[NA] = {"logsumexp", "softmax", "layernorm", "rmsnorm"};
static long a_n[NA], a_cr[NA]; static double a_worst[NA];
static void tally(int a, double u) { a_n[a]++; a_cr[a] += u <= 0.5; if (u > a_worst[a]) a_worst[a] = u; }
static int sec8(void)
{
  mpfr_t m, s, t, v; mpfr_inits2(600, m, s, t, v, (mpfr_ptr)0);
  mpfr_t *e = malloc(NV * sizeof *e); for (int i = 0; i < NV; i++) mpfr_init2(e[i], 600);
  for (int k = 0; k < NK; k++) {
    if (k == 7) continue;                                /* specials: the formula, not a real number */
    for (unsigned li = 0; li < sizeof LEN / sizeof LEN[0]; li++) {
      size_t n = LEN[li]; fill(k, n);
      /* logsumexp and softmax: m + log sum e^(x_i - m), e^(x_i - m) / sum */
      float mx = -INFINITY; for (size_t i = 0; i < n; i++) if (X[i] > mx) mx = X[i];
      mpfr_set_zero(s, 1);
      for (size_t i = 0; i < n; i++) { mpfr_set_d(t, X[i], MPFR_RNDN); mpfr_sub_d(t, t, mx, MPFR_RNDN); mpfr_exp(e[i], t, MPFR_RNDN); mpfr_add(s, s, e[i], MPFR_RNDN); }
      mpfr_log(t, s, MPFR_RNDN); mpfr_add_d(t, t, mx, MPFR_RNDN);
      tally(A_LSE, ulps(crnn_logsumexpf(X, n), t));
      crnn_softmaxf(Y0, X, n);
      for (size_t i = 0; i < n; i++) { mpfr_div(t, e[i], s, MPFR_RNDN); tally(A_SOFTMAX, ulps(Y0[i], t)); }
      /* layernorm and rmsnorm */
      mpfr_set_zero(s, 1); for (size_t i = 0; i < n; i++) mpfr_add_d(s, s, X[i], MPFR_RNDN);
      mpfr_div_ui(m, s, n, MPFR_RNDN);
      mpfr_set_zero(v, 1);
      for (size_t i = 0; i < n; i++) { mpfr_set_d(e[i], X[i], MPFR_RNDN); mpfr_sub(e[i], e[i], m, MPFR_RNDN); mpfr_fma(v, e[i], e[i], v, MPFR_RNDN); }
      mpfr_div_ui(v, v, n, MPFR_RNDN); mpfr_add_d(v, v, 1e-5f, MPFR_RNDN); mpfr_rec_sqrt(v, v, MPFR_RNDN);
      crnn_layernormf(Y0, X, n, G, B, 1e-5f);
      for (size_t i = 0; i < n; i++) {
        mpfr_mul(t, e[i], v, MPFR_RNDN); mpfr_mul_d(t, t, G[i], MPFR_RNDN); mpfr_add_d(t, t, B[i], MPFR_RNDN);
        tally(A_LAYERNORM, ulps(Y0[i], t));
      }
      mpfr_set_zero(v, 1); for (size_t i = 0; i < n; i++) { mpfr_set_d(t, X[i], MPFR_RNDN); mpfr_fma(v, t, t, v, MPFR_RNDN); }
      mpfr_div_ui(v, v, n, MPFR_RNDN); mpfr_add_d(v, v, 1e-6f, MPFR_RNDN); mpfr_rec_sqrt(v, v, MPFR_RNDN);
      crnn_rmsnormf(Y0, X, n, G, 1e-6f);
      for (size_t i = 0; i < n; i++) { mpfr_mul_d(t, v, X[i], MPFR_RNDN); mpfr_mul_d(t, t, G[i], MPFR_RNDN); tally(A_RMSNORM, ulps(Y0[i], t)); }
    }
  }
  for (int i = 0; i < NV; i++) mpfr_clear(e[i]);
  free(e); mpfr_clears(m, s, t, v, (mpfr_ptr)0);
  int r = 0;
  for (int a = 0; a < NA; a++) {
    printf("%-9s against the exact mathematics: %ld of %ld correctly rounded (%.4f%%), largest error %.3f ulp (must be < 1)\n",
           ANAME[a], a_cr[a], a_n[a], 100.0 * a_cr[a] / a_n[a], a_worst[a]);
    if (a_worst[a] >= 1) r = 1;
  }
  return r;
}

/* 9 */
static float naive_lse(const float *x, size_t n)
{
  float m = -INFINITY, s = 0;
  for (size_t i = 0; i < n; i++) if (x[i] > m) m = x[i];
  for (size_t i = 0; i < n; i++) s += expf(x[i] - m);
  return m + logf(s);
}
static int sec9(void)
{
  long diff = 0, order = 0, cases = 0;
  for (int rep = 0; rep < 200; rep++) {
    size_t n = 1000 + rep; fill(rep % 2 ? 0 : 1, n); cases++;
    double S, m; float want = spec_lse(X, n, &S, &m), a = naive_lse(X, n);
    diff += !same32(a, want);
    for (size_t i = 0; i < n; i++) IDX[i] = i;
    kit_shuffle(next(), IDX, n);
    for (size_t i = 0; i < n; i++) XS[i] = X[IDX[i]];
    order += !same32(naive_lse(XS, n), a);
  }
  printf("negative control, naive binary32 logsumexp: differs from the specification in %ld of %ld cases, "
         "and from itself shuffled in %ld (both must be > 0)\n", diff, cases, order);
  return diff && order ? 0 : 2;
}

static int sec10(void)
{
  typedef void (*fn16)(uint16_t *, const uint16_t *, size_t);
  static const fn16 F[2][CRNN_NFN1] = {
    {crnn_sigmoid_f16, crnn_silu_f16, crnn_gelu_f16, crnn_softplus_f16, crnn_rsqrt_f16},
    {crnn_sigmoid_bf16, crnn_silu_bf16, crnn_gelu_bf16, crnn_softplus_bf16, crnn_rsqrt_bf16}};
  static const kit_mpfr1 REF[CRNN_NFN1] = {crnn_mpfr_sigmoid, crnn_mpfr_silu, crnn_mpfr_gelu, crnn_mpfr_softplus, kit_mpfr_rsqrt};
  static const kit_fmt KF[2] = {KIT_B16, KIT_BF16};
  static const char *const NM[2] = {"binary16", "bfloat16"};
  static uint16_t x[65536], y[65536], yu[65536], want[2][CRNN_NFN1][65536];
  for (uint32_t u = 0; u < 65536; u++) x[u] = (uint16_t)u;
  long tot = 0, ctl = 0;
  for (int fmt = 0; fmt < 2; fmt++)
    for (int f = 0; f < CRNN_NFN1; f++) {
#pragma omp parallel for schedule(dynamic, 1024)
      for (uint32_t u = 0; u < 65536; u++) want[fmt][f][u] = (uint16_t)kit_ref1(KF[fmt], REF[f], u, MPFR_RNDN);
      F[fmt][f](y, x, 65536);
      fenv_t env; fegetenv(&env); ftz_on(); fesetround(FE_UPWARD);
      F[fmt][f](yu, x, 65536);
      fesetenv(&env);
      long bad = 0, badu = 0;
      for (uint32_t u = 0; u < 65536; u++) {
        int nan_w = fmt ? ((want[fmt][f][u] & 0x7f80) == 0x7f80 && (want[fmt][f][u] & 0x7f)) : ((want[fmt][f][u] & 0x7c00) == 0x7c00 && (want[fmt][f][u] & 0x3ff));
        int nan_y = fmt ? ((y[u] & 0x7f80) == 0x7f80 && (y[u] & 0x7f)) : ((y[u] & 0x7c00) == 0x7c00 && (y[u] & 0x3ff));
        bad += !(y[u] == want[fmt][f][u] || (nan_w && nan_y));
        badu += y[u] != yu[u];
      }
      tot += bad + badu;
      printf("%s %-8s all 65536 inputs: %ld differ from MPFR; %ld differ under round-upward with flush-to-zero\n", NM[fmt], crnn_fn1_name[f], bad, badu);
    }
  for (int fmt = 0; fmt < 2; fmt++) {   /* control: sigmoid judged against softplus */
    F[fmt][CRNN_SIGMOID](y, x, 65536);
    for (uint32_t u = 0; u < 65536; u++) ctl += y[u] != want[fmt][CRNN_SOFTPLUS][u];
  }
  printf("control: %s and %s sigmoid judged against softplus's references: %ld of 131072 differ (must be > 0)\n", NM[0], NM[1], ctl);
  if (!ctl) return 2;
  return tot ? 1 : 0;
}

int main(int argc, char **argv)
{
  int all = argc > 1 && !strcmp(argv[1], "all");
  printf("crnn's one-argument functions run through %s\n", crnn_vector_path() ? "crmvec's vector code (CRNN_CRMVEC)" : "the scalar path");
  int r = sec1();
  r = kit_worst(r, sec2(all));
  if (!all) {
    r = kit_worst(r, sec3());
    r = kit_worst(r, sec4());
    r = kit_worst(r, sec5());
    r = kit_worst(r, sec67());
    r = kit_worst(r, sec8());
    r = kit_worst(r, sec9());
    r = kit_worst(r, sec10());
  }
  return kit_verdict(r, all ? "every binary32 input of every one-argument function is correctly rounded, and every control differs"
                            : "crnn is correctly rounded where it says so, follows its specification bit for bit in any order and environment, and every control differs");
}
