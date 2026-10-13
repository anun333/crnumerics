/* check: crdist (crdist.h).

     1  the constants in crdist-fast.h against MPFR, and crdist-poly.h's
        first coefficient and pieces
     2  the three functions, to nearest, against the MPFR references
        (crdist-ref.c): the edge values and 2^18 inputs drawn at every
        exponent (kit_sample1; 2^16, as MPFR's erfc is slow far out), or with "all" every one of the 2^32
        (kit_exhaust1); each run with the kit's control
     3  the references against a bisection that has no start: over the
        binary32 encodings, comparing the root with the midpoint between
        neighbouring binary32 numbers (subnormals included), so it also
        checks kit_ref1's rounding of the 24-bit answer into the
        subnormals; 2^10 inputs a function, and the smallest inputs of
        erfinv, whose answers are subnormal
     4  the table: every entry is an input the fast path can't decide, and
        its result is MPFR's
     5  a negative control: a plain binary32 formula (Winitzki's start and
        one Newton step with the C library's erff, erfcf and expf, as the
        first version did in binary64) must differ from the references on
        the same inputs
     6  the environment: under round-upward with flush-to-zero set, every
        function gives the bits it gives in the default environment, in
        place (y = x) too, and the caller's environment is back afterwards.
        The control: the fast paths, run bare in that environment, give
        other bits
     7  the bound crdist-fast.h claims: the binary64 value d is within
        2^-50 |d| of the exact answer (CRDIST_BOUND), decided by comparing
        the root with d (1 - 2^-50) and d (1 + 2^-50); how many lie between
        2^-51 and 2^-50 is shown. 2^16 raw encodings a function (those inside the domain);
        "bound-all" runs it on every input. The control: d moved by 2^-48
        relative must fall outside on every input
     8  the binary16 and bfloat16 functions (added 2026-10-13): every one of
        the 2^16 inputs of each against MPFR (kit_ref1 in the format), in
        the default environment and under round-upward with flush-to-zero;
        the control moves one result in 64 by an ulp; and how often the
        binary32 result rounded again to 16 bits would be wrong
     9  the samplers (added 2026-10-13): an independent Philox4x32-10,
        checked against Random123's known answers, and the library's words
        against it on streams far apart; every sampler's specification
        (crdist.h) computed again with MPFR's log, exp and sqrt and the
        MPFR probit, 17 cases on two streams, bit for bit; the
        distributions (KS against the exact distribution function from
        MPFR for the normal, exponential, gamma and chi-squared, at the
        0.1% level; beta's mean and t's mean and mean square), each with a
        neighbouring distribution as its control, which must fail; the
        environment; and fingerprints of fixed streams, recorded
   The last line is the verdict.

     crdist-check            sections 1-8, sampled but for 8 (a minute or two of CPU)
     crdist-check all        sections 1, 2 on every input, and 7 (hundreds of hours of CPU: MPFR's erfc is slow far out)
     crdist-check bound-all  sections 1 and 7 on every input
     crdist-check hash-all   every output of every function on all 2^32 inputs, hashed, against the digests
                             of the build crdist-check all proved (minutes on a few cores: the way to carry
                             that proof to another compiler, system or machine)
   After all, bound-all or hash-all, function names (erfinv ... logit) limit the run to those. */
#include <fenv.h>
#include <float.h>
#ifdef _OPENMP   /* none on macOS with Apple clang: the loops run on one thread there */
#include <omp.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "kit.h"
#include "crdist.h"
#include "crdist-fast.h"
#include "crdist-ref.h"
#include "crdist-exceptions.h"
#include "crdist-round16.h"

static const kit_mpfr1 REF[CRDIST_NFN] = {crdist_mpfr_erfinv, crdist_mpfr_erfcinv, crdist_mpfr_probit, crdist_mpfr_ndtr,
                                             crdist_mpfr_log_ndtr, crdist_mpfr_normpdf, crdist_mpfr_logit};
typedef void (*arr1)(float *, const float *, size_t);
static const arr1 FN[CRDIST_NFN] = {crdist_erfinvf, crdist_erfcinvf, crdist_probitf, crdist_ndtrf, crdist_log_ndtrf, crdist_normpdff, crdist_logitf};
#define NINV 3   /* the inverses, erfinv .. probit: the functions with a root to compare (crdist_eqn_at) */
static unsigned fnmask = ~0u;   /* the functions to run: all, or those named after "all" or "bound-all" */

static uint64_t rng = 0x452821e638d01377ULL;
static uint64_t next(void)
{
  uint64_t z = (rng += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}
static uint32_t bits(float x) { uint32_t u; memcpy(&u, &x, 4); return u; }
static float flt(uint64_t u) { uint32_t v = (uint32_t)u; float x; memcpy(&x, &v, 4); return x; }
static int same32(float a, float b) { return (a != a && b != b) || bits(a) == bits(b); }

/* an input inside f's domain, away from its special points: raw bits,
   kept when they land there */
static float inside(int f)
{
  for (;;) {
    float x = flt(next()), y;
    if (!crdist_special(f, x, &y)) return x;
  }
}

/* 1 */
static int sec1(void)
{
  mpfr_t a, b; mpfr_inits2(300, a, b, (mpfr_ptr)0);
  mpfr_sqrt_ui(a, 2, MPFR_RNDN);
  int ok1 = mpfr_get_d(a, MPFR_RNDN) == CRDIST_SQRT2_HI;
  mpfr_set_d(b, CRDIST_SQRT2_HI, MPFR_RNDN); mpfr_add_d(b, b, CRDIST_SQRT2_LO, MPFR_RNDN);
  mpfr_sub(b, b, a, MPFR_RNDN); mpfr_div(b, b, a, MPFR_RNDN);
  int ok2 = mpfr_zero_p(b) || mpfr_get_exp(b) <= -106;
  /* P(0) = sqrt pi/2 and the pieces' ends: the polynomials' first
     coefficient rounded, the pieces in order and meeting 10.17 */
  mpfr_const_pi(a, MPFR_RNDN); mpfr_sqrt(a, a, MPFR_RNDN); mpfr_div_2ui(a, a, 1, MPFR_RNDN);
  mpfr_set_d(b, crdist_c[0], MPFR_RNDN); mpfr_sub(b, b, a, MPFR_RNDN); mpfr_div(b, b, a, MPFR_RNDN);
  int ok3 = mpfr_zero_p(b) || mpfr_get_exp(b) <= -52;   /* the interpolant's error and the rounding */
  int ok4 = crdist_t_end[CRDIST_NPIECE - 1] > 10.17;
  for (int k = 0; k < CRDIST_NPIECE; k++) ok4 &= (k == 0 || crdist_t_end[k] > crdist_t_end[k - 1]) && crdist_t_mid[k] < crdist_t_end[k];
  mpfr_clears(a, b, (mpfr_ptr)0);
  printf("constants: sqrt 2 hi rounded %s, hi + lo within 2^-106 %s; P(0) within 2^-52 of sqrt pi/2 %s; tail pieces in order %s\n",
         ok1 ? "yes" : "NO", ok2 ? "yes" : "NO", ok3 ? "yes" : "NO", ok4 ? "yes" : "NO");
  return ok1 && ok2 && ok3 && ok4 ? 0 : 1;
}

/* 2 and 5: the candidates, on encodings */
static int cur;
static uint64_t cand(uint64_t x) { float a = flt(x), y; FN[cur](&y, &a, 1); return bits(y); }

/* Winitzki in binary32 and one Newton step: what a careful float
   implementation without a proof might do */
static float plain_ierf(float x, int tail)
{
  float L = tail ? logf(x * (2 - x)) : log1pf(-x * x), t = 4.3307467f + 0.5f * L;
  float y = sqrtf(sqrtf(t * t - L * 6.8027211f) - t);
  return tail ? y + (erfcf(y) - x) / (1.1283792f * expf(-y * y)) : y - (erff(y) - x) / (1.1283792f * expf(-y * y));
}
static uint64_t naive(uint64_t u)
{
  float x = flt(u), y;
  if (crdist_special(cur, x, &y)) return bits(y);
  switch (cur) {
    case CRDIST_ERFINV: { float a = fabsf(x); return bits(copysignf(a <= 0.5f ? plain_ierf(a, 0) : plain_ierf(1 - a, 1), x)); }
    case CRDIST_ERFCINV: return bits(x <= 0.5f ? plain_ierf(x, 1) : x <= 1 ? plain_ierf(1 - x, 0) : x <= 1.5f ? -plain_ierf(x - 1, 0) : -plain_ierf(2 - x, 1));
    case CRDIST_PROBIT: return bits(x < 0.5f ? -1.4142135f * plain_ierf(2 * x, 1) : 1.4142135f * plain_ierf(2 - 2 * x, 1));
    case CRDIST_NDTR: return bits(0.5f * erfcf(-x * 0.70710678f));
    case CRDIST_LOG_NDTR: return bits(logf(0.5f * erfcf(-x * 0.70710678f)));
    case CRDIST_NORMPDF: return bits(expf(-0.5f * x * x) * 0.39894228f);
    default: return bits(logf(x / (1 - x)));
  }
}

static int sec2(int all)
{
  int r = 0;
  for (cur = 0; cur < CRDIST_NFN; cur++) {
    if (!(fnmask >> cur & 1)) continue;
    kit_tally c, t = all ? kit_exhaust1(KIT_B32, cand, REF[cur], MPFR_RNDN, &c)
                         : kit_sample1(KIT_B32, cand, REF[cur], MPFR_RNDN, 1 << 16, 0x5eed + cur, &c);
    char what[64]; snprintf(what, sizeof what, "%s%s", crdist_fn_name[cur], all ? ", every input" : "");
    r = kit_worst(r, kit_report(what, KIT_B32, t, c));
  }
  if (crdist_ref_long) printf("(%lu comparisons needed more than 512 bits of MPFR)\n", crdist_ref_long);
  return r;
}

/* 3: RN(r) in binary32 by bisection over the encodings of positive
   numbers: the first u whose upper midpoint lies above r */
static uint32_t bisect(const crdist_eqn *e)
{
  mpfr_t m; mpfr_init2(m, 64);
  uint32_t lo = 0, hi = 0x7f7fffff;   /* r < the midpoint above hi (r is finite and below 2^127) */
  while (lo < hi) {
    uint32_t mid = lo + (hi - lo) / 2;
    mpfr_set_d(m, flt(mid), MPFR_RNDN); mpfr_add_d(m, m, flt(mid + 1), MPFR_RNDN); mpfr_div_2ui(m, m, 1, MPFR_RNDN);   /* exact */
    if (crdist_cmp(e, m) < 0) hi = mid; else lo = mid + 1;
  }
  mpfr_clear(m);
  return lo;
}
static int sec3(void)
{
  long n = 0, wrong = 0, ctl = 0;
  mpfr_t v; mpfr_init2(v, 256);
  for (int f = 0; f < NINV; f++)
    for (int i = 0; i < (1 << 10) + (f == CRDIST_ERFINV ? 64 : 0); i++) {
      float x = i < 1 << 10 ? inside(f) : flt(i - (1 << 10) + 1);   /* erfinv: also the 64 smallest subnormals */
      crdist_eqn e; int sign = crdist_eqn_at(&e, v, f, x);
      uint32_t b = bisect(&e) | (sign < 0 ? 0x80000000u : 0);
      uint32_t want = (uint32_t)kit_ref1(KIT_B32, REF[f], bits(x), MPFR_RNDN);
      wrong += b != want;
      ctl += (b ^ (i % 64 == 0)) != want;   /* the control: one in 64 moved by an ulp */
      n++;
    }
  mpfr_clear(v);
  printf("references against bisection: %ld inputs, %ld differ (control: %ld differ)\n", n, wrong, ctl);
  return wrong ? 1 : !ctl ? 2 : 0;
}

/* 4 */
static int sec4(void)
{
  int r = 0;
  for (int f = 0; f < CRDIST_NFN; f++) {
    size_t n = crdist_exc[f].n, decided = 0, wrong = 0, naive_wrong = 0;
    for (size_t i = 0; i < n; i++) {
      float x = flt(crdist_exc[f].t[i][0]), y, z, fast;
      decided += crdist_decide32(f, x, &z);
      uint32_t want = (uint32_t)kit_ref1(KIT_B32, REF[f], crdist_exc[f].t[i][0], MPFR_RNDN);
      FN[f](&y, &x, 1);
      wrong += bits(y) != want || crdist_exc[f].t[i][1] != want;
      fast = (float)crdist_fast(f, x);
      naive_wrong += bits(fast) != want;
    }
    printf("table, %-8s %6zu entries: %zu decided by the fast path (must be 0), %zu differ from MPFR; "
           "the fast path's plain rounding is wrong on %zu of them\n", crdist_fn_name[f], n, decided, wrong, naive_wrong);
    if (decided || wrong) r = 1;
  }
  return r;
}

/* 5 */
static int sec5(void)
{
  int r = 0;
  for (cur = 0; cur < CRDIST_NFN; cur++) {
    kit_tally c, t = kit_sample1(KIT_B32, naive, REF[cur], MPFR_RNDN, 1 << 16, 0xbad + cur, &c);
    printf("negative control, plain binary32 %-8s %llu of %llu differ (must be > 0)\n", crdist_fn_name[cur], t.differ, t.tested);
    if (!t.differ) r = 2;
  }
  return r;
}

/* 6: flush-to-zero and denormals-are-zero, on x86 (MXCSR) and aarch64 (FPCR.FZ) */
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
static int sec6(void)
{
  enum { N = 1 << 16 };
  static float x[N], y0[N], y1[N];
  long diff = 0, ctl = 0, restored = 1;
  for (int f = 0; f < CRDIST_NFN; f++) {
    for (int i = 0; i < N; i++) x[i] = i % 4 ? inside(f) : flt(next());   /* the domain, and raw bits (specials too) */
    FN[f](y0, x, N);
    fesetround(FE_UPWARD); ftz(1);
    FN[f](y1, x, N);
    restored &= fegetround() == FE_UPWARD && ftz_on();
    for (int i = 0; i < N; i++) {
      diff += !same32(y0[i], y1[i]);
      float z; if (!crdist_special(f, x[i], &z)) ctl += !same32(y0[i], (float)crdist_fast(f, x[i]));
    }
    FN[f](x, x, N);   /* in place, still in that environment */
    for (int i = 0; i < N; i++) diff += !same32(y0[i], x[i]);
    ftz(0); fesetround(FE_TONEAREST);
  }
  printf("environment: round-upward with flush-to-zero, and in place: %ld results differ from the default environment's "
         "(control, the bare fast paths there: %ld differ, must be > 0); caller's environment restored: %s\n",
         diff, ctl, restored ? "yes" : "NO");
  return diff || !restored ? 1 : !ctl ? 2 : 0;
}

/* 7: is the root within d (1 +- 2^-k)? (|d|, the positive root) */
static int within(const crdist_eqn *e, double d, int k)
{
  mpfr_t a, b; mpfr_inits2(128, a, b, (mpfr_ptr)0);
  mpfr_set_d(b, d, MPFR_RNDN); mpfr_div_2si(b, b, k, MPFR_RNDN);   /* d 2^-k */
  mpfr_set_d(a, d, MPFR_RNDN); mpfr_sub(a, a, b, MPFR_RNDN);       /* d (1 - 2^-k), exact in 128 bits */
  int ok = crdist_cmp(e, a) >= 0;
  if (ok) { mpfr_set_d(a, d, MPFR_RNDN); mpfr_add(a, a, b, MPFR_RNDN); ok = crdist_cmp(e, a) <= 0; }
  mpfr_clears(a, b, (mpfr_ptr)0);
  return ok;
}
/* the functions of x: is d within 2^-k of the value v (256 bits, far closer than that)? */
static int within_v(mpfr_srcptr v, double d, int k)
{
  mpfr_t a, b; mpfr_inits2(256, a, b, (mpfr_ptr)0);
  mpfr_sub_d(a, v, d, MPFR_RNDN); mpfr_abs(a, a, MPFR_RNDN);
  mpfr_abs(b, v, MPFR_RNDN); mpfr_div_2si(b, b, k, MPFR_RNDN);
  int ok = mpfr_lessequal_p(a, b);
  mpfr_clears(a, b, (mpfr_ptr)0);
  return ok;
}
static int sec7(int all)
{
  int r = 0;
  for (int f = 0; f < CRDIST_NFN; f++) {
    if (!(fnmask >> f & 1)) continue;
    unsigned long long n = 0, out = 0, out52 = 0, ctl = 0;
    uint64_t total = all ? 1ULL << 32 : 1 << 16;
    uint64_t seed = 0x7a11 + f;
#pragma omp parallel reduction(+ : n, out, out52, ctl) if (mpfr_buildopt_tls_p())
    {
      mpfr_t v; mpfr_init2(v, 256);
#pragma omp for schedule(dynamic, 1 << 12)
      for (uint64_t i = 0; i < total; i++) {
        float x, y;
        if (all) x = flt(i);
        else {   /* the same draw as kit_sample1 would not matter here: raw bits from a hash of i */
          uint64_t z = (seed + i) * 0x9e3779b97f4a7c15ULL; z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL; z ^= z >> 27;
          x = flt(z);
        }
        if (crdist_special(f, x, &y)) continue;
        n++;
        if (f >= NINV) {   /* a function of x: against its value at 256 bits */
          mpfr_t xm; mpfr_init2(xm, 24); mpfr_set_d(xm, x, MPFR_RNDN);
          crdist_mpfr_value(f, v, xm);
          mpfr_clear(xm);
          double d = crdist_fast(f, x);
          if (!within_v(v, d, CRDIST_BOUND)) out++;
          else if (!within_v(v, d, CRDIST_BOUND + 1)) out52++;
          if (within_v(v, d * (1 + ldexp(1, 2 - CRDIST_BOUND)), CRDIST_BOUND)) ctl++;
          continue;
        }
        crdist_eqn e; crdist_eqn_at(&e, v, f, x);
        double d = fabs(crdist_fast(f, x));
        if (!within(&e, d, CRDIST_BOUND)) out++;
        else if (!within(&e, d, CRDIST_BOUND + 1)) out52++;
        if (within(&e, d * (1 + ldexp(1, 2 - CRDIST_BOUND)), CRDIST_BOUND)) ctl++;   /* must never hold: ctl counts failures to differ */
      }
      mpfr_clear(v);
      mpfr_free_cache2(MPFR_FREE_LOCAL_CACHE);
    }
    printf("bound, %-8s %llu inputs%s: %llu outside 2^-%d (must be 0), %llu between 2^-%d and 2^-%d (control: %llu inside after a move of 2^-%d, must be 0)\n",
           crdist_fn_name[f], n, all ? " (every one)" : "", out, CRDIST_BOUND, out52, CRDIST_BOUND + 1, CRDIST_BOUND, ctl, CRDIST_BOUND - 2);
    if (out) r = 1;
    else if (!n || ctl) r = kit_worst(r, 2);
  }
  return r;
}

/* 8: the 16-bit formats, every input */
static int nan16(int fmt, uint16_t v) { return fmt ? (v & 0x7f80) == 0x7f80 && (v & 0x7f) : (v & 0x7c00) == 0x7c00 && (v & 0x3ff); }
static int same16(int fmt, uint16_t a, uint16_t b) { return a == b || (nan16(fmt, a) && nan16(fmt, b)); }
static int sec8(void)
{
  typedef void (*fn16)(uint16_t *, const uint16_t *, size_t);
  static const fn16 F[2][CRDIST_NFN] = {
    {crdist_erfinv_f16, crdist_erfcinv_f16, crdist_probit_f16, crdist_ndtr_f16, crdist_log_ndtr_f16, crdist_normpdf_f16, crdist_logit_f16},
    {crdist_erfinv_bf16, crdist_erfcinv_bf16, crdist_probit_bf16, crdist_ndtr_bf16, crdist_log_ndtr_bf16, crdist_normpdf_bf16, crdist_logit_bf16}};
  static const kit_fmt KF[2] = {KIT_B16, KIT_BF16};
  static const char *const NM[2] = {"binary16", "bfloat16"};
  static uint16_t x[65536], y[65536], yu[65536], want[65536];
  for (uint32_t u = 0; u < 65536; u++) x[u] = (uint16_t)u;
  long tot = 0, ctl = 0;
  for (int fmt = 0; fmt < 2; fmt++)
    for (int f = 0; f < CRDIST_NFN; f++) {
#pragma omp parallel for schedule(dynamic, 256) if (mpfr_buildopt_tls_p())
      for (uint32_t u = 0; u < 65536; u++) want[u] = (uint16_t)kit_ref1(KF[fmt], REF[f], u, MPFR_RNDN);
      F[fmt][f](y, x, 65536);
      fesetround(FE_UPWARD); ftz(1);
      F[fmt][f](yu, x, 65536);
      ftz(0); fesetround(FE_TONEAREST);
      long bad = 0, badu = 0, twice = 0;
      for (uint32_t u = 0; u < 65536; u++) {
        bad += !same16(fmt, y[u], want[u]);
        badu += !same16(fmt, y[u], yu[u]);
        ctl += !same16(fmt, u % 64 ? y[u] : y[u] ^ 1, want[u]);   /* the control: one in 64 moved by an ulp */
        float a = (float)crdist16_decode(fmt, (uint16_t)u), r;   /* binary32's result, rounded again */
        FN[f](&r, &a, 1);
        twice += !same16(fmt, crdist16_encode(fmt, crdist16_rn(fmt, r)), want[u]);
      }
      tot += bad + badu;
      printf("%s %-8s all 65536 inputs: %ld differ from MPFR, %ld under round-upward with flush-to-zero; "
             "the binary32 result rounded again would be wrong on %ld\n", NM[fmt], crdist_fn_name[f], bad, badu, twice);
    }
  printf("16-bit control: one result in 64 moved by an ulp: %ld of %d differ (must be > 0)\n", ctl, 2 * CRDIST_NFN * 65536);
  return tot ? 1 : !ctl ? 2 : 0;
}

/* 9: the samplers. An independent Philox4x32-10 (Random123's known
   answers first), the specification in crdist.h computed again with MPFR's
   log, exp, sqrt and the MPFR probit, compared bit for bit; the
   distributions tested; the environment; and the fingerprints */
static void philox_ref(const uint32_t c[4], const uint32_t k[2], uint32_t o[4])
{
  uint32_t x[4] = {c[0], c[1], c[2], c[3]}, key[2] = {k[0], k[1]};
  for (int round = 1; round <= 10; round++) {
    uint64_t a = (uint64_t)x[0] * 0xD2511F53u, b = (uint64_t)x[2] * 0xCD9E8D57u;
    uint32_t y[4] = {(uint32_t)(b >> 32) ^ x[1] ^ key[0], (uint32_t)b, (uint32_t)(a >> 32) ^ x[3] ^ key[1], (uint32_t)a};
    memcpy(x, y, sizeof x);
    key[0] += 0x9E3779B9u; key[1] += 0xBB67AE85u;   /* for the next round */
  }
  memcpy(o, x, sizeof x);
}
typedef struct { uint64_t seed, sub, idx; } rstream;
static uint32_t rword(rstream *r)
{
  uint64_t i = r->idx++, j = i / 4;
  uint32_t c[4] = {(uint32_t)(j & 0xffffffffu), (uint32_t)(j >> 32), (uint32_t)(r->sub & 0xffffffffu), (uint32_t)(r->sub >> 32)};
  uint32_t k[2] = {(uint32_t)(r->seed & 0xffffffffu), (uint32_t)(r->seed >> 32)}, o[4];
  philox_ref(c, k, o);
  return o[i % 4];
}
static double mref64(kit_mpfr1 f, double x) { uint64_t u; memcpy(&u, &x, 8); u = kit_ref1(KIT_B64, f, u, MPFR_RNDN); double y; memcpy(&y, &u, 8); return y; }
static double rU(rstream *r) { return (2.0 * rword(r) + 1) / 8589934592.0; }   /* (2w + 1) 2^-33 */
static float rnormal(rstream *r)
{
  uint32_t w = rword(r);
  float q = (float)((2.0 * (w & 0x7fffffffu) + 1) / 8589934592.0);
  float p = flt(kit_ref1(KIT_B32, crdist_mpfr_probit, bits(q), MPFR_RNDN));
  return w >= 0x80000000u ? -p : p;
}
static double rgamma(double a, rstream *r)
{
  if (a < 1) { double g = rgamma(a + 1, r), u = rU(r); return g * mref64(mpfr_exp, mref64(mpfr_log, u) / a); }
  double d = a - 1.0 / 3, c = 1 / mref64(mpfr_sqrt, 9 * d);
  for (;;) {
    double z = rnormal(r), v = 1 + c * z;
    if (v <= 0) continue;
    v = v * v * v;
    double u = rU(r);
    if (mref64(mpfr_log, u) < ((0.5 * (z * z) + d) - d * v) + d * mref64(mpfr_log, v)) return d * v;
  }
}
static double rlgamma(double a, rstream *r)
{
  if (a < 1) { double g = rgamma(a + 1, r), u = rU(r); return mref64(mpfr_log, g) + mref64(mpfr_log, u) / a; }
  return mref64(mpfr_log, rgamma(a, r));
}
enum { S_UNIFORM, S_NORMAL, S_EXP, S_GAMMA, S_BETA, S_CHI2, S_T, S_N };
static const char *const SNAME[S_N] = {"uniform", "normal", "exponential", "gamma", "beta", "chisquared", "student_t"};
static float rdraw(int k, double a, double b, rstream *r)
{
  switch (k) {
    case S_UNIFORM: return (float)((2.0 * (rword(r) >> 9) + 1) / 16777216.0);
    case S_NORMAL: return rnormal(r);
    case S_EXP: return (float)-mref64(mpfr_log, rU(r));
    case S_GAMMA: return (float)rgamma(a, r);
    case S_BETA: { double la = rlgamma(a, r), lb = rlgamma(b, r); return (float)(1 / (1 + mref64(mpfr_exp, lb - la))); }
    case S_CHI2: return (float)(2 * rgamma(a / 2, r));
    default: { double z = rnormal(r), g = rgamma(a / 2, r); return (float)(z / mref64(mpfr_sqrt, (2 * g) / a)); }
  }
}
static void ldraws(int k, double a, double b, float *y, size_t n, crdist_stream *s)
{
  switch (k) {
    case S_UNIFORM: crdist_uniformf(y, n, s); break;
    case S_NORMAL: crdist_normalf(y, n, s); break;
    case S_EXP: crdist_exponentialf(y, n, s); break;
    case S_GAMMA: crdist_gammaf(y, n, a, s); break;
    case S_BETA: crdist_betaf(y, n, a, b, s); break;
    case S_CHI2: crdist_chisquaredf(y, n, a, s); break;
    default: crdist_student_tf(y, n, a, s);
  }
}
static const struct { int k; double a, b; } SCASES[] = {
  {S_UNIFORM, 0, 0}, {S_NORMAL, 0, 0}, {S_EXP, 0, 0}, {S_GAMMA, 0.05, 0}, {S_GAMMA, 0.3, 0}, {S_GAMMA, 1, 0},
  {S_GAMMA, 2.5, 0}, {S_GAMMA, 30, 0}, {S_GAMMA, 1e6, 0}, {S_BETA, 2, 5}, {S_BETA, 0.01, 0.02}, {S_BETA, 0.5, 50},
  {S_CHI2, 3, 0}, {S_CHI2, 0.7, 0}, {S_T, 5, 0}, {S_T, 1, 0}, {S_T, 0.5, 0}};
#define NSC (int)(sizeof SCASES / sizeof SCASES[0])
static uint64_t fnv(const float *y, size_t n)
{
  uint64_t h = 0xcbf29ce484222325ULL;
  for (size_t i = 0; i < n; i++) { uint32_t u = bits(y[i]); for (int b = 0; b < 4; b++) { h ^= (u >> (8 * b)) & 0xff; h *= 0x100000001b3ULL; } }
  return h;
}
/* Kolmogorov-Smirnov: the largest gap between the empirical distribution of y (sorted here) and F */
typedef double (*cdf)(double x, double a);
static int cmpf(const void *p, const void *q) { float a = *(const float *)p, b = *(const float *)q; return (a > b) - (a < b); }
static double ks(float *y, size_t n, cdf F, double a)
{
  qsort(y, n, sizeof *y, cmpf);
  double d = 0;
#pragma omp parallel for reduction(max : d) if (mpfr_buildopt_tls_p())
  for (size_t i = 0; i < n; i++) {
    double f = F(y[i], a), lo = f - (double)i / n, hi = (double)(i + 1) / n - f;
    d = fmax(d, fmax(lo, hi));
  }
  mpfr_free_cache2(MPFR_FREE_LOCAL_CACHE);
  return d;
}
static double cdf_normal(double x, double mu) { mpfr_t t; mpfr_init2(t, 64); mpfr_set_d(t, -(x - mu) * 0.70710678118654752, MPFR_RNDN); mpfr_erfc(t, t, MPFR_RNDN); double v = mpfr_get_d(t, MPFR_RNDN) / 2; mpfr_clear(t); return v; }
static double cdf_exp(double x, double rate) { return -expm1(-rate * x); }
static double cdf_gamma(double x, double a)   /* P(a, x) = 1 - Gamma(a, x)/Gamma(a) */
{
  if (x <= 0) return 0;
  mpfr_t A, X, g, G; mpfr_inits2(64, A, X, g, G, (mpfr_ptr)0);
  mpfr_set_d(A, a, MPFR_RNDN); mpfr_set_d(X, x, MPFR_RNDN);
  mpfr_gamma_inc(g, A, X, MPFR_RNDN); mpfr_gamma(G, A, MPFR_RNDN); mpfr_div(g, g, G, MPFR_RNDN);
  double v = 1 - mpfr_get_d(g, MPFR_RNDN);
  mpfr_clears(A, X, g, G, (mpfr_ptr)0);
  return v;
}
static double cdf_chi2(double x, double k) { return cdf_gamma(x / 2, k / 2); }
/* the discrete samplers, again from the specification: log k! from MPFR's lngamma (the table) and MPFR's log
   (Stirling's series), and every log, exp, log1p and sqrt from MPFR */
static double rlfact(double k)
{
  if (k < 128) return mref64(mpfr_lngamma, k + 1);
  double t = 1 / k;
  return ((k + 0.5) * mref64(mpfr_log, k) - k) + (0x1.d67f1c864beb5p-1 + t * (1.0 / 12 - (t * t) / 360));
}
static double rpoisson(double lam, rstream *r)
{
  if (lam == 0) return 0;
  if (lam < 10) {
    double u = rU(r), p = mref64(mpfr_exp, -lam), f = p, k = 0;
    while (u > f && k < 1000) { k = k + 1; p = (p * lam) / k; f = f + p; }
    return k;
  }
  double slam = mref64(mpfr_sqrt, lam), loglam = mref64(mpfr_log, lam), b = 0.931 + 2.53 * slam, a = -0.059 + 0.02483 * b;
  double invalpha = 1.1239 + 1.1328 / (b - 3.4), vr = 0.9277 - 3.6224 / (b - 2);
  for (;;) {
    double U = rU(r) - 0.5, V = rU(r), us = 0.5 - fabs(U), k = floor((2 * a / us + b) * U + lam + 0.43);
    if (us >= 0.07 && V <= vr) return k;
    if (k < 0 || (us < 0.013 && V > us)) continue;
    if ((mref64(mpfr_log, V) + mref64(mpfr_log, invalpha)) - mref64(mpfr_log, a / (us * us) + b) <= (-lam + k * loglam) - rlfact(k)) return k;
  }
}
static double rbinomial(double n, double p, rstream *r)
{
  double q = 1 - p, np = n * p;
  if (np < 10) {
    double qn = mref64(mpfr_exp, n * mref64(mpfr_log1p, -p)), bound = fmin(n, np + 10 * mref64(mpfr_sqrt, np * q + 1));
    double u = rU(r), px = qn, k = 0;
    while (u > px) {
      k = k + 1;
      if (k > bound) { k = 0; px = qn; u = rU(r); }
      else { u = u - px; px = (((n - k + 1) * p) * px) / (k * q); }
    }
    return k;
  }
  double spq = mref64(mpfr_sqrt, np * q), b = 1.15 + 2.53 * spq, a = (-0.0873 + 0.0248 * b) + 0.01 * p, c = np + 0.5;
  double vr = 0.92 - 4.2 / b, alpha = (2.83 + 5.1 / b) * spq, lpq = mref64(mpfr_log, p / q), m = floor((n + 1) * p);
  double h = rlfact(m) + rlfact(n - m);
  for (;;) {
    double U = rU(r) - 0.5, V = rU(r), us = 0.5 - fabs(U), k = floor((2 * a / us + b) * U + c);
    if (k < 0 || k > n) continue;
    if (us >= 0.07 && V <= vr) return k;
    if (mref64(mpfr_log, (V * alpha) / (a / (us * us) + b)) <= ((h - rlfact(k)) - rlfact(n - k)) + (k - m) * lpq) return k;
  }
}
static int64_t rdiscrete(int pois, double par, int64_t trials, rstream *r)
{
  if (pois) return !(par >= 0 && par <= 1e15) ? -1 : (int64_t)rpoisson(par, r);
  if (!(par >= 0 && par <= 1) || trials < 0 || trials > (1LL << 53)) return -1;
  if (trials == 0 || par == 0) return 0;
  if (par == 1) return trials;
  if (par > 0.5) return trials - (int64_t)rbinomial((double)trials, 1 - par, r);
  return (int64_t)rbinomial((double)trials, par, r);
}
static const struct { int pois; double par; int64_t trials; } DCASES[] = {
  {1, 0, 0}, {1, 0.001, 0}, {1, 0.5, 0}, {1, 3, 0}, {1, 9.999, 0}, {1, 10, 0}, {1, 25.5, 0}, {1, 1000, 0}, {1, 1e9, 0},
  {1, 1e15, 0}, {1, -1, 0}, {1, NAN, 0}, {0, 0.3, 0}, {0, 0.3, 10}, {0, 0.05, 100}, {0, 0.009, 1000}, {0, 0.5, 19},
  {0, 0.4, 25}, {0, 0.3, 1000}, {0, 0.5, 1000000}, {0, 0.9, 50}, {0, 1e-3, 1LL << 40}, {0, 0.25, 1LL << 53},
  {0, 1, 7}, {0, 0, 7}, {0, 1.5, 7}, {0, 0.5, -3}};
#define NDC (int)(sizeof DCASES / sizeof DCASES[0])
static void ddraws(int c, int64_t *y, size_t n, crdist_stream *s)
{
  if (DCASES[c].pois) crdist_poisson(y, n, DCASES[c].par, s);
  else crdist_binomial(y, n, DCASES[c].trials, DCASES[c].par, s);
}
/* chi-squared goodness of fit of the counts y against the probabilities pk (k = 0 .. K, the rest lumped into the
   last bin), bins merged until each expects 5; the statistic over its 0.1% point (Wilson and Hilferty) */
static double gof(const int64_t *y, size_t n, const double *pk, int K)
{
  static double obs[4096], exp_[4096];
  int nb = 0; double e = 0, o = 0;
  static long cnt[4096]; memset(cnt, 0, sizeof cnt);
  for (size_t i = 0; i < n; i++) cnt[y[i] < 0 ? 0 : y[i] > K ? K : y[i]]++;
  double rest = 1;
  for (int k = 0; k <= K; k++) {
    double pr = k < K ? pk[k] : rest;
    rest -= pk[k];
    e += pr * n; o += cnt[k];
    if (e >= 5 || k == K) { exp_[nb] = e; obs[nb] = o; nb++; e = o = 0; }
  }
  if (nb > 1 && exp_[nb - 1] < 5) { exp_[nb - 2] += exp_[nb - 1]; obs[nb - 2] += obs[nb - 1]; nb--; }
  double x2 = 0; for (int b = 0; b < nb; b++) x2 += (obs[b] - exp_[b]) * (obs[b] - exp_[b]) / exp_[b];
  double df = nb - 1, z = 3.0902, crit = df * pow(1 - 2 / (9 * df) + z * sqrt(2 / (9 * df)), 3);
  return x2 / crit;
}
static void pmf_poisson(double lam, double *pk, int K) { for (int k = 0; k <= K; k++) pk[k] = exp(k * log(lam) - lam - lgamma(k + 1.0)); }
static void pmf_binomial(double n, double p, double *pk, int K)
{
  for (int k = 0; k <= K; k++) pk[k] = k > n ? 0 : exp(lgamma(n + 1) - lgamma(k + 1.0) - lgamma(n - k + 1) + k * log(p) + (n - k) * log1p(-p));
}
static int sec9d(void)
{
  int r = 0;
  /* the specification, bit for bit, on two streams a case, and the index */
  enum { ND = 4096 };
  long bad = 0, total = 0;
#pragma omp parallel for schedule(dynamic, 1) reduction(+ : bad, total) if (mpfr_buildopt_tls_p())
  for (int c = 0; c < 2 * NDC; c++) {
    int64_t *y = malloc(ND * sizeof *y);
    crdist_stream s = {c % 2 ? 0x243f6a8885a308d3ULL : 11, c % 2 ? 3ULL << 62 : 0, c % 2 ? (1ULL << 33) + 1 : 0};
    rstream q = {s.seed, s.substream, s.index};
    ddraws(c / 2, y, ND, &s);
    for (int i = 0; i < ND; i++) bad += y[i] != rdiscrete(DCASES[c / 2].pois, DCASES[c / 2].par, DCASES[c / 2].trials, &q);
    bad += s.index != q.idx;
    total += ND;
    free(y);
    mpfr_free_cache2(MPFR_FREE_LOCAL_CACHE);
  }
  printf("samplers: poisson and binomial, %d cases (lam 0 to 1e15, n to 2^53, the parameters outside their ranges), %ld draws "
         "against the specification computed with MPFR: %ld differ\n", NDC, total, bad);
  if (bad) r = 1;
  /* the distributions: chi-squared against the exact probabilities, 2^16 draws; the control, a neighbouring
     parameter, must fail */
  enum { NG = 1 << 16 };
  static int64_t y[NG];
  static double pk[4096];
  static const struct { int pois; double par; int64_t trials; double wrong; } G[] = {
    {1, 0.3, 0, 0.32}, {1, 3, 0, 3.1}, {1, 25.5, 0, 26.2}, {1, 300, 0, 304}, {0, 0.05, 20, 0.06}, {0, 0.3, 100, 0.31},
    {0, 0.009, 1000, 0.0098}, {0, 0.4, 25, 0.42}, {0, 0.7, 60, 0.71}};
  int gfail = 0, gctl = 0;
  for (size_t i = 0; i < sizeof G / sizeof G[0]; i++) {
    crdist_stream s = {2026, 500 + i, 0};
    double mean = G[i].pois ? G[i].par : G[i].trials * G[i].par;
    int K = (int)(mean + 12 * sqrt(mean + 1) + 20); if (G[i].pois == 0 && K > G[i].trials) K = (int)G[i].trials;
    if (G[i].pois) crdist_poisson(y, NG, G[i].par, &s); else crdist_binomial(y, NG, G[i].trials, G[i].par, &s);
    if (G[i].pois) pmf_poisson(G[i].par, pk, K); else pmf_binomial(G[i].trials, G[i].par, pk, K);
    double g = gof(y, NG, pk, K);
    if (G[i].pois) pmf_poisson(G[i].wrong, pk, K); else pmf_binomial(G[i].trials, G[i].wrong, pk, K);
    double gw = gof(y, NG, pk, K);
    printf("  chi-squared %-8s %-6g %-5lld %.3f of the 0.1%% point; against %g: %.2f (must exceed 1)\n",
           G[i].pois ? "poisson" : "binomial", G[i].par, (long long)G[i].trials, g, G[i].wrong, gw);
    gfail += g > 1; gctl += gw > 1;
  }
  /* the large ones: the mean within 4.5 standard errors */
  static const struct { int pois; double par; int64_t trials; } M[] = {{1, 1e9, 0}, {1, 1e15, 0}, {0, 0.5, 1000000}, {0, 1e-3, 1LL << 40}};
  for (size_t i = 0; i < sizeof M / sizeof M[0]; i++) {
    crdist_stream s = {2026, 600 + i, 0};
    if (M[i].pois) crdist_poisson(y, NG, M[i].par, &s); else crdist_binomial(y, NG, M[i].trials, M[i].par, &s);
    double mu = M[i].pois ? M[i].par : M[i].trials * M[i].par, var = M[i].pois ? mu : mu * (1 - M[i].par), m = 0;
    for (int k = 0; k < NG; k++) m += (double)y[k] - mu;
    double z = (m / NG) / sqrt(var / NG);
    if (M[i].pois) printf("  mean of poisson(%g): %.2f standard errors from %g\n", M[i].par, z, mu);
    else printf("  mean of binomial(%lld, %g): %.2f standard errors from %g\n", (long long)M[i].trials, M[i].par, z, mu);
    gfail += fabs(z) > 4.5;
  }
  printf("samplers: poisson and binomial distributions, %d of 13 tests fail (must be 0); controls, %d of 9 fail (must be 9)\n", gfail, gctl);
  if (gfail) r = 1;
  if (gctl != 9) r = kit_worst(r, 2);
  /* the environment, and the fingerprints (FNV-1a over the 8-byte values of the first 4096 draws, stream (1, 0, 0)) */
  long ebad = 0;
  uint64_t hs[NDC];
  for (int c = 0; c < NDC; c++) {
    int64_t a[512], b[512]; crdist_stream s0 = {5, 100 + c, 0}, s1 = s0;
    ddraws(c, a, 512, &s0);
    fesetround(FE_UPWARD); ftz(1);
    ddraws(c, b, 512, &s1);
    ftz(0); fesetround(FE_TONEAREST);
    ebad += memcmp(a, b, sizeof a) != 0 || s0.index != s1.index;
    crdist_stream f = {1, 0, 0};
    ddraws(c, y, 4096, &f);
    uint64_t h = 0xcbf29ce484222325ULL;
    for (int i = 0; i < 4096; i++) for (int k = 0; k < 8; k++) { h ^= ((uint64_t)y[i] >> (8 * k)) & 0xff; h *= 0x100000001b3ULL; }
    hs[c] = h;
  }
  static const uint64_t DPRINT[NDC] = {   /* recorded 2026-10-13 on x86-64 (board5); every other machine's crdist-check compares */
    0x8f6955bf94ec2325ULL, 0xd9d07260cdd21ec4ULL, 0x8718e72c977d68a1ULL, 0xf651042b777b9148ULL,
    0x35d528680d664e3cULL, 0x2db205e0ebbe7f00ULL, 0x3554cb5b37ccef0bULL, 0x4321285d4e7c04d8ULL,
    0x31edf65f2fb7038aULL, 0x5ecb9873d658f2d2ULL, 0x9111afa91650a325ULL, 0x9111afa91650a325ULL,
    0x8f6955bf94ec2325ULL, 0x14b339d09afa2489ULL, 0x487cbeea6d5f4d79ULL, 0xc2a29fd746217bc2ULL,
    0x366268f51175a3c1ULL, 0x9b035c47e1316927ULL, 0xf7ddf5e327c935adULL, 0xfe4f8c4fce3feae2ULL,
    0x4e1fce12e1059caaULL, 0xa6d908b8c42e3eacULL, 0xa3a84bf210528944ULL, 0x3626c669c7f92325ULL,
    0x8f6955bf94ec2325ULL, 0x9111afa91650a325ULL, 0x9111afa91650a325ULL};
  int pbad = 0, have = 0;
  for (int c = 0; c < NDC; c++) {
    printf("  fingerprint %-8s %-6g %-16lld 0x%016llx\n", DCASES[c].pois ? "poisson" : "binomial", DCASES[c].par,
           (long long)DCASES[c].trials, (unsigned long long)hs[c]);
    if (DPRINT[c]) { have++; pbad += hs[c] != DPRINT[c]; }
  }
  printf("samplers: poisson and binomial under round-upward with flush-to-zero: %ld cases differ; %d of %d fingerprints differ\n", ebad, pbad, have);
  if (ebad || pbad) r = 1;
  return r;
}

static int sec9(void)
{
  int r = 0;
  /* Random123's known answers (tests/kat_vectors): counter, key -> words */
  static const uint32_t KAT[3][10] = {
    {0, 0, 0, 0, 0, 0, 0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8},
    {0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd},
    {0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344, 0xa4093822, 0x299f31d0, 0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1}};
  int kat = 0;
  for (int i = 0; i < 3; i++) { uint32_t o[4]; philox_ref(KAT[i], KAT[i] + 4, o); kat += !memcmp(o, KAT[i] + 6, sizeof o); }
  /* the library's words against this Philox, on streams far apart */
  static const uint64_t ST[4][3] = {{0, 0, 0}, {42, 7, 1}, {0xfedcba9876543210ULL, 0x8000000000000005ULL, 0x123456789abULL}, {1, 0xffffffffffffffffULL, 0x3ffffffffffffff0ULL}};
  long wbad = 0;
  for (int i = 0; i < 4; i++) {
    uint32_t w[257]; crdist_stream s = {ST[i][0], ST[i][1], ST[i][2]}; rstream q = {ST[i][0], ST[i][1], ST[i][2]};
    crdist_words(w, 257, &s);
    for (int j = 0; j < 257; j++) wbad += w[j] != rword(&q);
    wbad += s.index != q.idx;
  }
  printf("samplers: Philox4x32-10 matches Random123's known answers on %d of 3; the library's words differ from it on %ld of 1028 (and the index)\n", kat, wbad);
  if (kat != 3 || wbad) r = 1;
  /* the specification, bit for bit: each case on two streams, and the index left where the specification leaves it */
  enum { NB = 2048 };
  long sbad = 0, ictl = 0, total = 0;
#pragma omp parallel for schedule(dynamic, 1) reduction(+ : sbad, ictl, total) if (mpfr_buildopt_tls_p())
  for (int c = 0; c < 2 * NSC; c++) {
    int k = SCASES[c / 2].k; double a = SCASES[c / 2].a, b = SCASES[c / 2].b;
    float y[NB];
    crdist_stream s = {c % 2 ? 0x9e3779b97f4a7c15ULL : 7, c % 2 ? 1ULL << 63 : 0, c % 2 ? (1ULL << 40) + 3 : 0};
    rstream q = {s.seed, s.substream, s.index};
    ldraws(k, a, b, y, NB, &s);
    for (int i = 0; i < NB; i++) {
      float want = rdraw(k, a, b, &q);
      sbad += !same32(y[i], want);
      ictl += i % 64 == 0 && same32(flt(bits(y[i]) ^ 1), want);   /* the control: an ulp moved must not pass */
      total++;
    }
    sbad += s.index != q.idx;
    mpfr_free_cache2(MPFR_FREE_LOCAL_CACHE);
  }
  printf("samplers: %d cases (shapes 0.05 to 1e6), %ld draws against the specification computed with MPFR: %ld differ "
         "(control: %ld of the moved ones pass, must be 0)\n", NSC, total, sbad, ictl);
  if (sbad) r = 1;
  if (ictl) r = kit_worst(r, 2);
  /* the distributions: KS against the exact distribution function, 2^15 draws, at the 0.1% level (1.949/sqrt n);
     the control: the same draws against a neighbouring distribution must fail */
  enum { NK = 1 << 15 };
  static float y[NK];
  double crit = 1.949 / sqrt((double)NK);
  static const struct { int k; double a; cdf F; double fa, wrong; } KS[] = {
    {S_NORMAL, 0, cdf_normal, 0, 0.05}, {S_EXP, 0, cdf_exp, 1, 1.05}, {S_GAMMA, 0.05, cdf_gamma, 0.05, 0.06},
    {S_GAMMA, 0.3, cdf_gamma, 0.3, 0.32}, {S_GAMMA, 2.5, cdf_gamma, 2.5, 2.6}, {S_GAMMA, 30, cdf_gamma, 30, 30.6},
    {S_CHI2, 3, cdf_chi2, 3, 3.15}};
  int kfail = 0, kctl = 0;
  for (size_t i = 0; i < sizeof KS / sizeof KS[0]; i++) {
    crdist_stream s = {2026, 10 + i, 0};
    ldraws(KS[i].k, KS[i].a, 0, y, NK, &s);
    double d = ks(y, NK, KS[i].F, KS[i].fa), dw = ks(y, NK, KS[i].F, KS[i].wrong);
    printf("  KS %-11s %-5g D = %.4f (critical %.4f); against %g: %.4f (must exceed it)\n", SNAME[KS[i].k], KS[i].a, d, crit, KS[i].wrong, dw);
    kfail += d > crit; kctl += dw > crit;
  }
  /* beta and t: the sample mean (and t's variance, finite for nu > 4) within 4.5 standard errors; the control, a
     neighbouring parameter's mean, outside */
  {
    crdist_stream s = {2026, 99, 0};
    double a = 2, b = 5, m = a / (a + b), v = a * b / ((a + b) * (a + b) * (a + b + 1)), mw = a / (a + 5.3);
    crdist_betaf(y, NK, a, b, &s);
    double mean = 0; for (int i = 0; i < NK; i++) mean += y[i]; mean /= NK;
    double z = (mean - m) / sqrt(v / NK), zw = (mean - mw) / sqrt(v / NK);
    printf("  beta(2, 5) mean %.5f: %.2f standard errors from %.5f; from beta(2, 5.3)'s %.5f: %.1f (must exceed 4.5)\n", mean, z, m, mw, zw);
    kfail += fabs(z) > 4.5; kctl += fabs(zw) > 4.5;
    crdist_student_tf(y, NK, 9, &s);
    double m1 = 0, m2 = 0; for (int i = 0; i < NK; i++) { m1 += y[i]; m2 += (double)y[i] * y[i]; } m1 /= NK; m2 /= NK;
    double var = 9.0 / 7, kurt = 6.0 / 5 + 3, zt = m1 / sqrt(var / NK), zv = (m2 - var) / sqrt((kurt - 1) * var * var / NK);
    double zvw = (m2 - 11.0 / 9) / sqrt((kurt - 1) * var * var / NK);
    printf("  student_t(9) mean %.4f: %.2f standard errors; mean square %.4f: %.2f from 9/7, %.1f from t(11)'s 11/9 (must exceed 4.5)\n", m1, zt, m2, zv, zvw);
    kfail += fabs(zt) > 4.5 || fabs(zv) > 4.5; kctl += fabs(zvw) > 4.5;
  }
  printf("samplers: distributions, %d of 9 tests fail (must be 0); their controls, %d of 9 fail (must be 9)\n", kfail, kctl);
  if (kfail) r = 1;
  if (kctl != 9) r = kit_worst(r, 2);
  /* the environment, and a shape that is not positive: NaNs, no words */
  long ebad = 0;
  for (int c = 0; c < NSC; c++) {
    float y0[512], y1[512]; crdist_stream s0 = {5, c, 0}, s1 = s0;
    ldraws(SCASES[c].k, SCASES[c].a, SCASES[c].b, y0, 512, &s0);
    fesetround(FE_UPWARD); ftz(1);
    ldraws(SCASES[c].k, SCASES[c].a, SCASES[c].b, y1, 512, &s1);
    ftz(0); fesetround(FE_TONEAREST);
    for (int i = 0; i < 512; i++) ebad += !same32(y0[i], y1[i]);
    ebad += s0.index != s1.index;
  }
  crdist_stream sn = {1, 2, 3}; float yn[4]; crdist_gammaf(yn, 4, -1, &sn);
  int nanok = sn.index == 3 && yn[0] != yn[0] && yn[3] != yn[3];
  crdist_betaf(yn, 4, 1, NAN, &sn); nanok &= sn.index == 3 && yn[0] != yn[0];
  printf("samplers: under round-upward with flush-to-zero %ld draws differ; a shape not positive gives NaNs and takes no words: %s\n", ebad, nanok ? "yes" : "NO");
  if (ebad || !nanok) r = 1;
  /* the fingerprints: FNV-1a of the first 4096 draws of each case from the stream (seed 1, substream 0, index 0),
     so that another implementation of the specification can check itself */
  static const uint64_t PRINT[NSC] = {   /* recorded 2026-10-13 on x86-64 (board5); every other machine's crdist-check compares */
    0x4fbb7b01f6dc5a29ULL, 0xe0b080312afb39d4ULL, 0x8cdbe8e2536a7b47ULL, 0x880ca8a9822e2eedULL,
    0x7e7a1f7e0900f910ULL, 0x372d5cbee6f9e530ULL, 0xc6a5cb829535e89eULL, 0xc8f4100373e60fffULL,
    0x73b611877764ec5aULL, 0xb8e8fd9c51baa419ULL, 0xa972487992e10f5bULL, 0x4bb2d020bc88827aULL,
    0x22299892ced54bc3ULL, 0x16d422500ef7ab08ULL, 0x38c378c757c9d122ULL, 0xffd457d5c4445875ULL,
    0xbb8ad4d17887e53bULL};
  int pbad = 0, have = 0;
  for (int c = 0; c < NSC; c++) {
    float yy[4096]; crdist_stream s = {1, 0, 0};
    ldraws(SCASES[c].k, SCASES[c].a, SCASES[c].b, yy, 4096, &s);
    uint64_t h = fnv(yy, 4096);
    printf("  fingerprint %-11s %-5g %-4g 0x%016llx\n", SNAME[SCASES[c].k], SCASES[c].a, SCASES[c].b, (unsigned long long)h);
    if (PRINT[c]) { have++; pbad += h != PRINT[c]; }
  }
  if (have) { printf("samplers: %d of %d fingerprints differ from the recorded ones\n", pbad, have); if (pbad) r = 1; }
  return r;
}

/* hash-all: every output of every function on all 2^32 binary32 inputs, hashed (NaNs made one NaN, as the checks
   count any two NaNs equal), against the digests of the build that crdist-check all proved against MPFR. Equal
   digests carry that proof to this build in minutes (a sampled check cannot). Two 64-bit hashes a function, over
   4096 blocks of 2^20 inputs in order (FNV-1a, and a multiply-xorshift), combined in block order. */
static void block_hash(int f, uint64_t b, uint64_t h[2])
{
  enum { NB = 1 << 20 };
  float *x = malloc(2 * NB * sizeof *x), *y = x + NB;   /* not thread-local: static TLS comes out of each thread's stack */
  for (uint32_t i = 0; i < NB; i++) x[i] = flt((b << 20) | i);
  FN[f](y, x, NB);
  uint64_t a = 0xcbf29ce484222325ULL, c = 0x9e3779b97f4a7c15ULL;
  for (uint32_t i = 0; i < NB; i++) {
    uint32_t u = y[i] != y[i] ? 0x7fc00000u : bits(y[i]);
    for (int k = 0; k < 4; k++) { a ^= (u >> (8 * k)) & 0xff; a *= 0x100000001b3ULL; }
    c = (c ^ u) * 0xbf58476d1ce4e5b9ULL; c ^= c >> 31;
  }
  free(x);
  h[0] = a; h[1] = c;
}
static int hash_all(void)
{
  /* recorded 2026-10-13 on cfarm421 (x86-64, gcc 14) from the builds crdist-check all proved: 42432d0 (erfinv,
     erfcinv, probit) and 85ef764 (all seven), the same digests from both and from the branch that added this */
  static const uint64_t DIGEST[CRDIST_NFN][2] = {
    {0xeae91e37b5257860ULL, 0x2380874719e515c4ULL}, {0x43dd9d061ae17771ULL, 0xe4c276980bdd880eULL},
    {0xa9d2e179fe65fa28ULL, 0xf2d2bdffd0a67bf9ULL}, {0x371fab1c17c019daULL, 0xdad664ec43b26e14ULL},
    {0x7ab6f6eebbf046f3ULL, 0x0fc1a918b9a2d028ULL}, {0x49628d5fe32f05d5ULL, 0xc8bf4d19c155233fULL},
    {0x56e549204ec50ee3ULL, 0xbacb466e0f66d6d6ULL}};
  static uint64_t bh[CRDIST_NFN][4096][2];
  int r = 0, have = 0;
  for (int f = 0; f < CRDIST_NFN; f++) {
    if (!(fnmask >> f & 1)) continue;
#pragma omp parallel for schedule(dynamic, 4)
    for (int b = 0; b < 4096; b++) block_hash(f, (uint64_t)b, bh[f][b]);
    uint64_t d0 = 0xcbf29ce484222325ULL, d1 = 0;
    for (int b = 0; b < 4096; b++) {
      for (int k = 0; k < 8; k++) { d0 ^= (bh[f][b][0] >> (8 * k)) & 0xff; d0 *= 0x100000001b3ULL; }
      d1 = (d1 ^ bh[f][b][1]) * 0x94d049bb133111ebULL; d1 ^= d1 >> 29;
    }
    int rec = DIGEST[f][0] || DIGEST[f][1], same = d0 == DIGEST[f][0] && d1 == DIGEST[f][1];
    printf("hash, %-8s every input: 0x%016llx 0x%016llx%s\n", crdist_fn_name[f], (unsigned long long)d0, (unsigned long long)d1,
           !rec ? " (nothing recorded)" : same ? ", the proven build's" : ", NOT the proven build's");
    if (rec) { have++; if (!same) r = 1; }
  }
  /* the control: one output moved by an ulp must change the digest */
  uint64_t h0[2], h1[2];
  block_hash(CRDIST_NDTR, 2000, h0);
  block_hash(CRDIST_NDTR, 2000, h1);
  int ctl_same = h0[0] == h1[0] && h0[1] == h1[1];   /* the same block twice: the same */
  uint64_t a = h0[0], c = h0[1];
  { /* recompute with one value moved */
    enum { NB = 1 << 20 };
    static float x[NB], y[NB];
    for (uint32_t i = 0; i < NB; i++) x[i] = flt((2000ULL << 20) | i);
    FN[CRDIST_NDTR](y, x, NB);
    y[12345] = flt(bits(y[12345]) ^ 1);
    a = 0xcbf29ce484222325ULL; c = 0x9e3779b97f4a7c15ULL;
    for (uint32_t i = 0; i < NB; i++) {
      uint32_t u = y[i] != y[i] ? 0x7fc00000u : bits(y[i]);
      for (int k = 0; k < 4; k++) { a ^= (u >> (8 * k)) & 0xff; a *= 0x100000001b3ULL; }
      c = (c ^ u) * 0xbf58476d1ce4e5b9ULL; c ^= c >> 31;
    }
  }
  int ctl_moved = a != h0[0] && c != h0[1];
  printf("hash control: a block hashed twice the same %s; with one output moved by an ulp, both hashes change %s\n",
         ctl_same ? "yes" : "NO", ctl_moved ? "yes" : "NO");
  if (!ctl_same || !ctl_moved) r = kit_worst(r, 2);
  if (!have) r = kit_worst(r, 2);
  return r;
}

int main(int argc, char **argv)
{
  int all = argc > 1 && !strcmp(argv[1], "all"), bound_all = argc > 1 && !strcmp(argv[1], "bound-all");
  int hashing = argc > 1 && !strcmp(argv[1], "hash-all");
  if ((all || bound_all || hashing) && argc > 2) {   /* only the functions named */
    fnmask = 0;
    for (int i = 2; i < argc; i++) {
      int f = 0;
      while (f < CRDIST_NFN && strcmp(argv[i], crdist_fn_name[f])) f++;
      if (f == CRDIST_NFN) { fprintf(stderr, "crdist-check: no function %s\n", argv[i]); return 2; }
      fnmask |= 1u << f;
    }
  }
  if (hashing) return kit_verdict(hash_all(), "this build gives the proven build's output on every binary32 input of each function run, and the hash controls differ");
  int r = sec1();
  if (bound_all) return kit_verdict(kit_worst(r, sec7(1)), "the binary64 bound holds on every binary32 input of each function run, and every control differs");
  r = kit_worst(r, sec2(all));
  r = kit_worst(r, sec7(0));
  if (!all) {
    r = kit_worst(r, sec3());
    r = kit_worst(r, sec4());
    r = kit_worst(r, sec5());
    r = kit_worst(r, sec6());
    r = kit_worst(r, sec8());
    r = kit_worst(r, sec9());
    r = kit_worst(r, sec9d());
  }
  return kit_verdict(r, all ? "every binary32 input of each function run is correctly rounded, the binary64 bound holds on the sample, and every control differs"
                            : "crdist is correctly rounded where it says so, its binary64 bound holds, its samplers follow their specification, its results are the same in any environment, and every control differs");
}
