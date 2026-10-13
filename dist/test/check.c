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
   The last line is the verdict.

     crdist-check            sections 1-7, sampled (minutes of CPU)
     crdist-check all        sections 1, 2 on every input, and 7 (hundreds of hours of CPU: MPFR's erfc is slow far out)
     crdist-check bound-all  sections 1 and 7 on every input */
#include <fenv.h>
#include <float.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "kit.h"
#include "crdist.h"
#include "crdist-fast.h"
#include "crdist-ref.h"
#include "crdist-exceptions.h"

static const kit_mpfr1 REF[CRDIST_NFN] = {crdist_mpfr_erfinv, crdist_mpfr_erfcinv, crdist_mpfr_probit};
typedef void (*arr1)(float *, const float *, size_t);
static const arr1 FN[CRDIST_NFN] = {crdist_erfinvf, crdist_erfcinvf, crdist_probitf};

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
    default: return bits(x < 0.5f ? -1.4142135f * plain_ierf(2 * x, 1) : 1.4142135f * plain_ierf(2 - 2 * x, 1));
  }
}

static int sec2(int all)
{
  int r = 0;
  for (cur = 0; cur < CRDIST_NFN; cur++) {
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
  for (int f = 0; f < CRDIST_NFN; f++)
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
      decided += crdist_round32(crdist_fast(f, x), &z);
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
static int sec7(int all)
{
  int r = 0;
  for (int f = 0; f < CRDIST_NFN; f++) {
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
        crdist_eqn e; crdist_eqn_at(&e, v, f, x);
        double d = fabs(crdist_fast(f, x));
        n++;
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

int main(int argc, char **argv)
{
  int all = argc > 1 && !strcmp(argv[1], "all"), bound_all = argc > 1 && !strcmp(argv[1], "bound-all");
  int r = sec1();
  if (bound_all) return kit_verdict(kit_worst(r, sec7(1)), "the binary64 bound holds on every binary32 input of erfinv, erfcinv and probit, and every control differs");
  r = kit_worst(r, sec2(all));
  r = kit_worst(r, sec7(0));
  if (!all) {
    r = kit_worst(r, sec3());
    r = kit_worst(r, sec4());
    r = kit_worst(r, sec5());
    r = kit_worst(r, sec6());
  }
  return kit_verdict(r, all ? "every binary32 input of erfinv, erfcinv and probit is correctly rounded, the binary64 bound holds on the sample, and every control differs"
                            : "crdist is correctly rounded where it says so, its binary64 bound holds, its results are the same in any environment, and every control differs");
}
