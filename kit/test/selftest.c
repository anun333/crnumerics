/* selftest: the kit (numerics/kit/kit.h) against answers it did not make.

     A  formats: every encoding of the 4- to 16-bit formats and a sample
        of binary32, decoded, against the hardware's own conversion
        (binary32, binary16, bfloat16; E5M2 is binary16's top byte) or
        OCP's definitions (E4M3, and the MX element formats E2M3, E3M2,
        E2M1), and encoded back
     B  rounding: kit_round against a brute-force search through every
        value of each 4- to 16-bit format (E4M3 in both overflow modes)
        at every value, every midpoint and either side of it, and beyond
        the largest; binary32 and binary64 against the hardware's
        conversion, product and sum, which are correctly rounded
     C  the reference: kit_ref1 and kit_ref2 against CORE-MATH's binary16
        and bfloat16 functions, which are correctly rounded and were
        proven so without this kit; every input of the one-argument
        functions, and 2^16 pairs plus every pair of edge values of the
        two-argument ones
     D  binary32 and binary64 samples against CORE-MATH's expf, logf,
        sinf, atan2f, exp and log
     F  the 4-, 6- and 8-bit formats: every input (every pair) against
        CORE-MATH's binary64 function rounded again to the format (see
        there why rounding twice is safe)
     G  coverage: an exhaustive run visits every input of each format of 16
        bits or fewer exactly once, and an every-pair run every pair of each
        format of 8 bits or fewer (candidates that count what they are
        handed; a comparison can't see an input left out)
     E  a negative control: CORE-MATH's binary16 exp judged against MPFR's
        exp2 must differ

   B to F run in all four rounding modes. Every check has a control that
   must differ: the runner's own for C, D, F and E, and for A and B the
   same comparison against a wrong answer (a neighbouring encoding).
   About 12 seconds on four cores.

   Its own check, done once (2026-09-30): eleven bugs planted one at a
   time in fmt.c (the ternary value ignored by check_range or by
   subnormalize, no subnormalize, each end of the exponent range off by
   one, E4M3's overflow rules, the signaling NaN rule, a subnormal scale,
   a blind perturbation, a comparison that always agrees). Each made this
   fail: DIFFERS, VOID or an abort, and every rounding bug in B alone.
   The one that did not, the signaling NaN rule for one argument, changed
   nothing (MPFR gives NaN for every NaN) and was removed. */
#include <fenv.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "kit.h"

static const mpfr_rnd_t RND[4] = {MPFR_RNDN, MPFR_RNDU, MPFR_RNDD, MPFR_RNDZ};

static uint64_t rng = 0x243f6a8885a308d3ULL;
static uint64_t next(void)
{
  uint64_t z = (rng += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

/* a tally by hand, for B: the check, and its control against a
   neighbouring answer on about one point in 64. The inputs are binary64
   values and the results a format's encodings; the first difference is
   kept as binary64 encodings, for kit_report. */
typedef struct { kit_tally run, ctl; unsigned long long k; } hand;
static void tick(hand *h, kit_fmt f, int args, mpfr_rnd_t rnd, double x, double y, uint64_t got, uint64_t want)
{
  kit_tally *t[2] = {&h->run, &h->ctl};
  uint64_t g[2] = {got, h->k++ % 64 == 17 ? kit_perturb(f, got) : got};
  for (int i = 0; i < 2; i++) {
    t[i]->args = args;
    t[i]->tested++;
    if (kit_same(f, g[i], want)) continue;
    if (!t[i]->differ++) {
      t[i]->first_rnd = rnd;
      t[i]->first_x = kit_encode(KIT_B64, x);
      t[i]->first_y = kit_encode(KIT_B64, y);
      t[i]->first_got = kit_encode(KIT_B64, kit_decode(f, g[i]));
      t[i]->first_want = kit_encode(KIT_B64, kit_decode(f, want));
    }
  }
}

/* ---- A: formats ---- */

static double hw(kit_fmt f, uint64_t b)
{
  switch (f) {
  case KIT_B32: { uint32_t u = (uint32_t)b; float x; memcpy(&x, &u, 4); return x; }
  case KIT_B16: { uint16_t u = (uint16_t)b; _Float16 x; memcpy(&x, &u, 2); return x; }
  case KIT_BF16: { uint32_t u = (uint32_t)b << 16; float x; memcpy(&x, &u, 4); return x; }
  case KIT_E5M2: return hw(KIT_B16, b << 8);
  case KIT_E4M3: {   /* as OCP defines it: S.EEEE.MMM, bias 7, S.1111.111 the NaN */
    int s = b >> 7 & 1, e = b >> 3 & 15, m = b & 7;
    double v = e == 15 && m == 7 ? NAN : e ? (8 + m) / 8.0 * pow(2, e - 7) : m / 8.0 * pow(2, -6);
    return s ? -v : v;
  }
  case KIT_E2M3: {   /* OCP MX: S.EE.MMM, bias 1, no infinity or NaN */
    int s = b >> 5 & 1, e = b >> 3 & 3, m = b & 7;
    double v = e ? (8 + m) / 8.0 * pow(2, e - 1) : m / 8.0;
    return s ? -v : v;
  }
  case KIT_E3M2: {   /* OCP MX: S.EEE.MM, bias 3 */
    int s = b >> 5 & 1, e = b >> 2 & 7, m = b & 3;
    double v = e ? (4 + m) / 4.0 * pow(2, e - 3) : m / 4.0 * pow(2, -2);
    return s ? -v : v;
  }
  default: {   /* E2M1, from OCP MX's table of its eight magnitudes */
    static const double V[8] = {0, 0.5, 1, 1.5, 2, 3, 4, 6};
    return b & 8 ? -V[b & 7] : V[b & 7];
  }
  }
}
static int same_value(double a, double b) { return (isnan(a) && isnan(b)) || (a == b && signbit(a) == signbit(b)); }

static int sec_a(void)
{
  int r = 0;
  /* values from OCP's tables: the largest, the smallest normal and subnormal, 1, and the specials */
  static const struct { kit_fmt f; uint64_t b; double v; } OCP[] = {
    {KIT_E4M3, 0x7e, 448}, {KIT_E4M3, 0x08, 0x1p-6}, {KIT_E4M3, 0x01, 0x1p-9}, {KIT_E4M3, 0x38, 1}, {KIT_E4M3, 0x77, 240},
    {KIT_E4M3, 0x7f, NAN}, {KIT_E4M3, 0xff, NAN}, {KIT_E4M3, 0x80, -0.0}, {KIT_E5M2, 0x7b, 57344}, {KIT_E5M2, 0x04, 0x1p-14},
    {KIT_E5M2, 0x01, 0x1p-16}, {KIT_E5M2, 0x3c, 1}, {KIT_E5M2, 0x7c, INFINITY}, {KIT_E5M2, 0x7d, NAN}, {KIT_E5M2, 0xfc, -INFINITY},
    {KIT_E2M3, 0x1f, 7.5}, {KIT_E2M3, 0x08, 1}, {KIT_E2M3, 0x01, 0.125}, {KIT_E2M3, 0x3f, -7.5},
    {KIT_E3M2, 0x1f, 28}, {KIT_E3M2, 0x04, 0.25}, {KIT_E3M2, 0x01, 0.0625}, {KIT_E3M2, 0x0c, 1},
    {KIT_E2M1, 0x7, 6}, {KIT_E2M1, 0x1, 0.5}, {KIT_E2M1, 0x2, 1}, {KIT_E2M1, 0xf, -6},
  };
  static const double MAX[KIT_NFMT] = {DBL_MAX, FLT_MAX, 65504, 0x1.fep127, 57344, 448, 7.5, 28, 6};
  unsigned n = 0, bad = 0;
  for (unsigned k = 0; k < sizeof OCP / sizeof *OCP; k++, n++) bad += !same_value(kit_decode(OCP[k].f, OCP[k].b), OCP[k].v);
  for (kit_fmt f = 0; f < KIT_NFMT; f++, n++) bad += kit_max(f) != MAX[f];
  printf("%-30s %12u values checked, %u differ\n", "OCP's tables, the maxima", n, bad);
  if (bad) r = 1;
  for (kit_fmt f = KIT_B32; f < KIT_NFMT; f++) {
    hand h = {0};
    unsigned long long n = kit_info(f)->bits <= 16 ? 1ULL << kit_info(f)->bits : 1ULL << 22;
    for (unsigned long long k = 0; k < n; k++) {
      uint64_t b = kit_info(f)->bits <= 16 ? k : (uint32_t)next();
      double v = kit_decode(f, b), w = hw(f, b), wrong = kit_decode(f, b ^ 1);
      int bad = !same_value(v, w) || (isnan(v) ? !kit_isnan(f, kit_encode(f, v)) : kit_encode(f, v) != b) ||
                !kit_isnan(f, b) != !isnan(v);
      h.run.tested++; h.run.differ += bad; h.ctl.differ += !same_value(wrong, w);
      if (bad && h.run.differ == 1) { h.run.first_x = b; h.run.first_got = kit_encode(f, isnan(v) ? 0 : v); h.run.first_want = b; }
    }
    h.run.args = 1;
    char what[64];
    snprintf(what, sizeof what, "decode, encode %s", kit_info(f)->name);
    r = kit_worst(r, kit_report(what, f, h.run, h.ctl));
  }
  /* binary64: every encoding decodes to itself; the round trip, sampled */
  hand h = {0};
  for (int k = 0; k < 1 << 22; k++) {
    uint64_t b = next();
    double v = kit_decode(KIT_B64, b);
    uint64_t back = kit_encode(KIT_B64, v);
    h.run.tested++; h.run.differ += isnan(v) ? !kit_isnan(KIT_B64, back) : back != b;
    h.ctl.differ += kit_encode(KIT_B64, kit_decode(KIT_B64, b ^ 1)) != b;   /* a neighbour is not b */
  }
  r = kit_worst(r, kit_report("decode, encode binary64", KIT_B64, h.run, h.ctl));
  return r;
}

/* ---- B: rounding ---- */

/* the format's nonnegative finite values in order, then the virtual next
   one: the largest plus an ulp, the first value that overflows */
static double P[1 << 16];
static int par[1 << 16], np;
static void build(kit_fmt f)
{
  np = 0;
  for (uint64_t b = 0; b < 1ULL << (kit_info(f)->bits - 1); b++) {
    double v = kit_decode(f, b);
    if (isnan(v) || isinf(v)) continue;
    if (np && v <= P[np - 1]) { printf("build: %s not increasing at 0x%llx\n", kit_info(f)->name, (unsigned long long)b); exit(2); }
    P[np] = v;
    par[np++] = b & 1;
  }
  int e;
  frexp(kit_max(f), &e);
  P[np] = kit_max(f) + ldexp(1, e - 1 - kit_info(f)->mbits);
  /* 2^(emax+1) is even; E4M3's 1.111 * 2^8, below its NaN, is odd */
  par[np++] = kit_info(f)->has_nan && !kit_info(f)->has_inf;
}

/* x rounded to the format by search: the neighbours below and above, and
   in round-to-nearest the nearer, or on a tie the one whose last bit is
   0; the virtual next value overflows */
static uint64_t brute(kit_fmt f, double x, mpfr_rnd_t rnd)
{
  double max = kit_max(f), V = P[np - 1], a = fabs(x), r = 0;
  if (isnan(x)) return kit_info(f)->has_nan ? kit_encode(f, NAN) : KIT_NONE;   /* the MX formats have none */
  int neg = signbit(x), over = 0;
  int dir = rnd == MPFR_RNDN ? 0 : rnd == MPFR_RNDZ ? -1 : (rnd == MPFR_RNDU) != neg ? 1 : -1;   /* in magnitude */
  if (isinf(a)) { if (kit_info(f)->has_inf) r = a; else over = 1; }
  else if (a >= V) { if (dir < 0) r = max; else over = 1; }
  else {
    int lo = 0, hi = np - 1;   /* the first P[j] >= a */
    while (lo < hi) { int m = (lo + hi) / 2; if (P[m] >= a) hi = m; else lo = m + 1; }
    int j = lo;
    if (P[j] == a) r = a;
    else {
      double mid = (P[j - 1] + P[j]) / 2;
      int up = dir > 0 || (dir == 0 && (a > mid || (a == mid && par[j - 1])));
      r = up ? P[j] : P[j - 1];
      over = up && j == np - 1;
    }
  }
  if (over) {
    if (kit_info(f)->has_inf) r = INFINITY;
    else if (kit_e4m3_saturate || !kit_info(f)->has_nan) r = max;   /* the MX formats saturate */
    else return kit_encode(f, NAN);   /* E4M3's NaN, checked against OCP's in A */
  }
  return kit_encode(f, neg ? -r : r);
}

static uint64_t round_d(kit_fmt f, double x, mpfr_rnd_t rnd)
{
  mpfr_t m;
  mpfr_init2(m, 53);
  mpfr_set_d(m, x, MPFR_RNDN);
  uint64_t b = kit_round(f, m, rnd);
  mpfr_clear(m);
  return b;
}

static void small_points(kit_fmt f, hand *h)
{
  build(f);
  double V = P[np - 1], out[] = {V, V + (V - P[np - 2]) * 0x1p-20, 2 * V, 1e300, INFINITY, NAN};
  for (int m = 0; m < 4; m++)
    for (int s = 0; s < 2; s++) {
      for (int j = 0; j + 1 < np; j++) {
        double lo = P[j], mid = (P[j] + P[j + 1]) / 2, d = (P[j + 1] - P[j]) * 0x1p-20;
        double pt[4] = {lo, mid - d, mid, mid + d};
        for (int k = 0; k < 4; k++) {
          double x = s ? -pt[k] : pt[k];
          tick(h, f, 1, RND[m], x, 0, round_d(f, x, RND[m]), brute(f, x, RND[m]));
        }
      }
      for (unsigned k = 0; k < sizeof out / sizeof *out; k++) {
        double x = s ? -out[k] : out[k];
        tick(h, f, 1, RND[m], x, 0, round_d(f, x, RND[m]), brute(f, x, RND[m]));
      }
    }
}

static double random_double(int emin, int emax)
{
  uint64_t a = next();
  int e = emin + (int)(a % (uint64_t)(emax - emin + 1));
  double m = (double)((next() >> 11) | 1ULL << 52);   /* 53 bits */
  return (a >> 40 & 1 ? -1 : 1) * ldexp(m, e - 52);
}

static int sec_b(void)
{
  int r = 0;
  static const kit_fmt SMALL[] = {KIT_E5M2, KIT_E4M3, KIT_E4M3, KIT_B16, KIT_BF16, KIT_E2M3, KIT_E3M2, KIT_E2M1};
  for (unsigned k = 0; k < sizeof SMALL / sizeof *SMALL; k++) {
    hand h = {0};
    kit_e4m3_saturate = k == 2;
    small_points(SMALL[k], &h);
    char what[64];
    snprintf(what, sizeof what, "round %s%s", kit_info(SMALL[k])->name, k == 1 ? " (NaN)" : k == 2 ? " (saturating)" : "");
    r = kit_worst(r, kit_report(what, KIT_B64, h.run, h.ctl));
  }
  kit_e4m3_saturate = 0;
  /* binary32: the hardware's conversion from binary64, at random values
     from below the subnormals to beyond the largest, at midpoints, and
     either side of one */
  hand h = {0};
  for (int m = 0; m < 4; m++) {
    fesetround(kit_fenv_of(RND[m]));
    for (int k = 0; k < 1 << 20; k++) {
      double x;
      if (k % 3 == 0) x = random_double(-155, 129);
      else {
        uint32_t u = k % 4096 == 1 ? 0x7f7fffff : (uint32_t)(next() % 0x7f800000);   /* finite; the largest */
        float a;
        memcpy(&a, &u, 4);
        double ulp = u < 0x00800000 ? 0x1p-149 : ldexp(1, (int)(u >> 23) - 150);
        x = (double)a + ulp / 2;
        if (k % 3 == 2) x += (next() & 1 ? 1 : -1) * ulp * 0x1p-30;
        if (next() & 1) x = -x;
      }
      volatile double vx = x;
      volatile float y = (float)vx;
      float yy = y;
      uint32_t b;
      memcpy(&b, &yy, 4);
      tick(&h, KIT_B32, 1, RND[m], x, 0, round_d(KIT_B32, x, RND[m]), b);
    }
    fesetround(FE_TONEAREST);
  }
  r = kit_worst(r, kit_report("round binary32 (hardware)", KIT_B64, h.run, h.ctl));
  /* binary64: the hardware's product (random, and a value times a power
     of two falling into the subnormals, where ties are common) and sum
     (ties and near-ties of every exponent, the largest included) */
  hand g = {0};
  mpfr_t a, b, s;
  mpfr_inits2(256, a, b, s, (mpfr_ptr)0);
  for (int m = 0; m < 4; m++) {
    for (int k = 0; k < 1 << 20; k++) {
      double x, y;
      int sum = k % 3 == 2;
      if (k % 3 == 0) {
        int t = -1130 + (int)(next() % 2161);
        x = random_double(t / 2 - 3, t / 2 + 3);
        y = random_double(t - t / 2 - 3, t - t / 2 + 3);
      } else if (k % 3 == 1) {
        x = random_double(-1022, -990);
        y = ldexp(1, -(int)(next() % 64) - 1);
      } else {
        x = k % 1024 == 2 ? DBL_MAX : random_double(-1020, 1023);
        int e;
        frexp(x, &e);
        y = ldexp(1, e - 54) * (next() & 1 ? 1 : -1);
        if (next() & 1) y *= 1 + (next() & 1 ? 0x1p-20 : -0x1p-20);
      }
      mpfr_set_d(a, x, MPFR_RNDN);
      mpfr_set_d(b, y, MPFR_RNDN);
      if (sum) mpfr_add(s, a, b, MPFR_RNDN); else mpfr_mul(s, a, b, MPFR_RNDN);   /* exact */
      fesetround(kit_fenv_of(RND[m]));
      volatile double vx = x, vy = y;
      volatile double z = sum ? vx + vy : vx * vy;
      fesetround(FE_TONEAREST);
      double zz = z;
      tick(&g, KIT_B64, 2, RND[m], x, y, kit_round(KIT_B64, s, RND[m]), kit_encode(KIT_B64, zz));
    }
  }
  mpfr_clears(a, b, s, (mpfr_ptr)0);
  r = kit_worst(r, kit_report("round binary64 (hardware)", KIT_B64, g.run, g.ctl));
  return r;
}

/* ---- C: the reference against CORE-MATH's binary16 and bfloat16 ---- */

#define H1(f)                                                                                          \
  _Float16 cr_##f##f16(_Float16);                                                                      \
  __bf16 cr_##f##_bf16(__bf16);                                                                        \
  static uint64_t h_##f(uint64_t x) { uint16_t u = (uint16_t)x; _Float16 a; memcpy(&a, &u, 2); a = cr_##f##f16(a); memcpy(&u, &a, 2); return u; } \
  static uint64_t b_##f(uint64_t x) { uint16_t u = (uint16_t)x; __bf16 a; memcpy(&a, &u, 2); a = cr_##f##_bf16(a); memcpy(&u, &a, 2); return u; }
#define H2(f)                                                                                          \
  _Float16 cr_##f##f16(_Float16, _Float16);                                                            \
  __bf16 cr_##f##_bf16(__bf16, __bf16);                                                                \
  static uint64_t h_##f(uint64_t x, uint64_t y)                                                        \
  { uint16_t u = (uint16_t)x, v = (uint16_t)y; _Float16 a, b; memcpy(&a, &u, 2); memcpy(&b, &v, 2); a = cr_##f##f16(a, b); memcpy(&u, &a, 2); return u; } \
  static uint64_t b_##f(uint64_t x, uint64_t y)                                                        \
  { uint16_t u = (uint16_t)x, v = (uint16_t)y; __bf16 a, b; memcpy(&a, &u, 2); memcpy(&b, &v, 2); a = cr_##f##_bf16(a, b); memcpy(&u, &a, 2); return u; }
#define HSC(f)
#include "crmvec-f16-list.h"

/* where IEEE and MPFR name or define a function differently (kit/fns.c) */
#define mpfr_lgamma kit_mpfr_lgamma
#define mpfr_tgamma mpfr_gamma
#define mpfr_rsqrt kit_mpfr_rsqrt

static const struct { const char *name; kit_cand1 h, b; kit_mpfr1 ref; } F1[] = {
#define H1(f) {#f, h_##f, b_##f, mpfr_##f},
#define H2(f)
#define HSC(f)
#include "crmvec-f16-list.h"
};
static const struct { const char *name; kit_cand2 h, b; kit_mpfr2 ref; } F2[] = {
#define H1(f)
#define H2(f) {#f, h_##f, b_##f, mpfr_##f},
#define HSC(f)
#include "crmvec-f16-list.h"
};

static int sec_c(void)
{
  int r = 0;
  for (unsigned k = 0; k < sizeof F1 / sizeof *F1; k++)
    for (int bf = 0; bf < 2; bf++) {
      kit_fmt f = bf ? KIT_BF16 : KIT_B16;
      kit_tally run = {0}, ctl = {0};
      for (int m = 0; m < 4; m++) {
        kit_tally c, t = kit_exhaust1(f, bf ? F1[k].b : F1[k].h, F1[k].ref, RND[m], &c);
        kit_tally_add(&run, t);
        kit_tally_add(&ctl, c);
      }
      char what[64];
      snprintf(what, sizeof what, "%s %s", F1[k].name, kit_info(f)->name);
      r = kit_worst(r, kit_report(what, f, run, ctl));
    }
  for (unsigned k = 0; k < sizeof F2 / sizeof *F2; k++)
    for (int bf = 0; bf < 2; bf++) {
      kit_fmt f = bf ? KIT_BF16 : KIT_B16;
      kit_tally run = {0}, ctl = {0};
      for (int m = 0; m < 4; m++) {
        kit_tally c, t = kit_sample2(f, bf ? F2[k].b : F2[k].h, F2[k].ref, RND[m], 1 << 16, 0x9e37 + k, &c);
        kit_tally_add(&run, t);
        kit_tally_add(&ctl, c);
      }
      char what[64];
      snprintf(what, sizeof what, "%s %s", F2[k].name, kit_info(f)->name);
      r = kit_worst(r, kit_report(what, f, run, ctl));
    }
  return r;
}

/* ---- D: binary32 and binary64 samples ---- */

float cr_expf(float), cr_logf(float), cr_sinf(float), cr_atan2f(float, float);
double cr_exp(double), cr_log(double), cr_sin(double), cr_tgamma(double), cr_erfc(double), cr_pow(double, double),
  cr_atan2(double, double), cr_hypot(double, double);
#define S1(f)                                                                                          \
  static uint64_t s_##f(uint64_t x) { uint32_t u = (uint32_t)x; float a; memcpy(&a, &u, 4); a = cr_##f(a); memcpy(&u, &a, 4); return u; }
#define D1(f)                                                                                          \
  static uint64_t d_##f(uint64_t x) { double a; memcpy(&a, &x, 8); a = cr_##f(a); memcpy(&x, &a, 8); return x; }
S1(expf) S1(logf) S1(sinf) D1(exp) D1(log)
static uint64_t s_atan2f(uint64_t x, uint64_t y)
{ uint32_t u = (uint32_t)x, v = (uint32_t)y; float a, b; memcpy(&a, &u, 4); memcpy(&b, &v, 4); a = cr_atan2f(a, b); memcpy(&u, &a, 4); return u; }

static int sample(const char *what, kit_fmt f, kit_cand1 c1, kit_cand2 c2, kit_mpfr1 r1, kit_mpfr2 r2, unsigned long long n)
{
  kit_tally run = {0}, ctl = {0};
  for (int m = 0; m < 4; m++) {
    kit_tally c, t = c1 ? kit_sample1(f, c1, r1, RND[m], n, 0xd1b54a32d192ed03ULL + m, &c)
                        : kit_sample2(f, c2, r2, RND[m], n, 0xd1b54a32d192ed03ULL + m, &c);
    kit_tally_add(&run, t);
    kit_tally_add(&ctl, c);
  }
  return kit_report(what, f, run, ctl);
}

static int sec_d(void)
{
  int r = 0;
  r = kit_worst(r, sample("expf binary32", KIT_B32, s_expf, 0, mpfr_exp, 0, 1 << 18));
  r = kit_worst(r, sample("logf binary32", KIT_B32, s_logf, 0, mpfr_log, 0, 1 << 18));
  r = kit_worst(r, sample("sinf binary32", KIT_B32, s_sinf, 0, mpfr_sin, 0, 1 << 18));
  r = kit_worst(r, sample("atan2f binary32", KIT_B32, 0, s_atan2f, 0, mpfr_atan2, 1 << 18));
  r = kit_worst(r, sample("exp binary64", KIT_B64, d_exp, 0, mpfr_exp, 0, 1 << 16));
  r = kit_worst(r, sample("log binary64", KIT_B64, d_log, 0, mpfr_log, 0, 1 << 16));
  return r;
}

/* ---- F: the 8-bit formats through CORE-MATH's binary64 functions ----
   The candidate rounds CORE-MATH's binary64 result (correctly rounded)
   again, to the 8-bit format, in the same mode. Rounding twice in one
   direction is rounding once. To nearest it can differ only when the
   binary64 result is a midpoint of the 8-bit format (at most 5
   significant bits) and the exact result is not: when an 8-bit format's
   inputs give a result within 2^-53 of such a number without being it.
   For these functions that happens nowhere: the exact results that are
   short binary numbers (sqrt(4), pow(1.5, 2), hypot(3, 4)) are exact in
   binary64 too. */

static kit_fmt ff;
static double (*g1)(double);
static double (*g2)(double, double);
static mpfr_rnd_t mode_now(void)
{
  int m = fegetround();
  return m == FE_UPWARD ? MPFR_RNDU : m == FE_DOWNWARD ? MPFR_RNDD : m == FE_TOWARDZERO ? MPFR_RNDZ : MPFR_RNDN;
}
static int snan8(uint64_t x) { return ff == KIT_E5M2 && kit_isnan(ff, x) && !(x & 2); }
static uint64_t none8(void) { return kit_info(ff)->has_nan ? kit_encode(ff, NAN) : KIT_NONE; }   /* the MX formats: none */
static uint64_t twice(double v) { return round_d(ff, v, mode_now()); }
static uint64_t via1(uint64_t x) { return snan8(x) ? none8() : twice(g1(kit_decode(ff, x))); }
static uint64_t via2(uint64_t x, uint64_t y)
{ return snan8(x) || snan8(y) ? none8() : twice(g2(kit_decode(ff, x), kit_decode(ff, y))); }
static double libm_sqrt(double x) { return sqrt(x); }

static int sec_f(void)
{
  static const struct { const char *name; double (*g1)(double); kit_mpfr1 r1; double (*g2)(double, double); kit_mpfr2 r2; } G[] = {
    {"exp", cr_exp, mpfr_exp, 0, 0}, {"log", cr_log, mpfr_log, 0, 0}, {"sin", cr_sin, mpfr_sin, 0, 0},
    {"tgamma", cr_tgamma, mpfr_gamma, 0, 0}, {"erfc", cr_erfc, mpfr_erfc, 0, 0}, {"sqrt", libm_sqrt, mpfr_sqrt, 0, 0},
    {"pow", 0, 0, cr_pow, mpfr_pow},
    {"atan2", 0, 0, cr_atan2, mpfr_atan2}, {"hypot", 0, 0, cr_hypot, mpfr_hypot},
  };
  static const kit_fmt FF[6] = {KIT_E5M2, KIT_E4M3, KIT_E4M3, KIT_E2M3, KIT_E3M2, KIT_E2M1};
  int r = 0;
  for (int k = 0; k < 6; k++) {
    ff = FF[k];
    kit_e4m3_saturate = k == 2;
    for (unsigned j = 0; j < sizeof G / sizeof *G; j++) {
      g1 = G[j].g1;
      g2 = G[j].g2;
      kit_tally run = {0}, ctl = {0};
      for (int m = 0; m < 4; m++) {
        kit_tally c, t = g1 ? kit_exhaust1(ff, via1, G[j].r1, RND[m], &c) : kit_sample2(ff, via2, G[j].r2, RND[m], 0, 0, &c);
        kit_tally_add(&run, t);
        kit_tally_add(&ctl, c);
      }
      char what[64];
      snprintf(what, sizeof what, "%s %s%s", G[j].name, kit_info(ff)->name, k == 1 ? " (NaN)" : k == 2 ? " (saturating)" : "");
      r = kit_worst(r, kit_report(what, ff, run, ctl));
    }
  }
  kit_e4m3_saturate = 0;
  return r;
}

/* ---- G: coverage ---- */

static kit_fmt gf;
static unsigned char seen[1 << 16];
static int twice_seen;
static uint64_t count1(uint64_t x)
{
  unsigned char v;
#pragma omp atomic capture
  v = seen[x]++;
  if (v) twice_seen = 1;
  return kit_ref1(gf, mpfr_sqrt, x, mode_now());
}
static uint64_t count2(uint64_t x, uint64_t y)
{
  unsigned char v;
#pragma omp atomic capture
  v = seen[x << kit_info(gf)->bits | y]++;
  if (v) twice_seen = 1;
  return kit_ref2(gf, mpfr_hypot, x, y, mode_now());
}

static int sec_g(void)
{
  int r = 0;
  for (gf = KIT_B16; gf < KIT_NFMT; gf++)
    for (int args = 1; args <= 2; args++) {
      int bits = kit_info(gf)->bits * args;
      if (bits > 16) continue;
      memset(seen, 0, sizeof seen);
      twice_seen = 0;
      kit_tally c, t = args == 1 ? kit_exhaust1(gf, count1, mpfr_sqrt, MPFR_RNDN, &c)
                                 : kit_sample2(gf, count2, mpfr_hypot, MPFR_RNDN, 0, 0, &c);
      unsigned long long missed = 0;
      for (unsigned long long k = 0; k < 1ULL << bits; k++) missed += !seen[k];
      int ok = !missed && !twice_seen && t.tested == 1ULL << bits;
      char what[64];
      snprintf(what, sizeof what, "%s, every %s", kit_info(gf)->name, args == 1 ? "input" : "pair");
      printf("%-30s %12llu tested, %llu of %llu never handed over%s%s\n", what, t.tested, missed, 1ULL << bits,
             twice_seen ? ", some twice" : "", ok ? "" : "  NOT COVERED");
      if (!ok) r = 1;
    }
  return r;
}

/* ---- E: the negative control ---- */

static int sec_e(void)
{
  kit_tally c, t = kit_exhaust1(KIT_B16, h_exp, mpfr_exp2, MPFR_RNDN, &c);
  int r = kit_report("exp binary16 against exp2", KIT_B16, t, c);
  printf("%s\n", r == 1 ? "  (differs, as it must)" : "  NEGATIVE CONTROL FAILED: this must differ");
  return r == 1 ? 0 : 2;
}

int main(void)
{
  int r = 0;
  printf("A: formats\n");
  r = kit_worst(r, sec_a());
  printf("B: rounding, four modes\n");
  r = kit_worst(r, sec_b());
  printf("C: the reference against CORE-MATH's binary16 and bfloat16, four modes\n");
  r = kit_worst(r, sec_c());
  printf("D: binary32 and binary64 samples against CORE-MATH, four modes\n");
  r = kit_worst(r, sec_d());
  printf("F: 4-, 6- and 8-bit formats against CORE-MATH's binary64 rounded again, four modes\n");
  r = kit_worst(r, sec_f());
  printf("G: coverage of the exhaustive and every-pair runs\n");
  r = kit_worst(r, sec_g());
  printf("E: negative control\n");
  r = kit_worst(r, sec_e());
  return kit_verdict(r, "the kit agrees with every independent answer, and every control differs");
}
