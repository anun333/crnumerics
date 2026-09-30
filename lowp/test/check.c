/* check: lowp (lowp.h) on every input it has, in every mode.

     1  every one-argument function, every input, four modes, in E5M2 and
        E4M3 each with and without saturation: against the kit's MPFR
        reference (kit_exhaust1)
     2  the same against a second path that shares nothing with the first:
        CORE-MATH's binary64 function in the same rounding mode, rounded by
        lowp's own conversion (the four functions CORE-MATH has no binary64
        version of, exp10m1, exp2m1, log10p1 and log2p1, take MPFR at 200
        bits rounded to binary64 in the same direction instead; sqrt takes
        the C library's, which IEEE 754 requires to be correctly rounded)
     3  every two-argument function on every pair, four modes, the same
        four configurations: against the kit's reference
     4  the conversions from binary64 and binary32: every value of each
        format, every midpoint and either side of one, beyond the largest,
        the specials, and random values from far below the subnormals to
        far above the largest: against the kit's rounding
     5  a negative control (E5M2 exp judged against MPFR's exp2 must differ),
        and every mode lowp doesn't take must be refused, nothing written

   Every run has the kit's control, which must differ. The last line is the
   verdict.

   Its own check, done once (2026-09-30): ten bugs planted one at a time in
   lowp.c and lowp-tables.h:
   - in the rounding: the tie's parity, overflow toward zero, the
     subnormals' quantum, saturation;
   - in the functions: E4M3's overflow mask, the signaling NaN rule, a
     subnormal's decoding, E5M2's mode ignored, one table entry;
   - two() left in round-to-nearest.
   Nine made this DIFFER. The survivor, two() in round-to-nearest, changes
   no result on any pair, since no pair's exact result lies within 2^-53 of
   an FP8 value without being one. two() keeps the matching mode anyway:
   rounding twice in one direction is exact without that property.

   E5M2's saturating mode (added 2026-09-30, the fourth configuration): two
   bugs planted in lowp.c (the tables' infinities not replaced; round8
   keeping an infinite input's infinity) each made this DIFFER, and one in
   the kit (its E5M2 flag ignored) made the kit's self-test DIFFER. */
#include <fenv.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "kit.h"
#include "lowp.h"

static const mpfr_rnd_t RND[4] = {MPFR_RNDN, MPFR_RNDU, MPFR_RNDD, MPFR_RNDZ};
enum { NCFG = 4 };
static const kit_fmt FMT[NCFG] = {KIT_E5M2, KIT_E4M3, KIT_E4M3, KIT_E5M2};
static const char *CFG[NCFG] = {"E5M2", "E4M3", "E4M3 saturating", "E5M2 saturating"};
/* configuration c: E4M3 or E5M2, saturating or not; the kit's flags to match */
static int is4(int c) { return c == 1 || c == 2; }
static int sat(int c) { return c >= 2; }
static void kit_cfg(int c) { kit_e4m3_saturate = c == 2; kit_e5m2_saturate = c == 3; }

typedef int (*lf1)(const uint8_t *, uint8_t *, size_t, int);
typedef int (*lf2)(const uint8_t *, const uint8_t *, uint8_t *, size_t, int);
static const struct { const char *name; lf1 e5m2, e4m3; } L1[] = {
#define LOWP_F1(f) {#f, lowp_e5m2_##f, lowp_e4m3_##f},
#define LOWP_F2(f)
#include "lowp-list.h"
};
static const struct { const char *name; lf2 e5m2, e4m3; } L2[] = {
#define LOWP_F1(f)
#define LOWP_F2(f) {#f, lowp_e5m2_##f, lowp_e4m3_##f},
#include "lowp-list.h"
};
enum { N1 = sizeof L1 / sizeof *L1, N2 = sizeof L2 / sizeof *L2 };

/* CORE-MATH's binary64 functions, by lowp's names; 0 where it has none */
double cr_acos(double), cr_acosh(double), cr_acospi(double), cr_asin(double), cr_asinh(double), cr_asinpi(double),
  cr_atan(double), cr_atanh(double), cr_atanpi(double), cr_cbrt(double), cr_cos(double), cr_cosh(double),
  cr_cospi(double), cr_erf(double), cr_erfc(double), cr_exp(double), cr_exp10(double), cr_exp2(double),
  cr_expm1(double), cr_lgamma(double), cr_log(double), cr_log10(double), cr_log1p(double), cr_log2(double),
  cr_rsqrt(double), cr_sin(double), cr_sinh(double), cr_sinpi(double), cr_tan(double), cr_tanh(double),
  cr_tanpi(double), cr_tgamma(double);
static double c_sqrt(double x) { return sqrt(x); }
static double (*const B64[N1])(double) = {
  cr_acos, cr_acosh, cr_acospi, cr_asin, cr_asinh, cr_asinpi, cr_atan, cr_atanh, cr_atanpi, cr_cbrt, cr_cos,
  cr_cosh, cr_cospi, cr_erf, cr_erfc, cr_exp, cr_exp10, 0 /* exp10m1 */, cr_exp2, 0 /* exp2m1 */, cr_expm1,
  cr_lgamma, cr_log, cr_log10, 0 /* log10p1 */, cr_log1p, cr_log2, 0 /* log2p1 */, cr_rsqrt, cr_sin, cr_sinh,
  cr_sinpi, c_sqrt, cr_tan, cr_tanh, cr_tanpi, cr_tgamma};

/* the current function and configuration, for the candidates below (read
   only while a run is in flight) */
static int cur, cfg;
static int mode_now(void)
{
  int m = fegetround();
  return (m == FE_UPWARD ? 1 : m == FE_DOWNWARD ? 2 : m == FE_TOWARDZERO ? 3 : 0) | (sat(cfg) ? LOWP_SAT : 0);
}
static uint64_t cand1(uint64_t x)
{
  uint8_t a = (uint8_t)x, b = 0;
  (is4(cfg) ? L1[cur].e4m3 : L1[cur].e5m2)(&a, &b, 1, mode_now());
  return b;
}
static uint64_t cand2(uint64_t x, uint64_t y)
{
  uint8_t a = (uint8_t)x, b = (uint8_t)y, c = 0;
  (is4(cfg) ? L2[cur].e4m3 : L2[cur].e5m2)(&a, &b, &c, 1, mode_now());
  return c;
}

/* a tally by hand, with the control moving one result in 64 */
static void tick(kit_tally *run, kit_tally *ctl, kit_fmt f, mpfr_rnd_t rnd, uint64_t x, uint64_t got, uint64_t want,
                 unsigned long long k)
{
  kit_tally *t[2] = {run, ctl};
  uint64_t g[2] = {got, k % 64 == 29 ? kit_perturb(f, got) : got};
  for (int i = 0; i < 2; i++) {
    t[i]->args = 1;
    t[i]->tested++;
    if (kit_same(f, g[i], want)) continue;
    if (!t[i]->differ++) { t[i]->first_rnd = rnd; t[i]->first_x = x; t[i]->first_got = g[i]; t[i]->first_want = want; }
  }
}

static int sec1(void)
{
  int r = 0;
  for (cur = 0; cur < N1; cur++) {
    kit_tally run = {0}, ctl = {0};
    kit_fmt first = KIT_E4M3;
    for (cfg = 0; cfg < NCFG; cfg++) {
      kit_cfg(cfg);
      for (int m = 0; m < 4; m++) {
        kit_tally c, t = kit_exhaust1(FMT[cfg], cand1, kit_fns1[cur].ref, RND[m], &c);
        if (!run.differ && t.differ) first = FMT[cfg];
        kit_tally_add(&run, t);
        kit_tally_add(&ctl, c);
      }
    }
    kit_cfg(0);
    char what[64];
    snprintf(what, sizeof what, "%s, 4 configurations", L1[cur].name);
    r = kit_worst(r, kit_report(what, first, run, ctl));
  }
  return r;
}

static int sec2(void)
{
  static const int FE[4] = {FE_TONEAREST, FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO};
  int r = 0;
  mpfr_t a, y;
  mpfr_init2(a, 53);
  mpfr_init2(y, 200);
  for (cur = 0; cur < N1; cur++) {
    kit_tally run = {0}, ctl = {0};
    kit_fmt first = KIT_E4M3;
    unsigned long long k = 0;
    for (cfg = 0; cfg < NCFG; cfg++)
      for (int m = 0; m < 4; m++) {
        int mode = m | (sat(cfg) ? LOWP_SAT : 0);
        if (!run.differ) first = FMT[cfg];
        for (int x = 0; x < 256; x++) {
          uint8_t in = (uint8_t)x, got, want;
          double v, d;
          (is4(cfg) ? lowp_e4m3_to_f64 : lowp_e5m2_to_f64)(&in, &v, 1);
          int snan = !is4(cfg) && (x & 0x7f) == 0x7d;
          if (B64[cur]) {
            fesetround(FE[m]);
            d = B64[cur](v);
            fesetround(FE_TONEAREST);
          } else {   /* MPFR at 200 bits, then to binary64 in the same direction */
            mpfr_set_d(a, v, MPFR_RNDN);
            kit_fns1[cur].ref(y, a, RND[m]);
            d = mpfr_get_d(y, RND[m]);
          }
          (is4(cfg) ? lowp_e4m3_from_f64 : lowp_e5m2_from_f64)(&d, &want, 1, mode);
          if (snan) want = 0x7e;
          (is4(cfg) ? L1[cur].e4m3 : L1[cur].e5m2)(&in, &got, 1, mode);
          tick(&run, &ctl, FMT[cfg], RND[m], (uint64_t)x, got, want, k++);
        }
      }
    char what[64];
    snprintf(what, sizeof what, "%s, %s", L1[cur].name, B64[cur] ? "CORE-MATH binary64" : "MPFR 200 bits");
    r = kit_worst(r, kit_report(what, first, run, ctl));
  }
  mpfr_clears(a, y, (mpfr_ptr)0);
  return r;
}

static int sec3(void)
{
  int r = 0;
  for (cur = 0; cur < N2; cur++) {
    kit_tally run = {0}, ctl = {0};
    kit_fmt first = KIT_E4M3;
    for (cfg = 0; cfg < NCFG; cfg++) {
      kit_cfg(cfg);
      for (int m = 0; m < 4; m++) {
        kit_tally c, t = kit_sample2(FMT[cfg], cand2, kit_fns2[cur].ref, RND[m], 0, 0, &c);
        if (!run.differ && t.differ) first = FMT[cfg];
        kit_tally_add(&run, t);
        kit_tally_add(&ctl, c);
      }
    }
    kit_cfg(0);
    char what[64];
    snprintf(what, sizeof what, "%s, every pair, 4 configurations", L2[cur].name);
    r = kit_worst(r, kit_report(what, first, run, ctl));
  }
  return r;
}

static uint64_t rng = 0x5851f42d4c957f2dULL;
static uint64_t next(void)
{
  uint64_t z = (rng += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

/* the conversion of v, in format cfg and mode m, against kit_round */
static void conv(kit_tally *run, kit_tally *ctl, double v, int m, unsigned long long *k)
{
  int mode = m | (sat(cfg) ? LOWP_SAT : 0);
  uint8_t got, g32;
  (is4(cfg) ? lowp_e4m3_from_f64 : lowp_e5m2_from_f64)(&v, &got, 1, mode);
  mpfr_t x;
  mpfr_init2(x, 53);
  mpfr_set_d(x, v, MPFR_RNDN);
  uint64_t want = kit_round(FMT[cfg], x, RND[m]);
  mpfr_clear(x);
  tick(run, ctl, FMT[cfg], RND[m], kit_encode(KIT_B64, v), got, want, (*k)++);
  float f = (float)v;
  if ((double)f == v || isnan(v)) {   /* binary32 takes the same value */
    (is4(cfg) ? lowp_e4m3_from_f32 : lowp_e5m2_from_f32)(&f, &g32, 1, mode);
    tick(run, ctl, FMT[cfg], RND[m], kit_encode(KIT_B64, v), g32, want, (*k)++);
  }
}

static int sec4(void)
{
  int r = 0;
  for (cfg = 0; cfg < NCFG; cfg++) {
    kit_cfg(cfg);
    kit_tally run = {0}, ctl = {0};
    unsigned long long k = 0;
    kit_fmt f = FMT[cfg];
    double P[256];   /* the format's nonnegative finite values, in order */
    int np = 0;
    for (int b = 0; b < 128; b++) {
      double v = kit_decode(f, (uint64_t)b);
      if (!isnan(v) && !isinf(v)) P[np++] = v;
    }
    double next_up = 2 * P[np - 1] - P[np - 2];   /* the largest plus an ulp */
    for (int m = 0; m < 4; m++)
      for (int s = 0; s < 2; s++) {
        double sg = s ? -1 : 1;
        for (int j = 0; j < np; j++) {
          double hi = j + 1 < np ? P[j + 1] : next_up, mid = (P[j] + hi) / 2, d = (hi - P[j]) * 0x1p-40;
          double pts[5] = {P[j], mid - d, mid, mid + d, nextafter(mid, 0)};
          for (int i = 0; i < 5; i++) conv(&run, &ctl, sg * pts[i], m, &k);
        }
        double far[] = {next_up, 2 * next_up, 1e30, DBL_MAX, INFINITY, NAN, 0x1p-1074, 0x1p-1022, 1e-30, P[1] / 2,
                        P[1] / 2 * (1 + 0x1p-52), P[1] / 2 * (1 - 0x1p-53), P[1] / 4};
        for (unsigned i = 0; i < sizeof far / sizeof *far; i++) conv(&run, &ctl, sg * far[i], m, &k);
        for (int i = 0; i < 1 << 16; i++) {   /* random: every binary64 exponent, weighted to the format's */
          uint64_t a = next();
          int e = a & 1 ? (int)((a >> 1) & 2047) : 1023 - 40 + (int)((a >> 1) % 64);   /* biased */
          if (e == 2047) e = 2046;
          uint64_t b = (uint64_t)s << 63 | (uint64_t)e << 52 | (next() >> 12);
          double v;
          memcpy(&v, &b, 8);
          conv(&run, &ctl, v, m, &k);
        }
      }
    char what[64];
    snprintf(what, sizeof what, "from binary64 and binary32, %s", CFG[cfg]);
    r = kit_worst(r, kit_report(what, KIT_B64, run, ctl));
  }
  kit_cfg(0);
  return r;
}

static int sec5(void)
{
  cur = 15;   /* exp */
  cfg = 0;
  kit_tally c, t = kit_exhaust1(KIT_E5M2, cand1, mpfr_exp2, MPFR_RNDN, &c);
  int neg = kit_report("exp E5M2 against exp2", KIT_E5M2, t, c) == 1;
  printf("  %s\n", neg ? "(differs, as it must)" : "NEGATIVE CONTROL FAILED: this must differ");
  /* modes lowp doesn't take */
  uint8_t in[2] = {0x38, 0x3c}, out[2] = {0xaa, 0xaa};
  double dv = 1;
  float fv = 1;
  int refused = lowp_e5m2_exp(in, out, 2, 8) == -1 && lowp_e5m2_exp(in, out, 2, -1) == -1 &&
                lowp_e4m3_exp(in, out, 2, 8) == -1 && lowp_e5m2_pow(in, in, out, 2, 8) == -1 &&
                lowp_e4m3_pow(in, in, out, 2, -1) == -1 && lowp_e5m2_from_f64(&dv, out, 1, 8) == -1 &&
                lowp_e4m3_from_f32(&fv, out, 1, -1) == -1 && out[0] == 0xaa && out[1] == 0xaa;
  printf("%-30s %s\n", "modes it doesn't take", refused ? "refused, nothing written" : "NOT REFUSED");
  return neg && refused ? 0 : 2;
}

int main(void)
{
  if (N1 != kit_nfns1 || N2 != kit_nfns2) { printf("VERDICT: VOID: lowp-list.h and the kit's lists differ\n"); return 2; }
  if (strcmp(L1[15].name, "exp")) { printf("VERDICT: VOID: exp is not function 15\n"); return 2; }
  int r = 0;
  printf("1: one-argument functions against MPFR, every input, four modes\n");
  r = kit_worst(r, sec1());
  printf("2: one-argument functions against CORE-MATH's binary64, rounded by lowp's conversion\n");
  r = kit_worst(r, sec2());
  printf("3: two-argument functions against MPFR, every pair, four modes\n");
  r = kit_worst(r, sec3());
  printf("4: conversions against the kit's rounding, four modes\n");
  r = kit_worst(r, sec4());
  printf("5: negative control, refused modes\n");
  r = kit_worst(r, sec5());
  return kit_verdict(r, "lowp is correctly rounded on every input and pair, and every control differs");
}
