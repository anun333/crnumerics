/* check: ival (ival.h) against an MPFR reference that finds the bounds its
   own way.

   The reference, for an interval [a, b] and a function f:
     - intersects [a, b] with f's domain, written out here again (not
       shared with ival.c);
     - evaluates f by MPFR at both ends, rounding down and up, without
       assuming f is monotone: the lower bound is the least, the upper the
       greatest;
     - adds f's critical points and poles inside [a, b], found from their
       definitions: pi/2 + k pi (sin's extrema, tan's poles) and k pi
       (cos's) with pi to 2,200 bits, which is enough for any binary64
       argument; k + 1/2 and k for sinpi, cospi and tanpi; 0 for cosh.
       An extremum gives its value (1 or -1), a pole inside gives
       [-inf, +inf], and a pole at an end gives that side's limit.
   ival works differently (derivative signs at the ends of short pieces).
   The intervals tried:
     points      [x, x] at special values and 2^14 random binary64 values
     random      2^14 per function, widths from one ulp to 10^3, at every
                 magnitude
     critical    around sin's and cos's extrema and tan's poles, near and far
                 (the binary64 value nearest a multiple of pi/2 included),
                 around sinpi's, cospi's and tanpi's, exactly on them too
     domain      across the domain's edges, outside it, on it
     specials    infinite ends, NaN ends, lo > hi
   Each bound is compared as a number (+0 = -0; both NaN when empty). A
   second check needs no reference: at 8 points inside each interval, the
   exact f (MPFR, 300 bits) must lie within the bounds. Every run has a
   control: a bound moved by an ulp in one interval in 64, and in at least
   one. tgamma (2026-10-02): the reference has no table. It counts the
   poles exactly, finds each extremum by bisection on digamma's sign, and
   takes signs from lgamma. Its control is built by hand: ival.c with
   -DIVAL_PLANT_TG puts segment 3's minimum an ulp low, and 9 intervals
   must differ.

   Two-argument functions (atan2, hypot and pow) take boxes: the intervals above,
   paired at random and each against the specials. The reference evaluates
   by MPFR at every pair of candidate points, each interval's ends and its
   critical points (for hypot, 0 when the interval holds it: the only
   place its gradient (x, y)/r lets a minimum sit off a corner; for
   atan2, 0 in x, and in y both +0, its value on the axis, and -0, the
   limit from below, which reaches -pi on the negative x-axis), never the
   origin, where atan2 is undefined; for pow, x = 1 and y = 0 inside, and
   IEEE 1788's domain, x > 0 or x = 0 with y > 0, written out again). It
   assumes nothing about magnitudes or where the cut is. pow's negative
   control forgets that x reaches 0 (the least positive value instead). The same 8-point and control checks,
   and a negative control: the corners alone must differ. The last line is
   the verdict. */
#include <fenv.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "kit.h"
#include "ival.h"

typedef void (*ivf)(const double *, const double *, double *, double *, size_t);
static const struct { const char *name; ivf f; } L[] = {
#define IVAL_F(f) {#f, ival_##f},
#include "ival-list.h"
};
enum { NL = sizeof L / sizeof *L };
typedef void (*ivf2)(const double *, const double *, const double *, const double *, double *, double *, size_t);
static const struct { const char *name; ivf2 f; } L2[] = {
#define IVAL_F2(f) {#f, ival_##f},
#include "ival-list.h"
};
enum { NL2 = sizeof L2 / sizeof *L2 };

/* each function's domain [lo, hi], open where the function has a pole or
   no limit at the end; its critical points, as a kind */
enum { NONE, CRIT_COSH, CRIT_SIN, CRIT_COS, CRIT_TAN, CRIT_SINPI, CRIT_COSPI, CRIT_TANPI, CRIT_GAMMA };
typedef struct { const char *name; double lo, hi; int lo_open, hi_open, crit; } dom;
static const dom D[] = {
  {"acos", -1, 1, 0, 0, NONE},        {"acosh", 1, INFINITY, 0, 0, NONE}, {"acospi", -1, 1, 0, 0, NONE},
  {"asin", -1, 1, 0, 0, NONE},        {"asinpi", -1, 1, 0, 0, NONE},      {"atanh", -1, 1, 1, 1, NONE},
  {"log", 0, INFINITY, 1, 0, NONE},   {"log10", 0, INFINITY, 1, 0, NONE}, {"log2", 0, INFINITY, 1, 0, NONE},
  {"log1p", -1, INFINITY, 1, 0, NONE}, {"rsqrt", 0, INFINITY, 1, 0, NONE}, {"sqrt", 0, INFINITY, 0, 0, NONE},
  {"cosh", -INFINITY, INFINITY, 0, 0, CRIT_COSH}, {"sin", -INFINITY, INFINITY, 0, 0, CRIT_SIN},
  {"cos", -INFINITY, INFINITY, 0, 0, CRIT_COS},   {"tan", -INFINITY, INFINITY, 0, 0, CRIT_TAN},
  {"sinpi", -INFINITY, INFINITY, 0, 0, CRIT_SINPI}, {"cospi", -INFINITY, INFINITY, 0, 0, CRIT_COSPI},
  {"tanpi", -INFINITY, INFINITY, 0, 0, CRIT_TANPI}, {"tgamma", -INFINITY, INFINITY, 0, 0, CRIT_GAMMA},
};
static dom domain_of(const char *n)
{
  for (unsigned i = 0; i < sizeof D / sizeof *D; i++) if (!strcmp(D[i].name, n)) return D[i];
  return (dom){n, -INFINITY, INFINITY, 0, 0, NONE};
}
static kit_mpfr1 ref_of(const char *n)
{
  for (int i = 0; i < kit_nfns1; i++) if (!strcmp(kit_fns1[i].name, n)) return kit_fns1[i].ref;
  return 0;
}

/* f(x) correctly rounded to binary64 in direction rnd, by the kit */
static double eval(kit_mpfr1 r, double x, mpfr_rnd_t rnd)
{ return kit_decode(KIT_B64, kit_ref1(KIT_B64, r, kit_encode(KIT_B64, x), rnd)); }

/* the critical points c + k u in [a, b], u = pi or 1, c = u/2 or 0: how
   many (*count), whether a and b are ones, and the parity of the first
   k; exact, by MPFR at 2,200 bits (enough for any binary64 against pi) */
static void crit(double a, double b, int half, int with_pi, double *count, int *at_a, int *at_b, int *even)
{
  mpfr_t u, t, v, w;
  mpfr_inits2(2200, u, t, v, w, (mpfr_ptr)0);
  if (with_pi) mpfr_const_pi(u, MPFR_RNDN); else mpfr_set_ui(u, 1, MPFR_RNDN);
  mpfr_div_2ui(w, u, 1, MPFR_RNDN);   /* u / 2 */
  mpfr_set_d(t, a, MPFR_RNDN);
  if (half) mpfr_sub(t, t, w, MPFR_RNDN);
  mpfr_div(t, t, u, MPFR_RNDN);   /* (a - c) / u, exact for u = 1 */
  *at_a = !with_pi && mpfr_integer_p(t);
  mpfr_ceil(t, t);   /* the first k */
  mpfr_set_d(v, b, MPFR_RNDN);
  if (half) mpfr_sub(v, v, w, MPFR_RNDN);
  mpfr_div(v, v, u, MPFR_RNDN);
  *at_b = !with_pi && mpfr_integer_p(v);
  mpfr_floor(v, v);   /* the last k */
  mpfr_sub(v, v, t, MPFR_RNDN);
  *count = mpfr_get_d(v, MPFR_RNDN) + 1;
  mpfr_fmod_ui(w, t, 2, MPFR_RNDN);
  *even = mpfr_zero_p(w);
  mpfr_clears(u, t, v, w, (mpfr_ptr)0);
}

/* tgamma's reference (2026-10-02), its own way: the poles (0, -1, -2, ...) in [a, b] counted by MPFR as integers
   (crit, u = 1); the sign of Gamma inside, for a pole at an end, from MPFR at the interval's midpoint; an extremum
   inside where digamma goes from negative to positive between the ends (it increases from -inf to +inf on each
   pole-free piece), found by bisection on digamma's sign at 400 bits and evaluated there at 400 bits. ival.c uses a
   table of the extrema instead. */
static int digamma_sign(double x)
{
  /* MPFR rounds correctly, so the sign is exact at any output precision: 64 bits */
  mpfr_t t, y;
  mpfr_inits2(64, t, y, (mpfr_ptr)0);
  mpfr_set_d(t, x, MPFR_RNDN);
  mpfr_digamma(y, t, MPFR_RNDN);
  int sg = mpfr_sgn(y);
  mpfr_clears(t, y, (mpfr_ptr)0);
  return sg;
}
/* the extremum in (lo, hi) (digamma < 0 at lo, > 0 at hi, either may be a pole), to 400 bits: its value rounded
   down and up, and the root itself rounded down (for the intervals around it) */
static void gamma_ext(double lo, double hi, double *vd, double *vu, mpfr_t root)
{
  /* the root to 240 bits: Gamma is flat there (its error goes as the square of the root's), so the value comes out
     good to 400; each step's sign of digamma is exact at a 32-bit output (correct rounding) */
  mpfr_t l, h, m, y, s;
  mpfr_inits2(240, l, h, m, (mpfr_ptr)0);
  mpfr_init2(s, 32);
  mpfr_init2(y, 400);
  mpfr_set_d(l, lo, MPFR_RNDN);
  mpfr_set_d(h, hi, MPFR_RNDN);
  for (int i = 0; i < 250; i++) {
    mpfr_add(m, l, h, MPFR_RNDN);
    mpfr_div_2ui(m, m, 1, MPFR_RNDN);
    mpfr_digamma(s, m, MPFR_RNDN);
    if (mpfr_sgn(s) < 0) mpfr_set(l, m, MPFR_RNDN); else mpfr_set(h, m, MPFR_RNDN);
  }
  mpfr_clear(s);
  /* by lgamma, which cannot underflow: past n = 180 or so Gamma there is below MPFR's exponent range */
  int sg;
  mpfr_lgamma(y, &sg, l, MPFR_RNDN);
  if (mpfr_cmp_d(y, -745.2) < 0) {   /* |Gamma| < 2^-1075: rounds to 0 or the least subnormal */
    *vd = sg > 0 ? 0.0 : -0x1p-1074;
    *vu = sg > 0 ? 0x1p-1074 : -0.0;
  } else {
    mpfr_gamma(y, l, MPFR_RNDN);
    *vd = mpfr_get_d(y, MPFR_RNDD);
    *vu = mpfr_get_d(y, MPFR_RNDU);
  }
  if (root) mpfr_set(root, l, MPFR_RNDN);
  mpfr_clears(l, h, m, y, (mpfr_ptr)0);
}
static void ref_gamma(kit_mpfr1 r, double a, double b, double *lo, double *hi)
{
  *lo = *hi = NAN;
  if (!(a <= b)) return;
  if (a == 0) a = 0.0;
  if (b == 0) b = 0.0;
  int pole_a = a <= 0 && (isinf(a) || floor(a) == a), pole_b = b <= 0 && (isinf(b) || floor(b) == b);
  if (a == b) {
    if (pole_a) return;   /* a pole alone (or -inf): empty */
    *lo = eval(r, a, MPFR_RNDD);
    *hi = eval(r, a, MPFR_RNDU);
    return;
  }
  if (isinf(a)) { *lo = -INFINITY; *hi = INFINITY; return; }   /* every pole below b */
  if (a <= 0) {   /* the poles in [a, min(b, 0)], and whether any is strictly inside [a, b] */
    double cnt;
    int ata, atb, even;
    crit(a, b > 0 ? 0 : b, 0, 0, &cnt, &ata, &atb, &even);
    if (cnt > ata + (b <= 0 && atb)) { *lo = -INFINITY; *hi = INFINITY; return; }
  }
  double l = INFINITY, h = -INFINITY;
  if (!pole_a) { l = eval(r, a, MPFR_RNDD); h = eval(r, a, MPFR_RNDU); }
  if (!pole_b) {
    double l2 = eval(r, b, MPFR_RNDD), h2 = eval(r, b, MPFR_RNDU);
    if (l2 < l) l = l2;
    if (h2 > h) h = h2;
  }
  if (pole_a || pole_b) {   /* the limit at a pole end, from inside: Gamma's sign there */
    mpfr_t m, y;
    mpfr_inits2(2200, m, y, (mpfr_ptr)0);
    mpfr_set_d(m, a, MPFR_RNDN);
    mpfr_set_d(y, b, MPFR_RNDN);
    mpfr_add(m, m, y, MPFR_RNDN);
    mpfr_div_2ui(m, m, 1, MPFR_RNDN);
    int sg;
    mpfr_lgamma(y, &sg, m, MPFR_RNDN);   /* the sign by lgamma: Gamma itself underflows MPFR far out */
    if (sg > 0) h = INFINITY; else l = -INFINITY;
    mpfr_clears(m, y, (mpfr_ptr)0);
  }
  int dl = pole_a ? -1 : digamma_sign(a), dh = pole_b ? 1 : isinf(b) ? 1 : digamma_sign(b);
  if (dl < 0 && dh > 0) {
    double vd, vu;
    gamma_ext(a, isinf(b) ? 2 : b, &vd, &vu, NULL);
    if (vd < l) l = vd;
    if (vu > h) h = vu;
  }
  *lo = l;
  *hi = h;
}

/* the reference [*lo, *hi] of f over [a, b] */
static void reference(const dom *d, kit_mpfr1 r, double a, double b, double *lo, double *hi)
{
  if (d->crit == CRIT_GAMMA) { ref_gamma(r, a, b, lo, hi); return; }
  *lo = *hi = NAN;
  if (!(a <= b)) return;
  if (a < d->lo) a = d->lo;
  if (b > d->hi) b = d->hi;
  if (!(a <= b) || (d->lo_open && b <= d->lo) || (d->hi_open && a >= d->hi)) return;
  if (a == 0) a = 0.0;
  if (d->crit == CRIT_TANPI && a == b && fabs(a) < 0x1p52 && fmod(fabs(a), 1) == 0.5) return;   /* only a pole */
  /* the ends, but not an end that is a pole (tanpi at a half-integer):
     only its limit from inside counts, added below */
  int pole_a = d->crit == CRIT_TANPI && fabs(a) < 0x1p52 && fmod(fabs(a), 1) == 0.5;
  int pole_b = d->crit == CRIT_TANPI && fabs(b) < 0x1p52 && fmod(fabs(b), 1) == 0.5;
  double l = INFINITY, h = -INFINITY;
  if (!pole_a) { l = eval(r, a, MPFR_RNDD); h = eval(r, a, MPFR_RNDU); }
  if (!pole_b) {
    double l2 = eval(r, b, MPFR_RNDD), h2 = eval(r, b, MPFR_RNDU);
    if (l2 < l) l = l2;
    if (h2 > h) h = h2;
  }
  switch (d->crit) {
  case NONE: break;
  case CRIT_COSH:
    if (a <= 0 && 0 <= b) l = 1;
    break;
  default: {   /* sin, cos, tan: pi/2 + k pi, k pi; sinpi, cospi, tanpi: k + 1/2, k */
    int pole = d->crit == CRIT_TAN || d->crit == CRIT_TANPI;
    if (isinf(a) || isinf(b)) {
      if (pole) { l = -INFINITY; h = INFINITY; } else { l = -1; h = 1; }
      break;
    }
    double count;
    int at_a, at_b, even;
    int half = d->crit != CRIT_COS && d->crit != CRIT_COSPI, with_pi = d->crit <= CRIT_TAN;
    crit(a, b, half, with_pi, &count, &at_a, &at_b, &even);
    if (count <= 0) break;
    if (pole) {   /* tan's poles are never binary64 values; tanpi's are */
      if (count > at_a + at_b) { l = -INFINITY; h = INFINITY; }
      if (at_a) l = -INFINITY;
      if (at_b) h = INFINITY;
    } else if (count >= 2) { l = -1; h = 1; }   /* a maximum and a minimum */
    else {   /* sin(pi/2 + k pi) = cos(k pi) = sinpi(k + 1/2) = cospi(k) = (-1)^k */
      double v = even ? 1 : -1;
      if (v < l) l = v;
      if (v > h) h = v;
    }
    break;
  }
  }
  *lo = l;
  *hi = h;
}

static uint64_t rng = 0x94d049bb133111ebULL;
static uint64_t next(void)
{
  uint64_t z = (rng += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}
static double rand_double(void)   /* every binary64 exponent equally likely, both signs */
{
  uint64_t a = next(), b = (a & 1ULL << 63) | ((a >> 20) % 2047) << 52 | (next() >> 12);
  double d;
  memcpy(&d, &b, 8);
  return d;
}
static double rand_mag(double lo_exp, double hi_exp)   /* 2^e, e uniform */
{ return ldexp(1 + (double)(next() >> 11) * 0x1p-53, (int)(lo_exp + (double)(next() % 1000) / 1000 * (hi_exp - lo_exp))); }

/* the intervals to try, for a function */
static double *A, *B;
static size_t n, cap;
static void add(double a, double b)
{
  if (n == cap) { cap = cap ? 2 * cap : 1 << 15; A = realloc(A, cap * sizeof *A); B = realloc(B, cap * sizeof *B); }
  A[n] = a;
  B[n++] = b;
}
static void intervals(const dom *d)
{
  n = 0;
  static const double SPECIAL[] = {0, -0.0, 1, -1, 0.5, -0.5, 2, -2, 1e-310, -1e-310, DBL_MIN, DBL_MAX, -DBL_MAX,
                                   DBL_TRUE_MIN, 0x1p-30, 710, -745, 1e22, 0x1.921fb54442d18p+0, 3.141592653589793,
                                   0x1.921fb54442d18p+1, 0x1.921fb54442d18p+2, 6381956970095103.0 * 0x1p797,
                                   INFINITY, -INFINITY, 0.25, 0.75, 1.5, 4503599627370495.5, 0x1p52, 0x1p53};
  for (unsigned i = 0; i < sizeof SPECIAL / sizeof *SPECIAL; i++) add(SPECIAL[i], SPECIAL[i]);
  for (int i = 0; i < 1 << 14; i++) { double x = rand_double(); add(x, x); }   /* points */
  for (int i = 0; i < 1 << 14; i++) {   /* random widths at random magnitudes */
    double a = (next() & 1 ? -1 : 1) * rand_mag(-60, 60), w;
    switch (next() % 4) {
    case 0: w = nextafter(a, INFINITY) - a; break;
    case 1: w = fabs(a) * rand_mag(-50, -1); break;
    case 2: w = rand_mag(-10, 10); break;
    default: w = rand_mag(-3, 3);
    }
    add(a, a + w);
  }
  if (d->crit == CRIT_SIN || d->crit == CRIT_COS || d->crit == CRIT_TAN) {   /* around pi/2 multiples */
    double h = 0x1.921fb54442d18p+0;   /* pi/2 rounded */
    for (int k = -40; k <= 40; k++) {
      double c = k * h;
      for (int j = -3; j <= 3; j++) {
        double e = nextafter(c, j < 0 ? -INFINITY : INFINITY);
        for (int s = 0; s < abs(j); s++) e = nextafter(e, j < 0 ? -INFINITY : INFINITY);
        add(fmin(c, e), fmax(c, e));
        add(c - 0.1, c + 0.1);
        add(c - 1e-9, c);
        add(c, c + 1e-9);
      }
    }
    double far = 6381956970095103.0 * 0x1p797;   /* the binary64 nearest a multiple of pi/2 */
    add(nextafter(far, 0), far);
    add(far, nextafter(far, INFINITY));
    add(nextafter(far, 0), nextafter(far, INFINITY));
    for (int i = 0; i < 512; i++) { double a = rand_mag(20, 1000); add(a, nextafter(nextafter(a, INFINITY), INFINITY)); }
    add(0, 7); add(0, 6.3); add(-3.2, 3.2); add(1, 7.9); add(-8, 0); add(100, 108);
  }
  if (d->crit == CRIT_SINPI || d->crit == CRIT_COSPI || d->crit == CRIT_TANPI) {   /* around integers and halves */
    for (int k = -20; k <= 20; k++)
      for (int hlf = 0; hlf < 2; hlf++) {
        double c = k + 0.5 * hlf;
        add(c, c);
        add(c, c + 0.25);
        add(c - 0.25, c);
        add(c - 0.25, c + 0.25);
        add(nextafter(c, -INFINITY), nextafter(c, INFINITY));
        add(c - 0.75, c + 0.5);
      }
    add(0, 3.9); add(0, 4); add(0.5, 2.5); add(-1.5, 1.5); add(0x1p52, 0x1p52 + 3); add(0x1p53, 0x1p53 + 8);
    add(4503599627370495.5, 4503599627370496.0);
  }
  if (d->crit == CRIT_COSH) { add(-1e-300, 1e-300); add(-5, 0); add(0, 5); add(-3, 4); add(-0.0, 0.0); }
  if (d->crit == CRIT_GAMMA) {   /* the poles, the extrema (found here by bisection) and their neighbours */
    mpfr_t root;
    mpfr_init2(root, 420);
    for (int k = -1; k < 200; k++) {   /* k = -1: the minimum on the positives, in (1, 2) */
      double pa = k < 0 ? 1 : -k - 1, pb = k < 0 ? 2 : -k, vd, vu;
      gamma_ext(pa, pb, &vd, &vu, root);
      double xl = mpfr_get_d(root, MPFR_RNDD), xh = mpfr_get_d(root, MPFR_RNDU);
      add(xl, xh); add(xl, xl); add(xh, xh); add(nextafter(xl, -INFINITY), xl); add(xh, nextafter(xh, INFINITY));
      add(pa, xl); add(pa, xh); add(xl, pb); add(xh, pb); add(pa, pb);
      add(nextafter(pa, INFINITY), nextafter(pb, -INFINITY));
      add(pa, nextafter(pa, INFINITY)); add(nextafter(pb, -INFINITY), pb);
      add(pa - 0.5, pa); add(pb, pb + 0.5); add(pa - 0.5, pb - 0.25);
    }
    mpfr_clear(root);
    double big[] = {0x1p52, 0x1p53, 0x1p60};
    for (int i = 0; i < 3; i++) {
      double x = -big[i];
      add(x, x); add(nextafter(x, -INFINITY), x); add(x, nextafter(x, INFINITY));
      add(nextafter(x, -INFINITY), nextafter(x, INFINITY));
    }
    add(-0x1p52 - 0.5, -0x1p52); add(-0x1p52 + 0.5, -0x1p52 + 1);   /* half-integers just below 2^52 */
    add(171.5, 171.7); add(171.62, 200); add(0x1p-1074, 1); add(0, 1); add(-0.0, 1.5); add(-0.0, -0.0);
    add(-1, -0.0); add(-1e-300, 1e-300); add(-200, -150); add(-185, -183.5); add(-184.5, -183.5);
  }
  if (isfinite(d->lo) || isfinite(d->hi)) {   /* domain edges */
    double e[2] = {d->lo, d->hi};
    for (int k = 0; k < 2; k++) {
      if (!isfinite(e[k])) continue;
      double x = e[k];
      add(x, x);
      add(x - 1, x + 1);
      add(x - 1, x);
      add(x, x + 1);
      add(nextafter(x, -INFINITY), nextafter(x, INFINITY));
      add(x - 2, x - 1);
      add(x + 1, x + 2);
      add(-INFINITY, x);
      add(x, INFINITY);
    }
  }
  add(-INFINITY, INFINITY);
  add(0, INFINITY);
  add(-INFINITY, 0);
  add(NAN, 1);
  add(1, NAN);
  add(2, 1);
  add(-0.0, 0.0);
  add(-0.0, 4);   /* -0 at a domain's closed or open end: rsqrt(-0) is -inf, rsqrt over [-0, 4] ends at +inf */
  add(-0.0, 0.5);
}

static int same(double x, double y) { return x == y || (isnan(x) && isnan(y)); }

/* ---- two arguments ---- */

static kit_mpfr2 ref2_of(const char *n)
{
  for (int i = 0; i < kit_nfns2; i++) if (!strcmp(kit_fns2[i].name, n)) return kit_fns2[i].ref;
  return 0;
}
static double eval2(kit_mpfr2 r, double x, double y, mpfr_rnd_t rnd)
{ return kit_decode(KIT_B64, kit_ref2(KIT_B64, r, kit_encode(KIT_B64, x), kit_encode(KIT_B64, y), rnd)); }

/* the candidate points of [a, b], argument pos of f: its ends, and the
   points inside where f can be least or greatest (hypot: 0). For atan2's
   y (pos 0), a zero stands for two candidates: +0, atan2's value on the
   axis, when 0 is in [a, b], and -0, the limit from below (-pi on the
   negative x-axis), when negatives lie next to it. Zeros are unsigned in
   the input (sets): an end of either sign is 0. */
static int cands(const char *f, int pos, double a, double b, double *c)
{
  int k = 0, at2 = !strcmp(f, "atan2");
  if (a == 0) a = 0.0;
  if (b == 0) b = 0.0;
  c[k++] = a;
  if (b != a) c[k++] = b;
  if ((!strcmp(f, "hypot") || (at2 && pos == 1)) && a < 0 && 0 < b) c[k++] = 0;
  if (!strcmp(f, "pow")) {   /* where x^y turns: x = 1, y = 0 */
    double t = pos == 0 ? 1 : 0;
    if (a < t && t < b) c[k++] = t;
  }
  if (at2 && pos == 0 && a < 0 && 0 <= b) {
    if (b > 0) c[k++] = 0.0;
    c[k++] = -0.0;
  }
  return k;
}
/* the tightest box bounds by the candidates; corners = 1 leaves the
   critical points out (the negative control) */
static void reference2(const char *f, kit_mpfr2 r, double a, double b, double c, double d, int corners, double *lo,
                       double *hi)
{
  if (!(a <= b) || !(c <= d)) { *lo = *hi = NAN; return; }
  if (!strcmp(f, "pow")) {   /* IEEE 1788's domain, written out again: x > 0, or x = 0 with y > 0 */
    if (b < 0) { *lo = *hi = NAN; return; }
    if (a < 0) a = 0;
    if (b == 0) {   /* only x = 0: y > 0 alone, where x^y = 0 */
      *lo = *hi = d > 0 ? 0 : NAN;
      return;
    }
    /* the negative control forgets that x reaches 0 (as a limit, from
       inside the domain): the least positive binary64 instead */
    if (corners && a == 0) a = 0x1p-1074;
    corners = 0;
  }
  double xs[4], ys[4];
  int nx = corners ? 2 : cands(f, 0, a, b, xs), ny = corners ? 2 : cands(f, 1, c, d, ys);
  if (corners) { xs[0] = a; xs[1] = b; ys[0] = c; ys[1] = d; }
  int at2 = !strcmp(f, "atan2");
  double l = INFINITY, h = -INFINITY;
  for (int i = 0; i < nx; i++)
    for (int j = 0; j < ny; j++) {
      if (at2 && xs[i] == 0 && ys[j] == 0) continue;   /* the origin: undefined */
      double v = eval2(r, xs[i], ys[j], MPFR_RNDD), w = eval2(r, xs[i], ys[j], MPFR_RNDU);
      if (v < l) l = v;
      if (w > h) h = w;
    }
  if (l > h) l = h = NAN;   /* nothing left: atan2 on the origin alone */
  *lo = l;
  *hi = h;
}

static double *X0, *X1, *Y0, *Y1;
static size_t nb, capb;
static void addb(double a, double b, double c, double d)
{
  if (nb == capb) {
    capb = capb ? 2 * capb : 1 << 15;
    X0 = realloc(X0, capb * sizeof *X0); X1 = realloc(X1, capb * sizeof *X1);
    Y0 = realloc(Y0, capb * sizeof *Y0); Y1 = realloc(Y1, capb * sizeof *Y1);
  }
  X0[nb] = a; X1[nb] = b; Y0[nb] = c; Y1[nb++] = d;
}
/* boxes from the one-argument intervals: random pairs, and the first 64
   (the specials and points) against each other */
static void boxes(void)
{
  dom all = {"", -INFINITY, INFINITY, 0, 0, NONE};
  intervals(&all);
  nb = 0;
  for (size_t i = 0; i < 64 && i < n; i++)
    for (size_t j = 0; j < 64 && j < n; j++) addb(A[i], B[i], A[j], B[j]);
  for (int k = 0; k < 1 << 15; k++) {
    size_t i = next() % n, j = next() % n;
    addb(A[i], B[i], A[j], B[j]);
  }
  /* on and across the axes, at every scale: each side from negative,
     touching zero, zero alone (either sign), across, to positive */
  for (int k = 0; k < 256; k++) {
    double u = rand_mag(-60, 60), v = rand_mag(-60, 60), w = rand_mag(-60, 60);
    double lo[7] = {-u - v, -u, 0, -0.0, -u, 0, w}, hi[7] = {-u, 0, 0, -0.0, v, v, w + v};
    for (int i = 0; i < 7; i++)
      for (int j = 0; j < 7; j++) addb(lo[i], hi[i], lo[j], hi[j]);
  }
  /* around x = 1 and y = 0, where x^y turns */
  for (int k = 0; k < 256; k++) {
    double u = rand_mag(-50, 0), v = rand_mag(-50, 0), w = rand_mag(-60, 60);
    double lo[6] = {1 - u, 1, 1, 0, 1 - u, 1 + v}, hi[6] = {1 + v, 1 + v, 1, 1, 1, 1 + v + w};
    double ylo[5] = {-w, 0, -w, -0.0, w}, yhi[5] = {w, w, 0, 0, 2 * w};
    for (int i = 0; i < 6; i++)
      for (int j = 0; j < 5; j++) addb(lo[i], hi[i], ylo[j], yhi[j]);
  }
  /* across zero in one or both, at every scale */
  for (int k = 0; k < 4096; k++) {
    double u = rand_mag(-60, 60), v = rand_mag(-60, 60), w = rand_mag(-60, 60), z = rand_mag(-60, 60);
    addb(-u, v, w, w + z);
    addb(w, w + z, -u, v);
    addb(-u, v, -w, z);
  }
}

/* the flush modes, as a -ffast-math program sets them: x86-64 MXCSR FZ and DAZ, aarch64 FPCR.FZ */
#if defined(__x86_64__)
#include <immintrin.h>
#define FLUSH_BITS 0x8040ul
static unsigned long fpctl(void) { return _mm_getcsr(); }
static void set_fpctl(unsigned long r) { _mm_setcsr((unsigned)r); }
#elif defined(__aarch64__)
#define FLUSH_BITS (1ul << 24)
static unsigned long fpctl(void) { unsigned long r; __asm__ volatile("mrs %0, fpcr" : "=r"(r)); return r; }
static void set_fpctl(unsigned long r) { __asm__ volatile("msr fpcr, %0" : : "r"(r)); }
#else
#define FLUSH_BITS 0ul
static unsigned long fpctl(void) { return 0; }
static void set_fpctl(unsigned long r) { (void)r; }
#endif
/* how many of n results differ, bit for bit, when the flush modes are set for the call (-1: the call cleared them) */
static long flushed(void (*f)(const double *, const double *, double *, double *, size_t), const double *a,
                    const double *b, const double *lo, const double *hi, size_t n)
{
  double *l = malloc(n * sizeof *l), *h = malloc(n * sizeof *h);
  unsigned long c = fpctl();
  set_fpctl(c | FLUSH_BITS);
  f(a, b, l, h, n);
  int kept = (fpctl() & FLUSH_BITS) == FLUSH_BITS;
  set_fpctl(c);
  long d = 0;
  for (size_t k = 0; k < n; k++) d += memcmp(&l[k], &lo[k], 8) || memcmp(&h[k], &hi[k], 8);
  free(l);
  free(h);
  return kept ? d : -1;
}

/* the same for the two-argument functions on boxes */
static long flushed2(void (*f)(const double *, const double *, const double *, const double *, double *, double *, size_t),
                     const double *x0, const double *x1, const double *y0, const double *y1, const double *lo,
                     const double *hi, size_t n)
{
  double *l = malloc(n * sizeof *l), *h = malloc(n * sizeof *h);
  unsigned long c = fpctl();
  set_fpctl(c | FLUSH_BITS);
  f(x0, x1, y0, y1, l, h, n);
  int kept = (fpctl() & FLUSH_BITS) == FLUSH_BITS;
  set_fpctl(c);
  long d = 0;
  for (size_t k = 0; k < n; k++) d += memcmp(&l[k], &lo[k], 8) || memcmp(&h[k], &hi[k], 8);
  free(l);
  free(h);
  return kept ? d : -1;
}

static int check2(void)
{
  int r = 0;
  boxes();
  for (int i = 0; i < NL2; i++) {
    kit_mpfr2 ref = ref2_of(L2[i].name);
    if (!ref) { printf("%s: no reference\n", L2[i].name); return 2; }
    double *lo = malloc(nb * sizeof *lo), *hi = malloc(nb * sizeof *hi);
    L2[i].f(X0, X1, Y0, Y1, lo, hi, nb);
    unsigned long long bad = 0, ctl = 0, outside = 0, samples = 0, neg = 0;
    long first = -1;
    double fw_lo = 0, fw_hi = 0;
#pragma omp parallel for reduction(+ : bad, ctl, outside, samples, neg) schedule(dynamic, 64)
    for (size_t k = 0; k < nb; k++) {
      double wl, wh, cl2, ch2;
      reference2(L2[i].name, ref, X0[k], X1[k], Y0[k], Y1[k], 0, &wl, &wh);
      reference2(L2[i].name, ref, X0[k], X1[k], Y0[k], Y1[k], 1, &cl2, &ch2);
      neg += !same(lo[k], cl2) || !same(hi[k], ch2);
      int differ = !same(lo[k], wl) || !same(hi[k], wh);
      bad += differ;
      if (differ) {
#pragma omp critical
        if (first < 0 || (long)k < first) { first = (long)k; fw_lo = wl; fw_hi = wh; }
      }
      int moved = k == nb / 2 || (k * 0x9e3779b97f4a7c15ULL >> 58) == 9;
      double cl = moved ? (isnan(lo[k]) ? 0 : nextafter(lo[k], INFINITY)) : lo[k];
      ctl += !same(cl, wl) || !same(hi[k], wh);
      if (isnan(lo[k]) || !isfinite(X0[k]) || !isfinite(X1[k]) || !isfinite(Y0[k]) || !isfinite(Y1[k])) continue;
      mpfr_t x, y, z;
      mpfr_init2(x, 53);
      mpfr_init2(y, 53);
      mpfr_init2(z, 300);
      for (int j = 0; j < 8; j++) {
        uint64_t h1 = (k * 8 + (size_t)j) * 0x9e3779b97f4a7c15ULL, h2 = h1 * 0xbf58476d1ce4e5b9ULL;
        double t = X0[k] + (X1[k] - X0[k]) * ((double)(h1 >> 11) * 0x1p-53);
        double u = Y0[k] + (Y1[k] - Y0[k]) * ((double)(h2 >> 11) * 0x1p-53);
        if (!(t >= X0[k] && t <= X1[k] && u >= Y0[k] && u <= Y1[k])) continue;
        if (!strcmp(L2[i].name, "pow") && (t < 0 || (t == 0 && u <= 0))) continue;   /* outside pow's domain */
        mpfr_set_d(x, t, MPFR_RNDN);
        mpfr_set_d(y, u, MPFR_RNDN);
        ref(z, x, y, MPFR_RNDN);
        samples++;
        if (mpfr_nan_p(z)) continue;
        outside += mpfr_cmp_d(z, lo[k]) < 0 || mpfr_cmp_d(z, hi[k]) > 0;
      }
      mpfr_clears(x, y, z, (mpfr_ptr)0);
    }
    long fl = flushed2(L2[i].f, X0, X1, Y0, Y1, lo, hi, nb);   /* the same with the caller's flush modes set */
    int res = !nb || !ctl ? 2 : bad || outside || fl ? 1 : 0;
    printf("%-10s %7zu boxes, %llu differ, %llu of %llu points outside (control: %llu differ)", L2[i].name, nb, bad,
           outside, samples, ctl);
    if (fl) printf(fl < 0 ? "\n    the call cleared the caller's flush modes" : "\n    with the flush modes set, %ld differ", fl);
    if (first >= 0)
      printf("\n    first: [%a, %a] x [%a, %a]: got [%a, %a], want [%a, %a]", X0[first], X1[first], Y0[first], Y1[first],
             lo[first], hi[first], fw_lo, fw_hi);
    if (!ctl) printf("  VOID: the control did not differ");
    printf("\n%-10s %llu of %zu boxes differ %s\n", "corners", neg, nb,
           neg ? "(as they must: the critical points matter)" : "NEGATIVE CONTROL FAILED: the corners alone must differ");
    if (!neg) res = kit_worst(res, 2);
    r = kit_worst(r, res);
    free(lo);
    free(hi);
  }
  return r;
}

int main(void)
{
  int r = 0;
  printf("every function: ival against the MPFR reference, and the exact f at 8 points inside each interval\n");
  const char *only = getenv("IVAL_ONLY");   /* one function's run alone, for work on it */
  for (int i = 0; i < NL; i++) {
    if (only && strcmp(only, L[i].name)) continue;
    kit_mpfr1 ref = ref_of(L[i].name);
    if (!ref) { printf("%s: no reference\nVERDICT: VOID\n", L[i].name); return 2; }
    dom d = domain_of(L[i].name);
    intervals(&d);
    double *lo = malloc(n * sizeof *lo), *hi = malloc(n * sizeof *hi);
    L[i].f(A, B, lo, hi, n);
    unsigned long long bad = 0, ctl = 0, outside = 0, samples = 0;
    long first = -1, first_out = -1;
    double fw_lo = 0, fw_hi = 0, out_t = 0;
#pragma omp parallel for reduction(+ : bad, ctl, outside, samples) schedule(dynamic, 64)
    for (size_t k = 0; k < n; k++) {
      double wl, wh;
      reference(&d, ref, A[k], B[k], &wl, &wh);
      int differ = !same(lo[k], wl) || !same(hi[k], wh);
      bad += differ;
      if (differ) {
#pragma omp critical
        if (first < 0 || (long)k < first) { first = (long)k; fw_lo = wl; fw_hi = wh; }
      }
      int moved = k == n / 2 || (k * 0x9e3779b97f4a7c15ULL >> 58) == 9;
      double cl = moved ? (isnan(lo[k]) ? 0 : nextafter(lo[k], INFINITY)) : lo[k];
      ctl += !same(cl, wl) || !same(hi[k], wh);
      /* the exact f inside: 8 points */
      if (isnan(lo[k]) || !(A[k] <= B[k])) continue;
      double a = A[k] < d.lo ? d.lo : A[k], b = B[k] > d.hi ? d.hi : B[k];
      if (!isfinite(a) || !isfinite(b) || !(a < b)) continue;
      mpfr_t x, y;
      mpfr_init2(x, 53);
      mpfr_init2(y, 300);
      for (int j = 0; j < 8; j++) {
        double t = a + (b - a) * ((double)((k * 8 + (size_t)j) * 0x9e3779b97f4a7c15ULL >> 11) * 0x1p-53);
        if (!(t >= a && t <= b)) continue;
        if (d.lo_open && t == d.lo) continue;
        if (d.hi_open && t == d.hi) continue;
        if (d.crit == CRIT_TANPI && fabs(t) < 0x1p52 && fmod(fabs(t), 1) == 0.5) continue;   /* a pole: outside the domain */
        if (d.crit == CRIT_GAMMA && t <= 0 && floor(t) == t) continue;                         /* tgamma's poles too */
        mpfr_set_d(x, t, MPFR_RNDN);
        ref(y, x, MPFR_RNDN);
        samples++;
        if (mpfr_nan_p(y)) continue;
        if (mpfr_cmp_d(y, lo[k]) < 0 || mpfr_cmp_d(y, hi[k]) > 0) {
          outside++;
#pragma omp critical
          if (first_out < 0 || (long)k < first_out) { first_out = (long)k; out_t = t; }
        }
      }
      mpfr_clear(x);
      mpfr_clear(y);
    }
    long fl = flushed(L[i].f, A, B, lo, hi, n);   /* the same with the caller's flush modes set */
    int res = !n || !ctl ? 2 : bad || outside || fl ? 1 : 0;
    printf("%-10s %7zu intervals, %llu differ, %llu of %llu points outside (control: %llu differ)", L[i].name, n, bad,
           outside, samples, ctl);
    if (fl) printf(fl < 0 ? "\n    the call cleared the caller's flush modes" : "\n    with the flush modes set, %ld differ", fl);
    if (first >= 0)
      printf("\n    first: [%a, %a]: got [%a, %a], want [%a, %a]", A[first], B[first], lo[first], hi[first], fw_lo, fw_hi);
    if (first_out >= 0)
      printf("\n    first outside: f(%a), in [%a, %a], bounds [%a, %a]", out_t, A[first_out], B[first_out], lo[first_out],
             hi[first_out]);
    if (!ctl) printf("  VOID: the control did not differ");
    printf("\n");
    r = kit_worst(r, res);
    free(lo);
    free(hi);
  }
  /* in place, and the environment left alone */
  double a[4] = {-1, 0.5, 2, -3}, b[4] = {1, 1.5, 3, -2}, l[4], h[4];
  ival_sin(a, b, l, h, 4);
  int round0 = fegetround();
  ival_sin(a, b, a, b, 4);
  int inplace = !memcmp(a, l, sizeof a) && !memcmp(b, h, sizeof b) && fegetround() == round0;
  printf("%-10s %s\n", "in place", inplace ? "the same intervals, rounding mode unchanged" : "DIFFERENT");
  if (!inplace) r = 1;
  /* negative control: exp's bounds judged against exp2's */
  {
    dom d = domain_of("exp");
    intervals(&d);
    double *lo = malloc(n * sizeof *lo), *hi = malloc(n * sizeof *hi);
    ival_exp(A, B, lo, hi, n);
    unsigned long long bad = 0;
    for (size_t k = 0; k < n; k++) {
      double wl, wh;
      reference(&d, mpfr_exp2, A[k], B[k], &wl, &wh);
      bad += !same(lo[k], wl) || !same(hi[k], wh);
    }
    printf("%-10s %llu of %zu intervals differ %s\n", "exp against exp2", bad, n,
           bad ? "(as they must)" : "NEGATIVE CONTROL FAILED: they must differ");
    if (!bad) r = kit_worst(r, 2);
    free(lo);
    free(hi);
  }
  printf("two-argument functions: ival against the MPFR reference on boxes, and the exact f at 8 points in each\n");
  r = kit_worst(r, check2());
  return kit_verdict(r, "every interval is the tightest, and every control differs");
}
