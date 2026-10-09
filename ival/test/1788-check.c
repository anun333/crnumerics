/* 1788-check.c: ival's other IEEE 1788.1 operations (ival-1788.c), each against a reference of its own.

   The rounding ones against MPFR, on every pair of special intervals and random ones:
     - mid: (lo + hi) / 2 exactly, rounded to nearest (ties to even); wid: hi - lo rounded up; rad: the greater of
       mid - lo and hi - mid, exactly, rounded up (ival's mid, as 1788 defines rad from it); midrad equals both;
     - cancelminus and cancelplus: the widths compared exactly, then the ends' differences rounded down and up.
   The exact ones by definition, element by element (abs, sign, the roundings as the image of the ends; min, max,
   intersect, hull from the ends; mag, mig, inf, sup). And the predicates against overlap's state, which is computed
   another way: equal holds exactly for equals (or both empty), subset for equals, starts, containedBy, finishes (or
   A empty), disjoint for before and after (or an empty one); less(A, B) and less(B, A) together are equal; and
   overlap(B, A) is overlap(A, B)'s converse. pown(x, p) (ival.c) against MPFR's pow_si at the ends, with 0 and the
   limits at the pole added where the interval reaches them, for 22 powers (0, small, large, the int extremes) on
   every interval. mulrev and mulrevpair: points c / b in X must be in mulrev's result (exact rationals), and mulrev
   on the whole line is the hull of mulrevpair's pair. Negative control: the midpoint as the sum of the halves must differ
   somewhere (it rounds twice for subnormal ends). */
#include <float.h>
#include <math.h>
#include <mpfr.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ival.h"

static uint64_t rs = 0x2545f4914f6cdd1dULL;
static uint64_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }
static double anyd(void) { for (;;) { uint64_t u = rnd(); double d; memcpy(&d, &u, 8); if (isfinite(d)) return d; } }
static int empty(double lo, double hi) { return !(lo <= hi) || lo == INFINITY || hi == -INFINITY; }
static int same(double a, double b) { return (a != a && b != b) || a == b; }

static long checked, bad, control;
static char first[400];
static void want(const char *op, double al, double ah, double bl, double bh, double z0, double z1, double r0, double r1)
{
  checked++;
  if (same(z0, r0) && same(z1, r1)) return;
  if (!bad++) snprintf(first, sizeof first, " (first: %s [%a, %a] [%a, %a] = %a %a, want %a %a)", op, al, ah, bl, bh, z0, z1, r0, r1);
}

static mpfr_t X, Y;
static double rn_mid(double l, double h)
{
  if (isinf(l)) return isinf(h) ? 0.0 : -DBL_MAX;
  if (isinf(h)) return DBL_MAX;
  mpfr_set_d(X, l, MPFR_RNDN); mpfr_add_d(X, X, h, MPFR_RNDN); mpfr_div_2ui(X, X, 1, MPFR_RNDN);   /* exact */
  return mpfr_get_d(X, MPFR_RNDN) + 0.0;
}
static double ru_diff(double a, double b)   /* a - b rounded up, exactly */
{
  mpfr_set_d(X, a, MPFR_RNDN); mpfr_sub_d(X, X, b, MPFR_RNDN);
  return mpfr_get_d(X, MPFR_RNDU);
}
static double rd_diff(double a, double b)
{
  mpfr_set_d(X, a, MPFR_RNDN); mpfr_sub_d(X, X, b, MPFR_RNDN);
  return mpfr_get_d(X, MPFR_RNDD);
}
static void r_cancel(double al, double ah, double bl, double bh, double *zl, double *zh)
{
  int ea = empty(al, ah), eb = empty(bl, bh);
  int ua = !ea && (isinf(al) || isinf(ah)), ub = !eb && (isinf(bl) || isinf(bh));
  if (ea && !ub) { *zl = *zh = NAN; return; }
  if (ea || ua || ub || eb) { *zl = -INFINITY; *zh = INFINITY; return; }
  mpfr_set_d(X, ah, MPFR_RNDN); mpfr_sub_d(X, X, al, MPFR_RNDN);   /* the widths, exactly */
  mpfr_set_d(Y, bh, MPFR_RNDN); mpfr_sub_d(Y, Y, bl, MPFR_RNDN);
  if (mpfr_less_p(X, Y)) { *zl = -INFINITY; *zh = INFINITY; return; }
  *zl = rd_diff(al, bl) + 0.0; *zh = ru_diff(ah, bh) + 0.0;
}
static double pick(double lo, double hi)   /* a finite point of [lo, hi] */
{
  double l = isinf(lo) ? -DBL_MAX : lo, h = isinf(hi) ? DBL_MAX : hi;
  double t = (double)(rnd() >> 11) * 0x1p-53, x = l + t * (h - l);
  if (!isfinite(x)) x = t < 0.5 ? l : h;
  return x < l ? l : x > h ? h : x;
}
/* whether c / b is at least u (upper = 0) or at most u (upper = 1), exactly; u may be infinite */
static int inside_q(double u, double b, double c, int upper)
{
  if (isinf(u)) return upper ? u > 0 : u < 0;
  mpfr_set_d(X, u, MPFR_RNDN); mpfr_mul_d(X, X, b, MPFR_RNDN);   /* exact: 2200 bits */
  int cmp = mpfr_cmp_d(X, c);                                     /* u b against c */
  if (b < 0) cmp = -cmp;                                          /* u <= c / b  <=>  u b >= c for b < 0 */
  return upper ? cmp >= 0 : cmp <= 0;
}
/* whether d is in S = {x : x b in C for some b in B}, B and C nonempty: d B, the exact products (an infinite end is
   a limit, not a member), meets C. With no equal infinite ends to compare, that is max(d B) >= cl and min(d B) <= ch.
   d = 0 gives {0}. */
static int in_s(double d, double bl, double bh, double cl, double ch)
{
  if (empty(bl, bh) || empty(cl, ch)) return 0;
  if (d == 0) return cl <= 0 && 0 <= ch;
  mpfr_set_d(X, d, MPFR_RNDN); mpfr_mul_d(X, X, bl, MPFR_RNDN);
  mpfr_set_d(Y, d, MPFR_RNDN); mpfr_mul_d(Y, Y, bh, MPFR_RNDN);
  if (mpfr_greater_p(X, Y)) mpfr_swap(X, Y);   /* X = min, Y = max */
  return mpfr_cmp_d(Y, cl) >= 0 && mpfr_cmp_d(X, ch) <= 0;
}
static int converse(int s)
{
  static const int c[16] = { IVAL_BOTH_EMPTY, IVAL_SECOND_EMPTY, IVAL_FIRST_EMPTY, IVAL_AFTER, IVAL_MET_BY,
                             IVAL_OVERLAPPED_BY, IVAL_STARTED_BY, IVAL_CONTAINS, IVAL_FINISHED_BY, IVAL_EQUALS,
                             IVAL_FINISHES, IVAL_CONTAINED_BY, IVAL_STARTS, IVAL_OVERLAPS, IVAL_MEETS, IVAL_BEFORE };
  return c[s];
}

static const double SP[] = { -INFINITY, -DBL_MAX, -0x1.fffffffffffffp+1022, -3, -2, -1, -0.5, -0x1p-1021, -DBL_MIN,
                             -0x3p-1074, -DBL_TRUE_MIN, 0.0, DBL_TRUE_MIN, 0x3p-1074, DBL_MIN, 0x1p-1021, 0.5, 1, 1.5,
                             2, 3, 0x1.fffffffffffffp+1022, DBL_MAX, INFINITY };
enum { NSP = sizeof SP / sizeof SP[0] };

int main(void)
{
  mpfr_init2(X, 2200); mpfr_init2(Y, 2200);
  enum { NR = 1 << 13 };
  static double L[NSP * NSP + NR + 8], H[NSP * NSP + NR + 8];
  int ni = 0;
  for (int i = 0; i < NSP; i++) for (int j = i; j < NSP; j++) { L[ni] = SP[i]; H[ni] = SP[j]; ni++; }
  for (int k = 0; k < NR; k++) {
    double a = anyd(), b;
    switch (k % 5) {
      case 4: a = (double)(int64_t)(rnd() >> 12) * 0x1p-1074; b = (double)(int64_t)(rnd() >> 12) * 0x1p-1074;   /* small, */
        if (rnd() & 1) a = -a;                                                                    /* to subnormal */
        if (rnd() & 1) b = -b;
        break;
      case 0: b = anyd(); break;
      case 1: b = a; break;
      case 2: b = a; for (int s = (int)(rnd() % 4); s >= 0; s--) b = nextafter(b, INFINITY); break;
      default: b = a * (1 + (double)(rnd() >> 11) * 0x1p-60); break;
    }
    if (b < a) { double t = a; a = b; b = t; }
    L[ni] = a; H[ni] = b; ni++;
  }
  L[ni] = NAN; H[ni] = NAN; ni++; L[ni] = 2; H[ni] = 1; ni++; L[ni] = INFINITY; H[ni] = INFINITY; ni++;
  L[ni] = -INFINITY; H[ni] = -INFINITY; ni++;

  /* one interval */
  static double y[8][NSP * NSP + NR + 8];
  ival_mid(L, H, y[0], ni); ival_wid(L, H, y[1], ni); ival_rad(L, H, y[2], ni); ival_midrad(L, H, y[3], y[4], ni);
  ival_mag(L, H, y[5], ni); ival_mig(L, H, y[6], ni);
  for (int i = 0; i < ni; i++) {
    double l = L[i], h = H[i];
    int e = empty(l, h), ub = !e && (isinf(l) || isinf(h));
    double m = e ? NAN : rn_mid(l, h);
    want("mid", l, h, 0, 0, y[0][i], 0, m, 0);
    want("wid", l, h, 0, 0, y[1][i], 0, e ? NAN : ub ? INFINITY : ru_diff(h, l), 0);
    double r = e ? NAN : ub ? INFINITY : fmax(ru_diff(y[0][i], l), ru_diff(h, y[0][i]));
    want("rad", l, h, 0, 0, y[2][i], 0, r, 0);
    want("midrad", l, h, 0, 0, y[3][i], y[4][i], y[0][i], y[2][i]);
    want("mag", l, h, 0, 0, y[5][i], 0, e ? NAN : fmax(fabs(l), fabs(h)), 0);
    want("mig", l, h, 0, 0, y[6][i], 0, e ? NAN : l > 0 ? l : h < 0 ? -h : 0.0, 0);
    if (!e && !ub && !same(l * 0.5 + h * 0.5, m)) control++;   /* the negative control */
  }
  struct { void (*f)(const double *, const double *, double *, double *, size_t); double (*g)(double); const char *name; } mono[] = {
    { ival_ceil, ceil, "ceil" }, { ival_floor, floor, "floor" }, { ival_trunc, trunc, "trunc" },
    { ival_round, round, "round" }, { ival_roundeven, nearbyint, "roundeven" } };
  for (unsigned k = 0; k < sizeof mono / sizeof mono[0]; k++) {
    mono[k].f(L, H, y[0], y[1], ni);
    for (int i = 0; i < ni; i++) {
      int e = empty(L[i], H[i]);
      want(mono[k].name, L[i], H[i], 0, 0, y[0][i], y[1][i], e ? NAN : mono[k].g(L[i]) + 0.0, e ? NAN : mono[k].g(H[i]) + 0.0);
    }
  }
  ival_abs(L, H, y[0], y[1], ni); ival_sign(L, H, y[2], y[3], ni); ival_inf(L, H, y[4], ni); ival_sup(L, H, y[5], ni);
  for (int i = 0; i < ni; i++) {
    double l = L[i], h = H[i];
    int e = empty(l, h);
    double al = l >= 0 ? l : h <= 0 ? -h : 0.0, ah = fmax(fabs(l), fabs(h));
    want("abs", l, h, 0, 0, y[0][i], y[1][i], e ? NAN : al + 0.0, e ? NAN : ah);
    double sl = (l > 0) - (l < 0), sh = (h > 0) - (h < 0);
    want("sign", l, h, 0, 0, y[2][i], y[3][i], e ? NAN : sl, e ? NAN : sh);
    want("inf", l, h, 0, 0, y[4][i], signbit(y[4][i]), e ? INFINITY : l, e ? 0 : l == 0 || signbit(l));
    want("sup", l, h, 0, 0, y[5][i], signbit(y[5][i]), e ? -INFINITY : h, e ? 1 : h != 0 && signbit(h));
  }

  /* two intervals: every pair of the special ones and random pairs */
  int nsp = NSP * (NSP + 1) / 2, np = nsp * nsp + (1 << 17);
  double *al = malloc(np * 8), *ah = malloc(np * 8), *bl = malloc(np * 8), *bh = malloc(np * 8);
  double *zl = malloc(np * 8), *zh = malloc(np * 8);
  unsigned char *ov = malloc(np), *ov2 = malloc(np), *q = malloc(np), *q2 = malloc(np), *q3 = malloc(np);
  int m = 0;
  for (int i = 0; i < nsp; i++) for (int j = 0; j < nsp; j++) { al[m] = L[i]; ah[m] = H[i]; bl[m] = L[j]; bh[m] = H[j]; m++; }
  while (m < np) { int i = (int)(rnd() % ni), j = (int)(rnd() % ni); al[m] = L[i]; ah[m] = H[i]; bl[m] = L[j]; bh[m] = H[j]; m++; }
  for (int sub = 0; sub < 2; sub++) {
    (sub ? ival_cancelplus : ival_cancelminus)(al, ah, bl, bh, zl, zh, np);
    for (int k = 0; k < np; k++) {
      double rl, rh;
      if (sub) r_cancel(al[k], ah[k], -bh[k], -bl[k], &rl, &rh); else r_cancel(al[k], ah[k], bl[k], bh[k], &rl, &rh);
      want(sub ? "cancelplus" : "cancelminus", al[k], ah[k], bl[k], bh[k], zl[k], zh[k], rl, rh);
    }
  }
  for (int op = 0; op < 4; op++) {
    void (*f[4])(const double *, const double *, const double *, const double *, double *, double *, size_t) =
      { ival_min, ival_max, ival_intersect, ival_hull };
    const char *nm[4] = { "min", "max", "intersect", "hull" };
    f[op](al, ah, bl, bh, zl, zh, np);
    for (int k = 0; k < np; k++) {
      int ea = empty(al[k], ah[k]), eb = empty(bl[k], bh[k]);
      double rl, rh;
      if (op == 0) { rl = fmin(al[k], bl[k]); rh = fmin(ah[k], bh[k]); if (ea || eb) rl = rh = NAN; }
      else if (op == 1) { rl = fmax(al[k], bl[k]); rh = fmax(ah[k], bh[k]); if (ea || eb) rl = rh = NAN; }
      else if (op == 2) { rl = fmax(al[k], bl[k]); rh = fmin(ah[k], bh[k]); if (ea || eb || rl > rh) rl = rh = NAN; }
      else if (ea && eb) rl = rh = NAN;
      else if (ea || eb) { rl = ea ? bl[k] : al[k]; rh = ea ? bh[k] : ah[k]; }
      else { rl = fmin(al[k], bl[k]); rh = fmax(ah[k], bh[k]); }
      want(nm[op], al[k], ah[k], bl[k], bh[k], zl[k], zh[k], rl == rl ? rl + 0.0 : rl, rh == rh ? rh + 0.0 : rh);
    }
  }
  /* the predicates against overlap */
  ival_overlap(al, ah, bl, bh, ov, np); ival_overlap(bl, bh, al, ah, ov2, np);
  ival_equal(al, ah, bl, bh, q, np);
  for (int k = 0; k < np; k++) {
    want("overlap converse", al[k], ah[k], bl[k], bh[k], ov2[k], 0, converse(ov[k]), 0);
    want("equal", al[k], ah[k], bl[k], bh[k], q[k], 0, ov[k] == IVAL_BOTH_EMPTY || ov[k] == IVAL_EQUALS, 0);
  }
  ival_subset(al, ah, bl, bh, q, np);
  for (int k = 0; k < np; k++) {
    int s = ov[k];
    want("subset", al[k], ah[k], bl[k], bh[k], q[k], 0, s == IVAL_BOTH_EMPTY || s == IVAL_FIRST_EMPTY || s == IVAL_EQUALS
         || s == IVAL_STARTS || s == IVAL_CONTAINED_BY || s == IVAL_FINISHES, 0);
  }
  ival_disjoint(al, ah, bl, bh, q, np);
  for (int k = 0; k < np; k++) {
    int s = ov[k];
    want("disjoint", al[k], ah[k], bl[k], bh[k], q[k], 0, s <= IVAL_SECOND_EMPTY || s == IVAL_BEFORE || s == IVAL_AFTER, 0);
  }
  ival_less(al, ah, bl, bh, q, np); ival_less(bl, bh, al, ah, q2, np); ival_equal(al, ah, bl, bh, q3, np);
  for (int k = 0; k < np; k++) want("less both ways", al[k], ah[k], bl[k], bh[k], q[k] && q2[k], 0, q3[k], 0);
  /* interior and strictless against their definitions with the infinite ends spelled out */
  ival_interior(al, ah, bl, bh, q, np); ival_strictless(al, ah, bl, bh, q2, np);
  for (int k = 0; k < np; k++) {
    int ea = empty(al[k], ah[k]), eb = empty(bl[k], bh[k]);
    int lo_in = bl[k] < al[k] || (isinf(al[k]) && al[k] < 0 && isinf(bl[k]) && bl[k] < 0);
    int hi_in = ah[k] < bh[k] || (isinf(ah[k]) && ah[k] > 0 && isinf(bh[k]) && bh[k] > 0);
    want("interior", al[k], ah[k], bl[k], bh[k], q[k], 0, ea ? 1 : eb ? 0 : lo_in && hi_in, 0);
    int lo_lt = al[k] < bl[k] || (isinf(al[k]) && al[k] < 0 && isinf(bl[k]) && bl[k] < 0);
    want("strictless", al[k], ah[k], bl[k], bh[k], q2[k], 0, ea || eb ? ea && eb : lo_lt && hi_in, 0);
  }
  /* pown against MPFR: the hull of x^p at the ends, 0 when an even positive power crosses it, and the limits at 0 of
     a negative power from whichever side the interval reaches it, each rounded both ways (0 itself excluded then) */
  static const int P[] = { 0, 1, 2, 3, 4, 5, 7, 10, 31, 64, 1000, -1, -2, -3, -4, -7, -10, -1000, 2147483647,
                           2147483646, -2147483647 - 1, -2147483647 };
  enum { NP = sizeof P / sizeof P[0] };
  int *pp = malloc((size_t)ni * NP * sizeof *pp);
  double *pl = malloc((size_t)ni * NP * 8), *ph = malloc((size_t)ni * NP * 8), *xl = malloc((size_t)ni * NP * 8),
         *xh = malloc((size_t)ni * NP * 8);
  int npw = 0;
  for (int i = 0; i < ni; i++) for (int k = 0; k < NP; k++) { xl[npw] = L[i]; xh[npw] = H[i]; pp[npw] = P[k]; npw++; }
  ival_pown(xl, xh, pp, pl, ph, npw);
  for (int j = 0; j < npw; j++) {
    double a = xl[j], c = xh[j], lo = INFINITY, hi = -INFINITY;
    int p = pp[j];
    if (empty(a, c) || (p < 0 && a == 0 && c == 0)) { lo = hi = NAN; }
    else if (p == 0) lo = hi = 1;
    else {
      double ends[2] = { a, c };
      for (int e = 0; e < 2; e++) {
        if (p < 0 && ends[e] == 0) continue;   /* the pole */
        mpfr_set_d(Y, ends[e], MPFR_RNDN);
        mpfr_pow_si(X, Y, p, MPFR_RNDD); double d = mpfr_get_d(X, MPFR_RNDD);
        mpfr_pow_si(X, Y, p, MPFR_RNDU); double u = mpfr_get_d(X, MPFR_RNDU);
        if (d < lo) lo = d;
        if (u > hi) hi = u;
      }
      if (p > 0 && !(p & 1) && a < 0 && c > 0) lo = 0;
      if (p < 0 && a < 0 && c >= 0) { double lim = (p & 1) ? -INFINITY : INFINITY; if (lim < lo) lo = lim; if (lim > hi) hi = lim; }
      if (p < 0 && a <= 0 && c > 0) { if (INFINITY > hi) hi = INFINITY; }
      lo += 0.0; hi += 0.0;
    }
    char nm[32]; snprintf(nm, sizeof nm, "pown %d", p);
    want(nm, a, c, 0, 0, pl[j], ph[j], lo, hi);
  }
  /* pown's double-double path (ival-eft.h, pown_dd, p 2 to 64) where it is closest to failing: points [x, x], whose
     result must be x^p rounded down and up (MPFR's pow_si at 53 bits, RNDD and RNDU), for every p from 2 to 64 and x
     near 1 (1 +- k ulps, where x^p is near a double and the low part near the error bound), small integers and short
     dyadic numbers (exact powers), random magnitudes, and near the underflow and overflow cut-offs; negative too */
  {
    enum { NX = 1200 };
    static double px[63 * NX], pr0[63 * NX], pr1[63 * NX];
    static int pe[63 * NX];
    int m = 0;
    for (int p = 2; p <= 64; p++)
      for (int k = 0; k < NX; k++) {
        double x, u = (double)(rnd() >> 11) * 0x1p-53;
        switch (k % 6) {
          case 0: x = 1; for (int s = (int)(rnd() % 40); s > 0; s--) x = nextafter(x, rnd() % 2 ? 2.0 : 0.0); break;
          case 1: x = (double)(int)(rnd() % 1000) / (double)(1 << (rnd() % 8)); break;
          case 2: x = ldexp(1 + u, (int)(rnd() % 40) - 20); break;
          case 3: x = pow(2, ((double)(rnd() % 2000) - 1000) / p) * (1 + u * 0x1p-30); break;   /* x^p anywhere */
          case 4: x = pow(2, (rnd() % 2 ? 1000.0 : -900.0) / p) * (1 + (u - 0.5) * 0x1p-40); break;   /* the cut-offs */
          default: x = 1 + ldexp(u - 0.5, -(int)(rnd() % 30)); break;
        }
        if (rnd() % 2) x = -x;
        px[m] = x; pe[m] = p; m++;
      }
    ival_pown(px, px, pe, pr0, pr1, m);
    for (int j = 0; j < m; j++) {
      mpfr_set_prec(X, 53); mpfr_set_d(X, px[j], MPFR_RNDN);
      mpfr_set_prec(Y, 53);
      mpfr_pow_si(Y, X, pe[j], MPFR_RNDD); double lo = mpfr_get_d(Y, MPFR_RNDD) + 0.0;   /* down again: DBL_MAX past it, and the subnormals */
      mpfr_set_d(X, px[j], MPFR_RNDN);
      mpfr_pow_si(Y, X, pe[j], MPFR_RNDU); double hi = mpfr_get_d(Y, MPFR_RNDU) + 0.0;
      char nm[32]; snprintf(nm, sizeof nm, "pown %d (point)", pe[j]);
      want(nm, px[j], px[j], 0, 0, pr0[j], pr1[j], lo, hi);
    }
    mpfr_set_prec(X, 2200); mpfr_set_prec(Y, 2200);
  }
  /* mulrev(B, C, X) and mulrevpair(B, C): containment, by exact rationals: for points b in B and c in C (finite),
     x = c / b in X must lie in mulrev's result (zl b <= c <= zh b for b > 0, reversed for b < 0, as MPFR products);
     for b = 0 and c = 0 every x is in S, so the result must be X itself. And mulrev with X the whole line must be the
     hull of mulrevpair's two intervals, the first before the second. Triples from the pairs above with X random. */
  {
    double *xl3 = malloc(np * 8), *xh3 = malloc(np * 8), *ml = malloc(np * 8), *mh = malloc(np * 8);
    double *p1l = malloc(np * 8), *p1h = malloc(np * 8), *p2l = malloc(np * 8), *p2h = malloc(np * 8);
    double *ninf = malloc(np * 8), *pinf = malloc(np * 8), *el = malloc(np * 8), *eh = malloc(np * 8);
    for (int k = 0; k < np; k++) { int i = (int)(rnd() % ni); xl3[k] = L[i]; xh3[k] = H[i]; ninf[k] = -INFINITY; pinf[k] = INFINITY; }
    ival_mulrev(al, ah, bl, bh, xl3, xh3, ml, mh, np);
    ival_mulrevpair(al, ah, bl, bh, p1l, p1h, p2l, p2h, np);
    ival_mulrev(al, ah, bl, bh, ninf, pinf, el, eh, np);
    long pts = 0;
    for (int k = 0; k < np; k++) {   /* B = [al, ah], C = [bl, bh], X = [xl3, xh3] */
      double hl = p1l[k], hh = p1h[k];
      if (p2l[k] == p2l[k]) { if (!(p1h[k] <= p2l[k])) hl = NAN; hh = p2h[k]; }
      want("mulrev whole line = hull of the pair", al[k], ah[k], bl[k], bh[k], el[k], eh[k], hl, hh);
      if (empty(al[k], ah[k]) || empty(bl[k], bh[k]) || empty(xl3[k], xh3[k])) continue;
      for (int q = 0; q < 4; q++) {
        double b = q == 3 && al[k] <= 0 && ah[k] >= 0 ? 0.0 : pick(al[k], ah[k]), c = q == 3 && bl[k] <= 0 && bh[k] >= 0 ? 0.0 : pick(bl[k], bh[k]);
        if (b == 0) {
          if (c != 0) continue;
          want("mulrev with 0 in B and C", al[k], ah[k], bl[k], bh[k], ml[k], mh[k], xl3[k] + 0.0, xh3[k] + 0.0);
          continue;
        }
        /* x = c / b against an interval [u, v]: u b <= c <= v b (b > 0) */
        #define IN(u, v) (inside_q(u, b, c, 0) && inside_q(v, b, c, 1))
        int inx = IN(xl3[k], xh3[k]);
        if (!inx) continue;
        pts++;
        if (!(ml[k] == ml[k] && IN(ml[k], mh[k]))) {
          bad++; checked++;
          if (!first[0]) snprintf(first, sizeof first, " (first: mulrev B [%a, %a] C [%a, %a] X [%a, %a]: %a / %a outside [%a, %a])",
                                  al[k], ah[k], bl[k], bh[k], xl3[k], xh3[k], c, b, ml[k], mh[k]);
        } else checked++;
      }
    }
    if (!pts) { bad++; snprintf(first, sizeof first, " (VOID: no mulrev point inside X)"); }
    /* tightness at the ends: X a single point d, at each finite end of the pair's intervals and the doubles beside
       it, and two random points. mulrev must give [d, d] exactly when d is in S, which in_s decides another way */
    int nd = 0, cap = np * 8;
    double *d0 = malloc(cap * 8), *qb0 = malloc(cap * 8), *qb1 = malloc(cap * 8), *qc0 = malloc(cap * 8), *qc1 = malloc(cap * 8);
    for (int k = 0; k < np && nd + 16 < cap; k++) {
      double ends[4] = { p1l[k], p1h[k], p2l[k], p2h[k] }, ds[16];
      int m2 = 0;
      for (int e = 0; e < 4; e++)
        if (isfinite(ends[e])) { ds[m2++] = ends[e]; ds[m2++] = nextafter(ends[e], INFINITY); ds[m2++] = nextafter(ends[e], -INFINITY); }
      if (!empty(al[k], ah[k]) && !empty(bl[k], bh[k])) { ds[m2++] = anyd(); ds[m2++] = pick(-1, 1); }
      for (int j = 0; j < m2; j++) {
        if (!isfinite(ds[j])) continue;   /* a point at an infinity is no interval */
        d0[nd] = ds[j]; qb0[nd] = al[k]; qb1[nd] = ah[k]; qc0[nd] = bl[k]; qc1[nd] = bh[k]; nd++;
      }
    }
    double *dl = malloc(nd * 8), *dh = malloc(nd * 8);
    ival_mulrev(qb0, qb1, qc0, qc1, d0, d0, dl, dh, nd);
    long members = 0;
    for (int j = 0; j < nd; j++) {
      int in = in_s(d0[j], qb0[j], qb1[j], qc0[j], qc1[j]);
      members += in;
      want("mulrev at a point", qb0[j], qb1[j], qc0[j], qc1[j], dl[j], dh[j], in ? d0[j] + 0.0 : NAN, in ? d0[j] + 0.0 : NAN);
    }
    if (!members || members == nd) { bad++; snprintf(first, sizeof first, " (VOID: the points were all in S or all out)"); }
  }
  if (!bad && control > 0)
    printf("VERDICT: IDENTICAL (%ld results as the references give them; control: the sum of the halves differs from the midpoint %ld times)\n",
           checked, control);
  else
    printf("VERDICT: DIFFERS (%ld of %ld, control %ld)%s\n", bad, checked, control, first);
  return bad || !control;
}
