/* rev-check.c: ival's reverse operations (sqrrev, absrev, coshrev, pownrev) and rootn (ival-rev.c), against exact
   decisions made another way.

   A reverse operation's result is the tightest interval around {x in X : f(x) in C}. For a point X = [d, d] that is
   [d, d] when f(d) is in C and empty otherwise, and whether f(d) is in C is decided here exactly: sqr and abs in
   MPFR at 2200 bits; cosh at 2200 bits (cosh(d) is a binary64 value only at d = 0); pown by d^|p| exactly (53 |p| +
   64 bits) and, for p < 0, C's ends multiplied by it instead of dividing. The points: each finite end of the result
   for X the whole line, the doubles beside it, and random points, for every pair of special intervals as C and
   random ones; 21 powers.

   rootn(x, q): each bound proved by exact powers, as the double below the root at that end with the next one above it
   (or the root itself when it is a double), on every special and random interval and 13 values of q; for q < 0 the
   root falls as x grows and y is below it exactly when y^|q| x <= 1; q = 0 must give the empty interval.
   Negative control: rootn's lower bounds moved one double outward must fail the proof somewhere. */
#include <float.h>
#include <math.h>
#include <mpfr.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ival.h"

static uint64_t rs = 0x9b97f4a7c15e3779ULL;
static uint64_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }
static double anyd(void) { for (;;) { uint64_t u = rnd(); double d; memcpy(&d, &u, 8); if (isfinite(d)) return d; } }
static int empty(double lo, double hi) { return !(lo <= hi) || lo == INFINITY || hi == -INFINITY; }
static int same(double a, double b) { return (a != a && b != b) || a == b; }

static long checked, bad, members, outs, control, negdone;
static char first[400];
static void want(const char *op, double cl, double ch, double d, double zl, double zh, double rl, double rh)
{
  checked++;
  if (same(zl, rl) && same(zh, rh)) return;
  if (!bad++) snprintf(first, sizeof first, " (first: %s C [%a, %a] at %a: [%a, %a], want [%a, %a])", op, cl, ch, d, zl, zh, rl, rh);
}

static mpfr_t T, U;
/* whether f(d) is in C = [cl, ch], exactly; f: 0 sqr, 1 abs, 2 cosh, 3 pown(p) */
static int member(int f, int p, double d, double cl, double ch)
{
  if (empty(cl, ch)) return 0;
  if (f == 3) {
    if (p == 0) return cl <= 1 && 1 <= ch;
    if (p < 0 && d == 0) return 0;
    unsigned q = p < 0 ? (unsigned)-(long)p : (unsigned)p;
    mpfr_set_prec(T, 53 * (mpfr_prec_t)q + 64); mpfr_set_prec(U, 53 * (mpfr_prec_t)q + 128);
    mpfr_set_d(T, d, MPFR_RNDN); mpfr_pow_ui(T, T, q, MPFR_RNDN);   /* d^|p|, exact */
    if (p > 0) return mpfr_cmp_d(T, cl) >= 0 && mpfr_cmp_d(T, ch) <= 0;
    /* 1 / t in [cl, ch]: for t > 0, cl t <= 1 <= ch t; for t < 0 reversed. Infinite ends are open: cl = -inf holds */
    int pos = mpfr_sgn(T) > 0, lo_ok, hi_ok;
    if (isinf(cl)) lo_ok = cl < 0; else { mpfr_mul_d(U, T, cl, MPFR_RNDN); lo_ok = pos ? mpfr_cmp_ui(U, 1) <= 0 : mpfr_cmp_ui(U, 1) >= 0; }
    if (isinf(ch)) hi_ok = ch > 0; else { mpfr_mul_d(U, T, ch, MPFR_RNDN); hi_ok = pos ? mpfr_cmp_ui(U, 1) >= 0 : mpfr_cmp_ui(U, 1) <= 0; }
    return lo_ok && hi_ok;
  }
  mpfr_set_prec(T, 2200);
  mpfr_set_d(T, d, MPFR_RNDN);
  if (f == 0) mpfr_sqr(T, T, MPFR_RNDN);
  else if (f == 1) mpfr_abs(T, T, MPFR_RNDN);
  else mpfr_cosh(T, T, MPFR_RNDN);
  return mpfr_cmp_d(T, cl) >= 0 && mpfr_cmp_d(T, ch) <= 0;
}
static void rev(int f, const double *cl, const double *ch, const double *xl, const double *xh, const int *p, double *zl,
                double *zh, size_t n)
{
  if (f == 0) ival_sqrrev(cl, ch, xl, xh, zl, zh, n);
  else if (f == 1) ival_absrev(cl, ch, xl, xh, zl, zh, n);
  else if (f == 2) ival_coshrev(cl, ch, xl, xh, zl, zh, n);
  else ival_pownrev(cl, ch, xl, xh, p, zl, zh, n);
}

/* y^q against x exactly: the sign of y^q - x, for y, x finite and q a positive int */
static int cmp_pow(double y, unsigned q, double x)
{
  mpfr_set_prec(T, 53 * (mpfr_prec_t)q + 64);
  mpfr_set_d(T, y, MPFR_RNDN); mpfr_pow_ui(T, T, q, MPFR_RNDN);
  return mpfr_cmp_d(T, x);
}
/* whether lo is the root of x (q-th, q > 0) rounded down: lo^q <= x, and the next double's power above x unless lo^q
   is x (signs: odd q over the line). up: the same for rounding up */
static int root_ok(double x, unsigned q, double b, int up)
{
  if (isinf(x)) return b == x;
  if (x == 0) return b == 0;
  if (isinf(b)) return 0;
  int c = cmp_pow(b, q, x);
  if (c == 0) return 1;
  double nb = nextafter(b, up ? -INFINITY : INFINITY);
  return up ? c > 0 && cmp_pow(nb, q, x) < 0 : c < 0 && cmp_pow(nb, q, x) > 0;
}

/* ---- the periodic reverses' reference: the set {x : f(x) in C'} as pieces, each an inverse function's values at
   C's ends plus k periods (sin: [asin cl, asin ch] and [pi - asin ch, pi - asin cl], period 2 pi; cos: [acos ch,
   acos cl] and [-acos cl, -acos ch]; tan: [atan cl, atan ch], period pi), in MPFR at 2200 bits. The lower end is the
   least of max(xl, s_k) over the first piece of each kind ending at or after xl, k = ceil((xl - e_0) / period), when
   it starts by xh; the upper end likewise. Rounded once, down and up: none of these values is a binary64 but 0. */
static mpfr_t P0, P1, P2, P3, PER;
static void tref_piece(int kind, int which, double cl, double ch, mpfr_t s0, mpfr_t e0)
{
  mpfr_t a, b;
  mpfr_inits2(2200, a, b, (mpfr_ptr)0);
  mpfr_set_d(a, cl, MPFR_RNDN); mpfr_set_d(b, ch, MPFR_RNDN);
  if (kind == 0) {
    mpfr_asin(a, a, MPFR_RNDN); mpfr_asin(b, b, MPFR_RNDN);
    if (!which) { mpfr_set(s0, a, MPFR_RNDN); mpfr_set(e0, b, MPFR_RNDN); }
    else { mpfr_const_pi(s0, MPFR_RNDN); mpfr_sub(s0, s0, b, MPFR_RNDN); mpfr_const_pi(e0, MPFR_RNDN); mpfr_sub(e0, e0, a, MPFR_RNDN); }
  } else if (kind == 1) {
    mpfr_acos(a, a, MPFR_RNDN); mpfr_acos(b, b, MPFR_RNDN);
    if (!which) { mpfr_set(s0, b, MPFR_RNDN); mpfr_set(e0, a, MPFR_RNDN); }
    else { mpfr_neg(s0, a, MPFR_RNDN); mpfr_neg(e0, b, MPFR_RNDN); }
  } else {
    mpfr_atan(s0, a, MPFR_RNDN); mpfr_atan(e0, b, MPFR_RNDN);   /* atan(+-inf) = +-pi/2 */
  }
  mpfr_clears(a, b, (mpfr_ptr)0);
}
static int tref(int kind, double cl, double ch, double xl, double xh, double *l, double *u)
{
  if (empty(cl, ch) || empty(xl, xh)) return 0;
  if (kind != 2) {
    if (ch < -1 || cl > 1) return 0;
    if (cl <= -1 && ch >= 1) { *l = xl + 0.0; *u = xh + 0.0; return 1; }
    if (cl < -1) cl = -1;
    if (ch > 1) ch = 1;
  } else if (cl == -INFINITY && ch == INFINITY) { *l = xl + 0.0; *u = xh + 0.0; return 1; }
  mpfr_const_pi(PER, MPFR_RNDN);
  if (kind != 2) mpfr_mul_2ui(PER, PER, 1, MPFR_RNDN);
  int found = 0;
  double lo = INFINITY, hi = -INFINITY;
  for (int which = 0; which < (kind == 2 ? 1 : 2); which++) {
    tref_piece(kind, which, cl, ch, P0, P1);   /* s0, e0 */
    /* the first piece ending at or after xl, and the last starting at or before xh */
    if (isinf(xl)) { lo = -INFINITY; found |= 1; }
    else {
      mpfr_set_d(P2, xl, MPFR_RNDN); mpfr_sub(P2, P2, P1, MPFR_RNDN); mpfr_div(P2, P2, PER, MPFR_RNDN); mpfr_ceil(P2, P2);   /* k */
      mpfr_mul(P3, P2, PER, MPFR_RNDN); mpfr_add(P3, P3, P0, MPFR_RNDN);   /* s_k */
      if (mpfr_cmp_d(P3, xh) <= 0) {
        double v = mpfr_cmp_d(P3, xl) <= 0 ? xl : mpfr_get_d(P3, MPFR_RNDD);
        if (v < lo) lo = v;
        found = 1;
      }
    }
    if (isinf(xh)) { hi = INFINITY; found |= 1; }
    else {
      mpfr_set_d(P2, xh, MPFR_RNDN); mpfr_sub(P2, P2, P0, MPFR_RNDN); mpfr_div(P2, P2, PER, MPFR_RNDN); mpfr_floor(P2, P2);
      mpfr_mul(P3, P2, PER, MPFR_RNDN); mpfr_add(P3, P3, P1, MPFR_RNDN);   /* e_k */
      if (mpfr_cmp_d(P3, xl) >= 0) {
        double v = mpfr_cmp_d(P3, xh) >= 0 ? xh : mpfr_get_d(P3, MPFR_RNDU);
        if (v > hi) hi = v;
        found = 1;
      }
    }
  }
  if (!found || lo > hi) return 0;
  *l = lo + 0.0; *u = hi + 0.0;
  return 1;
}

/* for q < 0: the root r = x^(1/q) of x > 0 falls as x grows, and y (> 0) is below it exactly when y^|q| x <= 1. The
   sign of y^|q| x - 1, exactly */
static int cmp_negroot(double y, unsigned q, double x)
{
  mpfr_set_prec(T, 53 * (mpfr_prec_t)q + 64); mpfr_set_prec(U, 53 * (mpfr_prec_t)q + 128);
  mpfr_set_d(T, y, MPFR_RNDN); mpfr_pow_ui(T, T, q, MPFR_RNDN);   /* exact */
  mpfr_mul_d(U, T, x, MPFR_RNDN);                                 /* exact */
  return mpfr_cmp_ui(U, 1);
}
/* whether b is x^(1/q) rounded down (up = 0) or up, x > 0, q < 0 (uq = -q): b is on its side of the root and the next
   double inward is not, or b is the root */
static int negroot_ok(double x, unsigned q, double b, int up)
{
  if (isinf(x)) return b == 0;              /* the root of +inf is +0 */
  if (x == 0) return b == INFINITY;         /* of +0, +inf */
  /* a finite positive x has a positive real root, which may lie past DBL_MAX (rounding up to +inf, down to
     DBL_MAX) or below the least subnormal (rounding down to 0, up to it) */
  if (b == INFINITY) return up && cmp_negroot(DBL_MAX, q, x) < 0;
  if (b == 0) return !up && cmp_negroot(DBL_TRUE_MIN, q, x) > 0;
  if (!(b > 0)) return 0;
  int c = cmp_negroot(b, q, x);             /* > 0: b above the root */
  if (c == 0) return 1;
  double nb = nextafter(b, up ? -INFINITY : INFINITY);
  if (up) return c > 0 && (nb == 0 || cmp_negroot(nb, q, x) < 0);
  return c < 0 && (isinf(nb) || cmp_negroot(nb, q, x) > 0);
}

static const double SP[] = { -INFINITY, -DBL_MAX, -1e300, -27, -8, -4, -2, -1.5, -1, -0.5, -0x1p-1022, -DBL_TRUE_MIN, 0.0,
                             DBL_TRUE_MIN, 0x1p-1022, 0.25, 0.5, 1, 1.5, 2, 4, 8, 27, 1e300, DBL_MAX, INFINITY };
enum { NSP = sizeof SP / sizeof SP[0] };

int main(void)
{
  mpfr_init2(T, 2200); mpfr_init2(U, 2200);
  enum { NR = 1 << 11 };
  static double L[NSP * NSP + NR], H[NSP * NSP + NR];
  int ni = 0;
  for (int i = 0; i < NSP; i++) for (int j = i; j < NSP; j++) { L[ni] = SP[i]; H[ni] = SP[j]; ni++; }
  for (int k = 0; k < NR; k++) {
    double a = anyd(), b = k % 2 ? anyd() : a * (1 + (double)(rnd() >> 11) * 0x1p-60);
    if (k % 3 == 0) { a = fabs(a) < 1e300 ? 1 + a * 1e-300 : a; b = a + (double)(rnd() >> 11) * 0x1p-50; }   /* near 1 */
    if (b < a) { double t = a; a = b; b = t; }
    L[ni] = a; H[ni] = b; ni++;
  }
  L[ni] = NAN; H[ni] = NAN; ni++;
  static const int P[] = { 0, 1, 2, 3, 4, 5, 7, 10, 31, -1, -2, -3, -4, -5, -7, -10, -31, 64, -64, 101, -101 };
  enum { NP = sizeof P / sizeof P[0] };
  size_t cap = (size_t)ni * 16 * NP;
  double *cl = malloc(cap * 8), *ch = malloc(cap * 8), *d = malloc(cap * 8), *zl = malloc(cap * 8), *zh = malloc(cap * 8);
  double *wl = malloc(ni * 8), *wh = malloc(ni * 8), *ninf = malloc(ni * 8), *pinf = malloc(ni * 8);
  int *pp = malloc(cap * sizeof *pp), *pq = malloc(ni * sizeof *pq);
  for (int i = 0; i < ni; i++) { ninf[i] = -INFINITY; pinf[i] = INFINITY; }
  const char *names[4] = { "sqrrev", "absrev", "coshrev", "pownrev" };
  for (int f = 0; f < 4; f++)
    for (int pk = 0; pk < (f == 3 ? NP : 1); pk++) {
      int p = f == 3 ? P[pk] : 0;
      for (int i = 0; i < ni; i++) pq[i] = p;
      rev(f, L, H, ninf, pinf, pq, wl, wh, ni);   /* X the whole line */
      size_t m = 0;
      for (int i = 0; i < ni; i++) {
        double ds[8];
        int nd = 0;
        if (isfinite(wl[i])) { ds[nd++] = wl[i]; ds[nd++] = nextafter(wl[i], -INFINITY); ds[nd++] = nextafter(wl[i], INFINITY); }
        if (isfinite(wh[i])) { ds[nd++] = wh[i]; ds[nd++] = nextafter(wh[i], -INFINITY); ds[nd++] = nextafter(wh[i], INFINITY); }
        ds[nd++] = anyd(); ds[nd++] = (double)(int)(rnd() % 7) - 3;
        for (int j = 0; j < nd; j++) {
          if (!isfinite(ds[j])) continue;
          cl[m] = L[i]; ch[m] = H[i]; d[m] = ds[j]; pp[m] = p; m++;
        }
      }
      rev(f, cl, ch, d, d, pp, zl, zh, m);
      for (size_t k = 0; k < m; k++) {
        int in = member(f, p, d[k], cl[k], ch[k]);
        members += in; outs += !in;
        char nm[40]; snprintf(nm, sizeof nm, "%s %d", names[f], p);
        want(nm, cl[k], ch[k], d[k], zl[k], zh[k], in ? d[k] + 0.0 : NAN, in ? d[k] + 0.0 : NAN);
      }
    }
  /* rootn */
  static const int Q[] = { 1, 2, 3, 4, 5, 7, 31, 101, -1, -2, -3, -5, -31, 0 };
  enum { NQ = sizeof Q / sizeof Q[0] };
  for (int qk = 0; qk < NQ; qk++) {
    int q = Q[qk];
    for (int i = 0; i < ni; i++) pq[i] = q;
    ival_rootn(L, H, pq, wl, wh, ni);
    for (int i = 0; i < ni; i++) {
      double a = L[i], b = H[i];
      checked++;
      int e = empty(a, b) || q == 0 || (!(q & 1) && b < 0) || (q < 0 && a == 0 && b == 0);
      if (e) { if (!(wl[i] != wl[i] && wh[i] != wh[i])) { if (!bad++) snprintf(first, sizeof first, " (first: rootn %d [%a, %a] not empty)", q, a, b); } continue; }
      if (q < 0) {   /* decreasing: the lower end is the root at b, the upper the root at a */
        unsigned uq = (unsigned)-q;
        int ok;
        if (a >= 0) ok = negroot_ok(b, uq, wl[i], 0) && negroot_ok(a, uq, wh[i], 1);
        else if (b <= 0) ok = negroot_ok(-b, uq, -wl[i], 1) && negroot_ok(-a, uq, -wh[i], 0);   /* odd: minus the root of -x */
        else ok = wl[i] == -INFINITY && wh[i] == INFINITY;                                       /* odd, 0 inside */
        if (!(q & 1) && a < 0) ok = negroot_ok(b, uq, wl[i], 0) && negroot_ok(0, uq, wh[i], 1); /* even: x >= 0 */
        if (!ok) { if (!bad++) snprintf(first, sizeof first, " (first: rootn %d [%a, %a] = [%a, %a])", q, L[i], H[i], wl[i], wh[i]); }
        else negdone++;
        continue;
      }
      if (!(q & 1) && a < 0) a = 0;
      unsigned uq = (unsigned)q;
      int ok_lo = a < 0 ? root_ok(-a, uq, -wl[i], 1) : root_ok(a, uq, wl[i], 0);
      int ok_hi = b < 0 ? root_ok(-b, uq, -wh[i], 0) : root_ok(b, uq, wh[i], 1);
      if (!ok_lo || !ok_hi) { if (!bad++) snprintf(first, sizeof first, " (first: rootn %d [%a, %a] = [%a, %a])", q, L[i], H[i], wl[i], wh[i]); }
      /* the control: one double outward must fail */
      double ml = nextafter(wl[i], -INFINITY);
      if (isfinite(wl[i]) && wl[i] != 0 && !(a < 0 ? root_ok(-a, uq, -ml, 1) : root_ok(a, uq, ml, 0))) control++;
    }
  }
  /* sinrev, cosrev, tanrev against tref, on random C (in [-1, 1] for sin and cos: near 0, near +-1, points, wide;
     for tan anywhere, poles' neighbourhoods and infinities included) and X (around 0, near multiples of pi / 2,
     large, huge, half-lines) */
  mpfr_inits2(2200, P0, P1, P2, P3, PER, (mpfr_ptr)0);
  long tchecked = 0;
  for (int kind = 0; kind < 3; kind++) {
    enum { NT = 6000 };
    static double c0[NT], c1[NT], x0[NT], x1[NT], z0[NT], z1[NT];
    for (int k = 0; k < NT; k++) {
      double a, b, t = (double)(rnd() >> 11) * 0x1p-53, w = (double)(rnd() >> 11) * 0x1p-53;
      switch (rnd() % 6) {   /* C */
        case 0: a = 2 * t - 1; b = a + w * (1 - a); break;                                  /* anywhere in [-1, 1] */
        case 1: a = 1 - ldexp(t, -(int)(rnd() % 60)); b = a + ldexp(w, -(int)(rnd() % 60)); break;   /* near 1 */
        case 2: b = -1 + ldexp(t, -(int)(rnd() % 60)); a = b - ldexp(w, -(int)(rnd() % 60)); break;  /* near -1 */
        case 3: a = ldexp(t - 0.5, -(int)(rnd() % 1000)); b = a + ldexp(w, -(int)(rnd() % 1000)); break;  /* near 0 */
        case 4: a = b = 2 * t - 1; break;                                                   /* a point */
        default: a = (2 * t - 1) * 3; b = a + w * 3; break;                                /* past [-1, 1] too */
      }
      if (kind == 2) {   /* tan: values of any size */
        int e = (int)(rnd() % 120) - 60;
        a = ldexp(2 * t - 1, e); b = a + ldexp(w, e);
        if (rnd() % 10 == 0) a = -INFINITY;
        if (rnd() % 10 == 0) b = INFINITY;
      }
      if (b > (kind == 2 ? INFINITY : 3)) b = kind == 2 ? INFINITY : 3;
      c0[k] = a; c1[k] = b;
      double u = (double)(rnd() >> 11) * 0x1p-53, v = (double)(rnd() >> 11) * 0x1p-53;
      switch (rnd() % 6) {   /* X */
        case 0: a = (u - 0.5) * 20; b = a + v * 10; break;
        case 1: { double m = (double)((int)(rnd() % 40) - 20) * M_PI / 2; a = m - ldexp(u, -(int)(rnd() % 50)); b = m + ldexp(v, -(int)(rnd() % 50)); break; }
        case 2: a = (u - 0.5) * 1e6; b = a + v * 10; break;
        case 3: a = ldexp(u, (int)(rnd() % 1000)); b = a + ldexp(v, (int)(rnd() % 1000)); break;   /* up to huge */
        case 4: a = ldexp(u - 0.5, (int)(rnd() % 60)); b = a; break;                              /* a point */
        default: a = rnd() % 2 ? -INFINITY : (u - 0.5) * 100; b = rnd() % 2 ? INFINITY : a + v * 100; break;
      }
      if (b < a) { double t2 = a; a = b; b = t2; }
      x0[k] = a; x1[k] = b;
    }
    if (kind == 0) ival_sinrev(c0, c1, x0, x1, z0, z1, NT);
    else if (kind == 1) ival_cosrev(c0, c1, x0, x1, z0, z1, NT);
    else ival_tanrev(c0, c1, x0, x1, z0, z1, NT);
    for (int k = 0; k < NT; k++) {
      double rl = NAN, rh = NAN;
      if (!tref(kind, c0[k], c1[k], x0[k], x1[k], &rl, &rh)) rl = rh = NAN;
      tchecked++;
      want(kind == 0 ? "sinrev" : kind == 1 ? "cosrev" : "tanrev", c0[k], c1[k], x0[k], z0[k], z1[k], rl, rh);
    }
  }
  if (!negdone) { bad++; snprintf(first, sizeof first, " (VOID: no negative root proved)"); }
  if (!bad && control > 0 && members > 0 && outs > 0)
    printf("VERDICT: IDENTICAL (%ld results as the exact decisions give them, %ld points members and %ld not, %ld negative "
           "roots proved; control: rootn moved outward fails %ld times)\n", checked, members, outs, negdone, control);
  else
    printf("VERDICT: DIFFERS (%ld of %ld, members %ld, not %ld, control %ld)%s\n", bad, checked, members, outs, control, first);
  return bad || !control || !members || !outs;
}
