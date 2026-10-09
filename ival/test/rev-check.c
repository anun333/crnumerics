/* rev-check.c: ival's reverse operations (sqrrev, absrev, coshrev, pownrev) and rootn (ival-rev.c), against exact
   decisions made another way.

   A reverse operation's result is the tightest interval around {x in X : f(x) in C}. For a point X = [d, d] that is
   [d, d] when f(d) is in C and empty otherwise, and whether f(d) is in C is decided here exactly: sqr and abs in
   MPFR at 2200 bits; cosh at 2200 bits (cosh(d) is a binary64 value only at d = 0); pown by d^|p| exactly (53 |p| +
   64 bits) and, for p < 0, C's ends multiplied by it instead of dividing. The points: each finite end of the result
   for X the whole line, the doubles beside it, and random points, for every pair of special intervals as C and
   random ones; 21 powers.

   rootn(x, q), q > 0: each bound proved by exact powers, as the double below the root at that end with the next one
   above it (or the root itself when it is a double), on every special and random interval and 8 values of q; q = 0
   and negative q only for being empty where they must (pownrev's negative powers above run the same root search).
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

static long checked, bad, members, outs, control;
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
      if (q < 0) continue;   /* the reciprocal of the root: pownrev's negative powers above run the same search */
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
  if (!bad && control > 0 && members > 0 && outs > 0)
    printf("VERDICT: IDENTICAL (%ld results as the exact decisions give them, %ld points members and %ld not; control: "
           "rootn moved outward fails %ld times)\n", checked, members, outs, control);
  else
    printf("VERDICT: DIFFERS (%ld of %ld, members %ld, not %ld, control %ld)%s\n", bad, checked, members, outs, control, first);
  return bad || !control || !members || !outs;
}
