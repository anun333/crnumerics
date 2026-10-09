/* ival-rev.c: IEEE 1788.1's reverse operations for sqr, abs, pown and cosh, and rootn (ival.h), 2026-10-09.

   fRev(C, X) is the tightest interval around {x in X : f(x) in C}. The set {x : f(x) in C} comes as at most two real
   pieces whose ends are an inverse function's values at C's ends, each rounded both ways; meet_hull (ival-eft.h)
   intersects them with X exactly and takes the hull. sqr, abs and cosh are even, so their pieces are P and -P, P's
   ends sqrt, the identity or acosh (correctly rounded in every mode). pown's inverse is a root, which has no
   correctly rounded implementation here, so root2 finds it by search over the doubles: CORE-MATH's pow is correctly
   rounded, so whether a double is below the root is decided exactly (for p > 0, y^p <= c exactly when y^p rounded up
   is), and the largest double below it is the root rounded down. rootn(x, q) is the same root, used forward. */
#include <fenv.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include "ival.h"
#include "ival-eft.h"

double cr_acosh(double), cr_pow(double, double);

#define ENTER fenv_t env; fegetenv(&env); fesetround(FE_TONEAREST); flush_off();
#define LEAVE fesetenv(&env);

static double sqrt1(double x) { return sqrt(x); }
static double ident(double x) { return x; }
/* g(x) rounded down and up: one evaluation in each mode, the mode back to nearest after. x is read through a
   volatile each time: otherwise GCC computes an inlined sqrt(x) once for both modes (-frounding-math or not). */
static void both(double (*g)(double), double x, double *d, double *u)
{
#if IVAL_PLANT_ARITH == 26   /* 26: x read once, so GCC may compute sqrt(x) once for both modes */
  double v = x;
#else
  volatile double v = x;
#endif
  fesetround(FE_DOWNWARD); *d = g(v);
  fesetround(FE_UPWARD); *u = g(v);
  fesetround(FE_TONEAREST);
}

/* ---- the root c^(1/p), c >= 0, p != 0, rounded down and up ---- */
static double pw(double y, int p, int up)
{
  fesetround(up ? FE_UPWARD : FE_DOWNWARD);
  double r = cr_pow(y, (double)p);
  fesetround(FE_TONEAREST);
  return r;
}
static double from(uint64_t b) { double y; memcpy(&y, &b, 8); return y; }
static uint64_t bits(double y) { uint64_t b; memcpy(&b, &y, 8); return b; }
/* y (>= 0) is at most the root: y^p <= c for p > 0, y^p >= c for p < 0 (y^p decreasing), decided exactly */
static int below(double y, double c, int p)
{
#if IVAL_PLANT_ARITH == 24   /* 24: the predicate on the nearest y^p */
  fesetround(FE_TONEAREST); double r = cr_pow(y, (double)p); return p > 0 ? r <= c : r >= c;
#endif
  return p > 0 ? pw(y, p, 1) <= c : pw(y, p, 0) >= c;
}
static void root2(double c, int p, double *d, double *u)
{
  if (c == 0) { *d = *u = p > 0 ? 0 : INFINITY; return; }
  if (c == INFINITY) { *d = *u = p > 0 ? INFINITY : 0; return; }
  if (p == 1) { *d = *u = c; return; }
  if (p == -1) { *d = div_r(1, c, 0); *u = div_r(1, c, 1); return; }
  if (p == 2) { both(sqrt1, c, d, u); return; }
  /* |p| >= 2 keeps the root within [2^-537, 2^537]. Start from pow(c, 1/p) to nearest, bracket by doubling steps
     over the bit patterns (which order the positive doubles), then bisect: below(lo), not below(hi) */
  uint64_t b = bits(cr_pow(c, 1.0 / p)), lo, hi, step = 1;
  if (below(from(b), c, p)) {
    lo = b;
    while (below(from(b + step), c, p)) { lo = b + step; step *= 2; }
    hi = b + step;
  } else {
    hi = b;
    while (step < b && !below(from(b - step), c, p)) { hi = b - step; step *= 2; }
    lo = step < b ? b - step : 0;   /* +0 is below any root */
  }
  while (hi - lo > 1) {
    uint64_t m = lo + (hi - lo) / 2;
    if (below(from(m), c, p)) lo = m; else hi = m;
  }
  *d = from(lo);
  *u = pw(*d, p, 0) == c && pw(*d, p, 1) == c ? *d : from(lo + 1);   /* exact when d^p is c */
}

/* ---- the pieces ---- */
static struct piece pc(double ld, double lu, double hd, double hu) { struct piece p = { ld, lu, hd, hu, 0, 0 }; return p; }
/* P = [a, h] in [0, inf] by its ends rounded both ways, and its mirror -P */
static int even(double ad, double au, double hd, double hu, int lopen, struct piece q[2])
{
  q[0] = pc(-hu, -hd, -au, -ad); q[0].hopen = lopen;
  q[1] = pc(ad, au, hd, hu); q[1].lopen = lopen;
#if IVAL_PLANT_ARITH == 25   /* 25: the mirror's open end on the wrong side */
  q[0].hopen = 0; q[0].lopen = lopen;
#endif
  return 2;
}
/* {x : x^p in C}, C nonempty, p != 0 */
static int pown_pieces(double cl, double ch, int p, struct piece q[2])
{
  double ad, au, hd, hu;
  if (p > 0 && (p & 1)) {   /* odd: x^p is a bijection of the line; a root of a negative c is minus the root of -c */
    double d, u;
    if (cl < 0) { root2(-cl, p, &d, &u); ad = -u; au = -d; } else root2(cl, p, &ad, &au);
    if (ch < 0) { root2(-ch, p, &d, &u); hd = -u; hu = -d; } else root2(ch, p, &hd, &hu);
    q[0] = pc(ad, au, hd, hu);
    return 1;
  }
  if (p > 0) {              /* even: |x| in [root(max(cl, 0)), root(ch)] */
    if (ch < 0) return 0;
    root2(cl > 0 ? cl : 0, p, &ad, &au); root2(ch, p, &hd, &hu);
    return even(ad, au, hd, hu, 0, q);
  }
  /* p < 0: x^p is never 0; for x > 0 it falls from +inf to 0 */
  int k = 0;
  if (!(p & 1)) {           /* even: |x| in [root(ch), root(max(cl, 0))], open at 0 when ch is +inf (never reached) */
    if (ch <= 0) return 0;
    root2(ch, p, &ad, &au); root2(cl > 0 ? cl : 0, p, &hd, &hu);
    return even(ad, au, hd, hu, ch == INFINITY, q);
  }
  if (ch > 0) {             /* odd, x > 0: x^p in (0, ch] from x in [root(ch), root(max(cl, 0))] */
    root2(ch, p, &ad, &au); root2(cl > 0 ? cl : 0, p, &hd, &hu);
    q[k] = pc(ad, au, hd, hu); q[k].lopen = ch == INFINITY; k++;
  }
  if (cl < 0) {             /* x < 0: x^p = -(-x)^p in [cl, min(ch, 0)) from x in [-root(-min(ch, 0)), -root(-cl)] */
    root2(ch < 0 ? -ch : 0, p, &hd, &hu); root2(-cl, p, &ad, &au);
    q[k] = pc(-hu, -hd, -au, -ad); q[k].hopen = cl == -INFINITY; k++;
  }
  return k;
}

#define REV(name, ...)                                                                                        \
  void ival_##name(const double *clo, const double *chi, const double *xlo, const double *xhi, double *zlo,   \
                   double *zhi, size_t n)                                                                     \
  {                                                                                                           \
    ENTER                                                                                                     \
    for (size_t i = 0; i < n; i++) {                                                                          \
      double cl = clo[i], ch = chi[i], ad, au, hd, hu;                                                        \
      struct piece q[2];                                                                                      \
      int k = 0;                                                                                              \
      if (!empty(cl, ch) && !empty(xlo[i], xhi[i])) { __VA_ARGS__ }                                           \
      meet_hull(q, k, xlo[i], xhi[i], &zlo[i], &zhi[i]);                                                      \
      (void)ad; (void)au; (void)hd; (void)hu;                                                                 \
    }                                                                                                         \
    LEAVE                                                                                                     \
  }
/* sqr: |x| in [sqrt(max(cl, 0)), sqrt(ch)] */
REV(sqrrev, if (ch >= 0) { both(sqrt1, cl > 0 ? cl : 0, &ad, &au); both(sqrt1, ch, &hd, &hu); k = even(ad, au, hd, hu, 0, q); })
/* abs: |x| in [max(cl, 0), ch] */
REV(absrev, if (ch >= 0) { both(ident, cl > 0 ? cl : 0, &ad, &au); both(ident, ch, &hd, &hu); k = even(ad, au, hd, hu, 0, q); })
/* cosh: |x| in [acosh(max(cl, 1)), acosh(ch)] */
REV(coshrev, if (ch >= 1) { both(cr_acosh, cl > 1 ? cl : 1, &ad, &au); both(cr_acosh, ch, &hd, &hu); k = even(ad, au, hd, hu, 0, q); })

void ival_pownrev(const double *clo, const double *chi, const double *xlo, const double *xhi, const int *p,
                  double *zlo, double *zhi, size_t n)
{
  ENTER
  for (size_t i = 0; i < n; i++) {
    double cl = clo[i], ch = chi[i];
    struct piece q[2];
    int k = 0;
    if (!empty(cl, ch) && !empty(xlo[i], xhi[i])) {
      if (p[i] == 0) { if (cl <= 1 && 1 <= ch) { q[0] = pc(-INFINITY, -INFINITY, INFINITY, INFINITY); k = 1; } }
      else k = pown_pieces(cl, ch, p[i], q);
    }
    meet_hull(q, k, xlo[i], xhi[i], &zlo[i], &zhi[i]);
  }
  LEAVE
}

/* rootn(x, q): the real q-th root, for q != 0 (q = 0 gives the empty interval). Odd q: over the whole line, the root
   of a negative x minus the root of -x; even q: over x >= 0. A negative q is the reciprocal: decreasing, with a pole
   at 0 (an odd one reaches -inf from the left), and [0, 0] alone empty. Each bound is root2 at an end. */
void ival_rootn(const double *lo, const double *hi, const int *q, double *ylo, double *yhi, size_t n)
{
  ENTER
  for (size_t i = 0; i < n; i++) {
    double a = lo[i], b = hi[i], d, u, l = NAN, h = NAN;
    int p = q[i], odd = p & 1;
    if (empty(a, b) || p == 0 || (!odd && b < 0) || (p < 0 && a == 0 && b == 0)) { ylo[i] = yhi[i] = NAN; continue; }
    if (!odd && a < 0) a = 0;   /* even: within the domain */
    if (p > 0) {                /* increasing */
      if (a < 0) { root2(-a, p, &d, &u); l = -u; } else { root2(a, p, &d, &u); l = d; }
      if (b < 0) { root2(-b, p, &d, &u); h = -d; } else { root2(b, p, &d, &u); h = u; }
    } else if (a >= 0) {        /* decreasing on x >= 0, +inf at 0 */
      root2(b, p, &d, &u); l = d;
      root2(a, p, &d, &u); h = u;
    } else if (b <= 0) {        /* odd, x <= 0: -root(-x), decreasing, -inf at 0 */
      root2(-b, p, &d, &u); l = -u;
      root2(-a, p, &d, &u); h = -d;
    } else { l = -INFINITY; h = INFINITY; }   /* odd, 0 inside */
    ylo[i] = canon(l); yhi[i] = canon(h);
  }
  LEAVE
}
