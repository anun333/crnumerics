/* ival.c: interval versions of elementary functions (ival.h).

   Each bound is a CORE-MATH value: the lower one computed rounding down,
   the upper one rounding up, at the right point of the interval:
     - monotone functions: an endpoint (after intersecting with the domain);
     - cosh: its minimum 1 when the interval holds 0;
     - sinpi, cospi, tanpi: their extrema and poles are at integers and
       half-integers, which are binary64 values (or, beyond 2^52, lie
       between binary64 integers). They are counted exactly with floor and
       ceil, and an extremum inside gives 1 or -1, a pole infinity.
     - sin, cos, tan: the interval is cut into pieces shorter than half a
       period, so a piece holds at most one extremum or pole. The sign of
       the derivative at a piece's ends says which (cos for sin, -sin for
       cos; a sign change of cos is a pole of tan). Those signs are exact:
       CORE-MATH's results are correctly rounded, and cos is never zero at
       a binary64 value (sin is, at 0 only, where cos has its maximum 1:
       the value says which side the slope is on). Beyond 2^54, neighbours
       are 4 apart, more than pi: such a piece holds one or two critical
       points, and the parity of the sign changes says which. An interval
       8 wide or more holds a whole period.
   No high-precision pi is needed, and nothing depends on how a piece's
   ends round: any cut into short enough pieces gives the same union. */
#include <fenv.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ival.h"
#include "ival-eft.h"   /* flush_off */

double cr_acos(double), cr_acosh(double), cr_acospi(double), cr_asin(double), cr_asinh(double), cr_asinpi(double),
  cr_atan(double), cr_atanh(double), cr_atanpi(double), cr_cbrt(double), cr_cos(double), cr_cosh(double),
  cr_cospi(double), cr_erf(double), cr_erfc(double), cr_exp(double), cr_exp10(double), cr_exp2(double),
  cr_expm1(double), cr_log(double), cr_log10(double), cr_log1p(double), cr_log2(double), cr_rsqrt(double),
  cr_sin(double), cr_sinh(double), cr_sinpi(double), cr_tan(double), cr_tanh(double), cr_tanpi(double),
  cr_hypot(double, double), cr_atan2(double, double), cr_pow(double, double), cr_tgamma(double);

enum { INC, DEC, COSH, SIN, COS, TAN, SINPI, COSPI, TANPI };
/* a function, its shape, and its domain [dlo, dhi] (open at an end where
   the function has a pole or no limit: log at 0, atanh at +-1) */
typedef struct { double (*f)(double); int kind; double dlo, dhi; int lo_open, hi_open; } fn;

static int sgn(double v) { return (v > 0) - (v < 0); }

/* the derivative's sign just right (right = 1) or just left of x, for sin
   and cos: where it is zero (cos at 0), the value there says which way.
   Called rounding to nearest (run's first pass): rounding down or up could
   make a tiny nonzero result (sin near 0) zero */
static int slope(int kind, double x, int right)
{
  int d = kind == SIN ? sgn(cr_cos(x)) : -sgn(cr_sin(x));
  if (d) return d;
  double v = kind == SIN ? cr_sin(x) : cr_cos(x);   /* 1 or -1, exactly */
  return (v > 0) == right ? -1 : 1;
}

static double lesser(double a, double b) { return a < b ? a : b; }
static double greater(double a, double b) { return a > b ? a : b; }

/* sin, cos and tan over [a, b] shorter than 8 are cut into k pieces of 2.5
   and a hair, under pi unless neighbours are further apart; what each needs
   rounding to nearest, the slopes at its ends (tan: whether cos changes
   sign, a pole inside), is worked out once by pieces() in run's first pass,
   for both bounds. 2026-10-09: before, each bound switched the mode to
   nearest and back for every slope, most of the time sin and cos took. */
struct pinfo { int k; double p[5]; signed char du[4], dv[4], pole[4], longer[4]; };
static void pieces(const fn *F, double a, double b, struct pinfo *P)
{
  double w = b - a;
  P->k = 0;
  if (!(w < 8)) return;   /* a whole period inside (or an infinite end): no pieces */
  int k = (int)ceil(w / 2.5);
  if (k < 1) k = 1;
  P->k = k;
  P->p[0] = a;
  for (int i = 1; i <= k; i++) {
    double u = P->p[i - 1], v = i == k ? b : a + w * i / k;
    if (v < u) v = u;
    P->p[i] = v;
    P->longer[i - 1] = v - u > 3.2;   /* exact here: 4 (neighbours beyond 2^54), or at most 2.5 and a hair */
    if (F->kind == TAN) P->pole[i - 1] = sgn(cr_cos(u)) * sgn(cr_cos(v)) < 0;
    else if (u != v) { P->du[i - 1] = (signed char)slope(F->kind, u, 1); P->dv[i - 1] = (signed char)slope(F->kind, v, 0); }
  }
}
/* the lower (up = 0, rounding down) or upper (up = 1, rounding up) bound of
   sin, cos or tan over piece i of P: shorter than pi, or (sparse
   neighbours) shorter than 2 pi and longer than pi */
static double piece(const fn *F, const struct pinfo *P, int i, int up)
{
  double u = P->p[i], v = P->p[i + 1];
  if (u == v) return F->f(u);   /* a point: nothing inside */
  if (F->kind == TAN) {
    if (P->longer[i] || P->pole[i]) return up ? INFINITY : -INFINITY;   /* a pole inside */
    return F->f(up ? v : u);
  }
  int du = P->du[i], dv = P->dv[i];
  /* the extrema inside: at most one (shorter than pi), or one or two
     (longer): an odd number when the slope changes sign */
  int change = du != dv;
  if (P->longer[i] && !change) return up ? 1 : -1;   /* two: a maximum and a minimum */
  if (!change) return F->f((du > 0) == up ? v : u);   /* monotone, the slope's way: one end, rounded monotonically too */
  if (du > 0) return up ? 1 : lesser(F->f(u), F->f(v));   /* a maximum inside */
  return up ? greater(F->f(u), F->f(v)) : -1;             /* a minimum inside */
}

/* sinpi, cospi, tanpi over [a, b], under 4 wide: their critical points
   k + c (c = 1/2 for sinpi's extrema and tanpi's poles, 0 for cospi's
   extrema), counted exactly. Below 2^52, x - 1/2 is a binary64 value;
   from 2^52 on, every binary64 is an integer, so ceil(x - 1/2) = x and
   floor(x - 1/2) = x - 1. Counts are differences of integers under 4
   apart: exact. */
static double pibound(const fn *F, double a, double b, int up)
{
  double first, count;   /* the first k, and how many */
  int half = F->kind != COSPI;
  if (!half) {
    first = ceil(a);
    count = floor(b) - first + 1;
  } else {
    first = fabs(a) >= 0x1p52 ? a : ceil(a - 0.5);
    count = (fabs(b) >= 0x1p52 ? b - first - 1 : floor(b - 0.5) - first) + 1;
  }
  if (F->kind == TANPI) {
    if (count <= 0) return F->f(up ? b : a);   /* increasing, no pole */
    /* the poles are first + 1/2 on; at an end only if that is a binary64 */
    int at_a = fabs(first) < 0x1p52 && first + 0.5 == a;
    int at_b = count == 1 + at_a ? fabs(first) < 0x1p52 && first + count - 0.5 == b : 0;
    if (count > at_a + at_b) return up ? INFINITY : -INFINITY;   /* a pole inside */
    if (!up) return at_a ? -INFINITY : F->f(a);   /* a pole at a: coming from its right */
    return at_b ? INFINITY : F->f(b);             /* at b: from its left */
  }
  double fa = F->f(a), fb = F->f(b), lo = lesser(fa, fb), hi = greater(fa, fb);
  if (count >= 2) return up ? 1 : -1;   /* a maximum and a minimum */
  if (count == 1) {   /* sinpi(k + 1/2) = cospi(k) = (-1)^k */
    double v = fmod(fabs(first), 2) == 0 ? 1 : -1;
    lo = lesser(lo, v);
    hi = greater(hi, v);
  }
  return up ? hi : lo;
}

/* one bound over [a, b], inside the domain, a <= b, in the current mode (P: sin, cos and tan's pieces) */
static double bound(const fn *F, double a, double b, int up, const struct pinfo *P)
{
  switch (F->kind) {
  case INC: return F->f(up ? b : a);
  case DEC: return F->f(up ? a : b);
  case COSH:
    if (!up) return a <= 0 && 0 <= b ? 1 : F->f(a > 0 ? a : b);
    else {
      double u = F->f(a), v = F->f(b);
      return u > v ? u : v;
    }
  case SINPI: case COSPI: case TANPI:
    if (!(b - a < 4)) return F->kind == TANPI ? (up ? INFINITY : -INFINITY) : (up ? 1 : -1);   /* a whole period */
    return pibound(F, a, b, up);
  default: {   /* SIN, COS, TAN */
    if (!P->k) {   /* a whole period inside (or an infinite end) */
      if (F->kind == TAN) return up ? INFINITY : -INFINITY;
      return up ? 1 : -1;
    }
    double r = piece(F, P, 0, up);
    for (int i = 1; i < P->k; i++) {
      double v = piece(F, P, i, up);
      r = up ? greater(v, r) : lesser(v, r);
    }
    return r;
  }
  }
}

/* In blocks of 64: one pass rounding to nearest (the domain, tanpi's lone
   poles, sin, cos and tan's pieces), then every lower bound rounding down,
   then every upper bound rounding up: three mode switches a block, not
   two or more an interval. The block's results go out last, so ylo and yhi
   may be lo and hi. */
#define BLK 64
static void run(const fn *F, const double *lo, const double *hi, double *ylo, double *yhi, size_t n)
{
  fenv_t env;
  fegetenv(&env);
  flush_off();   /* a -ffast-math caller's flush modes would break the bounds; fesetenv gives them back */
  double A[BLK], Bd[BLK], L[BLK], U[BLK];
  unsigned char E[BLK];
  struct pinfo P[BLK];
  int trig = F->kind == SIN || F->kind == COS || F->kind == TAN;
  for (size_t i0 = 0; i0 < n; i0 += BLK) {
    size_t m = n - i0 < BLK ? n - i0 : BLK;
    fesetround(FE_TONEAREST);
    for (size_t k = 0; k < m; k++) {
      double a = lo[i0 + k], b = hi[i0 + k];
      /* the intersection with the domain; empty if there is none */
      if (a < F->dlo) a = F->dlo;
      if (b > F->dhi) b = F->dhi;
      int empty = !(a <= b) || (F->lo_open && b <= F->dlo) || (F->hi_open && a >= F->dhi);
      if (!empty && a == b && F->kind == TANPI && cr_cospi(a) == 0) empty = 1;   /* only a pole */
      if (a == 0) a = 0.0;   /* +0: a domain ending at 0 starts at +0 (rsqrt(+0) = +inf) */
      E[k] = (unsigned char)empty; A[k] = a; Bd[k] = b;
      if (!empty && trig) pieces(F, a, b, &P[k]);
    }
    fesetround(FE_DOWNWARD);
    for (size_t k = 0; k < m; k++) if (!E[k]) L[k] = bound(F, A[k], Bd[k], 0, &P[k]);
    fesetround(FE_UPWARD);
    for (size_t k = 0; k < m; k++) if (!E[k]) U[k] = bound(F, A[k], Bd[k], 1, &P[k]);
    for (size_t k = 0; k < m; k++) { ylo[i0 + k] = E[k] ? NAN : L[k]; yhi[i0 + k] = E[k] ? NAN : U[k]; }
  }
  fesetenv(&env);
}

#define ALL -INFINITY, INFINITY, 0, 0
static const fn
  F_acos = {cr_acos, DEC, -1, 1, 0, 0}, F_acosh = {cr_acosh, INC, 1, INFINITY, 0, 0},
  F_acospi = {cr_acospi, DEC, -1, 1, 0, 0}, F_asin = {cr_asin, INC, -1, 1, 0, 0}, F_asinh = {cr_asinh, INC, ALL},
  F_asinpi = {cr_asinpi, INC, -1, 1, 0, 0}, F_atan = {cr_atan, INC, ALL}, F_atanh = {cr_atanh, INC, -1, 1, 1, 1},
  F_atanpi = {cr_atanpi, INC, ALL}, F_cbrt = {cr_cbrt, INC, ALL}, F_cos = {cr_cos, COS, ALL},
  F_cosh = {cr_cosh, COSH, ALL}, F_cospi = {cr_cospi, COSPI, ALL}, F_erf = {cr_erf, INC, ALL},
  F_erfc = {cr_erfc, DEC, ALL}, F_exp = {cr_exp, INC, ALL}, F_exp10 = {cr_exp10, INC, ALL},
  F_exp2 = {cr_exp2, INC, ALL}, F_expm1 = {cr_expm1, INC, ALL}, F_log = {cr_log, INC, 0, INFINITY, 1, 0},
  F_log10 = {cr_log10, INC, 0, INFINITY, 1, 0}, F_log1p = {cr_log1p, INC, -1, INFINITY, 1, 0},
  F_log2 = {cr_log2, INC, 0, INFINITY, 1, 0}, F_rsqrt = {cr_rsqrt, DEC, 0, INFINITY, 1, 0},
  F_sin = {cr_sin, SIN, ALL}, F_sinh = {cr_sinh, INC, ALL}, F_sinpi = {cr_sinpi, SINPI, ALL},
  F_tan = {cr_tan, TAN, ALL}, F_tanh = {cr_tanh, INC, ALL},
  F_tanpi = {cr_tanpi, TANPI, ALL};

#define IVAL_F(f)                                                                    \
  void ival_##f(const double *lo, const double *hi, double *ylo, double *yhi, size_t n) \
  { run(&F_##f, lo, hi, ylo, yhi, n); }
#define IVAL_FX(f)   /* their own code, below */
#include "ival-list.h"

/* ---- tgamma (2026-10-02) ----
   The domain is the reals but the poles 0, -1, -2, ... Gamma is decreasing then increasing on (0, inf), its minimum
   at x0 = 1.4616...; on each segment (-n-1, -n) it has one extremum x_n, a minimum where Gamma > 0 (n odd) and a
   maximum where it is negative (n even), and runs to that sign's infinity at both poles. So:
     - a single pole is empty; a pole strictly inside the interval gives [-inf, +inf] (the signs differ across it);
     - otherwise the interval lies in one segment's closure (or in [0, inf]): the bounds are the ends' values rounded
       down and up, a pole at an end giving its side's infinity, and the extremum's rounded value replaces one bound
       when x0 or x_n lies inside. None of these points is a binary64 value: tgamma-table.h (gen-tgamma.py) gives each
       as the two binary64 values around it, and its value rounded down and up, for the segments with n < TG_NT;
       beyond, every binary64 value has |Gamma| < 2^-1074, so the extremum's value rounds to +0 (down, Gamma > 0) or
       -0 (up, Gamma < 0), which bounds the interval whether or not it holds the extremum. (First written as "the
       ends' rounded values are already the extremum's": not when both ends are poles, [k - 1, k] beyond 2^50,
       caught by the check.) */
#include "tgamma-table.h"
/* one bound in the current mode; 0 for the empty interval */
static int tg1(double a, double b, int p, int up, double *r)
{
  (void)p;
  if (!(a <= b)) return 0;                                    /* empty, NaN ends too */
  if (a == 0) a = 0.0;
  if (b == 0) b = 0.0;
  if (a == b && a <= 0 && floor(a) == a) return 0;            /* only a pole (or -inf) */
  /* a pole strictly inside: 0 when a < 0 < b; else the largest integer below b, if above a (beyond 2^53 every
     binary64 is an integer and its neighbours are 2 or more apart, so a < b alone says it) */
  int inside;
  if (b > 0) inside = a < 0;
  else if (fabs(b) >= 0x1p53) inside = a < b;
  else inside = (floor(b) == b ? b - 1 : floor(b)) > a;
  if (inside) { *r = up ? INFINITY : -INFINITY; return 1; }
  if (a >= 0) {                                               /* [0, inf]: Gamma(+0) = +inf, the limit from the right */
    double fa = cr_tgamma(a), fb = cr_tgamma(b);
    if (up) { *r = greater(fa, fb); return 1; }
    *r = a <= TG_X0LO && b >= TG_X0HI ? TG_MIN_RD : lesser(fa, fb);
    return 1;
  }
  double nb = floor(b) == b ? -b : -ceil(b);                  /* within [-n-1, -n]; n, and |b| < 2^53 here */
  double s = fmod(nb, 2) == 1 ? 1 : -1;                       /* Gamma's sign on the segment */
  int pa = a == -nb - 1, pb = b == -nb;                       /* a pole at an end */
  double fa = pa ? s * INFINITY : cr_tgamma(a), fb = pb ? s * INFINITY : cr_tgamma(b);
  double l = lesser(fa, fb), u = greater(fa, fb);
  if (nb < TG_NT) {
    const double *e = TG_EXT[(int)nb];
    if (a <= e[0] && b >= e[1]) {                             /* the extremum inside */
      if (s > 0) l = e[2]; else u = e[3];
#ifdef IVAL_PLANT_TG   /* the check's control: segment 3's tabulated minimum an ulp low (a neighbour one off cannot
                          show: Gamma is flat there, and an ulp from x_n it rounds as the extremum does) */
      if (nb == 3) l = nextafter(l, -INFINITY);
#endif
    }
  } else if (s > 0) l = lesser(l, 0.0);                       /* beyond the table every value inside rounds as the */
  else u = greater(u, -0.0);                                  /* extremum's, +0 down or -0 up: also when both ends are poles */
  *r = up ? u : l;
  return 1;
}
/* In blocks, as run: every lower bound rounding down, then every upper bound rounding up (p: pown's powers) */
typedef int (*bound1)(double a, double b, int p, int up, double *r);
static void run1(bound1 g, const double *lo, const double *hi, const int *p, double *ylo, double *yhi, size_t n)
{
  fenv_t env;
  fegetenv(&env);
  flush_off();   /* a -ffast-math caller's flush modes would break the bounds; fesetenv gives them back */
  double L[BLK], U[BLK];
  unsigned char E[BLK];
  for (size_t i0 = 0; i0 < n; i0 += BLK) {
    size_t m = n - i0 < BLK ? n - i0 : BLK;
    fesetround(FE_DOWNWARD);
    for (size_t k = 0; k < m; k++) E[k] = (unsigned char)!g(lo[i0 + k], hi[i0 + k], p ? p[i0 + k] : 0, 0, &L[k]);
    fesetround(FE_UPWARD);
    for (size_t k = 0; k < m; k++) if (!E[k]) g(lo[i0 + k], hi[i0 + k], p ? p[i0 + k] : 0, 1, &U[k]);
    for (size_t k = 0; k < m; k++) { ylo[i0 + k] = E[k] ? NAN : L[k]; yhi[i0 + k] = E[k] ? NAN : U[k]; }
  }
  fesetenv(&env);
}
typedef int (*bound2)(double a, double b, double c, double d, int up, double *r);
static void run2(bound2 g, const double *x0, const double *x1, const double *y0, const double *y1, double *zlo,
                 double *zhi, size_t n)
{
  fenv_t env;
  fegetenv(&env);
  flush_off();
  double L[BLK], U[BLK];
  unsigned char E[BLK];
  for (size_t i0 = 0; i0 < n; i0 += BLK) {
    size_t m = n - i0 < BLK ? n - i0 : BLK;
    fesetround(FE_DOWNWARD);
    for (size_t k = 0; k < m; k++) E[k] = (unsigned char)!g(x0[i0 + k], x1[i0 + k], y0[i0 + k], y1[i0 + k], 0, &L[k]);
    fesetround(FE_UPWARD);
    for (size_t k = 0; k < m; k++) if (!E[k]) g(x0[i0 + k], x1[i0 + k], y0[i0 + k], y1[i0 + k], 1, &U[k]);
    for (size_t k = 0; k < m; k++) { zlo[i0 + k] = E[k] ? NAN : L[k]; zhi[i0 + k] = E[k] ? NAN : U[k]; }
  }
  fesetenv(&env);
}
void ival_tgamma(const double *lo, const double *hi, double *ylo, double *yhi, size_t n)
{
  run1(tg1, lo, hi, NULL, ylo, yhi, n);
}

/* ---- two arguments ---- */

/* the least and greatest |x| over [a, b], a <= b */
static void mag(double a, double b, double *least, double *most)
{
  double u = fabs(a), v = fabs(b);
  *most = u > v ? u : v;
  *least = a <= 0 && 0 <= b ? 0 : u < v ? u : v;
}

/* hypot depends on |x| and |y| only, and grows with each: its least value
   over the box is at the least magnitudes, its greatest at the greatest */
static int hy1(double a, double b, double c, double d, int up, double *r)
{
  if (!(a <= b) || !(c <= d)) return 0;   /* NaN ends too */
  double xl, xm, yl, ym;
  mag(a, b, &xl, &xm);
  mag(c, d, &yl, &ym);
  *r = up ? cr_hypot(xm, ym) : cr_hypot(xl, yl);
  return 1;
}
void ival_hypot(const double *xlo, const double *xhi, const double *ylo, const double *yhi, double *zlo, double *zhi,
                size_t n)
{
  run2(hy1, xlo, xhi, ylo, yhi, zlo, zhi, n);
}

/* either zero as +0: sets have one zero (IEEE 1788), and atan2(+0, x < 0)
   is pi, its value on the negative x-axis */
static double z0(double v) { return v == 0 ? 0.0 : v; }

/* atan2(y, x) over Y x X, minus the origin (where it is undefined). Its
   range is (-pi, pi], with the cut on the negative x-axis (pi there).
   - The box holds points with x < 0 on both sides of the axis (y < 0 and
     y = 0): the values come arbitrarily close to -pi and reach pi, so the
     whole range [-pi rounded down, pi rounded up].
   - Otherwise atan2 is the angle of a point, continuous over the box
     without the origin, and the angle of a box seen from outside it (or
     from a point on its edge) is least and greatest at its corners: those,
     but the origin. */
static int at1(double c, double d, double a, double b, int up, double *r)   /* Y = [c, d], X = [a, b] */
{
  if (!(a <= b) || !(c <= d)) return 0;
  a = z0(a); b = z0(b); c = z0(c); d = z0(d);
  if (a < 0 && c < 0 && d >= 0) {   /* across the cut: -pi rounded down, pi rounded up */
    *r = up ? cr_atan2(0.0, -1) : cr_atan2(-0.0, -1);
    return 1;
  }
  double xs[2] = {a, b}, ys[2] = {c, d}, best = up ? -INFINITY : INFINITY;
  int any = 0;
  for (int j = 0; j < 2; j++)
    for (int k = 0; k < 2; k++) {
      if (xs[j] == 0 && ys[k] == 0) continue;   /* the origin */
      double v = cr_atan2(ys[k], xs[j]);
      any = 1;
      best = up ? greater(v, best) : lesser(v, best);
    }
  if (!any) return 0;   /* the box is the origin alone: empty */
  *r = best;
  return 1;
}
void ival_atan2(const double *ylo, const double *yhi, const double *xlo, const double *xhi, double *zlo, double *zhi,
                size_t n)
{
  run2(at1, ylo, yhi, xlo, xhi, zlo, zhi, n);
}

/* pow(x, y) = e^(y log x), with IEEE 1788's domain: x > 0, and x = 0 with
   y > 0 (negative x is pown's and rootn's business, not pow's). For fixed
   y it is monotone in x, and for fixed x monotone in y, so over a box it is
   least and greatest at corners. At the domain's edge C's pow gives the
   limits from inside it: pow(+0, y) is +0, 1 or +inf as y > 0, = 0 or < 0,
   and at infinities likewise. The one exception: x = 0 alone, where only
   y > 0 is in the domain (the box is empty otherwise, and [0, 0] if not). */
static int pw1(double a, double b, double c, double d, int up, double *r)
{
  if (!(a <= b) || !(c <= d) || b < 0) return 0;
  if (a < 0) a = 0;   /* x within its domain */
  a = z0(a); b = z0(b); c = z0(c); d = z0(d);
  if (b == 0) {   /* x = 0 alone */
    if (!(d > 0)) return 0;
    *r = 0;
    return 1;
  }
  double xs[2] = {a, b}, ys[2] = {c, d}, best = up ? -INFINITY : INFINITY;
  for (int j = 0; j < 2; j++)
    for (int k = 0; k < 2; k++) {
      double v = cr_pow(xs[j], ys[k]);
      best = up ? greater(v, best) : lesser(v, best);
    }
  *r = best;
  return 1;
}
void ival_pow(const double *xlo, const double *xhi, const double *ylo, const double *yhi, double *zlo, double *zhi,
              size_t n)
{
  run2(pw1, xlo, xhi, ylo, yhi, zlo, zhi, n);
}


/* pown(x, p), p an int: IEEE 1788's x^p for every real x, each bound CORE-MATH's pow at an end (p as a double is
   exact), rounded down or up. x^0 is 1, 0^0 included. For p > 0 odd, x^p increases; for p > 0 even it falls to 0 and
   rises, like sqr. For p < 0 there is a pole at 0: x = [0, 0] alone is empty; odd p decreases on each side, so an
   interval with 0 at an end reaches the infinity on that side, and 0 strictly inside gives the whole line; even p is
   positive and falls with |x|, so an interval reaching 0 goes up to +inf from its larger magnitude. */
static double pwc(double x, int p) { return cr_pow(x, (double)p); }   /* in the current mode */
static int pn1(double a, double b, int p, int up, double *r)
{
  if (!(a <= b) || a == INFINITY || b == -INFINITY) return 0;
  a = z0(a); b = z0(b);
  double v;
  int odd = p & 1;
  if (p == 0) v = 1;
  else if (p > 0) {
    if (odd || a >= 0) v = pwc(up ? b : a, p);
    else if (b <= 0) v = pwc(up ? a : b, p);
#if IVAL_PLANT_ARITH == 21   /* 21: an even power straddling 0 taken from its ends alone */
    else v = up ? greater(pwc(a, p), pwc(b, p)) : lesser(pwc(a, p), pwc(b, p));
#else
    else v = up ? greater(pwc(a, p), pwc(b, p)) : 0;
#endif
  } else {
    if (a == 0 && b == 0) return 0;
    if (a >= 0) v = up ? (a == 0 ? INFINITY : pwc(a, p)) : pwc(b, p);
    else if (b <= 0) {
      if (odd) v = up ? pwc(a, p) : (b == 0 ? -INFINITY : pwc(b, p));
      else v = up ? (b == 0 ? INFINITY : pwc(b, p)) : pwc(a, p);
    } else if (odd) v = up ? INFINITY : -INFINITY;
#if IVAL_PLANT_ARITH == 20   /* 20: a negative even power straddling 0 from the smaller magnitude */
    else v = up ? INFINITY : pwc(-a < b ? a : b, p);
#else
    else v = up ? INFINITY : pwc(-a > b ? a : b, p);
#endif
  }
  *r = z0(v);
  return 1;
}
void ival_pown(const double *lo, const double *hi, const int *p, double *ylo, double *yhi, size_t n)
{
  run1(pn1, lo, hi, p, ylo, yhi, n);
}

/* ---- the accurate mode (2026-10-09): ival_acc_f, each end within one ulp of the tightest (IEEE 1788's
   "accurate"), at vector speed. crmvec's four-lane functions (its libmvec.so.1, named by IVAL_CRMVEC, loaded at first
   use) are correctly rounded to nearest, so a value moved one ulp outward bounds the exact one, and lies at most one
   ulp outside the tightest bound (which is that value or its neighbour). Then each end is held to the function's
   range where that is exact (exp >= 0, |tanh| <= 1, ...). The monotone functions and cosh go four intervals at a
   time; sin, cos and tan too when narrower than 2.5 (one piece: the slopes at the ends from crmvec's cos and sin, as
   pieces() does, a zero slope left to the tight path); wide ones, the pi functions and tgamma take the tight path,
   which is accurate too. Without crmvec, or without AVX2 and FMA, the whole call takes it. ---- */
#if defined(__x86_64__)
#include <dlfcn.h>
#include <immintrin.h>
#include <sched.h>
typedef __m256d (*v4)(__m256d);
static struct { const char *name; v4 v; double rmin, rmax; } VT[] = {
#define IVAL_F(f) { #f, 0, -INFINITY, INFINITY },
#include "ival-list.h"
};
enum { NVT = sizeof VT / sizeof VT[0] };
static v4 v_sin, v_cos;
/* -1 not tried, 2 being loaded, 0 unavailable, 1 loaded. One thread claims the load and the others wait for it (a
   dlopen and 34 lookups), so two first calls at once cannot both write VT, and every call after the load gives the
   same bits: a thread that went the tight way meanwhile would not */
static int vt_state = -1;
static const struct { const char *name; double rmin, rmax; } RANGE[] = {
  { "exp", 0, INFINITY }, { "exp2", 0, INFINITY }, { "exp10", 0, INFINITY }, { "expm1", -1, INFINITY },
  { "sqrt", 0, INFINITY }, { "rsqrt", 0, INFINITY }, { "erf", -1, 1 }, { "erfc", 0, 2 }, { "tanh", -1, 1 },
  { "cosh", 1, INFINITY }, { "acosh", 0, INFINITY }, { "acos", 0, INFINITY }, { "acospi", 0, 1 },
  { "asinpi", -0.5, 0.5 }, { "atanpi", -0.5, 0.5 }, { "sin", -1, 1 }, { "cos", -1, 1 } };
static int vt_load(void)
{
  int st = __atomic_load_n(&vt_state, __ATOMIC_ACQUIRE);
  if (st == 0 || st == 1) return st;
  int expect = -1;
#if IVAL_PLANT_ARITH == 34   /* 34: no claim, every first caller loads and writes VT, as the first version did */
  if (0) {
#else
  if (!__atomic_compare_exchange_n(&vt_state, &expect, 2, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
#endif
    while ((st = __atomic_load_n(&vt_state, __ATOMIC_ACQUIRE)) == 2) sched_yield();   /* another thread is loading */
    return st;
  }
  st = 0;
  const char *p = getenv("IVAL_CRMVEC");
  __builtin_cpu_init();
  void *h = p && *p && __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") ? dlopen(p, RTLD_NOW | RTLD_LOCAL) : NULL;
  if (h) {
    char sym[64];
    for (int i = 0; i < NVT; i++) {
      snprintf(sym, sizeof sym, "_ZGVdN4v_%s", VT[i].name);
      VT[i].v = (v4)dlsym(h, sym);
      for (unsigned r = 0; r < sizeof RANGE / sizeof RANGE[0]; r++)
        if (!strcmp(RANGE[r].name, VT[i].name)) { VT[i].rmin = RANGE[r].rmin; VT[i].rmax = RANGE[r].rmax; }
    }
    v_sin = (v4)dlsym(h, "_ZGVdN4v_sin"); v_cos = (v4)dlsym(h, "_ZGVdN4v_cos");
    st = v_sin && v_cos;
  }
  __atomic_store_n(&vt_state, st, __ATOMIC_RELEASE);
  return st;
}
#define TGT __attribute__((target("avx2,fma")))
TGT static inline __m256d vstep(__m256d x, int up)   /* the next double down (up = 0) or up, as nextafter */
{
  const __m256d z = _mm256_setzero_pd(), sg = _mm256_set1_pd(-0.0);
  if (up) x = _mm256_xor_pd(x, sg);   /* up is minus down of minus */
  __m256i u = _mm256_castpd_si256(x), one = _mm256_set1_epi64x(1);
  __m256d pos = _mm256_cmp_pd(x, z, _CMP_GT_OQ), neg = _mm256_cmp_pd(x, z, _CMP_LT_OQ), isz = _mm256_cmp_pd(x, z, _CMP_EQ_OQ);
  __m256d ninf = _mm256_cmp_pd(x, _mm256_set1_pd(-INFINITY), _CMP_EQ_OQ);
  __m256d r = _mm256_blendv_pd(x, _mm256_castsi256_pd(_mm256_sub_epi64(u, one)), pos);
  r = _mm256_blendv_pd(r, _mm256_castsi256_pd(_mm256_add_epi64(u, one)), _mm256_andnot_pd(ninf, neg));
  r = _mm256_blendv_pd(r, _mm256_set1_pd(-0x1p-1074), isz);
  return up ? _mm256_xor_pd(r, sg) : r;
}
TGT static inline __m256d vsign(__m256d x)   /* -1, 0 or 1 */
{
  const __m256d z = _mm256_setzero_pd();
  return _mm256_sub_pd(_mm256_and_pd(_mm256_cmp_pd(x, z, _CMP_GT_OQ), _mm256_set1_pd(1)),
                       _mm256_and_pd(_mm256_cmp_pd(x, z, _CMP_LT_OQ), _mm256_set1_pd(1)));
}
/* four intervals at a time; lanes marked in *fb are for the tight path */
TGT static void acc4(const fn *F, int idx, const double *lo, const double *hi, double *yl, double *yh, int *fb)
{
  v4 g = VT[idx].v;
  __m256d a = _mm256_loadu_pd(lo), b = _mm256_loadu_pd(hi), l, u;
  const __m256d dlo = _mm256_set1_pd(F->dlo), dhi = _mm256_set1_pd(F->dhi), z = _mm256_setzero_pd();
  /* the domain: an empty lane (NaN ends, lo > hi, nothing left) goes to the tight path, which says so */
  __m256d bad = _mm256_cmp_pd(a, b, _CMP_NLE_UQ);
  a = _mm256_max_pd(a, dlo); b = _mm256_min_pd(b, dhi);
  bad = _mm256_or_pd(bad, _mm256_cmp_pd(a, b, _CMP_GT_OQ));
  if (F->lo_open) bad = _mm256_or_pd(bad, _mm256_cmp_pd(b, dlo, _CMP_LE_OQ));
  if (F->hi_open) bad = _mm256_or_pd(bad, _mm256_cmp_pd(a, dhi, _CMP_GE_OQ));
  a = _mm256_add_pd(a, z);   /* -0 to +0 at a domain's end (rsqrt(+0) = +inf); the mode is to nearest */
  a = _mm256_blendv_pd(a, z, bad); b = _mm256_blendv_pd(b, z, bad);   /* harmless arguments in bad lanes */
  if (F->kind == INC || F->kind == DEC) {
    __m256d fa = g(a), fb2 = g(b);
    l = vstep(F->kind == INC ? fa : fb2, 0); u = vstep(F->kind == INC ? fb2 : fa, 1);
  } else if (F->kind == COSH) {
    __m256d sg = _mm256_set1_pd(-0.0), ma = _mm256_andnot_pd(sg, a), mb = _mm256_andnot_pd(sg, b);
    __m256d in0 = _mm256_and_pd(_mm256_cmp_pd(a, z, _CMP_LE_OQ), _mm256_cmp_pd(b, z, _CMP_GE_OQ));
    l = _mm256_blendv_pd(vstep(g(_mm256_min_pd(ma, mb)), 0), _mm256_set1_pd(1), in0);
    u = vstep(g(_mm256_max_pd(ma, mb)), 1);
  } else {   /* SIN, COS, TAN, narrower than 2.5: one piece */
    bad = _mm256_or_pd(bad, _mm256_cmp_pd(_mm256_sub_pd(b, a), _mm256_set1_pd(2.5), _CMP_NLT_UQ));
    __m256d ca = v_cos(a), cb = v_cos(b), fa = g(a), fb2 = g(b);
    if (F->kind == TAN) {
      __m256d pole = _mm256_cmp_pd(_mm256_mul_pd(vsign(ca), vsign(cb)), z, _CMP_LT_OQ);
      l = _mm256_blendv_pd(vstep(fa, 0), _mm256_set1_pd(-INFINITY), pole);
      u = _mm256_blendv_pd(vstep(fb2, 1), _mm256_set1_pd(INFINITY), pole);
    } else {
      __m256d du, dv;
      if (F->kind == SIN) { du = vsign(ca); dv = vsign(cb); }
      else { du = vsign(_mm256_sub_pd(z, v_sin(a))); dv = vsign(_mm256_sub_pd(z, v_sin(b))); }
      bad = _mm256_or_pd(bad, _mm256_or_pd(_mm256_cmp_pd(du, z, _CMP_EQ_OQ), _mm256_cmp_pd(dv, z, _CMP_EQ_OQ)));
      __m256d inc = _mm256_cmp_pd(du, z, _CMP_GT_OQ), change = _mm256_cmp_pd(du, dv, _CMP_NEQ_OQ);
      __m256d lo_m = _mm256_blendv_pd(fb2, fa, inc), hi_m = _mm256_blendv_pd(fa, fb2, inc);   /* monotone: by the slope */
      __m256d mn = _mm256_min_pd(fa, fb2), mx = _mm256_max_pd(fa, fb2);
      /* a change from rising (du > 0): a maximum inside, 1; from falling: a minimum, -1 */
      l = _mm256_blendv_pd(vstep(lo_m, 0), _mm256_blendv_pd(_mm256_set1_pd(-1), vstep(mn, 0), inc), change);
      u = _mm256_blendv_pd(vstep(hi_m, 1), _mm256_blendv_pd(vstep(mx, 1), _mm256_set1_pd(1), inc), change);
    }
  }
  l = _mm256_max_pd(l, _mm256_set1_pd(VT[idx].rmin)); u = _mm256_min_pd(u, _mm256_set1_pd(VT[idx].rmax));
  _mm256_storeu_pd(yl, l); _mm256_storeu_pd(yh, u);
  *fb = _mm256_movemask_pd(bad);
}
static void acc_run(const fn *F, int idx, const double *lo, const double *hi, double *ylo, double *yhi, size_t n)
{
  int vec = (F->kind == INC || F->kind == DEC || F->kind == COSH || F->kind == SIN || F->kind == COS || F->kind == TAN);
  if (!vec || !vt_load() || !VT[idx].v) { run(F, lo, hi, ylo, yhi, n); return; }
  /* the lanes for the tight path are gathered, up to 256, and run together: one environment and three mode switches
     for all of them, not for each */
  enum { G = 256 };
  double ga[G], gb[G], gl[G], gu[G];
  size_t gi[G], ng = 0;
  fenv_t env;
  fegetenv(&env);
  fesetround(FE_TONEAREST);
  flush_off();
  size_t i = 0;
  for (; i + 4 <= n; i += 4) {
    double a[4], b[4], l[4], u[4];
    int fb;
    memcpy(a, lo + i, sizeof a); memcpy(b, hi + i, sizeof b);   /* ylo may be lo */
    acc4(F, idx, a, b, l, u, &fb);
    for (int k = 0; k < 4; k++) {
      ylo[i + k] = l[k]; yhi[i + k] = u[k];
      if (fb >> k & 1) { ga[ng] = a[k]; gb[ng] = b[k]; gi[ng++] = i + k; }
    }
    if (ng > G - 4) {
      fesetenv(&env);
      run(F, ga, gb, gl, gu, ng);
      for (size_t k = 0; k < ng; k++) { ylo[gi[k]] = gl[k]; yhi[gi[k]] = gu[k]; }
      ng = 0;
      fesetround(FE_TONEAREST);
      flush_off();
    }
  }
  fesetenv(&env);
  if (ng) {
    run(F, ga, gb, gl, gu, ng);
    for (size_t k = 0; k < ng; k++) { ylo[gi[k]] = gl[k]; yhi[gi[k]] = gu[k]; }
  }
  if (i < n) run(F, lo + i, hi + i, ylo + i, yhi + i, n - i);
}
#define IVAL_F(f)                                                                                \
  void ival_acc_##f(const double *lo, const double *hi, double *ylo, double *yhi, size_t n)          \
  {                                                                                              \
    int idx = 0;                                                                                 \
    while (strcmp(VT[idx].name, #f)) idx++;                                                      \
    acc_run(&F_##f, idx, lo, hi, ylo, yhi, n);                                                   \
  }
#else
#define IVAL_F(f)                                                                                \
  void ival_acc_##f(const double *lo, const double *hi, double *ylo, double *yhi, size_t n)          \
  { run(&F_##f, lo, hi, ylo, yhi, n); }
#endif
/* the functions with code of their own (tgamma): their tight results */
#define IVAL_FX(f)                                                                               \
  void ival_acc_##f(const double *lo, const double *hi, double *ylo, double *yhi, size_t n)          \
  { ival_##f(lo, hi, ylo, yhi, n); }
#include "ival-list.h"
