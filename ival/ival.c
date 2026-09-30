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
#include "ival.h"

double cr_acos(double), cr_acosh(double), cr_acospi(double), cr_asin(double), cr_asinh(double), cr_asinpi(double),
  cr_atan(double), cr_atanh(double), cr_atanpi(double), cr_cbrt(double), cr_cos(double), cr_cosh(double),
  cr_cospi(double), cr_erf(double), cr_erfc(double), cr_exp(double), cr_exp10(double), cr_exp2(double),
  cr_expm1(double), cr_log(double), cr_log10(double), cr_log1p(double), cr_log2(double), cr_rsqrt(double),
  cr_sin(double), cr_sinh(double), cr_sinpi(double), cr_tan(double), cr_tanh(double), cr_tanpi(double),
  cr_hypot(double, double);
static double cr_sqrt(double x) { return sqrt(x); }   /* correctly rounded in every mode (IEEE 754) */

enum { INC, DEC, COSH, SIN, COS, TAN, SINPI, COSPI, TANPI };
/* a function, its shape, and its domain [dlo, dhi] (open at an end where
   the function has a pole or no limit: log at 0, atanh at +-1) */
typedef struct { double (*f)(double); int kind; double dlo, dhi; int lo_open, hi_open; } fn;

static int sgn(double v) { return (v > 0) - (v < 0); }

/* the derivative's sign just right (right = 1) or just left of x, for sin
   and cos: where it is zero (cos at 0), the value there says which way */
static int slope(int kind, double x, int right)
{
  /* the sign rounding to nearest: rounding down or up could make a tiny
     nonzero result (sin near 0) zero */
  int m = fegetround();
  fesetround(FE_TONEAREST);
  int d = kind == SIN ? sgn(cr_cos(x)) : -sgn(cr_sin(x));
  fesetround(m);
  if (d) return d;
  double v = kind == SIN ? cr_sin(x) : cr_cos(x);   /* 1 or -1, exactly */
  return (v > 0) == right ? -1 : 1;
}

static double lesser(double a, double b) { return a < b ? a : b; }
static double greater(double a, double b) { return a > b ? a : b; }

/* the lower (up = 0, rounding down) or upper (up = 1, rounding up) bound of
   sin, cos or tan over the piece [u, v]: shorter than pi, or (sparse
   neighbours) shorter than 2 pi and longer than pi */
static double piece(const fn *F, double u, double v, int up)
{
  if (u == v) return F->f(u);   /* a point: nothing inside */
  int longer = v - u > 3.2;   /* exact here: 4 (neighbours beyond 2^54), or at most 2.5 and a hair */
  if (F->kind == TAN) {
    int m = fegetround();
    fesetround(FE_TONEAREST);
    int change = sgn(cr_cos(u)) * sgn(cr_cos(v)) < 0;
    fesetround(m);
    if (longer || change) return up ? INFINITY : -INFINITY;   /* a pole inside */
    return F->f(up ? v : u);
  }
  int du = slope(F->kind, u, 1), dv = slope(F->kind, v, 0);
  /* the extrema inside: at most one (shorter than pi), or one or two
     (longer): an odd number when the slope changes sign */
  int change = du != dv;
  if (longer && !change) return up ? 1 : -1;   /* two: a maximum and a minimum */
  if (!change) return up ? greater(F->f(u), F->f(v)) : lesser(F->f(u), F->f(v));   /* monotone */
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

/* one bound over [a, b], inside the domain, a <= b, in the current mode */
static double bound(const fn *F, double a, double b, int up)
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
    double w = b - a;
    if (!(w < 8)) {   /* a whole period inside (or an infinite end) */
      if (F->kind == TAN) return up ? INFINITY : -INFINITY;
      return up ? 1 : -1;
    }
    int k = (int)ceil(w / 2.5);   /* pieces of 2.5 and a hair, under pi, unless neighbours are further apart */
    if (k < 1) k = 1;
    double r = 0, p = a;
    for (int i = 1; i <= k; i++) {
      double q = i == k ? b : a + w * i / k;
      if (q < p) q = p;
      double v = piece(F, p, q, up);
      r = i == 1 ? v : up ? greater(v, r) : lesser(v, r);
      p = q;
    }
    return r;
  }
  }
}

static void run(const fn *F, const double *lo, const double *hi, double *ylo, double *yhi, size_t n)
{
  fenv_t env;
  fegetenv(&env);
  for (size_t i = 0; i < n; i++) {
    double a = lo[i], b = hi[i];
    /* the intersection with the domain; empty if there is none */
    if (a < F->dlo) a = F->dlo;
    if (b > F->dhi) b = F->dhi;
    int empty = !(a <= b) || (F->lo_open && b <= F->dlo) || (F->hi_open && a >= F->dhi);
    if (!empty && a == b && F->kind == TANPI && cr_cospi(a) == 0) empty = 1;   /* only a pole */
    if (empty) { ylo[i] = yhi[i] = NAN; continue; }
    if (a == 0) a = 0.0;   /* +0: a domain ending at 0 starts at +0 (rsqrt(+0) = +inf) */
    fesetround(FE_DOWNWARD);
    double l = bound(F, a, b, 0);
    fesetround(FE_UPWARD);
    double u = bound(F, a, b, 1);
    ylo[i] = l;
    yhi[i] = u;
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
  F_sqrt = {cr_sqrt, INC, 0, INFINITY, 0, 0}, F_tan = {cr_tan, TAN, ALL}, F_tanh = {cr_tanh, INC, ALL},
  F_tanpi = {cr_tanpi, TANPI, ALL};

#define IVAL_F(f)                                                                    \
  void ival_##f(const double *lo, const double *hi, double *ylo, double *yhi, size_t n) \
  { run(&F_##f, lo, hi, ylo, yhi, n); }
#include "ival-list.h"

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
void ival_hypot(const double *xlo, const double *xhi, const double *ylo, const double *yhi, double *zlo, double *zhi,
                size_t n)
{
  fenv_t env;
  fegetenv(&env);
  for (size_t i = 0; i < n; i++) {
    double a = xlo[i], b = xhi[i], c = ylo[i], d = yhi[i];
    if (!(a <= b) || !(c <= d)) { zlo[i] = zhi[i] = NAN; continue; }   /* NaN ends too */
    double xl, xm, yl, ym;
    mag(a, b, &xl, &xm);
    mag(c, d, &yl, &ym);
    fesetround(FE_DOWNWARD);
    double l = cr_hypot(xl, yl);
    fesetround(FE_UPWARD);
    double u = cr_hypot(xm, ym);
    zlo[i] = l;
    zhi[i] = u;
  }
  fesetenv(&env);
}

