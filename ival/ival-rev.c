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

/* ---- sinRev, cosRev, tanRev: the tightest interval around {x in X : f(x) in C} for the periodic functions.
   With C' = C met with f's range: empty if C' is; X itself if C' is the whole range. Otherwise the lower end is the
   first point of the set at or after X's lower end, rounded down, and the upper end likewise from the right; each is
   found in two steps:
     - near: the first crossing of C's nearer end after xl (or before xh) is a known point, an inverse function's
       value plus a multiple of the period, computed in double-double to a few ulps (below 2^48; beyond, the doubles
       are at most a few hundred a period and are scanned from X's end directly);
     - exact: from a few ulps before it, the doubles are scanned with exact tests. A double d is in the set when f(d)
       rounded down is >= cl and rounded up is <= ch (CORE-MATH's f is correctly rounded both ways). The open gap
       between neighbours d < e holds a point of the set when f's range over it meets C': f's values at d and e, and
       its extrema inside (sin, cos) or poles (tan), whose number the slopes' signs at d and e give by parity (at most
       one in a gap shorter than half a period; at least one in a longer gap, and every value in a gap a period long).
   The lower end is the first d such that d or the gap after it holds a point: the point's value rounded down. ---- */
double cr_sin(double), cr_cos(double), cr_tan(double), cr_asin(double), cr_acos(double), cr_atan(double);
enum { TSIN, TCOS, TTAN };
static int tsgn(double v) { return (v > 0) - (v < 0); }
static void fval(int kind, double x, double *d, double *u) { both(kind == TSIN ? cr_sin : kind == TCOS ? cr_cos : cr_tan, x, d, u); }
static int tmember(int kind, double x, double cl, double ch)
{
  double d, u;
  fval(kind, x, &d, &u);
  return d >= cl && u <= ch;
}
/* the slope's sign just right (right = 1) or left of x, rounding to nearest (the sign of a correctly rounded value
   is exact; sin is 0 only at 0, where cos falls to the right and rises to the left) */
static int tslope(int kind, double x, int right)
{
  if (kind == TSIN) return tsgn(cr_cos(x));
  int s = -tsgn(cr_sin(x));
  return s ? s : right ? -1 : 1;
}
static const double PI_UP = 0x1.921fb54442d19p+1, TWOPI_UP = 0x1.921fb54442d19p+2;   /* pi and 2 pi rounded up */
/* the open gap (u, v), u and v neighbours, holds a point of the set */
static int gap_meets(int kind, double u, double v, double cl, double ch)
{
  double ud, uu, vd, vu, w = v - u;   /* exact: neighbours */
  fval(kind, u, &ud, &uu); fval(kind, v, &vd, &vu);
  if (kind == TTAN) {   /* increasing between poles, a pole where cos changes sign; period pi */
#if IVAL_PLANT_ARITH != 30   /* 30: long gaps' poles counted by parity alone, the first version's bug */
    if (w >= TWOPI_UP) return 1;   /* two periods: two poles at least, every value */
#endif
    int odd = tsgn(cr_cos(u)) != tsgn(cr_cos(v)), poles = w >= PI_UP ? (odd ? 1 : 2) : odd;
    if (poles >= 2) return 1;
    if (poles == 1) return ud < ch || vu > cl;   /* (f(u), inf) or (-inf, f(v)) meets C */
    return vu > cl && ud < ch;                   /* (f(u), f(v)) meets C */
  }
  if (w >= TWOPI_UP) return 1;   /* a whole period */
  int su = tslope(kind, u, 1), sv = tslope(kind, v, 0), odd = su != sv, ext = w >= PI_UP ? (odd ? 1 : 2) : odd;
  if (ext >= 2) return 1;        /* a maximum and a minimum: every value */
  if (ext == 1) return su > 0 ? (ud < ch || vd < ch) : (uu > cl || vu > cl);   /* up to 1, or down to -1 */
  return (uu > cl || vu > cl) && (ud < ch || vd < ch);   /* between f(u) and f(v), open */
}

/* double-double: a + b exactly as hi + lo */
static void dd_add(double a, double b, double *hi, double *lo)
{
  double s = a + b, bb = s - a;
  *hi = s; *lo = (a - (s - bb)) + (b - bb);
}
/* the crossing nearest x on the side dir (+1: the least t > x; -1: the greatest t < x) of base + k period, base and
   period double-doubles; to a few ulps (|x| < 2^48, so k fits and k period_hi is exact as a double-double by fma) */
static double crossing(double bh, double bl, double ph, double pl, double x, int dir)
{
  double k = dir > 0 ? ceil((x - bh) / ph) : floor((x - bh) / ph);
  for (int pass = 0; pass < 4; pass++) {   /* the k that is the first past x, checked in double-double */
    double th, tl, ph_k = k * ph, e = fma(k, ph, -ph_k);
    dd_add(ph_k, bh, &th, &tl);
    tl += e + k * pl + bl;
    dd_add(th, tl, &th, &tl);
    int past = dir > 0 ? (th > x || (th == x && tl > 0)) : (th < x || (th == x && tl < 0));
    double kb = k - dir;                    /* the one before must not be past x */
    double bh2, bl2, pk2 = kb * ph, e2 = fma(kb, ph, -pk2);
    dd_add(pk2, bh, &bh2, &bl2);
    bl2 += e2 + kb * pl + bl;
    dd_add(bh2, bl2, &bh2, &bl2);
    int before_past = dir > 0 ? (bh2 > x || (bh2 == x && bl2 > 0)) : (bh2 < x || (bh2 == x && bl2 < 0));
    if (!past) k += dir;
    else if (before_past) k -= dir;
    else return th;
  }
  return x;   /* not settled: scan from x itself */
}
static const double PI_H = 0x1.921fb54442d18p+1, PI_L = 0x1.1a62633145c07p-53;
/* where to start scanning toward the set from x (an end of X not in the set), dir +1 from the left, -1 the right */
static double start(int kind, double x, double cl, double ch, int dir)
{
  if (!(fabs(x) < 0x1p48)) return x;   /* few doubles a period: scan from x */
  double d, u, bh, bl, ph = 2 * PI_H, pl = 2 * PI_L;
  fval(kind, x, &d, &u);
  int above = d > ch;   /* f(x) above C, else below: the crossing of ch or of cl */
  if (kind == TTAN) {
    ph = PI_H; pl = PI_L;
    double y = dir > 0 ? cl : ch;   /* forward, tan enters C rising through cl (after a pole if above); back, through ch */
    if (isinf(y)) { bh = PI_H / 2; bl = PI_L / 2; }   /* only past a pole */
    else { fesetround(FE_TONEAREST); bh = cr_atan(y); bl = 0; }
  } else {
    /* sin rises through y at asin(y) + 2 k pi and falls at pi - asin(y); cos falls at acos(y), rises at -acos(y).
       Forward from above C it falls to ch, from below it rises to cl; backward, the other way */
    int falling = (dir > 0) == above;
    double y = above ? ch : cl;
    fesetround(FE_TONEAREST);
    if (kind == TSIN) {
      double a = cr_asin(y);
      if (falling) dd_add(PI_H, -a, &bh, &bl), bl += PI_L; else { bh = a; bl = 0; }
    } else {
      double a = cr_acos(y);
      bh = falling ? a : -a; bl = 0;
    }
  }
  double t = crossing(bh, bl, ph, pl, x, dir);
  for (int k = 0; k < 8; k++) t = nextafter(t, dir > 0 ? -INFINITY : INFINITY);   /* a few ulps short of it */
  return dir > 0 ? (t > x ? t : x) : (t < x ? t : x);
}
/* the lower (dir = +1) or upper end of the set within [xl, xh]; 0 if the set misses X */
static int tend(int kind, double xl, double xh, double cl, double ch, int dir, double *r)
{
  double x = dir > 0 ? xl : xh;
  if (isinf(x)) { *r = x; return 1; }   /* periodic: the set reaches every infinity X does */
  if (tmember(kind, x, cl, ch)) { *r = x; return 1; }
  double d = start(kind, x, cl, ch, dir);
  for (int steps = 0; steps < 100000; steps++) {
    if (dir > 0 ? d > xh : d < xl) return 0;
    if (d != x && tmember(kind, d, cl, ch)) { *r = d; return 1; }
    double e = nextafter(d, dir > 0 ? INFINITY : -INFINITY);
    if (dir > 0 ? e > xh : e < xl) return 0;   /* the gap past d is outside X */
#if IVAL_PLANT_ARITH == 29   /* 29: the gaps left out, only doubles counted */
    (void)gap_meets;
#else
    if (dir > 0 ? gap_meets(kind, d, e, cl, ch) : gap_meets(kind, e, d, cl, ch)) { *r = d; return 1; }
#endif
    d = e;
  }
  *r = x;   /* not reached in the checks: X's own end, a valid bound */
  return 1;
}
static void trev(int kind, const double *clo, const double *chi, const double *xlo, const double *xhi, double *zlo,
                 double *zhi, size_t n)
{
  ENTER
  for (size_t i = 0; i < n; i++) {
    double cl = clo[i], ch = chi[i], xl = xlo[i], xh = xhi[i], l, u;
    zlo[i] = zhi[i] = NAN;
    if (empty(cl, ch) || empty(xl, xh)) continue;
    if (kind != TTAN) {   /* C' = C met with [-1, 1] */
      if (ch < -1 || cl > 1) continue;
      if (cl <= -1 && ch >= 1) { zlo[i] = canon(xl); zhi[i] = canon(xh); continue; }
      if (cl < -1) cl = -1;
      if (ch > 1) ch = 1;
    } else if (cl == -INFINITY && ch == INFINITY) { zlo[i] = canon(xl); zhi[i] = canon(xh); continue; }
    if (!tend(kind, xl, xh, cl, ch, 1, &l)) continue;
    if (!tend(kind, l, xh, cl, ch, -1, &u)) continue;
    zlo[i] = canon(l); zhi[i] = canon(u);
  }
  LEAVE
}
void ival_sinrev(const double *clo, const double *chi, const double *xlo, const double *xhi, double *zlo, double *zhi, size_t n)
{ trev(TSIN, clo, chi, xlo, xhi, zlo, zhi, n); }
void ival_cosrev(const double *clo, const double *chi, const double *xlo, const double *xhi, double *zlo, double *zhi, size_t n)
{ trev(TCOS, clo, chi, xlo, xhi, zlo, zhi, n); }
void ival_tanrev(const double *clo, const double *chi, const double *xlo, const double *xhi, double *zlo, double *zhi, size_t n)
{ trev(TTAN, clo, chi, xlo, xhi, zlo, zhi, n); }
