/* ival-rev.c: IEEE 1788.1's reverse operations for sqr, abs, pown, cosh, sin, cos, tan and pow, and rootn (ival.h), 2026-10-09.

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

double cr_acosh(double), cr_pow(double, double), cr_log(double);

#define ENTER ival_env env; env_save(&env); set_round(FE_TONEAREST); flush_off();
#define LEAVE env_restore(&env);

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
  set_round(FE_DOWNWARD); *d = g(v);
  set_round(FE_UPWARD); *u = g(v);
  set_round(FE_TONEAREST);
}

/* ---- the root c^(1/p), c >= 0, p != 0, rounded down and up ---- */
static double pw(double y, int p, int up)
{
  double r0;
  if (p >= 2 && pown_dd(y, p, up, &r0)) return r0;   /* root2 runs to nearest, as pown_dd needs */
  set_round(up ? FE_UPWARD : FE_DOWNWARD);
  double r = cr_pow(y, (double)p);
  set_round(FE_TONEAREST);
  return r;
}
static double from(uint64_t b) { double y; memcpy(&y, &b, 8); return y; }
static uint64_t bits(double y) { uint64_t b; memcpy(&b, &y, 8); return b; }
/* y (>= 0) is at most the root: y^p <= c for p > 0, y^p >= c for p < 0 (y^p decreasing), decided exactly */
static int below(double y, double c, int p)
{
#if IVAL_PLANT_ARITH == 24   /* 24: the predicate on the nearest y^p */
  set_round(FE_TONEAREST); double r = cr_pow(y, (double)p); return p > 0 ? r <= c : r >= c;
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
#if IVAL_PLANT_ARITH == 33   /* 33: a negative root's upper end rounded down */
      h = d;
#endif
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
/* where to start scanning toward the set from x (an end of X not in the set), dir +1 from the left, -1 the right.
   *near says the crossing may lie within its estimate's error of x itself: asin, acos and atan come rounded to
   nearest, so the estimate can fall just short of x when the crossing is just past it, and crossing() then takes the
   next period's (IBEX's tests found it: cosRev of [sin 0.5, sin 1.5] in [0.5 - pi/2, pi - 1.6 - pi/2] came out
   empty). The scan then tries a few doubles from x first. */
static double start(int kind, double x, double cl, double ch, int dir, int *near)
{
  *near = 0;
  if (!(fabs(x) < 0x1p48)) return x;   /* few doubles a period: scan from x */
  double d, u, bh, bl, ph = 2 * PI_H, pl = 2 * PI_L;
  fval(kind, x, &d, &u);
#if IVAL_PLANT_ARITH == 44   /* 44: above C read from f(x) rounded down, which can equal ch when f(x) is above it */
  int above = d > ch;
#else
  int above = u > ch;   /* f(x) above C, else below (x is not in the set): exactly, as ch is a double */
#endif
  if (kind == TTAN) {
    ph = PI_H; pl = PI_L;
    double y = dir > 0 ? cl : ch;   /* forward, tan enters C rising through cl (after a pole if above); back, through ch */
    if (isinf(y)) { bh = PI_H / 2; bl = PI_L / 2; }   /* only past a pole */
    else { set_round(FE_TONEAREST); bh = cr_atan(y); bl = 0; }
  } else {
    /* sin rises through y at asin(y) + 2 k pi and falls at pi - asin(y); cos falls at acos(y), rises at -acos(y).
       Forward from above C it falls to ch, from below it rises to cl; backward, the other way */
    int falling = (dir > 0) == above;
    double y = above ? ch : cl;
    set_round(FE_TONEAREST);
    if (kind == TSIN) {
      double a = cr_asin(y);
      if (falling) dd_add(PI_H, -a, &bh, &bl), bl += PI_L; else { bh = a; bl = 0; }
    } else {
      double a = cr_acos(y);
      bh = falling ? a : -a; bl = 0;
    }
  }
  double t = crossing(bh, bl, ph, pl, x, dir);
  double tn = dir > 0 ? t - ph : t + ph, slack = 64 * (nextafter(fabs(t), INFINITY) - fabs(t));
#if IVAL_PLANT_ARITH != 43   /* 43: the crossing at x itself left to the estimate */
  *near = dir > 0 ? tn >= x - slack : tn <= x + slack;   /* the crossing one period nearer is about at x */
#endif
  for (int k = 0; k < 8; k++) t = nextafter(t, dir > 0 ? -INFINITY : INFINITY);   /* a few ulps short of it */
  return dir > 0 ? (t > x ? t : x) : (t < x ? t : x);
}
/* the lower (dir = +1) or upper end of the set within [xl, xh]; 0 if the set misses X */
static int tend(int kind, double xl, double xh, double cl, double ch, int dir, double *r)
{
  double x = dir > 0 ? xl : xh;
  if (isinf(x)) { *r = x; return 1; }   /* periodic: the set reaches every infinity X does */
  if (tmember(kind, x, cl, ch)) { *r = x; return 1; }
  int near;
  double far = start(kind, x, cl, ch, dir, &near), d = near ? x : far;
  for (int steps = 0; steps < 100000; steps++) {
    if (near && steps == 256) { near = 0; if (dir > 0 ? far > d : far < d) d = far; }   /* not at x after all */
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

/* ---- powRev1 and powRev2, pow's domain being x > 0, and x = 0 with y > 0 (where x^y is 0).
   powRev1(B, C, X) is the tightest interval around {x in X : x^y in C for some y in B}. x = 0 is in the set when 0 is
   in C and B holds a y > 0. For x > 0, x^y lies in C+ = C met with (0, inf), [c0, ch]: for a fixed y > 0 exactly when
   x is in [c0^(1/y), ch^(1/y)], for y < 0 in [ch^(1/y), c0^(1/y)], and for y = 0 (x^0 = 1) everywhere when 1 is in C.
   Over a range of y of one sign these intervals move continuously, so their union is an interval; each of its ends
   is c^(1/y) at one end of the range, chosen by whether c is above or below 1, since c^(1/y) is monotone in y on
   either side of 0. So the set is at most four pieces: y > 0, y < 0, y = 0, x = 0.
   powRev2(A, C, Y), around {y in Y : x^y in C for some x in A}, is the same with log_x(c) for c^(1/y): x > 1 puts y in
   [log_x c0, log_x ch], 0 < x < 1 in [log_x ch, log_x c0], x = 1 everywhere when 1 is in C, and x = 0 makes every
   y > 0 when 0 is in C.
   An end at an excluded point is a limit and the piece is open there: c0 = 0 or ch = inf, a range ending at y = 0 or
   x = 1 (where the power runs to 0 or inf), or at y = +-inf, x = 0 or x = inf (where c^(1/y) tends to 1 and log_x(c)
   to 0). Any other end is a real number R, c^(1/y) or log_x(c), found by search: whether a double t is at most R is
   decided exactly by CORE-MATH's pow, correctly rounded, against the double c. ---- */
static double pw2(double x, double y, int up)
{
  set_round(up ? FE_UPWARD : FE_DOWNWARD);
  double r = cr_pow(x, y);
  set_round(FE_TONEAREST);
  return r;
}
/* the doubles in order, as unsigned integers */
static uint64_t ord(double t) { uint64_t b = bits(t); return b >> 63 ? ~b : b | 1ULL << 63; }
static double unord(uint64_t k) { return from(k >> 63 ? k & ~(1ULL << 63) : ~k); }
/* t is at most R = c^(1/a) (root, t >= 0: t^a is increasing in t for a > 0, decreasing for a < 0) or R = log_a(c)
   (a^t is increasing for a > 1, decreasing for a < 1); c is a double, so x^y <= c exactly when x^y rounded up is */
static int atmost(int root, double t, double a, double c)
{
#if IVAL_PLANT_ARITH == 37   /* 37: the predicate on the nearest power */
  double r = root ? cr_pow(t, a) : cr_pow(a, t);
  return (root ? a > 0 : a > 1) ? r <= c : r >= c;
#endif
  if (root) return a > 0 ? pw2(t, a, 1) <= c : pw2(t, a, 0) >= c;
  return a > 1 ? pw2(a, t, 1) <= c : pw2(a, t, 0) >= c;
}
/* R rounded down and up, for a double c in (0, 1) or (1, inf) and a finite a != 0 (root) or a in (0, 1) or (1, inf):
   the last double at most R, by galloping from an estimate and bisecting over the doubles in order (+0 for a root,
   -inf for a logarithm, is at most R; +inf is not); R is that double exactly when the power at it is c both ways */
static void solve(int root, double a, double c, double *d, double *u)
{
  double est = root ? cr_pow(c, 1 / a) : cr_log(c) / cr_log(a);
  uint64_t lo = ord(root ? 0.0 : -INFINITY), hi = ord(INFINITY), b = ord(est), L, H, step = 1;
  if (b < lo) b = lo;
  if (b > hi) b = hi;
  if (atmost(root, unord(b), a, c)) {
    L = b;
    while (step < hi - b && atmost(root, unord(b + step), a, c)) { L = b + step; step *= 2; }
    H = step < hi - b ? b + step : hi;
  } else {
    H = b;
    while (step < b - lo && !atmost(root, unord(b - step), a, c)) { H = b - step; step *= 2; }
    L = step < b - lo ? b - step : lo;
  }
  while (H - L > 1) {
    uint64_t m = L + (H - L) / 2;
    if (atmost(root, unord(m), a, c)) L = m; else H = m;
  }
  double t = unord(L), x = root ? t : a, y = root ? a : t;
  *d = t;
  *u = pw2(x, y, 0) == c && pw2(x, y, 1) == c ? t : unord(L + 1);
#if IVAL_PLANT_ARITH == 39   /* 39: R rounded down for both */
  *u = t;
#endif
}
/* c^(1/y) (root) or log_x(c) at the end r of a range of y (or x) on one side of 0 (of 1), the positive (above 1)
   side when pos, rounded both ways; open when it is a limit */
static void lim(int root, int pos, double c, double r, double *d, double *u, int *open)
{
  double low = root ? 0 : -INFINITY, one = root ? 1 : 0, mid = root ? 0 : 1, v;
  *open = 1;
  if (c == 1) { v = one; *open = 0; }
  else if (c == 0 || c == INFINITY) v = (c == 0) == pos ? low : INFINITY;   /* ln c is -inf or inf */
  else if (r == mid) v = (c > 1) == pos ? INFINITY : low;                    /* 1/y or 1/ln x is -inf or inf */
  else if (isinf(r) || r == 0) v = one;                                      /* 1/y or 1/ln x is 0 */
  else { solve(root, r, c, d, u); *open = 0; return; }
#if IVAL_PLANT_ARITH == 40   /* 40: a limit taken as reached */
  *open = 0;
#endif
  *d = *u = v;
}
/* the piece from a range [r1, r2] of y (or x) on one side, for C+ = [c0, ch]: the lower end is c0's power at the end
   of the range where it is least and the upper end ch's where it is greatest. On the positive side c^(1/y) (and
   log_x(c)) increase with y (x) for c < 1 and decrease for c > 1; on the negative side the reverse. */
static struct piece side(int root, int pos, double r1, double r2, double c0, double ch)
{
  struct piece p;
#if IVAL_PLANT_ARITH == 38   /* 38: the negative side's ends chosen as the positive side's */
  pos = 1;
#endif
  if (pos) { lim(root, 1, c0, c0 < 1 ? r1 : r2, &p.ld, &p.lu, &p.lopen); lim(root, 1, ch, ch > 1 ? r1 : r2, &p.hd, &p.hu, &p.hopen); }
  else { lim(root, 0, ch, ch > 1 ? r2 : r1, &p.ld, &p.lu, &p.lopen); lim(root, 0, c0, c0 < 1 ? r2 : r1, &p.hd, &p.hu, &p.hopen); }
  return p;
}
static struct piece open2(double l, double h) { struct piece p = { l, l, h, h, 1, 1 }; return p; }

void ival_powrev1(const double *blo, const double *bhi, const double *clo, const double *chi, const double *xlo,
                  const double *xhi, double *zlo, double *zhi, size_t n)
{
  ENTER
  for (size_t i = 0; i < n; i++) {
    double bl = blo[i], bh = bhi[i], cl = clo[i], ch = chi[i], c0 = cl > 0 ? cl : 0;
    struct piece q[4];
    int k = 0;
    if (!empty(bl, bh) && !empty(cl, ch) && !empty(xlo[i], xhi[i])) {
      if (cl <= 0 && 0 <= ch && bh > 0) q[k++] = pc(0, 0, 0, 0);                    /* x = 0 */
      if (ch > 0) {                                                                    /* x > 0 */
        if (bh > 0) q[k++] = side(1, 1, bl > 0 ? bl : 0, bh, c0, ch);
        if (bl < 0) q[k++] = side(1, 0, bl, bh < 0 ? bh : 0, c0, ch);
        if (bl <= 0 && 0 <= bh && cl <= 1 && 1 <= ch) q[k++] = open2(0, INFINITY);   /* y = 0 */
      }
    }
    meet_hull(q, k, xlo[i], xhi[i], &zlo[i], &zhi[i]);
  }
  LEAVE
}

void ival_powrev2(const double *alo, const double *ahi, const double *clo, const double *chi, const double *ylo,
                  const double *yhi, double *zlo, double *zhi, size_t n)
{
  ENTER
  for (size_t i = 0; i < n; i++) {
    double al = alo[i], ah = ahi[i], cl = clo[i], ch = chi[i], c0 = cl > 0 ? cl : 0;
    struct piece q[4];
    int k = 0;
    if (!empty(al, ah) && !empty(cl, ch) && !empty(ylo[i], yhi[i])) {
      if (al <= 0 && 0 <= ah && cl <= 0 && 0 <= ch) q[k++] = open2(0, INFINITY);      /* x = 0, y > 0 */
      if (ch > 0) {
        if (ah > 1) q[k++] = side(0, 1, al > 1 ? al : 1, ah, c0, ch);                  /* x > 1 */
        if (al < 1 && ah > 0) q[k++] = side(0, 0, al > 0 ? al : 0, ah < 1 ? ah : 1, c0, ch);   /* 0 < x < 1 */
        if (al <= 1 && 1 <= ah && cl <= 1 && 1 <= ch) q[k++] = open2(-INFINITY, INFINITY);   /* x = 1 */
      }
    }
    meet_hull(q, k, ylo[i], yhi[i], &zlo[i], &zhi[i]);
  }
  LEAVE
}
