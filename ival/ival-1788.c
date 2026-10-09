/* ival-1788.c: the rest of IEEE 1788.1's basic operations on bare intervals (ival.h): the integer roundings, abs,
   sign, min and max, intersection and hull, cancelMinus and cancelPlus, the numeric functions (inf, sup, mid, wid,
   rad, mag, mig, midRad) and the boolean ones (isEmpty and the other predicates, the comparisons, overlap).

   The conventions are ival-arith.c's: [NaN, NaN] is empty, as is any input with lo > hi, a NaN end, lo = +inf or
   hi = -inf; zero ends come out as +0. Most of these are exact. Those that round (wid, rad, mid and the cancel
   operations) round to nearest with error-free transformations (ival-eft.h); every call runs with the mode to nearest
   and the flush modes off, and gives the caller's state back. 2026-10-09. */
#include <fenv.h>
#include <float.h>
#include <math.h>
#include "ival.h"
#include "ival-eft.h"

#define ENTER ival_env env; env_save(&env); set_round(FE_TONEAREST); flush_off();
#define LEAVE env_restore(&env);
#define I1(name, ...)                                                                                         \
  void ival_##name(const double *lo, const double *hi, double *ylo, double *yhi, size_t n)                    \
  {                                                                                                           \
    ENTER                                                                                                     \
    for (size_t i = 0; i < n; i++) {                                                                          \
      double l = lo[i], h = hi[i], zl, zh;                                                                    \
      if (empty(l, h)) { ylo[i] = yhi[i] = NAN; continue; }                                                   \
      __VA_ARGS__                                                                                             \
      ylo[i] = canon(zl); yhi[i] = canon(zh);                                                                 \
    }                                                                                                         \
    LEAVE                                                                                                     \
  }

/* ---- one interval to one. Each is monotone, so its bounds are its values at the ends ---- */
I1(pos, zl = l; zh = h;)
I1(abs, if (l >= 0) { zl = l; zh = h; } else if (h <= 0) { zl = -h; zh = -l; } else { zl = 0; zh = -l > h ? -l : h; })
I1(sign, zl = l > 0 ? 1 : l < 0 ? -1 : 0; zh = h > 0 ? 1 : h < 0 ? -1 : 0;)
I1(ceil, zl = ceil(l); zh = ceil(h);)
I1(floor, zl = floor(l); zh = floor(h);)
I1(trunc, zl = trunc(l); zh = trunc(h);)
I1(round, zl = round(l); zh = round(h);)            /* 1788's roundTiesToAway */
I1(roundeven, zl = nearbyint(l); zh = nearbyint(h);) /* roundTiesToEven: the mode is to nearest here */

/* ---- two intervals to one ---- */
#define I2(name, ...)                                                                                         \
  void ival_##name(const double *alo, const double *ahi, const double *blo, const double *bhi, double *zlo,   \
                   double *zhi, size_t n)                                                                     \
  {                                                                                                           \
    ENTER                                                                                                     \
    for (size_t i = 0; i < n; i++) {                                                                          \
      double al = alo[i], ah = ahi[i], bl = blo[i], bh = bhi[i], zl, zh;                                      \
      int ea = empty(al, ah), eb = empty(bl, bh);                                                             \
      __VA_ARGS__                                                                                             \
      if (zl != zl) zh = NAN;                                                                                 \
      zlo[i] = canon(zl); zhi[i] = canon(zh);                                                                 \
    }                                                                                                         \
    LEAVE                                                                                                     \
  }
I2(min, if (ea || eb) zl = zh = NAN; else { zl = al < bl ? al : bl; zh = ah < bh ? ah : bh; })
I2(max, if (ea || eb) zl = zh = NAN; else { zl = al > bl ? al : bl; zh = ah > bh ? ah : bh; })
I2(intersect,
   if (ea || eb) zl = zh = NAN;
   else { zl = al > bl ? al : bl; zh = ah < bh ? ah : bh; if (zl > zh) zl = zh = NAN; })
I2(hull,
   if (ea && eb) zl = zh = NAN;
   else if (ea) { zl = bl; zh = bh; }
   else if (eb) { zl = al; zh = ah; }
   else { zl = al < bl ? al : bl; zh = ah > bh ? ah : bh; })

/* cancelMinus(A, B): the tightest Z with B + Z holding A, so [al - bl rounded down, ah - bh rounded up], which exists
   when A is at least as wide as B, compared exactly. Otherwise, or when either is unbounded, or B is empty and A is
   not, the whole line; A empty and B bounded (empty included) gives the empty interval. */
static void cancel(double al, double ah, double bl, double bh, double *zl, double *zh)
{
  int ea = empty(al, ah), eb = empty(bl, bh);
  int ua = !ea && (isinf(al) || isinf(ah)), ub = !eb && (isinf(bl) || isinf(bh));
  if (ea && !ub) { *zl = *zh = NAN; return; }
  if (ea || ua || ub || eb || cmp_diff(ah, al, bh, bl) < 0) { *zl = -INFINITY; *zh = INFINITY; return; }
  *zl = add_r(al, -bl, 0); *zh = add_r(ah, -bh, 1);
}
I2(cancelminus, (void)ea; (void)eb; cancel(al, ah, bl, bh, &zl, &zh);)
I2(cancelplus, (void)ea; (void)eb; cancel(al, ah, -bh, -bl, &zl, &zh);)

/* ---- numbers ---- */
#define N1(name, ...)                                                                                         \
  void ival_##name(const double *lo, const double *hi, double *y, size_t n)                                   \
  {                                                                                                           \
    ENTER                                                                                                     \
    for (size_t i = 0; i < n; i++) { double l = lo[i], h = hi[i], e = empty(l, h), r; __VA_ARGS__ y[i] = r; } \
    LEAVE                                                                                                     \
  }
/* 1788.1: inf of an interval starting at 0 is -0, sup of one ending at 0 is +0; of the empty one, +inf and -inf */
N1(inf, r = e ? INFINITY : l == 0 ? -0.0 : l;)
N1(sup, r = e ? -INFINITY : h == 0 ? 0.0 : h;)
/* the midpoint rounded to nearest; DBL_MAX or -DBL_MAX for a half-line, 0 for the whole line. (l + h) / 2 rounds
   once: l + h is exact when it lands below 2^-1021, and halving is exact above that; a sum that overflows is
   between ends at least 2^1022, whose halves are exact. */
static double mid1(double l, double h)
{
  if (isinf(l)) return isinf(h) ? 0.0 : -DBL_MAX;
  if (isinf(h)) return DBL_MAX;
#if IVAL_PLANT_ARITH == 17   /* 17: always the halves, which round twice below 2^-1021 */
  return l * 0.5 + h * 0.5;
#endif
  double s = l + h;
  return isinf(s) ? l * 0.5 + h * 0.5 : s * 0.5;
}
N1(mid, r = e ? NAN : mid1(l, h) + 0.0;)
N1(wid, r = e ? NAN : isinf(l) || isinf(h) ? INFINITY : add_r(h, -l, 1);)
/* the least r with [m - r, m + r] holding the interval, m its mid, rounded up */
static double rad1(double l, double h)
{
  if (isinf(l) || isinf(h)) return INFINITY;
  double m = mid1(l, h), a = add_r(m, -l, 1), b = add_r(h, -m, 1);
  return a > b ? a : b;
}
N1(rad, r = e ? NAN : rad1(l, h);)
N1(mag, r = e ? NAN : fabs(l) > fabs(h) ? fabs(l) : fabs(h);)
N1(mig, r = e ? NAN : l > 0 ? l : h < 0 ? -h : 0.0;)
void ival_midrad(const double *lo, const double *hi, double *m, double *r, size_t n)
{
  ENTER
  for (size_t i = 0; i < n; i++) {
    double l = lo[i], h = hi[i];
    if (empty(l, h)) { m[i] = r[i] = NAN; continue; }
    m[i] = mid1(l, h) + 0.0; r[i] = rad1(l, h);
  }
  LEAVE
}

/* ---- booleans, 1 or 0 ---- */
#define B1(name, ...)                                                                                         \
  void ival_##name(const double *lo, const double *hi, unsigned char *r, size_t n)                            \
  {                                                                                                           \
    for (size_t i = 0; i < n; i++) { double l = lo[i], h = hi[i]; int e = empty(l, h); r[i] = (__VA_ARGS__); } \
  }
B1(isempty, e)
B1(isentire, !e && l == -INFINITY && h == INFINITY)
B1(issingleton, !e && l == h)
B1(iscommon, !e && !isinf(l) && !isinf(h))
void ival_ismember(const double *x, const double *lo, const double *hi, unsigned char *r, size_t n)
{
  for (size_t i = 0; i < n; i++) r[i] = !empty(lo[i], hi[i]) && isfinite(x[i]) && lo[i] <= x[i] && x[i] <= hi[i];
}
#define B2(name, ...)                                                                                         \
  void ival_##name(const double *alo, const double *ahi, const double *blo, const double *bhi,                \
                   unsigned char *r, size_t n)                                                                \
  {                                                                                                           \
    for (size_t i = 0; i < n; i++) {                                                                          \
      double al = alo[i], ah = ahi[i], bl = blo[i], bh = bhi[i];                                              \
      int ea = empty(al, ah), eb = empty(bl, bh);                                                             \
      r[i] = (__VA_ARGS__);                                                                                   \
    }                                                                                                         \
  }
B2(equal, ea || eb ? ea && eb : al == bl && ah == bh)
B2(subset, ea ? 1 : eb ? 0 : bl <= al && ah <= bh)
B2(less, ea || eb ? ea && eb : al <= bl && ah <= bh)
B2(precedes, ea || eb ? 1 : ah <= bl)
#if IVAL_PLANT_ARITH == 18   /* 18: interior without the infinite ends' clause */
B2(interior, ea ? 1 : eb ? 0 : bl < al && ah < bh)
#else
B2(interior, ea ? 1 : eb ? 0 : (bl < al || (bl == -INFINITY && al == -INFINITY)) && (ah < bh || (ah == INFINITY && bh == INFINITY)))
#endif
B2(strictless, ea || eb ? ea && eb : (al < bl || (al == -INFINITY && bl == -INFINITY)) && (ah < bh || (ah == INFINITY && bh == INFINITY)))
B2(strictprecedes, ea || eb ? 1 : ah < bl)
B2(disjoint, ea || eb ? 1 : ah < bl || bh < al)

/* overlap: 1788.1's sixteen states (ival.h, enum ival_overlap), each a condition on the four ends that excludes the
   others for nonempty A and B */
static int overlap1(double al, double ah, double bl, double bh)
{
  int ea = empty(al, ah), eb = empty(bl, bh);
  if (ea || eb) return ea && eb ? IVAL_BOTH_EMPTY : ea ? IVAL_FIRST_EMPTY : IVAL_SECOND_EMPTY;
  if (ah < bl) return IVAL_BEFORE;
  if (bh < al) return IVAL_AFTER;
  if (al == bl && ah == bh) return IVAL_EQUALS;
#if IVAL_PLANT_ARITH == 19   /* 19: meets without its strictness, so a point at B's start meets it */
  if (ah == bl && bl < bh) return IVAL_MEETS;
#endif
  if (al < ah && ah == bl && bl < bh) return IVAL_MEETS;
  if (bl < bh && bh == al && al < ah) return IVAL_MET_BY;
  if (al == bl) return ah < bh ? IVAL_STARTS : IVAL_STARTED_BY;
  if (ah == bh) return bl < al ? IVAL_FINISHES : IVAL_FINISHED_BY;
  if (bl < al && ah < bh) return IVAL_CONTAINED_BY;
  if (al < bl && bh < ah) return IVAL_CONTAINS;
  return al < bl ? IVAL_OVERLAPS : IVAL_OVERLAPPED_BY;
}
void ival_overlap(const double *alo, const double *ahi, const double *blo, const double *bhi, unsigned char *r,
                  size_t n)
{
  for (size_t i = 0; i < n; i++) r[i] = (unsigned char)overlap1(alo[i], ahi[i], blo[i], bhi[i]);
}

/* ---- mulRev and mulRevToPair: S = {x : x b in C for some b in B}, division with gaps. Each of S's at most two
   pieces is a real interval whose ends are quotients of ends (or infinities), kept rounded both ways: the hull takes
   the outer roundings, and an exact test against X takes the inner ones (a real q is below a binary64 xl exactly
   when q rounded down is). An end that is 0 only as a limit (a divisor running to an infinity) is not in S. ---- */
static struct piece qpiece(double nl, double dl, double nh, double dh)   /* [nl / dl, nh / dh], quotients */
{
  struct piece p = { div_r(nl, dl, 0), div_r(nl, dl, 1), div_r(nh, dh, 0), div_r(nh, dh, 1), 0, 0 };
  p.lopen = isinf(dl) && !isinf(nl) && nl != 0; p.hopen = isinf(dh) && !isinf(nh) && nh != 0;   /* c = 0 is attained */
  return p;
}
static struct piece lohalf(double nh, double dh)   /* [-inf, nh / dh] */
{
  struct piece p = qpiece(1, 1, nh, dh);
  p.ld = p.lu = -INFINITY; p.lopen = 0;
  return p;
}
static struct piece hihalf(double nl, double dl)   /* [nl / dl, +inf] */
{
  struct piece p = qpiece(nl, dl, 1, 1);
  p.hd = p.hu = INFINITY; p.hopen = 0;
  return p;
}
static int pieces(double bl, double bh, double cl, double ch, struct piece p[2])
{
  if (empty(bl, bh) || empty(cl, ch)) return 0;
  int c0 = cl <= 0 && 0 <= ch;
  if (c0 && bl <= 0 && 0 <= bh) { p[0] = lohalf(1, 1); p[0].hd = p[0].hu = INFINITY; p[0].hopen = 0; return 1; }
  if (bl > 0) {                         /* 1788's division table, C over B > 0 */
    if (cl >= 0) p[0] = qpiece(cl, bh, ch, bl);
    else if (ch <= 0) p[0] = qpiece(cl, bl, ch, bh);
    else p[0] = qpiece(cl, bl, ch, bl);
    return 1;
  }
  if (bh < 0) {                         /* and over B < 0 */
    if (cl >= 0) p[0] = qpiece(ch, bh, cl, bl);
    else if (ch <= 0) p[0] = qpiece(ch, bl, cl, bh);
    else p[0] = qpiece(ch, bh, cl, bh);
    return 1;
  }
  if (bl == 0 && bh == 0) return 0;     /* 0 in B, not in C */
#if IVAL_PLANT_ARITH == 22              /* 22: the gap's sides swapped for C > 0 */
  if (cl > 0) cl = -cl, ch = -ch;
#endif
  if (cl > 0) {
    if (bl == 0) { p[0] = hihalf(cl, bh); return 1; }
    if (bh == 0) { p[0] = lohalf(cl, bl); return 1; }
    p[0] = lohalf(cl, bl); p[1] = hihalf(cl, bh);
    return 2;
  }
  if (bl == 0) { p[0] = lohalf(ch, bh); return 1; }
  if (bh == 0) { p[0] = hihalf(ch, bl); return 1; }
  p[0] = lohalf(ch, bh); p[1] = hihalf(ch, bl);
  return 2;
}
void ival_mulrevpair(const double *blo, const double *bhi, const double *clo, const double *chi, double *z1lo,
                     double *z1hi, double *z2lo, double *z2hi, size_t n)
{
  ENTER
  for (size_t i = 0; i < n; i++) {
    struct piece p[2];
    int k = pieces(blo[i], bhi[i], clo[i], chi[i], p);
    z1lo[i] = k > 0 ? canon(p[0].ld) : NAN; z1hi[i] = k > 0 ? canon(p[0].hu) : NAN;
    z2lo[i] = k > 1 ? canon(p[1].ld) : NAN; z2hi[i] = k > 1 ? canon(p[1].hu) : NAN;
  }
  LEAVE
}
void ival_mulrev(const double *blo, const double *bhi, const double *clo, const double *chi, const double *xlo,
                 const double *xhi, double *zlo, double *zhi, size_t n)
{
  ENTER
  for (size_t i = 0; i < n; i++) {
    struct piece p[2];
    int k = empty(xlo[i], xhi[i]) ? 0 : pieces(blo[i], bhi[i], clo[i], chi[i], p);
    meet_hull(p, k, xlo[i], xhi[i], &zlo[i], &zhi[i]);
  }
  LEAVE
}

const char *ival_version(void) { return IVAL_VERSION; }
