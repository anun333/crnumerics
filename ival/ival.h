/* ival.h: interval versions of elementary functions in binary64, with the
   tightest enclosure (ival/README.md).

   ival_f(lo, hi, ylo, yhi, n): for each i, the interval X = [lo[i], hi[i]]
   gives Y = [ylo[i], yhi[i]], the smallest binary64 interval that contains
   f(x) for every real x in X within f's domain. ylo is the least value
   rounded down, and yhi the greatest rounded up, both correctly: CORE-MATH
   is correctly rounded in every mode.
     - X is intersected with f's domain first (log on [-1, 4] is log on
       [0, 4]: [-inf, log 4 rounded up]); an empty intersection gives the
       empty interval.
     - The empty interval is written [NaN, NaN]. An input with a NaN
       endpoint or lo > hi counts as empty.
     - Endpoints may be infinite: [-inf, +inf] is the whole line.
     - A pole inside X gives an infinite endpoint (tan across pi/2:
       [-inf, +inf]).
   Two-argument functions take two intervals, a box:
   ival_f(xlo, xhi, ylo, yhi, zlo, zhi, n) gives Z = [zlo[i], zhi[i]], the
   smallest binary64 interval holding f(x, y) for every x in X, y in Y (in
   C's argument order); either interval empty gives the empty one.
   The C rounding mode and floating-point flags are left as they were.
   Zero endpoints compare equal whatever their sign (IEEE 1788's sets). */
#ifndef IVAL_H
#define IVAL_H
#include <stddef.h>

#define IVAL_F(f) void ival_##f(const double *lo, const double *hi, double *ylo, double *yhi, size_t n);
#define IVAL_F2(f)                                                                                      \
  void ival_##f(const double *xlo, const double *xhi, const double *ylo, const double *yhi, double *zlo, \
                double *zhi, size_t n);
#include "ival-list.h"

/* Interval arithmetic (ival-arith.c, 2026-10-08), the same conventions, each result the tightest enclosure:
   ival_add, ival_sub, ival_mul, ival_div take two intervals A = [alo, ahi], B = [blo, bhi] and give Z = A op B;
   division follows IEEE 1788 (B = [0, 0] gives the empty interval, 0 inside B the whole line unless A = [0, 0]).
   ival_neg, ival_sqr and ival_recip take one. At interval ends 0 * inf is 0. */
#define IVAL_A2(f)                                                                                      \
  void ival_##f(const double *alo, const double *ahi, const double *blo, const double *bhi, double *zlo, \
                double *zhi, size_t n);
IVAL_A2(add) IVAL_A2(sub) IVAL_A2(mul) IVAL_A2(div)
#undef IVAL_A2
void ival_neg(const double *lo, const double *hi, double *ylo, double *yhi, size_t n);
void ival_sqr(const double *lo, const double *hi, double *ylo, double *yhi, size_t n);
void ival_recip(const double *lo, const double *hi, double *ylo, double *yhi, size_t n);
/* fma(A, B, C): the tightest enclosure of a * b + c over the box (2026-10-09), 0 * inf being 0 at ends as above */
void ival_fma(const double *alo, const double *ahi, const double *blo, const double *bhi, const double *clo,
              const double *chi, double *zlo, double *zhi, size_t n);

/* The rest of IEEE 1788.1's basic operations (ival-1788.c, 2026-10-09), the same conventions.
   One interval to one: pos, abs, sign, and the integer roundings ceil, floor, trunc, round (1788's
   roundTiesToAway) and roundeven (roundTiesToEven), each the image of the interval. */
#define IVAL_O1(f) void ival_##f(const double *lo, const double *hi, double *ylo, double *yhi, size_t n);
IVAL_O1(pos) IVAL_O1(abs) IVAL_O1(sign) IVAL_O1(ceil) IVAL_O1(floor) IVAL_O1(trunc) IVAL_O1(round) IVAL_O1(roundeven)
#undef IVAL_O1
/* Two to one: min and max (elementwise over the two sets), intersect, hull (1788's convexHull), cancelminus and
   cancelplus (the tightest Z with B + Z, or Z - B, holding A; the whole line when there is none). */
#define IVAL_O2(f)                                                                                      \
  void ival_##f(const double *alo, const double *ahi, const double *blo, const double *bhi, double *zlo, \
                double *zhi, size_t n);
IVAL_O2(min) IVAL_O2(max) IVAL_O2(intersect) IVAL_O2(hull) IVAL_O2(cancelminus) IVAL_O2(cancelplus)
#undef IVAL_O2
/* Numbers: inf and sup (+inf and -inf for the empty interval; -0 for inf at a zero end), mid (rounded to nearest;
   DBL_MAX or -DBL_MAX for a half-line, 0 for the whole line), wid, rad (rounded up), mag, mig; NaN for the empty
   interval but in inf and sup. ival_midrad gives mid and rad together. */
#define IVAL_N1(f) void ival_##f(const double *lo, const double *hi, double *y, size_t n);
IVAL_N1(inf) IVAL_N1(sup) IVAL_N1(mid) IVAL_N1(wid) IVAL_N1(rad) IVAL_N1(mag) IVAL_N1(mig)
#undef IVAL_N1
void ival_midrad(const double *lo, const double *hi, double *m, double *r, size_t n);
/* mulrevpair: {x : x b in C for some b in B} (1788's mulRevToPair, division with gaps), at most two intervals, the
   first before the second, the second (or both) empty when there are fewer; mulrev: the hull of that set's
   intersection with X (1788's mulRev; X = [-inf, inf] for the two-argument form). */
void ival_mulrevpair(const double *blo, const double *bhi, const double *clo, const double *chi, double *z1lo,
                     double *z1hi, double *z2lo, double *z2hi, size_t n);
void ival_mulrev(const double *blo, const double *bhi, const double *clo, const double *chi, const double *xlo,
                 const double *xhi, double *zlo, double *zhi, size_t n);
/* pown(x, p[i]): x to an integer power, for every real x (ival.c, 2026-10-09); a negative power of [0, 0] is empty */
void ival_pown(const double *lo, const double *hi, const int *p, double *ylo, double *yhi, size_t n);
/* Booleans, 1 or 0: of one interval, isempty, isentire, issingleton, iscommon (nonempty and bounded); ismember
   (a finite x in the interval); of two, equal, subset (A in B), less, precedes, interior, strictless,
   strictprecedes, disjoint, as 1788.1 defines them. */
#define IVAL_B1(f) void ival_##f(const double *lo, const double *hi, unsigned char *r, size_t n);
IVAL_B1(isempty) IVAL_B1(isentire) IVAL_B1(issingleton) IVAL_B1(iscommon)
#undef IVAL_B1
void ival_ismember(const double *x, const double *lo, const double *hi, unsigned char *r, size_t n);
#define IVAL_B2(f) \
  void ival_##f(const double *alo, const double *ahi, const double *blo, const double *bhi, unsigned char *r, size_t n);
IVAL_B2(equal) IVAL_B2(subset) IVAL_B2(less) IVAL_B2(precedes) IVAL_B2(interior) IVAL_B2(strictless)
IVAL_B2(strictprecedes) IVAL_B2(disjoint)
#undef IVAL_B2
/* overlap: which of 1788.1's sixteen states A and B are in, as an enum ival_overlap value */
enum ival_overlap { IVAL_BOTH_EMPTY, IVAL_FIRST_EMPTY, IVAL_SECOND_EMPTY, IVAL_BEFORE, IVAL_MEETS, IVAL_OVERLAPS,
                    IVAL_STARTS, IVAL_CONTAINED_BY, IVAL_FINISHES, IVAL_EQUALS, IVAL_FINISHED_BY, IVAL_CONTAINS,
                    IVAL_STARTED_BY, IVAL_OVERLAPPED_BY, IVAL_MET_BY, IVAL_AFTER };
void ival_overlap(const double *alo, const double *ahi, const double *blo, const double *bhi, unsigned char *r,
                  size_t n);

#endif
