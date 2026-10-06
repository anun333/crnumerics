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

#endif
