/* crsum.h: correctly rounded sums and dot products, the same bits under any
   order, thread count or vector width (numerics/README.md, "crsum").

   Every term is added exactly into a fixed-point accumulator wide enough
   for any sum of binary64 values or of their products, and the total is
   rounded once, correctly, in the mode asked for. There is one right
   answer, so the order of the terms, how they are split between threads,
   and the vector width can't change it: reproducible by construction.

   mode: CRSUM_NEAREST (ties to even), CRSUM_UP, CRSUM_DOWN, CRSUM_ZERO.
   Special values are as a sequence of exact additions would give them:
   a NaN, or +inf and -inf together, give NaN; an infinity of one sign
   gives it. An exact zero is +0, except -0 when every term is -0, and -0
   in CRSUM_DOWN when the terms cancel (MPFR's mpfr_sum and IEEE 754).
   Overflow gives an infinity or the largest finite value, per the mode.
   The sum of no terms is +0. Nothing depends on the C rounding mode or
   floating-point flags, which are left as they were. A mode it doesn't
   know gives NaN.

   The accumulator is 134 limbs of 32 bits in 64-bit integers (1,072
   bytes): exponents 2^-2176 to 2^2112, enough for any product of two
   binary64 values, carries deferred and settled every 2^29 terms. */
#ifndef CRSUM_H
#define CRSUM_H
#include <stddef.h>
#include <stdint.h>

enum { CRSUM_NEAREST = 0, CRSUM_UP = 1, CRSUM_DOWN = 2, CRSUM_ZERO = 3 };

/* sum of x[0 .. n-1], and dot product of x and y, correctly rounded */
double crsum(const double *x, size_t n, int mode);
double crdot(const double *x, const double *y, size_t n, int mode);
/* the same over binary32 values, rounded to binary32 */
float crsumf(const float *x, size_t n, int mode);
float crdotf(const float *x, const float *y, size_t n, int mode);

/* the accumulator, for sums in pieces: across calls, or across threads
   (one accumulator each, merged). Merging is exact, so the result is the
   same however the terms were split. */
#define CRSUM_LIMBS 134
typedef struct {
  int64_t limb[CRSUM_LIMBS];
  uint64_t since;          /* terms since the carries were last settled */
  int nan, pinf, ninf;     /* a NaN; +inf; -inf seen */
  int nonzero, poszero, negzero;   /* a nonzero term; a +0; a -0 (an exact zero's sign) */
  uint64_t terms;          /* how many terms */
} crsum_acc;

void crsum_init(crsum_acc *a);
void crsum_add(crsum_acc *a, const double *x, size_t n);
void crsum_add_dot(crsum_acc *a, const double *x, const double *y, size_t n);
void crsum_merge(crsum_acc *into, const crsum_acc *from);
double crsum_round(const crsum_acc *a, int mode);
float crsum_roundf(const crsum_acc *a, int mode);

#endif
