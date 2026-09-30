/* lowp.h: correctly rounded math for the OCP 8-bit floating-point formats
   E4M3 and E5M2 (numerics/README.md, "lowp").

   Every result is the correctly rounded value of the exact function at the
   input, in the rounding mode asked for, with the format's subnormals and
   overflow. The same inputs give the same bits on every machine:
   - the one-argument functions are tables, generated from MPFR;
   - the two-argument functions go through CORE-MATH's correctly rounded
     binary64 functions and this library's own rounding.
   Each function was checked on every input (every pair) in every mode.

   Values are encodings, one byte each (sign, exponent, significand):
     E4M3  bias 7,  largest 448,   no infinity; S.1111.111 is NaN
     E5M2  bias 15, largest 57344, infinities and NaNs as in IEEE 754

   mode: one of LOWP_NEAREST (ties to even), LOWP_UP, LOWP_DOWN, LOWP_ZERO,
   plus LOWP_SAT for E4M3 only. E4M3 overflow gives NaN by default. With
   LOWP_SAT it gives the largest finite value of the right sign, as OCP's
   saturating mode does. Overflow toward zero gives it in either case, and an
   infinite exact result (an infinite input, or a pole) is treated as an
   overflow away from zero.

   Every function takes n inputs and writes n results. It returns 0, or -1
   for a mode it does not know, and then writes nothing. The C rounding
   mode and floating-point flags are left as they were. NaN results are
   the canonical quiet NaN: 0x7f (E4M3), 0x7e (E5M2). */
#ifndef LOWP_H
#define LOWP_H
#include <stddef.h>
#include <stdint.h>

enum { LOWP_NEAREST = 0, LOWP_UP = 1, LOWP_DOWN = 2, LOWP_ZERO = 3, LOWP_SAT = 4 };

/* conversions: to FP8 correctly rounded, from FP8 exact */
int lowp_e4m3_from_f64(const double *x, uint8_t *y, size_t n, int mode);
int lowp_e5m2_from_f64(const double *x, uint8_t *y, size_t n, int mode);
int lowp_e4m3_from_f32(const float *x, uint8_t *y, size_t n, int mode);
int lowp_e5m2_from_f32(const float *x, uint8_t *y, size_t n, int mode);
void lowp_e4m3_to_f64(const uint8_t *x, double *y, size_t n);
void lowp_e5m2_to_f64(const uint8_t *x, double *y, size_t n);
void lowp_e4m3_to_f32(const uint8_t *x, float *y, size_t n);
void lowp_e5m2_to_f32(const uint8_t *x, float *y, size_t n);

/* one argument: y[i] = f(x[i]) */
#define LOWP_F1(f)                                                        \
  int lowp_e4m3_##f(const uint8_t *x, uint8_t *y, size_t n, int mode); \
  int lowp_e5m2_##f(const uint8_t *x, uint8_t *y, size_t n, int mode);
/* two arguments: z[i] = f(x[i], y[i]) (atan2 and atan2pi: f(y, x) in C's
   order, so x here is C's y) */
#define LOWP_F2(f)                                                                          \
  int lowp_e4m3_##f(const uint8_t *x, const uint8_t *y, uint8_t *z, size_t n, int mode); \
  int lowp_e5m2_##f(const uint8_t *x, const uint8_t *y, uint8_t *z, size_t n, int mode);
#include "lowp-list.h"

#endif
