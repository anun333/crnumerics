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
   optionally | LOWP_SAT. By default an overflow away from zero gives NaN in
   E4M3 and infinity in E5M2. With LOWP_SAT it gives the largest finite value
   of the right sign in both (448, 57344), as OCP's saturating mode does
   (OFP8 1.0, Table 3). Overflow toward zero gives it in either case, and an
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

/* MX blocks (lowp/MX.md): LOWP_MX_K elements of one type, one per byte
   (6- and 4-bit types in the low bits), under one E8M0 scale byte:
   element i stands for 2^(x - 127) times its value, and x = 0xff is NaN.
     p[k * LOWP_MX_K + i], x[k]   block k's elements and scale, in
     q[k * LOWP_MX_K + i], y[k]   and out (q may be p, y may be x)
   The result is the correctly rounded function on the block: the exact
   results converted to a block by OCP MX v1.0's rules where it gives them
   and its reference implementation's where it is silent (lowp/MX.md says
   which is which): the scale from the exact largest result, each element
   rounded in mode (LOWP_NEAREST ... LOWP_ZERO), subnormals kept,
   saturating; a NaN or an infinity anywhere makes the block NaN (scale
   0xff, elements 0). Types: e5m2, e4m3, e3m2 (FP6), e2m3 (FP6), e2m1
   (FP4), int8 (MXINT8: k/64 in two's complement, k = -127 ... 127 written,
   0x80 read as -2). The functions are lowp-list.h's but exp10m1, exp2m1, log10p1
   and log2p1. Returns 0, or -1 for a mode it doesn't take (nothing
   written). */
#define LOWP_MX_K 32
#define LOWP_MX1(t, f) int lowp_mx_##t##_##f(const uint8_t *p, const uint8_t *x, uint8_t *q, uint8_t *y, size_t nblocks, int mode);
#define LOWP_MX2(t, f)                                                                                      \
  int lowp_mx_##t##_##f(const uint8_t *p, const uint8_t *x, const uint8_t *p2, const uint8_t *x2, uint8_t *q, \
                        uint8_t *y, size_t nblocks, int mode);
#include "lowp-mx-list.h"

#endif
