/* crdist.h: the inverse error functions and the inverse of the standard
   normal distribution function, binary32 (and binary16 and bfloat16,
   below), correctly rounded to nearest
   (added 2026-10-12; ROADMAP.md, item 5). No other library we know of
   rounds these correctly: CORE-MATH and RLIBM have no inverse of erf, and
   the vendors' best (Intel IPP's) is within an ulp (checked 2026-10-12
   and 13).

     erfinv(x)   the y with erf(y) = x, for -1 <= x <= 1
     erfcinv(x)  the y with erfc(y) = x, for 0 <= x <= 2
     probit(p)   the y with Phi(y) = p, for 0 <= p <= 1, where
                 Phi(y) = erfc(-y/sqrt 2)/2 is the standard normal
                 distribution function (probit = sqrt 2 erfinv(2p - 1),
                 R's qnorm, SciPy's ndtri)

   Element by element (y may be x). Each is computed in binary64 from
   polynomials and CORE-MATH's correctly rounded log, with a relative error
   below 2^-50 (crdist-fast.h; crdist-check bound-all checks it on every
   input), and rounded to binary32 when the error interval cannot round two
   ways. Otherwise the result comes from a table (crdist-exceptions.h),
   made by dist/gen-exceptions.c with MPFR from every input where that
   happens. crdist-check checks all 2^32 inputs of each (make
   crdist-check-all).

   At the ends of the domain the infinities: erfinv(+-1) = +-inf,
   erfcinv(0) = +inf, erfcinv(2) = -inf, probit(0) = -inf, probit(1) =
   +inf. Outside it, NaN. erfinv(+-0) = +-0; erfcinv(1) = probit(1/2) = +0.

   Every entry point runs in the default floating-point environment: round
   to nearest, no flush-to-zero. It saves the caller's environment, sets
   the default, and restores it after, so the caller's rounding mode or
   -ffast-math's flush-to-zero cannot change a result. The exception flags
   are restored too: these functions do not report exceptions. */
#ifndef CRDIST_H
#define CRDIST_H
#include <stddef.h>
#include <stdint.h>

#define CRDIST_VERSION "0.3.0"
#ifdef __cplusplus
extern "C" {
#endif
const char *crdist_version(void);   /* the library's CRDIST_VERSION */

void crdist_erfinvf(float *y, const float *x, size_t n);
void crdist_erfcinvf(float *y, const float *x, size_t n);
void crdist_probitf(float *y, const float *p, size_t n);

/* the same three in binary16 (f16) and bfloat16 (bf16), as bit patterns
   (uint16_t), in and out: correctly rounded to nearest in the format, from
   the same binary64 values, a rounding test per format and a table per
   format (crdist-exceptions16.h); crdist-check tries all 2^16 inputs of
   each (added 2026-10-13). Not the binary32 result rounded again: that
   double rounding is wrong where it lands on a 16-bit midpoint. */
void crdist_erfinv_f16(uint16_t *y, const uint16_t *x, size_t n);
void crdist_erfcinv_f16(uint16_t *y, const uint16_t *x, size_t n);
void crdist_probit_f16(uint16_t *y, const uint16_t *p, size_t n);
void crdist_erfinv_bf16(uint16_t *y, const uint16_t *x, size_t n);
void crdist_erfcinv_bf16(uint16_t *y, const uint16_t *x, size_t n);
void crdist_probit_bf16(uint16_t *y, const uint16_t *p, size_t n);

#ifdef __cplusplus
}
#endif
#endif
