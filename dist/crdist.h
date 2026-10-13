/* crdist.h: the inverse error functions and the inverse of the standard
   normal distribution function, binary32, correctly rounded to nearest
   (added 2026-10-12; ROADMAP.md, item 5). No other library we know of
   rounds these correctly: CORE-MATH has no inverse of erf, and the
   vendors' best (Intel IPP's) is within an ulp (checked 2026-10-12).

     erfinv(x)   the y with erf(y) = x, for -1 <= x <= 1
     erfcinv(x)  the y with erfc(y) = x, for 0 <= x <= 2
     probit(p)   the y with Phi(y) = p, for 0 <= p <= 1, where
                 Phi(y) = erfc(-y/sqrt 2)/2 is the standard normal
                 distribution function (probit = sqrt 2 erfinv(2p - 1),
                 R's qnorm, SciPy's ndtri)

   Element by element (y may be x). Each is computed in binary64 from
   CORE-MATH's correctly rounded erf, erfc, exp, log and log1p, with a
   proven relative error below 2^-51 (crdist-fast.h), and rounded to
   binary32 when the error interval cannot round two ways. Otherwise the
   result comes from a table (crdist-exceptions.h), made by
   dist/gen-exceptions.c with MPFR from every input where that happens.
   crdist-check checks all 2^32 inputs of each (make crdist-check-all).

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

void crdist_erfinvf(float *y, const float *x, size_t n);
void crdist_erfcinvf(float *y, const float *x, size_t n);
void crdist_probitf(float *y, const float *p, size_t n);

#endif
