/* crdist.h: the inverse error functions and the inverse of the standard
   normal distribution function, binary32 (and binary16 and bfloat16,
   below), correctly rounded to nearest; then the normal distribution's
   functions of x, and samplers (both below)
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

#define CRDIST_VERSION "0.4.0"
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

/* The normal distribution's functions of x, and logit (added 2026-10-13),
   correctly rounded to nearest in the same way, in binary32 and, with the
   _f16 and _bf16 suffixes, binary16 and bfloat16:

     ndtr(x)      = Phi(x) = erfc(-x/sqrt 2)/2, the standard normal
                    distribution function (SciPy's ndtr, R's pnorm); the
                    survival function is ndtr(-x), the negation being exact
     log_ndtr(x)  = log Phi(x), accurate where Phi underflows (SciPy's
                    log_ndtr): log_ndtr(-40) is -804.6084, though Phi(-40),
                    about 4e-350, is below even binary64's subnormals
     normpdf(x)   = e^(-x^2/2)/sqrt(2 pi), the standard normal density
     logit(p)     = log(p/(1 - p)), the log-odds, for 0 <= p <= 1 (its
                    inverse, the logistic sigmoid, is crnn's crnn_sigmoidf)

   At the ends: ndtr(-inf) = +0, ndtr(+inf) = 1; log_ndtr(-inf) = -inf,
   log_ndtr(+inf) = +0, and -0 for finite x from 15 up (the exact value is
   negative and below half the smallest subnormal); normpdf(+-inf) = +0;
   logit(0) = -inf, logit(1) = +inf, logit(1/2) = +0, NaN outside [0, 1]. */
void crdist_ndtrf(float *y, const float *x, size_t n);
void crdist_log_ndtrf(float *y, const float *x, size_t n);
void crdist_normpdff(float *y, const float *x, size_t n);
void crdist_logitf(float *y, const float *p, size_t n);
void crdist_ndtr_f16(uint16_t *y, const uint16_t *x, size_t n);
void crdist_log_ndtr_f16(uint16_t *y, const uint16_t *x, size_t n);
void crdist_normpdf_f16(uint16_t *y, const uint16_t *x, size_t n);
void crdist_logit_f16(uint16_t *y, const uint16_t *p, size_t n);
void crdist_ndtr_bf16(uint16_t *y, const uint16_t *x, size_t n);
void crdist_log_ndtr_bf16(uint16_t *y, const uint16_t *x, size_t n);
void crdist_normpdf_bf16(uint16_t *y, const uint16_t *x, size_t n);
void crdist_logit_bf16(uint16_t *y, const uint16_t *p, size_t n);

/* SAMPLERS (added 2026-10-13): random variates with the same bits on every
   machine, for the same stream. Not correctly rounded (a variate has no
   exact value to round): each is the fixed sequence of operations below,
   every one of them correctly rounded or exact, so the specification fixes
   the bits. crdist-check computes it independently with MPFR and compares
   bit for bit, and tests the distributions.

   THE STREAM: Philox4x32-10 (Salmon, Moraes, Dror and Shaw, SC 2011, as
   in Random123, whose known answers crdist-check reproduces). Word i of
   the stream (seed, substream) is word i mod 4 of Philox4x32-10 at the
   counter (j mod 2^32, j div 2^32, substream mod 2^32, substream div 2^32),
   j = i div 4, under the key (seed mod 2^32, seed div 2^32). Each call
   takes the words it needs from index on and leaves index after the last;
   so streams with different substreams never overlap, and one stream can
   be cut into pieces by index.

   THE VARIATES, from the next words w (32-bit integers), in binary64
   unless said otherwise; RN is rounding to nearest, log and exp are
   CORE-MATH's (correctly rounded), probit is crdist's binary32 probit:
     uniform:      (2 (w >> 9) + 1) 2^-24, in binary32 (exact): 2^23
                   values in (0, 1), symmetric about 1/2
     U (internal): (2w + 1) 2^-33, exact, in (0, 1)
     normal:       q = RN32((2 (w & 0x7fffffff) + 1) 2^-33), in (0, 1/2];
                   z = probit(q), negated when w >= 2^31 (tails to 6.4)
     exponential:  RN32(-log U)
     gamma(a):     Marsaglia and Tsang (2000), without the squeeze. For
                   a >= 1: d = a - RN(1/3), c = 1/sqrt(9 d); repeat: z a
                   normal (one word), v = 1 + c z; if v > 0: v = v v v,
                   u = U (one word), and accept d v if log u <
                   ((0.5 (z z) + d) - d v) + d log v. For a < 1: g =
                   gamma(a + 1), then u = U, and g exp(log(u)/a). RN32 at
                   the end.
     beta(a, b):   la, lb = the logs of gamma(a), gamma(b), in that order
                   (below 1: log g + log(u)/a, not log of the product);
                   RN32(1/(1 + exp(lb - la)))
     chisquared(k):  RN32(2 gamma(k/2))
     student_t(nu):  z a normal, then g = gamma(nu/2); RN32(z/sqrt((2 g)/nu))
   Every operation rounds to nearest in binary64 as written (no
   contraction), and each call runs in the default environment. A shape
   that is not positive and finite gives NaNs and takes no words.
   crdist_words gives the stream's words themselves. */
typedef struct { uint64_t seed, substream, index; } crdist_stream;
void crdist_words(uint32_t *w, size_t n, crdist_stream *s);
void crdist_uniformf(float *y, size_t n, crdist_stream *s);
void crdist_normalf(float *y, size_t n, crdist_stream *s);
void crdist_exponentialf(float *y, size_t n, crdist_stream *s);
void crdist_gammaf(float *y, size_t n, double shape, crdist_stream *s);
void crdist_betaf(float *y, size_t n, double a, double b, crdist_stream *s);
void crdist_chisquaredf(float *y, size_t n, double k, crdist_stream *s);
void crdist_student_tf(float *y, size_t n, double nu, crdist_stream *s);

#ifdef __cplusplus
}
#endif
#endif
