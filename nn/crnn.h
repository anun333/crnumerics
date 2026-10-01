/* crnn.h: neural-network primitives with one specified answer (added
   2026-09-30; ROADMAP.md, Phase 2, item 2). binary32 in and out.

   Every entry point runs in the default floating-point environment: round
   to nearest, no flush-to-zero. It saves the caller's environment, sets
   the default, and restores it after, so the caller's rounding mode or
   -ffast-math's flush-to-zero cannot change a result. The exception flags
   are restored too: these functions do not report exceptions.

   ONE-ARGUMENT FUNCTIONS, correctly rounded to nearest, element by
   element (y may be x). Each is computed in binary64 from CORE-MATH's
   correctly rounded functions, with a proven relative error below 2^-51,
   and rounded to binary32 when the error interval cannot round two ways.
   Otherwise the result comes from a table (crnn-exceptions.h), made by
   nn/gen-exceptions.c with MPFR from every input where that happens.
   crnn-check checks all 2^32 inputs of each (make crnn-check-all).

     sigmoid(x)  = 1 / (1 + e^-x)
     silu(x)     = x sigmoid(x)
     gelu(x)     = x/2 (1 + erf(x/sqrt 2)) = x/2 erfc(-x/sqrt 2)
     softplus(x) = log(1 + e^x)
     rsqrt(x)    = 1 / sqrt(x)   (IEEE rSqrt: rsqrt(-0) = -inf, x < 0 NaN)

   At infinities the limits: silu(-inf) = gelu(-inf) = -0, silu(+inf) =
   gelu(+inf) = softplus(+inf) = +inf, sigmoid(-inf) = softplus(-inf) = +0.

   COMPOSITES: a fixed sequence of binary64 operations, each correctly
   rounded to nearest, and crsum's exact sums (crsum.h), each rounded once.
   So the result is the same on every machine and in any order of the
   elements; it is not in general the correctly rounded value of the
   mathematical expression, but it is close (crnn-check measures how
   close, against MPFR). Below, RN(e) is e rounded to binary64, exp, log
   and rsqrt are CORE-MATH's (correctly rounded), and a sum over i is
   exact, then rounded once to binary64.

     logsumexp(x, n):  m = max x_i;  T = (sum_i exp(RN(x_i - m))) - 1, exact
                       and then rounded once;  result = binary32(RN(m + log1p(T)))
       n = 0: -inf.  A NaN: NaN.  m = +inf: +inf.  m = -inf: -inf.
       (log of the rounded sum, the first version, lost almost everything
       when the sum was 1 plus terms far below it: its rounding error, up to
       2^-53, then dominates log(S) ~ S - 1. log1p of the exact S - 1 keeps
       every bit; crnn-check's "swamped" vectors found it.)
     softmax(y, x, n): m as above, S = sum_i exp(RN(x_i - m)), rounded once;
                       y_i = binary32(RN(exp(RN(x_i - m)) / S))
       y may be x.  An infinite m or a NaN gives NaNs, as the formula does.
     layernorm(y, x, n, g, b, eps):
                       the mean as two binary64 numbers, m1 + m2:
                         s1 = sum_i x_i;  s2 = the exact sum's remainder,
                         (sum_i x_i) - s1, rounded once;  m1 = RN(s1 / n);
                         m2 = RN(RN(fma(-m1, n, s1) + s2) / n)  (the fma is
                         exact: it is the division's remainder)
                       d_i = RN(RN(x_i - m1) - m2);  v = RN(sum_i d_i^2 / n);
                       r = rsqrt(RN(v + eps));
                       y_i = binary32(RN(RN(RN(d_i r) g_i) + b_i))
       g or b may be NULL: the step that uses it is left out. With a single
       rounded mean, d_i lost up to |mean|/|d_i| times 2^-53 when the mean
       is large and the spread small (1e4 +- 0.01: 30 ulp in the result);
       the two-part mean keeps d_i within a few roundings of exact.
     rmsnorm(y, x, n, g, eps):
                       q = RN(sum_i x_i^2 / n);  r = rsqrt(RN(q + eps));
                       y_i = binary32(RN(RN(x_i r) g_i))
   (the sums of x_i^2 and d_i^2 are exact dot products, rounded once; n is
   converted to binary64 exactly)
*/
#ifndef CRNN_H
#define CRNN_H
#include <stddef.h>
#include <stdint.h>

void crnn_sigmoidf(float *y, const float *x, size_t n);
void crnn_siluf(float *y, const float *x, size_t n);
void crnn_geluf(float *y, const float *x, size_t n);
void crnn_softplusf(float *y, const float *x, size_t n);
void crnn_rsqrtf(float *y, const float *x, size_t n);

/* the same five in binary16 (f16) and bfloat16 (bf16), as bit patterns
   (uint16_t), in and out: correctly rounded to nearest in the format, from
   the same binary64 values and a table per format (crnn-exceptions16.h);
   crnn-check tries all 2^16 inputs of each (added 2026-10-01) */
void crnn_sigmoid_f16(uint16_t *y, const uint16_t *x, size_t n);
void crnn_silu_f16(uint16_t *y, const uint16_t *x, size_t n);
void crnn_gelu_f16(uint16_t *y, const uint16_t *x, size_t n);
void crnn_softplus_f16(uint16_t *y, const uint16_t *x, size_t n);
void crnn_rsqrt_f16(uint16_t *y, const uint16_t *x, size_t n);
void crnn_sigmoid_bf16(uint16_t *y, const uint16_t *x, size_t n);
void crnn_silu_bf16(uint16_t *y, const uint16_t *x, size_t n);
void crnn_gelu_bf16(uint16_t *y, const uint16_t *x, size_t n);
void crnn_softplus_bf16(uint16_t *y, const uint16_t *x, size_t n);
void crnn_rsqrt_bf16(uint16_t *y, const uint16_t *x, size_t n);

float crnn_logsumexpf(const float *x, size_t n);
void crnn_softmaxf(float *y, const float *x, size_t n);
void crnn_layernormf(float *y, const float *x, size_t n, const float *g, const float *b, float eps);
void crnn_rmsnormf(float *y, const float *x, size_t n, const float *g, float eps);

#endif
