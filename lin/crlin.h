/* crlin.h: verified linear systems (added 2026-10-13; ROADMAP.md, item 11).
   binary64, row-major: row i of a matrix M starts at M + i * ldm.

   crlin_solve encloses the solution of A x = b: on success, every
   component of the exact solution x* (of the system whose entries are
   the binary64 values of A and b, taken exactly) lies in [x_lo[i],
   x_hi[i]], and A is proved nonsingular. When the proof does not go
   through (A singular, or too ill-conditioned for binary64, roughly a
   condition number past 1e15), it says so: CRLIN_FAILED, with x_lo = -inf
   and x_hi = +inf. It never returns an enclosure that misses x*.

   The method is Rump's (1983): an approximate inverse R and solution x~,
   then enclosures of the exact residual b - A x~ and of the exact I - R A,
   and a box Y with |R (b - A x~)| + |I - R A| Y < Y componentwise; that
   proves A nonsingular and puts x* in x~ + R (b - A x~) + (I - R A) Y.
   Every enclosure comes from crsum's exact sums and products (crgemv,
   crdot, crgemm_oz) rounded once, then widened by an ulp, or rounded up
   where an upper bound is needed. So nothing depends on the rounding mode,
   which is never changed: a threaded BLAS's worker threads, which keep the
   mode they started with, can't break the proof as they break methods that
   switch to rounding upward around BLAS calls.

   R: an approximate inverse of A (n by n, row-major, leading dimension
   ldr), from anywhere (LAPACK, numpy, a GPU): the proof holds for any R,
   only its success depends on R being close. NULL: crlin computes one (LU
   with partial pivoting, unblocked: fine to a few hundred, slow beyond).
   mode: how I - R A is enclosed.
     CRLIN_EXACT: from the exact product R A (crgemm_oz, through gemm),
       rounded to nearest and widened by an ulp. It asks nothing of gemm
       but binary64 arithmetic (crgemm_oz's slices keep every value it
       computes an integer below 2^53), and costs about 35 GEMMs.
     CRLIN_CLASSICAL: from two products in binary64, M = fl(R A) and
       S = fl(|R| |A|), and the a priori bound |M - R A| <= g_(n+1) |R| |A|
       + 2 (n + 1) t, |R| |A| <= (S + 2 n t)/(1 - g_n), g_n = n u/(1 - n u),
       u = 2^-53, t = 2^-1022 (each rounding u relative, and any product or
       partial sum below t possibly lost, as flush-to-zero in a BLAS's
       worker threads would lose it; the bound holds for any order of the n
       products' sum). A subnormal entry in R or A, which
       denormals-are-zero would zero before multiplying, sends the solve to
       CRLIN_EXACT's product instead. Every step of the bound rounds to
       nearest and then steps an ulp up. It asks that gemm compute each
       element as a sum of products in binary64, rounding to nearest, in
       any order, with fused multiply-adds or not, threaded or not: any
       BLAS's dgemm, but not a Strassen-like algorithm or a lower-precision
       mode. Two GEMMs: about 10 times an unverified solve, where
       CRLIN_EXACT is about 70.
   gemm: the binary64 GEMM (crsum.h's crsum_dgemm; a BLAS's dgemm will do);
   NULL: crsum's internal one.

   Returns CRLIN_VERIFIED (0), CRLIN_FAILED (1), or -1 for n = 0, a mode
   it doesn't know, or a non-finite entry in A or b (nothing written). It runs in the default
   floating-point environment, set on entry, and the caller's is restored
   on exit, so flush-to-zero can't touch the bounds. */
#ifndef CRLIN_H
#define CRLIN_H
#include <stddef.h>
#include "crsum.h"

#define CRLIN_VERSION "0.5.0"
enum { CRLIN_VERIFIED = 0, CRLIN_FAILED = 1 };
enum { CRLIN_EXACT = 0, CRLIN_CLASSICAL = 1 };
#ifdef __cplusplus
extern "C" {
#endif
int crlin_solve(size_t n, const double *A, size_t lda, const double *b, const double *R, size_t ldr, double *x_lo,
                double *x_hi, int mode, crsum_dgemm gemm, void *ctx);
/* several right-hand sides: B is n by nrhs (row-major, leading dimension ldb), and so are X_lo and X_hi (ldx). R and
   the bound on |I - R A| are computed once; each column then costs about 3 n^2 exact products. CRLIN_VERIFIED when
   every column is; otherwise CRLIN_FAILED, and the columns that did verify still hold their enclosures, the others
   -inf and +inf. crlin_solve is this with one column. */
int crlin_solve_many(size_t n, size_t nrhs, const double *A, size_t lda, const double *B, size_t ldb, const double *R,
                     size_t ldr, double *X_lo, double *X_hi, size_t ldx, int mode, crsum_dgemm gemm, void *ctx);
const char *crlin_version(void);
#ifdef __cplusplus
}
#endif
#endif
