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
   gemm: the binary64 GEMM crgemm_oz uses for the exact product R A
   (crsum.h; a BLAS's dgemm will do); NULL: crsum's internal one.

   Returns CRLIN_VERIFIED (0), CRLIN_FAILED (1), or -1 for n = 0 or a
   non-finite entry in A or b (nothing written). It runs in the default
   floating-point environment, set on entry, and the caller's is restored
   on exit, so flush-to-zero can't touch the bounds. */
#ifndef CRLIN_H
#define CRLIN_H
#include <stddef.h>
#include "crsum.h"

#define CRLIN_VERSION "0.5.0"
enum { CRLIN_VERIFIED = 0, CRLIN_FAILED = 1 };
#ifdef __cplusplus
extern "C" {
#endif
int crlin_solve(size_t n, const double *A, size_t lda, const double *b, const double *R, size_t ldr, double *x_lo,
                double *x_hi, crsum_dgemm gemm, void *ctx);
const char *crlin_version(void);
#ifdef __cplusplus
}
#endif
#endif
