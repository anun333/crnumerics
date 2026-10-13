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
   only its success depends on R being close (a poor one, whose bound on
   |I - R A| has a row sum of 1/2 or more, is improved once to
   fl(inv(fl(R A)) R) before the proof). NULL: crlin computes one (LU
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
     CRLIN_EXTENDED (2026-10-13): for very ill-conditioned systems, to a
       condition number of about 1e30. R is carried as a pair R1 + R2,
       from Rump's iteration for extremely ill-conditioned matrices (R ->
       inv(R A) R, R A computed exactly and R kept as its exact value's
       nearest rounding plus the rounded rest), until |I - R A| is small;
       x~ is a pair too, and every product exact. R given: the iteration's
       start. Several times CRLIN_EXACT's cost (each round about three
       exact products of inner dimension 2 n), column by column; not for
       crlin_solve_interval or crlin_spd (-1).
   gemm: the binary64 GEMM (crlin_dgemm below; a BLAS's dgemm will do);
   NULL: crsum's internal one.

   Returns CRLIN_VERIFIED (0), CRLIN_FAILED (1), or -1 for n = 0, a mode
   it doesn't know, or a non-finite entry in A or b (nothing written). It runs in the default
   floating-point environment, set on entry, and the caller's is restored
   on exit, so flush-to-zero can't touch the bounds.

   Threads: built with OpenMP (as the Makefile builds it on Linux), the
   work outside the GEMM (the exact residuals and enclosures, the bound's
   entries, the inverse when R is NULL) is split by rows among
   OMP_NUM_THREADS threads, from 64 rows. The results are the same bits
   for any number of threads: each row is computed the same way whichever
   thread takes it. Each thread sets the default floating-point
   environment itself and restores its own on leaving, since a pool's
   threads keep the mode they were created with. */
#ifndef CRLIN_H
#define CRLIN_H
#include <stddef.h>

#define CRLIN_VERSION "0.6.0"
enum { CRLIN_VERIFIED = 0, CRLIN_FAILED = 1 };
enum { CRLIN_EXACT = 0, CRLIN_CLASSICAL = 1, CRLIN_EXTENDED = 2 };
#ifdef __cplusplus
extern "C" {
#endif
/* the GEMM: C = A B, row-major (crsum.h's crsum_dgemm, the same type) */
typedef void (*crlin_dgemm)(size_t m, size_t n, size_t k, const double *A, size_t lda, const double *B, size_t ldb,
                            double *C, size_t ldc, void *ctx);
int crlin_solve(size_t n, const double *A, size_t lda, const double *b, const double *R, size_t ldr, double *x_lo,
                double *x_hi, int mode, crlin_dgemm gemm, void *ctx);
/* several right-hand sides: B is n by nrhs (row-major, leading dimension ldb), and so are X_lo and X_hi (ldx). R and
   the bound on |I - R A| are computed once; each column then costs about 3 n^2 exact products. CRLIN_VERIFIED when
   every column is; otherwise CRLIN_FAILED, and the columns that did verify still hold their enclosures, the others
   -inf and +inf. crlin_solve is this with one column. */
int crlin_solve_many(size_t n, size_t nrhs, const double *A, size_t lda, const double *B, size_t ldb, const double *R,
                     size_t ldr, double *X_lo, double *X_hi, size_t ldx, int mode, crlin_dgemm gemm, void *ctx);
/* interval data (2026-10-13): A's entries in [A_lo, A_hi] and B's in [B_lo, B_hi] (each lo <= hi, finite; else -1),
   both n by nrhs for B. CRLIN_VERIFIED: every A in [A_lo, A_hi] is nonsingular, and for every such A and every B in
   [B_lo, B_hi] each column of the solution lies in [X_lo, X_hi]. The same proof as crlin_solve's, on the midpoints:
   the residual's box is the exact range of b - A x~ over the intervals, each end one exact sum rounded outward, and
   the bound on |I - R A| covers every A in [A] by adding |R| rad(A) (the exact product rounded up, or in
   CRLIN_CLASSICAL one more GEMM with its a priori bound). The enclosure is close to the hull of the solutions when
   the radii are small next to A's distance from singular, and wider as they grow (in rad(A) |x| over that
   distance); an [A] that holds a singular matrix never verifies. R: an approximate inverse of the midpoint, or
   NULL. Column by column, as CRLIN_EXACT goes. */
int crlin_solve_interval(size_t n, size_t nrhs, const double *A_lo, const double *A_hi, size_t lda, const double *B_lo,
                         const double *B_hi, size_t ldb, const double *R, size_t ldr, double *X_lo, double *X_hi,
                         size_t ldx, int mode, crlin_dgemm gemm, void *ctx);
/* least squares (2026-10-13): A is m by n, B m by nrhs, X n by nrhs. m >= n: each column of X encloses the exact
   least-squares solution, the x minimizing ||A x - b||_2, and CRLIN_VERIFIED proves A of full column rank (so that
   x is unique). m < n: the exact minimum-norm solution of A x = b, A proved of full row rank. Through the augmented
   system [alpha I, A; A^T, 0] (m >= n; [alpha I, A^T; A, 0] for m < n), of order m + n, exact in binary64 for alpha
   a power of two, solved as crlin_solve_many solves it (any mode): so its cost is a solve of order m + n, heavy for
   m much larger than n. alpha: 0 for crlin's choice (near sigma_min(A) / sqrt 2, which keeps the augmented matrix
   about as well-conditioned as A, from fl(A^T A)'s Cholesky factor and inverse iteration), or a power of two (else
   -1). R: an approximate inverse of the augmented matrix with that alpha, or NULL. */
int crlin_lstsq(size_t m, size_t n, size_t nrhs, const double *A, size_t lda, const double *B, size_t ldb, double alpha,
                const double *R, size_t ldr, double *X_lo, double *X_hi, size_t ldx, int mode, crlin_dgemm gemm,
                void *ctx);
/* symmetric eigenvalues (2026-10-13): A symmetric (exactly, else -1). CRLIN_VERIFIED: lam_lo[k] <= lambda_k <=
   lam_hi[k] for A's k-th smallest eigenvalue lambda_k, k = 0 .. n-1. V (columns: approximate eigenvectors, leading
   dimension ldv) and d (approximate eigenvalues), from anywhere (LAPACK's dsyevd, say), or both NULL for crlin's own
   (the cyclic Jacobi method: fine to a few hundred). The proof: with G = V^T V - I and R = A V - V diag(d), each
   entry one exact product (of inner dimension 2 n), V^T A V = diag(d) + E with ||E||_2 <= ||G|| max |d_i| +
   ||V|| ||R|| = e, so its k-th smallest eigenvalue lies within e of d's k-th smallest (Weyl), and A's is that
   divided by some theta in [1 - ||G||_2, 1 + ||G||_2] (Ostrowski), with ||G||_2 < 1 needed. Every enclosure is
   about n u ||A|| wide, absolutely: tight for the eigenvalues near ||A||, wide relative to small ones. mode:
   CRLIN_EXACT or CRLIN_CLASSICAL (-1 for CRLIN_EXTENDED). */
int crlin_eig_sym(size_t n, const double *A, size_t lda, const double *V, size_t ldv, const double *d, double *lam_lo,
                  double *lam_hi, int mode, crlin_dgemm gemm, void *ctx);
/* positive definiteness (2026-10-13). A is symmetric (A[i][j] == A[j][i] exactly, else -1). CRLIN_VERIFIED: A is
   proved positive definite, every eigenvalue of A at least *lambda_lo > 0 (or 0, in the one case below where the
   bound underflows). CRLIN_FAILED: *lambda_lo is what could be proved, a lower bound on A's smallest eigenvalue that
   is not positive, or -inf when nothing was.
   The proof: for any matrix L and any number c, A = L L^T + c I + E with E = A - c I - L L^T, and L L^T has no
   negative eigenvalue, so A's smallest is at least c - ||E||_2 (Weyl). ||E||_2 is bounded by the largest row sum
   of a symmetric F >= |E|, with E enclosed as crlin_solve encloses I - R A (mode: CRLIN_EXACT, the exact product
   L L^T; CRLIN_CLASSICAL, two products and the a priori bound, a subnormal entry of L sending it to the exact one).
   L: an approximate Cholesky factor of A - c I (n by n, row-major, leading dimension ldl; it need not be
   triangular), from anywhere: only how small E is depends on it, and *lambda_lo is close to A's smallest
   eigenvalue when c is. NULL (c then ignored): crlin factors A itself (unblocked, threaded), finds an estimate of
   the smallest eigenvalue by inverse iteration, and factors A - c I with c half of it, so that *lambda_lo comes out
   near half the smallest eigenvalue. If that proof fails, it tries D A D, scaled by powers of two D = diag(2^e_i)
   to a diagonal near 1: exact (not tried if an entry would leave the normal range), and positive definite exactly
   when A is, which proves badly scaled matrices whose smallest eigenvalue is far below n u max A_ii. *lambda_lo is
   then D A D's bound times the least 2^(-2 e_i), a looser one, rounded down (0 if it underflows). For large A a
   factor from LAPACK is much faster than crlin's own. */
int crlin_spd(size_t n, const double *A, size_t lda, const double *L, size_t ldl, double c, double *lambda_lo, int mode,
              crlin_dgemm gemm, void *ctx);
const char *crlin_version(void);
#ifdef __cplusplus
}
#endif
#endif
