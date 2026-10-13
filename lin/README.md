# crlin: verified linear systems

`lin/crlin.h` (2026-10-13): solve A x = b and get back, for every component,
an interval that provably holds the exact solution x* of the system whose
entries are the binary64 values given. Or get told that it can't be proved,
because A is singular or too ill-conditioned for binary64. It never returns
an enclosure that misses x*.

```c
int crlin_solve(size_t n, const double *A, size_t lda, const double *b,
                const double *R, size_t ldr,           /* an approximate inverse, or NULL */
                double *x_lo, double *x_hi,            /* the enclosure */
                int mode,                              /* CRLIN_CLASSICAL or CRLIN_EXACT */
                crsum_dgemm gemm, void *ctx);          /* your BLAS's dgemm, or NULL */
/* CRLIN_VERIFIED (0), CRLIN_FAILED (1), or -1 for bad arguments */
int crlin_solve_many(size_t n, size_t nrhs, const double *A, size_t lda,   /* several right-hand sides */
                     const double *B, size_t ldb, const double *R, size_t ldr,
                     double *X_lo, double *X_hi, size_t ldx, int mode, crsum_dgemm gemm, void *ctx);
int crlin_solve_interval(size_t n, size_t nrhs,        /* interval data: every A in [A], b in [b] */
                         const double *A_lo, const double *A_hi, size_t lda,
                         const double *B_lo, const double *B_hi, size_t ldb, const double *R, size_t ldr,
                         double *X_lo, double *X_hi, size_t ldx, int mode, crsum_dgemm gemm, void *ctx);
int crlin_lstsq(size_t m, size_t n, size_t nrhs,       /* least squares (m >= n), minimum norm (m < n) */
                const double *A, size_t lda, const double *B, size_t ldb, double alpha,
                const double *R, size_t ldr, double *X_lo, double *X_hi, size_t ldx,
                int mode, crsum_dgemm gemm, void *ctx);
int crlin_spd(size_t n, const double *A, size_t lda,   /* A symmetric: proved positive definite? */
              const double *L, size_t ldl, double c,   /* a Cholesky factor of A - c I, or NULL */
              double *lambda_lo,                       /* every eigenvalue of A is at least this */
              int mode, crsum_dgemm gemm, void *ctx);
```

Installed by `make install` (Linux): `libcrlin.so.0`, `libcrlin.a`, `crlin.h`
and `crlin.pc` for pkg-config; the library exports `crlin_` names only, its
copy of crsum local to it. From Python (`lin/python`, a wheel by
`lin/python/build-wheel.sh`, `crlin-wheels.yml` for PyPI): `lo, hi =
crlin.solve(A, b)` over NumPy arrays, b a vector or an n-by-k matrix,
with numpy's inverse and numpy's matmul doing the heavy work, or
`crlin.NotVerified`; `crlin.solve_interval(A_lo, A_hi, b_lo, b_hi)` for
interval data, `crlin.lstsq(A, b)` for least squares, `lam = crlin.spd(A)`
for positive definiteness.

**Why:** a dense solve whose answer you can trust to the last bit, without
rerunning it in higher precision. The gap checked on 2026-10-12: INTLAB's
`verifylss` is MATLAB, with a licence for commercial products; C-XSC's last
release was in 2014; from Python the rigorous route was FLINT/Arb's
`arb_mat_solve`, 140 to 450 times numpy's `solve`.

**How:** Rump's method (1983).
1. An approximate inverse R and solution x~ come from ordinary floating
   point, from anywhere: the proof holds for any R, and only whether it
   succeeds depends on R.
2. The exact residual b − A x~ and the exact I − R A are enclosed using
   crsum's exact sums and products, each rounded once.
3. A box Y is found with |R(b − A x~)| + |I − R A|·Y < Y in every
   component. That proves A nonsingular and puts x* inside x~ + R(b − A x~)
   + (I − R A)·Y, each end rounded outward.

**A poor R** is improved once before the proof: when the bound on
|I − R A| has a row sum of 1/2 or more, R becomes fl(inv(fl(R A))·R). That
leaves |I − R A| near u·cond(A) whatever R's own errors were. No proof
rests on R, so this only decides how many systems verify, and a system
whose first R passes is left as it was, bit for bit. It was added for
least squares (below), where LU's inverse of the augmented matrix left row
sums near 10 at a condition number of 10^9, and one round brought them to
10^−7. It also lifted the Hilbert matrices one size (to n = 12 in
`CRLIN_EXACT`), and every spoiled R in the checks now verifies.

The rounding mode is never switched. Methods that round upward around BLAS
calls are silently wrong when the BLAS is threaded, because worker threads
keep the mode they started with. Here every bound comes from exact sums
rounded once in the direction needed, or from an a priori error bound
computed in round-to-nearest.

**Three modes** for I − R A:
- **`CRLIN_CLASSICAL`** (the Python default): two binary64 products, R A
  and |R||A|, plus the a priori bound γ_(n+1)·|R||A| + 2(n+1)·2^−1022.
  - **What it asks of the GEMM:** each element must be a sum of products
    rounded to nearest, in any order, with or without fused multiply-adds,
    threaded or not. Any BLAS's `dgemm` qualifies; Strassen-like algorithms
    and lower-precision modes don't.
  - **Flushing:** the absolute term assumes any product or partial sum
    below 2^−1022 may be flushed to zero, as a worker thread with
    flush-to-zero would do.
  - **Subnormal inputs:** a subnormal entry in R or A sends the solve to
    the exact product instead, since denormals-are-zero would zero it
    before multiplying.
- **`CRLIN_EXACT`**: the exact product, through crsum's Ozaki-scheme
  `crgemm_oz`. It asks nothing of the GEMM but binary64 arithmetic, but
  costs about 35 GEMMs.
- **`CRLIN_EXTENDED`**: for systems too ill-conditioned for binary64, to a
  condition number of about 10^30. R is carried as a pair of doubles,
  R₁ + R₂, from Rump's iteration for extremely ill-conditioned matrices
  (2009).
  - **The iteration:** R ← inv(R A)·R, with R A computed exactly and
    rounded once, and the new R kept exactly as its nearest rounding plus
    the rounded rest. Each round gains about 1/u in the condition number
    that R A is left with, and it stops when |I − R A| is small.
  - **The rest of the solve:** x̃ is a pair too, refined with the exact
    residual, and every product is exact (the pair's two terms as one
    product of inner dimension 2n).
  - **What it proves:** Hilbert matrices, rounded to binary64, verify to
    n = 24, where the other modes stop at n = 11. Integer matrices of
    determinant 1 verify to condition numbers past 10^31.
  - **Cost:** 1.2 to 1.5 times `CRLIN_EXACT` on a well-conditioned system,
    where one round suffices. On ill-conditioned ones it is 6 to 8 times:
    1.1 s at n = 400 with a condition number near 10^21, on board5 under
    load.
  - **Not supported:** interval data and `crlin_spd` refuse it.

**Measured** on board5 (Ryzen 5 PRO 5650U), one thread, random
well-conditioned systems (`openpocl/harness/measure/crlin-bench.py`, the
machine's load about 4 from other work). The crlin times include numpy's
inverse; numpy's `solve` is unverified:

| n | numpy `solve` | crlin classical | crlin exact | Arb `arb_mat_solve` |
|---|---|---|---|---|
| 100 | 0.21 ms | 2.1 ms (10×) | 11.6 ms (56×) | 27.9 ms (134×) |
| 200 | 0.38 ms | 4.4 ms (12×) | 38.9 ms (103×) | 163 ms (430×) |
| 400 | 2.08 ms | 20.8 ms (10×) | 179 ms (86×) | 882 ms (424×) |
| 800 | 11.0 ms | 120 ms (11×) | 893 ms (81×) | |
| 1600 | 79.1 ms | 880 ms (11×) | 5.28 s (67×) | |

crlin's enclosures were 2.2×10^−16 wide relative to |x| (one ulp) in every
case; Arb's were 5×10^−15 to 7×10^−15 (53-bit balls).

**Threads** (OpenMP, in the Linux builds): the work outside the GEMM is split
by rows among `OMP_NUM_THREADS` threads, from 64 rows. That covers the exact
residuals, R times them, the box iteration, the bound's entries, and the
inverse when crlin computes it. Each row is computed the same way whichever
thread takes it, so the enclosures are the same bits for any number of
threads. Measured on cfarm421 (AMD EPYC 7773X, 64 cores, nice 19; other
users' load rose from 0.2 to about 7 during the run), `crlin_solve` given
numpy's inverse, classical mode, numpy's matmul as the GEMM
(`openpocl/harness/measure/crlin-threads.py`; every run's bits the same):

| n | one thread | BLAS on 32 threads, crlin on one | both on 32 |
|---|---|---|---|
| 1600 | 520 ms | 191 ms | 103 ms |
| 3200 | 3.53 s | 734 ms | 342 ms |

With numpy's inverse included, crlin took 10 to 13 times numpy's `solve` on
one thread, and 2.4 to 4.6 times on 16 or 32. numpy's own threaded `solve`
scales poorly, which flatters that ratio. `CRLIN_EXACT` gains less, 46 s to
9.5 s at n = 3200: crgemm_oz's exact accumulation still runs on one thread.

**Several right-hand sides** share R and the bound on |I − R A|. In the
classical mode the columns then go through together, as matrix products:
- **Kept exact:** the residual, so the enclosures stay an ulp wide.
- **Matrix products with a priori bounds:** R times the residual and |I − R A|
  times the box. Their errors are of order n·u² of x, below its ulp.
- **Per column:** one column failing (its solution overflowing, say) fails
  alone; the others keep their enclosures.

At n = 400, 10 right-hand sides took 45 ms against numpy's 2.0 (23×) and 50
took 105 ms (47×). At n = 800, 100 took 573 ms (33×).

**Interval data** (`crlin_solve_interval`, `crlin.solve_interval`): A's
entries and b's given as intervals, from measurement error, say. The result
encloses the solution of every system inside them, and proves every A in
[A] nonsingular.
- **The proof:** the same as for point data, with the residual's box the
  exact range of b − A x̃ over the intervals, each end one exact sum
  rounded outward. The bound on |I − R A| gains |R|·rad(A): the exact
  product rounded up, or one more GEMM with its a priori bound.
- **Tightness:** close to the hull of the solutions when the radii are
  small next to A's distance from singular, wider as they grow. Against the
  exact hull, computed from every vertex for n ≤ 3, enclosures were at most
  1.33 times its width. An [A] holding a singular matrix never verifies.
- **Cost:** about twice the point solve. At n = 800 on board5 under load,
  166 to 197 ms against 90 ms. With radii of 2^−40 relative, the median
  width was 2×10^−10 relative: summed over 800 entries a row, the radii
  move x that much.

**Least squares** (`crlin_lstsq`, `crlin.lstsq`): for m ≥ n, the exact
least-squares solution, the x minimizing ‖A x − b‖₂, with A proved of full
column rank; for m < n, the exact minimum-norm solution of A x = b.
- **The system:** for m ≥ n it is the augmented system [α I, A; Aᵀ, 0]
  [r/α; x] = [b; 0], of order m + n; for m < n, [α I, Aᵀ; A, 0]. It is
  exact in binary64 for α a power of two, so the solve's enclosure holds
  the exact solution, in any mode, extended included.
- **The scaling:** α near σ_min(A)/√2 keeps the augmented matrix about as
  well-conditioned as A. crlin estimates σ_min from fl(Aᵀ A)'s Cholesky
  factor and inverse iteration; the Python binding takes it from numpy's
  singular values.
- **Cost:** a solve of order m + n, heavy when m is much larger than n. On
  board5 under load: 42 times numpy's unverified `lstsq` at 200 by 50,
  5 times at 800 by 200 and at 300 by 300; enclosures an ulp wide.

**Positive definiteness** (`crlin_spd`, `crlin.spd`): a proof that a
symmetric A is positive definite, with a lower bound on its smallest
eigenvalue.
- **The argument:** for any matrix L and any number c, A = L Lᵀ + c I + E
  with E = A − c I − L Lᵀ. L Lᵀ has no negative eigenvalue, so A's smallest
  is at least c − ‖E‖₂ (Weyl). ‖E‖₂ is at most the largest row sum of a
  symmetric bound F ≥ |E|, and E is enclosed as I − R A is in the solve,
  in either mode. So the proof holds for any L, from anywhere; only how
  small E comes out depends on it being a good Cholesky factor of A − c I.
- **The bound's size:** it is close to the smallest eigenvalue when c is.
  Without a factor given, crlin factors A, estimates the smallest
  eigenvalue by inverse iteration, and factors again with c half of it, so
  the bound comes out near half the smallest eigenvalue.
- **Badly scaled matrices:** if that fails, crlin tries D A D, scaled by
  powers of two to a diagonal near 1. That is exact, and positive definite
  exactly when A is, so it proves matrices whose smallest eigenvalue lies
  far below n·u·max A_ii. The bound for A is then looser.
- **Cost:** at n = 800 on board5, under load, 63 ms with a LAPACK factor
  of A − c I passed in (numpy's `eigvalsh`, unverified, took 38). crlin's
  own factor is unblocked, and its default took 191 ms.

**Not the factorization:** Rump's method forms R·A, so it needs R itself.
An LU factorization would only save forming the inverse, about a third of
the classical mode's time at n = 800 (inverse 39 ms, the two products 39,
the exact O(n²) work 43). Doing without the inverse needs a different and
much heavier error analysis; not done.

**How it is checked** (`lin/test/check.c`, every section in each of the
three modes, on the same systems, except 8 and 9, which extended mode
doesn't apply to; about half a minute):
1. **Containment:** 516 systems from n = 1 to 64 against the exact rational
   solution. Each row is scaled to integers (the solution doesn't change)
   and solved by fraction-free Gauss–Jordan elimination in GMP's integers,
   checked exactly against A x = b. The systems are random, small-integer,
   triangular, and rows of wildly different scales.
   - **Result:** every one verified, and none missed x*. The widest
     enclosure was 3.9×10^−16 relative.
   - **The control:** shrunk by one ulp at one end, 506 of the 516 miss x*,
     so the comparison can see a miss and the enclosures are about as tight
     as binary64 allows.
2. **Never wrong:**
   - 100 exactly singular systems (rank-deficient, a repeated row, a zero
     row): none verified.
   - Hilbert matrices from n = 2 to 24: verified to n = 12 in
     `CRLIN_EXACT` and to 11 in `CRLIN_CLASSICAL`, failed beyond, never
     wrong; in `CRLIN_EXTENDED` all 23 verify. (To n = 8 until the box
     iteration computed every row in each round; it had stopped at the
     first failing row and stalled.)
3. **Any R:** 120 systems with another matrix's inverse, one spoiled by
   noise, or a singular one (zero, or rank one). Every singular R fails, as
   it must, since R A is then singular; the other 108 verify, the poor R
   improved first, and none is wrong. Before that improvement 35 of the
   120 failed.
4. **The environment:** the same results under round-upward with
   flush-to-zero, and the caller's mode back afterwards.
5. **Larger systems** (n = 100 to 400): verified, an ulp wide.
6. **Several right-hand sides:** 80 systems, 314 columns against the exact
   solutions, each column bit for bit what `crlin_solve` gives it alone. A
   column whose solution overflows fails alone, and the others stay finite
   and tight. That last test's first version accepted −∞ and +∞ as "kept",
   which hid a bug in the batched path; the Python tests caught it.
7. **Threads:** systems from n = 64 to 300 give the same bits with 1, 3
   and 4 threads.
   - **A poisoned pool:** they also give the same bits with every thread of
     OpenMP's pool left in round-upward with flush-to-zero, and each thread
     keeps its own mode afterwards. A pool's threads keep the mode they
     were created with, so each thread sets the default itself.
   - **The control:** a build whose threads don't set their mode gives
     different bits in 3 of the 16 solves, all nearly singular, where the
     last bits follow R's. A well-conditioned system's enclosure is the same
     ulp around x* whatever R was, and the first version of this test, with
     only those, passed that build. 2 of the 4 positive-definiteness bounds
     (below) differ there too.

8. **Positive definiteness:** every bound is checked exactly. A − λ_lo I
   must be positive definite by symmetric elimination in rationals, every
   pivot positive. The cases, 313 matrices in each mode:
   - random positive definite ones, some scaled over 60 binades;
   - Hilbert matrices to n = 14;
   - exactly singular and indefinite ones;
   - a factor given: good, spoiled by noise, or random.

   No bound is wrong, and none of the 78 that are not positive definite
   verified. Without the scaled second try, 34 of the 40 scaled matrices
   failed; with it, all verify. Hilbert matrices verify to n = 11. Rounded
   to binary64, n = 12 and 13 are still positive definite but fail, and
   n = 14 is not positive definite at all (the exact test says so). **The
   control:** for 141 of the 213 verified, A − 4 λ_lo I is not positive
   definite, so the bound is near the smallest eigenvalue and the exact
   test can see a failure.

9. **Interval data:** 200 systems in each mode.
   - **Small ones** (n ≤ 3): every vertex of [A] and [b] is solved exactly,
     which gives the hull of the solutions, since its ends lie at vertices.
   - **Larger ones** (n = 4 to 20): sampled vertices and points inside.

   All 167,520 of those exact solutions lie inside their enclosures. None
   of 41 interval matrices holding a singular matrix verifies. That test's
   first version built some 2-by-2 matrices that were not singular, 5 of
   which verified; the n = 2 case is now excluded. **The control:** in 73
   systems, an enclosure shrunk by 1% of its width at one end loses some
   vertex's solution.

10. **Very ill-conditioned:** integer matrices of determinant 1, built by
    random row operations until their entries reach 2^20 to 2^52, so that
    they stay exact in binary64. Their condition numbers, computed exactly
    from the integer inverse, run from 10^10 to past 10^35.
    - **Extended mode:** all 48 to 2^46 verify, 21 of the 24 beyond.
    - **The other modes** verify only those under about 10^16.
    - **In all three,** no enclosure misses the exact solution.

11. **Least squares:** 155 problems in each mode, m from 1 to 16 and n
    from 1 to 10. For m ≥ n each is checked against the exact solution of
    the normal equations, solved in rationals; for m < n against the exact
    A^T (A A^T)^−1 b.
    - **Results:** none misses. All 19 rank-deficient ones fail, and all
      17 with nearly dependent columns (condition numbers near 5×10^8)
      verify.
    - **Before the R improvement,** 7 of those 17 failed in the first two
      modes, at any α.
    - **The control:** every verified enclosure, shrunk by an ulp at one
      end, misses.

`make crlin-python-check` tests the binding: against exact fractions, a
singular system refused even when given an inverse, numpy's threaded BLAS,
and spoiled inverses.

**Not yet:** sparse and banded systems; eigenvalue enclosures; least
squares through a QR factorization (for m much larger than n); a first user (the
roadmap's intake rule); PyPI (the workflow is ready; the upload waits on a
trusted publisher and a release).
