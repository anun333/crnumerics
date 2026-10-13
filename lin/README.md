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
```

From Python (`lin/python`): `lo, hi = crlin.solve(A, b)` over NumPy arrays, b a
vector or an n-by-k matrix,
with numpy's inverse and numpy's matmul doing the heavy work, or
`crlin.NotVerified`.

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

The rounding mode is never switched. Methods that round upward around BLAS
calls are silently wrong when the BLAS is threaded, because worker threads
keep the mode they started with. Here every bound comes from exact sums
rounded once in the direction needed, or from an a priori error bound
computed in round-to-nearest.

**Two modes** for I − R A:
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

**Several right-hand sides** share R and the bound on |I − R A|. In the
classical mode the columns then go through together, as matrix products:
- **Kept exact:** the residual, so the enclosures stay an ulp wide.
- **Matrix products with a priori bounds:** R times the residual and |I − R A|
  times the box. Their errors are of order n·u² of x, below its ulp.
- **Per column:** one column failing (its solution overflowing, say) fails
  alone; the others keep their enclosures.

At n = 400, 10 right-hand sides took 45 ms against numpy's 2.0 (23×) and 50
took 105 ms (47×). At n = 800, 100 took 573 ms (33×).

**Not the factorization:** Rump's method forms R·A, so it needs R itself.
An LU factorization would only save forming the inverse, about a third of
the classical mode's time at n = 800 (inverse 39 ms, the two products 39,
the exact O(n²) work 43). Doing without the inverse needs a different and
much heavier error analysis; not done.

**How it is checked** (`lin/test/check.c`, every section in both modes, on
the same systems; a few seconds):
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
   - Hilbert matrices from n = 2 to 14: verified to n = 11, failed beyond,
     never wrong. (To n = 8 until the box iteration computed every row in
     each round; it had stopped at the first failing row and stalled.)
3. **Any R:** 120 systems with another matrix's inverse or one spoiled by
   noise: 85 verify, 35 fail, and none is wrong.
4. **The environment:** the same results under round-upward with
   flush-to-zero, and the caller's mode back afterwards.
5. **Larger systems** (n = 100 to 400): verified, an ulp wide.
6. **Several right-hand sides:** 80 systems, 314 columns against the exact
   solutions, each column bit for bit what `crlin_solve` gives it alone. A
   column whose solution overflows fails alone, and the others stay finite
   and tight. That last test's first version accepted −∞ and +∞ as "kept",
   which hid a bug in the batched path; the Python tests caught it.

`make crlin-python-check` tests the binding: against exact fractions, a
singular system refused even when given an inverse, numpy's threaded BLAS,
and spoiled inverses.

**Not yet:** sparse and banded systems; eigenvalues; a first user (the
roadmap's intake rule); PyPI.
