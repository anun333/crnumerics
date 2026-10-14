# crlin: verified linear algebra for Python

`lo, hi = crlin.solve(A, b)`: for every component, an interval that provably holds the exact solution of A x = b
(the binary64 values of A and b taken exactly), typically an ulp wide, at about 10 times the cost of
`numpy.linalg.solve`. Or `crlin.NotVerified`, when A is singular or too ill-conditioned for binary64. It never
returns an enclosure that misses the solution.

```python
import numpy as np, crlin
A = np.random.default_rng(1).standard_normal((500, 500)); b = np.ones(500)
lo, hi = crlin.solve(A, b)                    # numpy's inverse and its (threaded) BLAS do the heavy work
lo, hi = crlin.solve(A, b, mode="exact")      # asks nothing of the BLAS but binary64 arithmetic; slower
lo, hi = crlin.solve(A, b, mode="extended")   # for condition numbers to about 1e30
```

The rest, each checked against exact rational answers in crnumerics' tests:

- `crlin.solve(A, B)` with B an n-by-k matrix: several right-hand sides together.
- `crlin.solve_interval(A_lo, A_hi, b_lo, b_hi)`: the solutions of every system whose entries lie in the given
  intervals, which also proves every such A nonsingular.
- `crlin.lstsq(A, b)`: the exact least-squares solution (A m by n, m >= n, proved of full rank), or for m < n the
  exact minimum-norm one.
- `crlin.eig_sym(A)`: an interval for every eigenvalue of a symmetric A, from numpy's `eigh`.
- `crlin.spd(A)`: a proof that a symmetric A is positive definite, and a lower bound on its smallest eigenvalue.
- `crlin.solve_banded((kl, ku), ab, b)` (scipy's banded layout) and `crlin.solve_sparse(A, b)` (scipy.sparse, or
  compressed rows): banded and sparse systems, through a proven bound on the smallest eigenvalue of A or A^T A.

It never switches the rounding mode, so a threaded BLAS can't break the proof; every bound comes from exact sums
or from a priori error bounds in round-to-nearest. The methods and their checks are in crnumerics' `lin/README.md`.
`pip install crlin` brings a wheel with the library, `libcrlin.so.0` (or `libcrlin.0.dylib` on macOS);
`CRLIN_LIBRARY` names another.
