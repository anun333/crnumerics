# crlin: verified linear systems for Python

`lo, hi = crlin.solve(A, b)`: for every component, an interval that provably holds the exact solution of A x = b
(the binary64 values of A and b taken exactly), typically an ulp wide, at about 10 times the cost of
`numpy.linalg.solve`. Or `crlin.NotVerified`, when A is singular or too ill-conditioned for binary64. It never
returns an enclosure that misses the solution.

```python
import numpy as np, crlin
A = np.random.default_rng(1).standard_normal((500, 500)); b = np.ones(500)
lo, hi = crlin.solve(A, b)                    # numpy's inverse and its (threaded) BLAS do the heavy work
lo, hi = crlin.solve(A, b, mode="exact")      # asks nothing of the BLAS but binary64 arithmetic; slower
```

It never switches the rounding mode, so a threaded BLAS can't break the proof; every bound comes from exact sums
or from a priori error bounds in round-to-nearest. The method and its checks are in crnumerics' `lin/README.md`.
The library is `libcrlin.so.0` (`CRLIN_LIBRARY` names it); not on PyPI yet.
