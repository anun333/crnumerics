# crdist for Python (`pip install crdistr`)

`erfinv`, `erfcinv` and `probit` (the standard normal quantile: R's `qnorm`, SciPy's `ndtri`) over NumPy arrays,
correctly rounded to nearest in binary32, binary16 and bfloat16, from crnumerics' C library `libcrdist`. Every
input of every function in each format was checked against MPFR, on x86-64 and aarch64. The results are the same
bits on every machine, in any floating-point environment.

```python
import numpy as np, crdist
crdist.probit(np.float32(0.975))                  # 1.959964, the float32 nearest the exact quantile
u = np.random.default_rng(1).random(10, dtype=np.float32)
z = crdist.probit(u)                              # normal draws: the same everywhere for the same u
crdist.erfinv(np.array([0.1, 0.5], np.float16))   # binary16 in, binary16 out, rounded once
```

The format follows the input. float16 arrays use binary16, and ml_dtypes' `bfloat16` arrays use bfloat16; for
bfloat16 bit patterns given as `uint16`, use `erfinv_bf16`, `erfcinv_bf16` and `probit_bf16`. float32 uses
binary32. Anything else is converted to float32 first, rounded to nearest, so a float64 0.1 becomes float32's 0.1
before the function sees it.

The library is `libcrdist.so.0`, from crnumerics' `make install` (or `CRDIST_LIBRARY` naming it). The tests
(`make crdist-python-check` at crnumerics' root) compare results with mpmath, rounded once to each format.
Linux only for now. On PyPI it is `crdistr` ("crdist" there is an unrelated project); the module is `crdist`.
