# crdistr: crdist for Python

The normal distribution over NumPy arrays, from crnumerics' C library `libcrdist`: `erfinv`, `erfcinv`, `probit`
(the standard normal quantile: R's `qnorm`, SciPy's `ndtri`), `ndtr`, `log_ndtr`, `normpdf` and `logit`, correctly
rounded to nearest in binary32, binary16 and bfloat16, every input of each checked against MPFR on x86-64 and
aarch64; and samplers on a Philox4x32-10 stream. The results are the same bits on every machine, in any
floating-point environment.

```python
import numpy as np, crdistr
crdistr.probit(np.float32(0.975))                  # 1.959964, the float32 nearest the exact quantile
u = np.random.default_rng(1).random(10, dtype=np.float32)
z = crdistr.probit(u)                              # normal draws: the same everywhere for the same u
crdistr.erfinv(np.array([0.1, 0.5], np.float16))   # binary16 in, binary16 out, rounded once
crdistr.log_ndtr(np.float32(-40))                  # log Phi(-40) = -804.6084, where Phi underflows binary64
s = crdistr.Stream(seed=42)                        # Philox4x32-10, (seed, substream, index)
g = s.gamma(1000, shape=2.5)                      # the same bits on every machine
```

Also `ndtr` (Phi), `normpdf` and `logit`, correctly rounded like the inverses; and `Stream`'s `uniform`,
`normal`, `exponential`, `gamma`, `beta`, `chisquared`, `student_t` and `words`, whose bits are fixed by the
specification in `crdist.h` (the tests check the binding against its recorded fingerprints).

The format follows the input. float16 arrays use binary16, and ml_dtypes' `bfloat16` arrays use bfloat16; for
bfloat16 bit patterns given as `uint16`, use `erfinv_bf16`, `erfcinv_bf16` and `probit_bf16`. float32 uses
binary32. Anything else is converted to float32 first, rounded to nearest, so a float64 0.1 becomes float32's 0.1
before the function sees it.

The library is `libcrdist.so.0`, from crnumerics' `make install` (or `CRDIST_LIBRARY` naming it). The tests
(`make crdist-python-check` at crnumerics' root) compare results with mpmath, rounded once to each format.
Linux only for now. `pip install crdistr`, then `import crdistr`: the name is crdistr on PyPI and as a module,
"crdist" on PyPI being an unrelated project (the C library is still crdist: `libcrdist`, `crdist.h`).
