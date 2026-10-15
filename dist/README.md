# crdist: the normal distribution, correctly rounded, and samplers that agree everywhere

`dist/crdist.h` (2026-10-12): the inverse error functions and the inverse
of the standard normal distribution function in binary32, each correctly
rounded to nearest on every input. Every entry point runs in the default
floating-point environment (it saves the caller's, sets round-to-nearest
without flush-to-zero, and restores the caller's on return), so neither a
rounding mode nor `-ffast-math`'s flush-to-zero can change a result.

```c
void crdist_erfinvf(float *y, const float *x, size_t n);    /* erf(y) = x,   -1 <= x <= 1 */
void crdist_erfcinvf(float *y, const float *x, size_t n);   /* erfc(y) = x,   0 <= x <= 2 */
void crdist_probitf(float *y, const float *p, size_t n);    /* Phi(y) = p,    0 <= p <= 1 */
void crdist_erfinv_f16(uint16_t *y, const uint16_t *x, size_t n);    /* and erfcinv_f16, probit_f16: binary16 */
void crdist_erfinv_bf16(uint16_t *y, const uint16_t *x, size_t n);   /* and erfcinv_bf16, probit_bf16: bfloat16 */

void crdist_ndtrf(float *y, const float *x, size_t n);      /* Phi(x), the normal distribution function */
void crdist_log_ndtrf(float *y, const float *x, size_t n);  /* log Phi(x), also where Phi underflows */
void crdist_normpdff(float *y, const float *x, size_t n);   /* e^(-x^2/2)/sqrt(2 pi) */
void crdist_logitf(float *y, const float *p, size_t n);     /* log(p/(1 - p)) */
/* ... and each with _f16 and _bf16 */

typedef struct { uint64_t seed, substream, index; } crdist_stream;   /* Philox4x32-10 */
void crdist_normalf(float *y, size_t n, crdist_stream *s);           /* and uniformf, exponentialf, */
void crdist_uniform(double *y, size_t n, crdist_stream *s);          /* binary64, and exponential */
void crdist_gammaf(float *y, size_t n, double shape, crdist_stream *s); /* betaf, chisquaredf, student_tf */
void crdist_poisson(int64_t *y, size_t n, double lam, crdist_stream *s);
void crdist_binomial(int64_t *y, size_t n, int64_t trials, double p, crdist_stream *s);
```

**The normal distribution's functions** (2026-10-13): `ndtr` (Φ, SciPy's
`ndtr`, R's `pnorm`; the survival function is `ndtr(-x)`, the negation
being exact), `log_ndtr` (log Φ), `normpdf` (the density) and `logit` (the
log-odds; its inverse, the logistic sigmoid, is crnn's `crnn_sigmoidf`),
correctly rounded in the same three formats. `log_ndtr` is the one for
tail probabilities: log Φ(−40) is −804.6084, though Φ(−40), about 4e−350,
is below even binary64's subnormals. Below −30 it is kept as −x²/2 plus the
rest of the asymptotic series. x² is exact, and −x²/2 is often exactly a
binary32 midpoint, so the rest decides the rounding. A test of its own then
compares the pair against the midpoints on either side, exactly but for
one rounding. Before that test, 32,069 inputs were left to the table; after
it, 23.

**Samplers** (2026-10-13): uniform, normal, exponential, gamma, beta,
chi-squared and Student t variates with the same bits on every machine,
from a Philox4x32-10 stream (seed, substream, index). A variate has no exact
value to round, so the bits are fixed by a specification instead
(`crdist.h`). It is a fixed sequence of operations, each correctly rounded
or exact: Philox words, crdist's binary32 `probit`, CORE-MATH's `log` and
`exp` in binary64, and IEEE arithmetic. Normals come from `probit`
(one word each, tails to 6.36); gamma is Marsaglia and Tsang's method
without the squeeze; beta works with the logs of two gammas, so tiny
shapes keep their ratio. NumPy's samplers call the platform's `log`, `exp`
and `pow` (61 calls in its `distributions.c`, checked 2026-10-13), so the
same seed need not give the same draws on another system.
`crdist-check` checks the specification and the distributions (below), and
records fingerprints of fixed streams that any other implementation of the
specification can check itself against: the first 4,096 normals of the
stream (1, 0, 0) hash (FNV-1a over the bits) to 0xe0b080312afb39d4.

**Binary64 uniform and exponential** (2026-10-15): `crdist_uniform` gives
(2m + 1) 2^-53 with m the top 52 bits of two words (exact, 2^52 values in
(0, 1)), and `crdist_exponential` the correctly rounded −log of it. These
are the draws a double-precision simulation needs, such as the Gillespie
algorithm in [examples/ssa-xplat](../examples/ssa-xplat/). In Python,
`Stream.uniform(n, dtype=np.float64)` and `Stream.exponential(n,
dtype=np.float64)`.

**Poisson and binomial** (2026-10-13): counts as `int64_t`, by inversion
for small means, and by Hörmann's transformed rejection above that: PTRS
for Poisson (as NumPy uses it) and BTRS for the binomial (as TensorFlow and
JAX use it). Their log k! is part of the specification: a table below 128
(each entry rounded once from MPFR), Stirling's series above. CORE-MATH's
`lgamma` isn't used, because it writes the C library's `signgam`.

**Proving a build on another machine** (2026-10-13): `crdist-check
hash-all` hashes every output of every function on all 2^32 inputs and
compares with the digests of the builds `crdist-check all` proved against
MPFR. Equal digests carry that proof to the build at hand, in minutes on a
few cores instead of hours. Builds that matched every digest:
- the proven builds themselves, 42432d0 and 85ef764 (cfarm421, gcc 14);
- board5 (x86-64, gcc 13);
- cfarm424 (aarch64, gcc 14), 21 seconds on 32 threads;
- cfarm104 (macOS 26 on an Apple M1, Apple clang 21, one thread, no
  OpenMP), 6 minutes.

So the proof of every input holds on x86-64 and aarch64 Linux with gcc 13
and 14 and on Apple Silicon macOS with Apple clang.

**binary16 and bfloat16** (2026-10-13): the same three functions with 16-bit
inputs and outputs, as bit patterns, correctly rounded to nearest in the
format. They use the same binary64 values, a rounding test per format
(`crdist-round16.h`) and a table per format (`crdist-exceptions16.h`), which
is empty: no input of either format is left undecided. `crdist-check` tries
all 2^16 inputs of each, in the default environment and under round-upward
with flush-to-zero: 0 differ from MPFR. Rounding the binary32 result again
to 16 bits would have been wrong on 5 inputs (2 each for binary16's erfinv
and erfcinv, 1 for bfloat16's erfcinv), the binary32 result landing on a
16-bit midpoint.

**Installed** (2026-10-13): `make install` puts `libcrdist.so.0`,
`libcrdist.a`, `crdist.h` and `crdist.pc` (pkg-config) under `PREFIX`, on
Linux. `make check` installs into a scratch directory and builds against it
through pkg-config, shared, static and C++ (`dist/test/install-check.c`).
`crdist_version()` gives the library's version.

**From Python** (`dist/python`, 2026-10-13): `pip install crdistr`, then
`import crdistr` (the name on PyPI and as a module, "crdist" on PyPI
being an unrelated project). `crdistr.erfinv`, `erfcinv`, `probit`,
`ndtr`, `log_ndtr`, `normpdf` and `logit` over NumPy arrays, the format
following the dtype (float32, float16, ml_dtypes' bfloat16), and
`crdistr.Stream(seed, substream, index)` with the samplers. `make
crdist-python-check` tests it against mpmath and the C fingerprints.

`probit` is the quantile function of the standard normal distribution
(R's `qnorm`, SciPy's `ndtri`), the transform behind inverse-CDF sampling:
a uniform draw in, a normal draw out. With a counter-based generator
(Philox, Threefry) the uniform draws are the same everywhere; through a
`probit` that is correctly rounded, so are the normal ones. No other
library we know of rounds these three correctly: CORE-MATH and RLIBM have
no inverse of `erf`, and the best vendor accuracy we found is Intel IPP's,
within an ulp (checked 2026-10-12 and 13).

**Special values:** `erfinv(±1) = ±inf`, `erfcinv(0) = +inf`,
`erfcinv(2) = -inf`, `probit(0) = -inf`, `probit(1) = +inf`; NaN outside
the domain; `erfinv(±0) = ±0`, `erfcinv(1) = probit(1/2) = +0`. `ndtr`
is +0 below −15 and 1 above 6; `log_ndtr(+inf) = +0`, and −0 for finite
x from 15 up; `normpdf(±inf) = +0`; `logit(0) = -inf`, `logit(1) = +inf`,
`logit(1/2) = +0`.

**How it is computed** (`crdist-fast.h` has the derivations):
- **The fast path:** two inverses in binary64. Up to 1/2, erfinv(x) is
  x·P(x²), with P of degree 13. In the tail, erfcinv(q) is a function of
  s = √(−log q), evaluated in six pieces of degree 14. The polynomials
  (`crdist-poly.h`) are interpolated at Chebyshev nodes in 60 digits by
  `fit.py` (mpmath). Every argument is formed exactly, so the three
  functions share the two inverses. The error is below 2^-50. Only IEEE
  operations and CORE-MATH's correctly rounded `log` go into it, so every
  machine computes the same bits.
- **The rounding test:** when the error interval can round two ways, the
  result comes from a table, `crdist-exceptions.h`. `gen-exceptions` makes
  it from all 2^32 inputs with MPFR: 20 entries for `erfinv`, 52 for
  `erfcinv` and 46 for `probit`. On 3 of them, plainly rounding the binary64
  value would be wrong. The table comes out byte-identical when generated on
  x86-64 with gcc 13 and 14 and on aarch64 with gcc 14.
- **The references:** MPFR has no inverse of `erf`, so `crdist-ref.c`
  decides each answer by comparisons. Whether the root lies above or below
  an exact number m is whether `erf(m)` (or `erfc(m)`, or `erfc(m/√2)` for
  `probit`) lies below or above the exact right-hand side. An enclosure from
  directed roundings settles that, with precision doubled from 64 bits
  until it does. The answer is the number whose two midpoints bracket the
  root. A bisection over the binary32 encodings, with no starting estimate,
  agrees.
- **The first version** (also 2026-10-12) started from Winitzki's
  approximation and took three to five Newton steps through CORE-MATH's
  `erf` or `erfc` and `exp`. It was within 2^-51, with a 51-entry table
  that came out byte-identical on x86-64 and aarch64, but it took 290 to
  880 ns an element.

**How it is checked** (`dist/test/check.c`, about a minute of CPU):
1. **The constants** in the fast path, against MPFR.
2. **The seven functions** against the references: the edge values and
   2^16 inputs at every exponent. With `make crdist-check-all`, every one of
   the 2^32 inputs: run on 2026-10-13 on cfarm421 (x86-64) and cfarm424
   (aarch64, a Neoverse N1), gcc 14.2.0, 0 differ for each function, and
   the control differs on 67,111,000 (2 h 27 min on 64 threads; 6 h 6 min on
   32). The same for ndtr, log_ndtr, normpdf and logit on the same hosts
   later that day (13 min on 64 threads; 27 min on 32). Each run has the
   kit's control.
3. **The references against the bisection,** including the 64 smallest
   subnormal inputs of `erfinv`, whose answers are subnormal.
4. **The table:** every entry is an input the fast path can't decide, and
   its result is MPFR's.
5. **A negative control:** a plain binary32 formula (Winitzki's start and a
   Newton step with the C library's `erff`, `erfcf` and `expf`) must differ
   from the references. It does, on about a quarter of the inputs.
6. **The environment:** under round-upward with flush-to-zero, and in
   place, every result is the default environment's, and the caller's
   environment is back afterwards.
7. **The bound:** the binary64 value lies within 2^-50 of the exact answer,
   decided against MPFR. `crdist-check bound-all` checks every input: run
   on 2026-10-13 on cfarm421, within 2^-50 on all 4,269,801,466 inputs
   inside the domains, and beyond 2^-51 on only 51, all of `probit` (6 h
   21 min on 64 threads).
8. **binary16 and bfloat16:** every input, as above.
9. **The samplers** (and, for Poisson and binomial, 27 cases and 221,184
   draws against the specification, chi-squared tests against the exact
   probabilities and mean tests, each with a failing control):
   - **The stream:** an independent Philox4x32-10 reproduces Random123's
     known answers, and the library's words match it on streams far apart.
   - **The specification:** every sampler, computed again with MPFR's
     `log`, `exp`, `sqrt` and the MPFR `probit`, gives the library's bits:
     17 cases, shapes from 0.05 to 10^6, 69,632 draws, 0 differ.
   - **The distributions:** KS tests at the 0.1% level against the exact
     distribution functions from MPFR (normal, exponential, gamma at four
     shapes, chi-squared), and beta's mean and t's mean and mean square.
     Each control, a neighbouring distribution, must fail, and does.
   - **The environment:** draws unchanged under round-upward with
     flush-to-zero.
   - **The fingerprints:** recorded for all 17 cases.

**Platforms:** Linux on x86-64 and aarch64 (gcc 13 or later), and macOS
on Apple Silicon (Apple clang, without OpenMP; CI's macos job runs the
check and the binding; `make install` installs crdist on Linux only).

**Speed** (a Ryzen 5 PRO 5650U, gcc 13.3, an element):
- **The inverses:** 36 to 41 ns on uniform inputs and 47 to 54 ns deep in
  the tails (Horner's rule, plus one `log` and a square root in the tail).
- **The functions of x:** `ndtr` 115 ns and `log_ndtr` 137 ns (CORE-MATH's
  `erfc` and `exp`), `normpdf` 14, `logit` 29.
- **The samplers:** uniform 7 ns, normal 57, exponential 31, gamma 119
  (shape 2.5) and 171 (0.3), beta 297, Student t 179.

**Files:** `crdist.h` (the interface), `crdist.c`, `crdist-fast.h`,
`crdist-poly.h` (generated: `make crdist-poly`, which needs mpmath; make the
tables again after it), `fit.py`, `crdist-exceptions.h` and
`crdist-exceptions16.h` (generated: `make crdist-exceptions` and `make
crdist-exceptions16`), `gen-exceptions.c` and `gen-exceptions16.c`,
`crdist16.c` and `crdist-round16.h` (the 16-bit formats), `crdist.pc.in`,
`crdist-sample.c` (the samplers), `crdist-ref.c` and `crdist-ref.h` (the
MPFR references), `test/check.c`, `test/install-check.c`, `python/`. The
library carries its own copies of CORE-MATH's `log`, `log1p`, `exp` and
`erfc`, made local to it.
