# crdist: the inverse normal distribution, correctly rounded

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
```

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
the domain; `erfinv(±0) = ±0`, `erfcinv(1) = probit(1/2) = +0`.

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
2. **The three functions** against the references: the edge values and
   2^16 inputs at every exponent. With `make crdist-check-all`, every one of
   the 2^32 inputs: run on 2026-10-13 on cfarm421 (x86-64) and cfarm424
   (aarch64, a Neoverse N1), gcc 14.2.0, 0 differ for each function, and
   the control differs on 67,111,000 (2 h 27 min on 64 threads; 6 h 6 min on
   32). Each run has the kit's control.
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
   decided against MPFR. `crdist-check bound-all` checks every input.

**Speed:** 36 to 41 ns an element on uniform inputs and 47 to 54 ns deep in
the tails, measured on a Ryzen 5 PRO 5650U with gcc 13.3: Horner's rule,
plus one `log` and a square root in the tail.

**Files:** `crdist.h` (the interface), `crdist.c`, `crdist-fast.h`,
`crdist-poly.h` (generated: `make crdist-poly`, which needs mpmath; make the
table again after it), `fit.py`, `crdist-exceptions.h` (generated: `make
crdist-exceptions`), `gen-exceptions.c`, `crdist-ref.c` and `crdist-ref.h`
(the MPFR references), `test/check.c`. The library carries its own copy of
CORE-MATH's `log`, made local to it.
