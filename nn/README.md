# crnn: neural-network primitives with one answer

`nn/crnn.h` (2026-09-30): the functions where machine-learning code
usually keeps fast approximations, in binary32, each with one specified
result. Every entry point runs in the default floating-point environment
(it saves the caller's, sets round-to-nearest without flush-to-zero, and
restores the caller's on return), so neither a rounding mode nor
`-ffast-math`'s flush-to-zero can change a result.

```c
void crnn_sigmoidf(float *y, const float *x, size_t n);   /* and siluf, geluf, softplusf, rsqrtf */
float crnn_logsumexpf(const float *x, size_t n);
void crnn_softmaxf(float *y, const float *x, size_t n);
void crnn_layernormf(float *y, const float *x, size_t n, const float *g, const float *b, float eps);
void crnn_rmsnormf(float *y, const float *x, size_t n, const float *g, float eps);
```

**One argument, correctly rounded** (to nearest): sigmoid 1/(1 + e^−x),
SiLU x·sigmoid(x), GELU (x/2)(1 + erf(x/√2)), softplus log(1 + e^x), and
rsqrt.
- **The fast path:** binary64 from CORE-MATH's correctly rounded `exp`,
  `log1p`, `erfc` and `rsqrt`, with a proven error below 2^−51 (the
  derivations are in `crnn-fast.h`).
  - GELU keeps x/√2 to about 2^−104 as a double-double, because `erfc`
    magnifies an input error by 2t².
  - SiLU and GELU at |x| < 2^−120 are x/2 plus a positive term below
    binary64's reach, and x/2 is often a binary32 midpoint. They round with
    the tie broken upward, which is where the exact value lies.
- **The rounding test:** when the error interval can round two ways, the
  result comes from a table, `crnn-exceptions.h`. `gen-exceptions` makes it
  with MPFR from all 2^32 inputs: 355 entries for sigmoid, 24 for SiLU, 15
  for GELU, 10 for softplus and 127 for rsqrt. On 66 of sigmoid's entries,
  plainly rounding the fast path's value would be wrong. The whole table
  comes out byte-identical when generated on x86-64 (an EPYC 7773X) and on
  aarch64 (a Neoverse N1), as it must: the fast paths are correctly rounded
  operations throughout.

**binary16 and bfloat16** (2026-10-01): the same five functions with
16-bit inputs and outputs, as bit patterns, correctly rounded to nearest in
the format: `crnn_sigmoid_f16(uint16_t *y, const uint16_t *x, size_t n)`
and `crnn_sigmoid_bf16(...)`, likewise for the other four. The same
binary64 fast paths, a rounding test per format (`crnn-round16.h`), and a
table from MPFR over every input (`crnn-exceptions16.h`): binary16 needs no
entries at all; bfloat16 needs 256 each for SiLU and GELU, the tiny inputs
whose halves are bfloat16 midpoints. `crnn-check` tries all 2^16 inputs of
each, in the default environment and under round-upward with flush-to-zero.
With the table skipped, 128 inputs of each of those two come out wrong, so
the check sees it.

**Composites, specified bit for bit:** a fixed sequence of correctly
rounded binary64 operations and crsum's exact sums, each rounded once.
`crnn.h` spells each one out. The result is the same in any order of the
elements and on any machine. It isn't promised to be the correctly rounded
value of the formula, but the check measures how close it comes.
- **logsumexp:** m + log1p(T), where T is the exact sum of e^(x_i−m) minus
  1, rounded once. The first version took log of the rounded sum. When the
  sum is 1 plus terms far below it, that lost almost everything: 9.5
  million ulp on the check's "swamped" vectors.
- **softmax:** e^(x_i−m) over the exact sum, rounded once.
- **layernorm:** the mean is kept as two binary64 numbers: the exact sum
  rounded once, its remainder rounded once, and the division's remainder
  from an `fma`. With one rounded mean, data with a large mean and a small
  spread (1e4 ± 0.01) lost up to |mean|/|x − mean| times 2^−53: 30 ulp in
  the result.
- **rmsnorm:** the exact sum of squares, rounded once.

**How it is checked** (`nn/test/check.c`, about 2 s on three cores):
1. **The constants** in the fast paths, against MPFR.
2. **The one-argument functions** against MPFR references (`crnn-ref.c`,
   Ziv loops). The sample is the edge values and 2^18 inputs at every
   exponent. With `make crnn-check-all`, every one of the 2^32 inputs:
   0 differ for each function, and the control differs on 67,111,000.
   This ran on aarch64 (cfarm424, a Neoverse N1: 22 minutes on 32
   threads) and on x86-64 (cfarm420, an EPYC 7773X: 18 minutes), both on
   2026-10-01. On each, the table was regenerated identical to the
   committed one. Rerun later that day at `c47566b` (after the vector path
   and the binary16 and bfloat16 versions) on cfarm421: the same.
3. **The table:** every entry is an input the fast path can't decide, and
   its result is MPFR's.
4. **The environment:** under round-upward with flush-to-zero, every
   function and composite gives the bits it gives in the default
   environment, and the caller's environment comes back. The control: the
   bare fast paths there differ on 117,919 of the inputs.
5. **The composites:** 891,600 results against an independent computation
   of the specification (MPFR's `exp`, `log1p` and `rsqrt`, and
   `mpfr_sum`), 0 differ.
   - Lengths 1 to 4,097; vectors that are normal, wide, cancelling, huge,
     tiny, offset, constant, subnormal, swamped, or with NaN and
     infinities.
   - The same bits shuffled and in place.
6. **Against the exact mathematics** (MPFR at 600 bits): every logsumexp,
   softmax, layernorm and rmsnorm result tested is correctly rounded,
   53,478 each (90 for logsumexp). That is measured, not promised; the
   check requires under 1 ulp.
7. **Controls:**
   - the naive binary32 formulas differ from the references (2,563 to
     17,012 of 65,561 inputs);
   - a naive binary32 logsumexp differs from the specification in 37 of
     200 cases, and from itself shuffled in 35.

**Planted, each caught:**
- the rounding test always passing, or its bound at 2^−60 (the table's
  entries are then decided, and 233 or 66 of sigmoid's come out wrong);
- the tie-break dropped (504 SiLU and 549 GELU results wrong);
- a corrupted table entry;
- the caller's environment used (an undecided input then reaches a missing
  table entry, and the library aborts, as it must);
- a one-part mean (498 results off the specification, 177 ulp off the
  exact value);
- softmax over a binary32 sum (10,724 off);
- rmsnorm's eps outside the rsqrt;
- logsumexp and softmax through a plain binary64 sum, which only the
  swamped vectors expose (23 shuffled results differ);
- logsumexp through log of the rounded sum (27 off, 9.5 million ulp).

Dropping GELU's correction term is caught by the table check, since one of
its entries becomes decided. Its wrong results, a few among 2^32 inputs
near x = −10, are left to the exhaustive run; a sample won't find them.

**Speed**, one thread on this Zen 3 laptop, ns per element, against the
naive binary32 formulas through the C library (`nn/test/bench.c`; the
composites retimed 2026-10-01, quiet, after crsum's change below):

| | crnn | naive | |
|---|---|---|---|
| sigmoid / SiLU | 10.8 / 11.3 | 2.9 / 2.9 | 3.7x / 3.8x |
| GELU | 73.8 | 15.4 | 4.8x |
| softplus | 27.2 | 16.4 | 1.7x |
| rsqrt | 9.2 | 2.1 | 4.4x |
| logsumexp / softmax | 10.8 / 19.8 | 3.1 / 3.2 | 3.4x / 6.1x |
| layernorm / rmsnorm | 12.5 / 8.1 | 0.8 / 0.8 | 16x / 10x |

The naive composites are timed as their main loop only, so those ratios
flatter them. GELU pays for `erfc`, an `exp` and, on x86-64 without FMA
hardware, the C library's `fma`.

**Faster through crmvec, with the same bits** (2026-10-01). With
`CRNN_CRMVEC` naming [crmvec](https://github.com/anun333/crmvec)'s
`libmvec.so.1`, on x86-64 with AVX2 and FMA, sigmoid, SiLU, GELU and
softplus take their `exp`, `log1p` and `erfc` from crmvec's vector code,
four lanes a call, and do the rounding test four lanes at a time too;
lanes that are special or undecided take the scalar steps. The fast paths'
proof needs only that those functions be correctly rounded, and a
correctly rounded result is unique, so the bits cannot change:
`make crnn-vsame CRMVEC=...` hashes all 2^32 results of each function on
both paths and finds them identical (it says VOID if crmvec did not load;
`crnn_vector_path()` tells a program which path it has). A planted
library, crmvec with `exp` scaled by 1 + 2^-20, stops crnn at its first
input left undecided and missing from the table. Same machine, ns per
element:

| | scalar | through crmvec | naive |
|---|---|---|---|
| sigmoid / SiLU | 11.1 / 12.0 | 4.2 / 4.2 | 3.1 / 3.1 |
| GELU | 78.3 | 22.0 | 16.2 |
| softplus | 29.8 | 10.0 | 17.4 |
| logsumexp / softmax | 10.8 / 19.8 | 6.4 / 10.6 | 3.1 / 3.2 |

rsqrt stays scalar: crmvec's vector `rsqrt` was slower here than
CORE-MATH's scalar one (9.8 against 7.4 ns an element).

logsumexp and softmax take their `exp`s through crmvec too, the same
values (`crnn-vsame` hashes 20,000 seeded vectors of each, lengths 1 to
3000, narrow, wide and huge values and -inf entries: identical on both
paths; the planted library changes both hashes); in the table above.

**And crsum's sums, 3 times faster** (2026-10-01): `crsum_add` allocated
and cleared 4096 bins and emptied all of them on every call, about 11 of
its 12.25 ns a term on crnn's 256-term blocks. Its bins are now one set
per thread, left zero between calls, and a mask records the groups of 64
bins a call touched, so emptying visits only those: 4.26 ns a term. The
same bits (crsum's check, and crnn-vsame's composite hashes unchanged).
The composites' figures in both tables above are with it (one quiet run,
2026-10-01; before it, softmax took 25.7 ns an element and layernorm
19.1).

**Not yet:** FP8 outputs (from the same binary64 values, rounded once
more, with a table per format); layernorm and rmsnorm through crmvec
(they call it once, for rsqrt: their sums are the cost); and GPUs at
speed.
