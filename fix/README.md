# crfix: correctly rounded fixed-point functions

`fix/crfix.h` (2026-10-13): exp, exp2, log, log2, sigmoid, tanh, sqrt,
rsqrt, reciprocal, division, and (2026-10-14) SiLU and GELU on fixed-point
numbers. Each result is the
exact value rounded once, then saturated, in integer arithmetic only, with
no floating point anywhere, so it runs where there is no FPU.

```c
/* sigmoid of an int16 activation in Q3.12, to an int16 in Q0.15, to nearest */
int16_t y = crfix_sigmoid(x, 12, 16, 15, CRFIX_NEAREST);
/* an int8 kernel's lookup table: tanh from Q2.5 to Q0.7 */
int32_t t[256];
crfix_table(CRFIX_TANH, 5, 8, 7, CRFIX_NEAREST, t);   /* t[x + 128] */
/* TFLite's int8 LOGISTIC: input scale 0.0478, zero point 3; output 1/256, -128 */
crfix_affine in = {0.0478f, 3, 8}, out = {1.0 / 256, -128, 8};
crfix_table_affine(CRFIX_SIGMOID, &in, &out, CRFIX_NEAREST, t);
```

**Why:** integer-only kernels (int8 and int16 inference, microcontrollers)
compute these functions in gemmlowp, CMSIS-NN and TFLite's integer code
without stated error bounds or checks across processors (the roadmap's
survey, ROADMAP item 3). Here the result is specified exactly, so every
machine gives the same bits, and checked against MPFR.

**The specification** (in `crfix.h`):
- **Formats:** a number is an integer x with F fractional bits, meaning
  x·2^−F. A result goes to an int8, int16 or int32 with its own
  fractional bits. Both counts run from 0 to 31. Since 2026-10-14 a
  result can also go to an unsigned 8- or 16-bit container,
  `8 | CRFIX_UNSIGNED` or `16 | CRFIX_UNSIGNED`, saturated to [0, 2^B − 1].
- **Results:** the exact result, rounded once to the output's integer,
  then saturated to its range. The modes are nearest with ties to even
  (`CRFIX_NEAREST`), up, down and toward zero (crsum's names and
  numbers).
- **Domain edges:** fixed point has no NaN. log and log2 of x ≤ 0 give the
  most negative value, sqrt below 0 gives 0, rsqrt of x ≤ 0 and the
  reciprocal of 0 give the largest. a / 0 gives the largest or the most
  negative by a's sign, and 0 / 0 gives 0.

**How:**
- **exp and exp2** reduce to 2^f on [0, 1): a 64-entry table and a
  degree-7 polynomial in 64-bit fixed point. exp's argument is reduced with
  a 128-bit log2(e).
- **log and log2** reduce to ln(1 + s) with |s| < 2^−7, through a
  64-entry table of 16-bit reciprocals, and its series to s^9.
- **sigmoid and tanh** come from e^−|v| and e^−2|v| and one division,
  rounded in the direction that keeps the enclosure. Below |v| = 2^−16
  they use the series instead: sigmoid(v) = 1/2 + v/4 − c and
  tanh(v) = v − c, with c ≈ v³/48 or v³/3 computed in integers.
- **SiLU** is v·sigmoid(v): sigmoid's enclosure times the exact v.
- **GELU** is the exact v·Φ(v), with Φ the standard normal distribution
  function, not the tanh approximation.
  - **Below |v| = 8:** erf(|v|/√2) = 2c|v|·e^(−v²/2)·Σ v^(2n)/(2n+1)!!,
    with c = 1/√(2π). Every term of the series is positive, and v² is
    exact. The sum reaches 2^46, so e^(−v²/2) is kept as a power of two
    times a 61-bit mantissa until the end.
  - **At |v| ≥ 8:** v·(1 − Φ(|v|)) = c·e^(−v²/2)·S, with S = Σ (−1)^k
    (2k−1)!!/v^(2k), erfc's asymptotic series, to 30 terms. Its remainder
    is at most the next term (Abramowitz and Stegun 7.1.24), and |v|
    cancels, so nothing divides by it.
  - **Clipping:** both SiLU and GELU lie strictly between v/2 and
    max(v, 0), so their enclosures are clipped to those bounds. That
    settles the dyadic values a rounding boundary often is: v/2 is a tie
    for a small v.
- **sqrt, rsqrt, reciprocal and division** are exact in integers: an
  integer square root or quotient, and an exact comparison to round.
- **Integer only:** `uint64_t`, and `unsigned __int128` where the
  compiler has one. The portable code otherwise uses Hacker's Delight's
  128-by-64-bit division, and the check runs both.

**Each approximated value comes with an enclosure** of the exact value,
an open interval in units of 2^−64. The rounding is decided when no
boundary lies strictly inside it.
- **Undecided cases** go to a slow path (`crfix-mp.c`): interval
  arithmetic in 320-bit fixed point, integers only. Each lower bound is
  rounded down and each upper bound up, so the result encloses the exact
  value by construction, with no error analysis to trust.
  - **exp and exp2:** a series with its tail bounded, then eight squarings.
  - **log and log2:** the atanh series, using 1/(m + 1) from Newton's
    iteration, proven by two exact products.
  - **Cost:** 7 to 10 µs.
  - **Frequency:** across every 16-bit input and format, 83 results take
    it, all of exp. Until 2026-10-13 a table generated from MPFR answered
    them; the slow path made the table unnecessary, and made 32-bit inputs
    correct by construction too.
- **Two findings on the way:**
  - **sigmoid and tanh** near 0: 1/2 + v/4 and v are often exact rounding
    boundaries, and the true value falls short by only v³/48 or v³/3,
    below 2^−64 for tiny v. A generic enclosure straddled them, and random
    checks found 188 wrong results before the series path.
  - **exp and exp2** first left 340,000 cases undecided: results far past
    every container, whose rounding doesn't matter. They are now
    saturated before deciding.

**Speed** on board5 (Ryzen 5 PRO 5650U, under load), int16 inputs in
Q3.12 to Q0.15 (`exp` and `sigmoid` saturating at large inputs):

| | exp | exp2 | log | log2 | sigmoid | tanh | sqrt | rsqrt | recip |
|---|---|---|---|---|---|---|---|---|---|
| native 128-bit | 24 ns | 22 | 24 | 26 | 61 | 61 | 17 | 22 | 12 |
| portable | 46 ns | 41 | 38 | 39 | 99 | 101 | 37 | 43 | 18 |

A float `expf` with rounding, not correctly rounded, takes 5.6 ns; for
int8, `crfix_table` gives a lookup table of correctly rounded results.
SiLU takes 77 ns and GELU 0.81 µs (its series), each on int16 inputs from
Q3.12 to Q2.13 (2026-10-14, load 7.6; sigmoid 61 ns in the same run). An
int8 affine table of GELU takes 0.2 ms.

**How it is checked** (`fix/test/check.c`):
1. **The one-argument functions:** every result against the
   specification computed by MPFR (`crfix-ref.c`), in all five containers
   (three signed, two unsigned since 2026-10-14), every output format and
   mode. GELU's reference is v·erfc(−v/√2)/2, with √2 bounded both ways.
   - `make check` covers every int8 input and every 16th int16 input, at
     nine input formats.
   - `make crfix-check-all` covers every 16-bit input and every input
     format, with every enclosure checked to hold the exact value.
2. **Division:** every int8 pair (`all`: at 16 × 16 input formats), and
   random int16 pairs and formats.
3. **32-bit inputs:** random x over the whole int32 range.
4. **The rest:** tables, the domain edges, invalid parameters, and the
   version.
5. **The slow path:** the results the fast enclosure leaves undecided,
   each against the specification. Since 2026-10-14 the slow path is also
   called directly for every approximated function, on sampled 16-bit and
   32-bit inputs and affine values: SiLU's and GELU's fast paths leave it
   no cases in the sample, so they alone would test it little. Two
   deliberate errors in its SiLU and GELU code are both caught there.
6. **Affine formats:** random TFLite-like configurations, every int8 input
   and some int16 ones, every mode, unsigned outputs among them; a uint8
   input table against the single results; invalid scales refused.
7. **Softmax:** random int8 vectors of 1 to 64 elements and some int16
   ones, against MPFR, every mode. The edge cases: all equal (exact ties),
   one dominant (the rest vanish), one element (saturating).

**The controls:** each result made one larger must differ, and for each
function the specification's rounding down and to nearest must differ
somewhere. Both builds (native and `-DCRFIX_PORTABLE`) run the same
check. `make crfix-install-check` installs the library and builds
against it through pkg-config.

**Proven on every 16-bit input** (2026-10-13, `make crfix-check-all` on
cfarm421, x86-64, and cfarm424, aarch64, native and portable builds, all
IDENTICAL):
- **One-argument functions:** for each of the nine, 805,306,368 results
  equal the specification's. That is every x from −32768 to 32767, every
  fin and fout, every mode and container.
- **The enclosures:** all 9,875,714 hold their exact value.
- **Division:** 6,442,450,944 int8-pair results, and 400,000 random int16
  cases.
- **32-bit inputs:** 69,120,000 sampled results.
- **The slow path:** the same run again with it in place of the table
  (01:05–01:09Z): every verdict IDENTICAL on both hosts and builds. It
  decided the same 83 results, and all 2,534,400 affine results of 400
  random configurations were right.
- **The exceptions table** (since replaced by the slow path) had
  regenerated identically on both machines. It decided 83 results, and the enclosure's
  middle would have rounded all 83 correctly anyway: the table is what made
  them proven, not what made them right.

The first run found 3 random divisions "wrong" in both builds. The
reference had compared the unscaled quotient with 2^40, so 1.3×10^11 at
fout 31 overflowed its conversion to a negative number. The library had
saturated them correctly. Every conversion in the reference now checks
that the value fits. (The quick check in `make check`: 135 million
one-argument results, 49.5 million division results, 6.9 million at 32-bit
inputs.)

**On a microcontroller** (`make crfix-cross-check`, 2026-10-13; a CI job):
- **No floating point:** built for a Cortex-M0, the library calls no
  floating-point routine and no libm function, and uses no instruction
  beyond ARMv6-M. It needs only libgcc's 64-bit integer helpers,
  `memcpy` and `memset`, and the allocator, for softmax alone. That is
  50,010 bytes of code (`arm-none-eabi-size` of the two objects; 34,750
  before SiLU and GELU, 27,298 before the affine and softmax fast paths). Each count has a control: a file adding two doubles shows
  `__aeabi_dadd`, and the same code built for Cortex-M3 shows ARMv7-M
  instructions.
- **The same bits:** a digest of 23,393,816 results (every function at
  every int16 input over three input and three output formats and every
  mode, division over int8 pairs, affine tables, softmax). It is
  d7a3b901584298d7 on x86-64 (native and portable), on 32-bit ARM Linux
  with soft float under qemu, and on the Cortex-M0 build running bare
  metal under qemu-system-arm. One changed result changes it. (With SiLU
  and GELU it covers eleven functions; before them it was
  d5e26d9b05816b07, and f6fbcbe26271becf before the toward-zero fix for
  affine outputs, below.)

**32-bit inputs** go through the same code. Their fast-path enclosures
rest on the same arithmetic, whose bounds were proven on every 16-bit
input; an undecided one goes to the slow path. They are checked on
samples.

**Affine formats** (`crfix_eval_affine`, `crfix_table_affine`) use
TFLite's quantization:
- **The format:** a value is scale·(q − zero point), with any positive
  normal binary64 scale. A float32 scale converts exactly, and the scale's
  bits are read with no floating-point operation.
- **The result:** round(f(v)/scale_out) + zero_out, in the mode asked for,
  saturated.
- **A fast path first:** |v| goes into signed Q64.64, exactly or below it
  by less than one unit, and through the fixed-point functions' 64-bit
  cores. exp takes w = |v|·log2(e) from a 128-bit product; log takes the
  top 64 bits of |q − zero point|·mantissa; sigmoid and tanh go through
  e^−|v|. Each end of the enclosure is then divided by the output scale,
  rounded outward, and the zero point is added.
- **The slow path takes the rest:** results the enclosure leaves
  undecided, sigmoid and tanh for 0 < |v| < 2^−16, and exp results from
  2^62 up when the output scale is 2^29 or more. Scales are arbitrary, so
  no input range can be proven ahead. The fast path's bounds are argued in
  `crfix.c` and checked on samples. The slow path is correct by
  construction.
- **Cost:** an int8 table takes 13 to 26 µs, 0.05 to 0.1 µs a value; the
  slow path alone took 0.9 to 2.4 ms. Measured on board5 at load 11, the
  best of five passes, with the same results:

  | table | slow path alone | fast path |
  |---|---|---|
  | sigmoid int8 (TFLite's LOGISTIC) | 1,060 µs | 25.6 µs |
  | tanh int8 (TFLite's TANH) | 1,067 µs | 20.7 µs |
  | exp int8 to int16 | 908 µs | 13.0 µs |
  | log int8 | 2,360 µs | 15.6 µs |
  | sigmoid, 256 int16 inputs, input scale 2^−12 | 1,062 µs | 22.4 µs |

- **Checked against MPFR** (`make check`):
  - **Random configurations:** 253,440 results of TFLite-like
    configurations (input scales as float32 values, zero points, the
    standard sigmoid and tanh outputs, random ones, every mode), with int16
    inputs.
  - **Edge configurations:** 348,768 results. Input scales have 53-bit
    mantissas (so |v| is truncated), or are tiny (2^−70) or large;
    output scales run from 2^−60 to 2^50; inputs are 32-bit.
  - **The fast path's own decisions** are compared as well, since the slow
    path would otherwise hide a wrong one. It decides all 245,760 int8
    results of the random configurations and 304,474 of the edge ones.
  - **Controls:** four deliberate errors in the fast path; the check
    reports three. The fourth drops the 3-unit widening of log when the
    mantissa is truncated (32-bit inputs). That widening is needed for the
    proof, but it sits inside log's 64-unit margin, so no sample can see
    it.
- **Found on the way:**
  - **0.7.0's slow path** saturated exp and exp2 values from 2^62 up,
    whatever the output scale. That is wrong for output scales above 2^30:
    exp(46) at scale 2^50 is 84,342, not saturated. Such a value is now
    kept as 2^g·2^k.
  - **The MPFR reference** treated the sum with the output zero point as
    exact even when it rounded: −128 + 2^−2498 is not −128.

**Softmax** (`crfix_softmax`):
- **The specification:** y_i = round(p_i/scale_out) + zero_out,
  saturated. p_i is the exact softmax of c·q: c is β times the logits'
  scale, as TFLite has it, and the zero point cancels.
- **A fast path first:** each e^(c·d_i) is the affine exp's 64-bit
  enclosure, and the sum's bounds are 128-bit sums of bounds. Each p_i's
  bounds are a division rounded each way, by the sum's top 64 bits rounded
  the other way. Then comes the affine output stage. If all logits are
  equal, p = 1/n, exactly or between two units.
- **The slow path** takes the whole vector if any element is undecided.
  Each distinct difference q_i − max gets its exponential once (an int8
  vector has at most 256), the sum's bounds are exact sums of bounds, and
  the reciprocal is proven: correct by construction for any vector.
- **Exact ties are kept:** two equal logits give 1/2 exactly, 128 in
  TFLite's output (1/256, −128) before the zero point.
- **Cost:** 0.09 to 0.1 µs an element, 0.97 µs for 10 int8 logits and 91 µs
  for 1000. The slow path alone took 3.4 µs an element (1.2 for 1000
  logits, with more repeated differences). Measured on board5 at load 10,
  the best of five passes, with the same results.
- **Checked against MPFR** (`make check`):
  - **Random vectors:** 300 of them (3,000 in `all`) in every mode, as
    above.
  - **Edge vectors:** c from 2^−40 to 2^20 and 53-bit; output scales from
    2^−60 to 8; 32-bit logits; 1,000 elements. Two more vectors are built
    so that one bound's last unit decides:
    - {127, −128} at c = 1: the sum's upper bound, used as a divisor, must
      be rounded up, or 1 − e^−255 rounds down to 1. The output is 16-bit,
      since saturating to int8 would hide the error.
    - 3,072 equal logits at the output scale floor(2^54/3)·2^−64: 1/3072
      truncated is exactly one output unit, so it must not be taken as
      exact.
  - **The fast path's own results** are compared as well. It decides all
    the random vectors and 469 of the 488 edge ones.
  - **Controls:** one result in 101 is made one larger, and must differ.
    Four deliberate errors in the fast path are all caught.

**Arrays** (`crfix_eval_n`, `crfix_eval_affine_n`, `crfix_div_n`): a
whole array in one call, the same results as the scalar functions
(section 4 of the check compares them).

**Python** (`fix/python`, the module `crfix`; NumPy integer arrays in,
int32 arrays out):

```python
import numpy as np, crfix
y = crfix.sigmoid(np.arange(-128, 128), fin=4, fout=7, bits=8)
t = crfix.table_affine("sigmoid", crfix.Affine(0.0478, 3), crfix.Affine(1 / 256, -128))
p = crfix.softmax(logits, c=beta * 0.0478)
```

- **Its tests use a second reference**, independent of MPFR and of the C
  check: Python's `decimal`, whose exp and ln are correctly rounded,
  widened by an error bound and refined until each rounding is decided.
  sigmoid and tanh go through their complements to 1, and the exact cases
  and sqrt, rsqrt, reciprocal and division use fractions.
- **Coverage:** every int8 input and 40 int16 ones; nine functions at six
  formats in every mode; division; affine configurations and softmax
  vectors.
- **It found a bug in 0.7.0:** with an output zero point, CRFIX_ZERO
  rounded y + zero point toward 0, not the value y. Both the C check and
  its MPFR reference made the same mistake, so the check passed. With
  TFLite's logistic output (zero point −128), almost every inexact entry
  was one too large. The fast path, the slow path and the MPFR reference
  now round y toward 0 by the value's sign, then add the zero point. Only
  affine and softmax results in that mode change; the digest's other
  sections are identical, mode by mode.
- **A control:** the same tests on 0.7.0's library fail exactly on those
  cases, and pass elsewhere through the binding's fallback loop for a
  library without the array functions.
- **Tables from the command line:** `python -m crfix table tanh --fin 5
  --fout 7 --format c` prints a C array, and `table-affine` does the same
  for affine formats. `--f32` rounds decimal scales to float32 first, as
  TFLite stores them.
- **Running it:** `make crfix-python-check` needs NumPy and pytest.
  `.github/workflows/crfix-wheels.yml` builds wheels that carry the
  library, and tests them in clean venvs.
