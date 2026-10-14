# crfix: correctly rounded fixed-point functions

`fix/crfix.h` (2026-10-14): exp, exp2, log, log2, sigmoid, tanh, sqrt,
rsqrt, reciprocal and division on fixed-point numbers. Each result is the
exact value rounded once, then saturated, in integer arithmetic only, with
no floating point anywhere, so it runs where there is no FPU.

```c
/* sigmoid of an int16 activation in Q3.12, to an int16 in Q0.15, to nearest */
int16_t y = crfix_sigmoid(x, 12, 16, 15, CRFIX_NEAREST);
/* an int8 kernel's lookup table: tanh from Q2.5 to Q0.7 */
int32_t t[256];
crfix_table(CRFIX_TANH, 5, 8, 7, CRFIX_NEAREST, t);   /* t[x + 128] */
```

**Why:** integer-only kernels (int8 and int16 inference, microcontrollers)
compute these functions in gemmlowp, CMSIS-NN and TFLite's integer code
without stated error bounds or checks across processors (the roadmap's
survey, ROADMAP item 3). Here the result is specified exactly, so every
machine gives the same bits, and checked against MPFR.

**The specification** (in `crfix.h`):
- **Formats:** a number is an integer x with F fractional bits, meaning
  x·2^−F. A result goes to an int8, int16 or int32 with its own
  fractional bits. Both counts run from 0 to 31.
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
- **sqrt, rsqrt, reciprocal and division** are exact in integers: an
  integer square root or quotient, and an exact comparison to round.
- **Integer only:** `uint64_t`, and `unsigned __int128` where the
  compiler has one. The portable code otherwise uses Hacker's Delight's
  128-by-64-bit division, and the check runs both.

**Each approximated value comes with an enclosure** of the exact value,
an open interval in units of 2^−64. The rounding is decided when no
boundary lies strictly inside it.
- **Undecided cases:** across every 16-bit input and input format, 83
  results are undecided, all of exp. `crfix-exceptions.h` answers them,
  generated from MPFR. The generator checks every enclosure, 9,875,714 of
  them, to hold the exact value.
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

**How it is checked** (`fix/test/check.c`):
1. **The one-argument functions:** every result against the
   specification computed by MPFR (`crfix-ref.c`), in all three
   containers, every output format and mode.
   - `make check` covers every int8 input and every 16th int16 input, at
     nine input formats.
   - `make crfix-check-all` covers every 16-bit input and every input
     format, with every enclosure checked to hold the exact value.
2. **Division:** every int8 pair (`all`: at 16 × 16 input formats), and
   random int16 pairs and formats.
3. **32-bit inputs:** random x over the whole int32 range.
4. **The rest:** tables, the domain edges, invalid parameters, and the
   version.
5. **The exceptions table:** how many results it decides.

**The controls:** each result made one larger must differ, and for each
function the specification's rounding down and to nearest must differ
somewhere. Both builds (native and `-DCRFIX_PORTABLE`) run the same
check. `make crfix-install-check` installs the library and builds
against it through pkg-config.

**Proven on every 16-bit input** (2026-10-14, `make crfix-check-all` on
cfarm421, x86-64, and cfarm424, aarch64, native and portable builds, all
IDENTICAL):
- **One-argument functions:** for each of the nine, 805,306,368 results
  equal the specification's. That is every x from −32768 to 32767, every
  fin and fout, every mode and container.
- **The enclosures:** all 9,875,714 hold their exact value.
- **Division:** 6,442,450,944 int8-pair results, and 400,000 random int16
  cases.
- **32-bit inputs:** 69,120,000 sampled results.
- **The exceptions table** regenerates identically on both machines. It
  decides 83 results, and the enclosure's middle would have rounded all 83
  correctly anyway: the table is what makes them proven, not what makes
  them right.

The first run found 3 random divisions "wrong" in both builds. The
reference had compared the unscaled quotient with 2^40, so 1.3×10^11 at
fout 31 overflowed its conversion to a negative number. The library had
saturated them correctly. Every conversion in the reference now checks
that the value fits. (The quick check in `make check`: 135 million
one-argument results, 49.5 million division results, 6.9 million at 32-bit
inputs.)

**32-bit inputs** go through the same code but are only sampled: an input
whose enclosure straddles a boundary is not in the table, and is rounded
from the enclosure's middle, which is one of the two neighbours. The
samples have found none.

**Not yet:** TFLite's affine formats (scale and zero point, not a power of
two); softmax; 32-bit inputs proven; a Python binding.
