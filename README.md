# crnumerics

Floating-point results that are the same bits on every machine, with
proofs anyone can rerun.

Correctly rounded functions give one right answer, so they agree
everywhere; [crmvec](https://github.com/anun333/crmvec) does that for the
vector functions compilers call. But most differences between machines
come from elsewhere: the order of a sum, fast approximations inside larger
operations, formats with no standard math library, and tools that can't
tell you where a difference comes from. The libraries here close those
gaps one at a time ([ROADMAP.md](ROADMAP.md) has the order):

| | What it gives you | Who it's for |
|---|---|---|
| **crsum** (`crsum/`) | correctly rounded sums, dot products and matrix products in binary64 and binary32: the same bits in any order, split or thread count; `crgemm_oz`, exact matrix products through any binary64 BLAS | simulation, training and inference, finance: anyone who needs a reduction to come out the same twice |
| **crnn** (`nn/`) | neural-network primitives with one answer: `sigmoid`, `silu`, `gelu`, `softplus` and `rsqrt` correctly rounded in binary32, and `logsumexp`, `softmax`, `layernorm` and `rmsnorm` specified bit for bit on top of crsum | inference and training that must give the same output on every machine |
| **lowp** (`lowp/`) | correctly rounded math for FP8 (E4M3, E5M2) and OCP MX blocks (MXFP8, MXFP6, MXFP4, MXINT8), proven on every input or correct by construction | low-precision machine learning; hardware and emulator writers |
| **ival** (`ival/`) | the tightest binary64 interval enclosures of 31 elementary functions, and of `atan2`, `hypot` and `pow` on boxes | verified and interval computing |
| **repro-scan, repro-diff** (`tools/`) | what in a binary or a build makes its results machine-dependent; a program run under changed conditions (threads, flush-to-zero, an older CPU) and its output compared | anyone chasing a result that changes between machines |
| **the checking kit** (`kit/`) | number formats down to FP4, correctly rounded references through MPFR, runs with controls and verdicts | building checks like these |

Each is checked the way crmvec is: against answers it did not make, with
controls, and with deliberately planted bugs that the checks must catch.

## Build and check

```
make          # the kit, lowp, ival, crsum, and their checks
make check    # every check: the kit, repro-scan, lowp, lowp's MX, ival, crsum, crnn
```

Needs gcc 13 or later (for `_Float16` and `__bf16`), MPFR 4.2 or later, and
OpenMP. repro-scan needs Python 3 and binutils; its tests also use clang and
the aarch64 and riscv64 cross compilers, and skip, saying so, the cases
whose compiler is missing. CI runs both on x86-64 and natively on arm64
(`.github/workflows/check.yml`).

Every check ends in a verdict line, as crmvec's do: `IDENTICAL` (exit 0),
`DIFFERS` (exit 1), or `VOID` (exit 2: nothing was tested, a control did not
fail, or the check could not run). repro-scan's own verdicts are `CLEAN`,
`FLAGGED` and `VOID`, with the same exit codes.

## The checking kit

`kit/kit.h` is the whole interface.

**Formats.** Each format is handled as an encoding (a `uint64_t`), with
exact decoding to `double`, since every format here embeds in binary64:

| format | bits | exponent, significand | largest | overflow |
|---|---|---|---|---|
| binary64 | 64 | 11, 52 | `DBL_MAX` | infinity |
| binary32 | 32 | 8, 23 | `FLT_MAX` | infinity |
| binary16 | 16 | 5, 10 | 65504 | infinity |
| bfloat16 | 16 | 8, 7 | 0x1.fep127 | infinity |
| E5M2 (OCP FP8) | 8 | 5, 2 | 57344 | infinity, or 57344 in saturating mode (`kit_e5m2_saturate`) |
| E4M3 (OCP FP8) | 8 | 4, 3 | 448 | NaN, or 448 in saturating mode (`kit_e4m3_saturate`) |
| E2M3 (OCP MX FP6) | 6 | 2, 3 | 7.5 | saturates (no infinity, no NaN) |
| E3M2 (OCP MX FP6) | 6 | 3, 2 | 28 | saturates |
| E2M1 (OCP MX FP4) | 4 | 2, 1 | 6 | saturates |

**The reference.** `kit_ref1` and `kit_ref2` give the correctly rounded
value of any MPFR function at an input of the format, in any of the four
rounding modes:
- the function is computed at the format's precision in MPFR's wide
  exponent range;
- then MPFR's recipe for a narrower range, `mpfr_check_range` and then
  `mpfr_subnormalize`, applies the format's range and subnormals, using
  the ternary value so that nothing is rounded twice.

E4M3 has no infinity, so its overflow is handled separately: toward zero
it gives ±448, and away from zero NaN (or ±448 when saturating). The MX
element formats have neither infinity nor NaN: overflow saturates, and a
NaN result has no encoding (`KIT_NONE`), since MX leaves NaN to the block's
shared scale. A
signaling NaN input gives a NaN (MPFR has no signaling NaNs). Where IEEE
defines a function differently from MPFR, the caller passes a wrapper;
for example, rSqrt(−0) is −∞ (`kit/test/selftest.c` has it).
`kit_round` rounds any MPFR value to a format the same way.

**Runs.**
- `kit_exhaust1` tries every input of a format of 32 bits or fewer
  (about 1/40 s per function and rounding mode for a 16-bit format, on
  four cores).
- `kit_sample1` and `kit_sample2` try the format's edge values and then
  random inputs, with every exponent equally likely (for 8-bit formats,
  `kit_sample2` tries every pair).

The candidate runs in the C rounding mode that matches, set on every
thread. Every run also carries its **control**: the same comparison with
the candidate's result moved by one ulp on about one input in 64, and on
at least one (a NaN, which has no neighbour, is replaced by zero). A check
whose control does not differ is blind, and `kit_report` calls it VOID.

A whole check of a binary16 function in four modes:

```c
#include "kit.h"
_Float16 my_cbrtf16(_Float16);
static uint64_t cand(uint64_t x)
{ uint16_t u = x; _Float16 a; memcpy(&a, &u, 2); a = my_cbrtf16(a); memcpy(&u, &a, 2); return u; }

int main(void)
{
  static const mpfr_rnd_t R[4] = {MPFR_RNDN, MPFR_RNDU, MPFR_RNDD, MPFR_RNDZ};
  kit_tally run = {0}, ctl = {0};
  for (int m = 0; m < 4; m++) {
    kit_tally c, t = kit_exhaust1(KIT_B16, cand, mpfr_cbrt, R[m], &c);
    kit_tally_add(&run, t);
    kit_tally_add(&ctl, c);
  }
  return kit_verdict(kit_report("cbrt binary16", KIT_B16, run, ctl), "to MPFR");
}
```

**How the kit itself is checked** (`kit/test/selftest.c`, 12 s on four
cores), in all four rounding modes except where noted:

| part | the kit's | against |
|---|---|---|
| A | decoding and encoding, every encoding of each 4- to 16-bit format and 2^22 of binary32 (not mode-dependent) | the hardware's conversions (binary32, binary16, bfloat16; E5M2 is binary16's top byte), OCP's definitions and tables (E4M3 and the MX element formats) |
| B | rounding (`kit_round`) | a brute-force search through every value of each 4- to 16-bit format, at every value, midpoint and either side of one, and past the largest (E4M3 in both overflow modes; the MX formats saturating, with no encoding for NaN); for binary32 and binary64, the hardware's conversion, product and sum, including subnormal results and ties |
| C | the reference, on every input of 37 one-argument functions and on 2^16 pairs plus edge values of 4 two-argument ones, in binary16 and bfloat16 | CORE-MATH's binary16 and bfloat16 functions, proven correctly rounded without this kit |
| D | the reference, sampled in binary32 and binary64 | CORE-MATH's `expf`, `logf`, `sinf`, `atan2f`, `exp`, `log` |
| F | the reference in E5M2, E4M3 and the MX element formats, every input and every pair | CORE-MATH's binary64 functions rounded again to the format, which is safe here (the comment in F says why) |
| G | coverage: every input of each format of 16 bits or fewer, every pair of each of 8 bits or fewer | candidates that count what they are handed: each exactly once (a comparison can't see an input left out) |
| E | a negative control: binary16 `exp` judged against MPFR's `exp2` | must differ, and does |

On 2026-09-30 eleven bugs were planted in the kit, one at a time: the
ternary value ignored, subnormalize left out, each end of the exponent
range off by one, E4M3's overflow rules, the signaling NaN rule, a
subnormal scale, a blind perturbation, and a comparison that always
agrees. The self-test caught every one, with DIFFERS, VOID or an abort;
part B alone caught every rounding bug, without CORE-MATH's help. The one
planted bug that nothing caught was the signaling NaN rule for
one-argument functions. It turned out to change nothing (MPFR returns NaN
for every NaN input), so it was removed.

Six more were planted in the MX formats' support, also on 2026-09-30:
- no saturation;
- the largest value one step too small;
- the top exponent's codes taken for NaN;
- a NaN given a number's encoding;
- every-pair runs enumerating the wrong pairs;
- the control moving no result in a small run.

All six are caught now. Two of them needed a new check first:
- the NaN encoding: the self-test had asked the kit for its own answer,
  and now states it independently (KIT_NONE);
- the enumeration: candidate and reference agreed on the wrong pairs, and
  part G now counts what each run hands over.

## lowp: FP8, FP6, FP4 and MX math

`lowp/lowp.h` gives correctly rounded functions for the OCP 8-bit formats:
- **formats:** E4M3 and E5M2;
- **rounding:** all four rounding modes, and E4M3's saturating mode as
  well as its NaN-on-overflow default;
- **functions:** the 41 functions CORE-MATH has for binary16 (37 with one
  argument, plus `atan2`, `atan2pi`, `hypot` and `pow`);
- **conversions:** from binary64 and binary32 (correctly rounded), and
  back (exact).

```c
#include "lowp.h"
uint8_t x[4] = {0x38, 0x40, 0x48, 0xb8}, y[4];   /* E4M3: 1, 2, 4, -1 */
lowp_e4m3_exp(x, y, 4, LOWP_NEAREST);            /* 0 on success */
lowp_e4m3_exp(x, y, 4, LOWP_UP | LOWP_SAT);      /* rounded up, saturating */
lowp_e5m2_pow(x, x, y, 4, LOWP_NEAREST | LOWP_SAT); /* E5M2 saturates too (OFP8) */
```

Every function takes arrays and an explicit mode, so no hidden global
state (the C rounding mode) changes a result. The C rounding mode and the
floating-point flags are left as they were. Build it with `make`:
`build/liblowp.a` and `build/liblowp.so` export the 312 `lowp_`
functions (90 FP8, 222 MX) and nothing else.

**How it works.** The one-argument functions are tables, so a result is
the same on every machine by construction. `lowp/gen-tables.c` generates
them from MPFR through the kit, into the committed `lowp/lowp-tables.h`
(280 KB of text, 80 KB of data). E4M3's two overflow modes share a table,
with a bit per input marking the results that overflowed away from zero.
The two-argument functions take CORE-MATH's binary64 function in the same
rounding mode, then the library's own rounding to the format: 65,536 pairs
per mode would make tables of megabytes.

**How it is checked** (`lowp/test/check.c`, 2.4 s on four cores), on
every input and every pair, in four modes, in E5M2 and in E4M3 with and
without saturation:

| part | lowp's | against |
|---|---|---|
| 1 | 37 one-argument functions | the kit's MPFR reference |
| 2 | the same | a path that shares nothing with the first: CORE-MATH's binary64 function in the same mode, rounded by lowp's own conversion (MPFR at 200 bits for the 4 functions CORE-MATH has no binary64 version of) |
| 3 | 4 two-argument functions, all 65,536 pairs | the kit's MPFR reference |
| 4 | conversions from binary64 and binary32: every value, every midpoint and either side of one, beyond the largest, the specials, 2^16 random values per mode and sign | the kit's rounding |
| 5 | a negative control (E5M2 `exp` against MPFR's `exp2`), and modes lowp doesn't take | must differ; must be refused with nothing written |

`make check` also regenerates the tables and compares them byte for byte
with the committed ones. All IDENTICAL, and clean under ASan and UBSan.

Ten bugs were planted, one at a time, on 2026-09-30:
- in the rounding: the tie's parity, overflow toward zero, the
  subnormals' quantum, saturation;
- in the functions: E4M3's overflow mask, the signaling NaN rule, a
  subnormal's decoding, E5M2's mode ignored, one table entry;
- the two-argument functions left in round-to-nearest.

The check caught nine. The tenth changed no result on any pair: no FP8
pair has an exact result that close to an FP8 value. The two-argument
functions keep the matching mode anyway, since that is exact without
relying on this.

As in the kit, an infinite exact result in E4M3's saturating mode gives
±448, and NaN in the non-saturating mode: OCP's 8-bit specification
(OFP8 1.0, Table 3) says the same for converting ±Inf (checked
2026-09-30). The same table requires a saturating mode for E5M2, where
±Inf and an overflow away from zero give ±57344; `LOWP_SAT` does that for
E5M2 too since 2026-09-30. It is the non-saturating result with ±Inf
replaced, so the tables are shared.

### MX blocks

`lowp_mx_<type>_<function>` works on OCP MX blocks: 32 elements of type
E5M2, E4M3, E3M2 or E2M3 (FP6), E2M1 (FP4) or INT8, under one E8M0 scale:
all four of OCP MX v1.0's concrete formats (MXFP8, MXFP6, MXFP4, MXINT8). It
covers 37 functions, all of the above but `exp10m1`, `exp2m1`, `log10p1`
and `log2p1`. `lowp/MX.md` defines the result:
- the exact results converted to a block as OCP's reference
  implementation converts values;
- the scale from the exact largest result;
- elements rounded in the mode asked for, with subnormals kept, saturating;
- a NaN or infinity anywhere makes the block NaN.

The rules come from that implementation, microsoft/microxcaling, since
the specification couldn't be read from here, and they are marked to be
checked against its text.

**How it works.** Each element's result is taken from CORE-MATH's binary64
function rounded down and rounded up:
- the two bracket the exact result, and the one with an odd last bit is
  the result rounded to odd;
- rounding that again to a narrower format, in any direction, is
  correct;
- the one nearer zero keeps the result's exponent, which gives the
  block's scale.

So every block is correct by construction, given CORE-MATH's correct
rounding in those two modes. That matters, because 32 elements of up to
256 values can't be enumerated.

**How it is checked** (`lowp/test/mx-check.c`, 27 s on four cores):
blocks built with purpose against the kit's exact reference (`kit/mx.c`):
- every element value at every scale (every eighth for FP8), grouped so
  that domain errors don't hide their neighbours;
- domain errors, NaN scales and elements, zeros;
- for two-argument functions, every value against every value at 121
  pairs of scales;
- random blocks.

It checks all 5 types, 37 functions and 4 modes. Also: in place, refused
modes, a negative control. All IDENTICAL, clean under ASan and UBSan, and
ten planted bugs each caught. `lowp/MX.md` has the details, and the one
case no check reaches, which the construction covers.

## ival: interval functions

`ival/ival.h` gives interval versions of 31 of CORE-MATH's binary64
functions: `ival_f(lo, hi, ylo, yhi, n)` maps each interval
[lo[i], hi[i]] to the smallest binary64 interval that contains f over it.
That is the least value rounded down and the greatest rounded up, each
correctly. For example, `ival_sin` on [1, 2] gives [sin 1 rounded down, 1],
since the maximum at π/2 lies inside. The rules:
- an interval is intersected with the function's domain first: log on
  [−1, 4] is log on [0, 4], which is [−∞, log 4 rounded up];
- an empty intersection gives the empty interval [NaN, NaN];
- infinite ends are allowed;
- a pole inside gives an infinite end;
- the C rounding mode and flags are left as they were.

The functions: the monotone ones (`exp`, `log`, `atan`, `erf`, `sqrt`,
`acos` and 18 more), `cosh`, and the periodic `sin`, `cos`, `tan`, `sinpi`,
`cospi`, `tanpi`. Of the two-argument functions, `hypot` and `atan2`
(2026-09-30), which take a box, X × Y, in C's argument order:
- **`ival_hypot(xlo, xhi, ylo, yhi, zlo, zhi, n)`:** its bounds are at the
  least and greatest magnitudes, since hypot grows with |x| and |y|;
- **`ival_atan2(ylo, yhi, xlo, xhi, zlo, zhi, n)`:** over the box minus the
  origin, where atan2 is undefined (the origin alone is empty). A box with
  points at x < 0 on both sides of the axis gives the whole range, from
  −π rounded down to π rounded up; otherwise the bounds are at its corners.
  Zeros are unsigned, as in IEEE 1788: y = 0 at x < 0 is π;
- **`ival_pow(xlo, xhi, ylo, yhi, zlo, zhi, n)`:** IEEE 1788's `pow`, whose
  domain is x > 0, and x = 0 with y > 0 (negative bases are for `pown` and
  `rootn`). It is monotone in each argument with the other fixed, so its
  bounds are at the corners; at the domain's edge C's `pow` gives the
  limits from inside (`pow(+0, y)` is 0, 1 or +∞ as y is positive, zero or
  negative). x = 0 alone gives [0, 0] when y reaches above 0, and the
  empty interval otherwise.

Not yet: `lgamma`, `tgamma` (not monotone on the negatives), binary32.

**How it works.** Each bound is a CORE-MATH value, computed rounding down
or up, at the point of the interval where f is least or greatest:
- **monotone functions:** an end;
- **`cosh`:** 1 when 0 is inside;
- **`sinpi`, `cospi`, `tanpi`:** their extrema and poles are at integers
  and half-integers, and ival counts them exactly;
- **`sin`, `cos`, `tan`:** the interval is cut into pieces shorter than
  half a period, and the sign of the derivative at each piece's ends
  says whether the piece holds an extremum or a pole.
  - Those signs are exact, computed rounding to nearest: rounding down
    would turn sin of 2^−1074 into 0.
  - Beyond 2^54, neighbouring binary64 values are further apart than π.
    There the parity of the sign changes counts the one or two critical
    points.

**How it is checked** (`ival/test/check.c`, 10 s on four cores):
- **The reference** finds the bounds its own way: MPFR at both ends in
  both directions, without assuming monotonicity. It adds each
  function's critical points and poles inside the interval, located with
  π to 2,200 bits and exact counting.
- **The intervals, per function:**
  - points;
  - 2^14 random intervals of every width and magnitude;
  - intervals around every kind of critical point, near and far (the
    binary64 value nearest a multiple of π/2 included);
  - the domain's edges;
  - infinite and empty ends.
- **A check with no reference:** the exact function at 8 points inside
  each interval must lie within it.
- **Controls:** each run's control, a negative control (`exp` against
  `exp2`), and a test in place.
- **`hypot`, `atan2` and `pow` on boxes:** 69,376 boxes (the intervals
  above paired at random, the specials against each other, boxes on and
  across both axes and around x = 1 at every scale), against a reference
  that evaluates MPFR at every pair of candidate points (the ends; zeros
  inside; for atan2's y, +0 and −0, the value on the axis and the limit
  from below; for pow, x = 1 and y = 0, within its domain written out
  again), and 8 exact points in each box. Negative controls must differ:
  the corners alone (hypot: 14,585 boxes; atan2: 6,948), and for pow, x's
  reach to 0 forgotten (5,552). Planted bugs, each caught: for hypot, a
  zero crossing ignored and the upper bound rounded down; for atan2, the
  cut ignored, signed zeros kept, and π rounded down; for pow, the x = 0
  case, the domain's cut at 0, and the diagonal corners only. (The first
  try at that last plant changed no result, since an inner loop still ran
  every corner: a plant counts only once it changes an output.)

All IDENTICAL: about 33,000 intervals and up to 150,000 points per
function. Clean under ASan and UBSan.

**The check found five bugs** in the first version, all fixed:
- the pieces of `sin`, `cos` and `tan` where neighbours are further apart
  than π;
- `sinpi`, `cospi` and `tanpi` beyond 2^52;
- a point interval of `cos` at 0;
- `cos` on [0, 2^−1074], where rounding down made sin's sign 0;
- the reference evaluating `tanpi` at a pole on the interval's end.

Eleven bugs were then planted, one at a time, and each was caught. One of
them, −0 not replaced by +0 at an end, first needed new test intervals
[−0, x]: rsqrt(−0) is −∞, but rsqrt over [−0, 4] reaches +∞.

## crsum: sums and dot products, the same bits everywhere

`crsum/crsum.h` (2026-09-30): the correctly rounded sum of n binary64 or
binary32 values, and their correctly rounded dot product, in any of the
four rounding modes:

```c
double crsum(const double *x, size_t n, int mode);                 /* CRSUM_NEAREST ... CRSUM_ZERO */
double crdot(const double *x, const double *y, size_t n, int mode);
float crsumf(const float *x, size_t n, int mode);
float crdotf(const float *x, const float *y, size_t n, int mode);
```

And matrix products, every element correctly rounded (each is one exact
dot product), row-major with leading dimensions:

```c
int crgemv(int trans, size_t m, size_t n, const double *A, size_t lda, const double *x, double beta, double *y, int mode);
int crgemm(size_t m, size_t n, size_t k, const double *A, size_t lda, const double *B, size_t ldb, double beta,
           double *C, size_t ldc, int mode);   /* and crgemvf, crgemmf */
```

`crgemv` gives y = A x + βy, or Aᵀx + βy with `trans`; `crgemm` gives C = AB
+ βC. βy is added exactly, as one more product, and with β = 0, y is not
read, as in BLAS. They return 0, or −1 for a mode they don't take.

**Exact matrix products through any BLAS: `crgemm_oz`** (the Ozaki
scheme). The same bits as `crgemm`, with the multiplying done by a binary64
GEMM you pass in (or an internal one):

```c
int crgemm_oz(size_t m, size_t n, size_t k, const double *A, size_t lda, const double *B, size_t ldb, double beta,
              double *C, size_t ldc, int mode, crsum_dgemm gemm, void *ctx);
```

A's rows and B's columns are scaled to integers and cut into slices of w =
⌊(53 − ⌈log₂ k⌉)/2⌋ bits. Then every value the GEMM computes, and every
partial sum, is an integer below 2^53, so the GEMM is exact however it
works: any summation order, fused multiply-adds or not, threads, blocking.
Any BLAS's `dgemm` in binary64 gives the same products (not an emulated or
lower-precision mode). The slice products are combined exactly and each
element rounded once. NaN or infinities, or data spread too widely (more
than 64 slice products), fall back to `crgemm`.

**How.** Every term, or every product of two terms, is added exactly into
a fixed-point accumulator that covers 2^−2176 to 2^2112 (134 limbs of 32
bits held in 64-bit integers, so that carries can wait), and the total is
rounded once. With one right answer, the order of the terms, how they are
split between threads and the vector width cannot change it: reproducible
by construction, not by fixing an order. For threads there is an
accumulator API (`crsum_init`, `crsum_add`, `crsum_add_dot`,
`crsum_merge`, `crsum_round`); merging is exact.

Special values follow a sequence of exact additions, as MPFR's `mpfr_sum`
does: NaN, or both infinities, give NaN; an exact zero is +0, −0 when every
term is −0, and −0 when rounding down after cancellation. The C rounding
mode plays no part.

**How it is checked** (`crsum/test/check.c`, 2 s on two cores), sums and
dot products in binary64 and binary32, in all four modes, against the
kit's `mpfr_sum` reference (`kit_sum_ref`):
- **the cases:** random terms at every exponent range; cancellation;
  sums exactly halfway between two results, and just above or below by a
  term far down (for dot products, a product below the format's range,
  2^−1200 or 2^−200, which only an exact method sees); subnormal results;
  overflow and cancelling back; NaN, infinities, signed zeros, no terms;
  2^18 terms. 4,676 results per kind, 0 differ;
- **the same bits in any order:** each case shuffled, and split at random
  over 1, 2, 3 and 7 accumulators merged in a random order: 74,816 results,
  none different; and 2^20 terms on 1, 2, 3, 4 and 8 OpenMP threads;
- **controls:** each run's control, and a negative one (naive left-to-right
  summation differs on 512 of the 600 cancellation and midpoint cases);
- **matrix products:** 65,864 elements of `crgemv` (both orientations) and
  `crgemm`, binary64 and binary32, four modes, each against the reference:
  padded leading dimensions (the padding NaN, so reading it shows), β of
  0, ±1 and random (0 with NaN in y, which must not be read), cancelling
  rows, NaN and infinities, sizes on both sides of the binned path's
  threshold. 0 differ; unknown modes refused with nothing written. Six
  planted bugs (the leading dimension ignored, a column read with stride 1,
  β ignored, y read at β = 0, the product bins' high parts not emptied
  between rows, B's column transposed) were each caught;
- **`crgemm_oz`:** 510,400 elements against `crgemm`, bit for bit, in four
  modes, through four GEMMs: the internal one; one that sums each
  element's products in a random order, alternating fused and separate
  multiply-adds; the system's BLAS (`dgemm_` from `libblas.so.3`, which the
  check names in its output); and a counting copy that shows the Ozaki
  path ran (240 of 300 cases; the rest fall back). The system BLAS was
  netlib's reference BLAS on the development laptop, and multithreaded
  OpenBLAS (`openblas-pthread`) on both CI runners, x86-64 and arm64 (run
  36754104164, 2026-09-30): 0 differ through each. Ranges from
  one binade to 150, zero rows, subnormals, k up to 3,000; exact zeros of
  one sign; exact midpoints and just either side, at the subnormal quantum
  too. 0 differ. A GEMM computing in binary32 must differ, and does.
  Planted, each caught: slices 2 bits too wide, a slice digit off by one
  bit, a column scaled from its top bit, a slice pair skipped, the
  exponent off by one, the exact-zero path removed, and in the direct
  rounding: ties away, the sticky bit, the subnormal quantum, the sign.
  Three first passed and showed gaps in the cases (fixed): β only ever 0
  where no zero row was; rows and columns of zeros alone fall back to
  `crgemm` whole, so the exact-zero path never ran; and the subnormal
  midpoints spread so wide that they fell back too;
- **the carries:** the check also runs on a build that settles them every
  3 terms (the default, 2^29, no test reaches);
- **planted bugs, each caught:** ties away from zero, the sticky bit
  ignored, no subnormal quantum, overflow always infinite, a cancelled zero
  always +0, a product's top piece lost, a merge dropping NaN, a negative
  term's high half added; and in the bins, a full bin never emptied (sums
  and dot products), the sign lost, the subnormal exponent off by one, a
  product's high part a binade low. Dropping the carry into the next
  binade after rounding changes no result (M 2^q is the same value either
  way), so it proves nothing about the check: a plant counts only once it
  changes an output.
  The dot products' "never emptied" plant first passed, because every long
  dot case had random signs and the two signs' bins wrapped past 2^64 alike,
  cancelling: an all-positive long case was added, and then it failed.

**Speed.** For 64 terms or more, each term goes first into a bin for its
sign and exponent (a uint64 sum of significands, Neal's "large
superaccumulator"), and a bin moves into the wide accumulator only when it
nears 2^62, and at the end. For a dot product, the exact 106-bit product
goes in as two 53-bit parts. On this Zen 3 laptop, one thread, 2^22 terms:

| | crsum | a naive loop | |
|---|---|---|---|
| binary64 sum | 0.88 ns a term | 0.75 ns | 1.2 times |
| binary64 dot product | 3.13 ns | 0.92 ns | 3.4 times |
| binary32 sum | 1.09 ns | 0.74 ns | 1.5 times |
| binary32 dot product | 3.47 ns | 0.73 ns | 4.7 times |

(The first, scalar version took 7.8 and 11.5 ns in binary64: 10.5 and
12.7 times.)

Matrix products reuse one set of bins, emptying only the exponents a row
touched: `crgemv` on 1024 × 1024 takes 4.35 ns a multiply-add (5.9 times a
naive loop), `crgemm` on 256³ 6.97 ns (8 times a naive triple loop, which
is itself far from an optimized BLAS).

`crgemm_oz` through an optimized BLAS is the fast route. One thread,
entries over 2^20 of range, OpenBLAS (scipy's bundled copy, loaded for the
test, since installing it would change this machine's system BLAS):

| | 256³ | 512³ |
|---|---|---|
| `crgemm_oz` through OpenBLAS | 18.9 ms | 117 ms |
| `crgemm_oz`, internal GEMM | 73.5 ms | 568 ms |
| `crgemm` | 130 ms | 766 ms |
| a naive triple loop (not exact) | 34.4 ms | 354 ms |
| OpenBLAS's own `dgemm` (not exact) | 0.6 ms | 5.2 ms |

About 16 slice products per call: exactness costs roughly 20 to 30 times
an optimized `dgemm` here, and much less on hardware whose matrix units
are faster at low precision (int8 slices: ROADMAP.md, item 10).

## crnn: neural-network primitives with one answer

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
   committed one.
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
naive binary32 formulas through the C library (`nn/test/bench.c`, two
runs within 2%):

| | crnn | naive | |
|---|---|---|---|
| sigmoid / SiLU | 10.8 / 11.3 | 2.9 / 2.9 | 3.7x / 3.8x |
| GELU | 73.8 | 15.4 | 4.8x |
| softplus | 27.2 | 16.4 | 1.7x |
| rsqrt | 9.2 | 2.1 | 4.4x |
| logsumexp / softmax | 18.4 / 25.7 | 3.0 / 2.9 | 6.2x / 8.9x |
| layernorm / rmsnorm | 19.1 / 7.5 | 0.7 / 0.7 | 26x / 10x |

The naive composites are timed as their main loop only, so those ratios
flatter them. Nothing here is vectorized yet. GELU pays for `erfc`, an
`exp` and, on x86-64 without FMA hardware, the C library's `fma`.

**Not yet:** bfloat16, binary16 and FP8 outputs (from the same binary64
values, rounded once more, with the table rebuilt for each); vector code;
and GPUs.

## repro-scan

```
tools/repro-scan [--json] [--strict] FILE...
```

It reads an ELF executable, shared library, object file or archive for
x86-64, i386, AArch64 or RISC-V. It gets the symbols from `readelf` and
the machine code from `objdump`, or from `llvm-objdump` for another ISA.

| kind | category | what it means |
|---|---|---|
| FLAG | estimate | x86 reciprocal and square-root estimates (`rcpps`, `rsqrtps`, `vrcp14ps`, …): Intel's and AMD's SSE/AVX estimates give different bits |
| FLAG | x87 | x87 transcendentals (`fsin`, `fpatan`, `f2xm1`, …): accuracy and bits vary between CPUs |
| FLAG | fast-math | crtfastmath's startup code (`-ffast-math`, `-Ofast`): flush-to-zero and denormals-are-zero for the whole process |
| FLAG | vector-math | calls into libmvec (`_ZGV…`), SLEEF, SVML or AMD's vector math: the bits depend on the library and the vector width |
| FLAG | blas | BLAS and LAPACK calls: kernels picked per CPU, sums split by thread count |
| FLAG | reduction | MPI reductions, LLVM OpenMP reductions (`__kmpc_reduce`) |
| info | isa-estimate | AArch64 `frecpe`, `frsqrte`, SVE's `fexpa` and friends, RISC-V `vfrec7.v`, `vfrsqrt7.v`: the same on every CPU of the ISA, and different on any other ISA |
| info | fma | fused multiply-adds: a build without them rounds differently |
| info | libm | calls into the C library's math, which is not correctly rounded in general |
| info | fp-env | writes to the rounding mode or FP control register (`ldmxcsr`, `msr fpcr`, `fsrm`, `fesetround`) |
| info | dispatch | code picked by CPU at run time (ifuncs, `target_clones`, `__builtin_cpu_supports`) |
| info | threads | OpenMP parallel regions |

Info findings don't flag unless `--strict` is given. The scanner reports
where it found each instruction: the function it lies in, or "code
without a symbol" in a stripped file.

**What it cannot see**:
- a reduction written by hand over threads;
- an OpenMP reduction compiled by gcc (inlined atomics, so only its
  parallel region shows);
- a library loaded with `dlopen`, or code generated at run time;
- crtfastmath in a stripped binary.

`tools/test/run-tests` pins down two of these blind spots as test cases.
A CLEAN scan is evidence, not proof: the checks that come with each
library are the proof.

**How it is checked**: `tools/test/run-tests` builds 22 small programs
for x86-64, AArch64 and RISC-V. Each has the exact FLAG set, exit status
and required info findings it must produce. Clean programs are among
them, so a scanner that flags everything fails, as does one that flags
nothing. A control, the clean program judged against another case's
expectation, must fail. Four scanner bugs planted on 2026-09-30 (a
missing rule, a missing symbol, a wrong severity, every info turned into
a flag) were each caught.

**On real libraries** (this container, 2026-09-30, glibc 2.39):
- crmvec's `libmvec.so.1`, built from its main of 2026-09-29 (`5bf6f82`), is **CLEAN**.
  Its info findings are:
  - fused multiply-adds;
  - libm calls (see below);
  - `ldmxcsr` where CORE-MATH raises exception flags, and in libgcc's
    soft-float exception helper;
  - the CPU check that sends its SSE2 entry points to the AVX2 code.
- glibc's `libmvec.so.1` is **FLAGGED**, with 79 estimate instructions:
  `rcpps`, `vrcpps`, `vrcp14pd`, `vrsqrtps` and others.
- glibc's `libm.so.6` is **FLAGGED**, with 39 x87 instructions
  (`fpatan`, `fyl2xp1`, `f2xm1`, `fyl2x`). The 16 it can place are in
  `long double` functions (`acosl`, `atan2l`, `exp10l`, …); the other 23
  are in code without a symbol.

**Compiler options.** Given a `compile_commands.json`, repro-scan reads
the compiler options instead:
- **FLAG:**
  - `-ffast-math`, `-Ofast` and their parts (fast-math);
  - `-mrecip` (estimate);
  - x87 arithmetic, from `-mfpmath=387` or `-m32` without `-mfpmath=sse`
    (x87);
  - `-march=native` and `-mcpu=native` (native);
  - `-ffp-contract=fast` (contract).
- **Noted:** contraction not pinned off (fma). Compilers fuse `a*b+c` by
  default wherever the target has FMA, and only `-ffp-contract=off` stops
  it.

## repro-diff: the same program, changed conditions

`repro-scan` reads what a program could do; `repro-diff` runs it and
looks:

```
tools/repro-diff [--output FILE]... [--only COND,...] -- COMMAND [ARG]...
```

It runs the command as given, then once per condition, and compares
standard output, the exit status and any `--output` files, byte for byte:

| condition | what changes |
|---|---|
| repeat | nothing: a difference is nondeterminism (time, addresses, a race) |
| threads | `OMP_NUM_THREADS` (and OpenBLAS's, MKL's) set to 1, then to the processor count |
| ftz | flush-to-zero and denormals-are-zero at startup, through an `LD_PRELOAD` shim: what a library built with `-ffast-math` does to the whole process |
| cpu | an older CPU under qemu (x86-64: Nehalem, without AVX, AVX2 or FMA; AArch64: Cortex-A57, without SVE): every CPU dispatcher, the C library's included, picks other code |

A condition that can't run (no qemu, say) is skipped, and says so.
Verdicts: IDENTICAL, DIFFERS, VOID.

**What it found in glibc** (this container, 2026-09-30, glibc 2.39): a
program printing `exp`, `log`, `sin`, `cos`, `pow` and `atan` at 200,000
arguments gives other bits under the `cpu` condition. glibc picks versions
of these functions by CPU (with and without FMA), and the versions
disagree:

| function | results that differ |
|---|---|
| `exp` | 138 of 200,000 |
| `sin` | 128 |
| `pow` | 126 |
| `cos` | 119 |
| `atan` | 48 |
| `log` | 10 |

For example, `exp(-0x1.4e68ebb380bp-2)` is `0x1.715a5688c9d3fp-1` on this
CPU (with FMA), and `0x1.715a5688c9d4p-1` without FMA. The same program
built on CORE-MATH gives the same bits under every condition.

**How both tools are checked**: `tools/test/run-tests` has 39 cases:
- 22 binaries on three ISAs;
- 10 `compile_commands.json` files, clean ones among them;
- 7 programs for repro-diff: a pure computation (no condition may change
  it), an OpenMP sum (threads), a subnormal on standard output and in a
  file (ftz), random bytes (every condition), a function cloned for FMA
  (cpu), a program that fails (VOID).

Each case has the exact result it must give, and each tool has a control
that must fail. Eight bugs planted in the new parts were each caught.

**Its first finding.** crmvec's libm calls all come from CORE-MATH's
bare-name stand-ins (`sinf16`, `acos_bf16`, …). crmvec links them but
neither exports nor calls them, with one exception: CORE-MATH's
`cr_cbrtf16` calls the platform's `cbrtf` and rounds the result to half
precision. The kit measured how much that matters, on every binary16
input in four modes:
- correct with glibc 2.39's `cbrtf`, with CORE-MATH's, and with
  `(float)cbrt(double)`;
- wrong on 3 inputs with a `cbrtf` one ulp off on some inputs, which is
  still within a 1-ulp error bound.

So crmvec's binary16 `cbrt` was only as good as the platform's libm.
crmvec fixed it in its build on 2026-09-30 (`a5af5f9`: CORE-MATH's `cbrtf`
for that file, the stand-ins dropped by `--gc-sections`, and a check that
the library imports no rounding libm function).
