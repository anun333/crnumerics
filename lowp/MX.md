# Correctly rounded functions on MX blocks

A definition, written 2026-09-30, for lowp's MX functions to follow. The
implementation and its checks come next.

## Where the rules come from

The OCP Microscaling Formats (MX) specification, v1.0, could not be read
from the environment this was written in: the network policy blocks
opencompute.org. The conversion rules below (marked **ref**) follow the
reference implementation published with it, microsoft/microxcaling at
commit 7bc41952de39:
- `mx/mx_ops.py`: `_quantize_mx`, `_shared_exponents`;
- `mx/elemwise_ops.py`: `_quantize_elemwise_core`, `_round_mantissa`;
- `mx/formats.py`: the element formats' parameters.

**Check each ref rule against the specification's text before anyone
relies on it.**

## Blocks

A block holds k elements P₁ … Pₖ of one element type T (k = 32 in OCP's
concrete formats) and a shared scale X. Element i stands for the value
Vᵢ = X · Pᵢ, exactly.

The scale X is E8M0: the byte e gives 2^(e − 127) for e = 0 … 254, and
0xFF is NaN. A NaN scale makes the whole block NaN.

| T | bits | emax_T | largest value |
|---|---|---|---|
| E5M2 | 8 | 15 | 57344 |
| E4M3 | 8 | 8 | 448 |
| E3M2 | 6 | 4 | 28 |
| E2M3 | 6 | 2 | 7.5 |
| E2M1 | 4 | 2 | 6 |

emax_T is the exponent of T's largest normal value (**ref**,
`_get_format_params`). INT8 elements are not covered here yet.

## Converting exact values to a block: Q

Q_T,r(v₁ … vₖ), for exact real values vᵢ and a rounding direction r:

1. If any vᵢ is NaN or infinite, the scale is NaN and the whole block is
   NaN (**ref**: the block's maximum is then NaN or infinite, and so is
   the scale). The elements are then written as zero.
2. Let m = max |vᵢ|. Let E = floor(log₂ m), or E = −126 if m = 0 (**ref**:
   an all-zero block takes binary32's smallest normal as its maximum).
3. Let s = E − emax_T. If s > 127, the scale is NaN. If s < −127, s = −127
   (**ref**). The scale is X = 2^s.
4. Pᵢ = round_T,r(vᵢ / 2^s): the correct rounding of the exact scaled
   value to T in direction r. T's subnormals are kept, and a value beyond
   T's largest saturates to ±largest (**ref**: `allow_denorm=True`,
   `saturate_normals=True`).

The direction r is one of IEEE 754's four: to nearest with ties to even
(the default here), toward +∞, toward −∞, toward zero. The reference
rounds ties away from zero by default ("nearest", `floor(|x| + 0.5)`),
and offers ties to even ("even") and toward zero ("floor"). Ties away from
zero is an option here, to match it.

## A correctly rounded function on a block

For a function f, the correctly rounded F on a block of type T in
direction r is:

    F_T,r(block) = Q_T,r(f(V₁), …, f(Vₖ))

with each f(Vᵢ) taken exactly, never rounded first. A two-argument f takes
Vᵢ and Wᵢ from two blocks of the same type. What this means:

- **The scale comes from the exact results.** Rounding first (to binary32,
  say) and then converting can pick a scale one step too large when the
  largest result lies just below a power of two. Then every element of
  the block differs.
- **A domain error or a pole anywhere makes the whole block NaN:** some
  f(Vᵢ) is NaN or infinite.
- **Elements cannot overflow:** the scale follows the largest result, and
  elements saturate. The scale overflows to NaN only when the largest
  result reaches 2^(128 + emax_T).
- **Small elements round to T's subnormals or to zero,** relative to the
  block's scale, as in the conversion.

### The exponent of the maximum, computed in floating point

The reference's PyTorch path takes floor(log₂ m) of a value in the
tensor's own floating-point type. A value just below a power of two can
then have its log₂ round up to the integer. For example, the largest
binary32 below 1024 is 1023.99994, and its log₂ correctly rounded to
binary32 is exactly 10.0, so the floor is 10 instead of 9. It is the same
for every exponent from 4 to 127 tried (checked in Python on 2026-09-30).
This definition takes the exact exponent.

## Computing it exactly

- **floor(log₂ m):** round m toward zero at any precision. The result's
  exponent is floor(log₂ m), because a power of two is representable at
  every precision, and rounding toward zero never passes one.
  - With MPFR: `MPFR_RNDZ` at any precision.
  - With CORE-MATH: its binary64 result in round-toward-zero.
- **The elements:** vᵢ / 2^s rounded to T. Dividing by 2^s is exact, so
  this is f(Vᵢ) rounded to T's grid moved by s, with T's subnormals below
  2^(emin_T + s). The kit's recipe (the ternary value, then
  `mpfr_subnormalize`) does it.
- **Without MPFR (the library): round-to-odd.**
  - CORE-MATH's binary64 results toward zero (z) and upward (u) are equal
    when f(Vᵢ) is exact, and lie on either side of it otherwise.
  - z with its last bit set when z ≠ u is f(Vᵢ) rounded to odd in binary64.
  - Rounding that again to a format at least 2 bits narrower, in any
    direction, gives the correct rounding of f(Vᵢ) (Boldo and Melquiond,
    "Emulation of FMA and correctly rounded sums: proved algorithms using
    rounding to odd", IEEE Transactions on Computers, 2008). T has at most
    4 bits; binary64 has 53.
  - The largest |z| gives the scale.

  So the library is correct by construction, given CORE-MATH's correct
  rounding toward zero and upward. It does not rest on exhaustive checks,
  which blocks would not allow anyway (32 elements of 16 to 256 values
  each).
- **Where binary64 runs out:**
  - **Inputs fit:** Vᵢ lies between 2^−143 and 2^143 in magnitude.
  - **Overflow:** a result that overflows binary64 exceeds 2^(128 + emax_T),
    so the scale is NaN either way.
  - **Tiny results:** below T's resolution (at least 2^−143), so they round
    to zero or T's smallest subnormal, and binary64's grid there is far
    finer than T's.
  - **All results tiny:** if every result underflows binary64, s falls
    below −127 whether m is zero or tiny, and s = −127 either way.

## How it will be checked

- **The reference:** the kit computes Q exactly from MPFR (`MPFR_RNDZ` for
  the scale, the kit's recipe for the elements).
- **Blocks can't be enumerated, so the checks build blocks with purpose:**
  - every element value of T, at every scale: a block of the 16 FP4 values
    twice, or of 32 of the 64 FP6 values, for each of the 255 scales, in
    every direction;
  - largest results just below, at and just above a power of two (the
    scale's edge);
  - one domain error or pole among finite results;
  - all zeros;
  - the scale's limits (s near −127 and 127);
  - and random blocks.
- **Controls and planted bugs** as for the rest of numerics/.

## Open questions

- Every **ref** rule against the OCP MX v1.0 text, in particular:
  - ties;
  - the all-zero block's scale;
  - clamping of s below −127;
  - the whole-block NaN rule.
- INT8 elements (MXINT8).
- Functions whose output type differs from their input type.
