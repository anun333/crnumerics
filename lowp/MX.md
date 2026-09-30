# Correctly rounded functions on MX blocks

A definition, written 2026-09-30, and what lowp does with it: `lowp/mx.c`
implements it (`lowp_mx_<type>_<function>` in `lowp.h`), and
`lowp/test/mx-check.c` checks it against the kit's exact reference
(`kit/mx.c`).

## Where the rules come from

The OCP Microscaling Formats (MX) specification, v1.0, could not be read
from the environment this was written in: the network policy blocks
opencompute.org. The conversion rules below (marked **ref**) follow the
reference implementation published with it, microsoft/microxcaling at
commit 7bc41952de39:
- `mx/mx_ops.py`: `_quantize_mx`, `_shared_exponents`;
- `mx/elemwise_ops.py`: `_quantize_elemwise_core`, `_round_mantissa`;
- `mx/formats.py`: the element formats' parameters.

**Checked against the specification, 2026-09-30.** The OCP MX v1.0 text
(September 2023; fetched through the Wayback Machine, since opencompute.org
answers a browser challenge) settles some of the **ref** rules and is
silent on the rest:
- **The scale, from the spec (§6.3, step 1):** "the largest power-of-two
  less than or equal to max(|Vᵢ|), divided by the largest power-of-two
  representable in the element data type": E = floor(log₂ m) and
  s = E − emax_T, exactly, as below. (The reference's floating-point
  floor(log₂) can come out one too large just below a power of two;
  the spec's wording is exact.)
- **Saturation, from the spec (§6.3, step 2):** normal values beyond the
  element type's largest are clamped to it, keeping the sign.
- **Ties, from the spec (§5.3, §6.3):** implementations must support
  roundTiesToEven for the element conversion; other modes may be
  supported. So ties to even is the spec's mode, and the reference's
  default, ties away from zero, is an extra one.
- **Silent, so lowp's own choice (still marked ref):** the all-zero
  block's scale (no power of two is ≤ 0); s below −127 (E8M0 covers −127 to
  127, Table 7, and the spec doesn't say what to do outside it); s above
  127; and NaN or infinite inputs. Converting a NaN to FP6, FP4 or INT8 is
  "implementation-defined" (§5.3.2–5.3.4). A NaN scale makes every value
  NaN (§5.1); for FP8 elements the spec would also allow a NaN element in a
  block that is otherwise finite, so the whole-block rule is a choice.

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

## How it is checked

- **The reference:** the kit computes Q exactly from MPFR (`MPFR_RNDZ` for
  the scale, the kit's recipe for the elements), in `kit/mx.c`.
- **Blocks can't be enumerated, so `lowp/test/mx-check.c` builds them with
  purpose,** for every element type, function and rounding mode:
  - every element value that isn't NaN or infinite, at every scale (FP8:
    every eighth scale, and the 16 at each end). The values are grouped
    by sign and by whether the function's exact result is finite there,
    so that a domain error doesn't hide its neighbours;
  - one domain error or pole among finite results;
  - a NaN scale; FP8 blocks holding a NaN or an infinity; all zeros;
  - for two-argument functions, every value against rotations of every
    value, at 11 × 11 pairs of scales from 0 to 254;
  - random blocks (512, and 1,024 pairs).

  Each block is compared element by element, scale too. Also:
  - in place (`q = p`, `y = x`) gives the same blocks;
  - modes lowp doesn't take are refused, nothing written;
  - E2M1 `exp` judged against `exp2` must differ;
  - every run's control moves one output and must differ.
- **Results, 2026-09-30:**
  - IDENTICAL for all 185 type-and-function pairs, in 27 s on four cores;
  - clean under ASan and UBSan;
  - ten planted bugs, each caught: round-to-odd taking the wrong one of
    the two, the scale from the upward result, emax off by one, the clamp
    at −126, infinity not making the block NaN, no saturation, the tie's
    parity, the all-zero block's exponent, an FP8 NaN element not seen.
- **Not reached by any check:** the scale's edge. A largest result within
  one binary64 ulp of a power of two, without being it, takes an input
  that no function here has at these element values. The construction
  covers it instead (rounding toward zero never passes a power of two).

## Open questions

- ~~Every **ref** rule against the OCP MX v1.0 text~~ checked 2026-09-30
  (above): the scale, saturation and ties are the spec's; the all-zero
  block, the clamp below −127 and the whole-block NaN rule are choices the
  spec leaves open.
- INT8 elements (MXINT8).
- Functions whose output type differs from their input type.
