# The checking kit

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
