# lowp: FP8, FP6, FP4 and MX math

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

## MX blocks

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
