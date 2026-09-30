# numerics

Groundwork for a family of reproducible numerics libraries, beside
crmvec: correctly rounded or bit-exact math that gives the same bits on
every machine, and the tools to prove it. [ROADMAP.md](ROADMAP.md) lists
the libraries this is for and the order they come in.

Here so far:
- **the checking kit** (`kit/`): number formats, a correctly rounded
  reference through MPFR, and exhaustive or sampled runs, each with a
  control that must fail;
- **repro-scan** (`tools/`): a scanner that reads a binary and reports what
  makes its floating-point results depend on the machine or the build;
- **lowp** (`lowp/`), the first library: correctly rounded math for the
  8-bit formats E4M3 and E5M2, proven on every input.

Each is checked the way crmvec is: against answers it did not make, with
controls, and with deliberately planted bugs that the checks must catch.

## Build and check

```
make -C numerics          # the kit, lowp, and their checks
make -C numerics check    # the kit's self-test, repro-scan's tests, lowp's checks
```

Needs gcc 13 or later (for `_Float16` and `__bf16`), MPFR 4.2 or later, and
OpenMP. repro-scan needs Python 3 and binutils; its tests also use clang and
the aarch64 and riscv64 cross compilers, and skip, saying so, the cases
whose compiler is missing. CI runs both on x86-64 and natively on arm64
(`.github/workflows/numerics.yml`).

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
| E5M2 (OCP FP8) | 8 | 5, 2 | 57344 | infinity |
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
```

Every function takes arrays and an explicit mode, so no hidden global
state (the C rounding mode) changes a result. The C rounding mode and the
floating-point flags are left as they were. Build it with `make -C
numerics`: `build/liblowp.a` and `build/liblowp.so` export the 275 `lowp_`
functions (90 FP8, 185 MX) and nothing else.

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

**Open:** as in the kit, an infinite exact result in E4M3's saturating
mode gives ±448, which is still to be checked against the OCP
specification. E5M2 has no saturating mode here.

### MX blocks

`lowp_mx_<type>_<function>` works on OCP MX blocks: 32 elements of type
E5M2, E4M3, E3M2 or E2M3 (FP6) or E2M1 (FP4), under one E8M0 scale. It
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

## repro-scan

```
numerics/tools/repro-scan [--json] [--strict] FILE...
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
- crmvec's `libmvec.so.1`, built from this branch's base, is **CLEAN**.
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

So crmvec's binary16 `cbrt` is only as good as the platform's libm. The
fix belongs to crmvec's build, not to this directory; `BRANCH-LOG.md`
records it for the maintainer.
