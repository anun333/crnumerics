# numerics

Groundwork for a family of reproducible numerics libraries, beside
crmvec: correctly rounded or bit-exact math that gives the same bits on
every machine, and the tools to prove it. [ROADMAP.md](ROADMAP.md) lists
the libraries this is for and the order they come in.

This first step holds two things every one of them needs:
- **the checking kit** (`kit/`): number formats, a correctly rounded
  reference through MPFR, and exhaustive or sampled runs, each with a
  control that must fail;
- **repro-scan** (`tools/`): a scanner that reads a binary and reports what
  makes its floating-point results depend on the machine or the build.

Both are checked the way crmvec is: against answers they did not make, with
controls, and with deliberately planted bugs that the checks must catch.

## Build and check

```
make -C numerics          # the kit and its self-test
make -C numerics check    # the kit's self-test, then repro-scan's tests
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

**The reference.** `kit_ref1` and `kit_ref2` give the correctly rounded
value of any MPFR function at an input of the format, in any of the four
rounding modes:
- the function is computed at the format's precision in MPFR's wide
  exponent range;
- then MPFR's recipe for a narrower range, `mpfr_check_range` and then
  `mpfr_subnormalize`, applies the format's range and subnormals, using
  the ternary value so that nothing is rounded twice.

E4M3 has no infinity, so its overflow is handled separately: toward zero
it gives ±448, and away from zero NaN (or ±448 when saturating). A
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
the candidate's result moved by one ulp on about one input in 64. A check
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
| A | decoding and encoding, every encoding of each 8- and 16-bit format and 2^22 of binary32 (not mode-dependent) | the hardware's conversions (binary32, binary16, bfloat16; E5M2 is binary16's top byte), OCP's definition and tables (E4M3) |
| B | rounding (`kit_round`) | a brute-force search through every value of each 8- and 16-bit format, at every value, midpoint and either side of one, and past the largest (E4M3 in both overflow modes); for binary32 and binary64, the hardware's conversion, product and sum, including subnormal results and ties |
| C | the reference, on every input of 37 one-argument functions and on 2^16 pairs plus edge values of 4 two-argument ones, in binary16 and bfloat16 | CORE-MATH's binary16 and bfloat16 functions, proven correctly rounded without this kit |
| D | the reference, sampled in binary32 and binary64 | CORE-MATH's `expf`, `logf`, `sinf`, `atan2f`, `exp`, `log` |
| F | the reference in E5M2 and E4M3, every input and every pair | CORE-MATH's binary64 functions rounded again to the format, which is safe here (the comment in F says why) |
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
