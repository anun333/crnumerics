# repro-scan and repro-diff

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
