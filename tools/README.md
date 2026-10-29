# repro-scan and repro-diff

## repro-scan

```
tools/repro-scan [--json] [--strict] FILE...
```

It reads an ELF executable, shared library, object file or archive for
x86-64, i386, AArch64 or RISC-V. It gets the symbols from `readelf` and
the machine code from `objdump`, or from `llvm-objdump` for another ISA.
It also reads:
- a Python wheel (`*.whl`): each ELF file in it is scanned, and checked
  against the wheel's platform tag;
- a directory, such as an installed package: each ELF file under it is
  scanned, and checked against this machine's ISA.

| kind | category | what it means |
|---|---|---|
| FLAG | estimate | x86 reciprocal and square-root estimates (`rcpps`, `rsqrtps`, `vrcp14ps`, …): Intel's and AMD's SSE/AVX estimates give different bits |
| FLAG | x87 | x87 transcendentals (`fsin`, `fpatan`, `f2xm1`, …): accuracy and bits vary between CPUs |
| FLAG | fast-math | crtfastmath's startup code (`-ffast-math`, `-Ofast`): flush-to-zero and denormals-are-zero for the whole process |
| FLAG | vector-math | calls into libmvec (`_ZGV…`), SLEEF, SVML or AMD's vector math: the bits depend on the library and the vector width |
| FLAG | blas | BLAS and LAPACK calls: kernels picked per CPU, sums split by thread count |
| FLAG | reduction | MPI reductions, LLVM OpenMP reductions (`__kmpc_reduce`) |
| FLAG | libm-cpu | x86-64 calls into glibc's math functions that take FMA or non-FMA code by CPU (`exp`, `log`, `pow`, `sin`, `cos`, `atan2`, `sinf`, …), or glibc's FMA variants linked in statically: the same file gives other bits on a CPU without AVX2 and FMA |
| FLAG | arch | (wheels and directories) a library for another ISA than the platform it ships for: it cannot load there |
| info | isa-estimate | AArch64 `frecpe`, `frsqrte`, SVE's `fexpa` and friends, RISC-V `vfrec7.v`, `vfrsqrt7.v`: the same on every CPU of the ISA, and different on any other ISA |
| info | fma | fused multiply-adds: a build without them rounds differently |
| info | libm | calls into the C library's math, which is not correctly rounded in general; on x86-64, the functions that give the same bits on every CPU (`cbrt`, `hypot`, `powf`, `logf`, …) |
| info | fp-env | writes to the rounding mode or FP control register (`ldmxcsr`, `msr fpcr`, `fsrm`, `fesetround`) |
| info | dispatch | code picked by CPU at run time (ifuncs, `target_clones`, `__builtin_cpu_supports`) |
| info | threads | OpenMP parallel regions |

Info findings don't flag unless `--strict` is given. The scanner reports
where it found each instruction: the function it lies in, or "code
without a symbol" in a stripped file.

**Fused multiply-adds in a CLEAN verdict.** They give the same bits on
every machine that runs the file, so they don't flag; but a build or an
ISA without them gives other bits, so a CLEAN verdict names how many it
found rather than leave them in the info lines. EPA SWMM's published
aarch64 wheel scanned CLEAN while its 552 fused multiply-adds changed 10
of 34 test models against the x86-64 wheel.

**Where libm-cpu comes from.** glibc's x86-64 libm picks FMA or non-FMA
code at load for some functions, directly or through a function they call
(`sinh` calls `exp`). The list is measured: each function gave other bits
with AVX2 and FMA hidden from glibc, on at least one release from 2.27 to
2.44, on every input of the float functions and 2^26 of the double ones
([crmvec's `docs/distros.md`](https://github.com/anun333/crmvec/blob/main/docs/distros.md),
column "CPU-dependent"). CORE-MATH's imports into glibc ended it for some
(`erf`, `erfc`, `lgamma`, `tgamma` in 2.43; `sinh`, `cosh`, `tanh` in
2.44), but a binary can't say which glibc it will run on. RHEL 10's glibc
is built for x86-64-v3 with no choice at run time, so there the flag is
moot. A correctly rounded libm, such as crmvec's crpreload, removes it.
On aarch64 glibc keeps one path, so the same calls are only `libm`.

**What it cannot see**:
- a reduction written by hand over threads;
- an OpenMP reduction compiled by gcc (inlined atomics, so only its
  parallel region shows);
- a library loaded with `dlopen`, or code generated at run time;
- crtfastmath in a stripped binary.

`tools/test/run-tests` pins down two of these blind spots as test cases.
A CLEAN scan is evidence, not proof: the checks that come with each
library are the proof.

**How it is checked**: `tools/test/run-tests` builds 26 small programs
for x86-64, AArch64 and RISC-V. Each has the exact FLAG set, exit status
and required info findings it must produce. Clean programs are among
them, so a scanner that flags everything fails, as does one that flags
nothing. A control, the clean program judged against another case's
expectation, must fail. Four scanner bugs planted on 2026-09-30 (a
missing rule, a missing symbol, a wrong severity, every info turned into
a flag) were each caught.

**On real wheels** (2026-10-29, swmm-toolkit 0.17.0 from PyPI, EPA SWMM's
Python package):
- the x86-64 wheel is **FLAGGED**: its `libswmm5.so` calls 11 of glibc's
  CPU-dependent functions (`exp`, `log`, `pow`, `acos`, …). One binary run
  on a CPU without FMA fails the project's own `allclose` test on 277
  values of one model;
- the aarch64 wheel is **CLEAN**, and the verdict names its 555 fused
  multiply-adds, 552 of them in `libswmm5.so`.

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
| hashseed | `PYTHONHASHSEED` set to another value: Python's string hashes, and the order of sets and dicts built from them |
| layout | the addresses (Linux): address-space randomization turned the other way for the run, and the environment 4 KiB longer, which moves the stack: code that orders by address, such as a `std::set<T*>` |

`repeat` sees a layout or a hash seed change only when they are random to
begin with. Where they are pinned (`setarch -R`, a debugger, a container
that turns randomization off, `PYTHONHASHSEED` set), it runs on the same
ones, and only `hashseed` and `layout` move them. WNTR's simulator is the
case: its evaluator keeps variables in sets ordered by address, and one
network changes a pump's state from run to run, unless both are pinned
and the command line is the same. With randomization off, every condition
that changes the environment moves the stack too, so a program that
depends on stack addresses differs under all of them; `layout` names the
cause.

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

**How both tools are checked**: `tools/test/run-tests` has 54 cases:
- 26 binaries on three ISAs;
- 10 `compile_commands.json` files, clean ones among them;
- 7 wheels and directories: a library tagged or placed for this ISA,
  another ISA and any platform, and no native code at all;
- 2 checks of the verdict's words: FMA named when found, and only then;
- 9 programs for repro-diff: a pure computation (no condition may change
  it), an OpenMP sum (threads), a subnormal on standard output and in a
  file (ftz), random bytes (every condition), a function cloned for FMA
  (cpu), the address of a large allocation under `setarch -R` (layout, and
  ftz and cpu, which map other code), a Python set's order with
  `PYTHONHASHSEED` pinned (hashseed), and a program that fails (VOID).

Each case has the exact result it must give, and each tool has a control
that must fail. Eight bugs planted in the 2026-09-30 parts were each
caught. The cases added on 2026-10-29 were run against the tools as they
were before that day's change, and 15 of them failed there, as they must.

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
the library imports no rounding libm function). CORE-MATH fixed it upstream on 2026-10-05 (`398b235`): its `cr_cbrtf16` now
computes the cube root itself and calls no C library function, and
`core-math/f16/cbrtf16.c` here is that version.
