# Roadmap

The aim: numerics that give the same bits on every machine (every CPU,
every instruction set, and GPUs), with proofs a stranger can rerun.
Correctly rounded functions, crmvec's job, are necessary for that but not
sufficient. Most differences between machines come from elsewhere:
- summation order;
- fast approximations inside larger operations;
- formats with no standard math library;
- random-number transforms built on inexact functions.

This page lists the libraries that close those gaps, in the order they
come. Statements about other projects are from memory as of 2026-09 and
are marked where unsure; check them before relying on them.

## Where correctly rounded math stands

- **Scalar, on CPUs: largely solved.** CORE-MATH did the hard part. Some
  of its functions have been adopted by glibc and LLVM's libc (how many,
  today, is worth checking). C23 reserves `cr_`-prefixed names for
  correctly rounded functions.
- **Vector, on CPUs:** crmvec, a drop-in libmvec and SLEEF replacement.
  As far as we know it is the only correctly rounded one.
- **GPUs: mostly missing.** The vendor math libraries (CUDA's, AMD's) and
  the Metal and Vulkan specifications allow errors of an ulp or more.
  LLVM's libc builds for GPUs and may carry some correctly rounded
  functions; its coverage is unverified here.

## Step zero: done

2026-09-30, first as crmvec's `claude/numerics-groundwork` branch (see
[README.md](README.md)):
- **the checking kit:** formats down to FP8, a correctly rounded MPFR
  reference, runs with controls, verdicts;
- **repro-scan**, the first version of item 8.

Each is checked against answers it did not make, and against
deliberately planted bugs.

## Phase 1: small, and quick to prove

**8. A reproducibility sanitizer.** Done so far: repro-scan reads ELF
binaries for x86-64, AArch64 and RISC-V, and compiler options from
`compile_commands.json`. repro-diff runs a program under changed
conditions (threads, flush-to-zero, an older CPU, a repeat) and compares
its output, which on 2026-09-30 showed glibc 2.39's `exp`, `sin`, `pow` and
others giving other bits without FMA. Next:
- GPU code: PTX, AMDGPU and SPIR-V kernels (estimate instructions, fast
  division and square root, `-ffast-math`-style flags);
- Python wheels: scan the shared objects inside;
- repro-diff: more conditions, such as another ISA under qemu (the same
  source built twice), and a vector width forced lower.

Related tools cover accuracy more than reproducibility: Verrou,
FPChecker, Herbie.

**4. FP8, bfloat16 and MX math.**
- **FP8 (E4M3, E5M2): done** (2026-09-30), as lowp (README.md): 41
  functions and the conversions, every rounding mode and both of E4M3's
  overflow modes, proven on every input and every pair.
- **bfloat16:** vector versions of CORE-MATH's scalar bfloat16
  functions, through crmvec's portable core.
- **MX block formats** (a shared E8M0 scale over FP8, FP6 or FP4
  elements): **done** (2026-09-30), in lowp.
  - `lowp/MX.md` defines a correctly rounded function per block, with the
    scale from the exact results.
  - lowp computes it by rounding CORE-MATH's binary64 results to odd:
    correct by construction for all 5 element types and 37 functions.
  - The check uses blocks built with purpose.
  - Still to do: checking the conversion rules against OCP's text, INT8
    elements, and packed storage (two FP4 elements to a byte).
- **Kit additions:** FP6 (E2M3, E3M2) and FP4 (E2M1) are done
  (2026-09-30). E8M0 needs no kit format, since the scale is computed as
  an exponent.

As far as we know, no standard math library exists for FP8 or MX.

**6. Vector interval arithmetic.** First version **done** (2026-09-30),
as ival (README.md): 31 functions in binary64, each the tightest
enclosure, checked against an independent MPFR reference. Still to do:
`lgamma`, `tgamma`, the two-argument functions, binary32, and vector
code for the directed modes (ival calls CORE-MATH's scalar functions).
- **Why it's close:** crmvec is correctly rounded in all four modes, so
  rounding down and up gives both ends of an interval: vectorized
  interval elementary functions.
- **What it needs:**
  - handling of each function's extrema (`sin`, `cos`, `tgamma`);
  - vector code for the directed modes (crmvec's vector code runs in
    round-to-nearest, and hands other modes to scalar CORE-MATH).
- **What exists:** MPFI, filib++ and the IEEE 1788 reference are mostly
  scalar.
- **Kit additions:** a runner for two-result functions (both ends
  against MPFR's `RNDD` and `RNDU`).

## Phase 2: reproducible machine learning and simulation

**1. Reproducible reductions and BLAS.**
- **The problem:** sums, dot products and matrix products differ with
  thread count, vector width and GPU, whatever the math library does.
- **What exists:** published algorithms (binned summation, as in
  ReproBLAS; exact accumulation, as in ExBLAS or a Kulisch accumulator).
  Intel's MKL has a reproducibility mode within its own CPU families.
- **The plan:** CPU vector code first, GPU through PoCL later.
- **Proof:**
  - the same bits under every permutation, thread count and vector
    width tried;
  - for the exact variant, equality with MPFR's `mpfr_sum`, which is
    correctly rounded.
- **Kit additions:** a many-input reference (`mpfr_sum`) and an order
  shuffler.

**2. Specified neural-network primitives.**
- **What:** softmax, log-sum-exp, GELU, SiLU, sigmoid, layer norm, RMS
  norm and rsqrt, where fast approximations usually live.
- **How each is specified:**
  - one function: correctly rounded;
  - a composite: a fixed sequence of correctly rounded operations and
    item 1's reductions.
- **Formats:** binary32, bfloat16, binary16 and FP8.
- **Depends on:** crmvec, item 1, item 4.

**3. Integer and fixed-point math.**
- **What:** exp, log, sigmoid, tanh, division and rsqrt for integer-only
  kernels.
- **The plan:** the specification first (rounding and saturation rules),
  then the code.
- **What exists:** gemmlowp, CMSIS-NN and TFLite's integer code each have
  some of this, without stated error bounds or checks across ISAs.
- **Proof:** 8- and 16-bit inputs, exhaustively against MPFR.
- **Kit additions:** fixed-point formats.

**5. Reproducible random-number distributions.**
- **What already works:** counter-based generators (Philox, Threefry)
  give the same bits everywhere.
- **What breaks it:** the transforms to normal, gamma and other
  distributions call `log`, `sin`/`cos` or the inverse normal CDF.
- **The plan:** those transforms built only on correctly rounded
  functions, plus a correctly rounded inverse normal CDF (`erfinv`,
  `ndtri`). We know of none (unverified).
- **Proof:** the binary32 version by exhaustion (2^32 inputs); the
  binary64 version is research.

## Phase 3: research, hardware and standards

- **GPU builds.**
  - **What's known:** on the review branch, crmvec's portable core
    compiled for NVPTX and AMDGCN, and so did 147 of 148 CORE-MATH files
    (a compile probe only).
  - **What it needs:** a GPU to run and check on.
- **7. Double-double, quad precision and complex functions.**
  - **The gap:** libquadmath is not correctly rounded, and glibc's
    complex `cexp`, `clog` and `cpow` have loose error bounds.
  - **The plan:** real research, best done with the CORE-MATH team.
- **9. Standards.** Proposals and discussion more than code:
  - a standard RISC-V vector math interface (compilers call SLEEF's names
    today);
  - correctly rounded math for WebAssembly;
  - a correctly rounded option in glibc's libmvec;
  - deterministic float math for games that replay the same moves on
    every player's machine (today: fixed point, or streflop).

## Getting them to the people who need them

- **Upstream where there is a home:**
  - PoCL's kernel library;
  - ggml and llama.cpp;
  - NumPy's random module;
  - LLVM's libc (for GPUs);
  - ReproBLAS, if maintained.

  A merged upstream change reaches more people than a new repository.
- **Package where there isn't:** distributions, conda-forge, PyPI wheels,
  crates.io, and npm for WebAssembly. crmvec's recipes (Debian, Fedora,
  Nix, conda) carry over.
- **Earn trust:** publish every check and every control, as crmvec's
  README does.
- **Licence:** MIT, as crmvec and CORE-MATH, so nobody has a reason not
  to use it.

## Constraints

- **Some of this is new mathematics.** crmvec was quick because
  CORE-MATH had done the hard part. A correctly rounded inverse normal
  CDF, complex functions and quad precision have no such foundation yet.
  The reductions do: their algorithms are published.
- **Hardware:** a GPU, a RISC-V board with the vector extension, and an
  AVX-512 machine other than a cloud VM.
- **Maintenance costs more than building.** One kit, one CI and one
  documentation format keep a family of libraries maintainable.
- **Anything outward-facing is the owner's decision:** upstream pull
  requests, proposals, and reports to other projects.
