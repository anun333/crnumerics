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
are marked where unsure; check them before relying on them. Those marked
**(checked 2026-09-30)** were read that day from the project's own
repository or page.

Items 10 to 14, and the extension of item 5, were added on 2026-09-30 from
a list of nine proposed intents; two of the nine were left out ("Considered
and left out", at the end).

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

2026-09-30, first as a folder of crmvec (see
[README.md](README.md)):
- **the checking kit:** formats down to FP8, a correctly rounded MPFR
  reference, runs with controls, verdicts;
- **repro-scan**, the first version of item 8.

Each is checked against answers it did not make, and against
deliberately planted bugs.

## Next, in order (rebuilt 2026-09-30, evening)

Done on 2026-09-30, in brief:
- ival's `hypot`, `atan2` and `pow`;
- crsum, with `crgemv`, `crgemm` and `crgemm_oz` (items 1 and 10);
- crnn's first version (item 2).

README.md and HISTORY.md have the details. Now, with the GCC Compile
Farm's machines available:

1. **Item 10, int8 slices for the Ozaki scheme:** first version **done**
   (2026-10-01): `crgemm_oz8` in crsum, 7-bit signed slices through an
   int8 x int8 -> int32 GEMM (AVX512-VNNI, Arm SDOT or plain C, or the
   caller's), exact, the same bits as `crgemm` (check section 7: 560,304
   elements, 0 differ on all three kernels, natively on cfarm151 and
   cfarm424). At n = 256 it takes 0.061 s on cfarm151 (VNNI) against
   `crgemm_oz`'s 0.121 and `crgemm`'s 0.173; 0.080 s on cfarm424 (SDOT)
   against 0.125 and 0.160 (`crsum/test/bench-oz.c`). **Also done
   (2026-10-01):** an AVX2 kernel (`vpmaddwd` on sign-extended bytes:
   exact) for x86 without VNNI, 19 times plain C on the laptop (512³:
   0.19 s, `crgemm_oz` 0.44, quiet); both x86 kernels blocked two rows by four
   columns, which gained VNNI only about 7% at 512³; `CRSUM_I8_KERNEL`
   picks a kernel for checks (all three pass on cfarm151). **Blocking
   (2026-10-01, AVX2 kernel):** panels of B's rows, about 16 KB, each
   serving every pair of A's rows before the next: 1024³ 1.41 to 1.26 s
   on the laptop (11%), no change at 256³ and 512³, where B fits in L2 and
   the kernel is compute-bound. Section 7 got cases with several panels
   and a short last one; a planted panel bug is caught by them (1,601
   elements) and by nothing before. **The VNNI kernel too** (cfarm151,
   natively): `crgemm_oz8` 512³ 0.315 to 0.254 s (19%), 1024³ 2.13 to 1.54
   (28%), 256³ unchanged (best of two, load about 1); all three kernels
   pass section 7 there, and the planted panel bug is caught through VNNI
   (1,601 elements). **And SDOT**, now blocked two rows by four over the
   same panels: on cfarm424's N1, 256³ 0.070 to 0.045 s, 512³ 0.472 to
   0.246, 1024³ 6.13 to 4.50 (checked there, the plant caught). **And
   I8MM** (16-byte loads paired by zip, four accumulators, the panels):
   on the GB10, 512³ 0.106 to 0.079 s, 1024³ 0.81 to 0.56 (checked there,
   the plant caught). **Blocking along k, measured (2026-10-02):** the
   driver's k-chunk (`OZ8_KC`, 32768 for int32 exactness, now settable at
   compile time) at 4096 instead, so each kernel call's panel fits L1, timed
   by `crsum/test/bench-ozk.c` (m x n x k shapes, best of 3, result = crgemm's):
   on cfarm421 (Zen 3, AVX2, 768 MB L3) 3-15% slower at 64x64x32768 to
   512x512x4096; on cfarm151 (Cascade Lake) 6-7% faster with VNNI, 1-2% with
   AVX2. No clear win, so it stays at 32768; an in-kernel k block (partial
   sums stored between blocks) is the remaining form, if a machine with a
   small L3 shows a gap. **I8MM done
   2026-10-01**: `smmla`, checked on the GB10's Arm cores (all three Arm
   kernels pass; a swapped output lane, planted, is caught), 1.4 to 1.8
   times SDOT; the arm64 CI runner next, if it has I8MM.
2. **Item 2, crnn:**
   - ~~bfloat16, binary16~~ **done 2026-10-01** (all 2^16 inputs checked
     against MPFR; binary16 needs no table, bfloat16 256 entries each for
     SiLU and GELU); FP8 outputs next, from the same binary64 values with a
     table per format;
   - vector code: **first step done 2026-10-01**: through crmvec
     (`CRNN_CRMVEC`), sigmoid and SiLU 1.4 times the naive binary32
     formulas (were 3.7), GELU 1.4 (4.8), softplus 0.6 (1.7), the same bits
     on every input (`make crnn-vsame`); logsumexp and softmax through
     crmvec too (1.2 to 1.5 times faster, the same bits on 20,000
     vectors); next crsum's exact sums, now most of the composites' cost,
     and rsqrt: crmvec's published vector rsqrt is a scalar loop, slower
     than scalar here; its harness now has a vector path (CORE-MATH's fast
     path transcribed, about 2.8 times scalar), and crnn can use it once a
     crmvec release carries it;
   - ~~the exhaustive check on x86-64 too~~ done 2026-10-01 (README:
     cfarm420, then again at `c47566b` on cfarm421, 0 differ).
3. **GPUs, reachable since 2026-09-30:** cfarm107-109 (NVIDIA GB10 and
   Jetson Thor) run CUDA 13 for our account. **First step done
   (2026-10-01):** CORE-MATH's `expf` and `logf`, compiled as CUDA device
   code with nothing changed but `__device__` on each definition, give the
   CPU's bits on all 2^32 inputs on the GB10, where CUDA's own `expf` and
   `logf` differ from the correctly rounded result on 160 and 73 million.
   The same night, CORE-MATH's binary64 `exp`, `log1p`, `erfc` and `rsqrt`
   on the GPU (the CPU's bits on 2^30 inputs each), and **crnn's five
   one-argument functions on the GPU: the CPU's bits on all 2^32 inputs of
   each** (`crnn-fast.h` and the exception table, unchanged but for
   `__device__`). **And crnn's composites** (`crsum.c` and `crnn.c`
   compiled as device code, one GPU thread per vector): logsumexp,
   softmax, layernorm and rmsnorm give the CPU's bits on 2000 vectors (4
   million elements, the hard cases included). **At GPU speed too**
   (softmax and logsumexp, a block per vector, the exact sum by integer
   bins in shared memory and crsum's rounding): the CPU's bits, 0.25 ns an
   element on the GB10, 34 times one CPU thread, layernorm and rmsnorm
   too. **And the Ozaki scheme on
   tensor cores:** `crgemm_oz8` with its int8 GEMMs through cuBLAS gives
   the CPU's bits, a correctly rounded DGEMM (1024³ in 0.35 s against 2.0 on
   one of the GB10's Arm cores (Cortex-X925/A725), the tensor cores 4% of it). **Then
   end to end on the GPU** (the slicing, sums and rounding as kernels
   calling crsum.c's own code): the CPU's bits, a
   correctly rounded DGEMM at 2.6 times cuBLAS's own DGEMM time at 4096³.
   The GPU code (CUDA, built from this repository's sources by a script
   that marks them `__device__`) is not in this repository yet; it comes
   with the next step. **A second vendor, AMD (2026-10-01):** a Radeon Vega
   integrated GPU (gfx90c) through ROCm 5.7's HIP, the same CUDA sources
   compiled by hipcc with a header mapping CUDA's runtime names (and its
   32-lane warp operations emulated in each half of AMD's 64-lane
   wavefront): CORE-MATH's `expf` and `logf` on all 2^32 inputs, its
   binary64 `exp`, `log1p`, `erfc` and `rsqrt` on 2^30 each, crnn's five
   functions on all 2^32 each, and the composites, all give the CPU's bits.
   Not the int8 GEMMs: rocBLAS has no kernels for that GPU, and running its
   gfx900 ones there hung the GPU. Next: a discrete AMD GPU for those, then
   Intel.
4. **ival:** ~~`tgamma`~~ **done 2026-10-02** (README, "ival"):
   - a table of its 185 extrema, between binary64 neighbours and rounded
     both ways, from mpmath (`ival/gen-tgamma.py`);
   - the poles counted exactly;
   - the check's reference finds the extrema its own way, by bisection on
     digamma's sign in MPFR. 36,049 intervals give the tightest
     enclosure, and an extremum's value planted an ulp low is caught.

   **`lgamma` next.** Its segments on the negatives run on to 2^52, with
   minima at no binary64 value, so no table covers them all. Its minima
   are also nowhere near underflow. It needs the extremum located at run
   time to an ulp, or a proof that the binary64 values next to it round
   the same as the minimum. Then binary32.
5. **bfloat16 vector functions,** through crmvec's portable core.
6. **repro-scan and repro-diff:** GPU kernels, Python wheels, more
   conditions.
7. **Packed FP4 and FP6 storage:** the MX spec leaves the layout open, so
   this means choosing a convention (the common hardware ones, to be
   checked).

Alongside, not in the order: a first user. The intake rule applies to
these libraries too, and none has one yet.

**Open decisions:** whether item 1 goes before step 1; the first users
for items 5, 10 and 11 (the intake rule); telling microxcaling's authors
about the scale just below a power of two (`lowp/MX.md`).

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
    correct by construction for all 6 element types (INT8 since
    2026-09-30) and 37 functions.
  - The check uses blocks built with purpose.
  - Done 2026-09-30: the conversion rules checked against OCP's text,
    INT8 elements, and E5M2's saturating mode. Still to do: packed storage
    (two FP4 elements to a byte).
- **Kit additions:** FP6 (E2M3, E3M2) and FP4 (E2M1) are done
  (2026-09-30). E8M0 needs no kit format, since the scale is computed as
  an exponent.

As far as we know, no standard math library exists for FP8 or MX.

**6. Vector interval arithmetic.** First version **done** (2026-09-30),
as ival (README.md): 31 functions in binary64, and `atan2`, `hypot` and
`pow` on boxes, each the tightest enclosure, checked against an
independent MPFR reference; `tgamma` added 2026-10-02. Still to do: `lgamma`, binary32,
and vector code for the directed modes (ival calls CORE-MATH's scalar
functions).
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

**1. Reproducible reductions and BLAS.** First version **done**
(2026-09-30), as crsum (README.md): correctly rounded sums and dot
products, the same bits in any order or thread count, checked against
`mpfr_sum`. Next: speed, then matrix products.
- **The problem:** sums, dot products and matrix products differ with
  thread count, vector width and GPU, whatever the math library does.
- **What exists:** published algorithms (binned summation, as in
  ReproBLAS; exact accumulation, as in ExBLAS or a Kulisch accumulator).
  Both implementations look unmaintained **(checked 2026-09-30)**:
  ReproBLAS's page was last updated in August 2018, and ExBLAS's
  repository was last changed in 2021. Intel's MKL has a reproducibility
  mode within its own CPU families.
- **The plan:** CPU vector code first, GPU through PoCL later.
- **Proof:**
  - the same bits under every permutation, thread count and vector
    width tried;
  - for the exact variant, equality with MPFR's `mpfr_sum`, which is
    correctly rounded.
- **Kit additions:** a many-input reference (`mpfr_sum`) and an order
  shuffler.

**10. FP64 accuracy from low-precision units (the Ozaki scheme).** First
version **done** (2026-09-30), as `crgemm_oz` in crsum: exact products,
correctly rounded, through any binary64 GEMM (README.md). int8 slices next.
- **What:** matrix products accurate to binary64 or better, computed with
  int8 (or FP8) matrix units by splitting each input into slices whose
  products are exact.
- **Why it belongs here:** the slice products are exact integers, so the
  result doesn't depend on summation order. It is reproducible by
  construction, the same property item 1 needs, and it shares item 1's
  accumulators.
- **What exists (checked 2026-09-30):**
  - ozIMMU (MIT, NVIDIA int8 tensor cores; last changed 2025-12);
  - NVIDIA's cuBLAS exposes Ozaki-scheme FP64 emulation (NVIDIA's
    developer blog), vendor-specific;
  - research on FP8 and FP4 slices and on guaranteed accuracy (arXiv
    2508.00441, arXiv 2608.06812, and "Guaranteed DGEMM Accuracy While
    Using Reduced Precision Tensor Cores Through Extensions of the Ozaki
    Scheme", doi:10.1145/3773656.3773670).
- **The gap:** vendor-neutral code, CPU and GPU, with its accuracy stated,
  checked and bit-reproducible.
- **The plan:** CPU first, on int8 dot-product instructions (AVX512-VNNI
  on cfarm151's Cascade Lake, Arm SDOT/I8MM on the Neoverse N2 CI runner,
  RVV), then GPUs through PoCL.
- **Proof:** every slice product exact (checked against integer
  arithmetic); the whole product against MPFR on random and adversarial
  matrices (wide exponent ranges, cancellation); the same bits across
  thread counts and ISAs. Control: one slice dropped must fail.

**11. Verified linear algebra.**
- **What:** Ax = b and eigenvalue problems returning an interval
  guaranteed to contain the exact answer, at close to BLAS speed (Rump's
  midpoint-radius methods), built on items 1, 10 and 6.
- **What exists (checked 2026-09-30):**
  - INTLAB (MATLAB): free for private, academic and in-company use; a
    commercial product that needs it requires a licence from its author.
    Not open source;
  - IntervalLinearAlgebra.jl (Julia, MIT; small, last changed 2026-06);
  - from memory: C-XSC (C++, LGPL, old) and kv (C++).
- **The gap:** a permissively licensed C library at BLAS speed, with
  Python bindings.
- **A known trap:** Rump's methods set the rounding mode upward around
  BLAS calls, but a threaded BLAS's worker threads keep the rounding mode
  they started with, so the enclosure silently stops being one.
  Error-free transformations (items 1 and 10) avoid directed rounding.
- **Proof:** on small systems, the enclosure against the exact rational
  solution; control: an enclosure shrunk by one ulp must fail to contain
  it somewhere.

**2. Specified neural-network primitives.** First version **done**
(2026-09-30), as crnn (README.md): binary32, one-argument functions
correctly rounded (a table for the inputs a binary64 fast path can't
decide), composites specified on crsum's exact sums.
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

**5. Distributions: reproducible sampling, accurate tails, parameter
derivatives.**
- **What already works:** counter-based generators (Philox, Threefry)
  give the same bits everywhere.
- **What breaks it:** the transforms to normal, gamma and other
  distributions call `log`, `sin`/`cos` or the inverse normal CDF.
- **The plan:** those transforms built only on correctly rounded
  functions, plus a correctly rounded inverse normal CDF (`erfinv`,
  `ndtri`). We know of none (unverified).
- **Proof:** the binary32 version by exhaustion (2^32 inputs); the
  binary64 version is research.
- **Extended 2026-09-30 (proposed intents 4 and 5):**
  - CDFs, survival functions and quantiles in log space, correct far into
    the tails (the noncentral t, F and chi-squared, extreme degrees of
    freedom), vectorized;
  - derivatives with respect to the parameters (the incomplete gamma
    function's in its shape, a Bessel function's in its order), which
    probabilistic programming needs;
  - **what exists:** Boost.Math (BSL-1.0, permissive) has the noncentral
    t, F and chi-squared with quantiles **(checked 2026-09-30)**. From
    memory: R's nmath (GPL), and Stan and TensorFlow Probability each
    hand-wrote some parameter derivatives;
  - **the gap:** all of it together: vectorized, log-space throughout,
    with derivatives, accuracy stated and checked, the same bits
    everywhere;
  - **references for the checks:** MPFR (`mpfr_gamma_inc`), and Arb, now
    part of FLINT (LGPL-3.0, checked 2026-09-30), for hypergeometric
    forms;
  - **first users to ask:** Stan, PyMC, NumPyro.

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

**12. Constant-time math and privacy-noise samplers** (proposed intent
9; research).
- **The problem:** floating-point noise in differential privacy leaks
  through its low bits (Mironov, CCS 2012, from memory), and subnormal
  numbers make operations take measurably different time.
- **What exists (checked 2026-09-30):** Google's differential-privacy
  library has secure noise generation (a paper and Go and Java code);
  OpenDP (MIT).
- **A conflict to resolve first:** correctly rounded functions have slow
  paths that depend on the input, and flush-to-zero, which removes the
  subnormal timing, breaks correct rounding (crmvec and CORE-MATH,
  2026-09-30). Constant time and correct rounding together means always
  running the slow path.
- **Before any claim:** a security reviewer.

**13. Robust geometry** (proposed intent 8): check the gap first.
- **What exists (checked 2026-09-30):** Shewchuk's predicates (public
  domain); geogram (BSD-3, with a CSG tool); Manifold (Apache-2.0,
  "geometry library for topological robustness", active).
- So a permissively licensed CPU kernel exists. What may remain: GPU
  predicates, and mesh booleans that give the same bits on every machine.
  Only with a named user.

**14. Complex special functions** (proposed intent 3): only for specific
functions a user asks for. pFq, Bessel functions of complex order and the
rest are a large field; the references would be Arb/FLINT (LGPL-3.0) and
mpmath (BSD-3), both checked 2026-09-30.

## Considered and left out (2026-09-30)

- **Machine-checked proofs for solvers** (ODEs, quadrature, root finding,
  with Coq, Flocq or VCFloat): a different discipline, years per solver.
  The proofs here are by exhaustion and by construction. Validated
  (interval) solvers belong under item 11.
- **Automatic precision tuning** (Herbie, Precimonious, FPTuner): a
  heuristic search whose result depends on the inputs tried, so there is
  nothing to prove. repro-scan's findings could feed such tools.

## Getting them to the people who need them

- **Upstream where there is a home:**
  - PoCL's kernel library;
  - ggml and llama.cpp;
  - NumPy's random module;
  - LLVM's libc (for GPUs);
  - ReproBLAS, which looks unmaintained (its page was last updated in
    2018).

  A merged upstream change reaches more people than a new repository.
- **Package where there isn't:** distributions, conda-forge, PyPI wheels,
  crates.io, and npm for WebAssembly. crmvec's recipes (Debian, Fedora,
  Nix, conda) carry over.
- **Earn trust:** publish every check and every control, as crmvec's
  README does.
- **Licence:** MIT, as crmvec and CORE-MATH, so nobody has a reason not
  to use it.

## Constraints

- **Intake rule (2026-09-30):** an item names its first user or upstream
  home before work starts, and every statement about another project is
  marked checked (with its date and source) or from memory.

- **Some of this is new mathematics.** crmvec was quick because
  CORE-MATH had done the hard part. A correctly rounded inverse normal
  CDF, complex functions and quad precision have no such foundation yet.
  The reductions do: their algorithms are published.
- **Hardware:** a GPU, a RISC-V board with the vector extension, and an
  AVX-512 machine other than a cloud VM.
- **Maintenance costs more than building.** One kit, one CI and one
  documentation format keep a family of libraries maintainable.
- **Upstream work is paced:** pull requests, proposals and reports to
  other projects go one at a time, each checked before it is sent.
