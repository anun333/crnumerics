# Roadmap

The aim: numerics that give the same bits on every machine (every CPU,
every instruction set, and GPUs), with proofs a stranger can rerun.
Correctly rounded functions, crmvec's job, are necessary for that but not
sufficient. Most differences between machines come from elsewhere:
- summation order;
- fast approximations inside larger operations;
- formats with no standard math library;
- random-number transforms built on inexact functions.

This page lists the libraries that close those gaps: where each stands,
what comes next, and the plan behind each. What was done, and when, is in
[HISTORY.md](HISTORY.md), and how each library works and is checked is in
its own README.

Statements about other projects are from memory as of 2026-09 unless
marked **(checked, with a date)**, when they were read from the project's
own repository or page; check them before relying on them. Items 10 to 14,
and the extension of item 5, came on 2026-09-30 from a list of nine
proposed intents; two of the nine were left out (at the end).

## Where correctly rounded math stands

- **Scalar, on CPUs: largely solved.** CORE-MATH did the hard part, and
  glibc is importing it **(checked 2026-10-02, glibc's NEWS)**:
  - 2.41 made 23 float functions correctly rounded;
  - 2.42 added the float pi functions;
  - 2.43 double `acosh`, `asinh`, `atanh`, `erf`, `erfc`, `lgamma` and
    `tgamma`;
  - 2.44 `cosh`, `sinh` and `tanh`;
  - 2.45, in development, `cbrt`.

  Double `exp`, `log`, `sin`, `cos` and `pow` are not among them yet.
  LLVM's libc uses CORE-MATH too (how widely is unchecked). C23 reserves
  `cr_`-prefixed names for correctly rounded functions.
- **Vector, on CPUs:** crmvec, a drop-in libmvec and SLEEF replacement.
  As far as we know it is the only correctly rounded one.
- **GPUs: mostly missing.** The vendor math libraries (CUDA's, AMD's) and
  the Metal and Vulkan specifications allow errors of an ulp or more.
  LLVM's libc builds for GPUs and may carry some correctly rounded
  functions; its coverage is unverified here. Our experiments so far, on
  NVIDIA and AMD, are in [GPU.md](GPU.md).

## Where each item stands (2026-10-14)

Releases are crnumerics' (v0.1.0 to v0.9.0). The packages are on PyPI.

| # | Item | Library | State | Next | First user |
|---|---|---|---|---|---|
| 1 | Reproducible reductions and BLAS | [crsum](crsum/README.md) | in every release: correctly rounded sums, dot and matrix products in any order | the GPU code into this repository ([GPU.md](GPU.md)) | none named |
| 2 | Specified neural-network primitives | [crnn](nn/README.md) | in every release: five functions correctly rounded in binary32, binary16 and bfloat16; composites on crsum; a vector path through crmvec | FP8 outputs | none named |
| 3 | Integer and fixed-point math | [crfix](fix/README.md) | released 0.7.0 to 0.9.0; `pip install crfix` | a first user | none named |
| 4 | FP8, bfloat16 and MX math | [lowp](lowp/README.md) | FP8 and MX in every release | bfloat16 vector functions; packed FP4 and FP6 | none named |
| 5 | Distributions | [crdist](dist/README.md) | released 0.3.0 to 0.5.0; `pip install crdistr` | other distributions, derivatives, binary64 (on a user's request) | JAX, to look at |
| 6 | Vector interval arithmetic | [ival](ival/README.md) | a full IEEE 1788.1 library since 0.1.0; `pip install crival` | `lgamma`; binary32, Windows | IBEX; pyinterval's users |
| 7 | Double-double, quad, complex | | not started | research | |
| 8 | Reproducibility sanitizer | [repro-scan, repro-diff](tools/README.md) | in every release | GPU kernels, Python wheels, more conditions | none named |
| 9 | Standards | | not started | proposals | |
| 10 | The Ozaki scheme | [crsum](crsum/README.md) | `crgemm_oz` through any binary64 BLAS, `crgemm_oz8` on int8 units (CPU, and tensor cores in [GPU.md](GPU.md)) | vendor-neutral GPU code in this repository | none named |
| 11 | Verified linear algebra | [crlin](lin/README.md) | released 0.6.0 and 0.7.0; `pip install crlin` | nested dissection; the exact mode through an int8 GEMM | none named |
| 12 | Constant-time math, privacy noise | | research, not started | | |
| 13 | Robust geometry | | only with a named user | | |
| 14 | Complex special functions | | only for a function a user asks for | | |

## Next, in order

1. **First users** for what is released: crfix, crlin and crdist have
   none named, and ival's (IBEX) waits on an offer. Each approach to
   another project is the owner's call (see Constraints).
2. **crnn:** FP8 outputs, from the same binary64 values with a table per
   format; layernorm and rmsnorm through crmvec.
3. **GPUs:** the CUDA and HIP code into this repository, built and checked
   as the CPU code is; then a discrete AMD GPU for the int8 GEMMs, and
   Intel ([GPU.md](GPU.md)).
4. **ival:** `lgamma`. Its segments on the negatives run on to 2^52, with
   minima at no binary64 value, so no table covers them all, and the
   minima are nowhere near underflow. It needs each extremum located at
   run time to an ulp, or a proof that the binary64 values next to it
   round as the minimum does. binary32 and Windows when a user asks.
5. **bfloat16 vector functions,** through crmvec's portable core (item 4).
6. **repro-scan and repro-diff:** GPU kernels, Python wheels, more
   conditions (item 8).
7. **Packed FP4 and FP6 storage:** the MX spec leaves the layout open, so
   this means choosing a convention (the common hardware ones, to be
   checked).

## Decisions and exceptions

- **Built ahead of a first user,** as exceptions to the intake rule (see
  Constraints), each at the owner's direction:
  - ival as a full IEEE 1788.1 library (2026-10-08);
  - crdist (2026-10-13);
  - crlin and its nine-part build list (2026-10-13);
  - crfix and its plan (2026-10-13 and 14).

  Their first users are the open item now, not more scope. ival named
  IBEX and Python's pyinterval users on 2026-10-09. In Julia ival adds
  only speed, since IntervalArithmetic.jl's `:correct` gives the same
  bounds (measured).
- **Open, each the owner's call:**
  - the offer to IBEX's maintainers, which needs a public branch on
    IBEX's master ([ival/README.md](ival/README.md));
  - a note to IntervalArithmetic.jl's maintainers (drafted, not posted);
  - a note to JAX about crdist (JAX's `random.normal` is
    `sqrt(2) * erf_inv(uniform)`);
  - telling microxcaling's authors about the scale just below a power of
    two ([lowp/MX.md](lowp/MX.md)).

## The plan, item by item

### Phase 1: small, and quick to prove

**4. FP8, bfloat16 and MX math.**
- **FP8 (E4M3, E5M2): done** (2026-09-30), as lowp: 41 functions and the
  conversions, every rounding mode and both of E4M3's overflow modes,
  proven on every input and every pair.
- **MX block formats** (a shared E8M0 scale over FP8, FP6 or FP4
  elements): **done** (2026-09-30), in lowp. `lowp/MX.md` defines a
  correctly rounded function per block, with the scale from the exact
  results. lowp computes it by rounding CORE-MATH's binary64 results to
  odd, which is correct by construction for all 6 element types and 37
  functions.
- **Left:**
  - **bfloat16:** vector versions of CORE-MATH's scalar bfloat16
    functions, through crmvec's portable core.
  - **Packed storage:** two FP4 elements to a byte.

As far as we know, no standard math library exists for FP8 or MX.

**6. Vector interval arithmetic.**
- **Why it's close:** crmvec is correctly rounded in all four modes, so
  rounding down and up gives both ends of an interval.
- **What exists:** MPFI, filib++ and the IEEE 1788 reference, mostly
  scalar.
- **Done,** as ival: a full IEEE 1788.1 library, with the tightest
  enclosures of 32 functions and the arithmetic. It has an accurate mode
  vectorized through crmvec, and passes ITF1788's tests. One interval at a
  time is cheap (an add in 4.5 to 10 ns). It runs from C and from Python
  (crival), on Linux and macOS ([ival/README.md](ival/README.md)).
- **Left:**
  - `lgamma` (Next, item 4);
  - binary32;
  - vector code for the directed modes (crmvec's vector code runs to
    nearest);
  - Windows.

**8. A reproducibility sanitizer.**
- **Done:**
  - repro-scan reads ELF binaries for x86-64, AArch64 and RISC-V, and
    compiler options from `compile_commands.json`.
  - repro-diff runs a program under changed conditions (threads,
    flush-to-zero, an older CPU, a repeat) and compares its output. On
    2026-09-30 it showed glibc 2.39's `exp`, `sin`, `pow` and others
    giving other bits without FMA.
- **Next:**
  - GPU code: PTX, AMDGPU and SPIR-V kernels (estimate instructions, fast
    division and square root, `-ffast-math`-style flags);
  - Python wheels: scan the shared objects inside;
  - repro-diff: more conditions, such as another ISA under qemu (the same
    source built twice), and a vector width forced lower.

Related tools cover accuracy more than reproducibility: Verrou,
FPChecker, Herbie.

### Phase 2: reproducible machine learning and simulation

**1. Reproducible reductions and BLAS.**
- **The problem:** sums, dot products and matrix products differ with
  thread count, vector width and GPU, whatever the math library does.
- **What exists:** published algorithms (binned summation, as in
  ReproBLAS; exact accumulation, as in ExBLAS or a Kulisch accumulator).
  Both implementations look unmaintained **(checked 2026-09-30)**:
  ReproBLAS's page was last updated in August 2018, and ExBLAS's
  repository was last changed in 2021. Intel's MKL has a reproducibility
  mode within its own CPU families.
- **Done,** as crsum: correctly rounded sums, dot products and matrix
  products (`crgemv`, `crgemm`), the same bits in any order, split or
  thread count, checked against `mpfr_sum`. Its kernels and timings are
  in [crsum/README.md](crsum/README.md).
- **Next:** the GPU code into this repository ([GPU.md](GPU.md)).

**2. Specified neural-network primitives.**
- **What:** softmax, log-sum-exp, GELU, SiLU, sigmoid, layer norm, RMS
  norm and rsqrt, where fast approximations usually live.
- **How each is specified:** one function is correctly rounded; a
  composite is a fixed sequence of correctly rounded operations and item
  1's reductions.
- **Done,** as crnn ([nn/README.md](nn/README.md)): five functions
  correctly rounded in binary32, binary16 and bfloat16, every 16-bit input
  and every binary32 input checked. The composites are on crsum's exact
  sums. A vector path through crmvec gives the same bits. On GPUs, the
  CPU's bits ([GPU.md](GPU.md)).
- **Next:** FP8 outputs; layernorm and rmsnorm through crmvec.

**3. Integer and fixed-point math.**
- **What:** exp, log, sigmoid, tanh, division, rsqrt and the activations
  for integer-only kernels.
- **What exists:** gemmlowp, CMSIS-NN and TFLite's integer code each have
  some of this, without stated error bounds or checks across ISAs.
- **Proof:** 8- and 16-bit inputs, exhaustively against MPFR.
- **Done,** as crfix ([fix/README.md](fix/README.md)):
  - **The functions:** exp, exp2, log, log2, sigmoid, tanh, SiLU, the exact
    GELU, sqrt, rsqrt, reciprocal and division, in integer arithmetic
    only. They are proven on every 16-bit input and checked on samples
    beyond, with a slow path that is correct by construction for the
    cases the fast one leaves.
  - **Formats:** TFLite's affine formats with uint8, and softmax.
  - **Layer normalization,** computed exactly.
  - **Microcontrollers:** the same bits on a Cortex-M0.

  Released in 0.7.0 to 0.9.0, and on PyPI.
- **Next:** a first user.

**5. Distributions: reproducible sampling, accurate tails, parameter
derivatives.**
- **What already works:** counter-based generators (Philox, Threefry)
  give the same bits everywhere.
- **What breaks it:** the transforms to normal, gamma and other
  distributions call `log`, `sin`/`cos` or the inverse normal CDF.
- **The plan:** those transforms built only on correctly rounded
  functions, plus a correctly rounded inverse normal CDF (`erfinv`,
  `ndtri`). We know of none elsewhere: CORE-MATH and RLIBM have no
  inverse of erf, and Intel IPP's best is within an ulp **(checked
  2026-10-12 and 13)**.
- **Done,** as crdist ([dist/README.md](dist/README.md)):
  - **The normal distribution:** `erfinv`, `erfcinv`, `probit`, `ndtr`,
    `log_ndtr`, `normpdf` and `logit`, correctly rounded in binary32,
    binary16 and bfloat16. Every input is checked on x86-64 and aarch64.
  - **The samplers:** normal, exponential, gamma, beta, chi-squared, t,
    Poisson and binomial, from a Philox stream. A written specification
    fixes their bits.
  - **Proof elsewhere:** `crdist-check hash-all` carries the every-input
    proof to another build in minutes.
- **First user to look at:** JAX, whose `random.normal` is
  `sqrt(2) * erf_inv(uniform)` **(checked 2026-10-13, jax e0f38af4d2)**.
  PyTorch's `trunc_normal_`, once `erfinv`-based, now samples by rejection
  (checked the same day).
- **The extension (proposed intents 4 and 5, 2026-09-30), not started:**
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
- **binary64:** research.

**10. FP64 accuracy from low-precision units (the Ozaki scheme).**
- **What:** matrix products accurate to binary64 or better, computed with
  int8 (or FP8) matrix units by splitting each input into slices whose
  products are exact.
- **Why it belongs here:** the slice products are exact integers, so the
  result doesn't depend on summation order. It is reproducible by
  construction, the property item 1 needs, and it shares item 1's
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
- **Done,** in crsum:
  - `crgemm_oz`: exact products, correctly rounded, through any binary64
    GEMM;
  - `crgemm_oz8`: 7-bit slices through int8 GEMMs (kernels for
    AVX512-VNNI, AVX2, Arm SDOT and I8MM, or the caller's), the same bits
    as `crgemm`;
  - through cuBLAS on tensor cores, a correctly rounded DGEMM
    ([GPU.md](GPU.md)).
- **Next:** that GPU code in this repository, and other vendors.

**11. Verified linear algebra.**
- **What:** Ax = b and eigenvalue problems returning an interval
  guaranteed to contain the exact answer, at close to BLAS speed (Rump's
  midpoint-radius methods), built on items 1, 10 and 6.
- **What exists (checked 2026-09-30, again 2026-10-12):**
  - INTLAB (MATLAB): free for private, academic and in-company use; a
    commercial product that needs it requires a licence from its author.
    Not open source;
  - IntervalLinearAlgebra.jl (Julia, MIT; small, last changed 2026-06);
  - C-XSC (C++, LGPL): its last release is 2.5.4, from 2014, still
    packaged by Fedora and Arch;
  - FLINT's `arb_mat_solve` (Arb's ball arithmetic, C, LGPL; from Python
    through python-flint): rigorous, and measured 2026-10-12 at 140 to
    450 times the time of `numpy.linalg.solve` on random
    well-conditioned dense systems, both on one thread;
  - in Python, pyintval and verry have no linear solver;
  - from memory: kv (C++).
- **The gap:** a maintained, permissively licensed C library with Python
  bindings that encloses a solution at a few times a plain solve's cost.
- **A known trap:** Rump's methods set the rounding mode upward around
  BLAS calls, but a threaded BLAS's worker threads keep the rounding mode
  they started with, so the enclosure silently stops being one.
  Error-free transformations (items 1 and 10) avoid directed rounding.
- **Done,** as crlin ([lin/README.md](lin/README.md)):
  - **The method:** Rump's, with every bound from crsum's exact products
    or an a priori bound in round-to-nearest, so it never switches the
    rounding mode. It returns an enclosure about an ulp wide, or a
    refusal.
  - **Cost:** 8 to 17 times numpy's `solve`, against Arb's 113 to 470.
  - **The rest:** several right-hand sides, interval data, very
    ill-conditioned systems, least squares, symmetric eigenvalues, proofs
    of positive definiteness, and banded and sparse systems.

  Released in 0.6.0 and 0.7.0, and on PyPI.
- **Next:** nested dissection for sparse fill; the exact mode through an
  int8 GEMM.
- **Proof:** on small systems, the enclosure against the exact rational
  solution; control: an enclosure shrunk by one ulp must fail to contain
  it somewhere.

### Phase 3: research, hardware and standards

- **GPU builds:** experiments run on NVIDIA and AMD with the CPU's bits
  ([GPU.md](GPU.md)). Next, that code in this repository.
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
    every player's machine (today: fixed point, or streflop; crfix's
    integer functions are one route).

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
- **Package where there isn't:**
  - **Done:** PyPI wheels for crival, crdistr, crlin and crfix (x86-64
    and aarch64 Linux, Apple Silicon macOS), and crmvec on conda-forge.
  - **Not yet:** distribution packages for crnumerics' libraries,
    crates.io, and npm for WebAssembly. crmvec's recipes (Debian, Fedora,
    Nix, conda) carry over.
- **Earn trust:** publish every check and every control, as crmvec's
  README does.
- **Licence:** MIT, as crmvec and CORE-MATH, so nobody has a reason not
  to use it.

## Constraints

- **Intake rule (2026-09-30):** an item names its first user or upstream
  home before work starts, and every statement about another project is
  marked checked (with its date and source) or from memory. The
  exceptions so far are under "Decisions and exceptions".
- **Some of this is new mathematics.** crmvec was quick because
  CORE-MATH had done the hard part. Complex functions and quad precision
  have no such foundation yet. (The correctly rounded inverse normal CDF,
  once on this list, is done in binary32.)
- **Hardware, from the GCC Compile Farm:**
  - **Have:** NVIDIA GPUs (GB10 and Jetson Thor, cfarm107 to cfarm109);
    a RISC-V board with the vector extension (cfarm95, a SpacemiT K1,
    VLEN 256); AVX-512 on cfarm151's Cascade Lake, but as shared virtual
    machines, too noisy for timing.
  - **Still wanted:** a discrete AMD GPU, an Intel GPU, and an AVX-512
    machine that is not a cloud VM.
- **Maintenance costs more than building.** One kit, one CI and one
  documentation format keep a family of libraries maintainable.
- **Upstream work is paced:** pull requests, proposals and reports to
  other projects go one at a time, each checked before it is sent, and
  each is the owner's call.
