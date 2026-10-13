# History

**2026-10-13, after 0.3.0: crdist in binary16 and bfloat16, installed, and
from Python.**
- **binary16 and bfloat16:** `crdist_erfinv_f16` and `_bf16`, likewise for
  `erfcinv` and `probit`, on bit patterns, correctly rounded in the format
  from the same binary64 values with a rounding test per format; no input of
  either format is undecided, so the tables are empty. `crdist-check` tries
  all 2^16 inputs of each against MPFR, also under round-upward with
  flush-to-zero: 0 differ. Rounding the binary32 result again would have
  been wrong on 5 inputs.
- **Installed:** `make install` puts `libcrdist.so.0` (the soname now
  versioned), `libcrdist.a`, `crdist.h` and `crdist.pc` under `PREFIX`
  (Linux); `make check` builds against an installed copy through
  pkg-config, shared, static and C++; `crdist_version()`.
- **From Python:** `dist/python`, the module `crdist` over NumPy arrays,
  the format following the dtype; `make crdist-python-check` against mpmath,
  in CI on x86-64 and arm64. A planted error of 2^-30 in probit's sqrt 2
  failed its float32 test. Its PyPI name is not chosen ("crdist" is taken).

**2026-10-13: crnumerics 0.3.0** (tag `v0.3.0`). crdist: binary32 `erfinv`,
`erfcinv` and `probit`, correctly rounded and checked on every input (below).
ival, crsum, crnn and lowp unchanged; ival's soname is still `libival.so.0`,
and no crival upload, as its results did not change. The candidate passed
`make check` and ival's install check on cfarm421 (x86-64) and cfarm424
(aarch64) and CI; the tag's archive is the commit file for file.

**2026-10-12 to 13, after 0.2.2: crdist, the inverse normal distribution,
correctly rounded** (`dist/`). `crdist_erfinvf`,
`crdist_erfcinvf` and `crdist_probitf` (the standard normal quantile, R's
`qnorm`, SciPy's `ndtri`) in binary32, correctly rounded to nearest on
every input; CORE-MATH and RLIBM have no inverse of erf, and the best vendor
accuracy found is Intel IPP's, within an ulp. JAX's `random.normal` is
`sqrt(2) * erf_inv(uniform)` (jax e0f38af4d2), the first user to look at.
- **Computed** in binary64: up to 1/2, erfinv(x) = x P(x^2) (degree 13); in the
  tail, the inverse of erfc as six degree-14 pieces in s = sqrt(-log q), q
  down to 2^-149. The polynomials come from `dist/fit.py` (mpmath, Chebyshev
  nodes), and CORE-MATH's `log` is the only function called. Within 2^-50;
  rounded to binary32 when that can't round two ways, else a table made
  from all 2^32 inputs with MPFR: 20, 52 and 46 entries, byte-identical when
  generated on x86-64 with gcc 13 and 14 and on aarch64 with gcc 14. On 3 of
  them, plainly rounding the binary64 value would be wrong. 36 to 54 ns an
  element (Ryzen 5 PRO 5650U).
- **The references:** MPFR has no inverse of erf, so `crdist-ref.c` compares
  the root with exact midpoints, by erf, erfc or erfc(m/sqrt 2) under
  directed roundings from 64 bits up; a bisection with no start agrees.
- **Proven on every input,** `crdist-check all` on cfarm421 (x86-64) and
  cfarm424 (aarch64), gcc 14.2.0: 3 x 2^32 inputs, 0 differ, and the kit's
  control 67,111,000 differ for each function (2 h 27 min on 64 threads; 6 h
  6 min on 32). The sampled check runs in `make check`. The binary64 bound,
  `crdist-check bound-all` on cfarm421: within 2^-50 on every input inside
  each domain; only 51 inputs, all of `probit`, lie beyond 2^-51.
- **The first version,** the same evening, started from Winitzki's
  approximation and took three to five Newton steps through CORE-MATH's erf
  or erfc and exp: within 2^-51 and a 51-entry table, identical on both
  hosts, but 290 to 880 ns an element. Its generator measured the start at
  2.26e-3 of the result in the tail, not the 2e-3 its comment had assumed.
  The polynomials replaced it before its exhaustive runs finished.

**2026-10-12: crnumerics 0.2.2** (tag `v0.2.2`). CORE-MATH ea7a421 inside ival
and the kit, with the same results, and the bfloat16 references right under
GCC 13 on AArch64.
- **CORE-MATH ea7a421:** ten of the carried functions updated (below); old
  against new, the same results and flags on 686,469,600 calls on x86-64 and
  aarch64, every worst case included.
- **GCC 13 on AArch64:** the vendored bfloat16 functions, the self-test's
  references, built right with the fp16 extension on (below).
- **ival from Julia:** `make ival-julia-ccall` and `make ival-julia-threads`
  (below).

**2026-10-12, in 0.2.2: CORE-MATH ea7a421.** CORE-MATH
ea7a421's cosh, cospi, exp, exp10, exp2, log10, log2, sinh, sinpi and tanpi
(upstream's source-coverage work since 040ee48, then 10-12's sinh type fix,
log2 patch and tanpi change). Old (040ee48) against new in all four rounding
modes: the same results and flags on 686,469,600 calls on x86-64 and on
aarch64, among them every input of the ten functions' worst-case files and its
negative (1,922,435). 81a899d's sinh, which narrowed a variable to `int`, gave
the same results as ea7a421's on the same calls: it cost time, not results.
All 131 carried files equal ea7a421.

**2026-10-11, after 0.2.1: the bfloat16 references under GCC 13 on AArch64.**
Built by GCC 13 with AArch64's fp16 extension (`-march=native` on most
current cores, so Ubuntu 24.04's default compiler on such a machine), 17 of
the vendored CORE-MATH bfloat16 functions were wrong and the kit's self-test
said so (cos(0) = 0.0078125; cospi on 124,424 of its 262,144 inputs and
modes). GCC loads a `__bf16` constant with FMOV's half-precision immediate,
which encodes binary16, so 1.0 arrives as 0x3c00: GCC PR111867, fixed in
GCC 14 and never backported. The Makefile now compiles `core-math/bf16/`
with `core-math/gcc13-bf16-fix.h`, which turns fp16 off for those files
under GCC before 14, and GCC 13 then loads the constant from memory. On
cfarm424 (Debian gcc 13.3.0, `-march=armv8.2-a+fp16`) the self-test went
from DIFFERS to IDENTICAL; gcc 13 without fp16 and gcc 14 with it were
IDENTICAL before and after. The arm64 CI job now builds the self-test with
fp16 too. Outside `core-math/bf16/` only the self-test uses `__bf16`, and its
own bfloat16 checks passed in the faulty build.

**2026-10-11, after 0.2.1:** `make ival-julia-ccall` (`ival/julia/ccall-check.jl`).
ival called from Julia through `ccall`: on 4,096 intervals each of exp, exp2,
exp10, tanh, cbrt, asinh, acosh and atanh, every bound is the same as
IntervalArithmetic.jl's `:correct` (1.0.12, Julia 1.11.7), and Julia is in
round-to-nearest before and after every call (libc's `fegetround`, and an
addition that only nearest rounds to 1.0). ival changes the rounding mode
only inside its own C code, so nothing LLVM compiles for Julia runs in
another mode. The script first checks that its checks can fail.
`make ival-julia-threads` (`ival/julia/threads-check.jl`) does the same from
several Julia threads at once: with 3 and with 7 worker threads, 25,600
calls of 4,096 intervals gave every bound of `:correct`, while a sentinel
on another thread checked its own rounding mode 86 to 152 million times
and always found round-to-nearest. The mode is per thread (MXCSR, FPCR).

**2026-10-09 to 10: crnumerics 0.2.1** (tag `v0.2.1`). ival on macOS (Apple
Silicon).
- **macOS:** `make` builds `libival.a` and `libival.dylib` with Apple
  clang; a macos-14 CI job runs every ival check, ITF1788 (7,451 tests
  tight) and the Python binding. On macOS the build needs no objcopy (Apple's
  linker hides the CORE-MATH symbols itself), runs one check without
  OpenMP, and gives the thread check its own barrier.
- **`make install` on macOS:** `libival.<version>.dylib` with its links,
  its install name LIBDIR's, and `ival.pc`; the install check asks
  `otool` and `nm` there. Python wheels for macOS 11 and later.
- **Two faults the port found, both giving wrong intervals on macOS only:**
  - sqrRev's square roots rounded down and up came out to nearest, so a
    double just beyond √c was kept: Apple clang 15, which keeps no
    floating-point order on AArch64, moved the inlined roots past the
    rounding-mode changes. Each root is now stored through a volatile
    before the next change.
  - text was read to nearest, not outward: macOS's `strtod` ignores the
    rounding mode, which the parser relied on. Numbers are now rounded by
    deciding exactly, with big integers, on which side of the exact value
    each neighbouring double lies; no C library's rounding is relied on.

**2026-10-09, after 0.1.0: crnumerics 0.2.0** (tag `v0.2.0`). ival one
interval at a time; a fix to the periodic reverses; Python.
- **Fixed: sinRev, cosRev and tanRev could miss a set,** returning the
  empty set or too little, when a crossing of C's end lay within a few
  ulps of X's end. 0.1.0 gave the empty set for cosRev of
  [sin 0.5, sin 1.5] in [0.5 + π/2, π − 1.6 + π/2], where the set is
  about [2.0708, 3.0708]. Two faults: whether f at X's end lies above C
  was read from f rounded down (all three now read it rounded up; it
  broke sinRev and cosRev), and the crossing's estimate could fall a
  period short (it broke all three). Found by IBEX's tests; rev-check
  now puts X's ends on and next to the crossings, and planted bugs 43 and
  44 put each fault back.
- **A call on one interval** costs 20 to 60 ns, not 165 to 186: on x86-64
  ival saves and sets MXCSR alone (CORE-MATH's objects get their fenv
  calls renamed to MXCSR versions, since glibc's read the x87 unit); on
  aarch64 FPCR and FPSR directly; small arrays take the scalar code.
  `env-check` shows every call leaving the caller's rounding mode, flags,
  MXCSR and x87 words as they were.
- **`ival-scalar.h`,** installed: the arithmetic on one interval, inline
  and bit for bit the library's. To nearest by error-free
  transformations; rounding upward, as Gaol-style interval classes keep
  it, one operation a bound (4.5 ns an add, 7.7 a product); in any other
  mode, the library. `ival1_scale` multiplies a point by an interval in
  two products. The mode check sits behind a volatile barrier, and
  `mode-change-check`, built without `-frounding-math` as most callers
  are, changes the mode between calls (planted bug 47 fails it).
- **Integer powers** without pow when a double-double product decides
  them (2 ≤ p ≤ 64, error at most p·2^−100): pown 3 to 6 times faster,
  pownRev 2.
- **Against IntervalArithmetic.jl** (`make ival-julia-compare`): the same
  bounds bit for bit as its `:correct` rounding; exp2 and tanh 50 to 100
  times faster.
- **Python** (`ival/python`): a binding over NumPy arrays (ctypes). Numbers
  that are not doubles round outward. 27 tests, in `make
  ival-python-check` and CI. Against pyinterval it gives the same tight
  bounds on 11 functions, and is 6 times faster one interval at a time.
  `build-wheel.sh` makes a wheel that carries `libival.so.0`. On PyPI as
  `crival` (`pip install crival`, then `import ival`), wheels for Linux
  x86-64 and aarch64.
- **IBEX** (outside this repository): an `INTERVAL_LIB=ival` backend
  passes 61 of IBEX's 62 tests. The one failure asserts bit-identical
  Jacobians along two evaluation orders, and ival is one ulp tighter on
  one. On IBEX's 264 solver benchmarks it takes 28% less time than filib
  on aarch64 and 23% more than Gaol on x86-64 (ROADMAP item 4).

**2026-10-08 to 09.** ival to a full IEEE 1788.1 library;
**crnumerics 0.1.0** (tag `v0.1.0`, the first).
- **Arithmetic:** add, sub, mul, div, neg, sqr, recip, sqrt and fma, the
  tightest. Error-free transformations are the reference, and arrays run
  with the rounding mode set per block; four lanes at a time with AVX2
  and FMA, bit for bit. The caller's flush modes are off inside.
- **The rest of 1788.1:**
  - the roundings, abs, sign, min, max, the set operations, cancel,
    the numbers (mid, rad, ...), the predicates and overlap;
  - pown for every real x, rootn;
  - mulRev and mulRevToPair, and the reverse operations of sqr, abs,
    cosh, pown, sin, cos and tan, with powRev1 and powRev2;
  - numsToInterval and textToInterval, with every bound exact;
  - the reductions, through crsum.
- **The functions** switch the rounding mode per block, not per interval
  (exp 6 times faster, sin and cos 3 to 4). The accurate mode
  (`ival_acc_f`, one ulp at most) runs through crmvec's vector functions
  for all 32 functions and atan2, hypot and pow.
- **Checks:**
  - ITF1788, IEEE 1788's test suite, through a converter: 7,451 tests of
    ival's operations, every result tight. Expectations that are not the
    tightest are corrected, each with a proof.
  - The periodic reverses against an MPFR reference.
  - Thread safety, checked.
  - A CI job under ASan, UBSan and TSan.
  - Comparisons with MPFI, Boost.Interval, filib++ and libieeep1788.
- **CORE-MATH 040ee48:** sin.c, exp.c, log/log.c, f16/cbrtf16.c and
  bf16/acos_bf16.c.
- **`make install`:** `libival.so.0`, `libival.a`, `ival.h`, `ival.pc`,
  checked by building against the installed copy. The version is
  `IVAL_VERSION` in `ival.h`.

**2026-10-06.** CORE-MATH e78b460: cospi.c, sinpi.c, sin.c, pow/pow.c,
pow/pow.h and f16/cbrtf16.c. Upstream fixed the cbrtf16 finding (398b235).

**2026-10-05.** README: an overview, with each library's section moved
unchanged to its directory's `README.md`.

**2026-10-02.** ival's `tgamma`.
- **ival:** `ival_tgamma` over any interval, the tightest binary64
  enclosure:
  - the poles counted exactly; the extrema from a table of 185
    (`ival/gen-tgamma.py`), each between its binary64 neighbours and
    rounded both ways;
  - past −184, where every binary64 value has |Γ| < 2^−1074, the extremum
    rounds to ±0.
- **ival's check:** a reference for tgamma that finds the extrema by
  bisection on digamma's sign in MPFR, with signs from lgamma (Γ underflows
  MPFR's own exponent range past n = 180).
  - It caught one bug: [k − 1, k] beyond 2^50, both ends poles, had given
    [+∞, +∞]. That is now [+0, +∞].
  - A planted error, an extremum's value an ulp low, is caught. An
    extremum misplaced by one neighbour cannot show: Γ is flat there.
  - `IVAL_ONLY=f` runs one function alone.
- **crsum:** `OZ8_KC` settable at compile time, and `bench-ozk` for m × n ×
  k shapes. Blocking along k measured, no clear win (ROADMAP item 1).

**2026-10-01.** The Ozaki scheme on int8 units.
- **crsum:** `crgemm_oz8`, the same exact scheme as `crgemm_oz` with 7-bit
  slices on integer dot-product instructions instead of binary64 GEMMs:
  every slice product an exact int32 (k in chunks of 32,768), the sums
  added in int64, and `crgemm_oz`'s rounding step, now shared. Checked
  against `crgemm` bit for bit through the internal kernel (AVX512-VNNI on
  cfarm151, Arm SDOT on cfarm424, plain C on the laptop), plain C, a
  scrambled order and a counting GEMM; a GEMM dropping one term differs.
  With VNNI it takes half `crgemm_oz`'s time at n = 256.
- **crnn:** the five one-argument functions in binary16 and bfloat16,
  correctly rounded on all 2^16 inputs of each (checked against MPFR,
  and under round-upward with flush-to-zero). binary16 needs no exception
  table; bfloat16 needs 256 entries each for SiLU and GELU, and the check
  catches their removal.
- **crnn on an NVIDIA GPU:** its five functions (all 2^32 inputs each) and
  its composites (crsum and crnn compiled as CUDA device code, 2000
  vectors) give the CPU's bits on a GB10.
- **crgemm_oz8's kernels:** AVX2 for x86 without VNNI (bytes widened to
  16 bits, `vpmaddwd`), 19 times plain C; the x86 kernels blocked;
  `CRSUM_I8_KERNEL` to pick one. Two planted kernel bugs (a wrong column
  in an edge block, the k tail dropped) were caught by the check. Then an
  Arm I8MM kernel (`smmla`), 1.4 to 1.8 times SDOT on the GB10's cores.
- **The x86 int8 kernels over panels of B's rows** (about 16 KB, L1):
  `crgemm_oz8` at 1024³ 28% faster with VNNI on cfarm151 and 11% with AVX2
  on the laptop (19% and none at 512³). The check got cases with several
  panels; a planted panel bug is caught by them and by nothing before.
- **The SDOT kernel blocked** like the others (two rows by four, over the
  same panels): `crgemm_oz8` 1.4 to 1.9 times faster on a Neoverse N1.
- **The I8MM kernel reworked** (16-byte loads, four accumulators, the
  panels): `crgemm_oz8` 1.16 to 1.45 times faster on the GB10.
- **crnn through crmvec:** an optional vector path (`CRNN_CRMVEC`) for
  sigmoid, SiLU, GELU and softplus, 2.5 to 3.5 times faster, the same bits
  on all 2^32 inputs of each (hashed both ways), because the proof needs
  only correct rounding.
- **crsum_add 3 times faster** on short blocks: thread-local bins and a
  touched-groups mask instead of allocating, clearing and scanning 4096
  bins per call; crnn's composites gain with it.
- **crblas:** `dgemm_` and `dgemm_64_` through `crgemm_oz`, for programs
  that already call a BLAS. Julia's products, through its
  libblastrampoline with Julia's own OpenBLAS inside, become the exact
  products rounded once (`make julia-check`); libblastrampoline's probes
  needed `isamax`, `zdotc`, `cdotc` and `sdot` too.

**2026-09-30.** The first day, in one burst.
- **Beginnings:** the checking kit and repro-scan, first as a folder of
  [crmvec](https://github.com/anun333/crmvec), then this repository, with
  their history.
- **lowp:** correctly rounded FP8 math (E4M3, E5M2), proven on every input
  and pair, and MX block functions, correct by construction. The rules
  were then checked against OCP's OFP8 1.0 and MX v1.0 specifications:
  E4M3's saturating infinity and the MX scale, saturation and ties are the
  specifications'; the all-zero block, the scale clamp and the whole-block
  NaN rule are choices they leave open. E5M2's saturating mode, which OFP8
  requires, was added. MXINT8 elements followed, and their first check
  found a scale bug that every earlier element type had hidden, in lowp
  and in the kit's reference alike (a nonzero result too small to
  represent taken for a zero): fixed in both.
- **ival:** 31 interval functions, then `hypot`, `atan2` (across the branch
  cut) and `pow` on boxes.
- **crsum:** correctly rounded sums and dot products, the same bits in any
  order; binned for speed (1.2 times a naive binary64 sum); matrix
  products; and `crgemm_oz`, exact matrix products through any binary64
  BLAS (the Ozaki scheme), checked through netlib's BLAS and through
  threaded OpenBLAS on x86-64 and arm64.
- **crnn** (the same evening): neural-network primitives with one answer.
  sigmoid, SiLU, GELU, softplus and rsqrt are correctly rounded in
  binary32, through a binary64 fast path and a table of the 531 inputs it
  can't decide. logsumexp, softmax, layernorm and rmsnorm are specified bit
  for bit on crsum. Its own checks corrected it three times before it was
  committed:
  - SiLU and GELU at tiny inputs sit exactly on a binary32 midpoint in
    binary64. The first table had 16.7 million entries each; breaking the
    tie analytically left 24 and 15.
  - logsumexp took log of the rounded sum, 9.5 million ulp off when the sum
    is 1 plus terms far below it. It now takes log1p of the exact sum less
    1.
  - layernorm subtracted a rounded mean, 30 ulp off for a large mean and a
    small spread. The mean is now carried in two parts.
- **Found along the way, for other projects:** crmvec's binary16 `cbrt`
  called the C library's `cbrtf` (fixed in crmvec); glibc's `libm` gives
  other bits on CPUs without FMA; Microsoft's MX reference implementation
  picks a scale one step too large just below a power of two.

Every check here has a control that must fail, and was trusted only after
deliberately planted bugs showed it could see them. The planted bugs that
passed are part of the record: some changed no result and proved nothing,
and some showed a gap in the test cases, which was then closed. README.md
has the details, library by library.
