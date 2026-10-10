# History

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
