# History

**2026-10-01.** The Ozaki scheme on int8 units.
- **crsum:** `crgemm_oz8`, the same exact scheme as `crgemm_oz` with 7-bit
  slices on integer dot-product instructions instead of binary64 GEMMs:
  every slice product an exact int32 (k in chunks of 32,768), the sums
  added in int64, and `crgemm_oz`'s rounding step, now shared. Checked
  against `crgemm` bit for bit through the internal kernel (AVX512-VNNI on
  cfarm151, Arm SDOT on cfarm424, plain C on the laptop), plain C, a
  scrambled order and a counting GEMM; a GEMM dropping one term differs.
  With VNNI it takes half `crgemm_oz`'s time at n = 256.

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
