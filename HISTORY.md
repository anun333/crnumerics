# History

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
- **Found along the way, for other projects:** crmvec's binary16 `cbrt`
  called the C library's `cbrtf` (fixed in crmvec); glibc's `libm` gives
  other bits on CPUs without FMA; Microsoft's MX reference implementation
  picks a scale one step too large just below a power of two.

Every check here has a control that must fail, and was trusted only after
deliberately planted bugs showed it could see them. The planted bugs that
passed are part of the record: some changed no result and proved nothing,
and some showed a gap in the test cases, which was then closed. README.md
has the details, library by library.
