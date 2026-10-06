# crsum: sums and dot products, the same bits everywhere

`crsum/crsum.h` (2026-09-30): the correctly rounded sum of n binary64 or
binary32 values, and their correctly rounded dot product, in any of the
four rounding modes:

```c
double crsum(const double *x, size_t n, int mode);                 /* CRSUM_NEAREST ... CRSUM_ZERO */
double crdot(const double *x, const double *y, size_t n, int mode);
float crsumf(const float *x, size_t n, int mode);
float crdotf(const float *x, const float *y, size_t n, int mode);
```

And matrix products, every element correctly rounded (each is one exact
dot product), row-major with leading dimensions:

```c
int crgemv(int trans, size_t m, size_t n, const double *A, size_t lda, const double *x, double beta, double *y, int mode);
int crgemm(size_t m, size_t n, size_t k, const double *A, size_t lda, const double *B, size_t ldb, double beta,
           double *C, size_t ldc, int mode);   /* and crgemvf, crgemmf */
```

`crgemv` gives y = A x + βy, or Aᵀx + βy with `trans`; `crgemm` gives C = AB
+ βC. βy is added exactly, as one more product, and with β = 0, y is not
read, as in BLAS. They return 0, or −1 for a mode they don't take.

**Exact matrix products through any BLAS: `crgemm_oz`** (the Ozaki
scheme). The same bits as `crgemm`, with the multiplying done by a binary64
GEMM you pass in (or an internal one):

```c
int crgemm_oz(size_t m, size_t n, size_t k, const double *A, size_t lda, const double *B, size_t ldb, double beta,
              double *C, size_t ldc, int mode, crsum_dgemm gemm, void *ctx);
```

A's rows and B's columns are scaled to integers and cut into slices of w =
⌊(53 − ⌈log₂ k⌉)/2⌋ bits. Then every value the GEMM computes, and every
partial sum, is an integer below 2^53, so the GEMM is exact however it
works: any summation order, fused multiply-adds or not, threads, blocking.
Any BLAS's `dgemm` in binary64 gives the same products (not an emulated or
lower-precision mode). The slice products are combined exactly and each
element rounded once. NaN or infinities, or data spread too widely (more
than 64 slice products), fall back to `crgemm`.

**How.** Every term, or every product of two terms, is added exactly into
a fixed-point accumulator that covers 2^−2176 to 2^2112 (134 limbs of 32
bits held in 64-bit integers, so that carries can wait), and the total is
rounded once. With one right answer, the order of the terms, how they are
split between threads and the vector width cannot change it: reproducible
by construction, not by fixing an order. For threads there is an
accumulator API (`crsum_init`, `crsum_add`, `crsum_add_dot`,
`crsum_merge`, `crsum_round`); merging is exact.

Special values follow a sequence of exact additions, as MPFR's `mpfr_sum`
does: NaN, or both infinities, give NaN; an exact zero is +0, −0 when every
term is −0, and −0 when rounding down after cancellation. The C rounding
mode plays no part.

**How it is checked** (`crsum/test/check.c`, 2 s on two cores), sums and
dot products in binary64 and binary32, in all four modes, against the
kit's `mpfr_sum` reference (`kit_sum_ref`):
- **the cases:** random terms at every exponent range; cancellation;
  sums exactly halfway between two results, and just above or below by a
  term far down (for dot products, a product below the format's range,
  2^−1200 or 2^−200, which only an exact method sees); subnormal results;
  overflow and cancelling back; NaN, infinities, signed zeros, no terms;
  2^18 terms. 4,676 results per kind, 0 differ;
- **the same bits in any order:** each case shuffled, and split at random
  over 1, 2, 3 and 7 accumulators merged in a random order: 74,816 results,
  none different; and 2^20 terms on 1, 2, 3, 4 and 8 OpenMP threads;
- **controls:** each run's control, and a negative one (naive left-to-right
  summation differs on 512 of the 600 cancellation and midpoint cases);
- **matrix products:** 65,864 elements of `crgemv` (both orientations) and
  `crgemm`, binary64 and binary32, four modes, each against the reference:
  padded leading dimensions (the padding NaN, so reading it shows), β of
  0, ±1 and random (0 with NaN in y, which must not be read), cancelling
  rows, NaN and infinities, sizes on both sides of the binned path's
  threshold. 0 differ; unknown modes refused with nothing written. Six
  planted bugs (the leading dimension ignored, a column read with stride 1,
  β ignored, y read at β = 0, the product bins' high parts not emptied
  between rows, B's column transposed) were each caught;
- **`crgemm_oz`:** 510,400 elements against `crgemm`, bit for bit, in four
  modes, through four GEMMs: the internal one; one that sums each
  element's products in a random order, alternating fused and separate
  multiply-adds; the system's BLAS (`dgemm_` from `libblas.so.3`, which the
  check names in its output); and a counting copy that shows the Ozaki
  path ran (240 of 300 cases; the rest fall back). The system BLAS was
  netlib's reference BLAS on the development laptop, and multithreaded
  OpenBLAS (`openblas-pthread`) on both CI runners, x86-64 and arm64 (run
  36754104164, 2026-09-30): 0 differ through each. Ranges from
  one binade to 150, zero rows, subnormals, k up to 3,000; exact zeros of
  one sign; exact midpoints and just either side, at the subnormal quantum
  too. 0 differ. A GEMM computing in binary32 must differ, and does.
  Planted, each caught: slices 2 bits too wide, a slice digit off by one
  bit, a column scaled from its top bit, a slice pair skipped, the
  exponent off by one, the exact-zero path removed, and in the direct
  rounding: ties away, the sticky bit, the subnormal quantum, the sign.
  Three first passed and showed gaps in the cases (fixed): β only ever 0
  where no zero row was; rows and columns of zeros alone fall back to
  `crgemm` whole, so the exact-zero path never ran; and the subnormal
  midpoints spread so wide that they fell back too;
- **the carries:** the check also runs on a build that settles them every
  3 terms (the default, 2^29, no test reaches);
- **planted bugs, each caught:** ties away from zero, the sticky bit
  ignored, no subnormal quantum, overflow always infinite, a cancelled zero
  always +0, a product's top piece lost, a merge dropping NaN, a negative
  term's high half added; and in the bins, a full bin never emptied (sums
  and dot products), the sign lost, the subnormal exponent off by one, a
  product's high part a binade low. Dropping the carry into the next
  binade after rounding changes no result (M 2^q is the same value either
  way), so it proves nothing about the check: a plant counts only once it
  changes an output.
  The dot products' "never emptied" plant first passed, because every long
  dot case had random signs and the two signs' bins wrapped past 2^64 alike,
  cancelling: an all-positive long case was added, and then it failed.

**Speed.** For 64 terms or more, each term goes first into a bin for its
sign and exponent (a uint64 sum of significands, Neal's "large
superaccumulator"), and a bin moves into the wide accumulator only when it
nears 2^62, and at the end. For a dot product, the exact 106-bit product
goes in as two 53-bit parts. On this Zen 3 laptop, one thread, 2^22 terms:

| | crsum | a naive loop | |
|---|---|---|---|
| binary64 sum | 0.88 ns a term | 0.75 ns | 1.2 times |
| binary64 dot product | 3.13 ns | 0.92 ns | 3.4 times |
| binary32 sum | 1.09 ns | 0.74 ns | 1.5 times |
| binary32 dot product | 3.47 ns | 0.73 ns | 4.7 times |

(The first, scalar version took 7.8 and 11.5 ns in binary64: 10.5 and
12.7 times.)

Matrix products reuse one set of bins, emptying only the exponents a row
touched: `crgemv` on 1024 × 1024 takes 4.35 ns a multiply-add (5.9 times a
naive loop), `crgemm` on 256³ 6.97 ns (8 times a naive triple loop, which
is itself far from an optimized BLAS).

`crgemm_oz` through an optimized BLAS is the fast route. One thread,
entries over 2^20 of range, OpenBLAS (scipy's bundled copy, loaded for the
test, since installing it would change this machine's system BLAS):

| | 256³ | 512³ |
|---|---|---|
| `crgemm_oz` through OpenBLAS | 18.9 ms | 117 ms |
| `crgemm_oz`, internal GEMM | 73.5 ms | 568 ms |
| `crgemm` | 130 ms | 766 ms |
| a naive triple loop (not exact) | 34.4 ms | 354 ms |
| OpenBLAS's own `dgemm` (not exact) | 0.6 ms | 5.2 ms |

About 16 slice products per call: exactness costs roughly 20 to 30 times
an optimized `dgemm` here, and much less on hardware whose matrix units
are faster at low precision (int8 slices: ROADMAP.md, item 10).

**On int8 units: `crgemm_oz8`** (same arguments, an int8 GEMM in place of
the binary64 one). Slices of 7 bits with their sign, every slice product an
exact int32, the same bits as `crgemm`. Its internal kernels, picked at run
time (`crsum_i8_kernel()` names the one in use; `CRSUM_I8_KERNEL=plain`,
`avx2`, `vnni` or `sdot` picks one the CPU has, for checks and timing):
AVX512-VNNI (`vpdpbusd`, A's bytes biased by 128), AVX2 (bytes widened to
16 bits and multiplied in pairs with `vpmaddwd`, which cannot saturate),
Arm I8MM (`smmla`, a 2 x 8 by 8 x 2 block per instruction), Arm SDOT,
plain C; all but plain blocked two rows by four columns, and over panels
of B's rows of about 16 KB, so that a panel stays in L1 while every pair
of A's rows passes over it. On the
GB10's Arm cores (Cortex-X925/A725) I8MM makes `crgemm_oz8` 1.4 to 1.8
times faster than SDOT (512³: 0.230 against 0.407 s, one core). `n`³
products, entries in [2^-6, 1], one thread, seconds:

| | 256³ | 512³ |
|---|---|---|
| Ryzen 5 PRO 5650U (AVX2, laptop, quiet), `crgemm_oz8` | 0.031 | 0.190 |
| the same, plain C kernel | 0.596 | (7.44 at load ~9) |
| the same, `crgemm_oz` / `crgemm` | 0.057 / 0.105 | 0.441 / 0.683 |
| Xeon, Cascade Lake (cfarm151, idle), `crgemm_oz8`, VNNI | 0.063 | 0.350 |
| the same, AVX2 kernel | 0.065 | 0.433 |
| the same, `crgemm_oz` / `crgemm` | 0.137 / 0.187 | 1.17 / 1.24 |

The panels (2026-10-01) gain where B outgrows L2: at 1024³, VNNI 2.13 to
1.54 s on cfarm151 (512³: 0.315 to 0.254; both at load about 1, so not the
idle figures above), AVX2 1.41 to 1.26 s on the laptop; nothing at 256³.
SDOT gained most, being unblocked before: on a Neoverse N1 (cfarm424)
`crgemm_oz8` went from 0.070 to 0.045 s at 256³, 0.472 to 0.246 at 512³
and 6.13 to 4.50 at 1024³. I8MM, reworked as well (16-byte loads paired
by `zip`, four accumulators, the panels), on the GB10's fastest core: 0.0170
to 0.0147 s at 256³, 0.106 to 0.079 at 512³, 0.81 to 0.56 at 1024³.
The int8 GEMM is still most of the time (the data above needs about 81
slice products); blocking along k, for k beyond 4096, is next.

**In a program that already calls a BLAS: `libcrblas.so`** (`crsum/crblas.c`,
`make build/libcrblas.so`). It exports `dgemm_` and `dgemm_64_` (32- and
64-bit integers) doing `crgemm_oz`, column-major, with BLAS's arguments:
each element is the correctly rounded value of (αA)B + βC, βC added
exactly, so with α = 1 the exact product rounded once (another α scales A
first, one rounding per element). The multiplying inside goes to the BLAS
that `crblas_set_inner(path)` names, or to crsum's internal GEMM. Julia,
for example, through its BLAS switchboard (libblastrampoline):

```julia
using LinearAlgebra
openblas = BLAS.get_config().loaded_libs[1].libname          # Julia's own OpenBLAS
BLAS.lbt_forward("/path/to/libcrblas.so"; clear = false, suffix_hint = "64_")
ccall((:crblas_set_inner, "/path/to/libcrblas.so"), Cint, (Cstring,), openblas)
A * B          # every element now the exact product, rounded once
```

`make julia-check` (`crsum/julia/crblas.jl`, Julia 1.13.1) checks it on
150 × 96 × 110 matrices with entries over 2^40 of range and cancelling
columns: Julia's `A * B`, `A' * B'`, the mixed transposes and `mul!` with
α and β give **the exact BigFloat product rounded once, in every element**,
with OpenBLAS inside on 1 or 8 threads and with the internal GEMM, the same
bits. Controls: OpenBLAS's own `A * B` differs from the exact product on
13,441 of the 16,500 elements, and crblas's call count shows the product
went through it. libblastrampoline probes `isamax`, `zdotc`, `cdotc` and
`sdot` before it forwards anything, so crblas exports those too, the dot
products correctly rounded (checked there, called directly); Julia's own
`dot` goes through CBLAS and stays OpenBLAS's. The cost on a 1000³ product,
one thread, a busy laptop: 1.05 s against OpenBLAS's 54 ms; OpenBLAS's
threads help the slice products only (0.76 s on 8).
