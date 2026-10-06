/* crsum.h: correctly rounded sums and dot products, the same bits under any
   order, thread count or vector width (crsum/README.md).

   Every term is added exactly into a fixed-point accumulator wide enough
   for any sum of binary64 values or of their products, and the total is
   rounded once, correctly, in the mode asked for. There is one right
   answer, so the order of the terms, how they are split between threads,
   and the vector width can't change it: reproducible by construction.

   mode: CRSUM_NEAREST (ties to even), CRSUM_UP, CRSUM_DOWN, CRSUM_ZERO.
   Special values are as a sequence of exact additions would give them:
   a NaN, or +inf and -inf together, give NaN; an infinity of one sign
   gives it. An exact zero is +0, except -0 when every term is -0, and -0
   in CRSUM_DOWN when the terms cancel (MPFR's mpfr_sum and IEEE 754).
   Overflow gives an infinity or the largest finite value, per the mode.
   The sum of no terms is +0. Nothing depends on the C rounding mode or
   floating-point flags, which are left as they were. A mode it doesn't
   know gives NaN.

   The accumulator is 134 limbs of 32 bits in 64-bit integers (1,072
   bytes): exponents 2^-2176 to 2^2112, enough for any product of two
   binary64 values, carries deferred and settled every 2^29 terms. */
#ifndef CRSUM_H
#define CRSUM_H
#include <stddef.h>
#include <stdint.h>

enum { CRSUM_NEAREST = 0, CRSUM_UP = 1, CRSUM_DOWN = 2, CRSUM_ZERO = 3 };

/* sum of x[0 .. n-1], and dot product of x and y, correctly rounded */
double crsum(const double *x, size_t n, int mode);
double crdot(const double *x, const double *y, size_t n, int mode);
/* the same over binary32 values, rounded to binary32 */
float crsumf(const float *x, size_t n, int mode);
float crdotf(const float *x, const float *y, size_t n, int mode);

/* the accumulator, for sums in pieces: across calls, or across threads
   (one accumulator each, merged). Merging is exact, so the result is the
   same however the terms were split. */
#define CRSUM_LIMBS 134
typedef struct {
  int64_t limb[CRSUM_LIMBS];
  uint64_t since;          /* terms since the carries were last settled */
  int nan, pinf, ninf;     /* a NaN; +inf; -inf seen */
  int nonzero, poszero, negzero;   /* a nonzero term; a +0; a -0 (an exact zero's sign) */
  uint64_t terms;          /* how many terms */
} crsum_acc;

/* matrix products, every element correctly rounded (each is one exact dot
   product). Row-major: row i of a matrix M starts at M + i * ldm.
   crgemv: y = op(A) x + beta y, A with m rows and n columns, op(A) = A
   (trans 0: x has n elements, y m) or its transpose (trans 1: x has m, y
   n). crgemm: C = A B + beta C, A m by k, B k by n, C m by n. beta y is
   added exactly, as one more product; with beta = 0, y (or C) is not read,
   as in BLAS. They return 0, or -1 for a mode they don't take (nothing
   written). Each element's cost is a dot product's: for large matrices,
   the exact route is slow next to an optimized BLAS. */
int crgemv(int trans, size_t m, size_t n, const double *A, size_t lda, const double *x, double beta, double *y,
           int mode);
int crgemm(size_t m, size_t n, size_t k, const double *A, size_t lda, const double *B, size_t ldb, double beta,
           double *C, size_t ldc, int mode);
int crgemvf(int trans, size_t m, size_t n, const float *A, size_t lda, const float *x, float beta, float *y, int mode);
int crgemmf(size_t m, size_t n, size_t k, const float *A, size_t lda, const float *B, size_t ldb, float beta, float *C,
            size_t ldc, int mode);

/* crgemm through a binary64 GEMM (the Ozaki scheme): the same bits as
   crgemm. gemm(m, n, k, A, lda, B, ldb, C, ldc, ctx) must set C = A B,
   row-major, in binary64 arithmetic: in any order, with or without fused
   multiply-adds, threaded or not (a BLAS's dgemm will do, but not an
   emulated or lower-precision mode). NULL: an internal one. A's rows and
   B's columns are cut into integer slices small enough that everything the
   GEMM computes is an integer below 2^53, so exact however it computes;
   the slices' products are then added exactly and rounded once. NaN or
   infinities in A or B, or a range needing more than 64 slice products,
   fall back to crgemm. */
typedef void (*crsum_dgemm)(size_t m, size_t n, size_t k, const double *A, size_t lda, const double *B, size_t ldb,
                            double *C, size_t ldc, void *ctx);
int crgemm_oz(size_t m, size_t n, size_t k, const double *A, size_t lda, const double *B, size_t ldb, double beta,
              double *C, size_t ldc, int mode, crsum_dgemm gemm, void *ctx);

/* crgemm through an int8 GEMM (the Ozaki scheme on integer dot-product
   units): the same bits as crgemm. A's rows and B's columns are cut into
   signed 7-bit slices (int8 values in [-127, 127]); gemm(m, n, k, A, lda, Bt,
   ldbt, C, ldc, ctx) must set C = A Bt^T exactly in 32-bit integers, A row
   major (m x k) and Bt row major (n x k, so B's columns are rows: every
   output a dot product of two contiguous rows), as AVX512-VNNI's vpdpbusd,
   Arm's SDOT or a BLAS's s8s8s32 GEMM compute. k goes to the GEMM in chunks
   of at most 32768, so no int32 sum can overflow (32768 x 255 x 127 < 2^31).
   NULL: an internal one (AVX512-VNNI, AVX2, Arm I8MM or SDOT where the CPU
   has it, else plain C).
   NaN or infinities, or a range needing more than 400 slice products, fall
   back to crgemm. */
typedef void (*crsum_i8gemm)(size_t m, size_t n, size_t k, const int8_t *A, size_t lda, const int8_t *Bt, size_t ldbt,
                             int32_t *C, size_t ldc, void *ctx);
int crgemm_oz8(size_t m, size_t n, size_t k, const double *A, size_t lda, const double *B, size_t ldb, double beta,
               double *C, size_t ldc, int mode, crsum_i8gemm gemm, void *ctx);
/* which internal int8 kernel crgemm_oz8 uses here: "vnni", "avx2", "i8mm", "sdot" or "plain" */
const char *crsum_i8_kernel(void);

void crsum_init(crsum_acc *a);
void crsum_add(crsum_acc *a, const double *x, size_t n);
void crsum_add_dot(crsum_acc *a, const double *x, const double *y, size_t n);
void crsum_merge(crsum_acc *into, const crsum_acc *from);
double crsum_round(const crsum_acc *a, int mode);
float crsum_roundf(const crsum_acc *a, int mode);

#endif
