/* crblas.c: dgemm with every element correctly rounded, under the Fortran
   BLAS names, for a program that already calls a BLAS. Through a BLAS
   switchboard (Julia's libblastrampoline, FlexiBLAS) or LD_PRELOAD, its
   matrix products become crgemm_oz's: the same bits on any CPU, thread
   count or BLAS.

   Exports, nothing else:
   - dgemm_ (32-bit integers) and dgemm_64_ (64-bit, as Julia's ILP64 BLAS
     calls it): C = alpha op(A) op(B) + beta C, column-major, every element
     the correctly rounded (to nearest) value of (alpha A) B + beta C, beta C
     added exactly. With alpha = 1 (Julia's A * B) that is the exact product
     rounded once; another alpha scales A first, one rounding per element,
     which is still the same bits everywhere. As in the reference BLAS,
     beta = 0 means C is not read, and alpha = 0 or k = 0 gives beta C.
   - isamax_ and isamax_64_: libblastrampoline calls isamax to tell 32-bit
     from 64-bit integers when it loads a library;
   - zdotc_, cdotc_ and their _64_ forms (the complex dot products,
     conjugating x), each part correctly rounded: libblastrampoline calls
     zdotc and cdotc to see how a complex result is returned (here as C's
     complex types are), and then sends the program's zdotc and cdotc calls
     here too;
   - sdot_ and sdot_64_, correctly rounded, returning a float:
     libblastrampoline calls sdot to tell gfortran's convention from f2c's
     (which returns a double), and then sends sdot calls here.
   - crblas_gemm_calls(): how many dgemm calls came here (to show that a
     program's products took this path);
   - crblas_set_inner(path): the BLAS library whose dgemm does the
     multiplying inside crgemm_oz (its dgemm_64_, else its dgemm_), such as
     Julia's own OpenBLAS; NULL or unset: crsum's internal GEMM. Exact
     either way: crgemm_oz cuts the operands into integer slices whose
     products and sums every binary64 GEMM computes without error.
   crgemm_oz is row-major; a column-major C is a row-major C^T, so
   C^T = op(B)^T op(A)^T + beta C^T is passed, transposed copies made only
   for transposed operands, and A copied only for alpha != 1. */
#define _GNU_SOURCE
#include <complex.h>
#include <dlfcn.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "crsum.h"

#define EXPORT __attribute__((visibility("default")))
typedef void (*fdgemm64)(const char *, const char *, const int64_t *, const int64_t *, const int64_t *, const double *,
                         const double *, const int64_t *, const double *, const int64_t *, const double *, double *,
                         const int64_t *, size_t, size_t);
typedef void (*fdgemm32)(const char *, const char *, const int32_t *, const int32_t *, const int32_t *, const double *,
                         const double *, const int32_t *, const double *, const int32_t *, const double *, double *,
                         const int32_t *, size_t, size_t);
static fdgemm64 inner64;
static fdgemm32 inner32;
static long ncalls;
EXPORT long crblas_gemm_calls(void) { return __atomic_load_n(&ncalls, __ATOMIC_RELAXED); }

EXPORT int crblas_set_inner(const char *path)
{
  inner64 = NULL; inner32 = NULL;
  if (!path || !*path) return 0;
  void *h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
  if (!h) return -1;
  inner64 = (fdgemm64)dlsym(h, "dgemm_64_");
  if (!inner64) inner32 = (fdgemm32)dlsym(h, "dgemm_");
  return inner64 || inner32 ? 0 : -1;
}

/* crsum_dgemm (row-major C = A B, m by n, depth k) through a column-major
   dgemm: C^T = B^T A^T, with no copies */
static void via_inner(size_t m, size_t n, size_t k, const double *A, size_t lda, const double *B, size_t ldb, double *C,
                      size_t ldc, void *ctx)
{
  (void)ctx; const double one = 1, zero = 0;
  if (inner64) {
    int64_t M = (int64_t)n, N = (int64_t)m, K = (int64_t)k, LA = (int64_t)ldb, LB = (int64_t)lda, LC = (int64_t)ldc;
    inner64("N", "N", &M, &N, &K, &one, B, &LA, A, &LB, &zero, C, &LC, 1, 1);
  } else {
    int32_t M = (int32_t)n, N = (int32_t)m, K = (int32_t)k, LA = (int32_t)ldb, LB = (int32_t)lda, LC = (int32_t)ldc;
    inner32("N", "N", &M, &N, &K, &one, B, &LA, A, &LB, &zero, C, &LC, 1, 1);
  }
}

static int istrans(char t) { return t == 'T' || t == 't' || t == 'C' || t == 'c'; }

static void gemm(char ta, char tb, int64_t m, int64_t n, int64_t k, double alpha, const double *A, int64_t lda,
                 const double *B, int64_t ldb, double beta, double *C, int64_t ldc)
{
  __atomic_add_fetch(&ncalls, 1, __ATOMIC_RELAXED);
  if (m <= 0 || n <= 0) return;
  if (alpha == 0 || k <= 0) {   /* C = beta C, C not read when beta = 0 */
    if (beta == 1) return;
    for (int64_t j = 0; j < n; j++)
      for (int64_t i = 0; i < m; i++) C[i + j * ldc] = beta == 0 ? 0 : beta * C[i + j * ldc];
    return;
  }
  /* X = op(B)^T, row-major n by k; Y = op(A)^T, row-major k by m */
  const double *X = B, *Y = A; size_t ldx = (size_t)ldb, ldy = (size_t)lda;
  double *xc = NULL, *yc = NULL;
  if (istrans(tb)) {             /* B is n by k column-major: copy it row-major */
    xc = malloc(sizeof(double) * (size_t)n * (size_t)k);
    for (int64_t i = 0; i < n; i++) for (int64_t p = 0; p < k; p++) xc[i * k + p] = B[i + p * ldb];
    X = xc; ldx = (size_t)k;
  }
  if (istrans(ta) || alpha != 1) {   /* op(A)^T row-major k by m, scaled by alpha */
    yc = malloc(sizeof(double) * (size_t)k * (size_t)m);
    for (int64_t p = 0; p < k; p++)
      for (int64_t j = 0; j < m; j++) {
        double a = istrans(ta) ? A[p + j * lda] : A[j + p * lda];
        yc[p * m + j] = alpha == 1 ? a : alpha * a;
      }
    Y = yc; ldy = (size_t)m;
  }
  crgemm_oz((size_t)n, (size_t)m, (size_t)k, X, ldx, Y, ldy, beta, C, (size_t)ldc, CRSUM_NEAREST,
            inner64 || inner32 ? via_inner : NULL, NULL);
  free(xc); free(yc);
}

EXPORT void dgemm_64_(const char *ta, const char *tb, const int64_t *m, const int64_t *n, const int64_t *k,
                      const double *alpha, const double *A, const int64_t *lda, const double *B, const int64_t *ldb,
                      const double *beta, double *C, const int64_t *ldc)
{ gemm(*ta, *tb, *m, *n, *k, *alpha, A, *lda, B, *ldb, *beta, C, *ldc); }

EXPORT void dgemm_(const char *ta, const char *tb, const int32_t *m, const int32_t *n, const int32_t *k,
                   const double *alpha, const double *A, const int32_t *lda, const double *B, const int32_t *ldb,
                   const double *beta, double *C, const int32_t *ldc)
{ gemm(*ta, *tb, *m, *n, *k, *alpha, A, *lda, B, *ldb, *beta, C, *ldc); }

/* the index (from 1) of the first element of largest magnitude */
static int64_t isamax(int64_t n, const float *x, int64_t inc)
{
  if (n < 1 || inc <= 0) return 0;
  int64_t best = 1; float bv = x[0] < 0 ? -x[0] : x[0];
  for (int64_t i = 1; i < n; i++) { float v = x[i * inc] < 0 ? -x[i * inc] : x[i * inc]; if (v > bv) { bv = v; best = i + 1; } }
  return best;
}
EXPORT int64_t isamax_64_(const int64_t *n, const float *x, const int64_t *inc) { return isamax(*n, x, *inc); }
EXPORT int32_t isamax_(const int32_t *n, const float *x, const int32_t *inc) { return (int32_t)isamax(*n, x, *inc); }

/* sum of conj(x_i) y_i: the real part sum(xr yr + xi yi) and the imaginary
   part sum(xr yi - xi yr), each one exact dot product of 2n terms, rounded
   once */
static double _Complex zdotc(int64_t n, const double *x, int64_t incx, const double *y, int64_t incy)
{
  if (n < 1) return 0;
  double *a = malloc(sizeof(double) * 4 * (size_t)n), *b = a + 2 * n;
  int64_t ix = incx < 0 ? (1 - n) * incx : 0, iy = incy < 0 ? (1 - n) * incy : 0;
  for (int64_t i = 0; i < n; i++, ix += incx, iy += incy) {
    a[2 * i] = x[2 * ix]; a[2 * i + 1] = x[2 * ix + 1]; b[2 * i] = y[2 * iy]; b[2 * i + 1] = y[2 * iy + 1];
  }
  double re = crdot(a, b, 2 * (size_t)n, CRSUM_NEAREST);
  for (int64_t i = 0; i < n; i++) { double yr = b[2 * i]; b[2 * i] = b[2 * i + 1]; b[2 * i + 1] = yr; a[2 * i + 1] = -a[2 * i + 1]; }
  double im = crdot(a, b, 2 * (size_t)n, CRSUM_NEAREST);
  free(a);
  return CMPLX(re, im);
}
EXPORT double _Complex zdotc_64_(const int64_t *n, const double *x, const int64_t *incx, const double *y, const int64_t *incy)
{ return zdotc(*n, x, *incx, y, *incy); }
EXPORT double _Complex zdotc_(const int32_t *n, const double *x, const int32_t *incx, const double *y, const int32_t *incy)
{ return zdotc(*n, x, *incx, y, *incy); }

static float _Complex cdotc(int64_t n, const float *x, int64_t incx, const float *y, int64_t incy)
{
  if (n < 1) return 0;
  float *a = malloc(sizeof(float) * 4 * (size_t)n), *b = a + 2 * n;
  int64_t ix = incx < 0 ? (1 - n) * incx : 0, iy = incy < 0 ? (1 - n) * incy : 0;
  for (int64_t i = 0; i < n; i++, ix += incx, iy += incy) {
    a[2 * i] = x[2 * ix]; a[2 * i + 1] = x[2 * ix + 1]; b[2 * i] = y[2 * iy]; b[2 * i + 1] = y[2 * iy + 1];
  }
  float re = crdotf(a, b, 2 * (size_t)n, CRSUM_NEAREST);
  for (int64_t i = 0; i < n; i++) { float yr = b[2 * i]; b[2 * i] = b[2 * i + 1]; b[2 * i + 1] = yr; a[2 * i + 1] = -a[2 * i + 1]; }
  float im = crdotf(a, b, 2 * (size_t)n, CRSUM_NEAREST);
  free(a);
  return CMPLXF(re, im);
}
EXPORT float _Complex cdotc_64_(const int64_t *n, const float *x, const int64_t *incx, const float *y, const int64_t *incy)
{ return cdotc(*n, x, *incx, y, *incy); }
EXPORT float _Complex cdotc_(const int32_t *n, const float *x, const int32_t *incx, const float *y, const int32_t *incy)
{ return cdotc(*n, x, *incx, y, *incy); }

static float sdot(int64_t n, const float *x, int64_t incx, const float *y, int64_t incy)
{
  if (n < 1) return 0;
  float *a = malloc(sizeof(float) * 2 * (size_t)n), *b = a + n;
  int64_t ix = incx < 0 ? (1 - n) * incx : 0, iy = incy < 0 ? (1 - n) * incy : 0;
  for (int64_t i = 0; i < n; i++, ix += incx, iy += incy) { a[i] = x[ix]; b[i] = y[iy]; }
  float r = crdotf(a, b, (size_t)n, CRSUM_NEAREST);
  free(a);
  return r;
}
EXPORT float sdot_64_(const int64_t *n, const float *x, const int64_t *incx, const float *y, const int64_t *incy)
{ return sdot(*n, x, *incx, y, *incy); }
EXPORT float sdot_(const int32_t *n, const float *x, const int32_t *incx, const float *y, const int32_t *incy)
{ return sdot(*n, x, *incx, y, *incy); }
