/* cublas.c: crlin with an NVIDIA GPU doing the heavy work (2026-10-13). cuBLAS's DGEMM as crlin's GEMM, cuSOLVER's
   LU for the approximate inverse; crlin's exact O(n^2) work and the bounds stay on the CPU. The proof asks of the
   GEMM what crlin.h says: in CRLIN_CLASSICAL each element a sum of products rounded to nearest (any order, fused or
   not), which native DGEMM is; in CRLIN_EXACT only binary64 arithmetic. So the handle is set to
   CUBLAS_PEDANTIC_MATH: cuBLAS 13's fixed-point emulation of FP64 (CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH) is opt-in,
   and is not that.

     cublas N [classical|exact|extended] [DUMP]

   A: random entries in [-1, 1) plus 0.1 N on the diagonal (as openpocl's crlin benches), b random, from a fixed
   seed. Prints the time of cuSOLVER's unverified solve, then of the inverse, of crlin_solve and of its GEMM calls,
   and the enclosure's largest relative width. DUMP: a file of N, A, b, lo, hi (binary64, little-endian) for
   check-dump.py, which checks the enclosure against the exact solution in fractions (small N).

   Build: make crlin-cublas (CUDA_HOME, default /usr/local/cuda). */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include "crlin.h"

#define CK(x) do { int e_ = (int)(x); if (e_) { fprintf(stderr, "%s:%d: %s failed (%d)\n", __FILE__, __LINE__, #x, e_); exit(2); } } while (0)

typedef struct { cublasHandle_t h; double *dA, *dB, *dC; size_t ca, cb, cc; double secs; long calls; } gpu;

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }

static void grow(double **p, size_t *cap, size_t need)
{
  if (need <= *cap) return;
  if (*p) CK(cudaFree(*p));
  CK(cudaMalloc((void **)p, need * sizeof **p));
  *cap = need;
}

/* row-major C (m by n) = A (m by k) B (k by n): in cuBLAS's column-major terms C^T = B^T A^T, the same memory */
static void cu_gemm(size_t m, size_t n, size_t k, const double *A, size_t lda, const double *B, size_t ldb, double *C,
                    size_t ldc, void *ctx)
{
  gpu *g = ctx;
  double t0 = now(), one = 1, zero = 0;
  grow(&g->dA, &g->ca, m * k); grow(&g->dB, &g->cb, k * n); grow(&g->dC, &g->cc, m * n);
  CK(cudaMemcpy2D(g->dA, k * 8, A, lda * 8, k * 8, m, cudaMemcpyHostToDevice));
  CK(cudaMemcpy2D(g->dB, n * 8, B, ldb * 8, n * 8, k, cudaMemcpyHostToDevice));
  CK(cublasDgemm(g->h, CUBLAS_OP_N, CUBLAS_OP_N, (int)n, (int)m, (int)k, &one, g->dB, (int)n, g->dA, (int)k, &zero, g->dC, (int)n));
  CK(cudaMemcpy2D(C, ldc * 8, g->dC, n * 8, n * 8, m, cudaMemcpyDeviceToHost));
  g->secs += now() - t0;
  g->calls++;
}

/* with A's row-major memory read as column-major (A^T), getrf then getrs: trans 0 solves A^T X = B, which for B the
   identity is (A^T)^-1, whose column-major memory is A^-1 row-major; trans 1 solves A x = b. X overwrites B. */
static void cu_lu_solve(cusolverDnHandle_t s, size_t n, size_t nrhs, const double *A, double *B, int trans)
{
  double *dA, *dB, *work;
  int *piv, *info, lwork;
  CK(cudaMalloc((void **)&dA, n * n * 8)); CK(cudaMalloc((void **)&dB, n * nrhs * 8));
  CK(cudaMalloc((void **)&piv, n * sizeof *piv)); CK(cudaMalloc((void **)&info, sizeof *info));
  CK(cudaMemcpy(dA, A, n * n * 8, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dB, B, n * nrhs * 8, cudaMemcpyHostToDevice));
  CK(cusolverDnDgetrf_bufferSize(s, (int)n, (int)n, dA, (int)n, &lwork));
  CK(cudaMalloc((void **)&work, (size_t)lwork * 8));
  CK(cusolverDnDgetrf(s, (int)n, (int)n, dA, (int)n, work, piv, info));
  CK(cusolverDnDgetrs(s, trans ? CUBLAS_OP_T : CUBLAS_OP_N, (int)n, (int)nrhs, dA, (int)n, piv, dB, (int)n, info));
  CK(cudaMemcpy(B, dB, n * nrhs * 8, cudaMemcpyDeviceToHost));
  CK(cudaFree(dA)); CK(cudaFree(dB)); CK(cudaFree(piv)); CK(cudaFree(info)); CK(cudaFree(work));
}

int main(int argc, char **argv)
{
  size_t n = argc > 1 ? strtoul(argv[1], 0, 10) : 1000;
  const char *ms = argc > 2 ? argv[2] : "classical";
  int mode = !strcmp(ms, "exact") ? CRLIN_EXACT : !strcmp(ms, "extended") ? CRLIN_EXTENDED : CRLIN_CLASSICAL;
  double *A = malloc(n * n * 8), *b = malloc(n * 8), *R = malloc(n * n * 8), *x = malloc(n * 8), *lo = malloc(n * 8),
         *hi = malloc(n * 8);
  uint64_t q = 0x2545f4914f6cdd1dULL;
  for (size_t i = 0; i < n * n; i++) { q = q * 6364136223846793005ULL + 1442695040888963407ULL; A[i] = (double)(q >> 11) * 0x1p-52 - 1; }
  for (size_t i = 0; i < n; i++) { A[i * n + i] += 0.1 * (double)n; q = q * 6364136223846793005ULL + 1442695040888963407ULL; b[i] = (double)(q >> 11) * 0x1p-52 - 1; }

  gpu g = {0};
  cusolverDnHandle_t s;
  cublasMath_t mm;
  CK(cublasCreate(&g.h));
  CK(cublasSetMathMode(g.h, CUBLAS_PEDANTIC_MATH));
  CK(cublasGetMathMode(g.h, &mm));
  CK(cusolverDnCreate(&s));
  struct cudaDeviceProp pr;
  CK(cudaGetDeviceProperties(&pr, 0));
  printf("%s, cuBLAS math mode %d (pedantic: 2), CUBLAS_EMULATION_STRATEGY=%s; n = %zu, %s\n", pr.name, (int)mm,
         getenv("CUBLAS_EMULATION_STRATEGY") ? getenv("CUBLAS_EMULATION_STRATEGY") : "(unset)", n, ms);

  /* warm up the device, the handles and the libraries' kernels on a small problem first */
  { double w[16] = {2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2}, c[16]; cu_gemm(4, 4, 4, w, 4, w, 4, c, 4, &g);
    double e[4] = {1, 1, 1, 1}; cu_lu_solve(s, 4, 1, w, e, 1); g.secs = 0; g.calls = 0; }

  double best_lu = INFINITY, best_inv = INFINITY, best_crlin = INFINITY, gemm_secs = 0;
  long gemm_calls = 0;
  int st = -1;
  for (int pass = 0; pass < 3; pass++) {
    double t0 = now();
    memcpy(x, b, n * 8);
    cu_lu_solve(s, n, 1, A, x, 1);                         /* unverified: A x = b */
    best_lu = fmin(best_lu, now() - t0);
    t0 = now();
    for (size_t i = 0; i < n * n; i++) R[i] = 0;
    for (size_t i = 0; i < n; i++) R[i * n + i] = 1;
    cu_lu_solve(s, n, n, A, R, 0);                         /* R = A^-1, row-major */
    double t1 = now();
    g.secs = 0; g.calls = 0;
    st = crlin_solve(n, A, n, b, R, n, lo, hi, mode, cu_gemm, &g);
    double t2 = now();
    best_inv = fmin(best_inv, t1 - t0);
    if (t2 - t1 < best_crlin) { best_crlin = t2 - t1; gemm_secs = g.secs; gemm_calls = g.calls; }
  }
  double w = 0;
  for (size_t i = 0; i < n; i++) w = fmax(w, (hi[i] - lo[i]) / fmax(fabs(lo[i]), fabs(hi[i])));
  printf("cuSOLVER solve (unverified) %.2f ms | inverse %.2f ms | crlin_solve %.2f ms (%ld GEMM calls, %.2f ms in them) "
         "| total %.1fx the unverified solve | %s, widest %.2g relative\n", best_lu * 1e3, best_inv * 1e3,
         best_crlin * 1e3, gemm_calls, gemm_secs * 1e3, (best_inv + best_crlin) / best_lu,
         st == CRLIN_VERIFIED ? "verified" : "FAILED", w);
  if (best_crlin < 1e-6) printf("VOID: crlin_solve took no time\n");
  if (argc > 3) {
    FILE *f = fopen(argv[3], "wb");
    double nn = (double)n;
    fwrite(&nn, 8, 1, f); fwrite(A, 8, n * n, f); fwrite(b, 8, n, f); fwrite(lo, 8, n, f); fwrite(hi, 8, n, f);
    fclose(f);
  }
  cublasDestroy(g.h); cusolverDnDestroy(s);
  return st == CRLIN_VERIFIED ? 0 : 1;
}
