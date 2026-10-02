/* bench-ozk m n k [m n k ...]: seconds per m x n x k product through crgemm_oz8 (int8 slices through the internal
   int8 GEMM), best of 3, with the result checked equal to crgemm's (exact per element). For the shapes bench-oz's
   cubes do not reach: long k, where a panel of Bt's rows no longer fits L1 (2026-10-02, ROADMAP item 1, "blocking
   along k"). Values as bench-oz's: one sign and magnitude 2^-6..1 for A, both signs for B.
     cc -O2 -ffp-contract=off -frounding-math -I crsum [-DOZ8_KC=4096] crsum/test/bench-ozk.c crsum/crsum.c -lm -ldl */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include "crsum.h"

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
int main(int argc, char **argv)
{
  printf("int8 kernel: %s\n%6s %6s %7s %12s  (seconds, best of 3)\n", crsum_i8_kernel(), "m", "n", "k", "crgemm_oz8");
  for (int a = 1; a + 2 < argc; a += 3) {
    size_t m = (size_t)atol(argv[a]), n = (size_t)atol(argv[a + 1]), k = (size_t)atol(argv[a + 2]);
    double *A = malloc(m * k * 8), *B = malloc(k * n * 8), *C0 = calloc(m * n, 8), *C1 = calloc(m * n, 8);
    uint64_t s = 7;
    for (size_t i = 0; i < m * k; i++) { s = s * 6364136223846793005ULL + 1442695040888963407ULL; A[i] = 0.015625 + (double)(s >> 11) * 0x1p-53; }
    for (size_t i = 0; i < k * n; i++) { s = s * 6364136223846793005ULL + 1442695040888963407ULL; B[i] = (double)(s >> 11) * 0x1p-53 - 0.5; }
    crgemm(m, n, k, A, k, B, n, 0, C0, n, CRSUM_NEAREST);
    double t = 1e9;
    for (int r = 0; r < 3; r++) {
      double t0 = now();
      crgemm_oz8(m, n, k, A, k, B, n, 0, C1, n, CRSUM_NEAREST, NULL, NULL);
      double dt = now() - t0; if (dt < t) t = dt;
    }
    printf("%6zu %6zu %7zu %12.4f  %s\n", m, n, k, t, memcmp(C0, C1, m * n * 8) ? "RESULTS DIFFER from crgemm" : "(= crgemm)");
    free(A); free(B); free(C0); free(C1);
  }
  return 0;
}
