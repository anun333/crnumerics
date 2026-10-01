/* bench-oz [n...]: seconds per n x n x n product, values of one sign and
   magnitude range 2^-6..1 (a typical, narrow range: the Ozaki paths run),
   for crgemm (exact per element), crgemm_oz (binary64 slices through the
   internal GEMM) and crgemm_oz8 (int8 slices through the internal int8
   GEMM: VNNI, SDOT or plain); best of 3; the three results checked equal. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "../crsum.h"
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
int main(int argc, char **argv)
{
  printf("int8 kernel: %s\n%6s %12s %12s %12s  (seconds)\n", crsum_i8_kernel(), "n", "crgemm", "crgemm_oz", "crgemm_oz8");
  for (int a = 1; a < argc; a++) {
    size_t n = (size_t)atol(argv[a]);
    double *A = malloc(n * n * 8), *B = malloc(n * n * 8), *C[3];
    uint64_t s = 7;
    for (size_t i = 0; i < n * n; i++) {
      s = s * 6364136223846793005ULL + 1442695040888963407ULL; A[i] = 0.015625 + (double)(s >> 11) * 0x1p-53;
      s = s * 6364136223846793005ULL + 1442695040888963407ULL; B[i] = (double)(s >> 11) * 0x1p-53 - 0.5;
    }
    double t[3];
    for (int f = 0; f < 3; f++) {
      C[f] = calloc(n * n, 8); t[f] = 1e9;
      for (int r = 0; r < 3; r++) {
        double t0 = now();
        if (f == 0) crgemm(n, n, n, A, n, B, n, 0, C[f], n, CRSUM_NEAREST);
        else if (f == 1) crgemm_oz(n, n, n, A, n, B, n, 0, C[f], n, CRSUM_NEAREST, NULL, NULL);
        else crgemm_oz8(n, n, n, A, n, B, n, 0, C[f], n, CRSUM_NEAREST, NULL, NULL);
        double dt = now() - t0; if (dt < t[f]) t[f] = dt;
      }
    }
    int same = !memcmp(C[0], C[1], n * n * 8) && !memcmp(C[0], C[2], n * n * 8);
    printf("%6zu %12.4f %12.4f %12.4f  %s\n", n, t[0], t[1], t[2], same ? "(results equal)" : "RESULTS DIFFER");
    free(A); free(B); free(C[0]); free(C[1]); free(C[2]);
  }
  return 0;
}
