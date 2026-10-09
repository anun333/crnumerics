/* ia-bench.c: the C side of make ival-julia-compare. ival's tight exp, exp2 and tanh on 4096 narrow intervals
   (x in [-5, 5], about 2^-40 wide relative), ns per interval, the least of 7 passes after one; then writes the
   intervals and ival's results to ivals.bin for ia-bench.jl to compare */
#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include "ival.h"
enum { N = 4096 };
static double lo[N], hi[N], yl[3][N], yh[3][N];
int main(void) {
  uint64_t s = 0x9e3779b97f4a7c15ULL;
  for (int i = 0; i < N; i++) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; double x = ((double)(s >> 11) * 0x1p-53 - 0.5) * 10;
    lo[i] = x; hi[i] = x + (x < 0 ? -x : x) * 0x1p-40 + 0x1p-60; }
  void (*fs[3])(const double *, const double *, double *, double *, size_t) = { ival_exp, ival_exp2, ival_tanh };
  const char *nm[3] = { "exp", "exp2", "tanh" };
  for (int f = 0; f < 3; f++) {
    double best = 1e30;
    for (int r = 0; r < 8; r++) {
      struct timespec a, b; clock_gettime(CLOCK_MONOTONIC, &a); fs[f](lo, hi, yl[f], yh[f], N); clock_gettime(CLOCK_MONOTONIC, &b);
      double ns = ((b.tv_sec - a.tv_sec) * 1e9 + (b.tv_nsec - a.tv_nsec)) / N; if (r && ns < best) best = ns;
    }
    printf("ival %-5s %7.1f ns\n", nm[f], best);
  }
  FILE *o = fopen("ivals.bin", "wb"); fwrite(lo, 8, N, o); fwrite(hi, 8, N, o);
  for (int f = 0; f < 3; f++) { fwrite(yl[f], 8, N, o); fwrite(yh[f], 8, N, o); } fclose(o);
  return 0;
}
