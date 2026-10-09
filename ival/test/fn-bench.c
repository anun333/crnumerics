/* fn-bench.c: the cost of ival's functions, in ns per interval, on 4096 intervals at a time.

   For each function, narrow intervals (width up to 1e-3 relative, the common case in a computation) and wide ones
   (up to the whole of [-10, 10]), in its domain. Each figure is the least of 7 passes after a warm-up, each pass at
   least 20 ms; below 0.5 ns it prints FOLDED. The load average is printed beside the figures. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "ival.h"

static uint64_t rs = 0x853c49e6748fea9bULL;
static double unit(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (double)(rs >> 11) * 0x1p-53; }
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }

typedef void (*f1)(const double *, const double *, double *, double *, size_t);
typedef void (*f2)(const double *, const double *, const double *, const double *, double *, double *, size_t);
static const struct { const char *name; f1 f; double lo, hi; } F1[] = {
  { "exp", ival_exp, -10, 10 }, { "log", ival_log, 1e-3, 100 }, { "atan", ival_atan, -10, 10 },
  { "sin", ival_sin, -10, 10 }, { "cos", ival_cos, -10, 10 }, { "tan", ival_tan, -1.5, 1.5 },
  { "cosh", ival_cosh, -10, 10 }, { "sinpi", ival_sinpi, -10, 10 }, { "tgamma", ival_tgamma, 0.1, 10 } };
static const struct { const char *name; f2 f; double lo, hi, ylo, yhi; } F2[] = {
  { "pow", ival_pow, 0.1, 10, -3, 3 }, { "hypot", ival_hypot, -10, 10, -10, 10 }, { "atan2", ival_atan2, -10, 10, -10, 10 } };

enum { N = 4096 };
static double a[N], b[N], c[N], d[N], yl[N], yh[N];
static double best(int k2, int i, size_t n)
{
  size_t reps = 1;
  for (;;) {
    double t = now();
    for (size_t r = 0; r < reps; r++) { if (k2) F2[i].f(a, b, c, d, yl, yh, n); else F1[i].f(a, b, yl, yh, n); }
    if (now() - t > 0.02) break;
    reps *= 2;
  }
  double m = 1e30;
  for (int p = 0; p < 7; p++) {
    double t = now();
    for (size_t r = 0; r < reps; r++) { if (k2) F2[i].f(a, b, c, d, yl, yh, n); else F1[i].f(a, b, yl, yh, n); }
    t = (now() - t) / ((double)reps * n) * 1e9;
    if (t < m) m = t;
  }
  return m;
}
static void fill(double lo, double hi, int wide, double *x, double *y)
{
  for (int k = 0; k < N; k++) {
    double u = lo + unit() * (hi - lo), w = wide ? unit() * (hi - lo) : fabs(u) * unit() * 1e-3;
    x[k] = u; y[k] = u + w > hi ? hi : u + w;
  }
}
int main(void)
{
  double la[3] = { 0 };
  FILE *f = fopen("/proc/loadavg", "r");
  if (f) { if (fscanf(f, "%lf %lf %lf", &la[0], &la[1], &la[2]) != 3) la[0] = -1; fclose(f); }
  printf("ns per interval, n = %d (least of 7 passes; load %.2f)   narrow    wide\n", N, la[0]);
  for (unsigned i = 0; i < sizeof F1 / sizeof F1[0]; i++) {
    printf("%-10s                                              ", F1[i].name);
    for (int wide = 0; wide < 2; wide++) {
      fill(F1[i].lo, F1[i].hi, wide, a, b);
      double t = best(0, i, N);
      if (t < 0.5) printf("  FOLDED"); else printf(" %7.1f", t);
    }
    printf("\n");
  }
  for (unsigned i = 0; i < sizeof F2 / sizeof F2[0]; i++) {
    printf("%-10s                                              ", F2[i].name);
    for (int wide = 0; wide < 2; wide++) {
      fill(F2[i].lo, F2[i].hi, wide, a, b); fill(F2[i].ylo, F2[i].yhi, wide, c, d);
      double t = best(1, i, N);
      if (t < 0.5) printf("  FOLDED"); else printf(" %7.1f", t);
    }
    printf("\n");
  }
  return 0;
}
