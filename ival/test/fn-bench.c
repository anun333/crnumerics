/* fn-bench.c: the cost of ival's functions, in ns per interval, on 4096 intervals at a time.

   For each function, narrow intervals (width up to 1e-3 relative, the common case in a computation) and wide ones
   (up to the whole of [-10, 10]), in its domain; the tight mode (ival_f) and the accurate one (ival_acc_f, through
   crmvec when IVAL_CRMVEC names it). Each figure is the least of 7 passes after a warm-up, each pass at
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
static const struct { const char *name; f1 f, acc; double lo, hi; } F1[] = {
  { "exp", ival_exp, ival_acc_exp, -10, 10 }, { "log", ival_log, ival_acc_log, 1e-3, 100 },
  { "atan", ival_atan, ival_acc_atan, -10, 10 }, { "sin", ival_sin, ival_acc_sin, -10, 10 },
  { "cos", ival_cos, ival_acc_cos, -10, 10 }, { "tan", ival_tan, ival_acc_tan, -1.5, 1.5 },
  { "cosh", ival_cosh, ival_acc_cosh, -10, 10 }, { "sinpi", ival_sinpi, ival_acc_sinpi, -10, 10 },
  { "tgamma", ival_tgamma, ival_acc_tgamma, 0.1, 10 } };
static const struct { const char *name; f2 f, acc; double lo, hi, ylo, yhi; } F2[] = {
  { "pow", ival_pow, ival_acc_pow, 0.1, 10, -3, 3 }, { "hypot", ival_hypot, ival_acc_hypot, -10, 10, -10, 10 },
  { "atan2", ival_atan2, ival_acc_atan2, -10, 10, -10, 10 } };

enum { N = 4096 };
static double a[N], b[N], c[N], d[N], yl[N], yh[N];
static int acc;   /* time the accurate mode */
#define CALL(k2, i, n) do { if (k2) (acc ? F2[i].acc : F2[i].f)(a, b, c, d, yl, yh, n); else (acc ? F1[i].acc : F1[i].f)(a, b, yl, yh, n); } while (0)
static double best(int k2, int i, size_t n)
{
  size_t reps = 1;
  for (;;) {
    double t = now();
    for (size_t r = 0; r < reps; r++) CALL(k2, i, n);
    if (now() - t > 0.02) break;
    reps *= 2;
  }
  double m = 1e30;
  for (int p = 0; p < 7; p++) {
    double t = now();
    for (size_t r = 0; r < reps; r++) CALL(k2, i, n);
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
  const char *lib = getenv("IVAL_CRMVEC");
  printf("ns per interval, n = %d (least of 7 passes; load %.2f; accurate mode %s)\n", N, la[0],
         lib && *lib ? "through crmvec" : "WITHOUT crmvec, so the tight mode");
  printf("            tight narrow   wide   accurate narrow   wide\n");
  for (unsigned i = 0; i < sizeof F1 / sizeof F1[0]; i++) {
    printf("%-10s  ", F1[i].name);
    for (acc = 0; acc < 2; acc++)
      for (int wide = 0; wide < 2; wide++) {
        fill(F1[i].lo, F1[i].hi, wide, a, b);
        double t = best(0, i, N);
        if (t < 0.5) printf("    FOLDED"); else printf(" %9.1f", t);
      }
    printf("\n");
  }
  for (unsigned i = 0; i < sizeof F2 / sizeof F2[0]; i++) {
    printf("%-10s  ", F2[i].name);
    for (acc = 0; acc < 2; acc++)
      for (int wide = 0; wide < 2; wide++) {
        fill(F2[i].lo, F2[i].hi, wide, a, b); fill(F2[i].ylo, F2[i].yhi, wide, c, d);
        double t = best(1, i, N);
        if (t < 0.5) printf("    FOLDED"); else printf(" %9.1f", t);
      }
    printf("\n");
  }
  return 0;
}
