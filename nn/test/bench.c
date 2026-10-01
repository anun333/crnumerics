/* bench: crnn's speed against the naive binary32 formulas (the C
   library's expf, erff, log1pf, sqrtf), ns per element, one thread, best
   of 7 passes after a warm-up. The naive results are summed into a
   checksum that is printed, so the loops can't be dropped. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "crnn.h"

enum { N = 1 << 16 };
static float x[N], y[N], g[N], b[N];
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
#define BEST(out, body) do { double best_ = 1e9; for (int p_ = 0; p_ < 8; p_++) { double t0_ = now(); body; double t_ = now() - t0_; if (p_ && t_ < best_) best_ = t_; } out = best_ / N * 1e9; } while (0)

int main(void)
{
  srand(1);
  for (int i = 0; i < N; i++) { x[i] = (float)((rand() / (RAND_MAX + 1.0) - 0.5) * 16); g[i] = 1; b[i] = 0; }
  volatile float sink = 0; float acc;
  double c, n;
  printf("crnn's one-argument functions: %s\n", crnn_vector_path() ? "crmvec's vector code (CRNN_CRMVEC)" : "the scalar path");
  printf("%-10s %8s %8s %6s   (ns per element, %d elements, one thread)\n", "", "crnn", "naive", "ratio", N);
#define ROW(name, CR, NAIVE) BEST(c, CR); BEST(n, { acc = 0; for (int i = 0; i < N; i++) acc += (NAIVE); sink += acc; }); \
  printf("%-10s %8.2f %8.2f %5.1fx\n", name, c, n, c / n)
  ROW("sigmoid", crnn_sigmoidf(y, x, N), 1.0f / (1.0f + expf(-x[i])));
  ROW("silu", crnn_siluf(y, x, N), x[i] / (1.0f + expf(-x[i])));
  ROW("gelu", crnn_geluf(y, x, N), 0.5f * x[i] * (1.0f + erff(x[i] * 0.70710678f)));
  ROW("softplus", crnn_softplusf(y, x, N), log1pf(expf(x[i])));
  ROW("rsqrt", crnn_rsqrtf(y, g, N), 1.0f / sqrtf(g[i] + x[i] * x[i]));
  float m = -INFINITY; for (int i = 0; i < N; i++) if (x[i] > m) m = x[i];
  ROW("logsumexp", sink += crnn_logsumexpf(x, N), expf(x[i] - m));
  ROW("softmax", crnn_softmaxf(y, x, N), expf(x[i] - m) * 0.5f);
  ROW("layernorm", crnn_layernormf(y, x, N, g, b, 1e-5f), (x[i] - 0.1f) * 0.9f + b[i]);
  ROW("rmsnorm", crnn_rmsnormf(y, x, N, g, 1e-6f), x[i] * x[i] + g[i]);
  printf("(checksum %g; the naive composites are their main loop only, so the ratios there are generous to them)\n", (double)sink);
  return 0;
}
