/* vsame [FUNC...]: a hash of each one-argument function's results on all
   2^32 binary32 inputs, in input order, and which path computed them; and
   (FUNC "logsumexp" or "softmax", or none given) of the composites' results
   on 20,000 seeded vectors (lengths 1 to 3000; narrow, wide and huge
   values, ties at the maximum, -inf entries). Run
   once without CRNN_CRMVEC and once with it: the hashes must agree (the
   vector path claims the scalar path's bits on every input; crnn.c). Any
   change in any one result changes the hash. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "crnn.h"
#define CH (1u << 16)
static uint64_t fnv(const void *p, size_t n, uint64_t h)
{
  const unsigned char *b = p;
  for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 0x100000001b3ULL; }
  return h;
}
int main(int argc, char **argv)
{
  static const struct { const char *name; void (*f)(float *, const float *, size_t); } F[] = {
    {"sigmoid", crnn_sigmoidf}, {"silu", crnn_siluf}, {"gelu", crnn_geluf}, {"softplus", crnn_softplusf}, {"rsqrt", crnn_rsqrtf}};
  printf("path: %s\n", crnn_vector_path() ? "crmvec's vector code" : "scalar");
  static uint64_t hc[1u << 16];
  for (int f = 0; f < 5; f++) {
    int want = argc == 1;
    for (int a = 1; a < argc; a++) want |= !strcmp(argv[a], F[f].name);
    if (!want) continue;
#pragma omp parallel for schedule(dynamic, 16)
    for (long c = 0; c < (1L << 16); c++) {
      float x[CH], y[CH];
      for (uint32_t i = 0; i < CH; i++) { uint32_t u = (uint32_t)c * CH + i; memcpy(&x[i], &u, 4); }
      F[f].f(y, x, CH);
      hc[c] = fnv(y, sizeof y, 0xcbf29ce484222325ULL);
    }
    printf("%-9s %016llx\n", F[f].name, (unsigned long long)fnv(hc, sizeof hc, 0xcbf29ce484222325ULL));
    fflush(stdout);
  }
  for (int c = 0; c < 2; c++) {
    const char *nm = c ? "softmax" : "logsumexp";
    int want = argc == 1;
    for (int a = 1; a < argc; a++) want |= !strcmp(argv[a], nm);
    if (!want) continue;
    uint64_t h = 0xcbf29ce484222325ULL, seed = 20261001;
    float *x = malloc(3000 * sizeof *x), *y = malloc(3000 * sizeof *y);
    for (int v = 0; v < 20000; v++) {
#define NEXT (seed = seed * 6364136223846793005ULL + 1442695040888963407ULL, seed >> 11)
      size_t n = 1 + NEXT % (v % 10 == 0 ? 3000 : 64);
      int kind = v % 5;
      for (size_t i = 0; i < n; i++) {
        double u = (double)NEXT * 0x1p-53;
        x[i] = kind == 0 ? (float)(u * 8 - 4) : kind == 1 ? (float)(u * 200 - 100) : kind == 2 ? (float)(u * 2e30 - 1e30)
             : kind == 3 ? (float)(10 + (u < 0.5 ? 0 : u * 1e-3)) : (u < 0.1 ? -INFINITY : (float)(u * 50 - 25));
      }
      if (c) { crnn_softmaxf(y, x, n); h = fnv(y, n * sizeof *y, h); }
      else { float r = crnn_logsumexpf(x, n); h = fnv(&r, sizeof r, h); }
    }
    printf("%-9s %016llx (20000 vectors)\n", nm, (unsigned long long)h);
    free(x); free(y);
  }
  return 0;
}
