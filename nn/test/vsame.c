/* vsame [FUNC...]: a hash of each one-argument function's results on all
   2^32 binary32 inputs, in input order, and which path computed them. Run
   once without CRNN_CRMVEC and once with it: the hashes must agree (the
   vector path claims the scalar path's bits on every input; crnn.c). Any
   change in any one result changes the hash. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
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
  return 0;
}
