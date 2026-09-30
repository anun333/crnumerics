/* sum.c: the many-input reference (MPFR's mpfr_sum) and an order shuffler
   (kit.h), for reductions. */
#include <stdlib.h>
#include "kit.h"

double kit_sum_ref(kit_fmt f, const double *x, const double *y, size_t n, mpfr_rnd_t rnd)
{
  mpfr_t *t = malloc(sizeof(mpfr_t) * (n ? n : 1));
  mpfr_ptr *p = malloc(sizeof(mpfr_ptr) * (n ? n : 1));
  for (size_t i = 0; i < n; i++) {
    mpfr_init2(t[i], 106);   /* a product of two binary64 values, exactly */
    if (y) {
      mpfr_t a, b;
      mpfr_init2(a, 53);
      mpfr_init2(b, 53);
      mpfr_set_d(a, x[i], MPFR_RNDN);
      mpfr_set_d(b, y[i], MPFR_RNDN);
      mpfr_mul(t[i], a, b, MPFR_RNDN);   /* exact: 106 bits, MPFR's exponent range */
      mpfr_clears(a, b, (mpfr_ptr)0);
    } else
      mpfr_set_d(t[i], x[i], MPFR_RNDN);
    p[i] = t[i];
  }
  mpfr_t r;
  mpfr_init2(r, kit_info(f)->mbits + 1);
  int tern = mpfr_sum(r, p, n, rnd);
  double d = kit_decode(f, kit_round_t(f, r, tern, rnd));
  mpfr_clear(r);
  for (size_t i = 0; i < n; i++) mpfr_clear(t[i]);
  free(t);
  free(p);
  return d;
}

void kit_shuffle(uint64_t seed, size_t *idx, size_t n)
{
  for (size_t i = 0; i < n; i++) idx[i] = i;
  uint64_t z = seed;
  for (size_t i = n; i > 1; i--) {   /* Fisher-Yates, splitmix64 */
    z += 0x9e3779b97f4a7c15ULL;
    uint64_t r = z;
    r = (r ^ (r >> 30)) * 0xbf58476d1ce4e5b9ULL;
    r = (r ^ (r >> 27)) * 0x94d049bb133111ebULL;
    r ^= r >> 31;
    size_t j = (size_t)(r % i), k = idx[i - 1];
    idx[i - 1] = idx[j];
    idx[j] = k;
  }
}
