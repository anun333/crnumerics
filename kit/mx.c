/* mx.c: the correctly rounded function on an MX block, exactly, from MPFR
   (kit.h; the definition is lowp/MX.md). */
#include <math.h>
#include <stdlib.h>
#include "kit.h"

int kit_mx_emax(kit_fmt f)
{
  switch (f) {
  case KIT_E5M2: return 15;
  case KIT_E4M3: return 8;
  case KIT_E3M2: return 4;
  case KIT_E2M3: case KIT_E2M1: return 2;
  default: abort();   /* not an MX element type */
  }
}

/* the element's value times 2^(x - 127), exactly; 0 if it is NaN or
   infinite (FP8), which makes the block NaN */
static int value(kit_fmt f, uint64_t p, int x, mpfr_ptr v)
{
  double d = kit_decode(f, p);
  if (isnan(d) || isinf(d)) return 0;
  mpfr_set_d(v, d, MPFR_RNDN);             /* exact: v has 53 bits */
  mpfr_mul_2si(v, v, x - 127, MPFR_RNDN);  /* exact: far inside MPFR's range */
  return 1;
}

static void nan_block(int k, uint64_t *q, int *y)
{
  for (int i = 0; i < k; i++) q[i] = 0;
  *y = 0xff;
}

/* the block from the inputs a (and b), per lowp/MX.md */
static void block(kit_fmt f, kit_mpfr1 fn1, kit_mpfr2 fn2, int k, mpfr_t *a, mpfr_t *b, mpfr_rnd_t rnd, uint64_t *q,
                  int *y)
{
  const kit_fmtinfo *info = kit_info(f);
  mpfr_t z;
  mpfr_init2(z, 64);
  /* the scale: each result rounded toward zero keeps its exponent, which
     is floor(log2 |result|) + 1 in MPFR's convention; NaN or infinity
     anywhere makes the block NaN (an overflow past MPFR's own range
     rounds toward zero to its largest number, and gives a NaN scale) */
  long E = -126;
  int any = 0;
  for (int i = 0; i < k; i++) {
    if (fn1) fn1(z, a[i], MPFR_RNDZ); else fn2(z, a[i], b[i], MPFR_RNDZ);
    if (mpfr_nan_p(z) || mpfr_inf_p(z)) { mpfr_clear(z); nan_block(k, q, y); return; }
    if (mpfr_zero_p(z)) continue;
    long e = mpfr_get_exp(z) - 1;
    if (!any || e > E) E = e;
    any = 1;
  }
  long s = E - kit_mx_emax(f);
  if (s > 127) { mpfr_clear(z); nan_block(k, q, y); return; }
  if (s < -127) s = -127;
  /* the elements: each result rounded to the element's precision, moved
     by 2^-s (exact, or else rounded again in the same direction, only far
     below the element's smallest value), then to the element type,
     saturating (an infinity or NaN out of the kit's rounding can only be
     an overflow, since the exact result is finite) */
  mpfr_set_prec(z, info->mbits + 1);
  for (int i = 0; i < k; i++) {
    int t = fn1 ? fn1(z, a[i], rnd) : fn2(z, a[i], b[i], rnd);
    int t2 = mpfr_mul_2si(z, z, -s, rnd);
    if (t2) t = t2;
    int neg = mpfr_signbit(z);
    uint64_t r = kit_round_t(f, z, t, rnd);
    double d = kit_decode(f, r);
    if (r == KIT_NONE || isnan(d) || isinf(d)) r = kit_encode(f, neg ? -kit_max(f) : kit_max(f));
    q[i] = r;
  }
  *y = (int)(s + 127);
  mpfr_clear(z);
}

void kit_mx_ref1(kit_fmt f, kit_mpfr1 fn, int k, const uint64_t *p, int x, mpfr_rnd_t rnd, uint64_t *q, int *y)
{
  mpfr_t *a = malloc(sizeof(mpfr_t) * (size_t)k);
  int ok = x != 0xff;
  for (int i = 0; i < k; i++) {
    mpfr_init2(a[i], 53);
    if (ok) ok = value(f, p[i], x, a[i]);
  }
  if (ok) block(f, fn, 0, k, a, 0, rnd, q, y);
  else nan_block(k, q, y);
  for (int i = 0; i < k; i++) mpfr_clear(a[i]);
  free(a);
}

void kit_mx_ref2(kit_fmt f, kit_mpfr2 fn, int k, const uint64_t *p, int x, const uint64_t *p2, int x2, mpfr_rnd_t rnd,
                 uint64_t *q, int *y)
{
  mpfr_t *a = malloc(sizeof(mpfr_t) * (size_t)k), *b = malloc(sizeof(mpfr_t) * (size_t)k);
  int ok = x != 0xff && x2 != 0xff;
  for (int i = 0; i < k; i++) {
    mpfr_init2(a[i], 53);
    mpfr_init2(b[i], 53);
    if (ok) ok = value(f, p[i], x, a[i]) && value(f, p2[i], x2, b[i]);
  }
  if (ok) block(f, 0, fn, k, a, b, rnd, q, y);
  else nan_block(k, q, y);
  for (int i = 0; i < k; i++) { mpfr_clear(a[i]); mpfr_clear(b[i]); }
  free(a);
  free(b);
}
