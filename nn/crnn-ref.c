/* crnn-ref.c: MPFR references for crnn's one-argument functions, as
   kit_mpfr1 functions (the correctly rounded value at y's precision, with
   the ternary value), for kit_ref1. Round to nearest only: crnn rounds
   only that way.

   Each is a Ziv loop: the function in MPFR at precision p with RNDN, a
   bound on the error in ulps, and mpfr_can_round with MPFR_RNDZ at one
   more bit, which also fixes the ternary value when the exact result is
   not representable. It never is here: every function below is
   transcendental at a nonzero rational argument, and the arguments where
   the value is rational (0, and the infinities) are answered first.

   Where the result rounds, provably, to a value that no amount of
   precision would approach (sigmoid near 1, silu, softplus and gelu near
   x, and all four near 0 for very negative x), the answer is given
   directly; the bounds are beside each. Everywhere else the loop reaches
   its answer by 512 bits (crnn-check counts any input that needs more,
   and aborts past 8192). */
#include <stdio.h>   /* before mpfr.h, for mpfr_fprintf */
#include <stdlib.h>
#include <mpfr.h>
#include "crnn-ref.h"

typedef void (*approx)(mpfr_ptr t, mpfr_srcptr x);   /* at t's precision, RNDN */

static int ziv(mpfr_ptr y, mpfr_srcptr x, mpfr_rnd_t rnd, approx f, int errbits)
{
  if (rnd != MPFR_RNDN) { fprintf(stderr, "crnn-ref: round to nearest only\n"); abort(); }
  mpfr_prec_t py = mpfr_get_prec(y);
  for (mpfr_prec_t p = 2 * py + 64; p <= 8192; p *= 2) {
    mpfr_t t; mpfr_init2(t, p);
    f(t, x);
    if (mpfr_regular_p(t) && mpfr_can_round(t, p - errbits, MPFR_RNDN, MPFR_RNDZ, py + 1)) {
      int r = mpfr_set(y, t, MPFR_RNDN);
      mpfr_clear(t);
      if (p > 512) __atomic_fetch_add(&crnn_ref_long, 1, __ATOMIC_RELAXED);
      return r;
    }
    mpfr_clear(t);
  }
  mpfr_fprintf(stderr, "crnn-ref: no rounding decided at 8192 bits for %Ra\n", x);
  abort();
}
unsigned long crnn_ref_long;

static void a_sigmoid(mpfr_ptr t, mpfr_srcptr x)   /* 1/(1 + e^-x): 3 roundings, condition below 1 */
{
  mpfr_neg(t, x, MPFR_RNDN); mpfr_exp(t, t, MPFR_RNDN); mpfr_add_ui(t, t, 1, MPFR_RNDN); mpfr_ui_div(t, 1, t, MPFR_RNDN);
}
static void a_silu(mpfr_ptr t, mpfr_srcptr x)
{
  a_sigmoid(t, x); mpfr_mul(t, t, x, MPFR_RNDN);
}
static void a_softplus(mpfr_ptr t, mpfr_srcptr x)  /* log1p(e^x), or x + log1p(e^-x) for x > 0 */
{
  if (mpfr_sgn(x) > 0) { mpfr_neg(t, x, MPFR_RNDN); mpfr_exp(t, t, MPFR_RNDN); mpfr_log1p(t, t, MPFR_RNDN); mpfr_add(t, t, x, MPFR_RNDN); }
  else { mpfr_exp(t, x, MPFR_RNDN); mpfr_log1p(t, t, MPFR_RNDN); }
}
static void a_gelu(mpfr_ptr t, mpfr_srcptr x)      /* x/2 erfc(-x/sqrt 2) */
{
  mpfr_t s; mpfr_init2(s, mpfr_get_prec(t));
  mpfr_sqrt_ui(s, 2, MPFR_RNDN); mpfr_div(t, x, s, MPFR_RNDN); mpfr_neg(t, t, MPFR_RNDN);
  mpfr_erfc(t, t, MPFR_RNDN); mpfr_mul(t, t, x, MPFR_RNDN); mpfr_div_2ui(t, t, 1, MPFR_RNDN);
  mpfr_clear(s);
}

/* The direct answers. 2^-25 is half an ulp just below 1, and 2^-150 half
   the smallest subnormal. Each ternary is the sign of (answer - exact). */
static int set_si(mpfr_ptr y, long v, int sign_of_zero, int ternary)
{
  mpfr_set_si(y, v, MPFR_RNDN);
  if (v == 0 && sign_of_zero < 0) mpfr_neg(y, y, MPFR_RNDN);
  return ternary;
}

int crnn_mpfr_sigmoid(mpfr_ptr y, mpfr_srcptr x, mpfr_rnd_t rnd)
{
  if (mpfr_nan_p(x)) { mpfr_set_nan(y); return 0; }
  if (mpfr_zero_p(x)) { mpfr_set_d(y, 0.5, MPFR_RNDN); return 0; }
  if (mpfr_inf_p(x)) return set_si(y, mpfr_sgn(x) > 0, 1, 0);
  /* x >= 32: 1 - sigmoid = 1/(1 + e^x) < e^-32 < 2^-46 < 2^-25: 1 */
  if (mpfr_cmp_si(x, 32) >= 0) return set_si(y, 1, 1, 1);
  /* x <= -105: sigmoid < e^-105 < 2^-151: +0 */
  if (mpfr_cmp_si(x, -105) <= 0) return set_si(y, 0, 1, -1);
  return ziv(y, x, rnd, a_sigmoid, 3);
}

int crnn_mpfr_silu(mpfr_ptr y, mpfr_srcptr x, mpfr_rnd_t rnd)
{
  if (mpfr_nan_p(x)) { mpfr_set_nan(y); return 0; }
  if (mpfr_zero_p(x)) { mpfr_set(y, x, MPFR_RNDN); return 0; }
  if (mpfr_inf_p(x)) { if (mpfr_sgn(x) > 0) { mpfr_set_inf(y, 1); return 0; } return set_si(y, 0, -1, 1); }
  /* x >= 32: silu = x (1 - d), d < e^-32 < 2^-46: x (below it) */
  if (mpfr_cmp_si(x, 32) >= 0) { mpfr_set(y, x, MPFR_RNDN); return 1; }
  /* x <= -112: |silu| < |x| e^x, which decreases as x falls below -1, and
     112 e^-112 < 2^-154: -0 */
  if (mpfr_cmp_si(x, -112) <= 0) return set_si(y, 0, -1, 1);
  return ziv(y, x, rnd, a_silu, 4);
}

int crnn_mpfr_softplus(mpfr_ptr y, mpfr_srcptr x, mpfr_rnd_t rnd)
{
  if (mpfr_nan_p(x)) { mpfr_set_nan(y); return 0; }
  if (mpfr_inf_p(x)) { if (mpfr_sgn(x) > 0) { mpfr_set_inf(y, 1); return 0; } return set_si(y, 0, 1, 0); }
  /* x >= 32: softplus = x + log1p(e^-x), the addend below e^-32 < 2^-46
     and so below 2^-51 x < half an ulp of x: x (below it) */
  if (mpfr_cmp_si(x, 32) >= 0) { mpfr_set(y, x, MPFR_RNDN); return -1; }
  /* x <= -105: softplus < e^x < 2^-151: +0 */
  if (mpfr_cmp_si(x, -105) <= 0) return set_si(y, 0, 1, -1);
  return ziv(y, x, rnd, a_softplus, 3);
}

int crnn_mpfr_gelu(mpfr_ptr y, mpfr_srcptr x, mpfr_rnd_t rnd)
{
  if (mpfr_nan_p(x)) { mpfr_set_nan(y); return 0; }
  if (mpfr_zero_p(x)) { mpfr_set(y, x, MPFR_RNDN); return 0; }
  if (mpfr_inf_p(x)) { if (mpfr_sgn(x) > 0) { mpfr_set_inf(y, 1); return 0; } return set_si(y, 0, -1, 1); }
  /* x >= 16: gelu = x (1 - erfc(x/sqrt 2)/2), erfc(11.3) < 2^-190: x (above it) */
  if (mpfr_cmp_si(x, 16) >= 0) { mpfr_set(y, x, MPFR_RNDN); return 1; }
  /* x <= -16: |gelu| = |x|/2 erfc(|x|/sqrt 2), which decreases from 8
     erfc(11.3) < 2^-186: -0 */
  if (mpfr_cmp_si(x, -16) <= 0) return set_si(y, 0, -1, 1);
  /* in (-16, 16): t = -x/sqrt 2 carries 2 ulps; erfc's condition number,
     at most 2 t^2 + 1 < 2^8.1, makes that under 2^9.1; erfc, the product
     and the halving (exact) add 2: under 2^10 ulps */
  return ziv(y, x, rnd, a_gelu, 10);
}
