/* crfix-ref.c: crfix-ref.h, with MPFR (2026-10-14). Each value is enclosed: a function correctly rounded to
   nearest at precision p, with its ternary value, puts the exact value at the result (ternary 0) or strictly
   between it and its neighbour on the side the ternary names, an open interval (a, b). The rounding is decided
   when no boundary lies strictly inside (a, b) 2^fout; otherwise the precision is raised (the exact value of
   these functions at these inputs is never a rounding boundary unless it is exact, which the ternary value then
   says). Deciding from open ends matters: tanh(-16384) is -1 + 2 e^-32768, which no closed end short of 47,000
   bits separates from -1. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <gmp.h>
#include <mpfr.h>
#include "crfix.h"
#include "crfix-ref.h"

int32_t crfix_ref_sat(int64_t q, int bout)
{
  int64_t mx = ((int64_t)1 << (bout - 1)) - 1;
  return (int32_t)(q > mx ? mx : q < -mx - 1 ? -mx - 1 : q);
}

static const mpfr_rnd_t RND[4] = {MPFR_RNDN, MPFR_RNDU, MPFR_RNDD, MPFR_RNDZ};

/* an integer-valued s as a 64-bit integer; a value that does not fit stops the run, since a wrapped conversion
   would be a wrong expected result that looks like any other */
static intmax_t get(mpfr_srcptr s)
{
  if (!mpfr_fits_intmax_p(s, MPFR_RNDN)) { fprintf(stderr, "crfix-ref: a result does not fit 64 bits\n"); abort(); }
  return mpfr_get_sj(s, MPFR_RNDN);
}

/* a <= exact value <= b, at precision p (a, b set to p): 1 if exact (then a = b = it) */
static int enclose(int fn, int32_t x, int fin, mpfr_prec_t p, mpfr_t a, mpfr_t b)
{
  mpfr_t v;
  int t;
  mpfr_init2(v, 64);
  mpfr_set_si_2exp(v, x, -fin, MPFR_RNDN);   /* exact */
  mpfr_set_prec(a, p);
  mpfr_set_prec(b, p);
  switch (fn) {
    case CRFIX_EXP: t = mpfr_exp(a, v, MPFR_RNDN); break;
    case CRFIX_EXP2: t = mpfr_exp2(a, v, MPFR_RNDN); break;
    case CRFIX_LOG: t = mpfr_log(a, v, MPFR_RNDN); break;
    case CRFIX_LOG2: t = mpfr_log2(a, v, MPFR_RNDN); break;
    case CRFIX_SIGMOID: mpfr_neg(v, v, MPFR_RNDN); t = mpfr_exp(a, v, MPFR_RNDN); break;   /* e^-v, then below */
    case CRFIX_TANH: t = mpfr_tanh(a, v, MPFR_RNDN); break;
    case CRFIX_SQRT: t = mpfr_sqrt(a, v, MPFR_RNDN); break;
    case CRFIX_RSQRT: t = mpfr_rec_sqrt(a, v, MPFR_RNDN); break;
    default: t = mpfr_ui_div(a, 1, v, MPFR_RNDN); break;
  }
  mpfr_set(b, a, MPFR_RNDN);
  if (t > 0) mpfr_nextbelow(a);
  else if (t < 0) mpfr_nextabove(b);
  if (fn == CRFIX_SIGMOID) {   /* 1 / (1 + e^-v), each end rounded outward: relative accuracy at any v (the
                                   first version took (1 + tanh(v / 2)) / 2, which keeps none for e^-32768) */
    int t1 = mpfr_add_ui(b, b, 1, MPFR_RNDU), t2 = mpfr_ui_div(b, 1, b, MPFR_RNDD);
    int t3 = mpfr_add_ui(a, a, 1, MPFR_RNDD), t4 = mpfr_ui_div(a, 1, a, MPFR_RNDU);
    mpfr_swap(a, b);   /* a = 1 / (1 + the upper e^-v), b = 1 / (1 + the lower) */
    t = t || t1 || t2 || t3 || t4;
  }
  mpfr_clear(v);
  return t == 0;
}

/* the rounding of the exact value times 2^fout, from its enclosure: exact (a = b = it), rounded in the mode, ties
   to even; open (a, b), decided when no integer (nearest: no integer plus a half) lies strictly inside, and then
   floor and ceiling are the integers around it. 1 and the result, or 0 */
static int round_ends(mpfr_srcptr a, mpfr_srcptr b, int exact, int fout, int mode, mpfr_t s, int64_t *q)
{
  mpfr_set_prec(s, mpfr_get_prec(a) + 80);
  if (exact) {
    mpfr_mul_2si(s, a, fout, MPFR_RNDN);
    mpfr_rint(s, s, RND[mode]);
    *q = (int64_t)get(s);
    return 1;
  }
  mpfr_mul_2si(s, a, fout, MPFR_RNDN);
  if (mode == 0) mpfr_add_d(s, s, 0.5, MPFR_RNDN);   /* exact at this precision */
  mpfr_floor(s, s);
  intmax_t A = get(s);
  mpfr_mul_2si(s, b, fout, MPFR_RNDN);
  if (mode == 0) mpfr_add_d(s, s, 0.5, MPFR_RNDN);
  mpfr_ceil(s, s);
  intmax_t B = get(s);
  if (B - A > 1) return 0;
  switch (mode) {
    case 0: case 2: *q = (int64_t)A; break;   /* nearest: floor(v + 1/2); down: floor */
    case 1: *q = (int64_t)A + 1; break;
    default: *q = A >= 0 ? (int64_t)A : (int64_t)A + 1; break;
  }
  return 1;
}

void crfix_ref1(int fn, int32_t x, int fin, int64_t out[128])
{
  int64_t fill = 1;
  if ((fn == CRFIX_LOG || fn == CRFIX_LOG2) && x <= 0) fill = INT64_MIN;
  else if (fn == CRFIX_SQRT && x <= 0) fill = 0;
  else if ((fn == CRFIX_RSQRT && x <= 0) || (fn == CRFIX_RECIP && !x)) fill = INT64_MAX;
  if (fill != 1) { for (int i = 0; i < 128; i++) out[i] = fill; return; }
  mpfr_t a, b, s;
  mpfr_inits2(256, a, b, s, (mpfr_ptr)0);
  int done[128] = {0}, left = 128;
  for (mpfr_prec_t p = 256; left; p *= 4) {
    int ex = enclose(fn, x, fin, p, a, b);
    if (mpfr_cmp_ui_2exp(a, 1, 31) >= 0) {   /* >= 2^31: past every container */
      for (int i = 0; i < 128; i++) out[i] = INT64_MAX;
      break;
    }
    for (int i = 0; i < 128; i++)
      if (!done[i]) {
        /* past every container at this fout: saturates however it rounds */
        if (mpfr_cmp_ui_2exp(a, 1, 31 - i / 4) >= 0) { out[i] = INT64_MAX; done[i] = 1; left--; continue; }
        if (round_ends(a, b, ex, i / 4, i % 4, s, &out[i])) { done[i] = 1; left--; }
      }
    if (p > 1 << 20) { fprintf(stderr, "crfix_ref1: fn %d x %d fin %d undecided at %ld bits\n", fn, (int)x, fin, (long)p); abort(); }
  }
  mpfr_clears(a, b, s, (mpfr_ptr)0);
}

int64_t crfix_ref_div(int32_t a, int fa, int32_t b, int fb, int fout, int mode)
{
  if (!b) return a > 0 ? INT64_MAX : a < 0 ? INT64_MIN : 0;
  mpfr_t va, vb, y, z, s;
  int64_t q = 0;
  mpfr_inits2(64, va, vb, (mpfr_ptr)0);
  mpfr_inits2(256, y, z, s, (mpfr_ptr)0);
  mpfr_set_si_2exp(va, a, -fa, MPFR_RNDN);
  mpfr_set_si_2exp(vb, b, -fb, MPFR_RNDN);
  for (mpfr_prec_t p = 256;; p *= 4) {
    mpfr_set_prec(y, p); mpfr_set_prec(z, p);
    int t = mpfr_div(y, va, vb, MPFR_RNDN);
    int ex = t == 0;
    mpfr_set(z, y, MPFR_RNDN);
    if (t > 0) mpfr_nextbelow(y);
    else if (t < 0) mpfr_nextabove(z);
    /* past every container once scaled by 2^fout (the first version compared the unscaled quotient with 2^40, and
       a quotient of 1.3e11 at fout 31 overflowed the conversion to a negative number) */
    mpfr_t m;
    mpfr_init2(m, 8);
    mpfr_set_ui_2exp(m, 1, 40 - fout, MPFR_RNDN);
    if (mpfr_cmp(y, m) >= 0) { q = INT64_MAX; mpfr_clear(m); break; }
    mpfr_neg(m, m, MPFR_RNDN);
    if (mpfr_cmp(z, m) <= 0) { q = INT64_MIN; mpfr_clear(m); break; }
    mpfr_clear(m);
    if (round_ends(y, z, ex, fout, mode, s, &q)) break;
  }
  mpfr_clears(va, vb, y, z, s, (mpfr_ptr)0);
  return q;
}

int crfix_ref_inside(int fn, int32_t x, int fin, uint64_t lo_hi, uint64_t lo_lo, uint64_t hi_hi, uint64_t hi_lo,
                     int exact)
{
  mpfr_t a, b, l, h;
  mpz_t z;
  int r;
  mpfr_inits2(256, a, b, (mpfr_ptr)0);
  mpfr_inits2(160, l, h, (mpfr_ptr)0);
  mpz_init(z);
  /* the signed Q64.64 words as numbers, exactly */
  mpz_set_ui(z, lo_hi); mpz_mul_2exp(z, z, 64); mpz_add_ui(z, z, lo_lo);
  if (lo_hi >> 63) { mpz_t m; mpz_init(m); mpz_ui_pow_ui(m, 2, 128); mpz_sub(z, z, m); mpz_clear(m); }
  mpfr_set_z_2exp(l, z, -64, MPFR_RNDN);
  mpz_set_ui(z, hi_hi); mpz_mul_2exp(z, z, 64); mpz_add_ui(z, z, hi_lo);
  if (hi_hi >> 63) { mpz_t m; mpz_init(m); mpz_ui_pow_ui(m, 2, 128); mpz_sub(z, z, m); mpz_clear(m); }
  mpfr_set_z_2exp(h, z, -64, MPFR_RNDN);
  int ex = enclose(fn, x, fin, 256, a, b);
  if (exact) r = ex && mpfr_equal_p(a, l);
  else r = !ex ? mpfr_cmp(l, a) <= 0 && mpfr_cmp(b, h) <= 0    /* lo <= a < exact < b <= hi */
               : mpfr_cmp(l, a) < 0 && mpfr_cmp(a, h) < 0;     /* an exact value inside an open enclosure */
  mpfr_clears(a, b, l, h, (mpfr_ptr)0);
  mpz_clear(z);
  return r;
}
