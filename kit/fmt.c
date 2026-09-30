/* fmt.c: the kit's formats, and the correctly rounded reference through
   MPFR (kit.h). */
#include "kit.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const kit_fmtinfo INFO[KIT_NFMT] = {
  {"binary64", 64, 11, 52, 1023, 1, 1}, {"binary32", 32, 8, 23, 127, 1, 1}, {"binary16", 16, 5, 10, 15, 1, 1},
  {"bfloat16", 16, 8, 7, 127, 1, 1},    {"E5M2", 8, 5, 2, 15, 1, 1},        {"E4M3", 8, 4, 3, 7, 0, 1},
  {"E2M3", 6, 2, 3, 1, 0, 0},           {"E3M2", 6, 3, 2, 3, 0, 0},         {"E2M1", 4, 2, 1, 1, 0, 0},
};

int kit_e4m3_saturate, kit_nan_bits;

const kit_fmtinfo *kit_info(kit_fmt f) { return &INFO[f]; }

static uint64_t width(const kit_fmtinfo *i) { return i->bits == 64 ? ~0ULL : (1ULL << i->bits) - 1; }
static uint64_t sign(const kit_fmtinfo *i) { return 1ULL << (i->bits - 1); }
static uint64_t mmask(const kit_fmtinfo *i) { return (1ULL << i->mbits) - 1; }
static uint64_t eall(const kit_fmtinfo *i) { return (1ULL << i->ebits) - 1; }
static uint64_t efield(const kit_fmtinfo *i, uint64_t b) { return (b >> i->mbits) & eall(i); }
/* the exponent of the largest finite value */
static int emax(const kit_fmtinfo *i) { return (1 << i->ebits) - (i->has_inf ? 2 : 1) - i->bias; }
/* the canonical quiet NaN: the quiet bit alone (E4M3: its one NaN; the MX
   element formats: none) */
static uint64_t qnan(const kit_fmtinfo *i)
{ return !i->has_nan ? KIT_NONE : eall(i) << i->mbits | (i->has_inf ? 1ULL << (i->mbits - 1) : mmask(i)); }

int kit_isnan(kit_fmt f, uint64_t b)
{
  const kit_fmtinfo *i = &INFO[f];
  uint64_t m = b & mmask(i);
  return i->has_nan && efield(i, b) == eall(i) && (i->has_inf ? m != 0 : m == mmask(i));
}
static int is_snan(const kit_fmtinfo *i, kit_fmt f, uint64_t b)
{ return i->has_inf && kit_isnan(f, b) && !(b >> (i->mbits - 1) & 1); }

double kit_decode(kit_fmt f, uint64_t b)
{
  const kit_fmtinfo *i = &INFO[f];
  if (f == KIT_B64) { double d; memcpy(&d, &b, 8); return d; }
  b &= width(i);
  uint64_t e = efield(i, b), m = b & mmask(i);
  double v;
  if (kit_isnan(f, b)) v = NAN;
  else if (i->has_inf && e == eall(i)) v = INFINITY;
  else if (e == 0) v = ldexp((double)m, 1 - i->bias - i->mbits);   /* exact: every operation here is */
  else v = ldexp((double)(m | 1ULL << i->mbits), (int)e - i->bias - i->mbits);
  return b & sign(i) ? -v : v;
}

static _Noreturn void unrepresentable(kit_fmt f, double v)
{ fprintf(stderr, "kit_encode: %a is not a %s value\n", v, INFO[f].name); abort(); }

uint64_t kit_encode(kit_fmt f, double v)
{
  const kit_fmtinfo *i = &INFO[f];
  if (isnan(v)) return qnan(i);
  if (f == KIT_B64) { uint64_t b; memcpy(&b, &v, 8); return b; }
  uint64_t s = signbit(v) ? sign(i) : 0;
  double a = fabs(v);
  if (isinf(a)) { if (!i->has_inf) unrepresentable(f, v); return s | eall(i) << i->mbits; }
  if (a == 0) return s;
  int e;
  frexp(a, &e);
  e--;   /* a is in [2^e, 2^(e+1)) */
  if (e > emax(i)) unrepresentable(f, v);
  if (e < 1 - i->bias) {   /* subnormal: a multiple of the smallest one */
    double m = ldexp(a, i->bias - 1 + i->mbits);
    if (m != floor(m)) unrepresentable(f, v);
    return s | (uint64_t)m;
  }
  double m = ldexp(a, i->mbits - e);   /* in [2^mbits, 2^(mbits+1)) */
  if (m != floor(m)) unrepresentable(f, v);
  uint64_t b = s | (uint64_t)(e + i->bias) << i->mbits | ((uint64_t)m & mmask(i));
  if (kit_isnan(f, b)) unrepresentable(f, v);   /* E4M3's 480 */
  return b;
}

double kit_max(kit_fmt f)
{
  const kit_fmtinfo *i = &INFO[f];
  return ldexp((double)(1ULL << i->mbits | (mmask(i) - (i->has_nan && !i->has_inf))), emax(i) - i->mbits);
}

uint64_t kit_perturb(kit_fmt f, uint64_t b)
{
  const kit_fmtinfo *i = &INFO[f];
  if (b == KIT_NONE || kit_isnan(f, b)) return 0;   /* NaN, or none: a number instead (no neighbour) */
  if (i->has_inf && efield(i, b) == eall(i)) return b - 1;   /* infinity: the largest finite value */
  uint64_t p = b ^ 1;
  return kit_isnan(f, p) ? b ^ 2 : p;   /* E4M3: 448 would become NaN; 384 instead */
}

int kit_same(kit_fmt f, uint64_t a, uint64_t b)
{ return a == b || (!kit_nan_bits && kit_isnan(f, a) && kit_isnan(f, b)); }

/* y rounded to the format's precision in MPFR's default exponent range,
   with ternary value t, to an encoding (kit.h): MPFR's recipe for a
   narrower exponent range (check_range, then subnormalize, which uses t so
   that nothing is rounded twice) */
uint64_t kit_round_t(kit_fmt f, mpfr_ptr y, int t, mpfr_rnd_t rnd)
{
  const kit_fmtinfo *i = &INFO[f];
  if (mpfr_nan_p(y)) return qnan(i);
  mpfr_exp_t lo = mpfr_get_emin(), hi = mpfr_get_emax();
  /* MPFR's exponents: 2^k is 0.1 * 2^(k+1) */
  mpfr_set_emin(2 - i->bias - i->mbits);
  mpfr_set_emax(emax(i) + 1);
  t = mpfr_check_range(y, t, rnd);
  mpfr_subnormalize(y, t, rnd);
  mpfr_set_emin(lo);
  mpfr_set_emax(hi);
  double d = mpfr_get_d(y, MPFR_RNDN);   /* exact: every format embeds in binary64 */
  if (!i->has_inf) {
    /* E4M3: MPFR's range reaches 480 (the NaN's encoding), and its overflow
       gives 480 toward zero and infinity away. An infinity here is exact
       (an infinite input, or a pole) or an overflow away from zero; above
       448, an overflow in either direction. The MX element formats have
       no NaN to give, and saturate. */
    double max = kit_max(f);
    if (isinf(d) || fabs(d) > max) {
      int towardzero = !isinf(d) && (rnd == MPFR_RNDZ || (rnd == MPFR_RNDD && d > 0) || (rnd == MPFR_RNDU && d < 0));
      d = towardzero || kit_e4m3_saturate || !i->has_nan ? copysign(max, d) : NAN;
    }
  }
  return kit_encode(f, d);
}

uint64_t kit_round(kit_fmt f, mpfr_srcptr x, mpfr_rnd_t rnd)
{
  mpfr_t y;
  mpfr_init2(y, INFO[f].mbits + 1);
  int t = mpfr_set(y, x, rnd);
  uint64_t r = kit_round_t(f, y, t, rnd);
  mpfr_clear(y);
  return r;
}

/* one argument: MPFR gives NaN for every NaN, so no signaling NaN rule */
uint64_t kit_ref1(kit_fmt f, kit_mpfr1 fn, uint64_t x, mpfr_rnd_t rnd)
{
  const kit_fmtinfo *i = &INFO[f];
  mpfr_t a, y;
  mpfr_init2(a, i->mbits + 1);
  mpfr_init2(y, i->mbits + 1);
  mpfr_set_d(a, kit_decode(f, x), MPFR_RNDN);   /* exact */
  int t = fn(y, a, rnd);
  uint64_t r = kit_round_t(f, y, t, rnd);
  mpfr_clear(a);
  mpfr_clear(y);
  return r;
}

uint64_t kit_ref2(kit_fmt f, kit_mpfr2 fn, uint64_t x, uint64_t y, mpfr_rnd_t rnd)
{
  const kit_fmtinfo *i = &INFO[f];
  if (is_snan(i, f, x) || is_snan(i, f, y)) return qnan(i);
  mpfr_t a, b, z;
  mpfr_init2(a, i->mbits + 1);
  mpfr_init2(b, i->mbits + 1);
  mpfr_init2(z, i->mbits + 1);
  mpfr_set_d(a, kit_decode(f, x), MPFR_RNDN);
  mpfr_set_d(b, kit_decode(f, y), MPFR_RNDN);
  int t = fn(z, a, b, rnd);
  uint64_t r = kit_round_t(f, z, t, rnd);
  mpfr_clear(a);
  mpfr_clear(b);
  mpfr_clear(z);
  return r;
}
