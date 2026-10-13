/* crdist-ref.c: MPFR references for crdist's functions, as kit_mpfr1
   functions (the correctly rounded value at y's precision, with the
   ternary value), for kit_ref1. Round to nearest only: crdist rounds only
   that way.

   MPFR has no inverse of erf, so each reference finds its answer by
   comparisons, not by computing the inverse. Every value here is the
   positive root r of an equation G(r s) = v, with G = erf (increasing) or
   erfc (decreasing), s = 1 or 1/sqrt 2 (probit), and v exact; a negative
   answer is the negation of one. Whether r lies above or below a number m
   is whether G(m s) lies below or above v (the other way for erfc), and
   cmp() decides that with directed roundings: an enclosure of G(m s) at q
   bits, from 64 up, doubled until it lies wholly on one side of v.
   (G(m s) = v exactly at an m other than 0 would make cmp() give up at
   8192 bits and abort; crdist-check would report it.)

   The answer at y's precision p: from a start c (rounded from a binary64
   estimate made here with the C library, independent of crdist-fast.h),
   find the side of c that r lies on, and compare r with the midpoint
   between c and its neighbour on that side: r short of the midpoint makes
   c = RN(r), with the ternary value the side c lies on; otherwise step to
   the neighbour and repeat. c and the midpoints are exact in MPFR, so
   every step is a comparison of r with an exact number. The start only decides how many steps there are;
   crdist-check also runs a bisection that has no start (section 3). */
#include <stdio.h>   /* before mpfr.h, for mpfr_fprintf */
#include <stdlib.h>
#include <math.h>
#include <mpfr.h>
#include "crdist-ref.h"

unsigned long crdist_ref_long;

enum { G_ERF, G_ERFC };
typedef crdist_eqn eqn;

/* the sign of r - m, for m >= 0 */
int crdist_cmp(const eqn *e, mpfr_srcptr m)
{
  if (mpfr_zero_p(m)) return 1;   /* r > 0 */
  for (mpfr_prec_t q = 64; q <= 8192; q *= 2) {
    mpfr_t lo, hi, a, b, rt;
    mpfr_inits2(q, lo, hi, a, b, rt, (mpfr_ptr)0);
    int exact = 0;
    if (!e->scaled) {
      int t = e->g == G_ERF ? mpfr_erf(lo, m, MPFR_RNDD) : mpfr_erfc(lo, m, MPFR_RNDD);
      mpfr_set(hi, lo, MPFR_RNDN);
      if (t) mpfr_nextabove(hi);   /* RNDD: lo <= G(m) < the next number */
      else exact = 1;
    } else {
      /* a <= m/sqrt 2 <= b, and G monotone */
      mpfr_sqrt_ui(rt, 2, MPFR_RNDU); mpfr_div(a, m, rt, MPFR_RNDD);
      mpfr_sqrt_ui(rt, 2, MPFR_RNDD); mpfr_div(b, m, rt, MPFR_RNDU);
      if (e->g == G_ERF) { mpfr_erf(lo, a, MPFR_RNDD); mpfr_erf(hi, b, MPFR_RNDU); }
      else { mpfr_erfc(lo, b, MPFR_RNDD); mpfr_erfc(hi, a, MPFR_RNDU); }
    }
    int below = mpfr_less_p(hi, e->v), above = mpfr_greater_p(lo, e->v);   /* G(m s) vs v */
    int eq = exact && mpfr_equal_p(lo, e->v);
    mpfr_clears(lo, hi, a, b, rt, (mpfr_ptr)0);
    if (q > 512 && (below || above)) __atomic_fetch_add(&crdist_ref_long, 1, __ATOMIC_RELAXED);
    if (eq) return 0;
    /* erf rises: G(m) below v puts r above m; erfc falls: the other way */
    if (below) return e->g == G_ERF ? 1 : -1;
    if (above) return e->g == G_ERF ? -1 : 1;
  }
  mpfr_fprintf(stderr, "crdist-ref: no decision at 8192 bits for m = %Ra, v = %Ra\n", m, e->v);
  abort();
}

/* RN(r) at y's precision from the start c0, and the ternary value: the
   side of r that c lies on first, then the midpoint on that side; two
   comparisons when c is already RN(r) */
static int walk(mpfr_ptr y, const eqn *e, double c0)
{
  mpfr_prec_t p = mpfr_get_prec(y);
  mpfr_t c, n, mid;
  mpfr_init2(c, p); mpfr_init2(n, p); mpfr_init2(mid, p + 1);
  mpfr_set_d(c, c0 > 0 ? c0 : 0x1p-160, MPFR_RNDN);
  int t;
  for (int steps = 0;; steps++) {
    if (steps > 4096) { mpfr_fprintf(stderr, "crdist-ref: no answer after 4096 steps from %a (v = %Ra)\n", c0, e->v); abort(); }
    t = crdist_cmp(e, c);   /* the sign of r - c */
    if (t == 0) { mpfr_fprintf(stderr, "crdist-ref: exact root %Ra (v = %Ra)\n", c, e->v); abort(); }
    mpfr_set(n, c, MPFR_RNDN);
    if (t > 0) mpfr_nextabove(n); else mpfr_nextbelow(n);
    mpfr_add(mid, c, n, MPFR_RNDN); mpfr_div_2ui(mid, mid, 1, MPFR_RNDN);   /* exact at p + 1 bits */
    int s = crdist_cmp(e, mid);
    if (s == 0) { mpfr_fprintf(stderr, "crdist-ref: root on a midpoint (v = %Ra)\n", e->v); abort(); }
    if (s != t) break;   /* r between c and the midpoint */
    mpfr_set(c, n, MPFR_RNDN);
  }
  mpfr_set(y, c, MPFR_RNDN);
  mpfr_clears(c, n, mid, (mpfr_ptr)0);
  return -t;
}

/* binary64 estimates of the positive roots, with the C library: Winitzki's
   start and Newton's method */
static double est_ierf(double x)
{
  if (x < 0x1p-20) return 0.886226925452758 * x;
  double L = log1p(-x * x), t = 4.330746750799873 + 0.5 * L, y = sqrt(sqrt(t * t - L * 6.802721088435375) - t);
  for (int i = 0; i < 4; i++) y -= (erf(y) - x) / (1.1283791670955126 * exp(-y * y));
  return y;
}
static double est_ierfc(double q)
{
  double L = log(q * (2 - q)), t = 4.330746750799873 + 0.5 * L, y = sqrt(sqrt(t * t - L * 6.802721088435375) - t);
  for (int i = 0; i < 6; i++) y += (erfc(y) - q) / (1.1283791670955126 * exp(-y * y));
  return y;
}

/* y = sign * r, with v = the exact value of the expression that set it */
static int answer(mpfr_ptr y, int g, int scaled, mpfr_srcptr v, int sign)
{
  eqn e = {g, scaled, v};
  double d = mpfr_get_d(v, MPFR_RNDN);
  double c0 = g == G_ERF ? est_ierf(d) : est_ierfc(d);
  if (scaled) c0 *= M_SQRT2;
  int t = walk(y, &e, c0);
  if (sign < 0) { mpfr_neg(y, y, MPFR_RNDN); t = -t; }
  return t;
}

/* v = a + b x exactly (b a power of 2 or -1), or abort; v has 256 bits */
static void exact(mpfr_ptr v, long a, long b, mpfr_srcptr x)
{
  if (mpfr_mul_si(v, x, b, MPFR_RNDN) || mpfr_add_si(v, v, a, MPFR_RNDN)) {
    fprintf(stderr, "crdist-ref: inexact argument\n");
    abort();
  }
}

static int special(mpfr_ptr y, double inf_sign)
{
  if (inf_sign != inf_sign) mpfr_set_nan(y);
  else if (inf_sign == 0) mpfr_set_zero(y, 1);
  else mpfr_set_inf(y, inf_sign > 0 ? 1 : -1);
  return 0;
}

static void need_rndn(mpfr_rnd_t rnd)
{
  if (rnd != MPFR_RNDN) { fprintf(stderr, "crdist-ref: round to nearest only\n"); abort(); }
}

int crdist_mpfr_erfinv(mpfr_ptr y, mpfr_srcptr x, mpfr_rnd_t rnd)
{
  need_rndn(rnd);
  if (mpfr_nan_p(x)) return special(y, NAN);
  if (mpfr_zero_p(x)) { mpfr_set(y, x, MPFR_RNDN); return 0; }
  int c = mpfr_cmpabs_ui(x, 1), sign = mpfr_sgn(x);
  if (c > 0) return special(y, NAN);
  if (c == 0) return special(y, sign);
  mpfr_t a, v; mpfr_init2(a, mpfr_get_prec(x)); mpfr_init2(v, 256); mpfr_abs(a, x, MPFR_RNDN);
  int t;
  if (mpfr_cmp_d(a, 0.5) <= 0) { exact(v, 0, 1, a); t = answer(y, G_ERF, 0, v, sign); }   /* erf(r) = |x| */
  else { exact(v, 1, -1, a); t = answer(y, G_ERFC, 0, v, sign); }                          /* erfc(r) = 1 - |x| */
  mpfr_clears(a, v, (mpfr_ptr)0);
  return t;
}

int crdist_mpfr_erfcinv(mpfr_ptr y, mpfr_srcptr x, mpfr_rnd_t rnd)
{
  need_rndn(rnd);
  if (mpfr_nan_p(x) || mpfr_sgn(x) < 0 || mpfr_cmp_ui(x, 2) > 0) return special(y, NAN);
  if (mpfr_zero_p(x)) return special(y, 1);
  if (mpfr_cmp_ui(x, 2) == 0) return special(y, -1);
  if (mpfr_cmp_ui(x, 1) == 0) return special(y, 0);
  mpfr_t v; mpfr_init2(v, 256); int t;
  if (mpfr_cmp_d(x, 0.5) <= 0) { exact(v, 0, 1, x); t = answer(y, G_ERFC, 0, v, 1); }         /* erfc(r) = x */
  else if (mpfr_cmp_ui(x, 1) < 0) { exact(v, 1, -1, x); t = answer(y, G_ERF, 0, v, 1); }      /* erf(r) = 1 - x */
  else if (mpfr_cmp_d(x, 1.5) <= 0) { exact(v, -1, 1, x); t = answer(y, G_ERF, 0, v, -1); }   /* erf(-r) = x - 1 */
  else { exact(v, 2, -1, x); t = answer(y, G_ERFC, 0, v, -1); }                                /* erfc(-r) = 2 - x */
  mpfr_clear(v);
  return t;
}

/* probit(p) = r with Phi(r) = erfc(-r/sqrt 2)/2 = p */
int crdist_mpfr_probit(mpfr_ptr y, mpfr_srcptr x, mpfr_rnd_t rnd)
{
  need_rndn(rnd);
  if (mpfr_nan_p(x) || mpfr_sgn(x) < 0 || mpfr_cmp_ui(x, 1) > 0) return special(y, NAN);
  if (mpfr_zero_p(x)) return special(y, -1);
  if (mpfr_cmp_ui(x, 1) == 0) return special(y, 1);
  if (mpfr_cmp_d(x, 0.5) == 0) return special(y, 0);
  mpfr_t v; mpfr_init2(v, 256); int t;
  if (mpfr_cmp_d(x, 0.25) <= 0) { exact(v, 0, 2, x); t = answer(y, G_ERFC, 1, v, -1); }       /* erfc(-r/sqrt 2) = 2p */
  else if (mpfr_cmp_d(x, 0.5) < 0) { exact(v, 1, -2, x); t = answer(y, G_ERF, 1, v, -1); }    /* erf(-r/sqrt 2) = 1 - 2p */
  else if (mpfr_cmp_d(x, 0.75) <= 0) { exact(v, -1, 2, x); t = answer(y, G_ERF, 1, v, 1); }   /* erf(r/sqrt 2) = 2p - 1 */
  else { exact(v, 2, -2, x); t = answer(y, G_ERFC, 1, v, 1); }                                 /* erfc(r/sqrt 2) = 2 - 2p */
  mpfr_clear(v);
  return t;
}

/* the equation behind f at the binary32 input x (inside the domain, not a
   special point), for the bisection and the bound check in crdist-check:
   *e is set, v (256 bits, the caller's) holds its right side, and the
   return is the sign of the answer */
int crdist_eqn_at(crdist_eqn *e, mpfr_ptr v, int f, float x)
{
  mpfr_t a; mpfr_init2(a, 24); mpfr_set_d(a, x, MPFR_RNDN);
  int sign = 1, g;
  if (f == 0) {
    sign = x < 0 ? -1 : 1; mpfr_abs(a, a, MPFR_RNDN);
    if (fabsf(x) <= 0.5f) { exact(v, 0, 1, a); g = G_ERF; } else { exact(v, 1, -1, a); g = G_ERFC; }
  } else if (f == 1) {
    if (x <= 0.5f) { exact(v, 0, 1, a); g = G_ERFC; }
    else if (x < 1) { exact(v, 1, -1, a); g = G_ERF; }
    else if (x <= 1.5f) { exact(v, -1, 1, a); g = G_ERF; sign = -1; }
    else { exact(v, 2, -1, a); g = G_ERFC; sign = -1; }
  } else {
    if (x <= 0.25f) { exact(v, 0, 2, a); g = G_ERFC; sign = -1; }
    else if (x < 0.5f) { exact(v, 1, -2, a); g = G_ERF; sign = -1; }
    else if (x <= 0.75f) { exact(v, -1, 2, a); g = G_ERF; }
    else { exact(v, 2, -2, a); g = G_ERFC; }
  }
  mpfr_clear(a);
  e->g = g; e->scaled = f == 2; e->v = v;
  return sign;
}
