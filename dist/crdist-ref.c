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

/* The normal distribution's functions of x (2026-10-13): Ziv loops, as in
   crnn-ref.c. The function in MPFR at precision p with RNDN, a bound on the
   error in ulps (errbits), and mpfr_can_round with MPFR_RNDZ at one more
   bit, which also settles the ternary value; the value is never exactly
   representable (each is transcendental at a nonzero rational argument,
   and the rational answers, Phi(0) = 1/2 and logit(1/2) = 0, are given
   first). Where the result provably rounds to a value no precision would
   approach, it is given directly, with the bound beside it. */
typedef void (*approx)(mpfr_ptr t, mpfr_srcptr x);   /* at t's precision, RNDN */
static int ziv(mpfr_ptr y, mpfr_srcptr x, approx f, int errbits)
{
  mpfr_prec_t py = mpfr_get_prec(y);
  for (mpfr_prec_t p = 2 * py + 64; p <= 8192; p *= 2) {
    mpfr_t t; mpfr_init2(t, p);
    f(t, x);
    if (mpfr_regular_p(t) && mpfr_can_round(t, p - errbits, MPFR_RNDN, MPFR_RNDZ, py + 1)) {
      int r = mpfr_set(y, t, MPFR_RNDN);
      mpfr_clear(t);
      if (p > 512) __atomic_fetch_add(&crdist_ref_long, 1, __ATOMIC_RELAXED);
      return r;
    }
    mpfr_clear(t);
  }
  mpfr_fprintf(stderr, "crdist-ref: no rounding decided at 8192 bits for %Ra\n", x);
  abort();
}

/* Phi(x) = erfc(-x/sqrt 2)/2: x/sqrt 2 carries 2 ulps, which erfc's
   condition number, below 2 t^2 + 1 < 2^8 for |t| < 10.7, magnifies;
   erfc and the halving (exact) add 1: under 2^10 ulps */
static void a_ndtr(mpfr_ptr t, mpfr_srcptr x)
{
  mpfr_t s; mpfr_init2(s, mpfr_get_prec(t));
  mpfr_sqrt_ui(s, 2, MPFR_RNDN); mpfr_div(t, x, s, MPFR_RNDN); mpfr_neg(t, t, MPFR_RNDN);
  mpfr_erfc(t, t, MPFR_RNDN); mpfr_div_2ui(t, t, 1, MPFR_RNDN);
  mpfr_clear(s);
}
/* log Phi(x): for x >= 0, log1p(-Phi(-x)); otherwise log(Phi(x)). Phi
   carries under 2^10 ulps as above, at |t| up to 23170 under 2^31 (the
   condition number 2 t^2 + 1); log divides that relative error by
   |log Phi| >= log 2 for x < 0, and by more than t^2 - 1 once t > 1, so
   the result's error stays under 2^12 ulps; log1p's condition number is
   below 1.45 for x >= 0 */
static void a_log_ndtr(mpfr_ptr t, mpfr_srcptr x)
{
  mpfr_t s; mpfr_init2(s, mpfr_get_prec(t));
  mpfr_sqrt_ui(s, 2, MPFR_RNDN); mpfr_div(t, x, s, MPFR_RNDN);
  if (mpfr_sgn(x) >= 0) { mpfr_erfc(t, t, MPFR_RNDN); mpfr_div_2ui(t, t, 1, MPFR_RNDN); mpfr_neg(t, t, MPFR_RNDN); mpfr_log1p(t, t, MPFR_RNDN); }
  else { mpfr_neg(t, t, MPFR_RNDN); mpfr_erfc(t, t, MPFR_RNDN); mpfr_div_2ui(t, t, 1, MPFR_RNDN); mpfr_log(t, t, MPFR_RNDN); }
  mpfr_clear(s);
}
/* below -32768 (where erfc's value nears MPFR's exponent range): log Phi(x)
   = -x^2/2 - log(-x) - log sqrt(2 pi) + log R, R = sum_k (-1)^k (2k-1)!!
   / x^(2k), the series enveloping (each partial sum's error is below the
   first term left out). Terms are summed until one is below 2^-(p+16);
   with 1/x^2 <= 2^-30 each is at most (2k+1) 2^-30 times the last, so
   that takes at most p/29 + 2 of them. Four negative terms, each within a
   few ulps: under 2^4 ulps */
static void a_log_ndtr_far(mpfr_ptr t, mpfr_srcptr x)
{
  mpfr_prec_t p = mpfr_get_prec(t);
  mpfr_t r, term, sum, a; mpfr_inits2(p + 32, r, term, sum, a, (mpfr_ptr)0);
  mpfr_sqr(r, x, MPFR_RNDN); mpfr_ui_div(r, 1, r, MPFR_RNDN);                  /* 1/x^2 */
  mpfr_set_ui(term, 1, MPFR_RNDN); mpfr_set_ui(sum, 0, MPFR_RNDN);
  for (unsigned long k = 1; ; k++) {
    mpfr_mul(term, term, r, MPFR_RNDN); mpfr_mul_ui(term, term, 2 * k - 1, MPFR_RNDN); mpfr_neg(term, term, MPFR_RNDN);
    if (mpfr_get_exp(term) < -(mpfr_exp_t)(p + 16)) break;
    mpfr_add(sum, sum, term, MPFR_RNDN);
  }
  mpfr_log1p(sum, sum, MPFR_RNDN);                                              /* log R */
  mpfr_neg(a, x, MPFR_RNDN); mpfr_log(a, a, MPFR_RNDN); mpfr_sub(sum, sum, a, MPFR_RNDN);
  mpfr_const_pi(a, MPFR_RNDN); mpfr_mul_2ui(a, a, 1, MPFR_RNDN); mpfr_log(a, a, MPFR_RNDN); mpfr_div_2ui(a, a, 1, MPFR_RNDN);
  mpfr_sub(sum, sum, a, MPFR_RNDN);
  mpfr_sqr(a, x, MPFR_RNDN); mpfr_div_2ui(a, a, 1, MPFR_RNDN);                  /* exact */
  mpfr_sub(t, sum, a, MPFR_RNDN);
  mpfr_clears(r, term, sum, a, (mpfr_ptr)0);
}
/* e^(-x^2/2)/sqrt(2 pi): x^2/2 exact at p >= 48, exp, the root and the
   division: under 2^3 ulps */
static void a_normpdf(mpfr_ptr t, mpfr_srcptr x)
{
  mpfr_t s; mpfr_init2(s, mpfr_get_prec(t));
  mpfr_sqr(t, x, MPFR_RNDN); mpfr_div_2ui(t, t, 1, MPFR_RNDN); mpfr_neg(t, t, MPFR_RNDN); mpfr_exp(t, t, MPFR_RNDN);
  mpfr_const_pi(s, MPFR_RNDN); mpfr_mul_2ui(s, s, 1, MPFR_RNDN); mpfr_sqrt(s, s, MPFR_RNDN); mpfr_div(t, t, s, MPFR_RNDN);
  mpfr_clear(s);
}
/* logit(p): log(p) - log1p(-p) below 1/4 (|log1p(-p)| < 0.29 < |log p| / 4:
   no cancellation), log1p((2p - 1)/(1 - p)) up to 3/4 (log1p's condition
   number below 1), -logit(1 - p) above (1 - p exact): under 2^4 ulps */
static void a_logit(mpfr_ptr t, mpfr_srcptr x)
{
  mpfr_t a, b; mpfr_inits2(mpfr_get_prec(t), a, b, (mpfr_ptr)0);
  int neg = mpfr_cmp_d(x, 0.75) > 0;
  mpfr_set(a, x, MPFR_RNDN);
  if (neg) mpfr_ui_sub(a, 1, a, MPFR_RNDN);
  if (mpfr_cmp_d(a, 0.25) < 0) { mpfr_log(b, a, MPFR_RNDN); mpfr_neg(a, a, MPFR_RNDN); mpfr_log1p(a, a, MPFR_RNDN); mpfr_sub(t, b, a, MPFR_RNDN); }
  else { mpfr_mul_2ui(b, a, 1, MPFR_RNDN); mpfr_sub_ui(b, b, 1, MPFR_RNDN); mpfr_ui_sub(a, 1, a, MPFR_RNDN); mpfr_div(t, b, a, MPFR_RNDN); mpfr_log1p(t, t, MPFR_RNDN); }
  if (neg) mpfr_neg(t, t, MPFR_RNDN);
  mpfr_clears(a, b, (mpfr_ptr)0);
}

/* the direct answers; each ternary is the sign of (answer - exact) */
static int direct(mpfr_ptr y, double v, int ternary) { mpfr_set_d(y, v, MPFR_RNDN); return ternary; }

int crdist_mpfr_ndtr(mpfr_ptr y, mpfr_srcptr x, mpfr_rnd_t rnd)
{
  need_rndn(rnd);
  if (mpfr_nan_p(x)) return special(y, NAN);
  if (mpfr_zero_p(x)) return direct(y, 0.5, 0);
  if (mpfr_cmp_si(x, 6) >= 0) return direct(y, 1, 1);      /* 1 - Phi(6) < 1e-9 < 2^-25: 1, above the exact value */
  if (mpfr_cmp_si(x, -15) <= 0) return direct(y, 0, -1);   /* Phi(-15) < 4e-51 < 2^-150: +0, below it */
  return ziv(y, x, a_ndtr, 10);
}
int crdist_mpfr_log_ndtr(mpfr_ptr y, mpfr_srcptr x, mpfr_rnd_t rnd)
{
  need_rndn(rnd);
  if (mpfr_nan_p(x)) return special(y, NAN);
  if (mpfr_inf_p(x)) return mpfr_sgn(x) > 0 ? direct(y, 0, 0) : special(y, -1);
  if (mpfr_cmp_si(x, 15) >= 0) return direct(y, -0.0, 1);  /* log Phi(15) > -4e-51 > -2^-150: -0, above the exact value */
  if (mpfr_cmp_si(x, -32768) < 0) return ziv(y, x, a_log_ndtr_far, 4);
  return ziv(y, x, a_log_ndtr, 12);
}
int crdist_mpfr_normpdf(mpfr_ptr y, mpfr_srcptr x, mpfr_rnd_t rnd)
{
  need_rndn(rnd);
  if (mpfr_nan_p(x)) return special(y, NAN);
  if (mpfr_inf_p(x) || mpfr_cmpabs_ui(x, 15) >= 0) return direct(y, 0, -1);   /* phi(15) < 6e-50 < 2^-150: +0 */
  return ziv(y, x, a_normpdf, 3);
}
int crdist_mpfr_logit(mpfr_ptr y, mpfr_srcptr x, mpfr_rnd_t rnd)
{
  need_rndn(rnd);
  if (mpfr_nan_p(x) || mpfr_sgn(x) < 0 || mpfr_cmp_ui(x, 1) > 0) return special(y, NAN);
  if (mpfr_zero_p(x)) return special(y, -1);
  if (mpfr_cmp_ui(x, 1) == 0) return special(y, 1);
  if (mpfr_cmp_d(x, 0.5) == 0) return special(y, 0);
  return ziv(y, x, a_logit, 4);
}

/* f's value at x (f one of the functions of x above, x not a special point)
   at t's precision, within 2^errbits ulps (errbits as in the references),
   for crdist-check's bound */
void crdist_mpfr_value(int f, mpfr_ptr t, mpfr_srcptr x)
{
  switch (f) {
    case 3: a_ndtr(t, x); break;
    case 4: if (mpfr_cmp_si(x, -32768) < 0) a_log_ndtr_far(t, x); else a_log_ndtr(t, x); break;
    case 5: a_normpdf(t, x); break;
    default: a_logit(t, x);
  }
}
