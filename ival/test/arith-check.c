/* arith-check.c: ival's interval arithmetic (ival-arith.c) against MPFR, the tightest enclosure for every result.

   The reference finds each result its own way, not by ival's case analysis:
     - add, sub: MPFR rounding the two ends' sums down and up;
     - mul, sqr: the exact products of the ends (106 bits; 0 * inf = 0 at interval ends), the least and greatest
       taken exactly, then rounded down and up (sqr adds 0 when the interval holds it);
     - div, recip: every quotient of an end of A by a nonzero end of B, rounded down and up; the limits where B
       reaches 0 from either side (an A end of either sign then gives an infinity); 0 when A holds it; B = [0, 0] is
       empty and A = [0, 0] gives [0, 0]. Not 1788's table: the hull of the candidates, which holds the extremes
       because a / b is linear in a and monotone in b on each side of 0;
     - fma: the least exact corner product plus clo, exactly (4400 bits), rounded down; the greatest plus chi rounded
       up. Each pair of intervals below with a third, special or random; 2 points inside each.
   The intervals: every pair of 23 special ends (0, the infinities, DBL_MAX, the subnormals, 2^-969 and 2^-960 where
   the error-free transformations stop being exact), random intervals of every magnitude and width, points, and
   empty ones ([NaN, NaN], lo > hi, [inf, inf], [-inf, -inf]).
   Also, with no reference: 4 points inside each pair of finite intervals, the exact result (MPFR at 2200 bits) must
   lie in ival's interval. The negative control: the ends rounded to nearest, with no error term, must differ from
   the reference somewhere.
   Then the paths, compared bit for bit with the results above (which, on a CPU with AVX2 and FMA, come from the
   four-lane passes): the portable passes; the scalar code alone, by error-free transformations rounding to nearest,
   a second algorithm; the same calls with flush-to-zero and denormals-are-zero set, as a program built with
   -ffast-math has them (and they must still be set afterwards); and each operation in place, the result written over
   its first operand. Zero ends must be +0. */
#include <float.h>
#include <math.h>
#include <mpfr.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ival.h"
#if defined(__x86_64__)
#include <immintrin.h>
#endif
/* the flush modes, as a -ffast-math program sets them: x86-64 MXCSR FZ and DAZ, aarch64 FPCR.FZ */
#if defined(__x86_64__)
#define FLUSH_BITS 0x8040ul
static unsigned long fpctl(void) { return _mm_getcsr(); }
static void set_fpctl(unsigned long r) { _mm_setcsr((unsigned)r); }
#elif defined(__aarch64__)
#define FLUSH_BITS (1ul << 24)
static unsigned long fpctl(void) { unsigned long r; __asm__ volatile("mrs %0, fpcr" : "=r"(r)); return r; }
static void set_fpctl(unsigned long r) { __asm__ volatile("msr fpcr, %0" : : "r"(r)); }
#else
#define FLUSH_BITS 0ul
static unsigned long fpctl(void) { return 0; }
static void set_fpctl(unsigned long r) { (void)r; }
#endif
extern int ival__arith_path;   /* ival-arith.c: 0 the best code, 1 the portable passes, 2 the scalar code alone */

static uint64_t rs = 0x9e3779b97f4a7c15ULL;
static uint64_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }
static double anyd(void) { for (;;) { uint64_t u = rnd(); double d; memcpy(&d, &u, 8); if (isfinite(d)) return d; } }
static int empty(double lo, double hi) { return !(lo <= hi) || lo == INFINITY || hi == -INFINITY; }

/* ---- the reference ---- */
static mpfr_t T, U;
static double rdn(mpfr_t x) { return mpfr_get_d(x, MPFR_RNDD); }
static double rup(mpfr_t x) { return mpfr_get_d(x, MPFR_RNDU); }

static void r_add(double al, double ah, double bl, double bh, double *zl, double *zh, int sub)
{
  if (empty(al, ah) || empty(bl, bh)) { *zl = *zh = NAN; return; }
  double x0 = al, y0 = sub ? -bh : bl, x1 = ah, y1 = sub ? -bl : bh;
  mpfr_set_d(T, x0, MPFR_RNDN); mpfr_set_d(U, y0, MPFR_RNDN); mpfr_add(T, T, U, MPFR_RNDD); *zl = mpfr_get_d(T, MPFR_RNDD);
  mpfr_set_d(T, x1, MPFR_RNDN); mpfr_set_d(U, y1, MPFR_RNDN); mpfr_add(T, T, U, MPFR_RNDU); *zh = mpfr_get_d(T, MPFR_RNDU);
}

/* exact product of two ends, 0 * inf = 0, into x (106 bits) */
static void endprod(mpfr_t x, double a, double b)
{
  if (a == 0 || b == 0) { mpfr_set_zero(x, 1); return; }
  mpfr_set_d(x, a, MPFR_RNDN); mpfr_mul_d(x, x, b, MPFR_RNDN);   /* exact at 106 bits; inf * finite = inf */
}
static void r_mul(double al, double ah, double bl, double bh, double *zl, double *zh)
{
  if (empty(al, ah) || empty(bl, bh)) { *zl = *zh = NAN; return; }
  mpfr_t p[4], lo, hi;
  for (int k = 0; k < 4; k++) mpfr_init2(p[k], 106);
  mpfr_init2(lo, 106); mpfr_init2(hi, 106);
  endprod(p[0], al, bl); endprod(p[1], al, bh); endprod(p[2], ah, bl); endprod(p[3], ah, bh);
  mpfr_set(lo, p[0], MPFR_RNDN); mpfr_set(hi, p[0], MPFR_RNDN);
  for (int k = 1; k < 4; k++) { if (mpfr_less_p(p[k], lo)) mpfr_set(lo, p[k], MPFR_RNDN); if (mpfr_greater_p(p[k], hi)) mpfr_set(hi, p[k], MPFR_RNDN); }
  *zl = rdn(lo); *zh = rup(hi);
  for (int k = 0; k < 4; k++) mpfr_clear(p[k]);
  mpfr_clear(lo); mpfr_clear(hi);
}
/* fma: the least product plus clo, exactly, rounded down; the greatest plus chi rounded up. inf{a * b + c} is
   inf{a * b} + inf{c} because a * b and c vary independently over the box. 4400 bits hold any such sum exactly. */
static mpfr_t W;
static void r_fma(double al, double ah, double bl, double bh, double cl, double ch, double *zl, double *zh)
{
  if (empty(al, ah) || empty(bl, bh) || empty(cl, ch)) { *zl = *zh = NAN; return; }
  mpfr_t p[4], lo, hi;
  for (int k = 0; k < 4; k++) mpfr_init2(p[k], 106);
  mpfr_init2(lo, 106); mpfr_init2(hi, 106);
  endprod(p[0], al, bl); endprod(p[1], al, bh); endprod(p[2], ah, bl); endprod(p[3], ah, bh);
  mpfr_set(lo, p[0], MPFR_RNDN); mpfr_set(hi, p[0], MPFR_RNDN);
  for (int k = 1; k < 4; k++) { if (mpfr_less_p(p[k], lo)) mpfr_set(lo, p[k], MPFR_RNDN); if (mpfr_greater_p(p[k], hi)) mpfr_set(hi, p[k], MPFR_RNDN); }
  mpfr_add_d(W, lo, cl, MPFR_RNDN); *zl = rdn(W);   /* exact at 4400 bits */
  mpfr_add_d(W, hi, ch, MPFR_RNDN); *zh = rup(W);
  for (int k = 0; k < 4; k++) mpfr_clear(p[k]);
  mpfr_clear(lo); mpfr_clear(hi);
}
static void r_sqr(double al, double ah, double *zl, double *zh)
{
  if (empty(al, ah)) { *zl = *zh = NAN; return; }
  mpfr_t p, q, lo, hi;
  mpfr_inits2(106, p, q, lo, hi, (mpfr_ptr)0);
  endprod(p, al, al); endprod(q, ah, ah);
  if (mpfr_less_p(p, q)) { mpfr_set(lo, p, MPFR_RNDN); mpfr_set(hi, q, MPFR_RNDN); } else { mpfr_set(lo, q, MPFR_RNDN); mpfr_set(hi, p, MPFR_RNDN); }
  if (al <= 0 && 0 <= ah) mpfr_set_zero(lo, 1);
  *zl = rdn(lo); *zh = rup(hi);
  mpfr_clears(p, q, lo, hi, (mpfr_ptr)0);
}
/* one candidate quotient of ends, rounded down into *d and up into *u; returns 0 if undefined (inf / inf) */
static int endquot(double a, double b, double *d, double *u)
{
  if (a == 0) { *d = *u = 0; return 1; }
  if (isinf(a) && isinf(b)) return 0;
  if (isinf(b)) { *d = *u = 0; return 1; }
  if (isinf(a)) { *d = *u = (a > 0) == (b > 0) ? INFINITY : -INFINITY; return 1; }
  mpfr_set_d(T, a, MPFR_RNDN); mpfr_div_d(U, T, b, MPFR_RNDD); *d = mpfr_get_d(U, MPFR_RNDD);
  mpfr_div_d(U, T, b, MPFR_RNDU); *u = mpfr_get_d(U, MPFR_RNDU);
  return 1;
}
static void r_div(double al, double ah, double bl, double bh, double *zl, double *zh)
{
  if (empty(al, ah) || empty(bl, bh) || (bl == 0 && bh == 0)) { *zl = *zh = NAN; return; }
  if (al == 0 && ah == 0) { *zl = *zh = 0; return; }
  double lo = INFINITY, hi = -INFINITY, d, u;
  double as[2] = { al, ah }, bs[2] = { bl, bh };
  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 2; j++)
      if (bs[j] != 0 && endquot(as[i], bs[j], &d, &u)) { if (d < lo) lo = d; if (u > hi) hi = u; }
  int from_above = bh > 0 && bl <= 0, from_below = bl < 0 && bh >= 0;   /* B reaches 0 from that side */
  if (from_above) { if (al < 0) lo = -INFINITY; if (ah > 0) hi = INFINITY; }
  if (from_below) { if (al < 0) hi = INFINITY; if (ah > 0) lo = -INFINITY; }
  if (al <= 0 && 0 <= ah) { if (0 < lo) lo = 0; if (0 > hi) hi = 0; }
  *zl = lo; *zh = hi;
}

/* ---- comparison ---- */
static int same(double a, double b) { return (a != a && b != b) || a == b; }
static long checked, bad, inside, outside, neg_differs;
static char first[512];
static void cmp(const char *op, double al, double ah, double bl, double bh, double zl, double zh, double rl, double rh)
{
  checked++;
  if (!same(zl, rl) || !same(zh, rh)) {
    if (!bad++) snprintf(first, sizeof first, " (first: %s [%a, %a], [%a, %a] = [%a, %a], want [%a, %a])", op, al, ah, bl, bh, zl, zh, rl, rh);
  }
}

/* points inside: the exact result must lie in ival's interval */
static mpfr_t X, Y, Z;
static double pick(double lo, double hi)
{
  double l = isinf(lo) ? -DBL_MAX : lo, h = isinf(hi) ? DBL_MAX : hi;
  double t = (double)(rnd() >> 11) * 0x1p-53, x = l + t * (h - l);
  if (!isfinite(x)) x = t < 0.5 ? l : h;
  return x < l ? l : x > h ? h : x;
}
static void inside_check(int op, double al, double ah, double bl, double bh, double zl, double zh)
{
  if (empty(al, ah) || empty(bl, bh) || zl != zl) return;
  for (int k = 0; k < 4; k++) {
    double x = pick(al, ah), y = pick(bl, bh);
    mpfr_set_d(X, x, MPFR_RNDN); mpfr_set_d(Y, y, MPFR_RNDN);
    if (op == 0) mpfr_add(Z, X, Y, MPFR_RNDN);
    else if (op == 1) mpfr_sub(Z, X, Y, MPFR_RNDN);
    else if (op == 2) mpfr_mul(Z, X, Y, MPFR_RNDN);
    else { if (y == 0) continue; mpfr_div(Z, X, Y, MPFR_RNDN); }
    if (mpfr_cmp_d(Z, zl) >= 0 && mpfr_cmp_d(Z, zh) <= 0) inside++;
    else if (!outside++ && !bad) snprintf(first, sizeof first, " (first outside: op %d x %a y %a not in [%a, %a])", op, x, y, zl, zh);
  }
}

static const double SP[] = { -INFINITY, -DBL_MAX, -0x1p+960, -3, -2, -1, -0.5, -0x1p-960, -0x1p-969, -0x1.8p-1000, -DBL_MIN,
                             -DBL_TRUE_MIN, 0.0, DBL_TRUE_MIN, DBL_MIN, 0x1.8p-1000, 0x1p-969, 0x1p-960, 0.5, 1, 3, 0x1p+960, DBL_MAX,
                             INFINITY };
enum { NSP = sizeof SP / sizeof SP[0] };

int main(void)
{
  mpfr_init2(T, 2200); mpfr_init2(U, 2200); mpfr_init2(W, 4400); mpfr_init2(X, 64); mpfr_init2(Y, 64); mpfr_init2(Z, 2200);
  /* the intervals: special pairs, randoms, empties */
  enum { NR = 1 << 15 };
  static double L[NSP * NSP + NR + 8], H[NSP * NSP + NR + 8];
  int ni = 0;
  for (int i = 0; i < NSP; i++) for (int j = i; j < NSP; j++) { L[ni] = SP[i]; H[ni] = SP[j]; ni++; }
  for (int k = 0; k < NR; k++) {
    double a = anyd(), b;
    switch (k % 4) {
      case 0: b = anyd(); break;                                          /* any width */
      case 1: b = a; break;                                               /* a point */
      case 2: b = a; for (int s = (int)(rnd() % 4); s >= 0; s--) b = nextafter(b, INFINITY); break;   /* a few ulps */
      default: b = a * (1 + (double)(rnd() >> 11) * 0x1p-60); break;      /* relative width ~1e-3 */
    }
    if (b < a) { double t = a; a = b; b = t; }
    L[ni] = a; H[ni] = b; ni++;
  }
  L[ni] = NAN; H[ni] = NAN; ni++; L[ni] = 2; H[ni] = 1; ni++; L[ni] = INFINITY; H[ni] = INFINITY; ni++; L[ni] = -INFINITY; H[ni] = -INFINITY; ni++;

  /* one-interval operations on every interval */
  static double zl[NSP * NSP + NR + 8], zh[NSP * NSP + NR + 8];
  double rl, rh;
  ival_neg(L, H, zl, zh, ni);
  for (int i = 0; i < ni; i++) { int e = empty(L[i], H[i]); cmp("neg", L[i], H[i], 0, 0, zl[i], zh[i], e ? NAN : -H[i], e ? NAN : -L[i]); }
  ival_sqr(L, H, zl, zh, ni);
  for (int i = 0; i < ni; i++) { r_sqr(L[i], H[i], &rl, &rh); cmp("sqr", L[i], H[i], 0, 0, zl[i], zh[i], rl, rh); }
  ival_recip(L, H, zl, zh, ni);
  for (int i = 0; i < ni; i++) { r_div(1, 1, L[i], H[i], &rl, &rh); cmp("recip", L[i], H[i], 0, 0, zl[i], zh[i], rl, rh); }

  /* two-interval operations: every pair of the special intervals, and random pairs of all of them */
  int nsp = NSP * (NSP + 1) / 2, np = nsp * nsp + (1 << 18);
  double *al = malloc(np * sizeof *al), *ah = malloc(np * sizeof *ah), *bl = malloc(np * sizeof *bl), *bh = malloc(np * sizeof *bh);
  double *ol = malloc(np * sizeof *ol), *oh = malloc(np * sizeof *oh);
  int m = 0;
  for (int i = 0; i < nsp; i++) for (int j = 0; j < nsp; j++) { al[m] = L[i]; ah[m] = H[i]; bl[m] = L[j]; bh[m] = H[j]; m++; }
  while (m < np) { int i = (int)(rnd() % ni), j = (int)(rnd() % ni); al[m] = L[i]; ah[m] = H[i]; bl[m] = L[j]; bh[m] = H[j]; m++; }
  const char *names[4] = { "add", "sub", "mul", "div" };
  for (int op = 0; op < 4; op++) {
    if (op == 0) ival_add(al, ah, bl, bh, ol, oh, np);
    else if (op == 1) ival_sub(al, ah, bl, bh, ol, oh, np);
    else if (op == 2) ival_mul(al, ah, bl, bh, ol, oh, np);
    else ival_div(al, ah, bl, bh, ol, oh, np);
    for (int k = 0; k < np; k++) {
      if (op <= 1) r_add(al[k], ah[k], bl[k], bh[k], &rl, &rh, op);
      else if (op == 2) r_mul(al[k], ah[k], bl[k], bh[k], &rl, &rh);
      else r_div(al[k], ah[k], bl[k], bh[k], &rl, &rh);
      cmp(names[op], al[k], ah[k], bl[k], bh[k], ol[k], oh[k], rl, rh);
      inside_check(op, al[k], ah[k], bl[k], bh[k], ol[k], oh[k]);
      if (op == 0 && !empty(al[k], ah[k]) && !empty(bl[k], bh[k])) {   /* the negative control: ends rounded to nearest */
        double nl = al[k] + bl[k], nh = ah[k] + bh[k];
        if (!same(nl, rl) || !same(nh, rh)) neg_differs++;
      }
    }
  }
  /* fma: each pair above with a third interval, special or random */
  double *cl3 = malloc(np * sizeof *cl3), *ch3 = malloc(np * sizeof *ch3);
  for (int k = 0; k < np; k++) { int i = k < nsp * nsp ? (int)(rnd() % nsp) : (int)(rnd() % ni); cl3[k] = L[i]; ch3[k] = H[i]; }
  ival_fma(al, ah, bl, bh, cl3, ch3, ol, oh, np);
  for (int k = 0; k < np; k++) {
    r_fma(al[k], ah[k], bl[k], bh[k], cl3[k], ch3[k], &rl, &rh);
    if (!same(ol[k], rl) || !same(oh[k], rh)) {
      checked++;
      if (!bad++) snprintf(first, sizeof first, " (first: fma [%a, %a], [%a, %a], [%a, %a] = [%a, %a], want [%a, %a])", al[k], ah[k], bl[k],
                           bh[k], cl3[k], ch3[k], ol[k], oh[k], rl, rh);
    } else checked++;
    if (empty(al[k], ah[k]) || empty(bl[k], bh[k]) || empty(cl3[k], ch3[k])) continue;
    for (int q = 0; q < 2; q++) {   /* points inside: x * y + w exactly */
      double x = pick(al[k], ah[k]), y = pick(bl[k], bh[k]), w = pick(cl3[k], ch3[k]);
      mpfr_set_d(X, x, MPFR_RNDN); mpfr_mul_d(W, X, y, MPFR_RNDN); mpfr_add_d(W, W, w, MPFR_RNDN);
      if (mpfr_cmp_d(W, ol[k]) >= 0 && mpfr_cmp_d(W, oh[k]) <= 0) inside++;
      else if (!outside++ && !bad) snprintf(first, sizeof first, " (first outside: fma x %a y %a w %a not in [%a, %a])", x, y, w, ol[k], oh[k]);
    }
  }

  /* the paths, bit for bit: every operation run again on the same inputs */
  long pathc = 0, pathd = 0; char pfirst[256] = "";
  double *xl = malloc(np * sizeof *xl), *xh = malloc(np * sizeof *xh), *rl2 = malloc(np * sizeof *rl2), *rh2 = malloc(np * sizeof *rh2);
  const char *mode[4] = { "portable", "scalar", "flush modes set", "in place" };
  for (int op = 0; op < 9; op++) {   /* 8: sqrt, its MPFR reference in check.c */
    int two = op < 4 || op == 7, cnt = two ? np : ni;
    const double *a = two ? al : L, *b = two ? ah : H;
    for (int md = -1; md < 4; md++) {
      ival__arith_path = md == 0 ? 1 : md == 1 ? 2 : 0;
      unsigned long csr = fpctl();
      if (md == 2) set_fpctl(csr | FLUSH_BITS);
      double *ol2 = md == -1 ? rl2 : xl, *oh2 = md == -1 ? rh2 : xh;
      const double *a2 = a, *b2 = b;
      if (md == 3) { memcpy(xl, a, cnt * sizeof *xl); memcpy(xh, b, cnt * sizeof *xh); a2 = xl; b2 = xh; }
      switch (op) {
        case 0: ival_add(a2, b2, bl, bh, ol2, oh2, cnt); break;
        case 1: ival_sub(a2, b2, bl, bh, ol2, oh2, cnt); break;
        case 2: ival_mul(a2, b2, bl, bh, ol2, oh2, cnt); break;
        case 3: ival_div(a2, b2, bl, bh, ol2, oh2, cnt); break;
        case 4: ival_neg(a2, b2, ol2, oh2, cnt); break;
        case 5: ival_sqr(a2, b2, ol2, oh2, cnt); break;
        case 6: ival_recip(a2, b2, ol2, oh2, cnt); break;
        case 7: ival_fma(a2, b2, bl, bh, cl3, ch3, ol2, oh2, cnt); break;
        default: ival_sqrt(a2, b2, ol2, oh2, cnt); break;
      }
      if (md == 2) {
        if ((fpctl() & FLUSH_BITS) != FLUSH_BITS) { pathd++; if (!pfirst[0]) snprintf(pfirst, sizeof pfirst, " (first: op %d cleared the caller's flush modes)", op); }
        set_fpctl(csr);
      }
      if (md == -1) {   /* the default path, the reference for the runs after it; its zero ends must be +0 */
        for (int k = 0; k < cnt; k++)
          if ((rl2[k] == 0 && signbit(rl2[k])) || (rh2[k] == 0 && signbit(rh2[k]))) {
            if (!pathd++) snprintf(pfirst, sizeof pfirst, " (first: op %d gives a -0 end for [%a, %a])", op, a[k], b[k]);
          }
        continue;
      }
      for (int k = 0; k < cnt; k++) {
        pathc++;
        if (memcmp(&xl[k], &rl2[k], 8) || memcmp(&xh[k], &rh2[k], 8)) {
          if (!pathd++) snprintf(pfirst, sizeof pfirst, " (first: op %d %s: [%a, %a], [%a, %a] gives [%a, %a], the default [%a, %a])", op,
                                 mode[md], a[k], b[k], two ? bl[k] : 0, two ? bh[k] : 0, xl[k], xh[k], rl2[k], rh2[k]);
        }
      }
    }
    ival__arith_path = 0;
  }
#if defined(__x86_64__)
  int vec = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
  int vec = 0;
#endif
  printf("paths: %s; portable, scalar, flush modes set and in place all bit for bit the default on %ld of %ld results%s\n",
         vec ? "the default is the four-lane passes (AVX2, FMA)" : "NO VECTOR PATH on this CPU (the default is the portable passes)",
         pathc - pathd, pathc, pfirst);
  if (!bad && !outside && neg_differs > 0 && inside > 0 && !pathd)
    printf("VERDICT: IDENTICAL (%ld results the tightest enclosure, %ld points inside, none outside; control: rounding to nearest differs on %ld sums)\n",
           checked, inside, neg_differs);
  else
    printf("VERDICT: DIFFERS (%ld of %ld results not the tightest, %ld points outside, control %ld, %ld path differences)%s%s\n", bad, checked, outside, neg_differs, pathd, first, pfirst);
  return bad || outside || !neg_differs || pathd;
}
