/* crfix-mp.c: crfix's slow path (2026-10-14). Interval arithmetic in fixed point of five 64-bit words, 64 integer
   bits and 256 fractional, in integers only: every operation rounds a lower bound down and an upper bound up, so
   the result encloses the exact value by construction, with no error analysis to trust. It decides what the fast
   path's 64-bit enclosure cannot (one input in some millions), and every input of the affine formats.

     exp, exp2   2^w with w = v log2(e) (or v): 2^g for g in [0, 1] by e^u, u = g ln 2 / 2^8, its series to u^24 with
                 the tail bounded, squared eight times
     log, log2   v = 2^p m, m in [1, 2): ln m = 2 atanh((m - 1)/(m + 1)), its series to t^163 with the tail
                 bounded; 1/(m + 1) by Newton's iteration, then proven by two exact products
     sigmoid     1 / (1 + e^-|v|), or e^v / (1 + e^v) for v < 0 (relative accuracy as the value vanishes)
     tanh        2 / (1 + e^-2|v|) - 1
   A value too small for the format (below 2^-199) is kept as the open interval (0, 2^-199), and tanh and sigmoid
   within 2^-199 of 1 as (1 - 2^-199, 1): the rounding then knows on which side of an integer it lies. */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "crfix.h"
#include "crfix-internal.h"
#include "crfix-tables.h"

#define NW 5   /* value = sum of w[i] 2^(64 (i - 4)) */
typedef struct { uint64_t w[NW]; } mp;
/* an interval of magnitudes, its sign, which ends are open, and whether it is exact (lo = hi = the value) */
typedef struct { int neg, big, exact, lo_open, hi_open; mp lo, hi; } ival;

#if defined(__SIZEOF_INT128__) && !defined(CRFIX_PORTABLE)
typedef unsigned __int128 u128n;
static inline void mul64w(uint64_t a, uint64_t b, uint64_t *hi, uint64_t *lo) { u128n p = (u128n)a * b; *hi = (uint64_t)(p >> 64); *lo = (uint64_t)p; }
#else
static inline void mul64w(uint64_t a, uint64_t b, uint64_t *hi, uint64_t *lo)
{
  uint64_t a0 = (uint32_t)a, a1 = a >> 32, b0 = (uint32_t)b, b1 = b >> 32;
  uint64_t p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0, p11 = a1 * b1;
  uint64_t mid = (p00 >> 32) + (uint32_t)p01 + (uint32_t)p10;
  *hi = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
  *lo = mid << 32 | (uint32_t)p00;
}
#endif

static void set_u(mp *a, uint64_t v) { memset(a, 0, sizeof *a); a->w[4] = v; }
static void pow2m(mp *a, int k) { memset(a, 0, sizeof *a); a->w[(256 - k) / 64] = 1ULL << ((256 - k) % 64); }   /* 2^-k, 0 <= k <= 256 */
static int iszero(const mp *a) { for (int i = 0; i < NW; i++) if (a->w[i]) return 0; return 1; }
static int cmp(const mp *a, const mp *b)
{
  for (int i = NW - 1; i >= 0; i--) if (a->w[i] != b->w[i]) return a->w[i] < b->w[i] ? -1 : 1;
  return 0;
}
static int add(mp *r, const mp *a, const mp *b)   /* the carry out */
{
  uint64_t c = 0;
  for (int i = 0; i < NW; i++) { uint64_t s = a->w[i] + c; c = s < c; uint64_t t = s + b->w[i]; c += t < s; r->w[i] = t; }
  return (int)c;
}
static int sub(mp *r, const mp *a, const mp *b)   /* the borrow out */
{
  uint64_t c = 0;
  for (int i = 0; i < NW; i++) { uint64_t s = a->w[i] - c; uint64_t c1 = a->w[i] < c; uint64_t t = s - b->w[i]; c = c1 + (s < b->w[i]); r->w[i] = t; }
  return (int)c;
}
static void ulp_up(mp *a) { for (int i = 0; i < NW; i++) if (++a->w[i]) break; }
static void ulp_down(mp *a) { if (iszero(a)) return; for (int i = 0; i < NW; i++) if (a->w[i]--) break; }
/* a 2^-s, s >= 0: rounded up (if up) when bits fall off */
static void shr(mp *r, const mp *a, int s, int up)
{
  mp t = {{0}};
  int lost = 0;
  if (s >= 64 * NW) { lost = !iszero(a); }
  else {
    int q = s / 64, b = s % 64;
    for (int i = 0; i < q; i++) lost |= a->w[i] != 0;
    if (b) lost |= (a->w[q] << (64 - b)) != 0;
    for (int i = 0; i + q < NW; i++) {
      uint64_t lo = a->w[i + q] >> b, hi = (b && i + q + 1 < NW) ? a->w[i + q + 1] << (64 - b) : 0;
      t.w[i] = b ? lo | hi : a->w[i + q];
    }
  }
  *r = t;
  if (up && lost) ulp_up(r);
}
static int shl(mp *r, const mp *a, int s)   /* 1 if bits fall off the top */
{
  mp t = {{0}};
  int over = 0;
  if (s >= 64 * NW) over = !iszero(a);
  else {
    int q = s / 64, b = s % 64;
    for (int i = NW - q; i < NW; i++) over |= a->w[i] != 0;
    if (b) over |= (a->w[NW - 1 - q] >> (64 - b)) != 0;
    for (int i = NW - 1; i >= q; i--) {
      uint64_t hi = a->w[i - q] << b, lo = (b && i - q - 1 >= 0) ? a->w[i - q - 1] >> (64 - b) : 0;
      t.w[i] = b ? hi | lo : a->w[i - q];
    }
  }
  *r = t;
  return over;
}
/* the full product: ten words, value p 2^-512 */
static void mulfull(uint64_t *p, const mp *a, const mp *b)
{
  memset(p, 0, 2 * NW * sizeof *p);
  for (int i = 0; i < NW; i++) {
    uint64_t c = 0;
    for (int j = 0; j < NW; j++) {
      uint64_t hi, lo;
      mul64w(a->w[i], b->w[j], &hi, &lo);
      lo += c; hi += lo < c;
      uint64_t t = p[i + j] + lo;
      hi += t < lo;
      p[i + j] = t;
      c = hi;
    }
    p[i + NW] = c;
  }
}
/* a b, rounded down or up; 1 if it reaches 2^64 */
static int mul(mp *r, const mp *a, const mp *b, int up)
{
  uint64_t p[2 * NW];
  mulfull(p, a, b);
  int lost = p[0] || p[1] || p[2] || p[3];
  memcpy(r->w, p + 4, NW * sizeof *p);
  if (up && lost) ulp_up(r);
  return p[9] != 0;
}
static int mul_u(mp *r, const mp *a, uint64_t m)   /* exact; 1 if it reaches 2^64 */
{
  uint64_t c = 0;
  for (int i = 0; i < NW; i++) {
    uint64_t hi, lo;
    mul64w(a->w[i], m, &hi, &lo);
    lo += c; hi += lo < c;
    r->w[i] = lo; c = hi;
  }
  return c != 0;
}
/* (hi 2^64 + lo) / d for hi < d, its remainder */
static uint64_t div2(uint64_t hi, uint64_t lo, uint64_t d, uint64_t *rem)
{
#if defined(__SIZEOF_INT128__) && !defined(CRFIX_PORTABLE)
  u128n n = (u128n)hi << 64 | lo;
  *rem = (uint64_t)(n % d);
  return (uint64_t)(n / d);
#else
  uint64_t q = 0, r = hi;
  for (int i = 63; i >= 0; i--) {
    uint64_t top = r >> 63;
    r = r << 1 | (lo >> i & 1);
    q <<= 1;
    if (top || r >= d) { r -= d; q |= 1; }
  }
  *rem = r;
  return q;
#endif
}
/* a / d, rounded down or up; 1 when it was not exact */
static int div_u(mp *r, const mp *a, uint64_t d, int up)
{
  uint64_t rem = 0;
  for (int i = NW - 1; i >= 0; i--) r->w[i] = div2(rem, a->w[i], d, &rem);
  if (rem && up) ulp_up(r);
  return rem != 0;
}

static const mp *K_LN2(int up) { return (const mp *)CRFIX_MP_LN2[up]; }
static const mp *K_L2E(int up) { return (const mp *)CRFIX_MP_LOG2E[up]; }

/* 1/d for an exact d in [1, 4): a first guess from the top 64 bits, three Newton steps, then a bound proven by
   exact products: rounded down (d r <= 1) or up (d r >= 1) */
static void recip(mp *r, const mp *d, int up)
{
  uint64_t d64 = d->w[4] << 62 | d->w[3] >> 2, rem;   /* d 2^62, in [2^62, 2^64) */
  mp x;
  memset(&x, 0, sizeof x);
  if (d64 == 1ULL << 62) set_u(&x, 1);                 /* d = 1 (2^126 / d64 would not fit) */
  else x.w[3] = div2(1ULL << 62, 0, d64, &rem);        /* 2^126 / d64 = 2^64 / d */
  mp two, t;
  set_u(&two, 2);
  for (int it = 0; it < 3; it++) { mul(&t, d, &x, 0); sub(&t, &two, &t); mul(&x, &x, &t, 0); }
  /* the proof: a lower bound has d x <= 1, an upper one d x >= 1, each product exact (ten words) */
  for (int k = 0; k < 64; k++) {
    uint64_t p[2 * NW];
    mulfull(p, d, &x);
    int below = p[8] == 0 ? 1 : p[8] > 1 ? 0 : (p[0] | p[1] | p[2] | p[3] | p[4] | p[5] | p[6] | p[7]) ? 0 : 2;   /* 1: d x < 1, 2: = 1, 0: > 1 */
    if (up ? below != 1 : below != 0) break;
    if (up) ulp_up(&x); else ulp_down(&x);
  }
  *r = x;
}

/* 2^g for g in [gl, gh], 0 <= g <= 2: e^u, u = g ln 2 / 2^8, to u^24 (u^25 / 25! < 2^-284), squared eight times */
static void pow2g(const mp *gl, const mp *gh, mp *lo, mp *hi)
{
  mp ul, uh;
  mul(&ul, gl, K_LN2(0), 0); shr(&ul, &ul, 8, 0);
  mul(&uh, gh, K_LN2(1), 1); shr(&uh, &uh, 8, 1);
  for (int side = 0; side < 2; side++) {
    const mp *u = side ? &uh : &ul;
    mp sum, term;
    set_u(&sum, 1); set_u(&term, 1);
    for (unsigned k = 1; k <= 24; k++) {
      mul(&term, &term, u, side);
      div_u(&term, &term, k, side);
      add(&sum, &sum, &term);
    }
    if (side) {   /* the tail: below 2 u^25 / 25! */
      mul(&term, &term, u, 1); div_u(&term, &term, 25, 1); add(&term, &term, &term); add(&sum, &sum, &term);
    }
    for (int s = 0; s < 8; s++) mul(&sum, &sum, &sum, side);
    if (side) *hi = sum; else *lo = sum;
  }
}

/* |v| as an interval of mp: v = N 2^e exactly, N = (nhi, nlo); 1 if it reaches 2^64 */
static int from_dyadic(uint64_t nhi, uint64_t nlo, int e, mp *lo, mp *hi, int *exact)
{
  mp a;
  memset(&a, 0, sizeof a);
  a.w[0] = nlo; a.w[1] = nhi;   /* N 2^-256 */
  int s = 256 + e;              /* times 2^s */
  *exact = 1;
  if (s >= 0) { if (shl(&a, &a, s)) return 1; *lo = *hi = a; return 0; }
  mp b;
  shr(lo, &a, -s, 0);
  shr(&b, &a, -s, 1);
  *hi = b;
  *exact = cmp(lo, hi) == 0;
  return 0;
}

/* e^v or 2^v for v = (-1)^neg N 2^e: an interval, big when it reaches 2^63, or the open (0, 2^-199) */
static void exp_iv(int base2, int neg, uint64_t nhi, uint64_t nlo, int e, ival *r)
{
  memset(r, 0, sizeof *r);
  mp vl, vh, wl, wh;
  int ex;
  if (!nhi && !nlo) { set_u(&r->lo, 1); r->hi = r->lo; r->exact = 1; return; }
  if (from_dyadic(nhi, nlo, e, &vl, &vh, &ex) || vl.w[4] >= 4096) {   /* |v| >= 4096 */
    if (!neg) r->big = 1;
    else { pow2m(&r->hi, 199); r->lo_open = 1; }   /* (0, 2^-199) */
    return;
  }
  if (base2) { wl = vl; wh = vh; }
  else { mul(&wl, &vl, K_L2E(0), 0); mul(&wh, &vh, K_L2E(1), 1); }
  mp gl, gh, K;
  uint64_t k;
  if (!neg) {   /* 2^w = 2^k 2^g, k = floor(wl), g = w - k */
    k = wl.w[4];
    set_u(&K, k);
    sub(&gl, &wl, &K); sub(&gh, &wh, &K);
    if (base2 && ex && iszero(&gl)) {   /* an integer: 2^k exactly */
      if (k >= 63) { r->big = 1; return; }
      set_u(&r->lo, 1ULL << k); r->hi = r->lo; r->exact = 1; return;
    }
    pow2g(&gl, &gh, &r->lo, &r->hi);
    if (k >= 62 || shl(&r->lo, &r->lo, (int)k) || shl(&r->hi, &r->hi, (int)k)) r->big = 1;
  } else {      /* 2^-w = 2^-K 2^g, K = floor(wh) + 1, g = K - w in (0, 1 + (wh - wl)] */
    k = wh.w[4] + 1;
    if (base2 && ex) {
      mp f = wl; f.w[4] = 0;
      if (iszero(&f)) {   /* 2^-k exactly, k = wl */
        uint64_t kk = wl.w[4];
        if (kk > 199) { pow2m(&r->hi, 199); r->lo_open = 1; return; }
        memset(&r->lo, 0, sizeof r->lo); r->lo.w[4] = 1; shr(&r->lo, &r->lo, (int)kk, 0); r->hi = r->lo; r->exact = 1; return;
      }
    }
    if (k > 199) { pow2m(&r->hi, 199); r->lo_open = 1; return; }
    set_u(&K, k);
    sub(&gl, &K, &wh); sub(&gh, &K, &wl);
    pow2g(&gl, &gh, &r->lo, &r->hi);
    shr(&r->lo, &r->lo, (int)k, 0); shr(&r->hi, &r->hi, (int)k, 1);
    if (iszero(&r->lo)) r->lo_open = 1;   /* below the format but above 0 */
  }
}

/* ln or log2 of v = N 2^e > 0 */
static void log_iv(int base2, uint64_t nhi, uint64_t nlo, int e, ival *r)
{
  memset(r, 0, sizeof *r);
  int bl = nhi ? 128 - __builtin_clzll(nhi) : 64 - __builtin_clzll(nlo);
  int64_t p = (int64_t)bl - 1 + e;
  /* m = N 2^-(bl - 1) in [1, 2), exactly */
  mp m;
  memset(&m, 0, sizeof m);
  m.w[0] = nlo; m.w[1] = nhi;
  shl(&m, &m, 256 - (bl - 1));
  mp one, frac;
  set_u(&one, 1);
  sub(&frac, &m, &one);
  uint64_t ap = p < 0 ? (uint64_t)-p : (uint64_t)p;
  if (iszero(&frac)) {   /* a power of two: log2 exact; ln exact only at 1 */
    if (base2 || !p) { set_u(&r->lo, ap); r->hi = r->lo; r->exact = 1; r->neg = p < 0; return; }
    mul_u(&r->lo, K_LN2(0), ap); mul_u(&r->hi, K_LN2(1), ap); r->neg = p < 0; return;
  }
  /* t = (m - 1) / (m + 1) <= 1/3 */
  mp d, rl, rh, tl, th;
  add(&d, &m, &one);
  recip(&rl, &d, 0); recip(&rh, &d, 1);
  mul(&tl, &frac, &rl, 0); mul(&th, &frac, &rh, 1);
  mp S[2];
  for (int side = 0; side < 2; side++) {   /* sum of t^(2k+1) / (2k+1), k = 0 .. 81; then 2 S = ln m */
    const mp *t = side ? &th : &tl;
    mp t2, pw = *t, term, sum = *t;
    mul(&t2, t, t, side);
    for (unsigned k = 1; k <= 81; k++) {
      mul(&pw, &pw, &t2, side);
      div_u(&term, &pw, 2 * k + 1, side);
      add(&sum, &sum, &term);
    }
    if (side) {   /* the tail: below t^165 / (165 (1 - t^2)) <= 2 t^165 / 165 */
      mul(&pw, &pw, &t2, 1); div_u(&term, &pw, 165, 1); add(&term, &term, &term); add(&sum, &sum, &term);
    }
    add(&S[side], &sum, &sum);
  }
  mp a[2], b[2];   /* the integer part's contribution and ln m (or log2 m), each bound */
  if (base2) {
    set_u(&a[0], ap); a[1] = a[0];
    mul(&b[0], &S[0], K_L2E(0), 0); mul(&b[1], &S[1], K_L2E(1), 1);
  } else {
    mul_u(&a[0], K_LN2(0), ap); mul_u(&a[1], K_LN2(1), ap);
    b[0] = S[0]; b[1] = S[1];
  }
  if (p >= 0) { add(&r->lo, &a[0], &b[0]); add(&r->hi, &a[1], &b[1]); }
  else {          /* -(|p| ln 2 - ln m): |p| ln 2 > ln m, as m < 2 */
    r->neg = 1;
    sub(&r->lo, &a[0], &b[1]); sub(&r->hi, &a[1], &b[0]);
  }
}

/* 1/(1 + e) for e in [el, eh] (0 <= e <= 1) */
static void recip1p(const mp *el, const mp *eh, mp *rl, mp *rh)
{
  mp one, d;
  set_u(&one, 1);
  add(&d, &one, eh); recip(rl, &d, 0);
  add(&d, &one, el); recip(rh, &d, 1);
}
static void sigmoid_iv(int neg, uint64_t nhi, uint64_t nlo, int e, ival *r)
{
  memset(r, 0, sizeof *r);
  if (!nhi && !nlo) { r->lo.w[3] = 1ULL << 63; r->hi = r->lo; r->exact = 1; return; }   /* 1/2 */
  ival E;
  exp_iv(0, 1, nhi, nlo, e, &E);   /* e^-|v| */
  mp tiny;
  pow2m(&tiny, 199);
  if (E.lo_open && iszero(&E.lo) && cmp(&E.hi, &tiny) <= 0) {   /* e^-|v| below 2^-199 */
    if (neg) { r->hi = tiny; r->lo_open = 1; }                    /* (0, 2^-199) */
    else { set_u(&r->hi, 1); sub(&r->lo, &r->hi, &tiny); r->lo_open = r->hi_open = 1; }   /* (1 - 2^-199, 1) */
    return;
  }
  mp rl, rh;
  recip1p(&E.lo, &E.hi, &rl, &rh);
  if (!neg) { r->lo = rl; r->hi = rh; }
  else { mul(&r->lo, &E.lo, &rl, 0); mul(&r->hi, &E.hi, &rh, 1); if (iszero(&r->lo)) r->lo_open = 1; }   /* e^v / (1 + e^v) */
}
static void tanh_iv(int neg, uint64_t nhi, uint64_t nlo, int e, ival *r)
{
  memset(r, 0, sizeof *r);
  r->neg = neg;
  if (!nhi && !nlo) { r->exact = 1; r->neg = 0; return; }   /* 0 */
  ival E;
  exp_iv(0, 1, nhi, nlo, e + 1, &E);   /* e^-2|v| */
  mp tiny;
  pow2m(&tiny, 199);
  if (E.lo_open && iszero(&E.lo) && cmp(&E.hi, &tiny) <= 0) {   /* e^-2|v| below 2^-199: tanh |v| in (1 - 2^-198, 1) */
    pow2m(&tiny, 198);
    set_u(&r->hi, 1); sub(&r->lo, &r->hi, &tiny);
    r->lo_open = r->hi_open = 1;
    return;
  }
  mp rl, rh, one;
  recip1p(&E.lo, &E.hi, &rl, &rh);
  set_u(&one, 1);
  add(&rl, &rl, &rl); add(&rh, &rh, &rh);   /* 2 r */
  if (cmp(&rl, &one) <= 0) { memset(&r->lo, 0, sizeof r->lo); r->lo_open = 1; } else sub(&r->lo, &rl, &one);
  sub(&r->hi, &rh, &one);
}

/* the rounding: Y = (-1)^neg V / s + z, V in the interval, s = so 2^eo; each end's integer part and whether its
   fraction is 0, below, at or above 1/2 */
typedef struct { int64_t ip; int frac; } endpt;   /* frac: 0 none, 1 below 1/2, 2 exactly 1/2, 3 above */
/* returns 1 past 2^62 (saturation); *inexact when the scaling rounded */
static int scale_end(const mp *v, uint64_t so, int eo, int up, int neg, int64_t z, endpt *o, int *inexact)
{
  mp m, t;
  *inexact = div_u(&m, v, so, up);
  if (eo <= 0) { if (shl(&m, &m, -eo)) return 1; }
  else {
    mp u;
    shr(&t, &m, eo, 0);
    shl(&u, &t, eo);
    int lost = cmp(&u, &m) != 0;   /* bits shifted out */
    *inexact |= lost;
    m = t;
    if (up && lost) ulp_up(&m);
  }
  if (m.w[4] >> 62) return 1;
  int64_t ip = (int64_t)m.w[4];
  mp f = m; f.w[4] = 0;
  mp half; memset(&half, 0, sizeof half); half.w[3] = 1ULL << 63;
  int fc = iszero(&f) ? 0 : cmp(&f, &half) < 0 ? 1 : cmp(&f, &half) == 0 ? 2 : 3;
  if (!neg) { o->ip = z + ip; o->frac = fc; }
  else {   /* -(ip + f) = -(ip + 1) + (1 - f) */
    if (!fc) { o->ip = z - ip; o->frac = 0; }
    else { o->ip = z - ip - 1; o->frac = fc == 1 ? 3 : fc == 3 ? 1 : 2; }
  }
  return 0;
}
/* floor, ceiling and floor(Y + 1/2) of an end */
static int64_t e_floor(const endpt *e) { return e->ip; }
static int64_t e_ceil(const endpt *e) { return e->ip + (e->frac != 0); }
static int64_t e_floorh(const endpt *e) { return e->ip + (e->frac >= 2); }
static int64_t e_ceilh(const endpt *e) { return e->ip + 1 + (e->frac == 3); }   /* ceil(ip + f + 1/2) */

/* the result before saturation; 0 if undecided (never seen: then the middle) */
static int round_iv(const ival *v, uint64_t so, int eo, int64_t z, int mode, int64_t *q)
{
  if (v->big) { *q = v->neg ? INT64_MIN : INT64_MAX; return 1; }
  endpt L, H;
  int lo_open, hi_open, inl, inh;
  /* the signed interval's lower end comes from the magnitude's upper end when negative */
  if (scale_end(v->neg ? &v->hi : &v->lo, so, eo, v->neg, v->neg, z, &L, &inl) ||
      scale_end(v->neg ? &v->lo : &v->hi, so, eo, !v->neg, v->neg, z, &H, &inh)) {
    *q = v->neg ? INT64_MIN : INT64_MAX;
    return 1;
  }
  lo_open = v->neg ? v->hi_open : v->lo_open;
  hi_open = v->neg ? v->lo_open : v->hi_open;
  int point = v->exact || (!v->lo_open && !v->hi_open && cmp(&v->lo, &v->hi) == 0);   /* a closed point: exact */
  if (point && (inl || inh)) lo_open = hi_open = 1;   /* an exact value whose scaling rounded: strictly between */
  else if (point) {   /* one value, exactly: ties to even */
    switch (mode) {
      case CRFIX_DOWN: *q = e_floor(&L); break;
      case CRFIX_UP: *q = e_ceil(&L); break;
      case CRFIX_ZERO: *q = L.ip >= 0 ? e_floor(&L) : e_ceil(&L); break;
      default: *q = L.frac == 2 ? L.ip + (L.ip & 1) : e_floorh(&L); break;
    }
    return 1;
  }
  int64_t fmin = e_floor(&L), fmax = hi_open ? e_ceil(&H) - 1 : e_floor(&H);
  int64_t cmin = lo_open ? e_floor(&L) + 1 : e_ceil(&L), cmax = e_ceil(&H);
  int64_t nmin = e_floorh(&L), nmax = hi_open ? e_ceilh(&H) - 1 : e_floorh(&H);
  switch (mode) {
    case CRFIX_DOWN: *q = fmin; return fmin == fmax;
    case CRFIX_UP: *q = cmax; return cmin == cmax;
    case CRFIX_ZERO:
      if (fmin >= 0 && fmin == fmax) { *q = fmin; return 1; }   /* the value >= 0: toward zero is the floor */
      if (cmax <= 0 && cmin == cmax) { *q = cmax; return 1; }
      *q = fmin;
      return 0;
    default: *q = nmin; return nmin == nmax;
  }
}

static void enclose(int fn, int neg, uint64_t nhi, uint64_t nlo, int e, ival *r)
{
  switch (fn) {
    case CRFIX_EXP: exp_iv(0, neg, nhi, nlo, e, r); break;
    case CRFIX_EXP2: exp_iv(1, neg, nhi, nlo, e, r); break;
    case CRFIX_LOG: case CRFIX_LOG2: log_iv(fn == CRFIX_LOG2, nhi, nlo, e, r); break;
    case CRFIX_SIGMOID: sigmoid_iv(neg, nhi, nlo, e, r); break;
    default: tanh_iv(neg, nhi, nlo, e, r); break;
  }
}

/* the slow path: f((-1)^neg N 2^e) / (so 2^eo) + z, rounded in mode; the domain (log of v <= 0) is the caller's */
int64_t crfix_mp_internal(int fn, int neg, uint64_t nhi, uint64_t nlo, int e, uint64_t so, int eo, int64_t z, int mode)
{
  ival v;
  int64_t q;
  enclose(fn, neg, nhi, nlo, e, &v);
  round_iv(&v, so, eo, z, mode, &q);
  return q;
}

/* softmax (2026-10-14): p_i = e^(c d_i) / sum_j e^(c d_j), d_i = q_i - max q <= 0, c = cm 2^ce > 0; each p_i / (so 2^eo)
   + z rounded into out[i], before saturation. Each distinct d's exponential once (an int8 vector has at most 256),
   as an interval; their sum's bounds exact sums of bounds; 1/S from S = 2^k S', S' in [1, 2), by recip(). 0, or -1
   for want of memory */
int crfix_softmax_internal(size_t n, const int32_t *q, uint64_t cm, int ce, uint64_t so, int eo, int64_t z, int mode,
                           int64_t *out)
{
  int32_t mx = q[0], mn = q[0];
  for (size_t i = 1; i < n; i++) { if (q[i] > mx) mx = q[i]; if (q[i] < mn) mn = q[i]; }
  uint64_t range = (uint64_t)((int64_t)mx - mn);
  int cache = range < 65536;
  ival *E = malloc((cache ? range + 1 : n) * sizeof *E);
  unsigned char *have = cache ? calloc(range + 1, 1) : NULL;
  if (!E || (cache && !have)) { free(E); free(have); return -1; }
  mp Slo, Shi;
  memset(&Slo, 0, sizeof Slo); memset(&Shi, 0, sizeof Shi);
  for (size_t i = 0; i < n; i++) {
    uint64_t d = (uint64_t)((int64_t)mx - q[i]);   /* -d_i >= 0 */
    ival *e = cache ? &E[d] : &E[i];
    if (!cache || !have[d]) {
      uint64_t hi, lo;
      mul64w(d, cm, &hi, &lo);
      exp_iv(0, 1, hi, lo, ce, e);   /* e^(-c d) */
      if (cache) have[d] = 1;
    }
    add(&Slo, &Slo, &e->lo); add(&Shi, &Shi, &e->hi);
  }
  /* S in [1, n]: S = 2^k S', and 1/S' by recip() each way */
  int k = 63 - __builtin_clzll(Shi.w[4]);
  mp sl, sh, rl, rh;
  shr(&sl, &Slo, k, 0); shr(&sh, &Shi, k, 1);
  mp one;
  set_u(&one, 1);
  if (cmp(&sl, &one) < 0) sl = one;   /* S >= 1: the largest term is e^0 = 1 */
  recip(&rl, &sh, 0); recip(&rh, &sl, 1);
  for (size_t i = 0; i < n; i++) {
    uint64_t d = (uint64_t)((int64_t)mx - q[i]);
    const ival *e = cache ? &E[d] : &E[i];
    ival p;
    memset(&p, 0, sizeof p);
    mul(&p.lo, &e->lo, &rl, 0); shr(&p.lo, &p.lo, k, 0);
    mul(&p.hi, &e->hi, &rh, 1); shr(&p.hi, &p.hi, k, 1);
    p.lo_open = e->lo_open || (iszero(&p.lo) && !iszero(&e->hi));   /* a positive value below the format */
    round_iv(&p, so, eo, z, mode, &out[i]);
  }
  free(E); free(have);
  return 0;
}
