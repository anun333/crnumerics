/* crfix.c: crfix.h's functions (2026-10-13). Integer arithmetic only: uint64_t, and unsigned __int128 where the
   compiler has it; CRFIX_PORTABLE forces the portable code, which the check also runs. */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "crfix.h"
#include "crfix-internal.h"
#include "crfix-tables.h"

#if defined(__SIZEOF_INT128__) && !defined(CRFIX_PORTABLE)
#define NATIVE128 1
typedef unsigned __int128 n128;
#endif

const char *crfix_version(void) { return CRFIX_VERSION; }

/* ---- 128-bit words ---- */
typedef crfix_u128 u128;
static inline u128 W(uint64_t hi, uint64_t lo) { u128 r; r.hi = hi; r.lo = lo; return r; }
static inline u128 add(u128 a, u128 b) { u128 r; r.lo = a.lo + b.lo; r.hi = a.hi + b.hi + (r.lo < a.lo); return r; }
static inline u128 sub(u128 a, u128 b) { u128 r; r.lo = a.lo - b.lo; r.hi = a.hi - b.hi - (a.lo < b.lo); return r; }
static inline u128 neg(u128 a) { return sub(W(0, 0), a); }
static inline int isneg(u128 a) { return (int)(a.hi >> 63); }
static inline u128 from_i64(int64_t v) { return W(v < 0 ? ~0ULL : 0, (uint64_t)v); }
static inline u128 shl(u128 a, int s)   /* 0 <= s < 128 */
{
  if (s == 0) return a;
  if (s >= 64) return W(a.lo << (s - 64), 0);
  return W(a.hi << s | a.lo >> (64 - s), a.lo << s);
}
static inline u128 shr(u128 a, int s)   /* logical, s >= 0 */
{
  if (s == 0) return a;
  if (s >= 128) return W(0, 0);
  if (s >= 64) return W(0, a.hi >> (s - 64));
  return W(a.hi >> s, a.lo >> s | a.hi << (64 - s));
}
static inline u128 sar(u128 a, int s)   /* arithmetic, 0 <= s < 128: floor(a / 2^s) */
{
  if (s == 0) return a;
  if (s >= 64) return W(isneg(a) ? ~0ULL : 0, (uint64_t)((int64_t)a.hi >> (s - 64)));
  return W((uint64_t)((int64_t)a.hi >> s), a.lo >> s | a.hi << (64 - s));
}
static inline int ucmp(u128 a, u128 b) { return a.hi != b.hi ? (a.hi < b.hi ? -1 : 1) : (a.lo < b.lo ? -1 : a.lo > b.lo); }
static inline int scmp(u128 a, u128 b) { return ucmp(W(a.hi ^ (1ULL << 63), a.lo), W(b.hi ^ (1ULL << 63), b.lo)); }

static inline u128 mul64(uint64_t a, uint64_t b)
{
#ifdef NATIVE128
  n128 p = (n128)a * b;
  return W((uint64_t)(p >> 64), (uint64_t)p);
#else
  uint64_t a0 = (uint32_t)a, a1 = a >> 32, b0 = (uint32_t)b, b1 = b >> 32;
  uint64_t p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0, p11 = a1 * b1;
  uint64_t mid = (p00 >> 32) + (uint32_t)p01 + (uint32_t)p10;
  return W(p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32), mid << 32 | (uint32_t)p00);
#endif
}
static inline uint64_t mulhi(uint64_t a, uint64_t b) { return mul64(a, b).hi; }
/* (a b) >> 63 for Q0.63 values, |a|, |b| < 2^63, the magnitude truncated */
static inline int64_t mul63(int64_t a, int64_t b)
{
  uint64_t ua = a < 0 ? -(uint64_t)a : (uint64_t)a, ub = b < 0 ? -(uint64_t)b : (uint64_t)b;
  u128 p = mul64(ua, ub);
  uint64_t m = p.hi << 1 | p.lo >> 63;
  return (a < 0) != (b < 0) ? -(int64_t)m : (int64_t)m;
}
static int clz64(uint64_t v);
#ifndef NATIVE128
/* (u1 2^64 + u0) / v for u1 < v, in two 64-by-32-bit steps (Hacker's Delight, divlu) */
static uint64_t divlu(uint64_t u1, uint64_t u0, uint64_t v, uint64_t *r)
{
  const uint64_t b = 1ULL << 32;
  int s = clz64(v);
  v <<= s;
  uint64_t vn1 = v >> 32, vn0 = v & 0xffffffffu;
  uint64_t un32 = u1 << s | (s ? u0 >> (64 - s) : 0), un10 = u0 << s, un1 = un10 >> 32, un0 = un10 & 0xffffffffu;
  uint64_t q1 = un32 / vn1, rhat = un32 - q1 * vn1;
  while (q1 >= b || q1 * vn0 > b * rhat + un1) { q1--; rhat += vn1; if (rhat >= b) break; }
  uint64_t un21 = un32 * b + un1 - q1 * v, q0 = un21 / vn1;
  rhat = un21 - q0 * vn1;
  while (q0 >= b || q0 * vn0 > b * rhat + un0) { q0--; rhat += vn1; if (rhat >= b) break; }
  *r = (un21 * b + un0 - q0 * v) >> s;
  return q1 * b + q0;
}
#endif
/* n / d and n % d, d > 0 */
static u128 div128_64(u128 n, uint64_t d, uint64_t *rem)
{
#ifdef NATIVE128
  n128 x = (n128)n.hi << 64 | n.lo;
  *rem = (uint64_t)(x % d);
  x /= d;
  return W((uint64_t)(x >> 64), (uint64_t)x);
#else
  uint64_t qh = n.hi / d, ql = divlu(n.hi % d, n.lo, d, rem);
  return W(qh, ql);
#endif
}
/* 2^127 / d for 2^64 <= d <= 2^65: the quotient (below 2^64), and whether it is exact */
static uint64_t div2p127(u128 d, int *exact)
{
#ifdef NATIVE128
  n128 y = (n128)d.hi << 64 | d.lo, n = (n128)1 << 127;
  *exact = n % y == 0;
  return (uint64_t)(n / y);
#else
  if (d.hi >= 2) { *exact = 1; return 1ULL << 62; }   /* d = 2^65 */
  /* q' = 2^126 / floor(d / 2) is the quotient or one more (2^127 / (d - 1) - 2^127 / d < 1 for d >= 2^64) */
  uint64_t r, h = (d.lo >> 1) | 1ULL << 63, q = divlu(1ULL << 62, 0, h, &r);
  u128 p = add(mul64(q, d.lo), W(q, 0)), n = W(1ULL << 63, 0);   /* q d, d.hi = 1 */
  if (ucmp(p, n) > 0) { q--; p = sub(p, d); }
  *exact = ucmp(p, n) == 0;
  return q;
#endif
}
static uint64_t isqrt128(u128 n)   /* floor(sqrt(n)), n < 2^126: from the root's top bit down */
{
  uint64_t r = 0;
  if (!n.hi && !n.lo) return 0;
  int bits = n.hi ? 128 - clz64(n.hi) : 64 - clz64(n.lo);
  for (int i = (bits - 1) / 2; i >= 0; i--) {
    uint64_t c = r | 1ULL << i;
    if (ucmp(mul64(c, c), n) <= 0) r = c;
  }
  return r;
}
static int clz64(uint64_t v)   /* v > 0 */
{
#if defined(__GNUC__)
  return __builtin_clzll(v);
#else
  int n = 0;
  while (!(v >> 63)) { v <<= 1; n++; }
  return n;
#endif
}

/* ---- enclosures ---- */
static void iv_exact(crfix_iv *e, u128 v) { e->kind = CRFIX_IV_EXACT; e->lo = e->hi = v; }
static void iv_open(crfix_iv *e, u128 lo, u128 hi) { e->kind = CRFIX_IV_OPEN; e->lo = lo; e->hi = hi; }

/* 2^f 2^64 for f = F 2^-64 in [0, 1): 2^(j/64) (1 + q), q = 2^t - 1 by Taylor to t^7, t < 2^-6; in [2^64, 2^65),
   within 4 units */
static u128 pow2frac(uint64_t f)
{
  unsigned j = (unsigned)(f >> 58);
  uint64_t t = f & ((1ULL << 58) - 1), acc = CRFIX_C[7];
  for (int k = 6; k >= 1; k--) acc = CRFIX_C[k] + mulhi(acc, t);
  uint64_t q = mulhi(acc, t), a = CRFIX_T2[j];
  return add(add(add(W(1, 0), W(0, a)), W(0, q)), W(0, mulhi(a, q)));
}
/* 2^w for w (signed Q64.64), exact_w when w is exactly the argument (else within 16 units of it: the margin,
   m / 2^59 + 4, holds 2^k (4 + 1.39 delta) + 1 for an error of delta units up to about 20); BIG from 2^kbig
   (kbig <= 62), less those units */
static void pow2_ivk(crfix_iv *e, u128 w, int exact_w, int kbig)
{
  int64_t k = (int64_t)w.hi;   /* floor(w) */
  uint64_t f = w.lo;
  if (k >= kbig) { e->kind = CRFIX_IV_BIG; return; }
  if (k < -72) { iv_open(e, W(0, 0), W(0, 1)); return; }   /* 0 < 2^w < 2^-72 */
  if (exact_w && !f) {
    if (k >= -64) iv_exact(e, shl(W(0, 1), (int)(64 + k)));
    else iv_open(e, W(0, 0), W(0, 1));
    return;
  }
  u128 v = pow2frac(f), m = k >= 0 ? shl(v, (int)k) : shr(v, (int)-k), d = add(shr(m, 59), W(0, 4)), lo = sub(m, d);
  iv_open(e, isneg(lo) ? W(0, 0) : lo, add(m, d));
}
static void pow2_iv(crfix_iv *e, u128 w, int exact_w) { pow2_ivk(e, w, exact_w, 32); }
/* e^v for v = x 2^-fin, |x| <= 2^33: 2^w with w = v log2(e) from LOG2E = log2(e) 2^126, |w| 2^64 = |x| LOG2E >>
   (62 + fin), truncated (within 2 units) */
static void exp_iv(crfix_iv *e, int64_t x, int fin)
{
  if (!x) { iv_exact(e, W(1, 0)); return; }
  uint64_t ax = x < 0 ? -(uint64_t)x : (uint64_t)x;
  u128 pl = mul64(ax, CRFIX_LOG2E[1]), ph = mul64(ax, CRFIX_LOG2E[0]);
  uint64_t w0 = pl.lo, w1 = pl.hi + ph.lo, w2 = ph.hi + (w1 < pl.hi);
  int s = 62 + fin;   /* 62 .. 93 */
  u128 r;
  uint64_t over;
  if (s < 64) { r = W(w1 >> s | w2 << (64 - s), w0 >> s | w1 << (64 - s)); over = w2 >> s; }
  else { int t = s - 64; r = W(t ? w2 >> t : w2, t ? w1 >> t | w2 << (64 - t) : w1); over = 0; }
  if (over || r.hi >= 64) {   /* |w| >= 64 */
    if (x > 0) e->kind = CRFIX_IV_BIG;
    else iv_open(e, W(0, 0), W(0, 1));
    return;
  }
  pow2_iv(e, x > 0 ? r : neg(r), 0);
}
/* e^v for v = (-1)^negv V 2^-64, V exact (vexact) or below |v| 2^64 by less than one unit: w = V LOG2E >> 126,
   truncated, below |w| 2^64 by less than 1 + log2(e) units; BIG from 2^62 */
static void exp_q(crfix_iv *e, u128 V, int negv, int vexact)
{
  if (!V.hi && !V.lo && vexact) { iv_exact(e, W(1, 0)); return; }
  if (V.hi >= 64) {   /* |v| >= 64 */
    if (negv) iv_open(e, W(0, 0), W(0, 1));
    else e->kind = CRFIX_IV_BIG;
    return;
  }
  u128 a = mul64(V.lo, CRFIX_LOG2E[1]), b = mul64(V.lo, CRFIX_LOG2E[0]), c = mul64(V.hi, CRFIX_LOG2E[1]),
       d = mul64(V.hi, CRFIX_LOG2E[0]);
  u128 mid = add(add(W(0, a.hi), W(0, b.lo)), W(0, c.lo)), top = add(add(add(W(0, b.hi), W(0, c.hi)), d), W(0, mid.hi));
  u128 w = W(top.hi << 2 | top.lo >> 62, top.lo << 2 | mid.lo >> 62);   /* bits 126 .. 253 of the product */
  pow2_ivk(e, negv ? neg(w) : w, 0, 62);
}
/* log2 v or ln v for v = x 2^-fin, x > 0: x = 2^p (1 + t); 1 + t = (1 + s) / r with r = R16[j] / 2^16, so
   log(1 + t) = -log(r) + ln(1 + s), |s| < 2^-7, ln(1 + s) = s - s^2 g(s) to s^9; within 64 units */
/* the core, for v = 2^ip m, m (Q1.63, top bit set) its mantissa exactly (mexact) or truncated within 2^-63 */
static void log_core(crfix_iv *e, int fn, uint64_t m, int64_t ip, int mexact)
{
  if (mexact && m == 1ULL << 63 && (fn == CRFIX_LOG2 || ip == 0)) { iv_exact(e, W((uint64_t)ip, 0)); return; }
  uint64_t aip = ip < 0 ? -(uint64_t)ip : (uint64_t)ip;
  u128 base;
  if (fn == CRFIX_LOG2) base = W((uint64_t)ip, 0);
  else {   /* ip ln 2, from LN2 = ln(2) 2^128, truncated */
    u128 a = mul64(aip, CRFIX_LN2[1]), b = mul64(aip, CRFIX_LN2[0]), v = add(b, W(0, a.hi));
    base = ip < 0 ? neg(v) : v;
  }
  u128 v = base;
  if (m != 1ULL << 63) {
    unsigned j = (unsigned)(m >> 57) & 63;
    u128 P = mul64(m, CRFIX_R16[j]);
    int64_t s = (int64_t)((P.hi << 48 | P.lo >> 16) - (1ULL << 63));   /* Q0.63 */
    int64_t g = CRFIX_INV[9];
    for (int k = 8; k >= 2; k--) g = CRFIX_INV[k] - mul63(s, g);
    int64_t ln = s - mul63(mul63(s, s), g);
    if (fn == CRFIX_LOG2) v = add(add(v, W(0, CRFIX_L2R[j])), shl(from_i64(ln + mul63(ln, CRFIX_IL2M1)), 1));
    else v = add(add(v, W(0, CRFIX_LNR[j])), shl(from_i64(ln), 1));
  }
  iv_open(e, sub(v, W(0, 64)), add(v, W(0, 64)));
}
static void log_iv(crfix_iv *e, int fn, int32_t x, int fin)
{
  uint64_t ux = (uint64_t)x;
  int p = 63 - clz64(ux);
  log_core(e, fn, ux << (63 - p), p - fin, 1);
}
/* r = 1 / (1 + e) for e in (E.lo, E.hi) 2^-64, 0 <= e <= 1: rlo < r 2^63 < rhi, by division rounded each way */
static void recip1p(const crfix_iv *E, uint64_t *rlo, uint64_t *rhi)
{
  u128 elo = isneg(E->lo) ? W(0, 0) : E->lo, ehi = ucmp(E->hi, W(1, 0)) > 0 ? W(1, 0) : E->hi;
  int exact;
  *rlo = div2p127(add(W(1, 0), ehi), &exact);
  *rhi = div2p127(add(W(1, 0), elo), &exact);
  *rhi += !exact;
}
/* small |v| < 2^-16 (fin > 16, |x| < 2^(fin - 16)): for v > 0, f(v) = V - c with V exact and c between
   v^3 / K and v^3 / K less the next term of the alternating series (v^5 / 480 for sigmoid, 2 v^5 / 15 for tanh,
   below 2^-80, so less than one unit of 2^-64). The enclosure (V - C, V - max(0, C - 1)) of the value for v > 0, with
   C = ceil and floor of |x|^3 2^(64 - 3 fin) / K, sits strictly below V: a boundary at V, which the dyadic V often
   is, never straddles it (a generic enclosure, a few units wide, would, as v^3 / K falls below the units) */
static int small(int32_t x, int fin)
{
  return fin > 16 && (x < 0 ? -(int64_t)x : x) < ((int64_t)1 << (fin - 16));
}
static void small_iv(crfix_iv *e, u128 V, uint64_t ax, int fin, uint64_t K)
{
  uint64_t x3 = ax * ax * ax, cf, cc;   /* ax < 2^15 */
  int s = 64 - 3 * fin;
  if (s >= 0) { uint64_t n = x3 << s; cf = n / K; cc = cf + (n % K != 0); }
  else { uint64_t d = K << -s; cf = x3 / d; cc = cf + (x3 % d != 0); }
  iv_open(e, sub(V, W(0, cc)), sub(V, W(0, cf ? cf - 1 : 0)));
}
/* sigmoid from E, an enclosure of e^-|v|; and tanh from E of e^-2|v| */
static void sigmoid_from(crfix_iv *e, const crfix_iv *E, int negv)
{
  uint64_t rlo, rhi;
  recip1p(E, &rlo, &rhi);
  u128 lo = shl(W(0, rlo), 1), hi = shl(W(0, rhi), 1);   /* sigmoid(|v|) = r */
  if (!negv) iv_open(e, lo, hi);
  else iv_open(e, sub(W(1, 0), hi), sub(W(1, 0), lo));  /* sigmoid(-|v|) = 1 - r */
}
static void tanh_from(crfix_iv *e, const crfix_iv *E, int negv)
{
  uint64_t rlo, rhi;
  recip1p(E, &rlo, &rhi);
  u128 lo = sub(shl(W(0, rlo), 2), W(1, 0)), hi = sub(shl(W(0, rhi), 2), W(1, 0));   /* tanh |v| = 2 r - 1 */
  if (!negv) iv_open(e, lo, hi);
  else iv_open(e, neg(hi), neg(lo));
}
static void sigmoid_iv(crfix_iv *e, int32_t x, int fin)
{
  if (!x) { iv_exact(e, W(0, 1ULL << 63)); return; }
  int64_t ax = x < 0 ? -(int64_t)x : x;
  if (small(x, fin)) {   /* sigmoid(|v|) = 1/2 + |v|/4 - c, c ~ |v|^3 / 48; sigmoid(-|v|) = 1 - that */
    small_iv(e, W(0, (1ULL << 63) + ((uint64_t)ax << (62 - fin))), (uint64_t)ax, fin, 48);
    if (x < 0) iv_open(e, sub(W(1, 0), e->hi), sub(W(1, 0), e->lo));
    return;
  }
  crfix_iv E;
  exp_iv(&E, -ax, fin);   /* e^-|v| */
  sigmoid_from(e, &E, x < 0);
}
static void tanh_iv(crfix_iv *e, int32_t x, int fin)
{
  if (!x) { iv_exact(e, W(0, 0)); return; }
  int64_t ax = x < 0 ? -(int64_t)x : x;
  if (small(x, fin)) {   /* tanh |v| = |v| - c, c ~ |v|^3 / 3 */
    small_iv(e, W(0, (uint64_t)ax << (64 - fin)), (uint64_t)ax, fin, 3);
    if (x < 0) iv_open(e, neg(e->hi), neg(e->lo));
    return;
  }
  crfix_iv E;
  exp_iv(&E, fin ? -ax : -2 * ax, fin ? fin - 1 : 0);   /* e^-2|v| */
  tanh_from(e, &E, x < 0);
}

void crfix_enclose_internal(int fn, int32_t x, int fin, crfix_iv *e)
{
  switch (fn) {
    case CRFIX_EXP: exp_iv(e, x, fin); break;
    case CRFIX_EXP2:
      /* x 2^-fin as signed Q64.64: exact */
      pow2_iv(e, shl(from_i64(x), 64 - fin), 1);
      break;
    case CRFIX_LOG: case CRFIX_LOG2:
      if (x <= 0) e->kind = CRFIX_IV_DOMAIN;
      else log_iv(e, fn, x, fin);
      break;
    case CRFIX_SIGMOID: sigmoid_iv(e, x, fin); break;
    default: tanh_iv(e, x, fin); break;
  }
}

/* round v 2^fout to an integer: v in signed Q64.64, |v| < 2^32 */
int crfix_round_internal(const crfix_iv *e, int fout, int mode, int64_t *q)
{
  int D = 64 - fout;
  u128 half = shl(W(0, 1), D - 1);
  if (e->kind == CRFIX_IV_BIG) { *q = INT64_MAX; return 1; }
  if (e->kind == CRFIX_IV_EXACT) {
    u128 fl = sar(e->lo, D), rem = sub(e->lo, shl(fl, D));
    int64_t f = (int64_t)fl.lo;
    int nz = rem.hi || rem.lo, c;
    switch (mode) {
      case CRFIX_DOWN: *q = f; break;
      case CRFIX_UP: *q = f + nz; break;
      case CRFIX_ZERO: *q = isneg(e->lo) ? f + nz : f; break;
      default: c = ucmp(rem, half); *q = c < 0 ? f : c > 0 ? f + 1 : f + (f & 1); break;
    }
    return 1;
  }
  /* past every container at this fout (|v 2^fout| >= 2^31): saturates however it rounds */
  if (scmp(e->lo, shl(W(0, 1), 95 - fout)) >= 0) { *q = INT64_MAX; return 1; }
  if (scmp(e->hi, neg(shl(W(0, 1), 95 - fout))) <= 0) { *q = INT64_MIN; return 1; }
  /* open: decided when no integer (nearest: no integer plus a half) lies strictly inside */
  u128 lo = e->lo, hi = e->hi;
  if (mode == CRFIX_NEAREST) { lo = add(lo, half); hi = add(hi, half); }
  u128 A = sar(lo, D), B = neg(sar(neg(hi), D));
  if (scmp(sub(B, A), W(0, 1)) > 0) return 0;
  int64_t a = (int64_t)A.lo;
  switch (mode) {
    case CRFIX_DOWN: case CRFIX_NEAREST: *q = a; break;
    case CRFIX_UP: *q = a + 1; break;
    default: *q = a >= 0 ? a : a + 1; break;
  }
  return 1;
}

/* ---- the functions ---- */
static int ok(int fin, int bout, int fout, int mode)
{
  return (unsigned)fin < 32 && (unsigned)fout < 32 && (bout == 8 || bout == 16 || bout == 32) && (unsigned)mode < 4;
}
static int32_t sat(int64_t q, int bout)
{
  int64_t mx = ((int64_t)1 << (bout - 1)) - 1;
  return (int32_t)(q > mx ? mx : q < -mx - 1 ? -mx - 1 : q);
}
static int32_t approx(int fn, int32_t x, int fin, int bout, int fout, int mode)
{
  if (!ok(fin, bout, fout, mode)) return 0;
  crfix_iv e;
  int64_t q;
  crfix_enclose_internal(fn, x, fin, &e);
  if (e.kind == CRFIX_IV_DOMAIN) return sat(INT64_MIN, bout);
  if (!crfix_round_internal(&e, fout, mode, &q))   /* undecided (one input in millions): the slow path */
    q = crfix_mp_internal(fn, x < 0, 0, x < 0 ? -(uint64_t)(int64_t)x : (uint64_t)x, -fin, 1, -fout, 0, mode);
  return sat(q, bout);
}
int32_t crfix_exp(int32_t x, int fin, int bout, int fout, int mode) { return approx(CRFIX_EXP, x, fin, bout, fout, mode); }
int32_t crfix_exp2(int32_t x, int fin, int bout, int fout, int mode) { return approx(CRFIX_EXP2, x, fin, bout, fout, mode); }
int32_t crfix_log(int32_t x, int fin, int bout, int fout, int mode) { return approx(CRFIX_LOG, x, fin, bout, fout, mode); }
int32_t crfix_log2(int32_t x, int fin, int bout, int fout, int mode) { return approx(CRFIX_LOG2, x, fin, bout, fout, mode); }
int32_t crfix_sigmoid(int32_t x, int fin, int bout, int fout, int mode) { return approx(CRFIX_SIGMOID, x, fin, bout, fout, mode); }
int32_t crfix_tanh(int32_t x, int fin, int bout, int fout, int mode) { return approx(CRFIX_TANH, x, fin, bout, fout, mode); }

/* the exact ones: a nonnegative quotient or root q with what is left, rounded by sign and mode. cmp: the sign of
   (the exact magnitude's fraction - 1/2), for nearest; nz: whether the fraction is not 0 */
static int64_t round_exact(int negative, uint64_t q, int nz, int cmp, int mode)
{
  uint64_t r;
  switch (mode) {
    case CRFIX_NEAREST: r = cmp < 0 ? q : cmp > 0 ? q + 1 : q + (q & 1); break;
    case CRFIX_UP: r = negative ? q : q + nz; break;
    case CRFIX_DOWN: r = negative ? q + nz : q; break;
    default: r = q; break;
  }
  if (r > (1ULL << 40)) r = 1ULL << 40;   /* past every container: saturation follows */
  return negative ? -(int64_t)r : (int64_t)r;
}
int32_t crfix_sqrt(int32_t x, int fin, int bout, int fout, int mode)
{
  if (!ok(fin, bout, fout, mode)) return 0;
  if (x <= 0) return 0;
  int s = 2 * fout - fin;
  uint64_t q;
  int nz, cmp;
  if (s >= 0) {   /* y^2 = N = x 2^s, an integer: (q + 1/2)^2 = q^2 + q + 1/4 is never one, so no ties */
    u128 N = shl(W(0, (uint64_t)x), s), q2;
    q = isqrt128(N);
    q2 = mul64(q, q);
    nz = ucmp(q2, N) != 0;
    cmp = ucmp(N, add(q2, W(0, q))) > 0 ? 1 : -1;
  } else {        /* y^2 = x / 2^k: q = isqrt(floor(x / 2^k)); compare (2 q + 1)^2 2^k with 4 x */
    int k = -s;
    q = isqrt128(W(0, (uint64_t)x >> k));
    nz = ucmp(shl(mul64(q, q), k), W(0, (uint64_t)x)) != 0;
    int c = ucmp(shl(mul64(2 * q + 1, 2 * q + 1), k), W(0, 4 * (uint64_t)x));
    cmp = -c;
  }
  return sat(round_exact(0, q, nz, cmp, mode), bout);
}
int32_t crfix_rsqrt(int32_t x, int fin, int bout, int fout, int mode)
{
  if (!ok(fin, bout, fout, mode)) return 0;
  if (x <= 0) return sat(INT64_MAX, bout);
  int E = 2 * fout + fin;   /* y^2 = 2^E / x */
  uint64_t r;
  u128 N = shl(W(0, 1), E), Nq = div128_64(N, (uint64_t)x, &r);
  uint64_t q = isqrt128(Nq);
  u128 qq = mul64(q, q), t = mul64(2 * q + 1, 2 * q + 1);
  /* q^2 x against 2^E, and (2 q + 1)^2 x against 2^(E + 2): each product below 2^127 */
  u128 qqx = add(mul64(qq.lo, (uint64_t)x), W(qq.hi * (uint64_t)x, 0));
  u128 tx = add(mul64(t.lo, (uint64_t)x), W(t.hi * (uint64_t)x, 0));
  int nz = ucmp(qqx, N) != 0, cmp = -ucmp(tx, shl(W(0, 1), E + 2));
  return sat(round_exact(0, q, nz, cmp, mode), bout);
}
static int64_t divide(int negative, u128 N, uint64_t D, int mode)
{
  uint64_t rem;
  u128 q = div128_64(N, D, &rem);
  if (q.hi || q.lo > (1ULL << 40)) return negative ? -((int64_t)1 << 40) : (int64_t)1 << 40;
  int cmp = rem > D - rem ? 1 : rem < D - rem ? -1 : 0;
  return round_exact(negative, q.lo, rem != 0, cmp, mode);
}
int32_t crfix_div(int32_t a, int fa, int32_t b, int fb, int bout, int fout, int mode)
{
  if (!ok(fa, bout, fout, mode) || (unsigned)fb >= 32) return 0;
  if (!b) return a > 0 ? sat(INT64_MAX, bout) : a < 0 ? sat(INT64_MIN, bout) : 0;
  if (!a) return 0;
  uint64_t ua = a < 0 ? -(uint64_t)a : (uint64_t)a, ub = b < 0 ? -(uint64_t)b : (uint64_t)b;
  int s = fb + fout - fa, negative = (a < 0) != (b < 0);
  int64_t q = s >= 0 ? divide(negative, shl(W(0, ua), s), ub, mode) : divide(negative, W(0, ua), ub << -s, mode);
  return sat(q, bout);
}
int32_t crfix_recip(int32_t x, int fin, int bout, int fout, int mode)
{
  if (!ok(fin, bout, fout, mode)) return 0;
  return crfix_div(1, 0, x, fin, bout, fout, mode);
}

int32_t crfix_eval(int fn, int32_t x, int fin, int bout, int fout, int mode)
{
  switch (fn) {
    case CRFIX_EXP: case CRFIX_EXP2: case CRFIX_LOG: case CRFIX_LOG2: case CRFIX_SIGMOID: case CRFIX_TANH:
      return approx(fn, x, fin, bout, fout, mode);
    case CRFIX_SQRT: return crfix_sqrt(x, fin, bout, fout, mode);
    case CRFIX_RSQRT: return crfix_rsqrt(x, fin, bout, fout, mode);
    case CRFIX_RECIP: return crfix_recip(x, fin, bout, fout, mode);
    default: return 0;
  }
}
/* ---- affine formats (2026-10-13) ---- */
/* s = m 2^e from the bits of a positive normal binary64 (no floating-point operation); 0, or -1 */
static int scale_bits(double s, uint64_t *m, int *e)
{
  uint64_t b;
  memcpy(&b, &s, sizeof b);
  unsigned ex = (unsigned)(b >> 52) & 0x7ff;
  if (b >> 63 || ex == 0 || ex == 0x7ff) return -1;
  *m = (b & ((1ULL << 52) - 1)) | 1ULL << 52;
  *e = (int)ex - 1075;
  return 0;
}
/* the bound A 2^sh / so (so in [2^52, 2^53)) of X / (so 2^-sh) for X signed Q64.64: down (up = 0) or up, *inexact
   when not exact; *big (the value not computed) when its magnitude is 2^97 units (2^33) or more */
static u128 scale_end(u128 X, int up, uint64_t so, int sh, int *big, int *inexact)
{
  int negx = isneg(X);
  u128 A = negx ? neg(X) : X, Q;
  uint64_t r;
  *big = *inexact = 0;
  if (!A.hi && !A.lo) return A;
  int bl = A.hi ? 128 - clz64(A.hi) : 64 - clz64(A.lo);
  if (bl + sh >= 151) { *big = 1; return A; }   /* A 2^sh / so > 2^(bl - 1 + sh - 53) >= 2^97 */
  if (sh <= 0) {   /* floor(floor(A / so) / 2^-sh) */
    Q = div128_64(A, so, &r);
    *inexact = r != 0;
    if (-sh >= 128) { *inexact |= Q.hi || Q.lo; Q = W(0, 0); }
    else if (sh) { u128 t = shr(Q, -sh); *inexact |= ucmp(shl(t, -sh), Q) != 0; Q = t; }
  } else {         /* A 2^s1 below 2^127, then the remaining s2 <= 23 bits through the remainder */
    int s1 = sh < 127 - bl ? sh : 127 - bl, s2 = sh - s1;
    Q = div128_64(shl(A, s1), so, &r);
    if (s2) {
      uint64_t r2;
      Q = add(shl(Q, s2), div128_64(shl(W(0, r), s2), so, &r2));
      r = r2;
    }
    *inexact = r != 0;
  }
  if (*inexact && up != negx) Q = add(Q, W(0, 1));   /* the ceiling of the magnitude */
  return negx ? neg(Q) : Q;
}
/* V = |v| 2^64 for |v| = N 2^e: 1 when exact, 0 when truncated (below by less than one unit); |v| >= 128 as 128,
   inexact (every result the same) */
static int mag_q(u128 N, int e, u128 *V)
{
  int t = e + 64, bl = N.hi ? 128 - clz64(N.hi) : N.lo ? 64 - clz64(N.lo) : 0;
  if (!bl) { *V = W(0, 0); return 1; }
  if (bl + t > 71) { *V = W(128, 0); return 0; }
  if (t >= 0) { *V = shl(N, t); return 1; }
  *V = shr(N, -t);
  return -t < 128 && ucmp(shl(*V, -t), N) == 0;
}
/* an enclosure f (not DOMAIN) divided by so 2^eo with each end rounded outward, plus z, rounded: 1 and the result
   before saturation, or 0 when undecided */
static int out_round(const crfix_iv *f, uint64_t so, int eo, int64_t z, int mode, int64_t *q)
{
  if (f->kind == CRFIX_IV_BIG) {   /* f >= 2^62 nearly: past every container when the scale is below 2^29 */
    if (eo <= -24) { *q = INT64_MAX; return 1; }
    return 0;
  }
  if (mode == CRFIX_ZERO) {   /* toward zero is f / scale's, before z is added (crfix.h): down or up by f's sign */
    if (!isneg(f->lo)) mode = CRFIX_DOWN;
    else if (scmp(f->hi, W(0, 0)) <= 0) mode = CRFIX_UP;
    else return 0;   /* the sign undecided: the slow path */
  }
  crfix_iv y;
  int bigl, bigh, il, ih;
  u128 lo = scale_end(f->lo, 0, so, -eo, &bigl, &il), hi;
  if (f->kind == CRFIX_IV_EXACT) { bigh = bigl; ih = il; hi = il ? add(lo, W(0, 1)) : lo; }
  else hi = scale_end(f->hi, 1, so, -eo, &bigh, &ih);
  if (bigl && !isneg(f->lo)) { *q = INT64_MAX; return 1; }   /* f / scale >= 2^33 */
  if (bigh && isneg(f->hi)) { *q = INT64_MIN; return 1; }    /* <= -2^33 */
  if (bigl || bigh) return 0;
  u128 Z = W((uint64_t)z, 0);
  if (f->kind == CRFIX_IV_EXACT && !il) iv_exact(&y, add(lo, Z));
  else iv_open(&y, add(lo, Z), add(hi, Z));
  return crfix_round_internal(&y, 0, mode, q);
}
/* the fast path of an affine format: f(v) for v = (-1)^negv N 2^ei enclosed as in the fixed-point one, then
   out_round */
static int affine_fast(int fn, int negv, u128 N, int ei, uint64_t so, int eo, int64_t z, int mode, int64_t *q)
{
  crfix_iv f, E;
  int bl = N.hi ? 128 - clz64(N.hi) : N.lo ? 64 - clz64(N.lo) : 0;
  if (fn == CRFIX_LOG || fn == CRFIX_LOG2) {   /* N > 0: its top 64 bits, exact or truncated */
    int s = bl > 64 ? bl - 64 : 0, mexact = !s || !(N.lo & ((1ULL << s) - 1));
    log_core(&f, fn, s ? shr(N, s).lo : N.lo << (64 - bl), (int64_t)bl - 1 + ei, mexact);
    if (!mexact) f.hi = add(f.hi, W(0, 3));   /* log of the truncated m: below by less than 2 log2(e) units */
  } else {
    u128 V;
    int vexact = mag_q(N, ei, &V);
    switch (fn) {
      case CRFIX_EXP: exp_q(&f, V, negv, vexact); break;
      case CRFIX_EXP2: pow2_ivk(&f, negv ? neg(V) : V, vexact, 62); break;
      default:
        if (!V.hi && V.lo < 1ULL << 48) {   /* |v| < 2^-16: the slow path, but for 0 */
          if (V.lo || !vexact) return 0;
          iv_exact(&f, fn == CRFIX_SIGMOID ? W(0, 1ULL << 63) : W(0, 0));
        } else if (fn == CRFIX_SIGMOID) { exp_q(&E, V, 1, vexact); sigmoid_from(&f, &E, negv); }
        else { exp_q(&E, shl(V, 1), 1, vexact); tanh_from(&f, &E, negv); }
        break;
    }
  }
  return out_round(&f, so, eo, z, mode, q);
}
int crfix_affine_fast_internal(int fn, int32_t q, const crfix_affine *in, const crfix_affine *out, int mode, int64_t *r)
{
  uint64_t mi, mo;
  int ei, eo;
  if (scale_bits(in->scale, &mi, &ei) || scale_bits(out->scale, &mo, &eo)) return 0;
  int64_t d = (int64_t)q - in->zero_point;
  if ((fn == CRFIX_LOG || fn == CRFIX_LOG2) && d <= 0) { *r = INT64_MIN; return 1; }
  return affine_fast(fn, d < 0, mul64(d < 0 ? -(uint64_t)d : (uint64_t)d, mi), ei, mo, eo, out->zero_point, mode, r);
}
int32_t crfix_eval_affine(int fn, int32_t q, const crfix_affine *in, const crfix_affine *out, int mode)
{
  uint64_t mi, mo;
  int ei, eo;
  if ((unsigned)fn > CRFIX_TANH || !in || !out || scale_bits(in->scale, &mi, &ei) || scale_bits(out->scale, &mo, &eo) ||
      (out->bits != 8 && out->bits != 16 && out->bits != 32) || (unsigned)mode >= 4)
    return 0;
  int64_t d = (int64_t)q - in->zero_point;   /* v = scale (q - zero_point) = (+-|d| mi) 2^ei, exactly */
  if ((fn == CRFIX_LOG || fn == CRFIX_LOG2) && d <= 0) return sat(INT64_MIN, out->bits);
  u128 N = mul64(d < 0 ? -(uint64_t)d : (uint64_t)d, mi);
  int64_t r;
  if (!affine_fast(fn, d < 0, N, ei, mo, eo, out->zero_point, mode, &r))   /* undecided: the slow path */
    r = crfix_mp_internal(fn, d < 0, N.hi, N.lo, ei, mo, eo, out->zero_point, mode);
  return sat(r, out->bits);
}
int crfix_table_affine(int fn, const crfix_affine *in, const crfix_affine *out, int mode, int32_t table[256])
{
  uint64_t m;
  int e;
  if ((unsigned)fn > CRFIX_TANH || !in || !out || in->bits != 8 || scale_bits(in->scale, &m, &e) ||
      scale_bits(out->scale, &m, &e) || (out->bits != 8 && out->bits != 16 && out->bits != 32) || (unsigned)mode >= 4)
    return -1;
  for (int q = -128; q < 128; q++) table[q + 128] = crfix_eval_affine(fn, q, in, out, mode);
  return 0;
}

/* bounds of A 2^64 / S for 0 <= A < 2^65 and 2^64 <= S < 2^126, down (up = 0) or up: S's top 64 bits as the divisor,
   rounded the other way */
static u128 div_s(u128 A, u128 S, int up)
{
  int s = 64 - clz64(S.hi);   /* 1 .. 62 */
  uint64_t D = shr(S, s).lo, r;
  if (!up && ucmp(shl(W(0, D), s), S) != 0) {   /* the divisor rounded up */
    if (D == UINT64_MAX) { D = 1ULL << 63; s++; }
    else D++;
  }
  u128 Q = div128_64(shl(A, 64 - s), D, &r);
  return up && r ? add(Q, W(0, 1)) : Q;
}
/* softmax's fast path: e^(c d_i) by exp_q, d_i = q_i - max q <= 0, c = cm 2^ce; the sum's bounds sums of bounds; p_i's
   by div_s, then out_round. 1 when every element is decided, else 0 (the slow path then takes the vector). Two
   passes, each enclosure computed again rather than kept */
static int softmax_fast(size_t n, const int32_t *q, uint64_t cm, int ce, uint64_t so, int eo, int64_t z, int mode,
                        int64_t *out)
{
  int32_t mx = q[0];
  for (size_t i = 1; i < n; i++) if (q[i] > mx) mx = q[i];
  u128 Slo = W(0, 0), Shi = W(0, 0), V;
  crfix_iv E, p;
  int same = 1, vx;
  for (size_t i = 0; i < n; i++) {
    uint64_t d = (uint64_t)((int64_t)mx - q[i]);
    same &= !d;
    vx = mag_q(mul64(d, cm), ce, &V);
    exp_q(&E, V, 1, vx);   /* e^(-c |d|), in [0, 1] */
    Slo = add(Slo, E.lo); Shi = add(Shi, E.hi);
  }
  if (same) {   /* p = 1/n: exactly, or between two units */
    uint64_t r;
    u128 f = div128_64(W(1, 0), (uint64_t)n, &r);
    if (r) iv_open(&p, f, add(f, W(0, 1)));
    else iv_exact(&p, f);
    for (size_t i = 0; i < n; i++) if (!out_round(&p, so, eo, z, mode, &out[i])) return 0;
    return 1;
  }
  if (Shi.hi >> 61) return 0;   /* n of 2^61 or more */
  for (size_t i = 0; i < n; i++) {
    uint64_t d = (uint64_t)((int64_t)mx - q[i]);
    vx = mag_q(mul64(d, cm), ce, &V);
    exp_q(&E, V, 1, vx);
    iv_open(&p, div_s(E.lo, Shi, 0), div_s(E.hi, Slo, 1));   /* S open (some d > 0): strictly inside */
    if (!out_round(&p, so, eo, z, mode, &out[i])) return 0;
  }
  return 1;
}
int crfix_softmax_fast_internal(size_t n, const int32_t *q, double c, const crfix_affine *out, int mode, int64_t *r)
{
  uint64_t cm, so;
  int ce, eo;
  if (scale_bits(c, &cm, &ce) || scale_bits(out->scale, &so, &eo)) return 0;
  return softmax_fast(n, q, cm, ce, so, eo, out->zero_point, mode, r);
}
int crfix_softmax(size_t n, const int32_t *q, double c, const crfix_affine *out, int mode, int32_t *y)
{
  uint64_t cm, so;
  int ce, eo;
  if (!n || !q || !y || !out || scale_bits(c, &cm, &ce) || scale_bits(out->scale, &so, &eo) ||
      (out->bits != 8 && out->bits != 16 && out->bits != 32) || (unsigned)mode >= 4)
    return -1;
  int64_t *r = malloc(n * sizeof *r);
  if (!r) return -1;
  if (!softmax_fast(n, q, cm, ce, so, eo, out->zero_point, mode, r) &&   /* undecided: the slow path */
      crfix_softmax_internal(n, q, cm, ce, so, eo, out->zero_point, mode, r)) { free(r); return -1; }
  for (size_t i = 0; i < n; i++) y[i] = sat(r[i], out->bits);
  free(r);
  return 0;
}

int crfix_eval_n(int fn, size_t n, const int32_t *x, int fin, int bout, int fout, int mode, int32_t *y)
{
  if ((unsigned)fn >= CRFIX_NFN || !ok(fin, bout, fout, mode) || (n && (!x || !y))) return -1;
  for (size_t i = 0; i < n; i++) y[i] = crfix_eval(fn, x[i], fin, bout, fout, mode);
  return 0;
}
int crfix_eval_affine_n(int fn, size_t n, const int32_t *q, const crfix_affine *in, const crfix_affine *out, int mode,
                        int32_t *y)
{
  uint64_t m;
  int e;
  if ((unsigned)fn > CRFIX_TANH || !in || !out || scale_bits(in->scale, &m, &e) || scale_bits(out->scale, &m, &e) ||
      (out->bits != 8 && out->bits != 16 && out->bits != 32) || (unsigned)mode >= 4 || (n && (!q || !y)))
    return -1;
  for (size_t i = 0; i < n; i++) y[i] = crfix_eval_affine(fn, q[i], in, out, mode);
  return 0;
}
int crfix_div_n(size_t n, const int32_t *a, int fa, const int32_t *b, int fb, int bout, int fout, int mode, int32_t *y)
{
  if (!ok(fa, bout, fout, mode) || (unsigned)fb >= 32 || (n && (!a || !b || !y))) return -1;
  for (size_t i = 0; i < n; i++) y[i] = crfix_div(a[i], fa, b[i], fb, bout, fout, mode);
  return 0;
}
int crfix_table(int fn, int fin, int bout, int fout, int mode, int32_t table[256])
{
  if ((unsigned)fn >= CRFIX_NFN || !ok(fin, bout, fout, mode)) return -1;
  for (int x = -128; x < 128; x++) table[x + 128] = crfix_eval(fn, x, fin, bout, fout, mode);
  return 0;
}
