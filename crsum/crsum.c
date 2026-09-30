/* crsum.c: correctly rounded sums and dot products (crsum.h).

   The accumulator is a fixed-point integer: limb i holds the bits of
   weight 2^(E0 + 32 i) to 2^(E0 + 32 i + 31), E0 = -2176, in a signed
   64-bit integer, so that carries can wait. A term is m 2^e with m an
   integer (53 bits for a binary64 value, 106 for the exact product of two)
   and e >= -2148 (the least product's exponent). m is cut into 32-bit
   pieces; each piece, shifted to its place, touches two neighbouring
   limbs. A term changes a limb by less than 2^33, so 2^29 terms change it
   by less than 2^62: the carries are settled every 2^29 terms, and before
   rounding or merging. The top limb, 2^2080 to 2^2111, holds the sign; no
   sum of fewer than 2^60 products reaches 2^2111.

   Rounding reads the settled integer: its leading bit, the p bits from it
   (p = 53 or 24; fewer for a subnormal result), the next bit and whether
   anything below it is set. Integer arithmetic only: the C rounding mode
   plays no part. */
#include <math.h>
#include <string.h>
#include "crsum.h"

enum { E0 = -2176, L = CRSUM_LIMBS };
/* 2^29 by the bound above; a build may settle more often (the check builds
   one that settles every 3 terms, so the settling runs on every test) */
#ifndef CRSUM_SETTLE_EVERY
#define CRSUM_SETTLE_EVERY (1ULL << 29)
#endif
#define SETTLE_EVERY ((uint64_t)(CRSUM_SETTLE_EVERY))

void crsum_init(crsum_acc *a) { memset(a, 0, sizeof *a); }

/* the carries settled: every limb but the top in [0, 2^32), the top signed */
static void settle(crsum_acc *a)
{
  for (int i = 0; i < L - 1; i++) {
    int64_t v = a->limb[i], low = v & 0xffffffff;   /* v mod 2^32, two's complement */
    a->limb[i] = low;
    a->limb[i + 1] += (v - low) / 4294967296;      /* exact: v - low is a multiple of 2^32 */
  }
  a->since = 0;
}

/* sign * m * 2^e into the accumulator, m < 2^106 */
static void put(crsum_acc *a, int neg, unsigned __int128 m, int e)
{
  int p = e - E0, i = p >> 5, sh = p & 31;
  for (int k = 0; m; k++, m >>= 32) {
    uint64_t v = (uint64_t)(m & 0xffffffff) << sh;   /* under 2^63 */
    int64_t lo = (int64_t)(v & 0xffffffff), hi = (int64_t)(v >> 32);
    if (neg) { lo = -lo; hi = -hi; }
    a->limb[i + k] += lo;
    a->limb[i + k + 1] += hi;
  }
  if (++a->since >= SETTLE_EVERY) settle(a);
}

/* x's integer significand and exponent: x = +-m 2^e, finite, nonzero */
static void split(double x, uint64_t *m, int *e)
{
  uint64_t b;
  memcpy(&b, &x, 8);
  int f = (int)(b >> 52 & 0x7ff);
  *m = b & ((1ULL << 52) - 1);
  if (f) { *m |= 1ULL << 52; *e = f - 1075; } else *e = -1074;
}

static void zero_seen(crsum_acc *a, int neg)
{
  if (neg) a->negzero = 1; else a->poszero = 1;
}

void crsum_add(crsum_acc *a, const double *x, size_t n)
{
  for (size_t i = 0; i < n; i++) {
    double v = x[i];
    a->terms++;
    if (isnan(v)) a->nan = 1;
    else if (isinf(v)) { if (v > 0) a->pinf = 1; else a->ninf = 1; }
    else if (v == 0) zero_seen(a, signbit(v));
    else {
      uint64_t m;
      int e;
      split(v, &m, &e);
      a->nonzero = 1;
      put(a, signbit(v) != 0, m, e);
    }
  }
}

void crsum_add_dot(crsum_acc *a, const double *x, const double *y, size_t n)
{
  for (size_t i = 0; i < n; i++) {
    double u = x[i], v = y[i];
    int neg = (signbit(u) != 0) != (signbit(v) != 0);
    a->terms++;
    if (isnan(u) || isnan(v)) a->nan = 1;
    else if (isinf(u) || isinf(v)) {
      if (u == 0 || v == 0) a->nan = 1;   /* inf times 0 */
      else if (neg) a->ninf = 1; else a->pinf = 1;
    } else if (u == 0 || v == 0) zero_seen(a, neg);
    else {
      uint64_t mu, mv;
      int eu, ev;
      split(u, &mu, &eu);
      split(v, &mv, &ev);
      a->nonzero = 1;
      put(a, neg, (unsigned __int128)mu * mv, eu + ev);   /* exact: 106 bits */
    }
  }
}

void crsum_merge(crsum_acc *into, const crsum_acc *from)
{
  crsum_acc f = *from;
  settle(&f);
  settle(into);
  for (int i = 0; i < L; i++) into->limb[i] += f.limb[i];   /* each under 2^33 now */
  settle(into);
  into->nan |= f.nan;
  into->pinf |= f.pinf;
  into->ninf |= f.ninf;
  into->nonzero |= f.nonzero;
  into->poszero |= f.poszero;
  into->negzero |= f.negzero;
  into->terms += f.terms;
}

/* ---- rounding ---- */

/* bit P (weight 2^P) of a settled nonnegative accumulator */
static int bit(const crsum_acc *a, int P)
{
  int p = P - E0;
  if (p < 0 || p >= 32 * L) return 0;
  return (int)(a->limb[p >> 5] >> (p & 31) & 1);
}
/* any bit of weight below 2^P */
static int below(const crsum_acc *a, int P)
{
  int p = P - E0;
  if (p <= 0) return 0;
  if (p >= 32 * L) p = 32 * L;
  int i = p >> 5;
  for (int j = 0; j < i; j++) if (a->limb[j]) return 1;
  return i < L && (p & 31) && (a->limb[i] & ((1LL << (p & 31)) - 1)) != 0;
}

/* the value rounded to p bits, least quantum 2^qmin, overflow at 2^emax:
   the result as a double (exact: p <= 53) */
static double round_to(const crsum_acc *acc, int mode, int p, int qmin, int emax, double maxv)
{
  if (mode < 0 || mode > 3) return NAN;
  if (acc->nan || (acc->pinf && acc->ninf)) return NAN;
  if (acc->pinf) return INFINITY;
  if (acc->ninf) return -INFINITY;
  crsum_acc a = *acc;
  settle(&a);
  int neg = a.limb[L - 1] < 0;
  if (neg) {
    for (int i = 0; i < L; i++) a.limb[i] = -a.limb[i];
    settle(&a);
  }
  int h = L - 1;
  while (h >= 0 && !a.limb[h]) h--;
  if (h < 0) {   /* an exact zero: its sign as mpfr_sum and IEEE 754 give it */
    if (!acc->nonzero && acc->terms && !acc->poszero) return -0.0;   /* every term -0 */
    if (!acc->nonzero && !acc->negzero) return 0.0;                  /* every term +0, or none */
    return mode == CRSUM_DOWN ? -0.0 : 0.0;                         /* mixed zeros, or cancellation */
  }
  int E = E0 + 32 * h + 63 - __builtin_clzll((uint64_t)a.limb[h]);   /* the leading bit's weight */
  int q = E - p + 1;
  if (q < qmin) q = qmin;
  uint64_t M = 0;
  for (int P = E; P >= q; P--) M = M << 1 | (uint64_t)bit(&a, P);
  int r = bit(&a, q - 1), s = below(&a, q - 1), up;
  switch (mode) {
  case CRSUM_NEAREST: up = r && (s || (M & 1)); break;
  case CRSUM_UP: up = (r || s) && !neg; break;
  case CRSUM_DOWN: up = (r || s) && neg; break;
  default: up = 0;
  }
  M += (uint64_t)up;
  if (M >> p) { M >>= 1; q++; }   /* carried into the next binade (M 2^q is the same value either way:
                                     this keeps M to p bits, and no result depends on it) */
  double v;
  if (M && q + 64 - __builtin_clzll(M) > emax) {   /* 2^emax or more: overflow */
    int away = mode == CRSUM_NEAREST || (mode == CRSUM_UP && !neg) || (mode == CRSUM_DOWN && neg);
    v = away ? INFINITY : maxv;
  } else
    v = ldexp((double)M, q);   /* exact */
  return neg ? -v : v;
}

double crsum_round(const crsum_acc *a, int mode) { return round_to(a, mode, 53, -1074, 1024, 0x1.fffffffffffffp1023); }
float crsum_roundf(const crsum_acc *a, int mode) { return (float)round_to(a, mode, 24, -149, 128, 0x1.fffffep127); }

double crsum(const double *x, size_t n, int mode)
{
  crsum_acc a;
  crsum_init(&a);
  crsum_add(&a, x, n);
  return crsum_round(&a, mode);
}

double crdot(const double *x, const double *y, size_t n, int mode)
{
  crsum_acc a;
  crsum_init(&a);
  crsum_add_dot(&a, x, y, n);
  return crsum_round(&a, mode);
}

/* binary32: each value exactly as a binary64, in blocks */
float crsumf(const float *x, size_t n, int mode)
{
  crsum_acc a;
  crsum_init(&a);
  double buf[256];
  for (size_t i = 0; i < n; i += 256) {
    size_t k = n - i < 256 ? n - i : 256;
    for (size_t j = 0; j < k; j++) buf[j] = x[i + j];
    crsum_add(&a, buf, k);
  }
  return crsum_roundf(&a, mode);
}

float crdotf(const float *x, const float *y, size_t n, int mode)
{
  crsum_acc a;
  crsum_init(&a);
  double bx[256], by[256];
  for (size_t i = 0; i < n; i += 256) {
    size_t k = n - i < 256 ? n - i : 256;
    for (size_t j = 0; j < k; j++) { bx[j] = x[i + j]; by[j] = y[i + j]; }
    crsum_add_dot(&a, bx, by, k);
  }
  return crsum_roundf(&a, mode);
}
