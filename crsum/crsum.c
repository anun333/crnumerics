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
#include <stdlib.h>
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

/* ---- the fast path: bins ----
   A bin holds the sum of the integer significands of terms with one sign
   and one exponent, in a uint64; a bin that reaches 2^62 is moved into the
   limbs (put) and restarts, and at the end every bin is. Each addition is
   under 2^53, so a bin stays under 2^63. Same exact value, far fewer limb
   operations (Neal's "large superaccumulator", 2015). */
#define BIN_FULL (1ULL << 62)
#define FAST_MIN 64   /* shorter arrays take the direct path */

/* sums: a bin per sign and exponent field, the value's top 12 bits; NaN,
   infinities and zeros noted on the way (rare branches) */
static void add_binned(crsum_acc *a, const double *x, size_t n, uint64_t *bin)
{
  int nz = 0;
  for (size_t i = 0; i < n; i++) {
    uint64_t b;
    memcpy(&b, &x[i], 8);
    unsigned idx = (unsigned)(b >> 52), f = idx & 0x7ff;
    uint64_t m = b & ((1ULL << 52) - 1);
    if (__builtin_expect(f == 0x7ff, 0)) {
      if (m) a->nan = 1; else if (idx >> 11) a->ninf = 1; else a->pinf = 1;
      continue;
    }
    if (f) m |= 1ULL << 52;
    else if (__builtin_expect(!m, 0)) { zero_seen(a, (int)(idx >> 11)); continue; }
    nz = 1;
    uint64_t v = bin[idx] + m;
    if (__builtin_expect(v >= BIN_FULL, 0)) { put(a, (int)(idx >> 11), v, f ? (int)f - 1075 : -1074); v = 0; }
    bin[idx] = v;
  }
  if (nz) a->nonzero = 1;
  a->terms += n;
}
/* every bin into the limbs */
static void flush_binned(crsum_acc *a, uint64_t *bin)
{
  for (unsigned idx = 0; idx < 4096; idx++)
    if (bin[idx]) put(a, (int)(idx >> 11), bin[idx], (idx & 0x7ff) ? (int)(idx & 0x7ff) - 1075 : -1074);
}

/* dot products: the exact product m 2^e (m < 2^106) as two parts under
   2^53, at exponents e and e + 53; a bin per sign and exponent, e from
   -2148 to 1995; specials and zero products noted on the way. x and y are
   read with strides (a matrix's column), and the least and greatest e
   touched are kept in *elo, *ehi, so that emptying the bins visits only
   those, and leaves every bin zero for the next use (a matrix's next row). */
enum { DE0 = -2148, DNB = 1995 - DE0 + 1 };
static void add_dot_binned(crsum_acc *a, const double *x, size_t incx, const double *y, size_t incy, size_t n,
                           uint64_t *bin, int *elo, int *ehi)
{
  int nz = 0, lo_e = *elo, hi_e = *ehi;
  for (size_t i = 0; i < n; i++) {
    uint64_t bu, bv;
    memcpy(&bu, &x[i * incx], 8);
    memcpy(&bv, &y[i * incy], 8);
    unsigned fu = (unsigned)(bu >> 52) & 0x7ff, fv = (unsigned)(bv >> 52) & 0x7ff;
    int neg = (int)((bu ^ bv) >> 63);
    uint64_t mu = bu & ((1ULL << 52) - 1), mv = bv & ((1ULL << 52) - 1);
    if (__builtin_expect(fu == 0x7ff || fv == 0x7ff, 0)) {
      double u = x[i * incx], v = y[i * incy];
      if (isnan(u) || isnan(v) || u == 0 || v == 0) a->nan = 1;   /* a NaN, or inf times 0 */
      else if (neg) a->ninf = 1; else a->pinf = 1;
      continue;
    }
    int eu = fu ? (int)fu - 1075 : -1074, ev = fv ? (int)fv - 1075 : -1074;
    if (fu) mu |= 1ULL << 52;
    if (fv) mv |= 1ULL << 52;
    if (__builtin_expect(!mu || !mv, 0)) { zero_seen(a, neg); continue; }
    nz = 1;
    unsigned __int128 m = (unsigned __int128)mu * mv;
    uint64_t *b = bin + (neg ? DNB : 0);
    int e = eu + ev;
    if (e < lo_e) lo_e = e;
    if (e > hi_e) hi_e = e;
    uint64_t lo = (uint64_t)m & ((1ULL << 53) - 1), hi = (uint64_t)(m >> 53);
    uint64_t w = b[e - DE0] + lo;
    if (__builtin_expect(w >= BIN_FULL, 0)) { put(a, neg, w, e); w = 0; }
    b[e - DE0] = w;
    w = b[e + 53 - DE0] + hi;
    if (__builtin_expect(w >= BIN_FULL, 0)) { put(a, neg, w, e + 53); w = 0; }
    b[e + 53 - DE0] = w;
  }
  if (nz) a->nonzero = 1;
  a->terms += n;
  *elo = lo_e;
  *ehi = hi_e;
}
/* the bins with e in [elo, ehi + 53] into the limbs, and zeroed; the range
   reset to empty */
static void flush_dot_binned(crsum_acc *a, uint64_t *bin, int *elo, int *ehi)
{
  if (*elo <= *ehi)
    for (int s = 0; s < 2; s++)
      for (int k = *elo - DE0; k <= *ehi + 53 - DE0; k++) {
        uint64_t *b = &bin[s * DNB + k];
        if (*b) { put(a, s, *b, k + DE0); *b = 0; }
      }
  *elo = INT32_MAX;
  *ehi = INT32_MIN;
}

void crsum_add(crsum_acc *a, const double *x, size_t n)
{
  uint64_t *bin;
  if (n >= FAST_MIN && (bin = calloc(4096, sizeof *bin))) {
    add_binned(a, x, n, bin);
    flush_binned(a, bin);
    free(bin);
    return;
  }
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
  uint64_t *bin;
  if (n >= FAST_MIN && (bin = calloc(2 * DNB, sizeof *bin))) {
    int elo = INT32_MAX, ehi = INT32_MIN;
    add_dot_binned(a, x, 1, y, 1, n, bin, &elo, &ehi);
    flush_dot_binned(a, bin, &elo, &ehi);
    free(bin);
    return;
  }
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

/* binary32: each value exactly as a binary64, in blocks, into one set of
   bins (for more than FAST_MIN terms) emptied once at the end */
enum { BLK = 1024 };
float crsumf(const float *x, size_t n, int mode)
{
  crsum_acc a;
  crsum_init(&a);
  double buf[BLK];
  uint64_t *bin = n >= FAST_MIN ? calloc(4096, sizeof *bin) : NULL;
  for (size_t i = 0; i < n; i += BLK) {
    size_t k = n - i < BLK ? n - i : BLK;
    for (size_t j = 0; j < k; j++) buf[j] = x[i + j];
    if (bin) add_binned(&a, buf, k, bin); else crsum_add(&a, buf, k);
  }
  if (bin) { flush_binned(&a, bin); free(bin); }
  return crsum_roundf(&a, mode);
}

float crdotf(const float *x, const float *y, size_t n, int mode)
{
  crsum_acc a;
  crsum_init(&a);
  double bx[BLK], by[BLK];
  uint64_t *bin = n >= FAST_MIN ? calloc(2 * DNB, sizeof *bin) : NULL;
  int elo = INT32_MAX, ehi = INT32_MIN;
  for (size_t i = 0; i < n; i += BLK) {
    size_t k = n - i < BLK ? n - i : BLK;
    for (size_t j = 0; j < k; j++) { bx[j] = x[i + j]; by[j] = y[i + j]; }
    if (bin) add_dot_binned(&a, bx, 1, by, 1, k, bin, &elo, &ehi); else crsum_add_dot(&a, bx, by, k);
  }
  if (bin) { flush_dot_binned(&a, bin, &elo, &ehi); free(bin); }
  return crsum_roundf(&a, mode);
}

/* ---- matrix products: every element one exact dot product ---- */

/* into a (initialised): sum over j < n of u[j incu] v[j incv], plus beta *w
   (added exactly, as one more product; not read when beta is 0, as BLAS) */
static void element(crsum_acc *a, const double *u, size_t incu, const double *v, size_t incv, size_t n, double beta,
                    const double *w, uint64_t *bin, int *elo, int *ehi)
{
  if (bin && n >= FAST_MIN) {
    add_dot_binned(a, u, incu, v, incv, n, bin, elo, ehi);
    flush_dot_binned(a, bin, elo, ehi);
  } else
    for (size_t j = 0; j < n; j++) crsum_add_dot(a, &u[j * incu], &v[j * incv], 1);
  if (beta != 0) crsum_add_dot(a, &beta, w, 1);
}

int crgemv(int trans, size_t m, size_t n, const double *A, size_t lda, const double *x, double beta, double *y,
           int mode)
{
  if (mode < 0 || mode > 3) return -1;
  uint64_t *bin = calloc(2 * DNB, sizeof *bin);   /* NULL: the direct path */
  int elo = INT32_MAX, ehi = INT32_MIN;
  size_t outs = trans ? n : m;
  for (size_t i = 0; i < outs; i++) {
    crsum_acc a;
    crsum_init(&a);
    if (trans) element(&a, A + i, lda, x, 1, m, beta, &y[i], bin, &elo, &ehi);   /* column i */
    else element(&a, A + i * lda, 1, x, 1, n, beta, &y[i], bin, &elo, &ehi);     /* row i */
    y[i] = crsum_round(&a, mode);
  }
  free(bin);
  return 0;
}

int crgemm(size_t m, size_t n, size_t k, const double *A, size_t lda, const double *B, size_t ldb, double beta,
           double *C, size_t ldc, int mode)
{
  if (mode < 0 || mode > 3) return -1;
  uint64_t *bin = calloc(2 * DNB, sizeof *bin);
  double *col = malloc((k ? k : 1) * sizeof *col);
  int elo = INT32_MAX, ehi = INT32_MIN;
  for (size_t j = 0; j < n; j++) {
    if (col) for (size_t l = 0; l < k; l++) col[l] = B[l * ldb + j];   /* column j, contiguous */
    for (size_t i = 0; i < m; i++) {
      crsum_acc a;
      crsum_init(&a);
      if (col) element(&a, A + i * lda, 1, col, 1, k, beta, &C[i * ldc + j], bin, &elo, &ehi);
      else element(&a, A + i * lda, 1, B + j, ldb, k, beta, &C[i * ldc + j], bin, &elo, &ehi);
      C[i * ldc + j] = crsum_round(&a, mode);
    }
  }
  free(col);
  free(bin);
  return 0;
}

/* binary32: rows and columns converted to binary64 (exactly) in scratch */
int crgemvf(int trans, size_t m, size_t n, const float *A, size_t lda, const float *x, float beta, float *y, int mode)
{
  if (mode < 0 || mode > 3) return -1;
  size_t in = trans ? m : n, outs = trans ? n : m;
  double *xd = malloc((in ? in : 1) * sizeof *xd), *row = malloc((in ? in : 1) * sizeof *row);
  uint64_t *bin = calloc(2 * DNB, sizeof *bin);
  int elo = INT32_MAX, ehi = INT32_MIN;
  if (!xd || !row) { free(xd); free(row); free(bin); return -1; }
  for (size_t j = 0; j < in; j++) xd[j] = x[j];
  for (size_t i = 0; i < outs; i++) {
    for (size_t j = 0; j < in; j++) row[j] = trans ? A[j * lda + i] : A[i * lda + j];
    double bd = beta, yd = beta != 0 ? (double)y[i] : 0;
    crsum_acc a;
    crsum_init(&a);
    element(&a, row, 1, xd, 1, in, bd, &yd, bin, &elo, &ehi);
    y[i] = crsum_roundf(&a, mode);
  }
  free(xd);
  free(row);
  free(bin);
  return 0;
}

int crgemmf(size_t m, size_t n, size_t k, const float *A, size_t lda, const float *B, size_t ldb, float beta, float *C,
            size_t ldc, int mode)
{
  if (mode < 0 || mode > 3) return -1;
  size_t mk = m * k;
  double *Ad = malloc((mk ? mk : 1) * sizeof *Ad), *col = malloc((k ? k : 1) * sizeof *col);
  uint64_t *bin = calloc(2 * DNB, sizeof *bin);
  int elo = INT32_MAX, ehi = INT32_MIN;
  if (!Ad || !col) { free(Ad); free(col); free(bin); return -1; }
  for (size_t i = 0; i < m; i++)
    for (size_t l = 0; l < k; l++) Ad[i * k + l] = A[i * lda + l];
  for (size_t j = 0; j < n; j++) {
    for (size_t l = 0; l < k; l++) col[l] = B[l * ldb + j];
    for (size_t i = 0; i < m; i++) {
      double bd = beta, cd = beta != 0 ? (double)C[i * ldc + j] : 0;
      crsum_acc a;
      crsum_init(&a);
      element(&a, Ad + i * k, 1, col, 1, k, bd, &cd, bin, &elo, &ehi);
      C[i * ldc + j] = crsum_roundf(&a, mode);
    }
  }
  free(Ad);
  free(col);
  free(bin);
  return 0;
}

/* ---- the Ozaki scheme: exact products through any binary64 GEMM ----
   Each row of A (column of B) is scaled by its lowest set bit, 2^qa_i
   (2^qb_j), so that its entries are integers, and cut into slices of w
   bits: a_il = sum_p A_p[i][l] 2^(w p + qa_i), |A_p[i][l]| < 2^w. With
   w = floor((53 - ceil(log2 k)) / 2), every product of two slice entries,
   and every partial sum of k of them, is an integer below 2^53, so any
   binary64 GEMM computes A_p B_q exactly: whatever its order, with or
   without fused multiply-adds, threaded or not. Then
     C_ij = sum_d D_d[i][j] 2^(w d + qa_i + qb_j),   D_d = sum_{p+q=d} A_p B_q,
   with D_d exact in int64 (at most 8 terms under 2^53), goes into the
   accumulator and is rounded once: the same bits as crgemm. An element
   whose exact value is zero is recomputed directly, for its zero's sign
   (which the slices can't see). NaN or infinities, or a range needing more
   than 64 slice products, fall back to crgemm. Ozaki, Ogita, Oishi and
   Rump (2012) introduced the splitting; the slices' width here makes it
   exact rather than accurate. */

/* bits [lo, lo + len) of m, len < 64 (lo may be negative: zeros below) */
static uint64_t bits_of(uint64_t m, int lo, int len)
{
  if (lo >= 64 || lo + len <= 0) return 0;
  uint64_t v = lo >= 0 ? m >> lo : m << -lo;
  return v & ((1ULL << len) - 1);
}

/* the internal GEMM: C = A B, row-major, for the slices (integer values) */
static void gemm_internal(size_t m, size_t n, size_t k, const double *A, size_t lda, const double *B, size_t ldb,
                          double *C, size_t ldc, void *ctx)
{
  (void)ctx;
  for (size_t i = 0; i < m; i++) {
    double *c = C + i * ldc;
    for (size_t j = 0; j < n; j++) c[j] = 0;
    for (size_t l = 0; l < k; l++) {
      double a = A[i * lda + l];
      if (a == 0) continue;
      const double *b = B + l * ldb;
      for (size_t j = 0; j < n; j++) c[j] += a * b[j];
    }
  }
}

/* the least exponent (lowest set bit) and the top bit's exponent of the
   nonzero values among v[0], v[inc], ... (n of them); 0 if none: 1 when a
   NaN or an infinity is among them */
static int scale_of(const double *v, size_t inc, size_t n, int *q, int *top, int *any)
{
  *any = 0;
  for (size_t l = 0; l < n; l++) {
    double x = v[l * inc];
    if (x == 0) continue;
    if (!isfinite(x)) return 1;
    uint64_t m;
    int e;
    split(x, &m, &e);
    int lo = e + __builtin_ctzll(m), hi = e + 63 - __builtin_clzll(m);
    if (!*any || lo < *q) *q = lo;
    if (!*any || hi > *top) *top = hi;
    *any = 1;
  }
  return 0;
}

/* the compact path (beta = 0): an element's exact value as two unsigned
   multi-word integers, positive and negative parts, little-endian 64-bit
   words, then their difference rounded directly. */
static void add_at(uint64_t *v, int nw, uint64_t d, int shift)   /* v += d 2^shift */
{
  int i = shift >> 6, s = shift & 63;
  unsigned __int128 t = (unsigned __int128)d << s;
  uint64_t lo = (uint64_t)t, hi = (uint64_t)(t >> 64), c;
  c = (v[i] += lo) < lo;
  hi += c;   /* hi < 2^63: no overflow */
  for (int j = i + 1; j < nw && (hi || c); j++) {
    c = (v[j] += hi) < hi;
    hi = c;
  }
}
static int bit_w(const uint64_t *v, int nw, int i) { return i < 0 || i >= 64 * nw ? 0 : (int)(v[i >> 6] >> (i & 63) & 1); }
static int below_w(const uint64_t *v, int nw, int i)   /* any bit under index i */
{
  if (i <= 0) return 0;
  if (i > 64 * nw) i = 64 * nw;
  for (int j = 0; j < (i >> 6); j++) if (v[j]) return 1;
  return (i & 63) && (i >> 6) < nw && (v[i >> 6] & ((1ULL << (i & 63)) - 1)) != 0;
}
/* (-1)^neg v 2^e0, v nonzero, correctly rounded to binary64 */
static double round_words(const uint64_t *v, int nw, int neg, int e0, int mode)
{
  int h = nw - 1;
  while (!v[h]) h--;
  int top = 64 * h + 63 - __builtin_clzll(v[h]), E = top + e0, q = E - 52;
  if (q < -1074) q = -1074;
  int qi = q - e0;   /* the quantum's bit index in v (negative: below v's last bit) */
  uint64_t M = 0;
  for (int i = top; i >= qi; i--) M = M << 1 | (uint64_t)bit_w(v, nw, i);
  int r = bit_w(v, nw, qi - 1), s = below_w(v, nw, qi - 1), up;
  switch (mode) {
  case CRSUM_NEAREST: up = r && (s || (M & 1)); break;
  case CRSUM_UP: up = (r || s) && !neg; break;
  case CRSUM_DOWN: up = (r || s) && neg; break;
  default: up = 0;
  }
  M += (uint64_t)up;
  if (M >> 53) { M >>= 1; q++; }
  double x;
  if (q + 64 - __builtin_clzll(M) > 1024) {
    int away = mode == CRSUM_NEAREST || (mode == CRSUM_UP && !neg) || (mode == CRSUM_DOWN && neg);
    x = away ? INFINITY : 0x1.fffffffffffffp1023;
  } else
    x = ldexp((double)M, q);
  return neg ? -x : x;
}

/* the Ozaki schemes' last step, shared by crgemm_oz and crgemm_oz8: the
   slice products' sums D_d (at bit w d above the rows' and columns' least
   exponents qa, qb) added exactly per element and rounded once, beta C
   included; an element whose exact value is zero recomputed directly, for
   its sign */
static void oz_finish(size_t m, size_t n, size_t k, const double *A, size_t lda, const double *B, size_t ldb, double beta,
                      double *C, size_t ldc, int mode, int w, int nd, const int64_t *D, const int *qa, const int *qb,
                      const int *anya, const int *anyb)
{
  int nw = (w * (nd - 1) + 64) / 64 + 2;   /* D_d < 2^57 at bit w d: the words that hold the sum */
  uint64_t *vp = malloc(2 * (size_t)nw * sizeof *vp);
  for (size_t i = 0; i < m; i++)
    for (size_t j = 0; j < n; j++) {
      if (beta == 0 && vp && anya[i] && anyb[j]) {   /* the compact path */
        uint64_t *pos = vp, *neg = vp + nw;
        memset(vp, 0, 2 * (size_t)nw * sizeof *vp);
        for (int d = 0; d < nd; d++) {
          int64_t v = D[((size_t)d * m + i) * n + j];
          if (v > 0) add_at(pos, nw, (uint64_t)v, w * d);
          else if (v < 0) add_at(neg, nw, -(uint64_t)v, w * d);
        }
        int cmp = 0;   /* pos against neg */
        for (int t = nw - 1; t >= 0 && !cmp; t--) cmp = pos[t] > neg[t] ? 1 : pos[t] < neg[t] ? -1 : 0;
        if (cmp) {
          uint64_t *big = cmp > 0 ? pos : neg, *small = cmp > 0 ? neg : pos, borrow = 0;
          for (int t = 0; t < nw; t++) {   /* big -= small, in 128 bits: no edge case */
            unsigned __int128 d = (unsigned __int128)big[t] - small[t] - borrow;
            big[t] = (uint64_t)d;
            borrow = (uint64_t)(d >> 64) & 1;
          }
          C[i * ldc + j] = round_words(big, nw, cmp < 0, qa[i] + qb[j], mode);
          continue;
        }   /* an exact zero: below, for its sign */
      }
      crsum_acc a;
      crsum_init(&a);
      int nonzero = 0;
      if (anya[i] && anyb[j])
        for (int d = 0; d < nd; d++) {
          int64_t v = D[((size_t)d * m + i) * n + j];
          if (!v) continue;
          nonzero = 1;
          put(&a, v < 0, (unsigned __int128)(v < 0 ? -(uint64_t)v : (uint64_t)v), w * d + qa[i] + qb[j]);
        }
      if (!nonzero) {   /* an exact zero from the products: directly, for its sign */
        crsum_init(&a);
        element(&a, A + i * lda, 1, B + j, ldb, k, 0, NULL, NULL, NULL, NULL);
      } else
        a.nonzero = 1;
      if (beta != 0) crsum_add_dot(&a, &beta, &C[i * ldc + j], 1);
      C[i * ldc + j] = crsum_round(&a, mode);
    }
  free(vp);
}

int crgemm_oz(size_t m, size_t n, size_t k, const double *A, size_t lda, const double *B, size_t ldb, double beta,
              double *C, size_t ldc, int mode, crsum_dgemm gemm, void *ctx)
{
  if (mode < 0 || mode > 3) return -1;
  if (!gemm) gemm = gemm_internal;
  int lk = 0;
  while (lk < 63 && ((size_t)1 << lk) < k) lk++;
  int w = (53 - lk) / 2;
  if (!m || !n || !k || w < 1) return crgemm(m, n, k, A, lda, B, ldb, beta, C, ldc, mode);
  int *qa = calloc(m, sizeof *qa), *qb = calloc(n, sizeof *qb), *anya = calloc(m, sizeof *anya), *anyb = calloc(n, sizeof *anyb);
  int WA = 0, WB = 0, bad = !qa || !qb || !anya || !anyb;
  for (size_t i = 0; i < m && !bad; i++) {
    int top = 0;
    bad = scale_of(A + i * lda, 1, k, &qa[i], &top, &anya[i]);
    if (anya[i] && top - qa[i] + 1 > WA) WA = top - qa[i] + 1;
  }
  for (size_t j = 0; j < n && !bad; j++) {
    int top = 0;
    bad = scale_of(B + j, ldb, k, &qb[j], &top, &anyb[j]);
    if (anyb[j] && top - qb[j] + 1 > WB) WB = top - qb[j] + 1;
  }
  int sa = (WA + w - 1) / w, sb = (WB + w - 1) / w, nd = sa + sb - 1;
  if (bad || !sa || !sb || sa * sb > 64) {
    free(qa); free(qb); free(anya); free(anyb);
    return crgemm(m, n, k, A, lda, B, ldb, beta, C, ldc, mode);   /* specials, all zero, or too wide */
  }
  double *As = malloc((size_t)sa * m * k * sizeof *As), *Bs = malloc((size_t)sb * k * n * sizeof *Bs);
  double *P = malloc(m * n * sizeof *P);
  int64_t *D = calloc((size_t)nd * m * n, sizeof *D);
  int *za = calloc((size_t)sa, sizeof *za), *zb = calloc((size_t)sb, sizeof *zb);   /* a slice with a nonzero entry */
  if (!As || !Bs || !P || !D || !za || !zb) {
    free(As); free(Bs); free(P); free(D); free(za); free(zb); free(qa); free(qb); free(anya); free(anyb);
    return crgemm(m, n, k, A, lda, B, ldb, beta, C, ldc, mode);
  }
  for (size_t i = 0; i < m; i++)
    for (size_t l = 0; l < k; l++) {
      double x = A[i * lda + l];
      uint64_t mm = 0;
      int e = 0;
      if (x != 0) split(x, &mm, &e);
      for (int p = 0; p < sa; p++) {
        double d = (double)bits_of(mm, w * p - (e - qa[i]), w);
        if (d != 0) za[p] = 1;
        As[((size_t)p * m + i) * k + l] = signbit(x) ? -d : d;
      }
    }
  for (size_t l = 0; l < k; l++)
    for (size_t j = 0; j < n; j++) {
      double x = B[l * ldb + j];
      uint64_t mm = 0;
      int e = 0;
      if (x != 0) split(x, &mm, &e);
      for (int q = 0; q < sb; q++) {
        double d = (double)bits_of(mm, w * q - (e - qb[j]), w);
        if (d != 0) zb[q] = 1;
        Bs[((size_t)q * k + l) * n + j] = signbit(x) ? -d : d;
      }
    }
  for (int p = 0; p < sa; p++)
    for (int q = 0; q < sb; q++) {
      if (!za[p] || !zb[q]) continue;   /* a slice of zeros: nothing to add */
      gemm(m, n, k, As + (size_t)p * m * k, k, Bs + (size_t)q * k * n, n, P, n, ctx);
      int64_t *Dd = D + (size_t)(p + q) * m * n;
      for (size_t t = 0; t < m * n; t++) Dd[t] += (int64_t)P[t];   /* exact: an integer under 2^53 */
    }
  oz_finish(m, n, k, A, lda, B, ldb, beta, C, ldc, mode, w, nd, D, qa, qb, anya, anyb);
  free(As); free(Bs); free(P); free(D); free(za); free(zb); free(qa); free(qb); free(anya); free(anyb);
  return 0;
}

/* crgemm_oz8 (2026-10-01): the Ozaki scheme on int8 dot-product units.
   The same scales as crgemm_oz, slices of w = 7 bits with the sign applied
   (int8), and the products in int32: exact for any k up to 133,143 at 127 x
   127 (32768 per chunk here, which also covers VNNI's unsigned-by-signed
   form, 255 x 127). The chunks' int32 sums are added in int64, then
   oz_finish as for crgemm_oz. */
#define OZ8_KC 32768
#define OZ8_MAXPROD 400

static void i8_plain(size_t m, size_t n, size_t k, const int8_t *A, size_t lda, const int8_t *Bt, size_t ldbt, int32_t *C,
                     size_t ldc)
{
  for (size_t i = 0; i < m; i++)
    for (size_t j = 0; j < n; j++) {
      const int8_t *a = A + i * lda, *b = Bt + j * ldbt;
      int32_t s = 0;
      for (size_t l = 0; l < k; l++) s += (int32_t)a[l] * b[l];
      C[i * ldc + j] = s;
    }
}
#if defined(__x86_64__)
#include <immintrin.h>
/* AVX512-VNNI: vpdpbusd multiplies unsigned bytes by signed ones, so A's
   bytes go in shifted by 128 and 128 times the column's sum comes off:
   sum (a + 128) b = sum a b + 128 sum b, every term exact in int32.
   Blocked as the AVX2 kernel (2026-10-01): two rows of A against four rows
   of Bt, each loaded vector used two or four times; the masked tail loads
   zero bytes of b, which cancel the bias's lanes. */
__attribute__((target("avx512f,avx512bw,avx512vnni"))) static void i8_vnni(size_t m, size_t n, size_t k, const int8_t *A,
                                                                         size_t lda, const int8_t *Bt, size_t ldbt,
                                                                         int32_t *C, size_t ldc)
{
  const __m512i bias = _mm512_set1_epi8((char)0x80);
  int32_t *sb = malloc(sizeof(int32_t) * (n ? n : 1));
  for (size_t j = 0; j < n; j++) {
    const int8_t *b = Bt + j * ldbt; int32_t t = 0;
    for (size_t l = 0; l < k; l++) t += b[l];
    sb[j] = t;
  }
  size_t k64 = k & ~(size_t)63;
  __mmask64 mk = k > k64 ? (__mmask64)(~0ULL >> (64 - (k - k64))) : 0;
  for (size_t i = 0; i < m; i += 2) {
    size_t mi = m - i < 2 ? m - i : 2;
    const int8_t *a0 = A + i * lda, *a1 = mi > 1 ? a0 + lda : a0;
    for (size_t j = 0; j < n; j += 4) {
      size_t nj = n - j < 4 ? n - j : 4;
      const int8_t *b0 = Bt + j * ldbt, *b1 = nj > 1 ? b0 + ldbt : b0, *b2 = nj > 2 ? b0 + 2 * ldbt : b0,
                   *b3 = nj > 3 ? b0 + 3 * ldbt : b0;
      __m512i c00 = _mm512_setzero_si512(), c01 = c00, c02 = c00, c03 = c00, c10 = c00, c11 = c00, c12 = c00, c13 = c00;
#define I8V_STEP(LA, LB)                                                                                              \
  {                                                                                                                   \
    __m512i x0 = _mm512_xor_si512(LA(a0), bias), x1 = _mm512_xor_si512(LA(a1), bias), y;                              \
    y = LB(b0); c00 = _mm512_dpbusd_epi32(c00, x0, y); c10 = _mm512_dpbusd_epi32(c10, x1, y);                         \
    y = LB(b1); c01 = _mm512_dpbusd_epi32(c01, x0, y); c11 = _mm512_dpbusd_epi32(c11, x1, y);                         \
    y = LB(b2); c02 = _mm512_dpbusd_epi32(c02, x0, y); c12 = _mm512_dpbusd_epi32(c12, x1, y);                         \
    y = LB(b3); c03 = _mm512_dpbusd_epi32(c03, x0, y); c13 = _mm512_dpbusd_epi32(c13, x1, y);                         \
  }
#define I8V_LD(p) _mm512_loadu_si512((p) + l)
#define I8V_MLD(p) _mm512_maskz_loadu_epi8(mk, (p) + l)
      size_t l = 0;
      for (; l < k64; l += 64) I8V_STEP(I8V_LD, I8V_LD)
      if (mk) I8V_STEP(I8V_MLD, I8V_MLD)
#undef I8V_STEP
#undef I8V_LD
#undef I8V_MLD
      int32_t s[2][4] = {{_mm512_reduce_add_epi32(c00), _mm512_reduce_add_epi32(c01), _mm512_reduce_add_epi32(c02),
                          _mm512_reduce_add_epi32(c03)},
                         {_mm512_reduce_add_epi32(c10), _mm512_reduce_add_epi32(c11), _mm512_reduce_add_epi32(c12),
                          _mm512_reduce_add_epi32(c13)}};
      for (size_t r = 0; r < mi; r++)
        for (size_t t = 0; t < nj; t++) C[(i + r) * ldc + j + t] = s[r][t] - 128 * sb[j + t];
    }
  }
  free(sb);
}
/* AVX2, for x86 without VNNI (2026-10-01): bytes sign-extended to int16
   (vpmovsxbw) and multiplied in pairs into int32 (vpmaddwd), each product
   at most 127 x 127 and each pair 32,258, so exact (vpmaddubsw would
   saturate). Blocked: two rows of A against four rows of Bt, so each
   widened vector serves two or four outputs; an edge block repeats a row,
   and only the outputs that exist are written. */
#define I8X_HSUM(v) ({ __m128i h_ = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1)); \
                       h_ = _mm_add_epi32(h_, _mm_shuffle_epi32(h_, 0x4e)); h_ = _mm_add_epi32(h_, _mm_shuffle_epi32(h_, 0xb1)); \
                       _mm_cvtsi128_si32(h_); })
__attribute__((target("avx2"))) static void i8_avx2(size_t m, size_t n, size_t k, const int8_t *A, size_t lda,
                                                   const int8_t *Bt, size_t ldbt, int32_t *C, size_t ldc)
{
  size_t k16 = k & ~(size_t)15;
  for (size_t i = 0; i < m; i += 2) {
    size_t mi = m - i < 2 ? m - i : 2;
    const int8_t *a0 = A + i * lda, *a1 = mi > 1 ? a0 + lda : a0;
    for (size_t j = 0; j < n; j += 4) {
      size_t nj = n - j < 4 ? n - j : 4;
      const int8_t *b0 = Bt + j * ldbt, *b1 = nj > 1 ? b0 + ldbt : b0, *b2 = nj > 2 ? b0 + 2 * ldbt : b0,
                   *b3 = nj > 3 ? b0 + 3 * ldbt : b0;
      __m256i c00 = _mm256_setzero_si256(), c01 = c00, c02 = c00, c03 = c00, c10 = c00, c11 = c00, c12 = c00, c13 = c00;
      for (size_t l = 0; l < k16; l += 16) {
#define I8X_LD(p) _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)((p) + l)))
        __m256i x0 = I8X_LD(a0), x1 = I8X_LD(a1), y;
        y = I8X_LD(b0); c00 = _mm256_add_epi32(c00, _mm256_madd_epi16(x0, y)); c10 = _mm256_add_epi32(c10, _mm256_madd_epi16(x1, y));
        y = I8X_LD(b1); c01 = _mm256_add_epi32(c01, _mm256_madd_epi16(x0, y)); c11 = _mm256_add_epi32(c11, _mm256_madd_epi16(x1, y));
        y = I8X_LD(b2); c02 = _mm256_add_epi32(c02, _mm256_madd_epi16(x0, y)); c12 = _mm256_add_epi32(c12, _mm256_madd_epi16(x1, y));
        y = I8X_LD(b3); c03 = _mm256_add_epi32(c03, _mm256_madd_epi16(x0, y)); c13 = _mm256_add_epi32(c13, _mm256_madd_epi16(x1, y));
#undef I8X_LD
      }
      int32_t s[2][4] = {{I8X_HSUM(c00), I8X_HSUM(c01), I8X_HSUM(c02), I8X_HSUM(c03)},
                         {I8X_HSUM(c10), I8X_HSUM(c11), I8X_HSUM(c12), I8X_HSUM(c13)}};
      const int8_t *ar[2] = {a0, a1}, *bc[4] = {b0, b1, b2, b3};
      for (size_t r = 0; r < mi; r++)
        for (size_t t = 0; t < nj; t++) {
          int32_t v = s[r][t];
          for (size_t l = k16; l < k; l++) v += (int32_t)ar[r][l] * bc[t][l];
          C[(i + r) * ldc + j + t] = v;
        }
    }
  }
}
#undef I8X_HSUM
#endif
#if defined(__aarch64__)
#include <arm_neon.h>
#include <sys/auxv.h>
/* Arm SDOT: signed by signed bytes into int32 lanes */
__attribute__((target("+dotprod"))) static void i8_sdot(size_t m, size_t n, size_t k, const int8_t *A, size_t lda,
                                                       const int8_t *Bt, size_t ldbt, int32_t *C, size_t ldc)
{
  for (size_t i = 0; i < m; i++)
    for (size_t j = 0; j < n; j++) {
      const int8_t *a = A + i * lda, *b = Bt + j * ldbt;
      int32x4_t acc = vdupq_n_s32(0);
      size_t l = 0;
      for (; l + 16 <= k; l += 16) acc = vdotq_s32(acc, vld1q_s8(a + l), vld1q_s8(b + l));
      int32_t s = vaddvq_s32(acc);
      for (; l < k; l++) s += (int32_t)a[l] * b[l];
      C[i * ldc + j] = s;
    }
}
#endif
static int i8_which = -1;   /* 0 plain, 1 vnni, 2 sdot, 3 avx2 */
static void i8_pick(void)
{
  if (i8_which >= 0) return;
  int have[4] = {1, 0, 0, 0};
#if defined(__x86_64__)
  have[1] = __builtin_cpu_supports("avx512vnni") && __builtin_cpu_supports("avx512bw");
  have[3] = __builtin_cpu_supports("avx2");
#endif
#if defined(__aarch64__) && defined(HWCAP_ASIMDDP)
  have[2] = (getauxval(AT_HWCAP) & HWCAP_ASIMDDP) != 0;
#endif
  i8_which = have[1] ? 1 : have[2] ? 2 : have[3] ? 3 : 0;
  /* for checks and timing: CRSUM_I8_KERNEL names one (if the CPU has it);
     CRSUM_I8_PLAIN=1, the older switch, means plain */
  const char *e = getenv("CRSUM_I8_PLAIN"), *kn = getenv("CRSUM_I8_KERNEL");
  if (e && *e == '1') i8_which = 0;
  static const char *NM[4] = {"plain", "vnni", "sdot", "avx2"};
  if (kn) for (int w = 0; w < 4; w++) if (!strcmp(kn, NM[w]) && have[w]) i8_which = w;
}
const char *crsum_i8_kernel(void)
{
  i8_pick();
  return i8_which == 1 ? "vnni" : i8_which == 2 ? "sdot" : i8_which == 3 ? "avx2" : "plain";
}
static void i8_internal(size_t m, size_t n, size_t k, const int8_t *A, size_t lda, const int8_t *Bt, size_t ldbt, int32_t *C,
                        size_t ldc, void *ctx)
{
  (void)ctx;
  i8_pick();
#if defined(__x86_64__)
  if (i8_which == 1) { i8_vnni(m, n, k, A, lda, Bt, ldbt, C, ldc); return; }
  if (i8_which == 3) { i8_avx2(m, n, k, A, lda, Bt, ldbt, C, ldc); return; }
#endif
#if defined(__aarch64__)
  if (i8_which == 2) { i8_sdot(m, n, k, A, lda, Bt, ldbt, C, ldc); return; }
#endif
  i8_plain(m, n, k, A, lda, Bt, ldbt, C, ldc);
}

int crgemm_oz8(size_t m, size_t n, size_t k, const double *A, size_t lda, const double *B, size_t ldb, double beta,
               double *C, size_t ldc, int mode, crsum_i8gemm gemm, void *ctx)
{
  if (mode < 0 || mode > 3) return -1;
  if (!gemm) gemm = i8_internal;
  const int w = 7;
  if (!m || !n || !k) return crgemm(m, n, k, A, lda, B, ldb, beta, C, ldc, mode);
  int *qa = calloc(m, sizeof *qa), *qb = calloc(n, sizeof *qb), *anya = calloc(m, sizeof *anya), *anyb = calloc(n, sizeof *anyb);
  int WA = 0, WB = 0, bad = !qa || !qb || !anya || !anyb;
  for (size_t i = 0; i < m && !bad; i++) {
    int top = 0;
    bad = scale_of(A + i * lda, 1, k, &qa[i], &top, &anya[i]);
    if (anya[i] && top - qa[i] + 1 > WA) WA = top - qa[i] + 1;
  }
  for (size_t j = 0; j < n && !bad; j++) {
    int top = 0;
    bad = scale_of(B + j, ldb, k, &qb[j], &top, &anyb[j]);
    if (anyb[j] && top - qb[j] + 1 > WB) WB = top - qb[j] + 1;
  }
  int sa = (WA + w - 1) / w, sb = (WB + w - 1) / w, nd = sa + sb - 1;
  if (bad || !sa || !sb || sa * sb > OZ8_MAXPROD) {
    free(qa); free(qb); free(anya); free(anyb);
    return crgemm(m, n, k, A, lda, B, ldb, beta, C, ldc, mode);   /* specials, all zero, or too wide */
  }
  int8_t *As = malloc((size_t)sa * m * k), *Bt = malloc((size_t)sb * n * k);
  int32_t *P = malloc(m * n * sizeof *P);
  int64_t *D = calloc((size_t)nd * m * n, sizeof *D);
  int *za = calloc((size_t)sa, sizeof *za), *zb = calloc((size_t)sb, sizeof *zb);
  if (!As || !Bt || !P || !D || !za || !zb) {
    free(As); free(Bt); free(P); free(D); free(za); free(zb); free(qa); free(qb); free(anya); free(anyb);
    return crgemm(m, n, k, A, lda, B, ldb, beta, C, ldc, mode);
  }
  for (size_t i = 0; i < m; i++)
    for (size_t l = 0; l < k; l++) {
      double x = A[i * lda + l];
      uint64_t mm = 0;
      int e = 0;
      if (x != 0) split(x, &mm, &e);
      for (int p = 0; p < sa; p++) {
        int d = (int)bits_of(mm, w * p - (e - qa[i]), w);
        if (d) za[p] = 1;
        As[((size_t)p * m + i) * k + l] = (int8_t)(signbit(x) ? -d : d);
      }
    }
  for (size_t l = 0; l < k; l++)
    for (size_t j = 0; j < n; j++) {
      double x = B[l * ldb + j];
      uint64_t mm = 0;
      int e = 0;
      if (x != 0) split(x, &mm, &e);
      for (int q = 0; q < sb; q++) {
        int d = (int)bits_of(mm, w * q - (e - qb[j]), w);
        if (d) zb[q] = 1;
        Bt[((size_t)q * n + j) * k + l] = (int8_t)(signbit(x) ? -d : d);   /* transposed: B's columns as rows */
      }
    }
  for (int p = 0; p < sa; p++)
    for (int q = 0; q < sb; q++) {
      if (!za[p] || !zb[q]) continue;
      int64_t *Dd = D + (size_t)(p + q) * m * n;
      for (size_t l0 = 0; l0 < k; l0 += OZ8_KC) {
        size_t kc = k - l0 < OZ8_KC ? k - l0 : OZ8_KC;
        gemm(m, n, kc, As + (size_t)p * m * k + l0, k, Bt + (size_t)q * n * k + l0, k, P, n, ctx);
        for (size_t t = 0; t < m * n; t++) Dd[t] += P[t];   /* exact: int32 into int64 */
      }
    }
  oz_finish(m, n, k, A, lda, B, ldb, beta, C, ldc, mode, w, nd, D, qa, qb, anya, anyb);
  free(As); free(Bt); free(P); free(D); free(za); free(zb); free(qa); free(qb); free(anya); free(anyb);
  return 0;
}
