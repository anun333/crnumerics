/* arith-bench.c: the cost of ival's interval arithmetic (ival-arith.c), in ns per interval, against the alternatives.

   For add, mul and div, on n intervals at a time:
     - rn: the ends rounded to nearest, no error term. Not an enclosure; the floor any method pays.
     - ival: ival_add etc.; where the CPU has AVX2 and FMA, the four-lane passes.
     - portable: the same calls with the portable passes (plain C, the rounding mode set by fesetround).
     - eft: the same calls with the scalar code alone, by error-free transformations rounding to nearest.
     - bare (x86-64 with AVX2 and FMA): the rounding mode set per block of 256 intervals, four lanes at a time, with
       none of ival's care for empty intervals, 0 * inf, zero signs or divisors holding 0: what the method costs
       alone. On these inputs its results must equal ival's bit for bit, or the timing is void.
   The inputs are normal numbers of moderate size, any sign; for div, divisors that do not hold 0 (the common case,
   and the only one bare handles). Each figure is the least of 7 passes after a warm-up, each pass long enough to
   take at least 20 ms. A figure below 0.05 ns is printed FOLDED: the work was optimized away. */
#include <fenv.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#if defined(__x86_64__)
#include <immintrin.h>
#endif
#include "ival.h"
extern int ival__arith_path;

static uint64_t rs = 0x9e3779b97f4a7c15ULL;
static uint64_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }
static double unit(void) { return (double)(rnd() >> 11) * 0x1p-53; }
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }

#if defined(__x86_64__)
#define TGT __attribute__((target("avx2,fma"), noinline))
#else
#define TGT __attribute__((noinline))
#endif
#define ARGS const double *al, const double *ah, const double *bl, const double *bh, double *zl, double *zh, size_t n
static inline double mn(double a, double b) { return a < b ? a : b; }
static inline double mx(double a, double b) { return a > b ? a : b; }
TGT static void rn_add(ARGS) { for (size_t i = 0; i < n; i++) { zl[i] = al[i] + bl[i]; zh[i] = ah[i] + bh[i]; } }
TGT static void rn_mul(ARGS)
{
  for (size_t i = 0; i < n; i++) {
    double p0 = al[i] * bl[i], p1 = al[i] * bh[i], p2 = ah[i] * bl[i], p3 = ah[i] * bh[i];
    zl[i] = mn(mn(p0, p1), mn(p2, p3)); zh[i] = mx(mx(p0, p1), mx(p2, p3));
  }
}
TGT static void rn_div(ARGS)
{
  for (size_t i = 0; i < n; i++) {
    double q0 = al[i] / bl[i], q1 = al[i] / bh[i], q2 = ah[i] / bl[i], q3 = ah[i] / bh[i];
    zl[i] = mn(mn(q0, q1), mn(q2, q3)); zh[i] = mx(mx(q0, q1), mx(q2, q3));
  }
}

#if defined(__x86_64__)
/* directed: one pass of lower ends, one of upper ends, per block; each pass is its own call, after the mode is set */
TGT static void d_add_lo(ARGS) { for (size_t i = 0; i < n; i += 4) _mm256_storeu_pd(zl + i, _mm256_add_pd(_mm256_loadu_pd(al + i), _mm256_loadu_pd(bl + i))); (void)ah; (void)bh; (void)zh; }
TGT static void d_add_hi(ARGS) { for (size_t i = 0; i < n; i += 4) _mm256_storeu_pd(zh + i, _mm256_add_pd(_mm256_loadu_pd(ah + i), _mm256_loadu_pd(bh + i))); (void)al; (void)bl; (void)zl; }
TGT static void d_mul_lo(ARGS)
{
  for (size_t i = 0; i < n; i += 4) {
    __m256d a = _mm256_loadu_pd(al + i), b = _mm256_loadu_pd(ah + i), c = _mm256_loadu_pd(bl + i), d = _mm256_loadu_pd(bh + i);
    _mm256_storeu_pd(zl + i, _mm256_min_pd(_mm256_min_pd(_mm256_mul_pd(a, c), _mm256_mul_pd(a, d)), _mm256_min_pd(_mm256_mul_pd(b, c), _mm256_mul_pd(b, d))));
  }
  (void)zh;
}
TGT static void d_mul_hi(ARGS)
{
  for (size_t i = 0; i < n; i += 4) {
    __m256d a = _mm256_loadu_pd(al + i), b = _mm256_loadu_pd(ah + i), c = _mm256_loadu_pd(bl + i), d = _mm256_loadu_pd(bh + i);
    _mm256_storeu_pd(zh + i, _mm256_max_pd(_mm256_max_pd(_mm256_mul_pd(a, c), _mm256_mul_pd(a, d)), _mm256_max_pd(_mm256_mul_pd(b, c), _mm256_mul_pd(b, d))));
  }
  (void)zl;
}
/* 1788's table for a divisor without 0 (ival-arith.c, vdiv1): the numerator and divisor of each end by blends */
TGT static void d_div_lo(ARGS)
{
  const __m256d z = _mm256_setzero_pd();
  for (size_t i = 0; i < n; i += 4) {
    __m256d a = _mm256_loadu_pd(al + i), b = _mm256_loadu_pd(ah + i), c = _mm256_loadu_pd(bl + i), d = _mm256_loadu_pd(bh + i);
    __m256d P = _mm256_cmp_pd(c, z, _CMP_GT_OQ), ge = _mm256_cmp_pd(a, z, _CMP_GE_OQ), le = _mm256_andnot_pd(ge, _mm256_cmp_pd(b, z, _CMP_LE_OQ));
    __m256d num = _mm256_blendv_pd(b, a, P), den = _mm256_blendv_pd(_mm256_blendv_pd(d, c, le), _mm256_blendv_pd(c, d, ge), P);
    _mm256_storeu_pd(zl + i, _mm256_div_pd(num, den));
  }
  (void)zh;
}
TGT static void d_div_hi(ARGS)
{
  const __m256d z = _mm256_setzero_pd();
  for (size_t i = 0; i < n; i += 4) {
    __m256d a = _mm256_loadu_pd(al + i), b = _mm256_loadu_pd(ah + i), c = _mm256_loadu_pd(bl + i), d = _mm256_loadu_pd(bh + i);
    __m256d P = _mm256_cmp_pd(c, z, _CMP_GT_OQ), ge = _mm256_cmp_pd(a, z, _CMP_GE_OQ), le = _mm256_andnot_pd(ge, _mm256_cmp_pd(b, z, _CMP_LE_OQ));
    __m256d num = _mm256_blendv_pd(a, b, P), den = _mm256_blendv_pd(_mm256_blendv_pd(d, c, ge), _mm256_blendv_pd(c, d, le), P);
    _mm256_storeu_pd(zh + i, _mm256_div_pd(num, den));
  }
  (void)zl;
}
typedef void (*fn)(ARGS);
static void directed(fn lo, fn hi, ARGS)
{
  unsigned csr = _mm_getcsr(), base = csr & ~0x6000u & ~0x8040u;
  for (size_t i = 0; i < n; i += 256) {
    size_t m = n - i < 256 ? n - i : 256;
    _mm_setcsr(base | 0x2000u); lo(al + i, ah + i, bl + i, bh + i, zl + i, zh + i, m);   /* toward -inf */
    _mm_setcsr(base | 0x4000u); hi(al + i, ah + i, bl + i, bh + i, zl + i, zh + i, m);   /* toward +inf */
  }
  _mm_setcsr(csr);
}
static void dir_add(ARGS) { directed(d_add_lo, d_add_hi, al, ah, bl, bh, zl, zh, n); }
static void dir_mul(ARGS) { directed(d_mul_lo, d_mul_hi, al, ah, bl, bh, zl, zh, n); }
static void dir_div(ARGS) { directed(d_div_lo, d_div_hi, al, ah, bl, bh, zl, zh, n); }
#else
#define dir_add 0
#define dir_mul 0
#define dir_div 0
#endif
static void sc_add(ARGS) { ival__arith_path = 2; ival_add(al, ah, bl, bh, zl, zh, n); ival__arith_path = 0; }
static void sc_mul(ARGS) { ival__arith_path = 2; ival_mul(al, ah, bl, bh, zl, zh, n); ival__arith_path = 0; }
static void sc_div(ARGS) { ival__arith_path = 2; ival_div(al, ah, bl, bh, zl, zh, n); ival__arith_path = 0; }
static void pt_add(ARGS) { ival__arith_path = 1; ival_add(al, ah, bl, bh, zl, zh, n); ival__arith_path = 0; }
static void pt_mul(ARGS) { ival__arith_path = 1; ival_mul(al, ah, bl, bh, zl, zh, n); ival__arith_path = 0; }
static void pt_div(ARGS) { ival__arith_path = 1; ival_div(al, ah, bl, bh, zl, zh, n); ival__arith_path = 0; }

static double best(fn f, ARGS)
{
  size_t reps = 1;
  for (;;) { double t = now(); for (size_t r = 0; r < reps; r++) f(al, ah, bl, bh, zl, zh, n); if (now() - t > 0.02) break; reps *= 2; }
  double m = 1e30;
  for (int p = 0; p < 7; p++) {
    double t = now();
    for (size_t r = 0; r < reps; r++) f(al, ah, bl, bh, zl, zh, n);
    t = (now() - t) / ((double)reps * n) * 1e9;
    if (t < m) m = t;
  }
  return m;
}

int main(void)
{
#if defined(__x86_64__)
  int bare = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");   /* else no bare column */
#else
  int bare = 0;
#endif
  size_t sizes[2] = { 1024, (size_t)1 << 22 }, N = sizes[1];
  double *al = aligned_alloc(64, N * 8), *ah = aligned_alloc(64, N * 8), *bl = aligned_alloc(64, N * 8), *bh = aligned_alloc(64, N * 8);
  double *bdl = aligned_alloc(64, N * 8), *bdh = aligned_alloc(64, N * 8);   /* divisors without 0 */
  double *zl = aligned_alloc(64, N * 8), *zh = aligned_alloc(64, N * 8), *rl = aligned_alloc(64, N * 8), *rh = aligned_alloc(64, N * 8);
  for (size_t i = 0; i < N; i++) {
    double x = (unit() - 0.5) * 2e3, w = unit() * 10, y = (unit() - 0.5) * 2e3, v = unit() * 10;
    al[i] = x; ah[i] = x + w; bl[i] = y; bh[i] = y + v;
    double d = 0.5 + unit() * 1e3, dw = unit() * 10;
    if (rnd() & 1) { bdl[i] = d; bdh[i] = d + dw; } else { bdl[i] = -d - dw; bdh[i] = -d; }
  }
  const char *ops[3] = { "add", "mul", "div" };
  fn f[3][5] = { { rn_add, ival_add, pt_add, sc_add, dir_add }, { rn_mul, ival_mul, pt_mul, sc_mul, dir_mul }, { rn_div, ival_div, pt_div, sc_div, dir_div } };
  int bad = 0;
  printf("ns per interval (least of 7 passes)       rn   ival portable    eft   bare\n");
  for (int s = 0; s < 2; s++) {
    size_t n = sizes[s];
    for (int o = 0; o < 3; o++) {
      const double *b1 = o == 2 ? bdl : bl, *b2 = o == 2 ? bdh : bh;
      /* correctness in the same run: scalar and directed must give ival's results bit for bit */
      f[o][1](al, ah, b1, b2, rl, rh, n);
      for (int v = 2; v < 4 + bare; v++) {
        f[o][v](al, ah, b1, b2, zl, zh, n);
        if (memcmp(zl, rl, n * 8) || memcmp(zh, rh, n * 8)) { printf("VOID: %s variant %d differs from ival\n", ops[o], v); bad = 1; }
      }
      printf("%-4s n = %-8zu                    ", ops[o], n);
      for (int v = 0; v < 5; v++) {
        if (v == 4 && !bare) { printf("       -"); continue; }
        double t = best(f[o][v], al, ah, b1, b2, zl, zh, n);
        if (t < 0.05) printf("  FOLDED"); else printf(" %7.2f", t);
      }
      printf("\n");
    }
  }
  return bad;
}
