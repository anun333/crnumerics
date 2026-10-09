/* ival-eft.h: the rounding primitives ival-arith.c and ival-1788.c share (not API).

   add_r, mul_r and div_r give a + b, a * b and a / b rounded down (up = 0) or up (up = 1) while the rounding mode is to
   nearest: an error-free transformation gives the exact error, which says whether the result moves one ulp; near
   underflow, where the error stops being exact, they switch the mode for the one operation. cmp_diff compares two
   differences exactly. The callers run with the mode to nearest and the flush modes off (flush_off). */
#ifndef IVAL_EFT_H
#define IVAL_EFT_H
#include <fenv.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#if defined(__x86_64__)
#include <immintrin.h>
#endif
#ifndef IVAL_PLANT_ARITH
#define IVAL_PLANT_ARITH 0
#endif

static inline int empty(double lo, double hi) { return !(lo <= hi) || lo == INFINITY || hi == -INFINITY; }
/* the doubles next below and above x, as nextafter(x, -inf) and nextafter(x, inf), inline: a step of the bit pattern,
   since a call to libm's nextafter was most of the scalar code's cost (a product's bounds take eight) */
static inline double pred(double x)
{
  if (x != x || x == -INFINITY) return x;
  if (x == 0) return -DBL_TRUE_MIN;
  uint64_t b;
  memcpy(&b, &x, 8);
  b += x > 0 ? (uint64_t)-1 : 1;
  memcpy(&x, &b, 8);
  return x;
}
static inline double succ(double x)
{
  if (x != x || x == INFINITY) return x;
  if (x == 0) return DBL_TRUE_MIN;
  uint64_t b;
  memcpy(&b, &x, 8);
  b += x > 0 ? 1 : (uint64_t)-1;
  memcpy(&x, &b, 8);
  return x;
}

/* ---- The caller's floating-point state, saved, changed and given back cheaply. ival's arithmetic is SSE only, so on
   x86-64 it reads and writes MXCSR alone: about 10 ns a call, against 122 ns for glibc's fegetenv and fesetenv and 146
   for fegetround and fesetround there and back, which handle the x87 unit too. Its CORE-MATH objects are linked with
   their fenv calls renamed to MXCSR versions (ival-fenv.c), so they read the mode set here and raise their flags where
   env_restore gives the caller's back; the x87 unit is never touched. Elsewhere, fenv.h itself. The FE_ constants are
   x86's: shifted left 3 they are MXCSR's rounding-control bits. ---- */
#if defined(__x86_64__)
typedef unsigned ival_env;   /* writing MXCSR costs, reading it hardly: each write is made only when it changes it */
static inline void env_save(ival_env *e) { *e = _mm_getcsr(); }
static inline void env_restore(const ival_env *e)
{
#if IVAL_PLANT_ARITH == 42   /* 42: the flags raised inside left for the caller */
  _mm_setcsr((*e & ~0x3fu) | (_mm_getcsr() & 0x3fu)); return;
#endif
  if (_mm_getcsr() != *e) _mm_setcsr(*e);
}
static inline void set_round(int m)
{
  unsigned c = _mm_getcsr(), d = (c & ~0x6000u) | ((unsigned)m << 3);
  if (d != c) _mm_setcsr(d);
}
static inline int get_round(void) { return (int)((_mm_getcsr() >> 3) & 0xc00u); }
#elif defined(__aarch64__)
/* aarch64: FPCR (rounding, flush) and FPSR (flags) directly, each written only when it changes. glibc's fenv calls use
   the same two registers, so CORE-MATH's fegetround and feraiseexcept agree with them and need no renaming. The FE_
   rounding values are FPCR's RMode bits. */
typedef struct { uint64_t cr, sr; } ival_env;
static inline uint64_t rd_fpcr(void) { uint64_t r; __asm__ volatile("mrs %0, fpcr" : "=r"(r)); return r; }
static inline uint64_t rd_fpsr(void) { uint64_t r; __asm__ volatile("mrs %0, fpsr" : "=r"(r)); return r; }
static inline void wr_fpcr(uint64_t r) { __asm__ volatile("msr fpcr, %0" : : "r"(r)); }
static inline void wr_fpsr(uint64_t r) { __asm__ volatile("msr fpsr, %0" : : "r"(r)); }
static inline void env_save(ival_env *e) { e->cr = rd_fpcr(); e->sr = rd_fpsr(); }
static inline void env_restore(const ival_env *e)
{
  if (rd_fpcr() != e->cr) wr_fpcr(e->cr);
#if IVAL_PLANT_ARITH == 42   /* 42: the flags raised inside left for the caller */
  return;
#endif
  if (rd_fpsr() != e->sr) wr_fpsr(e->sr);
}
static inline void set_round(int m)
{
  uint64_t c = rd_fpcr(), d = (c & ~0xc00000ull) | (uint64_t)(unsigned)m;
  if (d != c) wr_fpcr(d);
}
static inline int get_round(void) { return (int)(rd_fpcr() & 0xc00000ull); }
#else
typedef fenv_t ival_env;
static inline void env_save(ival_env *e) { fegetenv(e); }
static inline void env_restore(const ival_env *e) { fesetenv(e); }
static inline void set_round(int m) { fesetround(m); }
static inline int get_round(void) { return fegetround(); }
#endif

/* one operation rounding down or up, for the cases the error-free transformations do not cover */
static inline double directed(int op, double a, double b, int up)
{
  int m = get_round();
  set_round(up ? FE_UPWARD : FE_DOWNWARD);
  volatile double x = a, y = b, r;
  r = op == 0 ? x + y : op == 1 ? x * y : x / y;
  set_round(m);
  return r;
}

/* a + b rounded down (up = 0) or up (up = 1) */
static inline double add_r(double a, double b, int up)
{
  double s = a + b;
  if (isinf(s)) {
    if (isinf(a) || isinf(b)) return s;                    /* exact */
#if IVAL_PLANT_ARITH == 6   /* overflow left at infinity both ways */
    return s;
#else
    return up ? (s > 0 ? s : -DBL_MAX) : (s > 0 ? DBL_MAX : s);   /* overflow */
#endif
  }
  double bb = s - a, e = (a - (s - bb)) + (b - bb);          /* TwoSum: a + b = s + e exactly */
#if IVAL_PLANT_ARITH == 1   /* the check's control: step up even when exact */
  if (up) return e >= 0 ? succ(s) : s;
#else
  if (up) return e > 0 ? succ(s) : s;
#endif
  return e < 0 ? pred(s) : s;
}

/* a * b rounded down or up, with 0 * inf = 0 */
static inline double mul_r(double a, double b, int up)
{
#if IVAL_PLANT_ARITH != 5   /* 5: 0 * inf left to IEEE (NaN) */
  if (a == 0 || b == 0) return 0.0;
#endif
  double p = a * b;
  if (isinf(p)) {
    if (isinf(a) || isinf(b)) return p;
    return up ? (p > 0 ? p : -DBL_MAX) : (p > 0 ? DBL_MAX : p);
  }
#if IVAL_PLANT_ARITH == 2   /* the residual trusted near underflow */
  if (1) {
#else
  if (fabs(p) >= 0x1p-969) {
#endif
    double e = __builtin_fma(a, b, -p);                      /* a * b = p + e exactly */
    if (up) return e > 0 ? succ(p) : p;
    return e < 0 ? pred(p) : p;
  }
  return directed(1, a, b, up);
}

/* a / b rounded down or up, b != 0; at the ends, finite / inf is 0 and inf / finite is inf */
static inline double div_r(double a, double b, int up)
{
  if (a == 0) return 0.0;
  if (isinf(b)) return 0.0;
  double q = a / b;
  if (isinf(a)) return q;
  if (isinf(q)) return up ? (q > 0 ? q : -DBL_MAX) : (q > 0 ? DBL_MAX : q);
  if (fabs(a) >= 0x1p-960 && fabs(b) >= 0x1p-960 && fabs(q) >= 0x1p-960 && fabs(b) <= 0x1p+960) {
    double r = __builtin_fma(-q, b, a);                      /* a = q * b + r exactly; a / b - q = r / b */
    if (r == 0) return q;
#if IVAL_PLANT_ARITH == 4   /* the remainder's sign read the wrong way */
    int above = (r > 0) != (b > 0);
#else
    int above = (r > 0) == (b > 0);                          /* the true quotient is above q */
#endif
    if (up) return above ? succ(q) : q;
    return above ? q : pred(q);
  }
  return directed(2, a, b, up);
}

static inline double canon(double x) { return x == 0 ? 0.0 : x; }

/* x^p rounded down (up = 0) or up, for an integer 2 <= p <= 64, without pow (2026-10-09; the IBEX backend spent a
   quarter of a polynomial problem in CORE-MATH's pow for pown and pownRev). The power is multiplied out in
   double-double: each step h * x = ph + pt exactly (fma), the low part's product added to pt with one rounding, then
   ph and that summed exactly (FastTwoSum), so each step adds a relative error below 2^-104, and p steps below
   p 2^-103; the bound taken is p 2^-100. If no step rounded (every low part 0 when multiplied), h + l is x^p exactly.
   The low part's sign, when it exceeds the bound, says on which side of h the power lies (|l| is below half an ulp
   of h, and the bound far below that). Returns 0 when that is undecided, or the power is near underflow or overflow
   (below 2^-900 or above 2^1000, where a residual would not be exact): the caller takes CORE-MATH's pow. Must run to
   nearest (FastTwoSum is exact only there). */
#define IVAL_POWDD_NAME pown_dd_c
#define IVAL_POWDD_ATTR
#include "ival-powdd.inc"
#undef IVAL_POWDD_NAME
#undef IVAL_POWDD_ATTR
#if defined(__x86_64__) && !defined(__FMA__)
/* the same compiled for the FMA instruction (fma() is otherwise a call into libm, twice a step), used when the CPU has
   it; the check is made once, the result kept with relaxed atomics (any thread may make it, all agree) */
#define IVAL_POWDD_NAME pown_dd_fma
#define IVAL_POWDD_ATTR __attribute__((target("fma")))
#include "ival-powdd.inc"
#undef IVAL_POWDD_NAME
#undef IVAL_POWDD_ATTR
static inline int pown_dd(double x, int p, int up, double *r)
{
  static int has_fma = -1;
  int f = __atomic_load_n(&has_fma, __ATOMIC_RELAXED);
  if (f < 0) { __builtin_cpu_init(); f = __builtin_cpu_supports("fma") != 0; __atomic_store_n(&has_fma, f, __ATOMIC_RELAXED); }
  return f ? pown_dd_fma(x, p, up, r) : pown_dd_c(x, p, up, r);
}
#else
static inline int pown_dd(double x, int p, int up, double *r) { return pown_dd_c(x, p, up, r); }
#endif

/* A real interval known by its ends rounded both ways (ld <= lower end <= lu, hd <= upper end <= hu), an end open when
   it is only a limit; meet_hull gives the tightest interval around the union of k such pieces' intersections with
   X = [xl, xh], or the empty one. The hull takes the outer roundings. Whether a piece meets X is decided exactly on the
   inner ones: a real q is below a binary64 xl exactly when q rounded down is. */
struct piece { double ld, lu, hd, hu; int lopen, hopen; };
static inline void meet_hull(const struct piece *p, int k, double xl, double xh, double *zl, double *zh)
{
  double l = INFINITY, h = -INFINITY;
  for (int j = 0; j < k; j++) {
    int hi_ok = p[j].hopen ? p[j].hd > xl : p[j].hd >= xl, lo_ok = p[j].lopen ? p[j].lu < xh : p[j].lu <= xh;
#if IVAL_PLANT_ARITH == 23   /* 23: the meeting test on the outer roundings */
    hi_ok = p[j].hu >= xl; lo_ok = p[j].ld <= xh;
#endif
    if (!hi_ok || !lo_ok) continue;
    double a = p[j].ld > xl ? p[j].ld : xl, b = p[j].hu < xh ? p[j].hu : xh;
    if (a < l) l = a;
    if (b > h) h = b;
  }
  if (l > h) *zl = *zh = NAN;
  else { *zl = canon(l); *zh = canon(h); }
}

/* the sign of (a - b) - (c - d), exactly, for finite a, b, c, d. Each difference is s + e exactly by TwoSum, and
   rounding to nearest is monotone, so different s order the exact values; equal s leave it to e1 - e2, whose rounded
   value has its sign. When a difference overflows, its operands are at least 2^970 in magnitude, so if both overflow
   all four are and halving them is exact; if one does, it is the larger. */
static inline int cmp_diff(double a, double b, double c, double d)
{
  double s1 = a - b, s2 = c - d;
  if (isinf(s1) || isinf(s2)) {
    if (!isinf(s2)) return s1 > 0 ? 1 : -1;
    if (!isinf(s1)) return s2 > 0 ? -1 : 1;
    a *= 0.5; b *= 0.5; c *= 0.5; d *= 0.5; s1 = a - b; s2 = c - d;
  }
  if (s1 != s2) return s1 > s2 ? 1 : -1;
  double t1 = s1 - a, e1 = (a - (s1 - t1)) + (-b - t1);
  double t2 = s2 - c, e2 = (c - (s2 - t2)) + (-d - t2);
  double g = e1 - e2;
#if IVAL_PLANT_ARITH == 16   /* 16: equal rounded differences taken as equal */
  g = 0;
#endif
  return g > 0 ? 1 : g < 0 ? -1 : 0;
}

/* Flush-to-zero and denormals-are-zero off for the call (x86-64: MXCSR's FZ and DAZ; aarch64: FPCR.FZ); env_restore
   gives them back. A program built with -ffast-math starts with them on, and either breaks the enclosures. */
static inline void flush_off(void)
{
#if IVAL_PLANT_ARITH != 10   /* 10: the flush modes left as the caller set them */
#if defined(__x86_64__)
  unsigned c = _mm_getcsr();
  if (c & 0x8040u) _mm_setcsr(c & ~0x8040u);
#elif defined(__aarch64__)
  uint64_t r = rd_fpcr();
  if (r & (1ull << 24)) wr_fpcr(r & ~(1ull << 24));
#endif
#endif
}

#endif
