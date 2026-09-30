/* lowp.c: correctly rounded math for E4M3 and E5M2 (lowp.h).

   One-argument functions: a table per format and rounding mode, generated
   from MPFR by gen-tables.c (lowp-tables.h). E4M3's two overflow modes
   share a table: it holds the saturating results, and a bit per input
   marks the results that overflowed away from zero, which are NaN unless
   saturating.

   Two-argument functions: CORE-MATH's binary64 function, run in the same
   rounding mode, then rounded to the format by round8() below. Rounding
   twice in one direction is rounding once. To nearest it could differ only
   where the binary64 result is a midpoint of the format and the exact
   result is not. lowp/test/check.c tries every pair in every mode and finds
   no such case. (It also finds that computing in round-to-nearest for
   every mode would change nothing on these pairs; the matching mode stays,
   since it is exact without leaning on that.) */
#include <fenv.h>
#include <math.h>
#include <string.h>
#include "lowp.h"
#include "lowp-tables.h"

double cr_atan2(double, double), cr_atan2pi(double, double), cr_hypot(double, double), cr_pow(double, double);

typedef struct { int ebits, mbits, bias, has_inf; uint8_t qnan; } fmt;
static const fmt E4M3 = {4, 3, 7, 0, 0x7f}, E5M2 = {5, 2, 15, 1, 0x7e};

enum {
#define LOWP_F1(f) I_##f,
#define LOWP_F2(f)
#include "lowp-list.h"
  NF1
};
_Static_assert(NF1 == sizeof LOWP_T / sizeof *LOWP_T, "lowp-tables.h has a table per one-argument function");

/* E4M3 takes LOWP_SAT; E5M2 doesn't */
static int mode_ok(const fmt *f, int mode) { return mode >= 0 && mode <= (f->has_inf ? 3 : 7); }

static uint8_t maxf(const fmt *f)
{ return (uint8_t)((((1u << f->ebits) - 1 - f->has_inf) << f->mbits) | ((1u << f->mbits) - 1 - !f->has_inf)); }

/* v rounded to the format, in mode rnd (LOWP_NEAREST ... LOWP_ZERO),
   saturating or not (E4M3). Integer arithmetic only, so the C rounding mode
   doesn't matter: v's significand is split at the format's quantum, the
   part below decides the rounding, and a result beyond the largest finite
   value overflows as IEEE 754 says (E4M3: NaN, or the largest value). */
static uint8_t round8(const fmt *f, double v, int rnd, int sat)
{
  uint64_t b;
  memcpy(&b, &v, 8);
  uint8_t s = (uint8_t)(b >> 63 << 7), top = maxf(f), inf = (uint8_t)(((1u << f->ebits) - 1) << f->mbits);
  uint64_t a = b & 0x7fffffffffffffffULL;
  int away = rnd == LOWP_NEAREST || (rnd == LOWP_UP && !s) || (rnd == LOWP_DOWN && s);
  if (a > 0x7ff0000000000000ULL) return f->qnan;
  if (a == 0x7ff0000000000000ULL) return f->has_inf ? s | inf : sat ? s | top : f->qnan;
  if (a == 0) return s;
  int e = (int)(a >> 52);
  uint64_t m = a & ((1ULL << 52) - 1);
  if (e) m |= 1ULL << 52;
  else e = 1;
  /* |v| = m * 2^(e - 1075); the exponent of its leading bit, then of the
     format's quantum there (the subnormals' below the smallest normal) */
  int ev = e - 1075 + 63 - __builtin_clzll(m), emin = 1 - f->bias;
  int qe = (ev > emin ? ev : emin) - f->mbits, shift = qe - (e - 1075);
  uint64_t q;
  if (shift <= 0) q = m << -shift;   /* a multiple of the quantum: exact */
  else {
    uint64_t rem, half;
    if (shift >= 64) { q = 0; rem = 1; half = ~0ULL; }   /* far below half the quantum */
    else { q = m >> shift; rem = m & ((1ULL << shift) - 1); half = 1ULL << (shift - 1); }
    if (rem && (rnd == LOWP_NEAREST ? rem > half || (rem == half && (q & 1)) : away)) q++;
  }
  if (!q) return s;
  if (q >> (f->mbits + 1)) { q >>= 1; qe++; }   /* carried into the next binade */
  uint32_t biased = q >> f->mbits ? (uint32_t)(qe + f->mbits + f->bias) : 0;
  uint32_t enc = biased << f->mbits | (uint32_t)(q & ((1u << f->mbits) - 1));
  if (enc > top) return away ? (f->has_inf ? s | inf : sat ? s | top : f->qnan) : s | top;
  return s | (uint8_t)enc;
}

static double decode(const fmt *f, uint8_t x)
{
  int e = x >> f->mbits & ((1 << f->ebits) - 1), m = x & ((1 << f->mbits) - 1);
  double v;
  if (e == (1 << f->ebits) - 1 && (f->has_inf ? m != 0 : m == (1 << f->mbits) - 1)) v = NAN;
  else if (e == (1 << f->ebits) - 1 && f->has_inf) v = INFINITY;
  else if (e == 0) v = ldexp(m, 1 - f->bias - f->mbits);   /* exact */
  else v = ldexp(m | 1 << f->mbits, e - f->bias - f->mbits);
  return x & 0x80 ? -v : v;
}

/* a signaling NaN (E5M2's quiet bit is bit 1; E4M3 has one NaN, quiet) */
static int snan(const fmt *f, uint8_t x) { return f->has_inf && (x & 0x7f) == 0x7d; }

#define CONVERT(name, F, T)                                                 \
  int lowp_##name##_from_##T(const T##_t *x, uint8_t *y, size_t n, int mode) \
  {                                                                         \
    if (!mode_ok(&F, mode)) return -1;                                      \
    for (size_t i = 0; i < n; i++) y[i] = round8(&F, x[i], mode & 3, mode >> 2); \
    return 0;                                                               \
  }                                                                         \
  void lowp_##name##_to_##T(const uint8_t *x, T##_t *y, size_t n)            \
  { for (size_t i = 0; i < n; i++) y[i] = (T##_t)decode(&F, x[i]); }
typedef double f64_t;
typedef float f32_t;
CONVERT(e4m3, E4M3, f64) CONVERT(e5m2, E5M2, f64) CONVERT(e4m3, E4M3, f32) CONVERT(e5m2, E5M2, f32)

#define LOWP_F1(f)                                                                    \
  int lowp_e5m2_##f(const uint8_t *x, uint8_t *y, size_t n, int mode)                \
  {                                                                                   \
    if (!mode_ok(&E5M2, mode)) return -1;                                             \
    const uint8_t *t = LOWP_T[I_##f][0][mode];                                        \
    for (size_t i = 0; i < n; i++) y[i] = t[x[i]];                                    \
    return 0;                                                                         \
  }                                                                                   \
  int lowp_e4m3_##f(const uint8_t *x, uint8_t *y, size_t n, int mode)                \
  {                                                                                   \
    if (!mode_ok(&E4M3, mode)) return -1;                                             \
    const uint8_t *t = LOWP_T[I_##f][1][mode & 3], *o = LOWP_OVF[I_##f][mode & 3];    \
    if (mode & LOWP_SAT) for (size_t i = 0; i < n; i++) y[i] = t[x[i]];               \
    else for (size_t i = 0; i < n; i++) { uint8_t v = x[i]; y[i] = o[v >> 3] >> (v & 7) & 1 ? 0x7f : t[v]; } \
    return 0;                                                                         \
  }
#define LOWP_F2(f)
#include "lowp-list.h"

/* the two-argument functions, in the C rounding mode that matches; the
   caller's floating-point environment (mode and flags) is restored */
static const int FE[4] = {FE_TONEAREST, FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO};
static int two(const fmt *f, double (*g)(double, double), const uint8_t *x, const uint8_t *y, uint8_t *z, size_t n,
               int mode)
{
  if (!mode_ok(f, mode)) return -1;
  fenv_t env;
  fegetenv(&env);
  fesetround(FE[mode & 3]);
  for (size_t i = 0; i < n; i++) {
    uint8_t a = x[i], b = y[i];
    z[i] = snan(f, a) || snan(f, b) ? f->qnan : round8(f, g(decode(f, a), decode(f, b)), mode & 3, mode >> 2);
  }
  fesetenv(&env);
  return 0;
}
#define LOWP_F1(f)
#define LOWP_F2(f)                                                                                        \
  int lowp_e4m3_##f(const uint8_t *x, const uint8_t *y, uint8_t *z, size_t n, int mode)                  \
  { return two(&E4M3, cr_##f, x, y, z, n, mode); }                                                        \
  int lowp_e5m2_##f(const uint8_t *x, const uint8_t *y, uint8_t *z, size_t n, int mode)                  \
  { return two(&E5M2, cr_##f, x, y, z, n, mode); }
#include "lowp-list.h"
