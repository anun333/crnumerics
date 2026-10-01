/* crnn-round16.h: the 16-bit formats of crnn's one-argument functions
   (added 2026-10-01): binary16 (IEEE half: 11 significant bits, exponents
   -14 .. 15, subnormals to 2^-24) and bfloat16 (8 bits, binary32's range,
   subnormals to 2^-133). Decoding is exact. crnn16_rn rounds a binary64 to
   the format, to nearest with ties to even, overflowing to infinity, in the
   default floating-point environment crnn sets: the scaling by the
   quantum is exact (a power of two, far from binary64's limits), and
   nearbyint rounds to an integer in the current mode. crnn16_round is the
   rounding test: the format's value of anything within |r| 2^-50 of r, or
   0 when that interval holds two roundings. Shared by the library and the
   exception generator, so the two can't disagree on which inputs are hard. */
#ifndef CRNN_ROUND16_H
#define CRNN_ROUND16_H
#include <math.h>
#include <stdint.h>
#include <string.h>

enum { CRNN16_F16, CRNN16_BF16, CRNN16_NFMT };
static const struct { int p, emin, emax; } crnn16_fmt[CRNN16_NFMT] = {{11, -14, 15}, {8, -126, 127}};

static inline double crnn16_decode(int fmt, uint16_t u)
{
  if (fmt == CRNN16_BF16) { uint32_t w = (uint32_t)u << 16; float f; memcpy(&f, &w, 4); return f; }
  int s = u >> 15, e = (u >> 10) & 31, m = u & 1023;
  double v = e == 31 ? (m ? NAN : INFINITY) : e == 0 ? ldexp(m, -24) : ldexp(1024 + m, e - 25);
  return s ? -v : v;
}

/* v exactly representable in the format (or an infinity, or a NaN) */
static inline uint16_t crnn16_encode(int fmt, double v)
{
  if (fmt == CRNN16_BF16) {
    float f = (float)v; uint32_t w; memcpy(&w, &f, 4);
    return v != v ? (uint16_t)((w >> 16) | 0x40) : (uint16_t)(w >> 16);   /* a NaN stays a quiet NaN */
  }
  uint16_t s = signbit(v) ? 0x8000 : 0;
  if (v != v) return s | 0x7e00;
  double a = fabs(v);
  if (a == INFINITY) return s | 0x7c00;
  if (a == 0) return s;
  int e = ilogb(a);
  if (e < -14) return s | (uint16_t)ldexp(a, 24);   /* subnormal: a multiple of 2^-24 */
  return s | (uint16_t)((e + 15) << 10) | (uint16_t)(ldexp(a, 10 - e) - 1024);
}

static inline double crnn16_rn(int fmt, double r)
{
  if (r == 0 || !isfinite(r)) return r;
  int p = crnn16_fmt[fmt].p, emin = crnn16_fmt[fmt].emin, emax = crnn16_fmt[fmt].emax;
  int e = ilogb(r);
  if (e < emin) e = emin;
  double v = ldexp(nearbyint(ldexp(r, p - 1 - e)), e - p + 1);
  double max = ldexp(2.0 - ldexp(1.0, 1 - p), emax);
  return fabs(v) > max ? copysign(INFINITY, r) : v;
}

static inline int crnn16_round(int fmt, double r, double *y)
{
  double e = fabs(r) * 0x1p-50, lo = crnn16_rn(fmt, r - e), hi = crnn16_rn(fmt, r + e);
  if (lo != hi) return 0;
  *y = lo;
  return 1;
}
#endif
