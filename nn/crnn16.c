/* crnn16.c: crnn's one-argument functions in binary16 and bfloat16
   (2026-10-01; crnn.h). The same binary64 fast paths as binary32
   (crnn-fast.h), then the format's rounding test (crnn-round16.h), then a
   table for the inputs it can't decide (crnn-exceptions16.h, made from MPFR
   over every input). Special inputs (NaN, infinities, zeros, and rsqrt's
   negatives) go through crnn_special, whose values (0.5, zeros,
   infinities, NaN) every format holds exactly. */
#include <fenv.h>
#include <stdio.h>
#include <stdlib.h>
#include "crnn.h"
#include "crnn-fast.h"
#include "crnn-round16.h"
#include "crnn-exceptions16.h"

static uint16_t from_table16(int fmt, int f, uint16_t u)
{
  const uint16_t (*t)[2] = crnn_exc16[fmt][f].t; size_t lo = 0, hi = crnn_exc16[fmt][f].n;
  while (lo < hi) { size_t mid = lo + (hi - lo) / 2; if (t[mid][0] < u) lo = mid + 1; else hi = mid; }
  if (lo == crnn_exc16[fmt][f].n || t[lo][0] != u) {
    fprintf(stderr, "crnn: %s (%s) at 0x%04x undecided and not in crnn-exceptions16.h (make crnn-exceptions16)\n",
            crnn_fn1_name[f], fmt ? "bfloat16" : "binary16", u);
    abort();
  }
  return t[lo][1];
}

static inline uint16_t one16(int fmt, int f, uint16_t u)
{
  double x = crnn16_decode(fmt, u), y;
  float s;
  int special = (x != x) || isinf(x) || x == 0 || (f == CRNN_RSQRT && x < 0);
  if (special && crnn_special(f, (float)x, &s)) return crnn16_encode(fmt, (double)s);
  if (crnn16_round(fmt, crnn_fast(f, x), &y)) return crnn16_encode(fmt, y);
  return from_table16(fmt, f, u);
}

#define ENV_ENTER fenv_t env_; fegetenv(&env_); fesetenv(FE_DFL_ENV)
#define ENV_LEAVE fesetenv(&env_)
#define MAP16(name, F, FMT, SFX)                                                                 \
  void crnn_##name##_##SFX(uint16_t *y, const uint16_t *x, size_t n)                           \
  { ENV_ENTER; for (size_t i = 0; i < n; i++) y[i] = one16(FMT, F, x[i]); ENV_LEAVE; }
#define BOTH(name, F) MAP16(name, F, CRNN16_F16, f16) MAP16(name, F, CRNN16_BF16, bf16)
BOTH(sigmoid, CRNN_SIGMOID) BOTH(silu, CRNN_SILU) BOTH(gelu, CRNN_GELU) BOTH(softplus, CRNN_SOFTPLUS) BOTH(rsqrt, CRNN_RSQRT)
