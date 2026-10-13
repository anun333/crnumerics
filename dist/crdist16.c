/* crdist16.c: crdist's functions in binary16 and bfloat16 (2026-10-13;
   crdist.h). The same binary64 fast paths as binary32 (crdist-fast.h),
   then the format's rounding test (crdist-round16.h), then a table for the
   inputs it can't decide (crdist-exceptions16.h, made from MPFR over every
   input). Special inputs go through crdist_special, whose values (zeros,
   infinities, NaN) every format holds exactly; every 16-bit value is a
   binary32 value, so the conversion is exact. */
#include <fenv.h>
#include <stdio.h>
#include <stdlib.h>
#include "crdist.h"
#include "crdist-fast.h"
#include "crdist-round16.h"
#include "crdist-exceptions16.h"

static uint16_t from_table16(int fmt, int f, uint16_t u)
{
  const uint16_t (*t)[2] = crdist_exc16[fmt][f].t; size_t lo = 0, hi = crdist_exc16[fmt][f].n;
  while (lo < hi) { size_t mid = lo + (hi - lo) / 2; if (t[mid][0] < u) lo = mid + 1; else hi = mid; }
  if (lo == crdist_exc16[fmt][f].n || t[lo][0] != u) {
    fprintf(stderr, "crdist: %s (%s) at 0x%04x undecided and not in crdist-exceptions16.h (make crdist-exceptions16)\n",
            crdist_fn_name[f], crdist16_fmt_name[fmt], u);
    abort();
  }
  return t[lo][1];
}

static inline uint16_t one16(int fmt, int f, uint16_t u)
{
  double y;
  if (crdist16_decide(fmt, f, crdist16_decode(fmt, u), &y)) return crdist16_encode(fmt, y);
  return from_table16(fmt, f, u);
}

#define ENV_ENTER fenv_t env_; fegetenv(&env_); fesetenv(FE_DFL_ENV)
#define ENV_LEAVE fesetenv(&env_)
#define MAP16(name, F, FMT, SFX)                                                                 \
  void crdist_##name##_##SFX(uint16_t *y, const uint16_t *x, size_t n)                         \
  { ENV_ENTER; for (size_t i = 0; i < n; i++) y[i] = one16(FMT, F, x[i]); ENV_LEAVE; }
#define BOTH(name, F) MAP16(name, F, CRDIST16_F16, f16) MAP16(name, F, CRDIST16_BF16, bf16)
BOTH(erfinv, CRDIST_ERFINV) BOTH(erfcinv, CRDIST_ERFCINV) BOTH(probit, CRDIST_PROBIT)
BOTH(ndtr, CRDIST_NDTR) BOTH(log_ndtr, CRDIST_LOG_NDTR) BOTH(normpdf, CRDIST_NORMPDF) BOTH(logit, CRDIST_LOGIT)
