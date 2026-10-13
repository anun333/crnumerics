/* crdist.c: crdist.h's functions. Each runs in the default floating-point
   environment, set on entry and the caller's restored on exit (crdist.h). */
#include <fenv.h>
#include <stdio.h>
#include <stdlib.h>
#include "crdist.h"
#include "crdist-fast.h"
#include "crdist-exceptions.h"

#define ENV_ENTER fenv_t env_; fegetenv(&env_); fesetenv(FE_DFL_ENV)
#define ENV_LEAVE fesetenv(&env_)

/* an input the fast path left undecided: the table must have it
   (gen-exceptions tried every input), so a miss is a bug */
static float from_table(int f, float x)
{
  uint32_t u; memcpy(&u, &x, 4);
  size_t lo = 0, hi = crdist_exc[f].n;
  while (lo < hi) {
    size_t mid = lo + (hi - lo) / 2;
    if (crdist_exc[f].t[mid][0] < u) lo = mid + 1; else hi = mid;
  }
  if (lo == crdist_exc[f].n || crdist_exc[f].t[lo][0] != u) {
    fprintf(stderr, "crdist: %s(%a) undecided and not in crdist-exceptions.h (regenerate it: make crdist-exceptions)\n", crdist_fn_name[f], x);
    abort();
  }
  float y; memcpy(&y, &crdist_exc[f].t[lo][1], 4);
  return y;
}

static inline float one(int f, float x)
{
  float y;
  if (crdist_decide32(f, x, &y)) return y;
  return from_table(f, x);
}

static void each(int f, float *y, const float *x, size_t n)
{
  ENV_ENTER;
  for (size_t i = 0; i < n; i++) y[i] = one(f, x[i]);
  ENV_LEAVE;
}

void crdist_erfinvf(float *y, const float *x, size_t n) { each(CRDIST_ERFINV, y, x, n); }
void crdist_erfcinvf(float *y, const float *x, size_t n) { each(CRDIST_ERFCINV, y, x, n); }
void crdist_probitf(float *y, const float *p, size_t n) { each(CRDIST_PROBIT, y, p, n); }
void crdist_ndtrf(float *y, const float *x, size_t n) { each(CRDIST_NDTR, y, x, n); }
void crdist_log_ndtrf(float *y, const float *x, size_t n) { each(CRDIST_LOG_NDTR, y, x, n); }
void crdist_normpdff(float *y, const float *x, size_t n) { each(CRDIST_NORMPDF, y, x, n); }
void crdist_logitf(float *y, const float *p, size_t n) { each(CRDIST_LOGIT, y, p, n); }
const char *crdist_version(void) { return CRDIST_VERSION; }
