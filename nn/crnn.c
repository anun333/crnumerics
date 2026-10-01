/* crnn.c: crnn.h's functions. Each runs in the default floating-point
   environment, set on entry and the caller's restored on exit (crnn.h). */
#include <fenv.h>
#include <stdio.h>
#include <stdlib.h>
#include "crnn.h"
#include "crnn-fast.h"
#include "crnn-exceptions.h"
#include "../crsum/crsum.h"

#define ENV_ENTER fenv_t env_; fegetenv(&env_); fesetenv(FE_DFL_ENV)
#define ENV_LEAVE fesetenv(&env_)

/* an input the fast path left undecided: the table must have it
   (gen-exceptions tried every input), so a miss is a bug */
static float from_table(int f, float x)
{
  uint32_t u; memcpy(&u, &x, 4);
  size_t lo = 0, hi = crnn_exc[f].n;
  while (lo < hi) {
    size_t mid = lo + (hi - lo) / 2;
    if (crnn_exc[f].t[mid][0] < u) lo = mid + 1; else hi = mid;
  }
  if (lo == crnn_exc[f].n || crnn_exc[f].t[lo][0] != u) {
    fprintf(stderr, "crnn: %s(%a) undecided and not in crnn-exceptions.h (regenerate it: make crnn-exceptions)\n", crnn_fn1_name[f], x);
    abort();
  }
  float y; memcpy(&y, &crnn_exc[f].t[lo][1], 4);
  return y;
}

static inline float one(int f, float x)
{
  float y;
  if (crnn_special(f, x, &y)) return y;
  if (crnn_round32(crnn_fast(f, x), &y)) return y;
  return from_table(f, x);
}

#define MAP1(name, F) \
  void crnn_##name##f(float *y, const float *x, size_t n) \
  { ENV_ENTER; for (size_t i = 0; i < n; i++) y[i] = one(F, x[i]); ENV_LEAVE; }
MAP1(sigmoid, CRNN_SIGMOID)
MAP1(silu, CRNN_SILU)
MAP1(gelu, CRNN_GELU)
MAP1(softplus, CRNN_SOFTPLUS)
MAP1(rsqrt, CRNN_RSQRT)

static void *xmalloc(size_t n)
{
  void *p = malloc(n);
  if (!p) { fprintf(stderr, "crnn: out of memory\n"); abort(); }
  return p;
}

/* sums of many terms through a buffer, so that crsum takes its binned path */
#define BUF 256
typedef struct { crsum_acc a; double x[BUF], y[BUF]; size_t k; int dot; } acc_t;
static void acc_init(acc_t *s, int dot) { crsum_init(&s->a); s->k = 0; s->dot = dot; }
static void acc_flush(acc_t *s)
{
  if (s->dot) crsum_add_dot(&s->a, s->x, s->y, s->k); else crsum_add(&s->a, s->x, s->k);
  s->k = 0;
}
static inline void acc_put(acc_t *s, double x, double y) { s->x[s->k] = x; s->y[s->k] = y; if (++s->k == BUF) acc_flush(s); }
static double acc_round(acc_t *s) { acc_flush(s); return crsum_round(&s->a, CRSUM_NEAREST); }

/* max over the non-NaN elements (-inf if none), and whether one is NaN */
static float max_of(const float *x, size_t n, int *nan)
{
  float m = -INFINITY; *nan = 0;
  for (size_t i = 0; i < n; i++) { if (x[i] != x[i]) *nan = 1; else if (x[i] > m) m = x[i]; }
  return m;
}

static double sum_exp(const float *x, size_t n, double m, double less)   /* S of crnn.h, or S - 1 (T) */
{
  acc_t *s = xmalloc(sizeof *s); acc_init(s, 0);
  for (size_t i = 0; i < n; i++) acc_put(s, cr_exp((double)x[i] - m), 0);
  acc_put(s, -less, 0);
  double S = acc_round(s); free(s);
  return S;
}

float crnn_logsumexpf(const float *x, size_t n)
{
  if (n == 0) return -INFINITY;
  ENV_ENTER;
  int nan; float m = max_of(x, n, &nan), r;
  if (nan) r = NAN;
  else if (isinf(m)) r = m;
  else r = (float)((double)m + cr_log1p(sum_exp(x, n, m, 1)));
  ENV_LEAVE;
  return r;
}

void crnn_softmaxf(float *y, const float *x, size_t n)
{
  if (n == 0) return;
  ENV_ENTER;
  int nan; double m = max_of(x, n, &nan), S = sum_exp(x, n, m, 0);
  for (size_t i = 0; i < n; i++) y[i] = (float)(cr_exp((double)x[i] - m) / S);
  ENV_LEAVE;
}

void crnn_layernormf(float *y, const float *x, size_t n, const float *g, const float *b, float eps)
{
  if (n == 0) return;
  ENV_ENTER;
  acc_t *s = xmalloc(sizeof *s); acc_init(s, 0);
  for (size_t i = 0; i < n; i++) acc_put(s, x[i], 0);
  double s1 = acc_round(s), ms1 = -s1;
  crsum_add(&s->a, &ms1, 1);                         /* the remainder, exact, then rounded */
  double s2 = crsum_round(&s->a, CRSUM_NEAREST), dn = (double)n;
  double m1 = s1 / dn, m2 = (__builtin_fma(-m1, dn, s1) + s2) / dn;
  acc_init(s, 1);
  for (size_t i = 0; i < n; i++) { double d = ((double)x[i] - m1) - m2; acc_put(s, d, d); }
  double r = cr_rsqrt(acc_round(s) / dn + (double)eps);
  free(s);
  for (size_t i = 0; i < n; i++) {
    double t = (((double)x[i] - m1) - m2) * r;
    if (g) t = t * g[i];
    if (b) t = t + b[i];
    y[i] = (float)t;
  }
  ENV_LEAVE;
}

void crnn_rmsnormf(float *y, const float *x, size_t n, const float *g, float eps)
{
  if (n == 0) return;
  ENV_ENTER;
  acc_t *s = xmalloc(sizeof *s); acc_init(s, 1);
  for (size_t i = 0; i < n; i++) acc_put(s, x[i], x[i]);
  double r = cr_rsqrt(acc_round(s) / (double)n + (double)eps);
  free(s);
  for (size_t i = 0; i < n; i++) {
    double t = (double)x[i] * r;
    if (g) t = t * g[i];
    y[i] = (float)t;
  }
  ENV_LEAVE;
}
