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

/* The vector path (2026-10-01), optional: crmvec's AVX2 entry points for
   exp, log1p and erfc, four binary64 lanes a call, loaded at first
   use from the library CRNN_CRMVEC names (crmvec's libmvec.so.1). Every
   other step is the same binary64 operation as crnn-fast.h's, lane by
   lane, and specials and the table are handled per element as in one().
   crmvec's functions are correctly rounded, as CORE-MATH's are, so they
   return the same binary64 values, and crnn-fast.h's proof, which needs
   nothing but correct rounding, holds unchanged: the same bits with or
   without crmvec (nn/test/vsame.c checks every input). Unset, unloadable,
   or a CPU without AVX2 and FMA: the scalar path. */
#if defined(__x86_64__)
#include <dlfcn.h>
#include <immintrin.h>
typedef __m256d (*crnn_v4)(__m256d);
static crnn_v4 v_exp, v_log1p, v_erfc;
static int v_state = -1;   /* -1 not tried, 0 scalar, 1 vector */
static int v_on(void)
{
  int st = __atomic_load_n(&v_state, __ATOMIC_ACQUIRE);
  if (st >= 0) return st;
  st = 0;
  const char *p = getenv("CRNN_CRMVEC");
  void *h = p && *p && __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") ? dlopen(p, RTLD_NOW | RTLD_LOCAL) : NULL;
  if (h) {
    v_exp = (crnn_v4)dlsym(h, "_ZGVdN4v_exp"); v_log1p = (crnn_v4)dlsym(h, "_ZGVdN4v_log1p");
    v_erfc = (crnn_v4)dlsym(h, "_ZGVdN4v_erfc");
    st = v_exp && v_log1p && v_erfc;
  }
  __atomic_store_n(&v_state, st, __ATOMIC_RELEASE);
  return st;
}
/* crnn-fast.h's binary64 values for four elements: the same operations,
   lane by lane (no contraction: -ffp-contract=off, and GELU's one fma is
   written as one) */
__attribute__((target("avx2,fma"))) static inline __m256d fast4(int f, __m256d x)
{
  const __m256d one = _mm256_set1_pd(1.0), sgn = _mm256_set1_pd(-0.0);
  __m256d nx = _mm256_xor_pd(x, sgn);   /* -x, exact */
  switch (f) {
    case CRNN_SIGMOID: return _mm256_div_pd(one, _mm256_add_pd(one, v_exp(nx)));
    case CRNN_SILU: return _mm256_div_pd(x, _mm256_add_pd(one, v_exp(nx)));
    case CRNN_SOFTPLUS: {   /* x > 0: x + log1p(exp(-x)); else log1p(exp(x)) */
      __m256d pos = _mm256_cmp_pd(x, _mm256_setzero_pd(), _CMP_GT_OQ);
      __m256d l = v_log1p(v_exp(_mm256_blendv_pd(x, nx, pos)));
      return _mm256_blendv_pd(l, _mm256_add_pd(x, l), pos);
    }
    default: {   /* GELU */
      const __m256d hi = _mm256_set1_pd(CRNN_RSQRT2_HI), lo = _mm256_set1_pd(CRNN_RSQRT2_LO);
      __m256d th = _mm256_mul_pd(nx, hi);
      __m256d tl = _mm256_add_pd(_mm256_fmadd_pd(nx, hi, _mm256_xor_pd(th, sgn)), _mm256_mul_pd(nx, lo));
      __m256d q = _mm256_xor_pd(_mm256_mul_pd(th, th), sgn);
      __m256d corr = _mm256_mul_pd(_mm256_mul_pd(_mm256_set1_pd(CRNN_TWO_RSQRTPI), v_exp(q)), tl);
      return _mm256_mul_pd(_mm256_mul_pd(_mm256_set1_pd(0.5), x), _mm256_sub_pd(v_erfc(th), corr));
    }
  }
}
/* four at a time: when no lane can be special (finite, |x| >= 2^-120)
   and every lane's interval rounds one way (crnn_round32's test, in
   vector form), the four results are stored at once; otherwise each lane
   goes through one()'s steps with the value computed here */
__attribute__((target("avx2,fma"))) static void map1_vec(int f, float *y, const float *x, size_t n)
{
  const __m256d eps = _mm256_set1_pd(CRNN_EPS), absm = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7fffffffffffffffLL));
  const __m128 tiny = _mm_set1_ps(0x1p-120f), inf = _mm_set1_ps(INFINITY), absf = _mm_castsi128_ps(_mm_set1_epi32(0x7fffffff));
  size_t i = 0;
  for (; i + 4 <= n; i += 4) {
    __m128 xf = _mm_loadu_ps(x + i), ax = _mm_and_ps(xf, absf);
    __m256d r = fast4(f, _mm256_cvtps_pd(xf));
    __m256d e = _mm256_mul_pd(_mm256_and_pd(r, absm), eps);
    __m128 lo = _mm256_cvtpd_ps(_mm256_sub_pd(r, e)), hi = _mm256_cvtpd_ps(_mm256_add_pd(r, e));
    /* lanes that are plain: tiny <= |x| < inf (false for NaN), and lo == hi */
    __m128 ok = _mm_and_ps(_mm_and_ps(_mm_cmpge_ps(ax, tiny), _mm_cmplt_ps(ax, inf)), _mm_cmpeq_ps(lo, hi));
    if (_mm_movemask_ps(ok) == 15) { _mm_storeu_ps(y + i, lo); continue; }
    double rd[4]; _mm256_storeu_pd(rd, r);
    float xs[4]; _mm_storeu_ps(xs, xf);   /* read before y is written: y may be x */
    for (int t = 0; t < 4; t++) {
      float yi;
      if (!crnn_special(f, xs[t], &yi) && !crnn_round32(rd[t], &yi)) yi = from_table(f, xs[t]);
      y[i + t] = yi;
    }
  }
  for (; i < n; i++) y[i] = one(f, x[i]);
}
/* rsqrt stays scalar: crmvec's vector rsqrt was slower here than
   CORE-MATH's scalar one (9.8 against 7.4 ns an element, 2026-10-01) */
#define MAP1_BODY(F) if ((F) != CRNN_RSQRT && v_on()) map1_vec(F, y, x, n); else for (size_t i = 0; i < n; i++) y[i] = one(F, x[i]);
int crnn_vector_path(void) { return v_on(); }
#else
#define MAP1_BODY(F) for (size_t i = 0; i < n; i++) y[i] = one(F, x[i]);
int crnn_vector_path(void) { return 0; }
#endif

#define MAP1(name, F) \
  void crnn_##name##f(float *y, const float *x, size_t n) \
  { ENV_ENTER; MAP1_BODY(F) ENV_LEAVE; }
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

/* e_i = exp(RN(x_i - m)) for i < k: through crmvec's vector exp when the
   vector path is on (the same values: both correctly rounded; NaN payloads
   aside, which crnn.h leaves unspecified), else CORE-MATH's */
#if defined(__x86_64__)
__attribute__((target("avx2,fma"))) static void exp_shift_vec(const float *x, size_t k, double m, double *e)
{
  size_t i = 0;
  for (; i + 4 <= k; i += 4)
    _mm256_storeu_pd(e + i, v_exp(_mm256_sub_pd(_mm256_cvtps_pd(_mm_loadu_ps(x + i)), _mm256_set1_pd(m))));
  for (; i < k; i++) e[i] = cr_exp((double)x[i] - m);
}
#endif
static void exp_shift(const float *x, size_t k, double m, double *e)
{
#if defined(__x86_64__)
  if (v_on()) { exp_shift_vec(x, k, m, e); return; }
#endif
  for (size_t i = 0; i < k; i++) e[i] = cr_exp((double)x[i] - m);
}

static double sum_exp(const float *x, size_t n, double m, double less)   /* S of crnn.h, or S - 1 (T) */
{
  acc_t *s = xmalloc(sizeof *s); acc_init(s, 0);
  double e[BUF];
  for (size_t i = 0; i < n; i += BUF) {
    size_t k = n - i < BUF ? n - i : BUF;
    exp_shift(x + i, k, m, e);
    for (size_t t = 0; t < k; t++) acc_put(s, e[t], 0);
  }
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
  int nan; double m = max_of(x, n, &nan), S = sum_exp(x, n, m, 0), e[BUF];
  for (size_t i = 0; i < n; i += BUF) {
    size_t k = n - i < BUF ? n - i : BUF;
    exp_shift(x + i, k, m, e);   /* before y is written: y may be x */
    for (size_t t = 0; t < k; t++) y[i + t] = (float)(e[t] / S);
  }
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
