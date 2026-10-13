/* crdist-sample.c: crdist.h's samplers (2026-10-13). Each is the
   specification in crdist.h computed as written: Philox4x32-10 words, the
   binary32 probit of crdist.c, CORE-MATH's correctly rounded log and exp
   in binary64, and IEEE operations in binary64, each rounded to nearest in
   the default environment (set on entry, the caller's restored on exit),
   with no contraction (the Makefile's -ffp-contract=off). So every machine
   gives the same bits. crdist-check (section 9) computes the same
   specification independently and compares bit for bit. */
#include <fenv.h>
#include <math.h>
#include "crdist.h"
#include "crdist-fast.h"

#define ENV_ENTER fenv_t env_; fegetenv(&env_); fesetenv(FE_DFL_ENV)
#define ENV_LEAVE fesetenv(&env_)

/* Philox4x32-10 (Salmon, Moraes, Dror and Shaw, "Parallel random numbers:
   as easy as 1, 2, 3", SC 2011): ten rounds on a 128-bit counter under a
   64-bit key, the key bumped by the Weyl constants before every round but
   the first */
static void philox(uint32_t out[4], const uint32_t ctr[4], const uint32_t key[2])
{
  uint32_t c0 = ctr[0], c1 = ctr[1], c2 = ctr[2], c3 = ctr[3], k0 = key[0], k1 = key[1];
  for (int r = 0; r < 10; r++) {
    if (r) { k0 += 0x9E3779B9u; k1 += 0xBB67AE85u; }
    uint64_t p0 = (uint64_t)0xD2511F53u * c0, p1 = (uint64_t)0xCD9E8D57u * c2;
    uint32_t n0 = (uint32_t)(p1 >> 32) ^ c1 ^ k0, n1 = (uint32_t)p1, n2 = (uint32_t)(p0 >> 32) ^ c3 ^ k1, n3 = (uint32_t)p0;
    c0 = n0; c1 = n1; c2 = n2; c3 = n3;
  }
  out[0] = c0; out[1] = c1; out[2] = c2; out[3] = c3;
}

/* the stream's words, a block of four at a time */
typedef struct { crdist_stream *s; uint64_t blk; uint32_t w[4]; int have; } reader;
static uint32_t word(reader *r)
{
  uint64_t i = r->s->index++, j = i >> 2;
  if (!r->have || j != r->blk) {
    uint32_t ctr[4] = {(uint32_t)j, (uint32_t)(j >> 32), (uint32_t)r->s->substream, (uint32_t)(r->s->substream >> 32)};
    uint32_t key[2] = {(uint32_t)r->s->seed, (uint32_t)(r->s->seed >> 32)};
    philox(r->w, ctr, key);
    r->blk = j; r->have = 1;
  }
  return r->w[i & 3];
}

/* binary32 probit, correctly rounded (crdist.c's path: the decision, then the table) */
float crdist_one_internal(int f, float x);
static float probit32(float q) { return crdist_one_internal(CRDIST_PROBIT, q); }

static double ud(reader *r) { return (2.0 * word(r) + 1) * 0x1p-33; }   /* (2w + 1) 2^-33, exact */
static float nf(reader *r)
{
  uint32_t w = word(r);
  float q = (float)((2.0 * (w & 0x7fffffffu) + 1) * 0x1p-33);   /* exact in binary64, then rounded once */
  float p = probit32(q);
  return w >> 31 ? -p : p;
}

/* a gamma(a) variate in binary64 (Marsaglia and Tsang, 2000, without the squeeze), a > 0 finite */
static double gamma64(double a, reader *r)
{
  if (a < 1) {
    double g = gamma64(a + 1, r), u = ud(r);
    return g * cr_exp(cr_log(u) / a);
  }
  double d = a - 0x1.5555555555555p-2, c = 1 / __builtin_sqrt(9 * d);
  for (;;) {
    double z = nf(r), v = 1 + c * z;
    if (!(v > 0)) continue;
    v = v * v * v;
    double u = ud(r);
    if (cr_log(u) < ((0.5 * (z * z) + d) - d * v) + d * cr_log(v)) return d * v;
  }
}
/* its log, for beta: below 1, log g + log(u)/a, so that tiny shapes keep their ratio */
static double lgamma64(double a, reader *r)
{
  if (a < 1) {
    double g = gamma64(a + 1, r), u = ud(r);
    return cr_log(g) + cr_log(u) / a;
  }
  return cr_log(gamma64(a, r));
}

static int bad(double a) { return !(a > 0) || isinf(a); }
static void nans(float *y, size_t n) { for (size_t i = 0; i < n; i++) y[i] = NAN; }

void crdist_uniformf(float *y, size_t n, crdist_stream *s)
{
  reader r = {s, 0, {0}, 0};
  ENV_ENTER;
  for (size_t i = 0; i < n; i++) y[i] = (float)((2.0 * (word(&r) >> 9) + 1) * 0x1p-24);
  ENV_LEAVE;
}
void crdist_normalf(float *y, size_t n, crdist_stream *s)
{
  reader r = {s, 0, {0}, 0};
  ENV_ENTER;
  for (size_t i = 0; i < n; i++) y[i] = nf(&r);
  ENV_LEAVE;
}
void crdist_exponentialf(float *y, size_t n, crdist_stream *s)
{
  reader r = {s, 0, {0}, 0};
  ENV_ENTER;
  for (size_t i = 0; i < n; i++) y[i] = (float)-cr_log(ud(&r));
  ENV_LEAVE;
}
void crdist_gammaf(float *y, size_t n, double shape, crdist_stream *s)
{
  if (bad(shape)) { nans(y, n); return; }
  reader r = {s, 0, {0}, 0};
  ENV_ENTER;
  for (size_t i = 0; i < n; i++) y[i] = (float)gamma64(shape, &r);
  ENV_LEAVE;
}
void crdist_betaf(float *y, size_t n, double a, double b, crdist_stream *s)
{
  if (bad(a) || bad(b)) { nans(y, n); return; }
  reader r = {s, 0, {0}, 0};
  ENV_ENTER;
  for (size_t i = 0; i < n; i++) {
    double la = lgamma64(a, &r), lb = lgamma64(b, &r);
    y[i] = (float)(1 / (1 + cr_exp(lb - la)));
  }
  ENV_LEAVE;
}
void crdist_chisquaredf(float *y, size_t n, double k, crdist_stream *s)
{
  if (bad(k)) { nans(y, n); return; }
  reader r = {s, 0, {0}, 0};
  ENV_ENTER;
  for (size_t i = 0; i < n; i++) y[i] = (float)(2 * gamma64(k / 2, &r));
  ENV_LEAVE;
}
void crdist_student_tf(float *y, size_t n, double nu, crdist_stream *s)
{
  if (bad(nu)) { nans(y, n); return; }
  reader r = {s, 0, {0}, 0};
  ENV_ENTER;
  for (size_t i = 0; i < n; i++) {
    double z = nf(&r), g = gamma64(nu / 2, &r);
    y[i] = (float)(z / __builtin_sqrt((2 * g) / nu));
  }
  ENV_LEAVE;
}
void crdist_words(uint32_t *w, size_t n, crdist_stream *s)
{
  reader r = {s, 0, {0}, 0};
  for (size_t i = 0; i < n; i++) w[i] = word(&r);
}
