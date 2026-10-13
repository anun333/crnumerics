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

/* log k! (crdist.h): the table below 128 (each entry log k! rounded once to binary64, from MPFR's
   lngamma), Stirling's series above, its first omitted term t^5/1260 below 3e-14 */
static const double crdist_lfact_tab[128] = {
  0x0p+0, 0x0p+0, 0x1.62e42fefa39efp-1, 0x1.cab0bfa2a2002p+0,
  0x1.96ca77c922cf9p+1, 0x1.326643c4479c9p+2, 0x1.a51273acf01cap+2, 0x1.10ce1f32dcc3p+3,
  0x1.5358e82fcb70dp+3, 0x1.99a8921a7f7cfp+3, 0x1.e357590954d15p+3, 0x1.180973f3a8d74p+4,
  0x1.3fcba16d50143p+4, 0x1.68d5a9c3b32cep+4, 0x1.930f3df162a42p+4, 0x1.be636a63fd346p+4,
  0x1.eabff061f1a84p+4, 0x1.0c0a63f2f353ap+5, 0x1.2329df2d5ee52p+5, 0x1.3ab8153363985p+5,
  0x1.52af57aed77bep+5, 0x1.6b0a8643472a9p+5, 0x1.83c4faba84f06p+5, 0x1.9cda78b856a45p+5,
  0x1.b6472034e8d14p+5, 0x1.d007622cd65e7p+5, 0x1.ea17f717c6794p+5, 0x1.023aeb67e4fefp+6,
  0x1.0f8f18d33024p+6, 0x1.1d07353917231p+6, 0x1.2aa208b59d0e5p+6, 0x1.385e6fd9e5a4p+6,
  0x1.463b59b942084p+6, 0x1.5437c633ace4ap+6, 0x1.6252c474896bap+6, 0x1.708b719e11658p+6,
  0x1.7ee0f79b26758p+6, 0x1.8d528c1243d96p+6, 0x1.9bdf6f75257a3p+6, 0x1.aa86ec2969812p+6,
  0x1.b94855c702ba2p+6, 0x1.c8230869ca105p+6, 0x1.d7166813e12eep+6, 0x1.e621e01eeba4fp+6,
  0x1.f544e2ba69cf1p+6, 0x1.023f743addd9fp+7, 0x1.09e7b7ea41ea9p+7, 0x1.119afe762626bp+7,
  0x1.19590c853a559p+7, 0x1.2121a930c6ec3p+7, 0x1.28f49ddeb1f31p+7, 0x1.30d1b61e86335p+7,
  0x1.38b8bf8931ddbp+7, 0x1.40a989a33a6cdp+7, 0x1.48a3e5c12af19p+7, 0x1.50a7a6ee08711p+7,
  0x1.58b4a1d39da73p+7, 0x1.60caaca474746p+7, 0x1.68e99f0757979p+7, 0x1.711152043b2c4p+7,
  0x1.79419ff26dc59p+7, 0x1.817a6467f6fb9p+7, 0x1.89bb7c2a0aea1p+7, 0x1.9204c51e7c761p+7,
  0x1.9a561e3e1a4bdp+7, 0x1.a2af6787e4609p+7, 0x1.ab1081f509726p+7, 0x1.b3794f6d9d7afp+7,
  0x1.bbe9b2bdfb621p+7, 0x1.c4618f8cc56f7p+7, 0x1.cce0ca51791p+7, 0x1.d567484b8b7b6p+7,
  0x1.ddf4ef7a05a7p+7, 0x1.e689a69396befp+7, 0x1.ef2554ff15148p+7, 0x1.f7c7e2cc66183p+7,
  0x1.00389c56e3462p+8, 0x1.04909ff8b652bp+8, 0x1.08ebf13dbf263p+8, 0x1.0d4a85602b129p+8,
  0x1.11ac51df8932ap+8, 0x1.16114c7e34736p+8, 0x1.1a796b3ede1acp+8, 0x1.1ee4a46236d3ep+8,
  0x1.2352ee64b46d5p+8, 0x1.27c43ffc72962p+8, 0x1.2c3890172d057p+8, 0x1.30afd5d851956p+8,
  0x1.352a089728f1bp+8, 0x1.39a71fdd14947p+8, 0x1.3e271363e0df7p+8, 0x1.42a9db142a36ap+8,
  0x1.472f6f03d410cp+8, 0x1.4bb7c77491066p+8, 0x1.5042dcd27af64p+8, 0x1.54d0a7b2ba658p+8,
  0x1.596120d23c4ecp+8, 0x1.5df4411475a1cp+8, 0x1.628a018233bedp+8, 0x1.67225b4879462p+8,
  0x1.6bbd47b7669b6p+8, 0x1.705ac0412d89fp+8, 0x1.74fabe790f7bep+8, 0x1.799d3c1265c0ep+8,
  0x1.7e4232dfb367dp+8, 0x1.82e99cd1c0368p+8, 0x1.879373f6bc4fep+8, 0x1.8c3fb2796c21cp+8,
  0x1.90ee52a05c35fp+8, 0x1.959f4ecd1c8b3p+8, 0x1.9a52a17b831ccp+8, 0x1.9f084540f545ep+8,
  0x1.a3c034cbb7b2cp+8, 0x1.a87a6ae24493ap+8, 0x1.ad36e262a7ccp+8, 0x1.b1f59641e0db5p+8,
  0x1.b6b6818b4a3ebp+8, 0x1.bb799f600610ap+8, 0x1.c03eeaf66facdp+8, 0x1.c5065f9992226p+8,
  0x1.c9cff8a8a340dp+8, 0x1.ce9bb196830eap+8, 0x1.d36985e93f7b8p+8, 0x1.d83971399c213p+8,
  0x1.dd0b6f329dea4p+8, 0x1.e1df7b911a74cp+8, 0x1.e6b592234b0c9p+8, 0x1.eb8daec863182p+8};
static double lfact(double k)
{
  if (k < 128) return crdist_lfact_tab[(int)k];
  double t = 1 / k;
  return ((k + 0.5) * cr_log(k) - k) + (0x1.d67f1c864beb5p-1 + t * (1.0 / 12 - (t * t) / 360));
}

/* Poisson(lam), 0 <= lam <= 1e15 (crdist.h): inversion below 10, Hormann's PTRS (1993, as in NumPy) above */
static double poisson64(double lam, reader *r)
{
  if (lam == 0) return 0;
  if (lam < 10) {
    double u = ud(r), p = cr_exp(-lam), f = p, k = 0;
    while (u > f && k < 1000) { k = k + 1; p = (p * lam) / k; f = f + p; }
    return k;
  }
  double slam = __builtin_sqrt(lam), loglam = cr_log(lam), b = 0.931 + 2.53 * slam, a = -0.059 + 0.02483 * b;
  double invalpha = 1.1239 + 1.1328 / (b - 3.4), vr = 0.9277 - 3.6224 / (b - 2);
  for (;;) {
    double U = ud(r) - 0.5, V = ud(r), us = 0.5 - fabs(U), k = floor((2 * a / us + b) * U + lam + 0.43);
    if (us >= 0.07 && V <= vr) return k;
    if (k < 0 || (us < 0.013 && V > us)) continue;
    if ((cr_log(V) + cr_log(invalpha)) - cr_log(a / (us * us) + b) <= (-lam + k * loglam) - lfact(k)) return k;
  }
}

/* binomial(n, p), p <= 1/2, n p > 0 (crdist.h): inversion (BINV, as in NumPy) below n p = 10, Hormann's BTRS
   (1993, as in TensorFlow and JAX) above */
static double binomial64(double n, double p, reader *r)
{
  double q = 1 - p, np = n * p;
  if (np < 10) {
    double qn = cr_exp(n * cr_log1p(-p)), bound = fmin(n, np + 10 * __builtin_sqrt(np * q + 1));
    double u = ud(r), px = qn, k = 0;
    while (u > px) {
      k = k + 1;
      if (k > bound) { k = 0; px = qn; u = ud(r); }
      else { u = u - px; px = (((n - k + 1) * p) * px) / (k * q); }
    }
    return k;
  }
  double spq = __builtin_sqrt(np * q), b = 1.15 + 2.53 * spq, a = (-0.0873 + 0.0248 * b) + 0.01 * p, c = np + 0.5;
  double vr = 0.92 - 4.2 / b, alpha = (2.83 + 5.1 / b) * spq, lpq = cr_log(p / q), m = floor((n + 1) * p);
  double h = lfact(m) + lfact(n - m);
  for (;;) {
    double U = ud(r) - 0.5, V = ud(r), us = 0.5 - fabs(U), k = floor((2 * a / us + b) * U + c);
    if (k < 0 || k > n) continue;
    if (us >= 0.07 && V <= vr) return k;
    if (cr_log((V * alpha) / (a / (us * us) + b)) <= ((h - lfact(k)) - lfact(n - k)) + (k - m) * lpq) return k;
  }
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
void crdist_poisson(int64_t *y, size_t n, double lam, crdist_stream *s)
{
  if (!(lam >= 0 && lam <= 1e15)) { for (size_t i = 0; i < n; i++) y[i] = -1; return; }
  reader r = {s, 0, {0}, 0};
  ENV_ENTER;
  for (size_t i = 0; i < n; i++) y[i] = (int64_t)poisson64(lam, &r);
  ENV_LEAVE;
}
void crdist_binomial(int64_t *y, size_t n, int64_t trials, double p, crdist_stream *s)
{
  if (!(p >= 0 && p <= 1) || trials < 0 || trials > (1LL << 53)) { for (size_t i = 0; i < n; i++) y[i] = -1; return; }
  reader r = {s, 0, {0}, 0};
  double t = (double)trials;
  ENV_ENTER;
  for (size_t i = 0; i < n; i++) {
    if (trials == 0 || p == 0) y[i] = 0;
    else if (p == 1) y[i] = trials;
    else if (p > 0.5) y[i] = trials - (int64_t)binomial64(t, 1 - p, &r);
    else y[i] = (int64_t)binomial64(t, p, &r);
  }
  ENV_LEAVE;
}
void crdist_words(uint32_t *w, size_t n, crdist_stream *s)
{
  reader r = {s, 0, {0}, 0};
  for (size_t i = 0; i < n; i++) w[i] = word(&r);
}
