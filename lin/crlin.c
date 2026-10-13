/* crlin.c: crlin.h's verified solve (2026-10-13). */
#include <fenv.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "crlin.h"
#include "crsum.h"

#define ENV_ENTER fenv_t env_; fegetenv(&env_); fesetenv(FE_DFL_ENV)
#define ENV_LEAVE fesetenv(&env_)
/* threads (2026-10-13): rows of the O(n^2) work, and the columns of the inverse, go to OpenMP threads. Each row is
   computed the same way whichever thread takes it (the exact sums rounded once, the plain loops in the same order), so
   the bits don't depend on how many threads there are. Each thread sets the default floating-point environment itself
   and puts its own back: a pool's worker keeps the mode it was created with, which ENV_ENTER on the calling thread
   doesn't reach, and the exact sums and the bounds' upward steps need round-to-nearest without flushing. */
#define PAR_ENTER fenv_t tenv_; fegetenv(&tenv_); fesetenv(FE_DFL_ENV)
#define PAR_LEAVE fesetenv(&tenv_)
#define PAR_MIN 64   /* below this many rows, one thread */

const char *crlin_version(void) { return CRLIN_VERSION; }

/* an approximate inverse by LU with partial pivoting (no proof rests on it); 0, or -1 if a pivot is zero */
static int inverse(size_t n, const double *A, size_t lda, double *R)
{
  double *LU = malloc(n * n * sizeof *LU);
  size_t *p = malloc(n * sizeof *p);
  if (!LU || !p) { free(LU); free(p); return -1; }
  for (size_t i = 0; i < n; i++) { memcpy(LU + i * n, A + i * lda, n * sizeof *LU); p[i] = i; }
  int ok = 0;
  for (size_t k = 0; k < n && !ok; k++) {
    size_t m = k;
    for (size_t i = k + 1; i < n; i++) if (fabs(LU[i * n + k]) > fabs(LU[m * n + k])) m = i;
    if (LU[m * n + k] == 0) { ok = -1; break; }
    if (m != k) {
      for (size_t j = 0; j < n; j++) { double t = LU[k * n + j]; LU[k * n + j] = LU[m * n + j]; LU[m * n + j] = t; }
      size_t t = p[k]; p[k] = p[m]; p[m] = t;
    }
    double piv = LU[k * n + k];
#pragma omp parallel if (n - k >= 2 * PAR_MIN)
    {
      PAR_ENTER;
#pragma omp for schedule(static)
      for (size_t i = k + 1; i < n; i++) {
        double l = LU[i * n + k] / piv;
        LU[i * n + k] = l;
        for (size_t j = k + 1; j < n; j++) LU[i * n + j] -= l * LU[k * n + j];
      }
      PAR_LEAVE;
    }
  }
  if (!ok) {
    /* R = A^-1: solve L U R = P for the identity's permuted columns, row by row, in blocks of columns (each
       column's arithmetic the same whichever thread takes its block) */
    for (size_t i = 0; i < n; i++) for (size_t j = 0; j < n; j++) R[i * n + j] = p[i] == j;
#pragma omp parallel if (n >= PAR_MIN)
    {
      PAR_ENTER;
#pragma omp for schedule(static)
      for (size_t j0 = 0; j0 < n; j0 += 32) {
        size_t j1 = j0 + 32 < n ? j0 + 32 : n;
        for (size_t i = 0; i < n; i++)                   /* L, unit diagonal */
          for (size_t k = 0; k < i; k++) { double l = LU[i * n + k]; if (l != 0) for (size_t j = j0; j < j1; j++) R[i * n + j] -= l * R[k * n + j]; }
        for (size_t i = n; i-- > 0;) {                   /* U */
          for (size_t k = i + 1; k < n; k++) { double u = LU[i * n + k]; if (u != 0) for (size_t j = j0; j < j1; j++) R[i * n + j] -= u * R[k * n + j]; }
          double d = LU[i * n + i];
          for (size_t j = j0; j < j1; j++) R[i * n + j] /= d;
        }
      }
      PAR_LEAVE;
    }
  }
  free(LU); free(p);
  return ok;
}

/* y = b - A x exactly, rounded to nearest: each row an exact sum, so the rows can go to threads and the bits don't
   depend on how many */
static void residual(size_t n, const double *A, size_t lda, const double *b, const double *x, double *nx, double *y)
{
  for (size_t j = 0; j < n; j++) nx[j] = -x[j];
#pragma omp parallel if (n >= PAR_MIN)
  {
    PAR_ENTER;
#pragma omp for schedule(static)
    for (size_t i = 0; i < n; i++) {
      crsum_acc acc;
      crsum_init(&acc);
      crsum_add_dot(&acc, A + i * lda, nx, n);
      crsum_add(&acc, b + i, 1);
      y[i] = crsum_round(&acc, CRSUM_NEAREST);
    }
    PAR_LEAVE;
  }
}

/* the internal GEMM for CRLIN_CLASSICAL when none is given: C = A B, row-major, a classical sum of products */
static void gemm_plain(size_t m, size_t n, size_t k, const double *A, size_t lda, const double *B, size_t ldb, double *C,
                       size_t ldc, void *ctx)
{
  (void)ctx;
  for (size_t i = 0; i < m; i++) {
    double *Ci = C + i * ldc;
    for (size_t j = 0; j < n; j++) Ci[j] = 0;
    for (size_t l = 0; l < k; l++) { double a = A[i * lda + l]; const double *Bl = B + l * ldb; for (size_t j = 0; j < n; j++) Ci[j] += a * Bl[j]; }
  }
}
static double up(double t) { return nextafter(t, INFINITY); }   /* above any value t (rounded to nearest) stands for */

/* |I - R A| bounded from M = fl(R A) and S = fl(|R| |A|) (crlin.h): into C; 0, or -1 if anything overflowed */
static int classical(size_t n, const double *R, const double *A, size_t lda, double *C, crsum_dgemm gemm, void *ctx)
{
  double *aR = malloc(n * n * sizeof *aR), *aA = malloc(n * n * sizeof *aA), *S = malloc(n * n * sizeof *S);
  int r = -1;
  if (!aR || !aA || !S) goto done;
  if (!gemm) gemm = gemm_plain;
  for (size_t i = 0; i < n; i++) for (size_t j = 0; j < n; j++) { aR[i * n + j] = fabs(R[i * n + j]); aA[i * n + j] = fabs(A[i * lda + j]); }
  gemm(n, n, n, R, n, A, lda, C, n, ctx);                  /* M */
  gemm(n, n, n, aR, n, aA, n, S, n, ctx);                  /* S */
  /* the absolute term: 2^-1022 for each of the n products and n - 1 additions, which covers gradual underflow
     (eta/2 each) and flushing to zero alike, since a BLAS's worker threads keep whatever flush-to-zero mode they
     started with; subnormal inputs, which denormals-are-zero would zero before multiplying, don't reach here */
  const double u = 0x1p-53, tiny = 0x1p-1022, dn = (double)n;
  /* g = g_{n+1}/(1 - g_n), each step rounded up */
  double gn = up(up(dn * u) / (1 - up(dn * u))), gn1 = up(up((dn + 1) * u) / (1 - up((dn + 1) * u)));
  if (!(gn < 0.5)) goto done;                              /* n u too large for the bound to mean anything */
  double g = up(gn1 / (1 - gn)), neta = up(2 * dn * tiny), n1eta = up(2 * (dn + 1) * tiny);
  int inf = 0;
#pragma omp parallel if (n >= PAR_MIN) reduction(| : inf)
  {
    PAR_ENTER;
#pragma omp for schedule(static)
    for (size_t i = 0; i < n; i++)
      for (size_t j = 0; j < n; j++) {
        double m = C[i * n + j], s = S[i * n + j];
        if (!isfinite(m) || !isfinite(s)) inf = 1;
        double d = i == j ? up(fabs(1 - m)) : fabs(m);    /* |I - M| */
        C[i * n + j] = up(up(d + up(g * up(s + neta))) + n1eta);
      }
    PAR_LEAVE;
  }
  if (inf) goto done;
  r = 0;
done:
  free(aR); free(aA); free(S);
  return r;
}

/* |I - R A| bounded entry by entry into C; R left as it was; 0 by the classical bound, 1 by the exact product, or
   -1 */
static int bound_c(size_t n, const double *A, size_t lda, double *R, double *C, int mode, crsum_dgemm gemm, void *ctx)
{
  int sub = 0;   /* a subnormal entry: the classical bound would not survive denormals-are-zero in the GEMM */
  if (mode == CRLIN_CLASSICAL)
    for (size_t i = 0; i < n && !sub; i++)
      for (size_t j = 0; j < n; j++) {
        double r = fabs(R[i * n + j]), a = fabs(A[i * lda + j]);
        if ((r != 0 && r < 0x1p-1022) || (a != 0 && a < 0x1p-1022)) { sub = 1; break; }
      }
  if (mode == CRLIN_CLASSICAL && !sub) return classical(n, R, A, lda, C, gemm, ctx);   /* 0, or -1 */
  /* exactly, rounded to nearest; the exact value lies within an ulp, so its magnitude is at most the next number
     away from zero from the rounded one's (R negated for the product, and back after: both exact) */
  for (size_t i = 0; i < n; i++) for (size_t j = 0; j < n; j++) { C[i * n + j] = i == j; R[i * n + j] = -R[i * n + j]; }
  int r = crgemm_oz(n, n, n, R, n, A, lda, 1, C, n, CRSUM_NEAREST, gemm, ctx);
#pragma omp parallel if (n >= PAR_MIN)
  {
    PAR_ENTER;
#pragma omp for schedule(static)
    for (size_t k = 0; k < n * n; k++) { R[k] = -R[k]; C[k] = up(fabs(C[k])); }
    PAR_LEAVE;
  }
  return r ? -1 : 1;   /* 1: the exact product (asked for, or the classical mode's fallback) */
}

/* one right-hand side b, with R and the bound C shared: x~, the enclosures, the box; w holds 9 n doubles. Interval
   data (2026-10-13): A and b the midpoints, rA (n by n, row-major) and rb their radii, or NULL for point data */
static int column(size_t n, const double *A, size_t lda, const double *b, const double *R, const double *C,
                  double *x_lo, double *x_hi, double *w, const double *rA, const double *rb)
{
  double *x = w, *r = w + n, *nx = w + 2 * n, *zlo = w + 3 * n, *zhi = w + 4 * n, *az = w + 5 * n, *y = w + 6 * n, *v = w + 7 * n;
  double *rh = w + 8 * n;
  int status = CRLIN_FAILED;
  /* x~: R b, then two refinements with the exact residual */
  memset(x, 0, n * sizeof *x);
  for (int it = 0; it < 3; it++) {
    if (it) residual(n, A, lda, b, x, nx, r); else memcpy(r, b, n * sizeof *r);
#pragma omp parallel if (n >= 4 * PAR_MIN)
    {
      PAR_ENTER;
#pragma omp for schedule(static)
      for (size_t i = 0; i < n; i++) { double s = 0; for (size_t j = 0; j < n; j++) s += R[i * n + j] * r[j]; x[i] += s; }
      PAR_LEAVE;
    }
  }
  for (size_t i = 0; i < n; i++) if (!isfinite(x[i])) goto out;

  /* the residual's box [r, rh]. Point data: the exact b - A x~ lies within an ulp of its nearest rounding. Interval
     data: b - A x~ for every A and b in theirs lies between bm - rb - Am x~ - rA |x~| and bm + rb - Am x~ + rA |x~|,
     each an exact sum, rounded down and up */
  residual(n, A, lda, b, x, nx, r);
  if (!rA)
    for (size_t j = 0; j < n; j++) { rh[j] = nextafter(r[j], INFINITY); r[j] = nextafter(r[j], -INFINITY); }
  else {
    double *ax = malloc(2 * n * sizeof *ax), *nax = ax + n;
    if (!ax) goto out;
    for (size_t j = 0; j < n; j++) { ax[j] = fabs(x[j]); nax[j] = -ax[j]; }
#pragma omp parallel if (n >= PAR_MIN)
    {
      PAR_ENTER;
#pragma omp for schedule(static)
      for (size_t i = 0; i < n; i++) {
        crsum_acc acc;
        double e[2] = {b[i], -rb[i]};
        crsum_init(&acc);
        crsum_add_dot(&acc, A + i * lda, nx, n);
        crsum_add_dot(&acc, rA + i * n, nax, n);
        crsum_add(&acc, e, 2);
        r[i] = crsum_round(&acc, CRSUM_DOWN);
        e[1] = rb[i];
        crsum_init(&acc);
        crsum_add_dot(&acc, A + i * lda, nx, n);
        crsum_add_dot(&acc, rA + i * n, ax, n);
        crsum_add(&acc, e, 2);
        rh[i] = crsum_round(&acc, CRSUM_UP);
      }
      PAR_LEAVE;
    }
    free(ax);
  }
  /* z = R [r, rh]: for each row, the smallest and largest of the linear form over the box, each an exact sum
     rounded down or up */
  int nomem = 0;
#pragma omp parallel if (n >= PAR_MIN) reduction(| : nomem)
  {
    PAR_ENTER;
    double *vt = malloc(n * sizeof *vt);   /* each thread's own */
    if (!vt) nomem = 1;
#pragma omp for schedule(static)
    for (size_t i = 0; i < n; i++) {
      if (!vt) continue;
      const double *Ri = R + i * n;
      for (size_t j = 0; j < n; j++) vt[j] = Ri[j] >= 0 ? r[j] : rh[j];
      zlo[i] = crdot(Ri, vt, n, CRSUM_DOWN);
      for (size_t j = 0; j < n; j++) vt[j] = Ri[j] >= 0 ? rh[j] : r[j];
      zhi[i] = crdot(Ri, vt, n, CRSUM_UP);
      az[i] = fmax(fabs(zlo[i]), fabs(zhi[i]));
    }
    free(vt);
    PAR_LEAVE;
  }
  if (nomem) goto out;

  /* the box Y = [-y, y]: from |z|, inflated, until |z| + |C| y < y in every component (each left side an exact
     sum rounded up) */
  for (size_t i = 0; i < n; i++) y[i] = az[i];
  for (int it = 0; it < 12 && status == CRLIN_FAILED; it++) {
    for (size_t i = 0; i < n; i++) y[i] = y[i] * 1.0625 + 0x1p-1000;   /* epsilon-inflation: any y would do */
    int ok = 1;
#pragma omp parallel if (n >= PAR_MIN)
    {
      PAR_ENTER;
#pragma omp for schedule(static) reduction(& : ok)
      for (size_t i = 0; i < n; i++) {   /* every row, so that y grows from this round's values (stopping at the
                                            first failure, the first version, left the later rows stale and stalled) */
        v[i] = crdot(C + i * n, y, n, CRSUM_UP);                        /* |C| y, rounded up */
        double t[2] = {az[i], v[i]};
        nx[i] = crsum(t, 2, CRSUM_UP);                                  /* |z| + |C| y, rounded up */
        ok &= nx[i] < y[i];
      }
      PAR_LEAVE;
    }
    if (ok) status = CRLIN_VERIFIED;
    else for (size_t i = 0; i < n; i++) if (isfinite(nx[i])) y[i] = fmax(y[i], nx[i]);
  }
  if (status == CRLIN_VERIFIED)
    for (size_t i = 0; i < n; i++) {                                    /* x~ + z + C Y, each end rounded outward */
      double lo[3] = {x[i], zlo[i], -v[i]}, hi[3] = {x[i], zhi[i], v[i]};
      x_lo[i] = crsum(lo, 3, CRSUM_DOWN);
      x_hi[i] = crsum(hi, 3, CRSUM_UP);
    }
out:
  if (status != CRLIN_VERIFIED) for (size_t i = 0; i < n; i++) { x_lo[i] = -INFINITY; x_hi[i] = INFINITY; }
  return status;
}

/* E = B - A X exactly, rounded to nearest, all k columns (row-major, n by k; nA = -A): through crgemm_oz, or for a
   few columns one crgemv each, the same correctly rounded bits either way, but without crgemm_oz's GEMM calls */
static int residuals(size_t n, size_t k, const double *nA, const double *X, const double *B, size_t ldb, double *E,
                     double *xc, double *ec, crsum_dgemm gemm, void *ctx)
{
  if (k >= 8) {
    for (size_t i = 0; i < n; i++) for (size_t j = 0; j < k; j++) E[i * k + j] = B[i * ldb + j];
    return crgemm_oz(n, k, n, nA, n, X, k, 1, E, k, CRSUM_NEAREST, gemm, ctx) ? -1 : 0;
  }
  double *xn = malloc(n * sizeof *xn);
  if (!xn) return -1;
  for (size_t j = 0; j < k; j++) {   /* E = B - A X = B + nA X: residual() takes A and negates x itself */
    for (size_t i = 0; i < n; i++) { xc[i] = -X[i * k + j]; ec[i] = B[i * ldb + j]; }
    residual(n, nA, n, ec, xc, xn, ec);
    for (size_t i = 0; i < n; i++) E[i * k + j] = ec[i];
  }
  free(xn);
  return 0;
}

/* CRLIN_CLASSICAL with all the columns at once (2026-10-13): what column() does one column at a time, as matrix
   products. The residual stays exact (crgemm_oz, rounded to nearest, an ulp each way), since that is what makes the
   enclosure an ulp wide; R times the residual and |C| times the box come from the GEMM with a priori bounds, as C's
   own (its absolute term covering flush-to-zero): for P = R E with E the exact residual box, |R r - fl(R r)| <=
   g_n |R| |r| + 2 n t, plus |R| times the residual's ulp; for nonnegative factors, the exact product is at most
   fl(.)/(1 - g_n) + 2 n t. Those errors are of the order n u^2 of x, far below its ulp. W, of 8 n k doubles. 1 when
   every column verified; each column's enclosure, or -inf and +inf, written. */
static int batch(size_t n, size_t k, const double *A, size_t lda, const double *B, size_t ldb, const double *R,
                 const double *C, double *X_lo, double *X_hi, size_t ldx, crsum_dgemm gemm, void *ctx, double *W)
{
  size_t nk = n * k;
  double *X = W, *E = W + nk, *T = W + 2 * nk, *Z = W + 3 * nk, *Zr = W + 4 * nk, *Y = W + 5 * nk, *V = W + 6 * nk, *Q = W + 7 * nk;
  /* Y: the boxes' half-widths; V: fl(|C| Y); Q: first the bound quantity for Zr, then each column's |C| Y bound */
  double *nA = malloc(n * n * sizeof *nA), *aR = malloc(n * n * sizeof *aR), *xc = malloc(2 * n * sizeof *xc), *ec = xc + n;
  int *done = calloc(k, sizeof *done), *bad = calloc(k, sizeof *bad), all = 0;
  if (!nA || !aR || !xc || !done || !bad) goto out;
  if (!gemm) gemm = gemm_plain;
  for (size_t i = 0; i < n; i++) for (size_t j = 0; j < n; j++) { nA[i * n + j] = -A[i * lda + j]; aR[i * n + j] = fabs(R[i * n + j]); }
  const double u = 0x1p-53, tiny = 0x1p-1022, dn = (double)n;
  double gn = up(up(dn * u) / (1 - up(dn * u))), f = up(1 / (1 - gn)), ab = up(2 * (dn + 1) * tiny), ab2 = up(2 * ab);
  if (!(gn < 0.5)) goto out;
  /* X~ = R B, then two refinements with the exact residual: E = B - A X~ (A negated, B added exactly), X~ += R E */
  for (size_t i = 0; i < n; i++) for (size_t j = 0; j < k; j++) E[i * k + j] = B[i * ldb + j];
  memset(X, 0, nk * sizeof *X);
  for (int it = 0; it < 3; it++) {
    if (it && residuals(n, k, nA, X, B, ldb, E, xc, ec, gemm, ctx)) goto out;
    gemm(n, k, n, R, n, E, k, T, k, ctx);
    for (size_t q = 0; q < nk; q++) X[q] += T[q];
  }
  /* a column whose x~ is not finite fails by itself: its x~ set to 0 so that the products stay finite (columns don't
     mix in a matrix product), and it is never marked done */
  for (size_t q = 0; q < nk; q++) if (!isfinite(X[q])) bad[q % k] = 1;
  for (size_t q = 0; q < nk; q++) if (bad[q % k]) X[q] = 0;
  /* the exact residual, rounded to nearest: within an ulp of E */
  if (residuals(n, k, nA, X, B, ldb, E, xc, ec, gemm, ctx)) goto out;
  /* Z = fl(R E); Zr >= |R r - Z| for every exact residual r in E's box. Q = ulp(E) + g_n |E| bounds |r - E| plus
     fl(R E)'s relative error, so |R| Q covers both; a subnormal E entry, which denormals-are-zero would zero in the
     GEMM, counts in full (+ |E|), and Q is kept at least 2^-1022 (normal, so the GEMM sees it as given). fl(R E)'s
     absolute error and fl(|R| Q)'s are each at most 2 (n + 1) 2^-1022: ab2 */
  gemm(n, k, n, R, n, E, k, Z, k, ctx);
  for (size_t q = 0; q < nk; q++) {
    double e = fabs(E[q]), g = up(up(e) - e + up(gn * e));
    if (e != 0 && e < tiny) g = up(g + e);
    Q[q] = g < tiny ? tiny : g;
  }
  gemm(n, k, n, aR, n, Q, k, Zr, k, ctx);
  for (size_t q = 0; q < nk; q++) {
    if (!isfinite(Z[q]) || !isfinite(Zr[q])) bad[q % k] = 1;
    Zr[q] = up(up(Zr[q] * f) + ab2);
    T[q] = up(fabs(Z[q]) + Zr[q]);                                    /* |z|, up */
    Y[q] = T[q];
  }
  /* the boxes: |z| + |C| Y < Y in every column. |C| Y through the GEMM (C and Y nonnegative and normal: C's
     entries are at least 2 (n + 1) 2^-1022, Y's at least 2^-1000), bounded up by f and ab; a column that passes
     keeps its bound in Q (free now), its Y no longer inflated */
  for (int it = 0; it < 12; it++) {
    for (size_t q = 0; q < nk; q++) if (!done[q % k]) Y[q] = Y[q] * 1.0625 + 0x1p-1000;
    gemm(n, k, n, C, n, Y, k, V, k, ctx);
    int left = 0;
    for (size_t j = 0; j < k; j++) {
      if (done[j] || bad[j]) continue;
      int ok = 1;
      for (size_t i = 0; i < n; i++) {
        size_t q = i * k + j;
        double v = up(up(V[q] * f) + ab), w = up(T[q] + v);
        Q[q] = v;
        if (!(w < Y[q])) { ok = 0; if (isfinite(w)) Y[q] = fmax(Y[q], w); }
      }
      if (ok) done[j] = 1;
      else left++;
    }
    if (!left) break;
  }
  all = 1;
  for (size_t j = 0; j < k; j++) {
    if (!done[j]) { all = 0; for (size_t i = 0; i < n; i++) { X_lo[i * ldx + j] = -INFINITY; X_hi[i * ldx + j] = INFINITY; } continue; }
    for (size_t i = 0; i < n; i++) {                                    /* x~ + z + C Y, each end rounded outward */
      size_t q = i * k + j;
      double lo[4] = {X[q], Z[q], -Zr[q], -Q[q]}, hi[4] = {X[q], Z[q], Zr[q], Q[q]};
      X_lo[i * ldx + j] = crsum(lo, 4, CRSUM_DOWN);
      X_hi[i * ldx + j] = crsum(hi, 4, CRSUM_UP);
    }
  }
  if (!all) { free(nA); free(aR); free(xc); free(done); free(bad); return 0; }
out:
  if (!all) for (size_t j = 0; j < k; j++) if (!done || !done[j]) for (size_t i = 0; i < n; i++) { X_lo[i * ldx + j] = -INFINITY; X_hi[i * ldx + j] = INFINITY; }
  free(nA); free(aR); free(xc); free(done); free(bad);
  return all;
}

/* interval data (2026-10-13): C += |R| rA, bounded up. CRLIN_EXACT: the exact product rounded up (both factors
   nonnegative). CRLIN_CLASSICAL: fl(|R| rA), which for nonnegative factors is within g_n of the exact product plus
   2 (n + 1) 2^-1022 for flushing (rA's nonzero entries are at least 2^-1022, and a subnormal entry of R has already
   sent bound_c to the exact product, which mode then says); 0, or -1 */
static int add_radius(size_t n, const double *R, const double *rA, double *C, int mode, crsum_dgemm gemm, void *ctx)
{
  double *aR = malloc(n * n * sizeof *aR), *P = malloc(n * n * sizeof *P);
  int r = -1;
  if (!aR || !P) goto done;
  if (!gemm) gemm = gemm_plain;
  for (size_t k = 0; k < n * n; k++) aR[k] = fabs(R[k]);
  if (mode == CRLIN_EXACT) {
    if (crgemm_oz(n, n, n, aR, n, rA, n, 0, P, n, CRSUM_UP, gemm, ctx)) goto done;
  } else {
    gemm(n, n, n, aR, n, rA, n, P, n, ctx);
    const double u = 0x1p-53, dn = (double)n;
    double gn = up(up(dn * u) / (1 - up(dn * u))), f = up(1 / (1 - gn)), ab = up(2 * (dn + 1) * 0x1p-1022);
    if (!(gn < 0.5)) goto done;
    for (size_t k = 0; k < n * n; k++) P[k] = up(up(P[k] * f) + ab);
  }
  for (size_t k = 0; k < n * n; k++) { C[k] = up(C[k] + P[k]); if (!isfinite(C[k])) goto done; }
  r = 0;
done:
  free(aR); free(P);
  return r;
}

/* crlin_solve_many's work, and crlin_solve_interval's: A and B the midpoints, rA (n by n, row-major) and rB (n by
   nrhs) their radii, or NULL for point data. Interval data goes column by column (the batched classical path is for
   point data) */
static int solve(size_t n, size_t nrhs, const double *A, size_t lda, const double *B, size_t ldb, const double *rA,
                 const double *rB, const double *R_in, size_t ldr, double *X_lo, double *X_hi, size_t ldx, int mode,
                 crsum_dgemm gemm, void *ctx)
{
  int status = CRLIN_FAILED, ran = 0;
  ENV_ENTER;
  double *R = malloc(n * n * sizeof *R), *C = malloc(n * n * sizeof *C), *w = malloc(13 * n * sizeof *w);
  if (!R || !C || !w) goto out;
  if (R_in) for (size_t i = 0; i < n; i++) memcpy(R + i * n, R_in + i * ldr, n * sizeof *R);
  else if (inverse(n, A, lda, R)) goto out;
  int how = bound_c(n, A, lda, R, C, mode, gemm, ctx), classical_fell_back = how == 1;
  if (how < 0) goto out;
  if (rA && add_radius(n, R, rA, C, classical_fell_back ? CRLIN_EXACT : mode, gemm, ctx)) goto out;
  if (mode == CRLIN_CLASSICAL && !classical_fell_back && !rA) {
    double *W = malloc(8 * n * nrhs * sizeof *W);
    if (!W) goto out;
    ran = 1;
    status = batch(n, nrhs, A, lda, B, ldb, R, C, X_lo, X_hi, ldx, gemm, ctx, W) ? CRLIN_VERIFIED : CRLIN_FAILED;
    free(W);
    goto out;
  }
  double *b = w + 9 * n, *lo = w + 10 * n, *hi = w + 11 * n, *rb = w + 12 * n;
  int all = 1;
  ran = 1;
  for (size_t k = 0; k < nrhs; k++) {   /* each column writes its enclosure, or opens itself to +-inf */
    for (size_t i = 0; i < n; i++) { b[i] = B[i * ldb + k]; if (rB) rb[i] = rB[i * nrhs + k]; }
    all &= column(n, A, lda, b, R, C, lo, hi, w, rA, rB ? rb : NULL) == CRLIN_VERIFIED;
    for (size_t i = 0; i < n; i++) { X_lo[i * ldx + k] = lo[i]; X_hi[i * ldx + k] = hi[i]; }
  }
  if (all) status = CRLIN_VERIFIED;
out:
  if (!ran)   /* nothing proved for any column */
    for (size_t i = 0; i < n; i++) for (size_t k = 0; k < nrhs; k++) { X_lo[i * ldx + k] = -INFINITY; X_hi[i * ldx + k] = INFINITY; }
  free(R); free(C); free(w);
  ENV_LEAVE;
  return status;
}

int crlin_solve_many(size_t n, size_t nrhs, const double *A, size_t lda, const double *B, size_t ldb, const double *R_in,
                     size_t ldr, double *X_lo, double *X_hi, size_t ldx, int mode, crsum_dgemm gemm, void *ctx)
{
  if (!n || !nrhs || (mode != CRLIN_EXACT && mode != CRLIN_CLASSICAL)) return -1;
  for (size_t i = 0; i < n; i++) {
    for (size_t k = 0; k < nrhs; k++) if (!isfinite(B[i * ldb + k])) return -1;
    for (size_t j = 0; j < n; j++) if (!isfinite(A[i * lda + j])) return -1;
  }
  return solve(n, nrhs, A, lda, B, ldb, NULL, NULL, R_in, ldr, X_lo, X_hi, ldx, mode, gemm, ctx);
}

/* a midpoint m and a radius r with [lo, hi] inside [m - r, m + r]: m any double (lo/2 + hi/2, rounded), r the larger
   distance rounded up (0 when lo = hi), and at least 2^-1022 when not 0, so that denormals-are-zero in a GEMM
   can't drop it */
static void midrad(double lo, double hi, double *m, double *r)
{
  *m = 0.5 * lo + 0.5 * hi;
  double d = fmax(*m - lo, hi - *m);
  *r = d == 0 ? 0 : fmax(up(d), 0x1p-1022);
}

int crlin_solve_interval(size_t n, size_t nrhs, const double *A_lo, const double *A_hi, size_t lda, const double *B_lo,
                         const double *B_hi, size_t ldb, const double *R_in, size_t ldr, double *X_lo, double *X_hi,
                         size_t ldx, int mode, crsum_dgemm gemm, void *ctx)
{
  if (!n || !nrhs || (mode != CRLIN_EXACT && mode != CRLIN_CLASSICAL)) return -1;
  for (size_t i = 0; i < n; i++) {
    for (size_t k = 0; k < nrhs; k++) { double l = B_lo[i * ldb + k], h = B_hi[i * ldb + k]; if (!isfinite(l) || !isfinite(h) || !(l <= h)) return -1; }
    for (size_t j = 0; j < n; j++) { double l = A_lo[i * lda + j], h = A_hi[i * lda + j]; if (!isfinite(l) || !isfinite(h) || !(l <= h)) return -1; }
  }
  int status = CRLIN_FAILED;
  double *Am = malloc(n * n * sizeof *Am), *rA = malloc(n * n * sizeof *rA), *Bm = malloc(n * nrhs * sizeof *Bm),
         *rB = malloc(n * nrhs * sizeof *rB);
  if (Am && rA && Bm && rB) {
    ENV_ENTER;
    for (size_t i = 0; i < n; i++) {
      for (size_t j = 0; j < n; j++) midrad(A_lo[i * lda + j], A_hi[i * lda + j], &Am[i * n + j], &rA[i * n + j]);
      for (size_t k = 0; k < nrhs; k++) midrad(B_lo[i * ldb + k], B_hi[i * ldb + k], &Bm[i * nrhs + k], &rB[i * nrhs + k]);
    }
    ENV_LEAVE;
    status = solve(n, nrhs, Am, n, Bm, nrhs, rA, rB, R_in, ldr, X_lo, X_hi, ldx, mode, gemm, ctx);
  } else
    for (size_t i = 0; i < n; i++) for (size_t k = 0; k < nrhs; k++) { X_lo[i * ldx + k] = -INFINITY; X_hi[i * ldx + k] = INFINITY; }
  free(Am); free(rA); free(Bm); free(rB);
  return status;
}

int crlin_solve(size_t n, const double *A, size_t lda, const double *b, const double *R_in, size_t ldr, double *x_lo,
                double *x_hi, int mode, crsum_dgemm gemm, void *ctx)
{
  return crlin_solve_many(n, 1, A, lda, b, 1, R_in, ldr, x_lo, x_hi, 1, mode, gemm, ctx);
}

/* positive definiteness (2026-10-13). The Cholesky factor of A - c I (lower, n by n, row-major, zero above the
   diagonal), by columns: each entry one dot product of two rows' beginnings, the entries below the diagonal split
   among threads; 0, or -1 if a pivot is not positive. No proof rests on it. */
static int cholesky(size_t n, const double *A, size_t lda, double c, double *L)
{
  memset(L, 0, n * n * sizeof *L);
  for (size_t j = 0; j < n; j++) {
    const double *Lj = L + j * n;
    double s = A[j * lda + j] - c;
    for (size_t k = 0; k < j; k++) s -= Lj[k] * Lj[k];
    if (!(s > 0) || !isfinite(s)) return -1;
    double d = sqrt(s);
    L[j * n + j] = d;
#pragma omp parallel if ((n - j) * j >= PAR_MIN * PAR_MIN)
    {
      PAR_ENTER;
#pragma omp for schedule(static)
      for (size_t i = j + 1; i < n; i++) {
        const double *Li = L + i * n;
        double t = A[i * lda + j];
        for (size_t k = 0; k < j; k++) t -= Li[k] * Lj[k];
        L[i * n + j] = t / d;
      }
      PAR_LEAVE;
    }
  }
  return 0;
}

/* an estimate of L L^T's smallest eigenvalue, from above: inverse iteration, 12 steps from a fixed pseudo-random
   start (no proof rests on it); 0 if it overflowed. v and y: n doubles each */
static double smallest(size_t n, const double *L, double *v, double *y)
{
  uint64_t q = 0x9e3779b97f4a7c15ULL;
  double nrm = 0, est = 0;
  for (size_t i = 0; i < n; i++) { q = q * 6364136223846793005ULL + 1442695040888963407ULL; v[i] = (double)(q >> 11) * 0x1p-53 - 0.5; nrm += v[i] * v[i]; }
  nrm = sqrt(nrm);
  for (size_t i = 0; i < n; i++) v[i] /= nrm;
  for (int it = 0; it < 12; it++) {
    for (size_t i = 0; i < n; i++) {                       /* L y = v */
      const double *Li = L + i * n;
      double t = v[i];
      for (size_t k = 0; k < i; k++) t -= Li[k] * y[k];
      y[i] = t / Li[i];
    }
    for (size_t i = n; i-- > 0;) {                         /* L^T v = y, v overwritten */
      const double *Li = L + i * n;
      v[i] = y[i] / Li[i];
      for (size_t k = 0; k < i; k++) y[k] -= Li[k] * v[i];
    }
    nrm = 0;
    for (size_t i = 0; i < n; i++) nrm += v[i] * v[i];
    nrm = sqrt(nrm);
    if (!(nrm > 0) || !isfinite(nrm)) return 0;
    est = 1 / nrm;                                         /* ||(L L^T)^-1 v|| <= 1/lambda_min for a unit v */
    for (size_t i = 0; i < n; i++) v[i] /= nrm;
  }
  return est;
}

/* an upper bound on ||E||_2, E = A - c I - L L^T (L any n by n matrix, row-major): F >= |E| entry by entry, made
   symmetric as E is (the larger of F_ij and F_ji), and its largest row sum rounded up, which bounds the spectral
   radius of |E| and so ||E||_2. In CRLIN_EXACT, L L^T exactly (crgemm_oz), A - L L^T rounded to nearest; in
   CRLIN_CLASSICAL, M = fl(L L^T) and S = fl(|L| |L^T|) with the a priori bound of classical() (a subnormal entry of
   L: the exact product instead). NaN if anything overflowed, or for want of memory. */
static double bound_e(size_t n, const double *A, size_t lda, const double *L, double c, int mode, crsum_dgemm gemm, void *ctx)
{
  double *Lt = malloc(n * n * sizeof *Lt), *D = malloc(n * n * sizeof *D), *X = malloc(n * n * sizeof *X), *S = NULL;
  double beta = NAN;
  if (!Lt || !D || !X) goto done;
  if (!gemm) gemm = gemm_plain;
  for (size_t i = 0; i < n; i++) for (size_t j = 0; j < n; j++) Lt[j * n + i] = L[i * n + j];
  int sub = 0;
  for (size_t k = 0; k < n * n && !sub; k++) { double a = fabs(L[k]); sub = a != 0 && a < 0x1p-1022; }
  if (mode == CRLIN_CLASSICAL && !sub) {
    S = malloc(n * n * sizeof *S);
    if (!S) goto done;
    gemm(n, n, n, L, n, Lt, n, D, n, ctx);                 /* M */
    for (size_t k = 0; k < n * n; k++) { X[k] = fabs(L[k]); }
    double *aLt = Lt;                                      /* Lt is not needed again: |L^T| in its place */
    for (size_t k = 0; k < n * n; k++) aLt[k] = fabs(aLt[k]);
    gemm(n, n, n, X, n, aLt, n, S, n, ctx);                /* S */
    const double u = 0x1p-53, tiny = 0x1p-1022, dn = (double)n;
    double gn = up(up(dn * u) / (1 - up(dn * u))), gn1 = up(up((dn + 1) * u) / (1 - up((dn + 1) * u)));
    if (!(gn < 0.5)) goto done;
    double g = up(gn1 / (1 - gn)), neta = up(2 * dn * tiny), n1eta = up(2 * (dn + 1) * tiny);
#pragma omp parallel if (n >= PAR_MIN)
    {
      PAR_ENTER;
#pragma omp for schedule(static)
      for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < n; j++) {
          double m = D[i * n + j], a = A[i * lda + j], d;
          if (i == j) {                                    /* |A_ii - c - M_ii|, from its exact sum rounded each way */
            double t[3] = {a, -c, -m};
            d = fmax(fabs(crsum(t, 3, CRSUM_DOWN)), fabs(crsum(t, 3, CRSUM_UP)));
          } else d = up(fabs(a - m));
          X[i * n + j] = up(up(d + up(g * up(S[i * n + j] + neta))) + n1eta);
        }
      PAR_LEAVE;
    }
  } else {
    for (size_t i = 0; i < n; i++) { memcpy(D + i * n, A + i * lda, n * sizeof *D); for (size_t j = 0; j < n; j++) X[i * n + j] = -L[i * n + j]; }
    if (crgemm_oz(n, n, n, X, n, Lt, n, 1, D, n, CRSUM_NEAREST, gemm, ctx)) goto done;   /* A - L L^T, to nearest */
#pragma omp parallel if (n >= PAR_MIN)
    {
      PAR_ENTER;
#pragma omp for schedule(static)
      for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < n; j++) {
          double e = D[i * n + j];                         /* the exact value lies within half an ulp of e */
          if (i == j) {
            double lo[2] = {nextafter(e, -INFINITY), -c}, hi[2] = {nextafter(e, INFINITY), -c};
            X[i * n + j] = fmax(fabs(crsum(lo, 2, CRSUM_DOWN)), fabs(crsum(hi, 2, CRSUM_UP)));
          } else X[i * n + j] = up(fabs(e));
        }
      PAR_LEAVE;
    }
  }
  /* symmetric, then each row's sum rounded up (D reused for the rows) */
  double worst = 0;
  int bad = 0;
#pragma omp parallel if (n >= PAR_MIN) reduction(max : worst) reduction(| : bad)
  {
    PAR_ENTER;
#pragma omp for schedule(static)
    for (size_t i = 0; i < n; i++) {
      double *row = D + i * n;
      for (size_t j = 0; j < n; j++) row[j] = fmax(X[i * n + j], X[j * n + i]);
      double r = crsum(row, n, CRSUM_UP);
      if (!isfinite(r)) bad = 1;
      else worst = fmax(worst, r);
    }
    PAR_LEAVE;
  }
  if (!bad) beta = worst;
done:
  free(Lt); free(D); free(X); free(S);
  return beta;
}

/* crlin_spd without a factor given: A's own factor, a little shifted; then c about half the smallest eigenvalue that
   inverse iteration finds, so that the bound comes out near half of it (a factor that fails halves c, three times,
   and a bound that fails goes back to the first factor). The bound c - ||E||_2 rounded down into *lam, or -inf;
   L, L2: n^2 doubles each, v: 2 n */
static void spd_auto(size_t n, const double *A, size_t lda, int mode, crsum_dgemm gemm, void *ctx, double *L, double *L2,
                     double *v, double *lam)
{
  *lam = -INFINITY;
  double dmax = 0;
  for (size_t i = 0; i < n; i++) dmax = fmax(dmax, A[i * lda + i]);
  double c0 = up(4 * (double)n * 0x1p-53 * dmax), c = c0;
  if (!(dmax > 0) || cholesky(n, A, lda, c0, L)) return;
  double est = smallest(n, L, v, v + n) + c0;
  for (int t = 0; t < 3 && est / 2 > c0; t++, est /= 2)
    if (!cholesky(n, A, lda, est / 2, L2)) { memcpy(L, L2, n * n * sizeof *L); c = est / 2; break; }
  double beta = bound_e(n, A, lda, L, c, mode, gemm, ctx);
  if (!(beta < c) && c != c0 && !cholesky(n, A, lda, c0, L)) { c = c0; beta = bound_e(n, A, lda, L, c, mode, gemm, ctx); }
  if (isnan(beta)) return;
  double t[2] = {c, -beta};
  *lam = crsum(t, 2, CRSUM_DOWN);                          /* lambda_min(A) >= c - ||E||_2 */
}

int crlin_spd(size_t n, const double *A, size_t lda, const double *L_in, size_t ldl, double c, double *lambda_lo, int mode,
              crsum_dgemm gemm, void *ctx)
{
  if (!n || !lambda_lo || (mode != CRLIN_EXACT && mode != CRLIN_CLASSICAL) || (L_in && !isfinite(c))) return -1;
  for (size_t i = 0; i < n; i++)
    for (size_t j = 0; j < n; j++) {
      if (!isfinite(A[i * lda + j]) || A[i * lda + j] != A[j * lda + i]) return -1;
      if (L_in && !isfinite(L_in[i * ldl + j])) return -1;
    }
  *lambda_lo = -INFINITY;
  int status = CRLIN_FAILED;
  ENV_ENTER;
  double *L = malloc(n * n * sizeof *L), *L2 = malloc(n * n * sizeof *L2), *v = malloc(2 * n * sizeof *v), *As = NULL;
  int *e = NULL;
  if (!L || !L2 || !v) goto out;
  if (L_in) {
    for (size_t i = 0; i < n; i++) memcpy(L + i * n, L_in + i * ldl, n * sizeof *L);
    double beta = bound_e(n, A, lda, L, c, mode, gemm, ctx);
    if (!isnan(beta)) { double t[2] = {c, -beta}; *lambda_lo = crsum(t, 2, CRSUM_DOWN); }
  } else {
    spd_auto(n, A, lda, mode, gemm, ctx, L, L2, v, lambda_lo);
    /* failing that, scaled to a diagonal near 1 by powers of two: D A D with D = diag(2^e_i), exact unless an entry
       leaves the normal range (then not tried), and positive definite exactly when A is. A's smallest eigenvalue is
       at least D A D's times the least 2^(-2 e_i): x^T A x = y^T (D A D) y with y = D^-1 x */
    As = malloc(n * n * sizeof *As);
    e = malloc(n * sizeof *e);
    if (!(*lambda_lo > 0) && As && e) {
      int emax = INT_MIN, exact = 1;
      for (size_t i = 0; i < n; i++) {
        int k;
        frexp(A[i * lda + i], &k);
        e[i] = A[i * lda + i] > 0 ? -(k >= 0 ? k / 2 : -((1 - k) / 2)) : 0;
        if (e[i] > emax) emax = e[i];
      }
      for (size_t i = 0; i < n && exact; i++)
        for (size_t j = 0; j < n; j++) {
          double a = A[i * lda + j], b = ldexp(a, e[i] + e[j]);
          if (ldexp(b, -(e[i] + e[j])) != a || (b != 0 && fabs(b) < 0x1p-1022) || !isfinite(b)) { exact = 0; break; }
          As[i * n + j] = b;
        }
      double lam;
      if (exact) spd_auto(n, As, n, mode, gemm, ctx, L, L2, v, &lam);
      if (exact && lam > 0) {
        double l = ldexp(lam, -2 * emax);                  /* rounded down where it is not exact */
        if (ldexp(l, 2 * emax) != lam) l = nextafter(l, 0);
        *lambda_lo = l;
        if (!(l > 0)) status = CRLIN_VERIFIED;             /* proved, the bound underflowing (crlin.h) */
      }
    }
  }
  if (*lambda_lo > 0) status = CRLIN_VERIFIED;
out:
  free(L); free(L2); free(v); free(As); free(e);
  ENV_LEAVE;
  return status;
}
