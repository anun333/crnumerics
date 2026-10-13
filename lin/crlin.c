/* crlin.c: crlin.h's verified solve (2026-10-13). */
#include <fenv.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "crlin.h"
#include "crsum.h"

#define ENV_ENTER fenv_t env_; fegetenv(&env_); fesetenv(FE_DFL_ENV)
#define ENV_LEAVE fesetenv(&env_)

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
    for (size_t i = k + 1; i < n; i++) {
      double l = LU[i * n + k] / piv;
      LU[i * n + k] = l;
      for (size_t j = k + 1; j < n; j++) LU[i * n + j] -= l * LU[k * n + j];
    }
  }
  if (!ok) {
    /* R = A^-1: solve L U R = P for the identity's permuted columns, all columns at once, row by row */
    for (size_t i = 0; i < n; i++) for (size_t j = 0; j < n; j++) R[i * n + j] = p[i] == j;
    for (size_t i = 0; i < n; i++)                       /* L, unit diagonal */
      for (size_t k = 0; k < i; k++) { double l = LU[i * n + k]; if (l != 0) for (size_t j = 0; j < n; j++) R[i * n + j] -= l * R[k * n + j]; }
    for (size_t i = n; i-- > 0;) {                       /* U */
      for (size_t k = i + 1; k < n; k++) { double u = LU[i * n + k]; if (u != 0) for (size_t j = 0; j < n; j++) R[i * n + j] -= u * R[k * n + j]; }
      double d = LU[i * n + i];
      for (size_t j = 0; j < n; j++) R[i * n + j] /= d;
    }
  }
  free(LU); free(p);
  return ok;
}

/* y = b - A x exactly, rounded to nearest (crgemv adds beta y exactly) */
static void residual(size_t n, const double *A, size_t lda, const double *b, const double *x, double *nx, double *y)
{
  for (size_t j = 0; j < n; j++) nx[j] = -x[j];
  memcpy(y, b, n * sizeof *y);
  crgemv(0, n, n, A, lda, nx, 1, y, CRSUM_NEAREST);
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
  for (size_t i = 0; i < n; i++)
    for (size_t j = 0; j < n; j++) {
      double m = C[i * n + j], s = S[i * n + j];
      if (!isfinite(m) || !isfinite(s)) goto done;
      double d = i == j ? up(fabs(1 - m)) : fabs(m);      /* |I - M| */
      C[i * n + j] = up(up(d + up(g * up(s + neta))) + n1eta);
    }
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
  for (size_t k = 0; k < n * n; k++) { R[k] = -R[k]; C[k] = up(fabs(C[k])); }
  return r ? -1 : 1;   /* 1: the exact product (asked for, or the classical mode's fallback) */
}

/* one right-hand side b, with R and the bound C shared: x~, the enclosures, the box; w holds 8 n doubles */
static int column(size_t n, const double *A, size_t lda, const double *b, const double *R, const double *C,
                  double *x_lo, double *x_hi, double *w)
{
  double *x = w, *r = w + n, *nx = w + 2 * n, *zlo = w + 3 * n, *zhi = w + 4 * n, *az = w + 5 * n, *y = w + 6 * n, *v = w + 7 * n;
  int status = CRLIN_FAILED;
  /* x~: R b, then two refinements with the exact residual */
  memset(x, 0, n * sizeof *x);
  for (int it = 0; it < 3; it++) {
    if (it) residual(n, A, lda, b, x, nx, r); else memcpy(r, b, n * sizeof *r);
    for (size_t i = 0; i < n; i++) { double s = 0; for (size_t j = 0; j < n; j++) s += R[i * n + j] * r[j]; x[i] += s; }
  }
  for (size_t i = 0; i < n; i++) if (!isfinite(x[i])) goto out;

  /* the exact residual b - A x~ lies within an ulp of its nearest rounding r: [r-, r+] */
  residual(n, A, lda, b, x, nx, r);
  /* z = R [r-, r+]: for each row, the smallest and largest of the linear form over the box, each an exact sum
     rounded down or up */
  for (size_t i = 0; i < n; i++) {
    const double *Ri = R + i * n;
    for (size_t j = 0; j < n; j++) v[j] = Ri[j] >= 0 ? nextafter(r[j], -INFINITY) : nextafter(r[j], INFINITY);
    zlo[i] = crdot(Ri, v, n, CRSUM_DOWN);
    for (size_t j = 0; j < n; j++) v[j] = Ri[j] >= 0 ? nextafter(r[j], INFINITY) : nextafter(r[j], -INFINITY);
    zhi[i] = crdot(Ri, v, n, CRSUM_UP);
    az[i] = fmax(fabs(zlo[i]), fabs(zhi[i]));
  }

  /* the box Y = [-y, y]: from |z|, inflated, until |z| + |C| y < y in every component (each left side an exact
     sum rounded up) */
  for (size_t i = 0; i < n; i++) y[i] = az[i];
  for (int it = 0; it < 12 && status == CRLIN_FAILED; it++) {
    for (size_t i = 0; i < n; i++) y[i] = y[i] * 1.0625 + 0x1p-1000;   /* epsilon-inflation: any y would do */
    int ok = 1;
    for (size_t i = 0; i < n; i++) {   /* every row, so that y grows from this round's values (stopping at the first
                                          failure, the first version, left the later rows stale and stalled) */
      v[i] = crdot(C + i * n, y, n, CRSUM_UP);                          /* |C| y, rounded up */
      double t[2] = {az[i], v[i]};
      nx[i] = crsum(t, 2, CRSUM_UP);                                    /* |z| + |C| y, rounded up */
      ok &= nx[i] < y[i];
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
  for (size_t j = 0; j < k; j++) {
    for (size_t i = 0; i < n; i++) { xc[i] = X[i * k + j]; ec[i] = B[i * ldb + j]; }
    crgemv(0, n, n, nA, n, xc, 1, ec, CRSUM_NEAREST);
    for (size_t i = 0; i < n; i++) E[i * k + j] = ec[i];
  }
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

int crlin_solve_many(size_t n, size_t nrhs, const double *A, size_t lda, const double *B, size_t ldb, const double *R_in,
                     size_t ldr, double *X_lo, double *X_hi, size_t ldx, int mode, crsum_dgemm gemm, void *ctx)
{
  if (!n || !nrhs || (mode != CRLIN_EXACT && mode != CRLIN_CLASSICAL)) return -1;
  for (size_t i = 0; i < n; i++) {
    for (size_t k = 0; k < nrhs; k++) if (!isfinite(B[i * ldb + k])) return -1;
    for (size_t j = 0; j < n; j++) if (!isfinite(A[i * lda + j])) return -1;
  }
  int status = CRLIN_FAILED, ran = 0;
  ENV_ENTER;
  double *R = malloc(n * n * sizeof *R), *C = malloc(n * n * sizeof *C), *w = malloc(11 * n * sizeof *w);
  if (!R || !C || !w) goto out;
  if (R_in) for (size_t i = 0; i < n; i++) memcpy(R + i * n, R_in + i * ldr, n * sizeof *R);
  else if (inverse(n, A, lda, R)) goto out;
  int how = bound_c(n, A, lda, R, C, mode, gemm, ctx), classical_fell_back = how == 1;
  if (how < 0) goto out;
  if (mode == CRLIN_CLASSICAL && !classical_fell_back) {
    double *W = malloc(8 * n * nrhs * sizeof *W);
    if (!W) goto out;
    ran = 1;
    status = batch(n, nrhs, A, lda, B, ldb, R, C, X_lo, X_hi, ldx, gemm, ctx, W) ? CRLIN_VERIFIED : CRLIN_FAILED;
    free(W);
    goto out;
  }
  double *b = w + 8 * n, *lo = w + 9 * n, *hi = w + 10 * n;
  int all = 1;
  ran = 1;
  for (size_t k = 0; k < nrhs; k++) {   /* each column writes its enclosure, or opens itself to +-inf */
    for (size_t i = 0; i < n; i++) b[i] = B[i * ldb + k];
    all &= column(n, A, lda, b, R, C, lo, hi, w) == CRLIN_VERIFIED;
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

int crlin_solve(size_t n, const double *A, size_t lda, const double *b, const double *R_in, size_t ldr, double *x_lo,
                double *x_hi, int mode, crsum_dgemm gemm, void *ctx)
{
  return crlin_solve_many(n, 1, A, lda, b, 1, R_in, ldr, x_lo, x_hi, 1, mode, gemm, ctx);
}
