/* crlin.c: crlin.h's verified solve (2026-10-13). */
#include <fenv.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "crlin.h"

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

int crlin_solve(size_t n, const double *A, size_t lda, const double *b, const double *R_in, size_t ldr, double *x_lo,
                double *x_hi, crsum_dgemm gemm, void *ctx)
{
  if (!n) return -1;
  for (size_t i = 0; i < n; i++) {
    if (!isfinite(b[i])) return -1;
    for (size_t j = 0; j < n; j++) if (!isfinite(A[i * lda + j])) return -1;
  }
  int status = CRLIN_FAILED;
  ENV_ENTER;
  double *R = malloc(n * n * sizeof *R), *C = malloc(n * n * sizeof *C), *w = malloc(8 * n * sizeof *w);
  if (!R || !C || !w) goto out;
  double *x = w, *r = w + n, *nx = w + 2 * n, *zlo = w + 3 * n, *zhi = w + 4 * n, *az = w + 5 * n, *y = w + 6 * n, *v = w + 7 * n;
  if (R_in) for (size_t i = 0; i < n; i++) memcpy(R + i * n, R_in + i * ldr, n * sizeof *R);
  else if (inverse(n, A, lda, R)) goto out;

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

  /* C = I - R A, exactly, rounded to nearest; the exact value lies within an ulp, so |C| <= the next number away
     from zero from |C| as rounded */
  for (size_t i = 0; i < n; i++) for (size_t j = 0; j < n; j++) { C[i * n + j] = i == j; R[i * n + j] = -R[i * n + j]; }
  if (crgemm_oz(n, n, n, R, n, A, lda, 1, C, n, CRSUM_NEAREST, gemm, ctx)) goto out;
  for (size_t k = 0; k < n * n; k++) C[k] = nextafter(fabs(C[k]), INFINITY);

  /* the box Y = [-y, y]: from |z|, inflated, until |z| + |C| y < y in every component (each left side an exact
     sum rounded up) */
  for (size_t i = 0; i < n; i++) y[i] = az[i];
  for (int it = 0; it < 12 && status == CRLIN_FAILED; it++) {
    for (size_t i = 0; i < n; i++) y[i] = y[i] * 1.0625 + 0x1p-1000;   /* epsilon-inflation: any y would do */
    int ok = 1;
    for (size_t i = 0; i < n && ok; i++) {
      v[i] = crdot(C + i * n, y, n, CRSUM_UP);                          /* |C| y, rounded up */
      double t[2] = {az[i], v[i]};
      nx[i] = crsum(t, 2, CRSUM_UP);                                    /* |z| + |C| y, rounded up */
      ok = nx[i] < y[i];
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
  free(R); free(C); free(w);
  ENV_LEAVE;
  return status;
}
