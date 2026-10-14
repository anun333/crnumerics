/* band.c: crlin.h's banded and sparse solves (2026-10-14; crlin item 9).

   No inverse: A^-1 of a banded or sparse A is dense. Instead, Rump's route for sparse systems: a proven lower
   bound on the smallest eigenvalue of an SPD matrix G, which is A itself when A is symmetric positive definite, or
   A^T A otherwise (then its square root bounds A's smallest singular value). With it, any x~ and the exact residual
   r = b - A x~ give |x* - x~|_i <= ||x* - x~||_2 = ||A^-1 r||_2 <= ||r||_2 / lambda (or / sigma).
     - The bound is crlin_spd's: G = L L^T + c I + E for any L and c, so lambda_min(G) >= c - ||E||_2, and E is
       banded like G, each entry one exact sum (for A^T A, G's entries are exact dot products of A's columns and are
       never rounded); ||E||_2 <= its largest row sum. L is a banded Cholesky factor of fl(G) - c I with c about
       half G's smallest eigenvalue, from inverse iteration on a first factor of fl(G) (no proof rests on either).
     - x~ is a pair of doubles, refined with the exact residual through the first factor (for A^T A, through
       (L L^T)^-1 A^T, the normal equations, whose squared condition number limits the route to cond(A) of about
       1e8).
   Sparse A (compressed rows) is reordered by reverse Cuthill-McKee into a band and solved as one. */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "crlin.h"
#include "crsum.h"
#include "crlin-internal.h"

/* A[i][j] at AB[i * ld + (j + ks - i)], ks the subdiagonal count the caller's storage has; kl and ku the band's
   (at most n - 1) */
typedef struct { size_t n, kl, ku, ks, ld; const double *AB; } band;
static inline double at(const band *A, size_t i, size_t j)
{
  return j + A->kl < i || j > i + A->ku ? 0 : A->AB[i * A->ld + (j + A->ks - i)];
}
/* symmetric lower band storage, half bandwidth k: (i, j), i - k <= j <= i, at S[i * (k + 1) + (j + k - i)] */
#define S(M, k, i, j) (M)[(i) * ((k) + 1) + ((j) + (k) - (i))]

/* G = A (normal 0, k = kl) or fl(A^T A) (normal 1, k = kl + ku), lower band: the factor's input only */
static void build_g(const band *A, int normal, size_t k, double *G)
{
  size_t n = A->n;
  for (size_t i = 0; i < n; i++)
    for (size_t j = i > k ? i - k : 0; j <= i; j++) {
      double s = 0;
      if (!normal) s = at(A, i, j);
      else {   /* sum over the rows r holding both columns: r in [i - ku, j + kl] */
        size_t r0 = i > A->ku ? i - A->ku : 0, r1 = j + A->kl < n - 1 ? j + A->kl : n - 1;
        for (size_t r = r0; r <= r1; r++) s += at(A, r, i) * at(A, r, j);
      }
      S(G, k, i, j) = s;
    }
}
/* the Cholesky factor of G - c I, by rows; 0, or -1 for a pivot not positive */
static int chol(size_t n, size_t k, const double *G, double c, double *L)
{
  for (size_t i = 0; i < n; i++) {
    size_t j0 = i > k ? i - k : 0;
    for (size_t j = j0; j <= i; j++) {
      double s = S(G, k, i, j) - (i == j ? c : 0);
      for (size_t m = j0; m < j; m++) s -= S(L, k, i, m) * S(L, k, j, m);   /* row j's band starts at j - k <= j0 */
      if (i == j) {
        if (!(s > 0) || !isfinite(s)) return -1;
        S(L, k, i, i) = sqrt(s);
      } else S(L, k, i, j) = s / S(L, k, j, j);
    }
  }
  return 0;
}
static void lsolve(size_t n, size_t k, const double *L, double *y)    /* L y = y */
{
  for (size_t i = 0; i < n; i++) {
    double s = y[i];
    for (size_t m = i > k ? i - k : 0; m < i; m++) s -= S(L, k, i, m) * y[m];
    y[i] = s / S(L, k, i, i);
  }
}
static void ltsolve(size_t n, size_t k, const double *L, double *y)   /* L^T y = y */
{
  for (size_t i = n; i-- > 0;) {
    y[i] /= S(L, k, i, i);
    for (size_t m = i > k ? i - k : 0; m < i; m++) y[m] -= S(L, k, i, m) * y[i];
  }
}
/* an estimate of L L^T's smallest eigenvalue, from above: inverse iteration, 12 steps */
static double smallest(size_t n, size_t k, const double *L, double *v)
{
  uint64_t q = 0x9e3779b97f4a7c15ULL;
  double nrm = 0, est = 0;
  for (size_t i = 0; i < n; i++) { q = q * 6364136223846793005ULL + 1442695040888963407ULL; v[i] = (double)(q >> 11) * 0x1p-53 - 0.5; nrm += v[i] * v[i]; }
  nrm = sqrt(nrm);
  for (size_t i = 0; i < n; i++) v[i] /= nrm;
  for (int it = 0; it < 12; it++) {
    lsolve(n, k, L, v);
    ltsolve(n, k, L, v);
    nrm = 0;
    for (size_t i = 0; i < n; i++) nrm += v[i] * v[i];
    nrm = sqrt(nrm);
    if (!(nrm > 0) || !isfinite(nrm)) return 0;
    est = 1 / nrm;
    for (size_t i = 0; i < n; i++) v[i] /= nrm;
  }
  return est;
}
/* an upper bound on ||G - c I - L L^T||_2. CRLIN_EXACT: each entry one exact sum (G's own entries, for A^T A, exact
   dot products of A's columns), rounded to nearest and stepped away from zero. CRLIN_CLASSICAL: each entry's T terms
   summed left to right in binary64, s, and their magnitudes, a; |E_ij - s| <= g_T (a (1 + 2 g_T) + T 2^-1074) +
   T 2^-1074 (each product and sum within u of its value, plus an underflow's half quantum: this code's own loops, in
   the default environment, compiled without contraction or reassociation). E is symmetric, so its 2-norm is at most
   its largest row sum, rounded up. NaN when something overflowed or for want of memory */
static double bound_e(const band *A, int normal, size_t k, const double *L, double c, int mode)
{
  size_t n = A->n, cap = A->kl + A->ku + k + 2;
  double worst = 0;
  int bad = 0;
#pragma omp parallel if (n >= PAR_MIN) reduction(max : worst) reduction(| : bad)
  {
    PAR_ENTER;
    double *t1 = malloc(cap * sizeof *t1), *t2 = malloc(cap * sizeof *t2), *row = malloc((2 * k + 1) * sizeof *row);
    if (!t1 || !t2 || !row) bad = 1;
#pragma omp for schedule(static)
    for (size_t i = 0; i < n; i++) {
      if (!t1 || !t2 || !row) continue;
      size_t j0 = i > k ? i - k : 0, j1 = i + k < n - 1 ? i + k : n - 1, nr = 0;
      for (size_t j = j0; j <= j1; j++) {
        size_t lo = i > j ? i : j, hi = i < j ? i : j, len = 0;
        if (mode == CRLIN_CLASSICAL) {
          double sv = 0, av = 0;
          size_t T = 0;
          if (!normal) { double a = at(A, i, j); sv = a; av = fabs(a); T = 1; }
          else {
            size_t r0 = lo > A->ku ? lo - A->ku : 0, r1 = hi + A->kl < n - 1 ? hi + A->kl : n - 1;
            for (size_t r = r0; r <= r1; r++) { double p = at(A, r, i) * at(A, r, j); sv += p; av += fabs(p); T++; }
          }
          if (i == j) { sv -= c; av += fabs(c); T++; }
          for (size_t m = lo > k ? lo - k : 0; m <= hi; m++) { double p = S(L, k, i, m) * S(L, k, j, m); sv -= p; av += fabs(p); T++; }
          double dT = (double)T, gT = up(up(dT * 0x1p-53) / (1 - up(dT * 0x1p-53))), tt = up(dT * 0x1p-1074);
          double g = up(gT * up(1 + up(2 * gT)));   /* the magnitudes' own sum within a factor 1 / ((1 - u)(1 - g_T)) */
          row[nr++] = up(up(fabs(sv) + up(g * up(av + tt))) + tt);
          continue;
        }
        crsum_acc acc;
        crsum_init(&acc);
        if (!normal) { double a = at(A, i, j); crsum_add(&acc, &a, 1); }
        else {
          size_t r0 = lo > A->ku ? lo - A->ku : 0, r1 = hi + A->kl < n - 1 ? hi + A->kl : n - 1;
          for (size_t r = r0; r <= r1; r++) { t1[len] = at(A, r, i); t2[len++] = at(A, r, j); }
          crsum_add_dot(&acc, t1, t2, len);
        }
        if (i == j) { double mc = -c; crsum_add(&acc, &mc, 1); }
        len = 0;   /* - sum over m of L[i][m] L[j][m], m from max(i, j) - k to min(i, j) */
        for (size_t m = lo > k ? lo - k : 0; m <= hi; m++) { t1[len] = -S(L, k, i, m); t2[len++] = S(L, k, j, m); }
        crsum_add_dot(&acc, t1, t2, len);
        row[nr++] = up(fabs(crsum_round(&acc, CRSUM_NEAREST)));
      }
      double s = crsum(row, nr, CRSUM_UP);
      if (!isfinite(s)) bad = 1;
      else worst = fmax(worst, s);
    }
    free(t1); free(t2); free(row);
    PAR_LEAVE;
  }
  return bad ? NAN : worst;
}
/* r = b - A (x + x2), each row an exact sum rounded to nearest; nx and nx2: scratch, n each */
static void residual(const band *A, const double *b, const double *x, const double *x2, double *r, double *nx, double *nx2)
{
  size_t n = A->n;
  for (size_t j = 0; j < n; j++) { nx[j] = -x[j]; nx2[j] = -x2[j]; }
#pragma omp parallel if (n >= PAR_MIN)
  {
    PAR_ENTER;
#pragma omp for schedule(static)
    for (size_t i = 0; i < n; i++) {
      size_t j0 = i > A->kl ? i - A->kl : 0, j1 = i + A->ku < n - 1 ? i + A->ku : n - 1;
      const double *a = A->AB + i * A->ld + (j0 + A->ks - i);
      crsum_acc acc;
      crsum_init(&acc);
      /* the band's nonzeros only (a sparse matrix's band is mostly zeros), each with -x and -x2 */
      double t1[64], t2[64];
      size_t m = 0;
      for (size_t j = j0; j <= j1; j++) {
        if (a[j - j0] == 0) continue;
        t1[m] = a[j - j0]; t2[m++] = nx[j];
        t1[m] = a[j - j0]; t2[m++] = nx2[j];
        if (m == 64) { crsum_add_dot(&acc, t1, t2, m); m = 0; }
      }
      crsum_add_dot(&acc, t1, t2, m);
      crsum_add(&acc, b + i, 1);
      r[i] = crsum_round(&acc, CRSUM_NEAREST);
    }
    PAR_LEAVE;
  }
}

/* one route: a lower bound lam on G's smallest eigenvalue (-inf if none), and x~ (x + x2) refined; L0 the first
   factor (of fl(G) - c0 I), L1 the bound's */
static double route(const band *A, int normal, const double *b, double *x, double *x2, double *w, int mode)
{
  size_t n = A->n, k = normal ? A->kl + A->ku : A->kl, sz = n * (k + 1);
  double *G = malloc(sz * sizeof *G), *L0 = malloc(sz * sizeof *L0), *L1 = malloc(sz * sizeof *L1), lam = -INFINITY;
  if (!G || !L0 || !L1) goto done;
  build_g(A, normal, k, G);
  double dmax = 0;
  for (size_t i = 0; i < n; i++) dmax = fmax(dmax, S(G, k, i, i));
  double c0 = up(4 * (double)n * 0x1p-53 * dmax), c = c0;
  if (!(dmax > 0) || !isfinite(dmax) || chol(n, k, G, c0, L0)) goto done;
  double est = smallest(n, k, L0, w) + c0;
  int have = 0;
  for (int t = 0; t < 3 && est / 2 > c0 && !have; t++, est /= 2)
    if (!chol(n, k, G, est / 2, L1)) { c = est / 2; have = 1; }
  if (!have) memcpy(L1, L0, sz * sizeof *L1);
  double beta = bound_e(A, normal, k, L1, c, mode), t[2] = {c, -beta};
  if (!isnan(beta)) lam = crsum(t, 2, CRSUM_DOWN);
  if (!(lam > 0) && have) {   /* the second factor's bound failed: the first's */
    beta = bound_e(A, normal, k, L0, c0, mode);
    t[0] = c0; t[1] = -beta;
    if (!isnan(beta)) lam = fmax(lam, crsum(t, 2, CRSUM_DOWN));
  }
  if (!(lam > 0)) goto done;
  /* x~ from 0, each step d = (L0 L0^T)^-1 s for s = r (or A^T r), r the exact residual, added to the pair exactly */
  double *r = w, *s = w + n, *nx = w + 2 * n, *nx2 = w + 3 * n;
  memset(x, 0, n * sizeof *x); memset(x2, 0, n * sizeof *x2);
  for (int it = 0; it < 10; it++) {
    residual(A, b, x, x2, r, nx, nx2);
    if (!normal) memcpy(s, r, n * sizeof *s);
    else {
      memset(s, 0, n * sizeof *s);
      for (size_t i = 0; i < n; i++) {   /* s = A^T r: row i of A adds r_i A[i][j] to s_j */
        size_t j0 = i > A->kl ? i - A->kl : 0, j1 = i + A->ku < n - 1 ? i + A->ku : n - 1;
        for (size_t j = j0; j <= j1; j++) s[j] += at(A, i, j) * r[i];
      }
    }
    lsolve(n, k, L0, s);
    ltsolve(n, k, L0, s);
    int moved = 0;
    for (size_t i = 0; i < n; i++) {
      double p[3] = {x[i], x2[i], s[i]}, hi = crsum(p, 3, CRSUM_NEAREST), q[4] = {x[i], x2[i], s[i], -hi};
      moved |= hi != x[i];
      x2[i] = crsum(q, 4, CRSUM_NEAREST);
      x[i] = hi;
    }
    if (!moved && it >= 2) break;
  }
done:
  free(G); free(L0); free(L1);
  return lam;
}

int crlin_solve_band(size_t n, size_t kl, size_t ku, const double *AB, size_t ldab, const double *b, double *x_lo,
                     double *x_hi, int mode)
{
  if (!n || ldab < kl + ku + 1 || (mode != CRLIN_EXACT && mode != CRLIN_CLASSICAL)) return -1;
  band A = {n, kl < n ? kl : n - 1, ku < n ? ku : n - 1, kl, ldab, AB};   /* a band wider than the matrix: as LAPACK */
  kl = A.kl; ku = A.ku;
  for (size_t i = 0; i < n; i++) {
    if (!isfinite(b[i])) return -1;
    for (size_t j = i > kl ? i - kl : 0; j <= i + ku && j < n; j++) if (!isfinite(at(&A, i, j))) return -1;
  }
  int status = CRLIN_FAILED;
  ENV_ENTER;
  double *x = malloc(n * sizeof *x), *x2 = malloc(n * sizeof *x2), *w = malloc(4 * n * sizeof *w), *r = malloc(n * sizeof *r);
  if (!x || !x2 || !w || !r) goto out;
  int sym = kl == ku;
  for (size_t i = 0; i < n && sym; i++)
    for (size_t j = i > kl ? i - kl : 0; j < i; j++) if (at(&A, i, j) != at(&A, j, i)) { sym = 0; break; }
  for (int normal = !sym; normal <= 1 && status != CRLIN_VERIFIED; normal++) {
    double lam = route(&A, normal, b, x, x2, w, mode);
    if (!(lam > 0)) continue;
    /* delta >= ||r||_2 / lam (A SPD) or / sqrt(lam) (A^T A), r the exact residual, each step rounded up */
    residual(&A, b, x, x2, r, w + 2 * n, w + 3 * n);
    for (size_t i = 0; i < n; i++) w[i] = up(fabs(r[i]));
    double rn = up(sqrt(crdot(w, w, n, CRSUM_UP))), den = normal ? nextafter(sqrt(lam), 0) : lam, delta = up(rn / den);
    if (!isfinite(delta) || !(den > 0)) continue;
    for (size_t i = 0; i < n; i++) {
      double lo[3] = {x[i], x2[i], -delta}, hi[3] = {x[i], x2[i], delta};
      x_lo[i] = crsum(lo, 3, CRSUM_DOWN);
      x_hi[i] = crsum(hi, 3, CRSUM_UP);
    }
    status = CRLIN_VERIFIED;
  }
out:
  if (status != CRLIN_VERIFIED) for (size_t i = 0; i < n; i++) { x_lo[i] = -INFINITY; x_hi[i] = INFINITY; }
  free(x); free(x2); free(w); free(r);
  ENV_LEAVE;
  return status;
}

/* reverse Cuthill-McKee on the pattern of A + A^T: perm[new] = old */
static int deg_cmp_key(const size_t *deg, size_t a, size_t b) { return deg[a] < deg[b] ? -1 : deg[a] > deg[b]; }
static int rcm(size_t n, const size_t *ptr, const size_t *idx, size_t *perm)
{
  size_t *cnt = calloc(n + 1, sizeof *cnt), *adj = NULL, *deg = calloc(n, sizeof *deg), *fill = NULL, *level = NULL;
  unsigned char *seen = calloc(n, 1);
  int r = -1;
  if (!cnt || !deg || !seen) goto done;
  for (size_t i = 0; i < n; i++)
    for (size_t q = ptr[i]; q < ptr[i + 1]; q++) if (idx[q] != i) { cnt[i + 1]++; cnt[idx[q] + 1]++; }
  for (size_t i = 0; i < n; i++) cnt[i + 1] += cnt[i];
  adj = malloc((cnt[n] + 1) * sizeof *adj); fill = malloc(n * sizeof *fill); level = malloc(n * sizeof *level);
  if (!adj || !fill || !level) goto done;
  memcpy(fill, cnt, n * sizeof *fill);
  for (size_t i = 0; i < n; i++)
    for (size_t q = ptr[i]; q < ptr[i + 1]; q++) if (idx[q] != i) { adj[fill[i]++] = idx[q]; adj[fill[idx[q]]++] = i; }
  for (size_t i = 0; i < n; i++) {   /* each list sorted and without repeats: its length is the degree */
    size_t *a = adj + cnt[i], m = fill[i] - cnt[i], u = 0;
    for (size_t s = 1; s < m; s++) { size_t v = a[s], t = s; while (t > 0 && a[t - 1] > v) { a[t] = a[t - 1]; t--; } a[t] = v; }
    for (size_t s = 0; s < m; s++) if (!s || a[s] != a[s - 1]) a[u++] = a[s];
    deg[i] = u;
  }
  size_t done_n = 0;
  while (done_n < n) {
    size_t start = n;
    for (size_t i = 0; i < n; i++) if (!seen[i] && (start == n || deg[i] < deg[start])) start = i;
    /* a pseudo-peripheral start: twice, the least-degree node of the last level of a breadth-first search */
    for (int pass = 0; pass < 2; pass++) {
      size_t head = 0, tail = 0, last0 = 0;
      level[tail++] = start; seen[start] = 2;
      while (head < tail) {
        size_t lend = tail;
        last0 = head;
        for (; head < lend; head++)
          for (size_t q = cnt[level[head]]; q < cnt[level[head]] + deg[level[head]]; q++)
            if (!seen[adj[q]]) { seen[adj[q]] = 2; level[tail++] = adj[q]; }
      }
      size_t best = level[last0];
      for (size_t s = last0; s < tail; s++) if (deg[level[s]] < deg[best]) best = level[s];
      for (size_t s = 0; s < tail; s++) seen[level[s]] = 0;
      start = best;
    }
    /* Cuthill-McKee from it, neighbours by increasing degree */
    size_t head = done_n, tail = done_n;
    perm[tail++] = start; seen[start] = 1;
    while (head < tail) {
      size_t v = perm[head++], from = tail;
      for (size_t q = cnt[v]; q < cnt[v] + deg[v]; q++) if (!seen[adj[q]]) { seen[adj[q]] = 1; perm[tail++] = adj[q]; }
      for (size_t s = from + 1; s < tail; s++) {
        size_t u = perm[s], t = s;
        while (t > from && deg_cmp_key(deg, perm[t - 1], u) > 0) { perm[t] = perm[t - 1]; t--; }
        perm[t] = u;
      }
    }
    done_n = tail;
  }
  for (size_t i = 0; i < n / 2; i++) { size_t t = perm[i]; perm[i] = perm[n - 1 - i]; perm[n - 1 - i] = t; }
  r = 0;
done:
  free(cnt); free(adj); free(deg); free(fill); free(level); free(seen);
  return r;
}

int crlin_solve_sparse(size_t n, const size_t *rowptr, const size_t *colidx, const double *val, const double *b,
                       double *x_lo, double *x_hi, int mode)
{
  if (!n || rowptr[0] != 0 || (mode != CRLIN_EXACT && mode != CRLIN_CLASSICAL)) return -1;
  for (size_t i = 0; i < n; i++) {
    if (rowptr[i + 1] < rowptr[i] || !isfinite(b[i])) return -1;
    for (size_t q = rowptr[i]; q < rowptr[i + 1]; q++) {
      if (colidx[q] >= n || !isfinite(val[q])) return -1;
      for (size_t p = rowptr[i]; p < q; p++) if (colidx[p] == colidx[q]) return -1;   /* a repeated entry */
    }
  }
  size_t *perm = malloc(n * sizeof *perm), *inv = malloc(n * sizeof *inv);
  double *AB = NULL, *bp = malloc(n * sizeof *bp), *lo = malloc(n * sizeof *lo), *hi = malloc(n * sizeof *hi);
  int status = -1;
  if (!perm || !inv || !bp || !lo || !hi || rcm(n, rowptr, colidx, perm)) goto out;
  for (size_t s = 0; s < n; s++) inv[perm[s]] = s;
  size_t kl = 0, ku = 0;
  for (size_t i = 0; i < n; i++)
    for (size_t q = rowptr[i]; q < rowptr[i + 1]; q++) {
      size_t a = inv[i], c = inv[colidx[q]];
      if (a > c && a - c > kl) kl = a - c;
      if (c > a && c - a > ku) ku = c - a;
    }
  size_t ld = kl + ku + 1;
  if (ld > ((size_t)1 << 27) / n) goto out;   /* past a gigabyte of band: -1 */
  AB = calloc(n * ld, sizeof *AB);
  if (!AB) goto out;
  for (size_t i = 0; i < n; i++) {
    for (size_t q = rowptr[i]; q < rowptr[i + 1]; q++) AB[inv[i] * ld + (inv[colidx[q]] + kl - inv[i])] = val[q];
    bp[inv[i]] = b[i];
  }
  status = crlin_solve_band(n, kl, ku, AB, ld, bp, lo, hi, mode);
  for (size_t i = 0; i < n; i++) { x_lo[i] = lo[inv[i]]; x_hi[i] = hi[inv[i]]; }
out:
  if (status < 0) for (size_t i = 0; i < n; i++) { x_lo[i] = -INFINITY; x_hi[i] = INFINITY; }
  free(perm); free(inv); free(AB); free(bp); free(lo); free(hi);
  return status;
}
