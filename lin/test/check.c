/* check: crlin (crlin.h), 2026-10-13.

     1  containment: systems whose exact rational solution is known (each row
        scaled to integers, which keeps the solution, then fraction-free
        Gauss-Jordan elimination in GMP's integers: x*_i = det(A_i)/det(A)),
        random of every size to 64, integer ones with small exact answers,
        triangular ones, rows of wildly different scales; every verified
        enclosure must hold x* exactly. The control: each enclosure shrunk
        by an ulp at one end must miss x* somewhere (else the comparison could
        not see a miss)
     2  never wrong: exactly singular matrices (rank-deficient integer ones,
        a repeated row, a zero row) must not verify; Hilbert matrices to
        n = 14 (condition numbers to 1e18) either verify, holding x*, or
        fail
     3  any R: an inverse spoiled by noise of every size, and one of
        another matrix, may fail but never verifies wrongly
     4  the environment: under round-upward with flush-to-zero the same
        results, and the caller's environment back afterwards
     5  larger systems (n = 100 to 400): verified, how wide the enclosures
        are relative to |x|, and the time against the unverified solve
     6  several right-hand sides (crlin_solve_many): each column against the exact solution and bit for bit
        crlin_solve's; a column that overflows fails alone
     7  threads (OpenMP): the same bits with 1, 3 and 4 threads (crlin_spd's bound too, with 1 and 4), on sizes past
        every parallel loop's threshold; and
        with every thread of the pool left in round-upward with flush-to-zero (a worker keeps its own mode, which the
        caller's can't reach), the same bits again and each worker's mode back afterwards
     8  positive definiteness (crlin_spd): each lower bound lambda_lo checked exactly, A - lambda_lo I positive
        definite by symmetric elimination in rationals (every pivot positive); random positive definite matrices,
        some with rows and columns scaled over many binades, and Hilbert matrices; exactly singular and indefinite
        ones never verify; a factor given (good, spoiled, random) never verifies wrongly; not symmetric: -1. The
        control: A - 4 lambda_lo I is not positive definite for some (the bound is near the smallest eigenvalue,
        and the exact test can see a failure)
     9  interval data (crlin_solve_interval): for n <= 3 every vertex of [A] and [b] solved exactly, which gives the
        hull of the solutions (its ends lie at vertices): every enclosure holds them all, and how much wider it is
        than the hull is printed; larger systems against sampled vertices and interior points; an [A] holding a
        singular matrix never verifies; lo > hi refused. The control: an enclosure shrunk by 1% of its width at one
        end misses some vertex's solution
    10  very ill-conditioned: integer matrices of determinant 1 built from random row operations, their entries
        grown to 2^20 to 2^52 (exact in binary64), condition numbers to past 1e30, computed exactly from the integer
        inverse; every enclosure against the exact solution. CRLIN_EXTENDED should verify them to about 1e30, the
        other modes to about 1e15; none may be wrong
   Every section runs in each mode, CRLIN_EXACT, CRLIN_CLASSICAL and CRLIN_EXTENDED (sections 8 and 9 in the first
   two: crlin_spd and crlin_solve_interval refuse CRLIN_EXTENDED), on the same systems.
   The last line is the verdict. */
#include <fenv.h>
#include <float.h>
#include <gmp.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "crlin.h"

static int MODE;   /* CRLIN_EXACT or CRLIN_CLASSICAL: every section runs in both */
static uint64_t rng = 0x5851f42d4c957f2dULL;
static uint64_t next(void)
{
  uint64_t z = (rng += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}
static double unit(void) { return (next() >> 11) * 0x1p-53; }
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }

static long exact_bad;   /* systems whose "exact" solution fails A x = b: a bug in this check, not in crlin */
/* the exact solution: 1 and x[i] = num[i] / den, or 0 if A is singular */
static int exact(size_t n, const double *A, const double *b, mpq_t *x)
{
  size_t w = n + 1;
  mpz_t *M = malloc(n * w * sizeof *M), prev, t, u;
  mpz_inits(prev, t, u, NULL);
  for (size_t i = 0; i < n; i++) {
    int e = INT32_MAX;   /* the row's smallest exponent: scaling by 2^-e makes it integers */
    for (size_t j = 0; j < w; j++) {
      double v = j < n ? A[i * n + j] : b[i];
      if (v != 0) { int ex; frexp(v, &ex); if (ex - 53 < e) e = ex - 53; }
    }
    for (size_t j = 0; j < w; j++) {
      double v = j < n ? A[i * n + j] : b[i];
      mpz_init(M[i * w + j]);
      mpz_set_d(M[i * w + j], e == INT32_MAX ? 0 : ldexp(v, -e));   /* exact: an integer below 2^53 times a power */
    }
  }
  mpz_set_ui(prev, 1);
  int ok = 1;
  for (size_t k = 0; k < n && ok; k++) {
    size_t p = k;
    while (p < n && mpz_sgn(M[p * w + k]) == 0) p++;
    if (p == n) { ok = 0; break; }
    if (p != k) for (size_t j = 0; j < w; j++) mpz_swap(M[p * w + j], M[k * w + j]);
    for (size_t i = 0; i < n; i++) {
      if (i == k) continue;
      for (size_t j = 0; j < w; j++) {
        if (j == k) continue;
        mpz_mul(t, M[k * w + k], M[i * w + j]);
        mpz_mul(u, M[i * w + k], M[k * w + j]);
        mpz_sub(t, t, u);
        mpz_divexact(M[i * w + j], t, prev);
      }
    }
    for (size_t i = 0; i < n; i++) if (i != k) mpz_set_ui(M[i * w + k], 0);
    /* the rows not yet pivoted keep their other entries as minors; the pivot row is unchanged, as in Bareiss */
    mpz_set(prev, M[k * w + k]);
  }
  if (ok) {
    for (size_t i = 0; i < n; i++) { mpq_set_num(x[i], M[i * w + n]); mpq_set_den(x[i], M[i * w + i]); mpq_canonicalize(x[i]); }
    /* the solver checks itself: A x = b exactly */
    mpq_t s, a, p; mpq_inits(s, a, p, NULL);
    for (size_t i = 0; i < n; i++) {
      mpq_set_ui(s, 0, 1);
      for (size_t j = 0; j < n; j++) { mpq_set_d(a, A[i * n + j]); mpq_mul(p, a, x[j]); mpq_add(s, s, p); }
      mpq_set_d(a, b[i]);
      if (!mpq_equal(s, a)) exact_bad++;
    }
    mpq_clears(s, a, p, NULL);
  }
  for (size_t k = 0; k < n * w; k++) mpz_clear(M[k]);
  free(M);
  mpz_clears(prev, t, u, NULL);
  return ok;
}

/* does [lo, hi] hold x exactly? */
static int holds(double lo, double hi, const mpq_t x)
{
  mpq_t a; mpq_init(a);
  int r = 1;
  if (isfinite(lo)) { mpq_set_d(a, lo); r &= mpq_cmp(a, x) <= 0; } else r &= lo < 0;
  if (isfinite(hi)) { mpq_set_d(a, hi); r &= mpq_cmp(a, x) >= 0; } else r &= hi > 0;
  mpq_clear(a);
  return r;
}

static struct { long cases, verified, failed, wrong, shrunk_miss; double width_max; } T;
/* one system: solve, compare with the exact solution; singular: whether it was singular */
static int one(size_t n, const double *A, const double *b, const double *R, int *singular)
{
  double *lo = malloc(n * sizeof *lo), *hi = malloc(n * sizeof *hi);
  mpq_t *x = malloc(n * sizeof *x);
  for (size_t i = 0; i < n; i++) mpq_init(x[i]);
  int s = crlin_solve(n, A, n, b, R, n, lo, hi, MODE, NULL, NULL), ex = exact(n, A, b, x);
  *singular = !ex;
  T.cases++;
  if (s == CRLIN_VERIFIED) {
    T.verified++;
    if (!ex) T.wrong++;   /* verified a singular matrix */
    else {
      int bad = 0, miss = 0;
      for (size_t i = 0; i < n; i++) {
        bad += !holds(lo[i], hi[i], x[i]);
        miss += !holds(nextafter(lo[i], INFINITY), hi[i], x[i]) || !holds(lo[i], nextafter(hi[i], -INFINITY), x[i]);
        double m = fmax(fabs(lo[i]), fabs(hi[i]));
        if (m > 0) T.width_max = fmax(T.width_max, (hi[i] - lo[i]) / m);
      }
      T.wrong += bad != 0;
      T.shrunk_miss += miss != 0;
    }
  } else T.failed++;
  for (size_t i = 0; i < n; i++) mpq_clear(x[i]);
  free(lo); free(hi); free(x);
  return s;
}

static double rnd_entry(int spread) { double v = unit() * 2 - 1; return ldexp(v, spread ? (int)(next() % (2 * spread + 1)) - spread : 0); }

static int sec1(void)
{
  memset(&T, 0, sizeof T);
  long sing = 0;
  for (size_t n = 1; n <= 64; n = n < 8 ? n + 1 : n * 2)
    for (int rep = 0; rep < (n <= 8 ? 60 : 12); rep++) {
      double *A = malloc(n * n * sizeof *A), *b = malloc(n * sizeof *b);
      int kind = rep % 4, sg;
      for (size_t i = 0; i < n; i++) {
        double rs = kind == 3 ? ldexp(1, (int)(next() % 200) - 100) : 1;   /* rows of different scales */
        for (size_t j = 0; j < n; j++) {
          double v = kind == 1 ? (double)((int)(next() % 21) - 10)          /* small integers */
                   : kind == 2 ? (j < i ? 0 : rnd_entry(3))                 /* upper triangular */
                   : rnd_entry(kind == 0 ? 6 : 0);
          A[i * n + j] = (kind == 2 && i == j && v == 0 ? 1 : v) * rs;
        }
        b[i] = (kind == 1 ? (double)((int)(next() % 41) - 20) : rnd_entry(4)) * rs;
      }
      one(n, A, b, NULL, &sg);
      sing += sg;
      free(A); free(b);
    }
  printf("containment: %ld systems (n 1 to 64), %ld verified, %ld failed (%ld exactly singular); %ld enclosures miss x* "
         "(must be 0); widest relative width %.2g; control: %ld of the verified miss x* when shrunk by an ulp (must be > 0)\n",
         T.cases, T.verified, T.failed, sing, T.wrong, T.width_max, T.shrunk_miss);
  return T.wrong ? 1 : !T.shrunk_miss || !T.verified ? 2 : 0;
}

static int sec2(void)
{
  memset(&T, 0, sizeof T);
  long singular_verified = 0, sing_cases = 0;
  /* exactly singular */
  for (size_t n = 2; n <= 32; n *= 2)
    for (int rep = 0; rep < 20; rep++) {
      double *A = malloc(n * n * sizeof *A), *b = malloc(n * sizeof *b);
      int kind = rep % 3, sg;
      size_t r = 1 + next() % (n - 1);   /* rank r < n: A = U V, U n by r, V r by n, small integers (exact) */
      for (size_t i = 0; i < n; i++) for (size_t j = 0; j < n; j++) {
        double s = 0;
        for (size_t k = 0; k < r; k++) s += (double)((int)((i * 31 + k * 7 + rep) % 7) - 3) * (double)((int)((k * 13 + j * 5 + rep) % 5) - 2);
        A[i * n + j] = kind == 0 ? s : rnd_entry(2);
      }
      if (kind == 1) memcpy(A + (n - 1) * n, A, n * sizeof *A);          /* a repeated row */
      if (kind == 2) memset(A + (n / 2) * n, 0, n * sizeof *A);          /* a zero row */
      for (size_t i = 0; i < n; i++) b[i] = rnd_entry(2);
      int s = one(n, A, b, NULL, &sg);
      sing_cases += sg;
      singular_verified += sg && s == CRLIN_VERIFIED;
      free(A); free(b);
    }
  long scases = T.cases;
  /* Hilbert matrices, rounded to binary64 (the system solved is the rounded one) */
  long hv = 0, hf = 0, hwrong_before = T.wrong;
  for (size_t n = 2; n <= 24; n++) {
    double *A = malloc(n * n * sizeof *A), *b = malloc(n * sizeof *b);
    int sg;
    for (size_t i = 0; i < n; i++) { for (size_t j = 0; j < n; j++) A[i * n + j] = 1.0 / (double)(i + j + 1); b[i] = 1; }
    int s = one(n, A, b, NULL, &sg);
    hv += s == CRLIN_VERIFIED; hf += s != CRLIN_VERIFIED;
    free(A); free(b);
  }
  printf("never wrong: %ld singular-built systems, %ld exactly singular, %ld of those verified (must be 0); Hilbert n 2 to 24: "
         "%ld verified, %ld failed, %ld verified wrongly (must be 0); in all, %ld enclosures miss x* (must be 0)\n",
         scases, sing_cases, singular_verified, hv, hf, T.wrong - hwrong_before, T.wrong);
  return singular_verified || T.wrong ? 1 : !sing_cases || !hv ? 2 : 0;
}

static int sec3(void)
{
  memset(&T, 0, sizeof T);
  long noisy_cases = 0;
  for (int rep = 0; rep < 120; rep++) {
    size_t n = 2 + next() % 24;
    double *A = malloc(n * n * sizeof *A), *B = malloc(n * n * sizeof *B), *b = malloc(n * sizeof *b), *R = malloc(n * n * sizeof *R);
    for (size_t k = 0; k < n * n; k++) { A[k] = rnd_entry(3); B[k] = rnd_entry(3); }
    for (size_t i = 0; i < n; i++) b[i] = rnd_entry(3);
    /* R: B's inverse (another matrix), or A's spoiled by relative noise 2^-50 .. 2^-1 */
    double *lo = malloc(n * sizeof *lo), *hi = malloc(n * sizeof *hi);
    int sg;
    /* an inverse of A, or of B: A's or B's systems against the identity's columns (the enclosures' centres) */
    const double *M = rep % 4 == 0 ? B : A;
    for (size_t j = 0; j < n; j++) {
      double *e = calloc(n, sizeof *e); e[j] = 1;
      if (crlin_solve(n, M, n, e, NULL, 0, lo, hi, MODE, NULL, NULL) == 0) for (size_t i = 0; i < n; i++) R[i * n + j] = 0.5 * (lo[i] + hi[i]);
      else for (size_t i = 0; i < n; i++) R[i * n + j] = i == j;
      free(e);
    }
    if (rep % 4) { double s = ldexp(1, -1 - (int)(rep % 50)); for (size_t k = 0; k < n * n; k++) R[k] *= 1 + s * (2 * unit() - 1); }
    one(n, A, b, R, &sg);
    noisy_cases++;
    free(A); free(B); free(b); free(R); free(lo); free(hi);
  }
  printf("any R: %ld systems with another matrix's inverse or a spoiled one: %ld verified, %ld failed, %ld enclosures miss x* "
         "(must be 0)\n", noisy_cases, T.verified, T.failed, T.wrong);
  /* some must fail, or the spoiled ones tested nothing; but CRLIN_EXTENDED's iteration repairs any R it starts from
     (R -> inv(R A) R), so there all may verify */
  return T.wrong ? 1 : !T.verified || (!T.failed && MODE != CRLIN_EXTENDED) ? 2 : 0;
}

static void ftz(int on)
{
#if defined(__x86_64__)
  unsigned m; __asm__ volatile("stmxcsr %0" : "=m"(m)); m = on ? m | 0x8040 : m & ~0x8040u; __asm__ volatile("ldmxcsr %0" :: "m"(m));
#elif defined(__aarch64__)
  uint64_t c; __asm__ volatile("mrs %0, fpcr" : "=r"(c)); c = on ? c | (1u << 24) : c & ~(uint64_t)(1u << 24); __asm__ volatile("msr fpcr, %0" :: "r"(c));
#else
  (void)on;
#endif
}
static int sec4(void)
{
  long diff = 0, cases = 0, restored = 1;
  for (int rep = 0; rep < 40; rep++) {
    size_t n = 1 + next() % 40;
    double *A = malloc(n * n * sizeof *A), *b = malloc(n * sizeof *b), *l0 = malloc(n * 8), *h0 = malloc(n * 8), *l1 = malloc(n * 8), *h1 = malloc(n * 8);
    for (size_t k = 0; k < n * n; k++) A[k] = ldexp(rnd_entry(2), rep % 3 ? 0 : -1060);   /* some subnormal throughout */
    for (size_t i = 0; i < n; i++) b[i] = ldexp(rnd_entry(2), rep % 3 ? 0 : -1060);
    int s0 = crlin_solve(n, A, n, b, NULL, 0, l0, h0, MODE, NULL, NULL);
    fesetround(FE_UPWARD); ftz(1);
    int s1 = crlin_solve(n, A, n, b, NULL, 0, l1, h1, MODE, NULL, NULL);
    restored &= fegetround() == FE_UPWARD;
    ftz(0); fesetround(FE_TONEAREST);
    diff += s0 != s1 || memcmp(l0, l1, n * 8) || memcmp(h0, h1, n * 8);
    cases++;
    free(A); free(b); free(l0); free(h0); free(l1); free(h1);
  }
  printf("environment: under round-upward with flush-to-zero, %ld of %ld results differ (must be 0); the caller's mode back: %s\n",
         diff, cases, restored ? "yes" : "NO");
  return diff || !restored ? 1 : 0;
}

static int sec5(void)
{
  int r = 0;
  for (size_t n = 100; n <= 400; n *= 2) {
    double *A = malloc(n * n * sizeof *A), *b = malloc(n * sizeof *b), *lo = malloc(n * 8), *hi = malloc(n * 8);
    for (size_t k = 0; k < n * n; k++) A[k] = rnd_entry(0);
    for (size_t i = 0; i < n; i++) b[i] = rnd_entry(0);
    double t0 = now();
    int s = crlin_solve(n, A, n, b, NULL, 0, lo, hi, MODE, NULL, NULL);
    double t = now() - t0, wmax = 0, wmed;
    double *wd = malloc(n * sizeof *wd);
    for (size_t i = 0; i < n; i++) { wd[i] = (hi[i] - lo[i]) / fmax(fabs(lo[i]), fabs(hi[i])); wmax = fmax(wmax, wd[i]); }
    for (size_t i = 1; i < n; i++) for (size_t j = i; j > 0 && wd[j] < wd[j - 1]; j--) { double q = wd[j]; wd[j] = wd[j - 1]; wd[j - 1] = q; }
    wmed = wd[n / 2];
    printf("  n = %3zu random: %s in %.3f s; relative width median %.2g, largest %.2g\n", n,
           s == CRLIN_VERIFIED ? "verified" : "FAILED", t, wmed, wmax);
    if (s != CRLIN_VERIFIED) r = 1;
    free(A); free(b); free(lo); free(hi); free(wd);
  }
  return r;
}

/* 6: several right-hand sides: each column against the exact solution, and bit for bit what crlin_solve gives it
   alone; a column whose solution overflows fails by itself while the others keep their enclosures */
static int sec6(void)
{
  long cols = 0, wrong = 0, differ = 0, systems = 0;
  for (int rep = 0; rep < 80; rep++) {
    size_t n = 1 + next() % 16, m = 1 + next() % 7;
    double *A = malloc(n * n * 8), *B = malloc(n * m * 8), *L = malloc(n * m * 8), *H = malloc(n * m * 8);
    double *b = malloc(n * 8), *l = malloc(n * 8), *h = malloc(n * 8);
    for (size_t k = 0; k < n * n; k++) A[k] = rnd_entry(4);
    for (size_t k = 0; k < n * m; k++) B[k] = rnd_entry(4);
    int s = crlin_solve_many(n, m, A, n, B, m, NULL, 0, L, H, m, MODE, NULL, NULL);
    systems++;
    mpq_t *x = malloc(n * sizeof *x);
    for (size_t i = 0; i < n; i++) mpq_init(x[i]);
    for (size_t k = 0; k < m; k++) {
      for (size_t i = 0; i < n; i++) b[i] = B[i * m + k];
      crlin_solve(n, A, n, b, NULL, 0, l, h, MODE, NULL, NULL);
      for (size_t i = 0; i < n; i++) differ += memcmp(&l[i], &L[i * m + k], 8) || memcmp(&h[i], &H[i * m + k], 8);
      if (s == CRLIN_VERIFIED && exact(n, A, b, x)) for (size_t i = 0; i < n; i++) wrong += !holds(L[i * m + k], H[i * m + k], x[i]);
      cols++;
    }
    for (size_t i = 0; i < n; i++) mpq_clear(x[i]);
    free(A); free(B); free(L); free(H); free(b); free(l); free(h); free(x);
  }
  /* a column that overflows: A = I/2, B's last column 1e308 */
  size_t n = 5, m = 3;
  double A[25] = {0}, B[15], L[15], H[15];
  for (size_t i = 0; i < n; i++) { A[i * n + i] = 0.5; B[i * m] = 1; B[i * m + 1] = (double)i; B[i * m + 2] = 1e308; }
  int s = crlin_solve_many(n, m, A, n, B, m, NULL, 0, L, H, m, MODE, NULL, NULL);
  int alone = s == CRLIN_FAILED && isinf(L[2]) && isinf(H[2]);
  /* the others: finite and tight, not merely -inf and +inf (which would hold anything: the first version of this
     test accepted them) */
  for (size_t i = 0; i < n; i++)
    alone &= L[i * m] <= 2 && H[i * m] >= 2 && H[i * m] - L[i * m] < 1e-12 && L[i * m + 1] <= 2.0 * i && H[i * m + 1] >= 2.0 * i &&
             H[i * m + 1] - L[i * m + 1] < 1e-12;
  printf("several right-hand sides: %ld systems, %ld columns: %ld miss x* (must be 0), %ld differ from solving alone "
         "(must be 0); an overflowing column fails alone, the others kept: %s\n", systems, cols, wrong, differ, alone ? "yes" : "NO");
  return wrong || differ || !alone ? 1 : 0;
}

/* 7: threads. Systems of n = 64 to 300 (every parallel loop's threshold passed, the inverse's elimination from n =
   128 and the plain x~ loop from 256), one and three right-hand sides, well-conditioned and nearly singular (the last
   row the sum of the first two plus 1e-9 noise). The nearly singular ones are the witnesses: a well-conditioned
   system's enclosure is the same ulp or two around x* whatever R and x~ were, so a thread computing them in another
   mode leaves no trace in it, while a nearly singular one's last bits move with R's and x~'s. The first version had
   only well-conditioned systems and passed on a build whose threads never set their mode; with these, that build
   gives 3 of the 16 solves differing in each mode, all nearly singular, and 2 of the 4 bounds of crlin_spd (whose
   own factor, computed in the workers' mode, moves them) (2026-10-13). */
#ifdef _OPENMP
static void pool_mode(int nt, int poison)   /* every thread of an nt-thread team: round-upward with flush-to-zero, or back */
{
#pragma omp parallel num_threads(nt)
  {
    if (poison) { fesetround(FE_UPWARD); ftz(1); }
    else { ftz(0); fesetround(FE_TONEAREST); }
  }
}
#endif
static int sec7(void)
{
#ifndef _OPENMP
  printf("threads: built without OpenMP, nothing to compare\n");
  return 0;
#else
  static const size_t ns[] = {64, 100, 150, 300};
  long cases = 0, differ = 0, poisoned_differ = 0, verified = 0, near_verified = 0;
  int kept = 1, nt0 = omp_get_max_threads();
  for (size_t t = 0; t < sizeof ns / sizeof *ns; t++)
    for (size_t m = 1; m <= 3; m += 2)
    for (int near = 0; near <= 1; near++) {
      size_t n = ns[t];
      double *A = malloc(n * n * 8), *B = malloc(n * m * 8), *L0 = malloc(n * m * 8), *H0 = malloc(n * m * 8), *L = malloc(n * m * 8), *H = malloc(n * m * 8);
      for (size_t k = 0; k < n * n; k++) A[k] = rnd_entry(near ? 0 : 3);
      if (near) for (size_t j = 0; j < n; j++) A[(n - 1) * n + j] = A[j] + A[n + j] + 1e-9 * rnd_entry(0);
      for (size_t k = 0; k < n * m; k++) B[k] = rnd_entry(3);
      omp_set_num_threads(1);
      int s0 = crlin_solve_many(n, m, A, n, B, m, NULL, 0, L0, H0, m, MODE, NULL, NULL);
      verified += s0 == CRLIN_VERIFIED;
      near_verified += near && s0 == CRLIN_VERIFIED;
      for (int nt = 3; nt <= 4; nt++) {
        omp_set_num_threads(nt);
        int s = crlin_solve_many(n, m, A, n, B, m, NULL, 0, L, H, m, MODE, NULL, NULL);
        differ += s != s0 || memcmp(L, L0, n * m * 8) || memcmp(H, H0, n * m * 8);
      }
      /* the pool's threads in round-upward with flush-to-zero, the caller's in the default mode */
      pool_mode(4, 1);
      ftz(0); fesetround(FE_TONEAREST);
      int s = crlin_solve_many(n, m, A, n, B, m, NULL, 0, L, H, m, MODE, NULL, NULL);
      poisoned_differ += s != s0 || memcmp(L, L0, n * m * 8) || memcmp(H, H0, n * m * 8);
      int workers = 1;   /* the workers' own modes, which crlin must have put back (the master's was reset above) */
#pragma omp parallel num_threads(4) reduction(& : workers)
      if (omp_get_thread_num()) workers &= fegetround() == FE_UPWARD;
      kept &= workers;
      pool_mode(4, 0);
      cases++;
      if (m == 1 && !near && MODE != CRLIN_EXTENDED) {   /* crlin_spd on G = A^T A / n + I/4 (symmetric to the bit):
                                                            the same bound (crlin_spd refuses CRLIN_EXTENDED) */
        double *G = malloc(n * n * 8), l0, l1, l2;
        for (size_t i = 0; i < n; i++)
          for (size_t j = 0; j <= i; j++) {
            double t = 0;
            for (size_t k = 0; k < n; k++) t += A[k * n + i] * A[k * n + j];
            G[i * n + j] = G[j * n + i] = t / (double)n + (i == j ? 0.25 : 0);
          }
        omp_set_num_threads(1);
        int s1 = crlin_spd(n, G, n, NULL, 0, 0, &l0, MODE, NULL, NULL);
        omp_set_num_threads(4);
        int s2 = crlin_spd(n, G, n, NULL, 0, 0, &l1, MODE, NULL, NULL);
        pool_mode(4, 1);
        ftz(0); fesetround(FE_TONEAREST);
        int s3 = crlin_spd(n, G, n, NULL, 0, 0, &l2, MODE, NULL, NULL);
        pool_mode(4, 0);
        verified += s1 == CRLIN_VERIFIED;
        differ += s1 != s2 || memcmp(&l0, &l1, 8);
        poisoned_differ += s1 != s3 || memcmp(&l0, &l2, 8);
        cases++;
        free(G);
      }
      free(A); free(B); free(L0); free(H0); free(L); free(H);
    }
  omp_set_num_threads(nt0);
  printf("threads: %ld systems (%ld verified, %ld of them nearly singular), 3 and 4 threads against 1: %ld differ "
         "(must be 0); every thread in round-upward with flush-to-zero: %ld differ (must be 0), the workers' mode back: %s\n",
         cases, verified, near_verified, differ, poisoned_differ, kept ? "yes" : "NO");
  if (!near_verified) return 2;
  return differ || poisoned_differ || !kept ? 1 : 0;
#endif
}

/* 8: positive definiteness. A - t I positive definite, exactly: symmetric elimination in rationals, every pivot > 0
   (Sylvester's criterion through the pivots); A's entries and t taken exactly */
static int pd_exact(size_t n, const double *A, double t)
{
  mpq_t *M = malloc(n * n * sizeof *M), f, g;
  mpq_init(f); mpq_init(g);
  for (size_t i = 0; i < n; i++)
    for (size_t j = 0; j < n; j++) {
      mpq_init(M[i * n + j]); mpq_set_d(M[i * n + j], A[i * n + j]);
      if (i == j) { mpq_set_d(g, t); mpq_sub(M[i * n + j], M[i * n + j], g); }
    }
  int pd = 1;
  for (size_t k = 0; k < n && pd; k++) {
    if (mpq_sgn(M[k * n + k]) <= 0) { pd = 0; break; }
    for (size_t i = k + 1; i < n; i++) {
      if (!mpq_sgn(M[i * n + k])) continue;
      mpq_div(f, M[i * n + k], M[k * n + k]);
      for (size_t j = k + 1; j < n; j++) { mpq_mul(g, f, M[k * n + j]); mpq_sub(M[i * n + j], M[i * n + j], g); }
    }
  }
  for (size_t k = 0; k < n * n; k++) mpq_clear(M[k]);
  mpq_clear(f); mpq_clear(g); free(M);
  return pd;
}
static struct { long cases, verified, wrong, loose, notpd, notpd_verified; } P;
static void spd_one(size_t n, const double *A, const double *L, double c)
{
  double lam;
  int s = crlin_spd(n, A, n, L, n, c, &lam, MODE, NULL, NULL);
  int pd = pd_exact(n, A, 0);
  P.cases++;
  if (!pd) { P.notpd++; P.notpd_verified += s == CRLIN_VERIFIED; }
  if (s == CRLIN_VERIFIED) {
    P.verified++;
    if (!(lam > 0) || !pd_exact(n, A, lam)) P.wrong++;
    else if (!pd_exact(n, A, 4 * lam)) P.loose++;          /* the control: 4 lambda_lo is past the smallest */
  } else if (s == CRLIN_FAILED && lam > -INFINITY && !pd_exact(n, A, nextafter(lam, -INFINITY))) P.wrong++;
}
/* A = B^T B + d I for a random integer B (m by n) */
static void gram(size_t n, size_t m, double d, double *A)
{
  double *B = malloc(m * n * sizeof *B);
  for (size_t k = 0; k < m * n; k++) B[k] = (double)(int)(next() % 9) - 4;
  for (size_t i = 0; i < n; i++)
    for (size_t j = 0; j < n; j++) { double t = 0; for (size_t k = 0; k < m; k++) t += B[k * n + i] * B[k * n + j]; A[i * n + j] = t + (i == j ? d : 0); }
  free(B);
}
static int sec8(void)
{
  memset(&P, 0, sizeof P);
  long sym = 0;
  for (int rep = 0; rep < 160; rep++) {                    /* positive definite, integer; every fourth scaled by 2^k D A D */
    size_t n = 1 + next() % 20;
    double *A = malloc(n * n * sizeof *A);
    gram(n, n + next() % 4, (double)(next() % 3), A);
    if (rep % 4 == 3) {
      int *e = malloc(n * sizeof *e);
      for (size_t i = 0; i < n; i++) e[i] = (int)(next() % 61) - 30;
      for (size_t i = 0; i < n; i++) for (size_t j = 0; j < n; j++) A[i * n + j] = ldexp(A[i * n + j], e[i] + e[j]);
      free(e);
    }
    spd_one(n, A, NULL, 0);
    free(A);
  }
  for (size_t n = 2; n <= 14; n++) {                       /* Hilbert, its binary64 rounding */
    double *A = malloc(n * n * sizeof *A);
    for (size_t i = 0; i < n; i++) for (size_t j = 0; j < n; j++) A[i * n + j] = 1.0 / (double)(i + j + 1);
    spd_one(n, A, NULL, 0);
    free(A);
  }
  for (int rep = 0; rep < 80; rep++) {                     /* singular (B with fewer rows than columns), indefinite */
    size_t n = 2 + next() % 18;
    double *A = malloc(n * n * sizeof *A);
    if (rep % 2) gram(n, n - 1 - next() % (n - 1), 0, A);
    else {
      gram(n, n + 2, 1, A);
      double t = (double)(1 + next() % 40);               /* A - t I, its diagonal kept positive where it can */
      for (size_t i = 0; i < n; i++) A[i * n + i] -= t;
    }
    spd_one(n, A, NULL, 0);
    free(A);
  }
  for (int rep = 0; rep < 60; rep++) {                     /* a factor given: good, spoiled, random */
    size_t n = 1 + next() % 16;
    double *A = malloc(n * n * sizeof *A), *L = calloc(n * n, sizeof *L);
    gram(n, n + 1, 1, A);
    double c = rep % 3 == 2 ? 0 : 0.25;
    for (size_t j = 0; j < n; j++) {                       /* Cholesky of A - c I, then spoiled or replaced */
      double d = A[j * n + j] - c;
      for (size_t k = 0; k < j; k++) d -= L[j * n + k] * L[j * n + k];
      d = d > 0 ? sqrt(d) : 1;
      L[j * n + j] = d;
      for (size_t i = j + 1; i < n; i++) { double t = A[i * n + j]; for (size_t k = 0; k < j; k++) t -= L[i * n + k] * L[j * n + k]; L[i * n + j] = t / d; }
    }
    if (rep % 3 == 1) for (size_t k = 0; k < n * n; k++) L[k] *= 1 + 1e-3 * (unit() - 0.5);
    if (rep % 3 == 2) for (size_t k = 0; k < n * n; k++) L[k] = rnd_entry(1);
    spd_one(n, A, L, c);
    free(A); free(L);
  }
  {                                                        /* not symmetric, or not finite: -1 */
    double A[4] = {2, 1, 1.5, 2}, B[4] = {2, NAN, NAN, 2}, lam;
    sym = crlin_spd(2, A, 2, NULL, 0, 0, &lam, MODE, NULL, NULL) == -1 && crlin_spd(2, B, 2, NULL, 0, 0, &lam, MODE, NULL, NULL) == -1;
  }
  printf("positive definiteness: %ld matrices, %ld verified, %ld exactly not positive definite of which %ld verified (must be "
         "0); %ld lower bounds wrong (must be 0); not symmetric or not finite refused: %s; control: %ld verified with A - 4 "
         "lambda_lo I not positive definite (must be > 0)\n", P.cases, P.verified, P.notpd, P.notpd_verified, P.wrong,
         sym ? "yes" : "NO", P.loose);
  if (P.wrong || P.notpd_verified || !sym) return 1;
  return !P.loose || !P.notpd || !P.verified ? 2 : 0;
}

/* 9: interval data */
static void interval_of(double m, int k, double *lo, double *hi)   /* m +- m 2^-k, or degenerate when k = 0 */
{
  double r = k ? ldexp(fabs(m), -k) : 0;
  *lo = m - r; *hi = m + r;
}
static int sec9(void)
{
  long systems = 0, verified = 0, miss = 0, points = 0, shrunk_miss = 0, small = 0, singular = 0, singular_verified = 0;
  double over = 1;
  static const int ks[] = {0, 12, 24, 40, 52};
  for (int rep = 0; rep < 200; rep++) {
    size_t n = rep < 120 ? (size_t)(1 + rep % 3) : (size_t)(4 + next() % 17);
    double *Al = malloc(n * n * 8), *Ah = malloc(n * n * 8), *bl = malloc(n * 8), *bh = malloc(n * 8), *L = malloc(n * 8),
           *H = malloc(n * 8), *A = malloc(n * n * 8), *b = malloc(n * 8);
    int k = ks[rep % 5];
    for (size_t q = 0; q < n * n; q++) interval_of(rnd_entry(2) + (q % (n + 1) == 0 ? 3 : 0), next() % 4 ? k : 0, &Al[q], &Ah[q]);
    for (size_t i = 0; i < n; i++) interval_of(rnd_entry(2), k, &bl[i], &bh[i]);
    int s = crlin_solve_interval(n, 1, Al, Ah, n, bl, bh, 1, NULL, 0, L, H, 1, MODE, NULL, NULL);
    systems++;
    mpq_t *x = malloc(n * sizeof *x), *xmin = malloc(n * sizeof *xmin), *xmax = malloc(n * sizeof *xmax);
    for (size_t i = 0; i < n; i++) { mpq_init(x[i]); mpq_init(xmin[i]); mpq_init(xmax[i]); }
    if (s == CRLIN_VERIFIED) {
      verified++;
      if (n <= 3) {                                        /* every vertex: the hull */
        small++;
        size_t bits = n * n + n;
        for (unsigned long v = 0; v < (1UL << bits); v++) {
          for (size_t q = 0; q < n * n; q++) A[q] = v >> q & 1 ? Ah[q] : Al[q];
          for (size_t i = 0; i < n; i++) b[i] = v >> (n * n + i) & 1 ? bh[i] : bl[i];
          if (!exact(n, A, b, x)) { miss++; continue; }    /* a singular vertex inside a verified [A] */
          for (size_t i = 0; i < n; i++) {
            if (!holds(L[i], H[i], x[i])) miss++;
            if (!v || mpq_cmp(x[i], xmin[i]) < 0) mpq_set(xmin[i], x[i]);
            if (!v || mpq_cmp(x[i], xmax[i]) > 0) mpq_set(xmax[i], x[i]);
          }
          points++;
        }
        int lost = 0;
        for (size_t i = 0; i < n; i++) {
          double hw = mpq_get_d(xmax[i]) - mpq_get_d(xmin[i]), w = H[i] - L[i];
          if (hw > 0) over = fmax(over, w / hw);
          mpq_t t;
          mpq_init(t);
          mpq_set_d(t, L[i] + 0.01 * w); lost |= mpq_cmp(xmin[i], t) < 0;
          mpq_set_d(t, H[i] - 0.01 * w); lost |= mpq_cmp(xmax[i], t) > 0;
          mpq_clear(t);
        }
        shrunk_miss += lost;
      } else
        for (int t = 0; t < 12; t++) {                     /* sampled: vertices, then points inside */
          for (size_t q = 0; q < n * n; q++) A[q] = t < 6 ? (next() & 1 ? Ah[q] : Al[q]) : fmin(Ah[q], Al[q] + unit() * (Ah[q] - Al[q]));
          for (size_t i = 0; i < n; i++) b[i] = t < 6 ? (next() & 1 ? bh[i] : bl[i]) : fmin(bh[i], bl[i] + unit() * (bh[i] - bl[i]));
          if (!exact(n, A, b, x)) { miss++; continue; }
          for (size_t i = 0; i < n; i++) miss += !holds(L[i], H[i], x[i]);
          points++;
        }
    }
    for (size_t i = 0; i < n; i++) { mpq_clear(x[i]); mpq_clear(xmin[i]); mpq_clear(xmax[i]); }
    free(x); free(xmin); free(xmax); free(Al); free(Ah); free(bl); free(bh); free(L); free(H); free(A); free(b);
  }
  for (int rep = 0; rep < 40; rep++) {                     /* [A] holding a singular matrix */
    size_t n = 3 + next() % 10;   /* from 3: with n = 2 the "last row" is row 1 itself, and the first version of this
                                     test built 2-by-2 matrices that were not singular, 5 of which verified */
    double *Al = malloc(n * n * 8), *Ah = malloc(n * n * 8), *bl = malloc(n * 8), *bh = malloc(n * 8), *L = malloc(n * 8), *H = malloc(n * 8);
    for (size_t q = 0; q < n * n; q++) { double a = (double)(int)(next() % 7) - 3; Al[q] = Ah[q] = a; }
    for (size_t j = 0; j < n; j++) Al[(n - 1) * n + j] = Ah[(n - 1) * n + j] = Al[j] + Al[n + j];   /* singular */
    for (size_t q = 0; q < n * n; q++) {                   /* widened around it, the midpoint moved off it */
      double r = ldexp(1, -(int)(next() % 30) - 1);
      Al[q] -= rep % 2 ? r : r / 2; Ah[q] += r;
    }
    for (size_t i = 0; i < n; i++) { bl[i] = 1; bh[i] = 1; }
    singular++;
    singular_verified += crlin_solve_interval(n, 1, Al, Ah, n, bl, bh, 1, NULL, 0, L, H, 1, MODE, NULL, NULL) == CRLIN_VERIFIED;
    free(Al); free(Ah); free(bl); free(bh); free(L); free(H);
  }
  {                                                        /* [[1, [0.5, 2]], [1, 1]] holds a12 = 1, singular */
    double Al[4] = {1, 0.5, 1, 1}, Ah[4] = {1, 2, 1, 1}, bl[2] = {1, 1}, bh[2] = {1, 1}, L[2], H[2];
    singular++;
    singular_verified += crlin_solve_interval(2, 1, Al, Ah, 2, bl, bh, 1, NULL, 0, L, H, 1, MODE, NULL, NULL) == CRLIN_VERIFIED;
  }
  double Al[1] = {2}, Ah[1] = {1}, bl[1] = {1}, L[1], H[1];
  int refused = crlin_solve_interval(1, 1, Al, Ah, 1, bl, bl, 1, NULL, 0, L, H, 1, MODE, NULL, NULL) == -1;
  printf("interval data: %ld systems, %ld verified; %ld exact solutions of vertices and points inside, %ld outside "
         "(must be 0); hull known for %ld, the enclosure at most %.4g times its width; %ld holding a singular matrix, "
         "%ld verified (must be 0); lo > hi refused: %s; control: %ld enclosures shrunk by 1%% lose a vertex (must be > 0)\n",
         systems, verified, points, miss, small, over, singular, singular_verified, refused ? "yes" : "NO", shrunk_miss);
  if (miss || singular_verified || !refused) return 1;
  return !shrunk_miss || !points ? 2 : 0;
}

/* 10: very ill-conditioned. A = product of random row operations row_i += c row_j (c in -2..2, i != j) on the
   identity, until the largest entry reaches 2^bits: an integer matrix of determinant 1, whose inverse is an integer
   matrix as large, so its condition number is about the largest entry squared (computed below in rationals) */
static int unimodular(size_t n, int bits, double *A)
{
  for (size_t i = 0; i < n; i++) for (size_t j = 0; j < n; j++) A[i * n + j] = i == j;
  for (int t = 0; t < 100000; t++) {
    size_t i = next() % n, j = next() % n;
    int c = (int)(next() % 5) - 2;
    if (i == j || !c) continue;
    double big = 0;
    for (size_t k = 0; k < n; k++) big = fmax(big, fabs(A[i * n + k] + c * A[j * n + k]));
    if (big >= ldexp(1, bits)) return 1;                   /* the next step would pass 2^bits: stop here */
    for (size_t k = 0; k < n; k++) A[i * n + k] += c * A[j * n + k];
  }
  return 1;
}
static double cond_inf(size_t n, const double *A)   /* ||A|| ||A^-1|| in the infinity norm, A^-1 exactly */
{
  double na = 0, ni = 0, *e = calloc(n, sizeof *e);
  mpq_t *x = malloc(n * sizeof *x), *col = malloc(n * n * sizeof *col);
  for (size_t i = 0; i < n; i++) mpq_init(x[i]);
  for (size_t k = 0; k < n * n; k++) mpq_init(col[k]);
  for (size_t j = 0; j < n; j++) {
    memset(e, 0, n * sizeof *e); e[j] = 1;
    exact(n, A, e, x);
    for (size_t i = 0; i < n; i++) mpq_set(col[i * n + j], x[i]);
  }
  for (size_t i = 0; i < n; i++) {
    double ra = 0, ri = 0;
    for (size_t j = 0; j < n; j++) { ra += fabs(A[i * n + j]); ri += fabs(mpq_get_d(col[i * n + j])); }
    na = fmax(na, ra); ni = fmax(ni, ri);
  }
  for (size_t i = 0; i < n; i++) mpq_clear(x[i]);
  for (size_t k = 0; k < n * n; k++) mpq_clear(col[k]);
  free(x); free(col); free(e);
  return na * ni;
}
static int sec10(void)
{
  static const int bits[] = {20, 30, 40, 46, 50, 52};
  long before = T.wrong;
  printf("very ill-conditioned (determinant 1, integer):");
  for (size_t q = 0; q < sizeof bits / sizeof *bits; q++) {
    long v = 0, cases = 0;
    double cmin = INFINITY, cmax = 0;
    for (int rep = 0; rep < 12; rep++) {
      size_t n = 3 + rep % 8;
      double *A = malloc(n * n * sizeof *A), *b = malloc(n * sizeof *b);
      unimodular(n, bits[q], A);
      for (size_t i = 0; i < n; i++) b[i] = (double)(int)(next() % 21) - 10;
      int sg, s = one(n, A, b, NULL, &sg);
      double c = cond_inf(n, A);
      cmin = fmin(cmin, c); cmax = fmax(cmax, c);
      v += s == CRLIN_VERIFIED; cases++;
      free(A); free(b);
    }
    printf(" entries to 2^%d (cond %.0e to %.0e): %ld of %ld verified;", bits[q], cmin, cmax, v, cases);
  }
  printf(" %ld enclosures miss x* (must be 0)\n", T.wrong - before);
  return T.wrong != before;
}

static int worst(int a, int b) { return a == 1 || b == 1 ? 1 : a > b ? a : b; }
int main(void)
{
  int r = 0;
  for (MODE = CRLIN_EXACT; MODE <= CRLIN_EXTENDED; MODE++) {
    printf("== mode %s\n", MODE == CRLIN_EXACT ? "CRLIN_EXACT (I - R A from the exact product)" : MODE == CRLIN_CLASSICAL ?
           "CRLIN_CLASSICAL (I - R A from two products and an a priori bound)" : "CRLIN_EXTENDED (R and x~ as pairs, Rump's iteration)");
    rng = 0x5851f42d4c957f2dULL;   /* the same systems in both modes */
    r = worst(r, sec1());
    r = worst(r, sec2());
    r = worst(r, sec3());
    r = worst(r, sec4());
    r = worst(r, sec5());
    r = worst(r, sec6());
    r = worst(r, sec7());
    if (MODE != CRLIN_EXTENDED) {
      r = worst(r, sec8());
      r = worst(r, sec9());
    }
    r = worst(r, sec10());
  }
  printf("the exact solver: %ld systems whose solution failed A x = b in rationals (must be 0)\n", exact_bad);
  if (exact_bad) r = 2;
  if (r == 0) printf("VERDICT: IDENTICAL crlin's enclosures hold the exact solution, singular systems never verify, and every control differs\n");
  else if (r == 1) printf("VERDICT: DIFFERS: an enclosure missed the exact solution, a singular system verified, or threads changed the bits\n");
  else printf("VERDICT: VOID: a run above tested nothing, or its control did not differ\n");
  return r;
}
