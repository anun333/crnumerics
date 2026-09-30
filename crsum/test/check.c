/* check: crsum (crsum.h) against MPFR's correctly rounded mpfr_sum
   (kit_sum_ref), and against itself under every reordering tried.

     1  sums and dot products, binary64 and binary32, four modes, against
        the reference; the cases:
          random        terms at every exponent range of the format
          cancel        terms and their negatives, with small ones left over
          midpoint      sums exactly halfway between two results, and just
                        above or below by a term far down (for dot products,
                        a product below the format's range: 2^-1200 in
                        binary64, 2^-200 in binary32)
          subnormal     results in and below the subnormal range
          overflow      at and past the largest value, and cancelling back
          specials      NaN, infinities, zeros of either sign, no terms
          long          2^18 terms
     2  the same bits whatever the order: each case shuffled twice, and
        split at random over 2, 3 and 7 accumulators merged in a random
        order, in every mode
     3  real threads: 2^20 terms summed on 1, 2, 3, 4 and 8 OpenMP threads,
        one accumulator each, merged: the same bits each time
     4  a negative control: naive left-to-right summation, to nearest, must
        differ from the reference on the cancel and midpoint cases
     5  matrix products (crgemv, both orientations, and crgemm; binary64 and
        binary32; four modes): every element against the reference, with
        leading dimensions padded, beta 0, +-1 or random (0 with NaN in y,
        which must not be read), rows that cancel, NaN and infinities; sizes
        across the direct and binned paths; modes refused with nothing
        written; naive products must differ on the cancelling rows
     6  crgemm_oz (the Ozaki scheme) against crgemm, bit for bit, through
        four GEMMs: the internal one, one that sums every element's products
        in a random order alternating fused and separate multiply-adds, the
        system's BLAS (dgemm_ from libblas.so.3, when there is one), and a
        counting copy of the internal one that shows the Ozaki path ran
        (not the fallback). Ranges from one binade to 150 (the widest fall
        back), zero rows, subnormals, k up to 3000 (narrower slices).
        Negative control: a GEMM that rounds to binary32 must differ
   Every run has a control: a result moved by an ulp in one case in 64,
   and in at least one. The last line is the verdict. */
#include <dlfcn.h>
#include <fenv.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "kit.h"
#include "crsum.h"

static const mpfr_rnd_t RND[4] = {MPFR_RNDN, MPFR_RNDU, MPFR_RNDD, MPFR_RNDZ};   /* CRSUM_NEAREST ... CRSUM_ZERO */

static uint64_t rng = 0x243f6a8885a308d3ULL;
static uint64_t next(void)
{
  uint64_t z = (rng += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

/* a format's parameters: precision p, least exponent of a normal (emin),
   largest exponent (emax), least quantum (qmin) */
typedef struct { kit_fmt f; int p, emin, emax, qmin; const char *name; } fmtp;
static const fmtp F64 = {KIT_B64, 53, -1022, 1023, -1074, "binary64"}, F32 = {KIT_B32, 24, -126, 127, -149, "binary32"};

/* a random value of the format with exponent e (a subnormal below emin) */
static double val(const fmtp *F, int e)
{
  double s = next() & 1 ? -1 : 1;
  if (e < F->emin) return s * ldexp((double)(next() >> (64 - (F->p - 1))), F->qmin);   /* a subnormal */
  return s * ldexp((double)((next() >> (64 - (F->p - 1))) | 1ULL << (F->p - 1)), e - F->p + 1);
}
static int rexp(int lo, int hi) { return lo + (int)(next() % (uint64_t)(hi - lo + 1)); }

typedef struct { double *x, *y; size_t n; int kind; } tcase;
enum { RANDOM, CANCEL, MIDPOINT, SUBNORMAL, OVERFLOW, SPECIALS, LONG, NKIND };
static const char *KIND[NKIND] = {"random", "cancel", "midpoint", "subnormal", "overflow", "specials", "long"};
static tcase *T;
static size_t nt, capt;
static double *bx, *by;
static size_t bn, bcap;
static void push(double x, double y)
{
  if (bn == bcap) { bcap = bcap ? 2 * bcap : 1024; bx = realloc(bx, bcap * sizeof *bx); by = realloc(by, bcap * sizeof *by); }
  bx[bn] = x;
  by[bn++] = y;
}
/* the terms pushed so far as a case, shuffled (the order must not matter) */
static void done(int kind)
{
  if (nt == capt) { capt = capt ? 2 * capt : 256; T = realloc(T, capt * sizeof *T); }
  size_t *idx = malloc((bn ? bn : 1) * sizeof *idx);
  kit_shuffle(next(), idx, bn);
  tcase c = {malloc((bn ? bn : 1) * sizeof(double)), malloc((bn ? bn : 1) * sizeof(double)), bn, kind};
  for (size_t i = 0; i < bn; i++) { c.x[i] = bx[idx[i]]; c.y[i] = by[idx[i]]; }
  free(idx);
  T[nt++] = c;
  bn = 0;
}

/* the cases for format F; dot: y matters (else y is 1) */
static void cases(const fmtp *F, int dot)
{
  nt = 0;
  int E = F->emax, Q = F->qmin, P = F->p;
  /* random, at every exponent range */
  const int win[5][2] = {{-10, 10}, {-E / 3, E / 3}, {Q, E}, {E - 60, E}, {Q, F->emin + 20}};
  for (int w = 0; w < 5; w++)
    for (int k = 0; k < 60; k++) {
      int n = 1 + (int)(next() % 300);
      for (int i = 0; i < n; i++) {
        int lo = win[w][0], hi = win[w][1];
        if (dot) { lo /= 2; hi /= 2; }   /* products in range, mostly */
        push(val(F, rexp(lo, hi)), dot ? val(F, rexp(lo, hi)) : 1);
      }
      done(RANDOM);
    }
  /* cancellation: terms and their negatives, and a few small ones left */
  for (int k = 0; k < 200; k++) {
    int n = 1 + (int)(next() % 100), top = rexp(-E / 4, E / 4);
    for (int i = 0; i < n; i++) {
      double v = val(F, top - rexp(0, 40)), w = dot ? val(F, rexp(-20, 20)) : 1;
      push(v, w);
      push(-v, w);
    }
    for (int i = 0; i < 3; i++) push(val(F, top - rexp(60, 200 < E ? 200 : E / 2)), 1);
    done(CANCEL);
  }
  /* midpoints: a + half an ulp of a, then just above and below by a far
     term; disguised by a large pair that cancels */
  for (int k = 0; k < 400; k++) {
    int e = rexp(F->emin + P + 80, E - 40);
    double a = fabs(val(F, e)), half = ldexp(1, e - P);   /* a in [2^e, 2^(e+1)): its ulp is 2^(e-P+1) */
    push(a, 1);
    push(half, 1);
    int which = (int)(next() % 3);
    if (which) {
      double far = ldexp(1, e - P - 40);
      if (dot) {   /* a product below the format's range: binary64 2^-1200, binary32 2^-200 */
        int t = F->f == KIT_B64 ? -600 : -100;
        push(ldexp(1, t), ldexp(which == 1 ? 1 : -1, t));
      }
      else push(which == 1 ? far : -far, 1);
    }
    double big = fabs(val(F, e + 30 < E ? e + 30 : E));
    push(big, 1);
    push(-big, 1);
    if (next() & 1) for (size_t i = 0; i < bn; i++) bx[i] = -bx[i];   /* the negative side too */
    done(MIDPOINT);
  }
  /* subnormal results: small terms, and for dot products, tiny products */
  for (int k = 0; k < 200; k++) {
    int n = 1 + (int)(next() % 20);
    for (int i = 0; i < n; i++) {
      if (dot) push(val(F, rexp(Q / 2 - 10, Q / 2 + 10)), val(F, rexp(Q / 2 - 10, Q / 2 + 10)));
      else push(val(F, rexp(Q, F->emin + 2)), 1);
    }
    done(SUBNORMAL);
  }
  /* overflow: at and past the largest value, and back */
  {
    double M = F->f == KIT_B64 ? DBL_MAX : FLT_MAX, tiny = ldexp(1, E - P - 1);
    double sets[][4] = {{M, M, 0, 0}, {M, M, -M, 0}, {M, tiny, 0, 0}, {M, tiny, tiny, 0}, {-M, -M, M, -M}, {M, -tiny, tiny, 0}};
    for (int s = 0; s < 6; s++) {
      for (int j = 0; j < 4; j++) if (sets[s][j] != 0) push(sets[s][j], 1);
      done(OVERFLOW);
      for (int j = 0; j < 4; j++) if (sets[s][j] != 0) push(-sets[s][j], 1);
      done(OVERFLOW);
    }
    for (int k = 0; k < 40; k++) {
      int n = 2 + (int)(next() % 6);
      for (int i = 0; i < n; i++) push(dot ? val(F, rexp(E / 2 - 2, E / 2 + 1)) : val(F, rexp(E - 2, E)), dot ? val(F, rexp(E / 2 - 2, E / 2 + 1)) : 1);
      done(OVERFLOW);
    }
  }
  /* specials */
  {
    double I = INFINITY, N = NAN;
    double sets[][3] = {{I, 1, 0}, {-I, 1, 0}, {I, -I, 0}, {N, 1, 0}, {I, N, 0}, {0.0, 0.0, 0}, {-0.0, -0.0, 0},
                        {0.0, -0.0, 0}, {1, -1, 0}, {-0.0, 0, 0}, {I, 0, 0}};
    int len[] = {2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1};
    for (int s = 0; s < 11; s++) {
      for (int j = 0; j < len[s]; j++) push(sets[s][j], dot && s == 10 ? 0.0 : dot ? 1 : 1);   /* inf * 0 for dot */
      done(SPECIALS);
    }
    done(SPECIALS);   /* no terms */
    push(-0.0, 1);
    push(-0.0, -0.0);   /* dot: -0 * -0 = +0 */
    done(SPECIALS);
  }
  /* long */
  for (int k = 0; k < 4; k++) {
    for (int i = 0; i < 1 << 18; i++) {
      int e = k < 2 ? rexp(-30, 30) : 5;   /* k >= 2: every term the same size, the limbs fill up */
      /* k == 3: every term (every product) positive, so no bin's overflow
         can cancel against the other sign's: the bins must be emptied */
      push(k == 3 ? fabs(val(F, e)) : val(F, e), dot ? (k == 3 ? fabs(val(F, 0)) : val(F, rexp(-5, 5))) : 1);
    }
    done(LONG);
  }
}

/* crsum on case c, format F, mode m; through an accumulator split kways
   (0: the plain call), merged in a shuffled order */
static double run1(const fmtp *F, int dot, const tcase *c, int m, int kways, uint64_t seed)
{
  if (!kways) {
    if (F->f == KIT_B64) return dot ? crdot(c->x, c->y, c->n, m) : crsum(c->x, c->n, m);
    float *x = malloc((c->n ? c->n : 1) * sizeof *x), *y = malloc((c->n ? c->n : 1) * sizeof *y);
    for (size_t i = 0; i < c->n; i++) { x[i] = (float)c->x[i]; y[i] = (float)c->y[i]; }
    float r = dot ? crdotf(x, y, c->n, m) : crsumf(x, c->n, m);
    free(x);
    free(y);
    return r;
  }
  crsum_acc *acc = malloc((size_t)kways * sizeof *acc);
  for (int k = 0; k < kways; k++) crsum_init(&acc[k]);
  uint64_t z = seed;
  for (size_t i = 0; i < c->n; i++) {
    z = z * 6364136223846793005ULL + 1442695040888963407ULL;
    int k = (int)((z >> 33) % (uint64_t)kways);
    if (dot) crsum_add_dot(&acc[k], &c->x[i], &c->y[i], 1); else crsum_add(&acc[k], &c->x[i], 1);
  }
  size_t order[8];
  kit_shuffle(seed ^ 0x5bd1e995, order, (size_t)kways);
  crsum_acc all;
  crsum_init(&all);
  for (int k = 0; k < kways; k++) crsum_merge(&all, &acc[order[k]]);
  free(acc);
  return F->f == KIT_B64 ? crsum_round(&all, m) : crsum_roundf(&all, m);
}

static int same(double a, double b) { return (isnan(a) && isnan(b)) || !memcmp(&a, &b, 8); }

/* sections 1 and 2 for one format and kind of reduction */
static int section(const fmtp *F, int dot)
{
  cases(F, dot);
  unsigned long long bad[NKIND] = {0}, cnt[NKIND] = {0}, ctl = 0, reorder = 0, reorder_n = 0;
  long first = -1;
  int first_m = 0;
  double first_got = 0, first_want = 0;
  int round0 = fegetround();
#pragma omp parallel for reduction(+ : ctl, reorder, reorder_n) schedule(dynamic, 1)
  for (size_t t = 0; t < nt; t++) {
    const tcase *c = &T[t];
    for (int m = 0; m < 4; m++) {
      double got = run1(F, dot, c, m, 0, 0);
      double want = kit_sum_ref(F->f, c->x, dot ? c->y : NULL, c->n, RND[m]);
      int differ = !same(got, want);
#pragma omp atomic
      cnt[c->kind]++;
      if (differ) {
#pragma omp atomic
        bad[c->kind]++;
#pragma omp critical
        if (first < 0 || (long)t < first) { first = (long)t; first_m = m; first_got = got; first_want = want; }
      }
      int moved = (t == nt / 2 && m == 0) || ((t * 4 + (size_t)m) * 0x9e3779b97f4a7c15ULL >> 58) == 11;
      double cg = got;
      if (moved) cg = isnan(got) ? 0 : got == INFINITY ? DBL_MAX : nextafter(got, INFINITY);
      ctl += !same(cg, want);
      /* the same bits whatever the order and split */
      static const int KW[4] = {1, 2, 3, 7};
      for (int w = 0; w < 4; w++) {
        double o = run1(F, dot, c, m, KW[w], t * 131 + (size_t)w);
        reorder_n++;
        reorder += !same(o, got);
      }
    }
  }
  int r = 0;
  unsigned long long tb = 0, tc = 0;
  for (int k = 0; k < NKIND; k++) { tb += bad[k]; tc += cnt[k]; }
  printf("%s %-4s %6llu results, %llu differ (control: %llu differ)", F->name, dot ? "dot" : "sum", tc, tb, ctl);
  for (int k = 0; k < NKIND; k++) if (bad[k]) printf(" [%s: %llu]", KIND[k], bad[k]);
  if (first >= 0) printf("\n    first: case %ld (%s, n %zu), mode %d: got %a, want %a", first, KIND[T[first].kind], T[first].n, first_m, first_got, first_want);
  printf("\n%s %-4s %6llu reordered and split results, %llu differ from the plain call\n", F->name, dot ? "dot" : "sum",
         reorder_n, reorder);
  if (fegetround() != round0) { printf("    the C rounding mode changed\n"); r = 1; }
  if (!ctl) { printf("    VOID: the control did not differ\n"); r = kit_worst(r, 2); }
  if (tb || reorder) r = kit_worst(r, 1);
  for (size_t t = 0; t < nt; t++) { free(T[t].x); free(T[t].y); }
  return r;
}

/* section 3: real threads */
static int threads(void)
{
  size_t n = 1 << 20;
  double *x = malloc(n * sizeof *x);
  for (size_t i = 0; i < n; i++) x[i] = val(&F64, rexp(-40, 40));
  double want = kit_sum_ref(KIT_B64, x, NULL, n, MPFR_RNDN), first = 0;
  int bad = 0;
  static const int NT[5] = {1, 2, 3, 4, 8};
  for (int k = 0; k < 5; k++) {
    crsum_acc all;
    crsum_init(&all);
#pragma omp parallel num_threads(NT[k])
    {
      crsum_acc mine;
      crsum_init(&mine);
#pragma omp for schedule(static)
      for (size_t i = 0; i < n; i++) crsum_add(&mine, &x[i], 1);
#pragma omp critical
      crsum_merge(&all, &mine);   /* in whatever order the threads arrive */
    }
    double r = crsum_round(&all, CRSUM_NEAREST);
    if (k == 0) first = r;
    bad += !same(r, first) || !same(r, want);
  }
  printf("threads    2^20 terms on 1, 2, 3, 4, 8 threads: %s\n", bad ? "DIFFERENT" : "the same bits, and MPFR's");
  free(x);
  return bad ? 1 : 0;
}

/* section 4: the naive sum must differ somewhere on hard cases */
static int negative(void)
{
  cases(&F64, 0);
  unsigned long long differ = 0, tried = 0;
  for (size_t t = 0; t < nt; t++) {
    if (T[t].kind != CANCEL && T[t].kind != MIDPOINT) continue;
    double s = 0;
    for (size_t i = 0; i < T[t].n; i++) s += T[t].x[i];
    tried++;
    differ += !same(s, kit_sum_ref(KIT_B64, T[t].x, NULL, T[t].n, MPFR_RNDN));
  }
  for (size_t t = 0; t < nt; t++) { free(T[t].x); free(T[t].y); }
  printf("naive sum  %llu of %llu cancel and midpoint cases differ %s\n", differ, tried,
         differ ? "(as they must)" : "NEGATIVE CONTROL FAILED: they must differ");
  return differ ? 0 : 2;
}

/* section 5: matrices */
typedef struct { const fmtp *F; int kind; } mcfg;   /* kind 0: random, 1: rows that cancel, 2: specials */
static double mval(const fmtp *F, int kind, size_t i, size_t j)
{
  if (kind == 2 && next() % 40 == 0) { double sp[4] = {NAN, INFINITY, -INFINITY, 0.0}; return sp[next() % 4]; }
  (void)i; (void)j;
  return val(F, rexp(F->f == KIT_B64 ? -200 : -30, F->f == KIT_B64 ? 200 : 30));
}
/* the reference for one element: products u[j] v[j] and beta w */
static double mref(const fmtp *F, const double *u, const double *v, size_t n, double beta, double w, int m)
{
  double *uu = malloc((n + 1) * sizeof *uu), *vv = malloc((n + 1) * sizeof *vv);
  memcpy(uu, u, n * sizeof *uu);
  memcpy(vv, v, n * sizeof *vv);
  size_t len = n;
  if (beta != 0) { uu[len] = beta; vv[len++] = w; }
  double r = kit_sum_ref(F->f, uu, vv, len, RND[m]);
  free(uu);
  free(vv);
  return r;
}
static int matrices(void)
{
  unsigned long long tested = 0, bad = 0, ctl = 0, neg_tried = 0, neg_differ = 0;
  int refused_ok = 1;
  const fmtp *FS[2] = {&F64, &F32};
  for (int fi = 0; fi < 2; fi++) {
    const fmtp *F = FS[fi];
    for (int c = 0; c < 240; c++) {
      int kind = c % 3, trans = (c / 3) % 2, gemm = c % 8 == 7;
      size_t m = 1 + next() % (c % 5 == 0 ? 150 : 40), n = 1 + next() % (c % 5 == 1 ? 150 : 40), k = 1 + next() % (c % 5 == 2 ? 100 : 20);
      if (gemm) { m = 1 + next() % 12; n = 1 + next() % 12; }
      size_t cols = gemm ? k : n, lda = cols + next() % 4, ldb = n + next() % 4, ldc = n + next() % 4;
      size_t arows = m;
      double *A = malloc(arows * lda * sizeof *A), *B = malloc(k * ldb * sizeof *B);
      for (size_t i = 0; i < arows; i++)
        for (size_t j = 0; j < lda; j++) A[i * lda + j] = j < cols ? mval(F, kind, i, j) : NAN;   /* padding: must not be read */
      for (size_t i = 0; i < k; i++)
        for (size_t j = 0; j < ldb; j++) B[i * ldb + j] = j < n ? mval(F, kind, i, j) : NAN;
      size_t xin = gemm ? 0 : trans ? m : n, yout = gemm ? m * ldc : trans ? n : m;
      double *x = malloc((xin ? xin : 1) * sizeof *x), *y0 = malloc(yout * sizeof *y0);
      for (size_t j = 0; j < xin; j++) x[j] = mval(F, kind == 2 ? 0 : kind, 0, j);
      if (kind == 1 && !gemm) {   /* rows that cancel: every row (column) pairs each product with its negative, plus a small one */
        size_t len = trans ? m : n, outs = trans ? n : m;
        for (size_t o = 0; o < outs; o++)
          for (size_t j = 0; j + 1 < len; j += 2) {
            double *p = trans ? &A[j * lda + o] : &A[o * lda + j], *q = trans ? &A[(j + 1) * lda + o] : &A[o * lda + j + 1];
            *q = -*p;
            x[j + 1] = x[j];
            if (j == 0 && len > 2) *q = -*p + ldexp(*p, -F->p - 20);   /* not quite: a small remainder */
          }
        for (size_t o = 0; o < outs; o++) {   /* keep it exact in the format */
          size_t len2 = trans ? m : n;
          for (size_t j = 0; j < len2; j++) {
            double *p = trans ? &A[j * lda + o] : &A[o * lda + j];
            if (F->f == KIT_B32) *p = (float)*p;
          }
        }
      }
      double betas[5] = {0, 1, -1, val(F, rexp(-3, 3)), 0};
      double beta = betas[c % 5];
      for (size_t i = 0; i < yout; i++) y0[i] = c % 5 == 4 ? NAN : val(F, rexp(-10, 10));   /* beta 0 with NaN: not read */
      for (int md = 0; md < 4; md++) {
        double *y = malloc(yout * sizeof *y);
        memcpy(y, y0, yout * sizeof *y);
        int rc;
        if (F->f == KIT_B64) {
          rc = gemm ? crgemm(m, n, k, A, lda, B, ldb, beta, y, ldc, md) : crgemv(trans, m, n, A, lda, x, beta, y, md);
        } else {   /* binary32: through float copies */
          float *Af = malloc(arows * lda * sizeof *Af), *Bf = malloc(k * ldb * sizeof *Bf), *xf = malloc((xin ? xin : 1) * sizeof *xf), *yf = malloc(yout * sizeof *yf);
          for (size_t i = 0; i < arows * lda; i++) Af[i] = (float)A[i];
          for (size_t i = 0; i < k * ldb; i++) Bf[i] = (float)B[i];
          for (size_t i = 0; i < xin; i++) xf[i] = (float)x[i];
          for (size_t i = 0; i < yout; i++) yf[i] = (float)y[i];
          rc = gemm ? crgemmf(m, n, k, Af, lda, Bf, ldb, (float)beta, yf, ldc, md) : crgemvf(trans, m, n, Af, lda, xf, (float)beta, yf, md);
          for (size_t i = 0; i < yout; i++) y[i] = yf[i];
          free(Af); free(Bf); free(xf); free(yf);
        }
        if (rc) { bad++; free(y); continue; }
        size_t outs = gemm ? m * n : yout;
        for (size_t o = 0; o < outs; o++) {
          double want, got;
          if (gemm) {
            size_t i = o / n, j = o % n;
            double *col = malloc(k * sizeof *col);
            for (size_t l = 0; l < k; l++) col[l] = B[l * ldb + j];
            want = mref(F, &A[i * lda], col, k, beta, y0[i * ldc + j], md);
            got = y[i * ldc + j];
            free(col);
          } else {
            size_t len = trans ? m : n;
            double *row = malloc(len * sizeof *row);
            for (size_t j = 0; j < len; j++) row[j] = trans ? A[j * lda + o] : A[o * lda + j];
            want = mref(F, row, x, len, beta, y0[o], md);
            got = y[o];
            if (md == 0 && kind == 1) {   /* the negative control: a naive product */
              double s = 0;
              for (size_t j = 0; j < len; j++) s += row[j] * x[j];
              if (beta != 0) s += beta * y0[o];
              if (F->f == KIT_B32) s = (float)s;
              neg_tried++;
              neg_differ += !same(s, want);
            }
            free(row);
          }
          tested++;
          bad += !same(got, want);
          int moved = (tested * 0x9e3779b97f4a7c15ULL >> 58) == 7 || tested == 1;
          double cg = moved ? (isnan(got) ? 0 : got == INFINITY ? DBL_MAX : nextafter(got, INFINITY)) : got;
          ctl += !same(cg, want);
        }
        free(y);
      }
      /* a mode it doesn't take: -1, nothing written */
      double *y = malloc(yout * sizeof *y);
      for (size_t i = 0; i < yout; i++) y[i] = 12345;
      int rc = gemm ? crgemm(m, n, k, A, lda, B, ldb, beta, y, ldc, 4) : crgemv(trans, m, n, A, lda, x, beta, y, -1);
      for (size_t i = 0; i < yout; i++) if (y[i] != 12345) refused_ok = 0;
      if (rc != -1) refused_ok = 0;
      free(y);
      free(A); free(B); free(x); free(y0);
    }
  }
  printf("matrices   %llu elements (gemv both ways, gemm; binary64 and binary32; four modes), %llu differ (control: %llu differ)\n",
         tested, bad, ctl);
  printf("matrices   modes they don't take: %s\n", refused_ok ? "refused, nothing written" : "NOT REFUSED");
  printf("matrices   naive products differ on %llu of %llu cancelling rows %s\n", neg_differ, neg_tried,
         neg_differ ? "(as they must)" : "NEGATIVE CONTROL FAILED: they must differ");
  if (!ctl) return 2;
  return bad || !refused_ok ? 1 : neg_differ ? 0 : 2;
}

/* section 6: the Ozaki scheme */
static void gemm_plain(size_t m, size_t n, size_t k, const double *A, size_t lda, const double *B, size_t ldb, double *C,
                       size_t ldc, void *ctx)
{
  if (ctx) ++*(unsigned long long *)ctx;   /* the Ozaki path ran */
  for (size_t i = 0; i < m; i++)
    for (size_t j = 0; j < n; j++) {
      double c = 0;
      for (size_t l = 0; l < k; l++) c += A[i * lda + l] * B[l * ldb + j];
      C[i * ldc + j] = c;
    }
}
static void gemm_scrambled(size_t m, size_t n, size_t k, const double *A, size_t lda, const double *B, size_t ldb,
                           double *C, size_t ldc, void *ctx)
{
  uint64_t seed = ctx ? (*(uint64_t *)ctx)++ : 7;
  size_t *ord = malloc(k * sizeof *ord);
  for (size_t i = 0; i < m; i++)
    for (size_t j = 0; j < n; j++) {
      kit_shuffle(seed + i * 131 + j, ord, k);
      double c = 0;
      for (size_t t = 0; t < k; t++) {
        size_t l = ord[t];
        c = t & 1 ? fma(A[i * lda + l], B[l * ldb + j], c) : c + A[i * lda + l] * B[l * ldb + j];
      }
      C[i * ldc + j] = c;
    }
  free(ord);
}
static void gemm_sloppy(size_t m, size_t n, size_t k, const double *A, size_t lda, const double *B, size_t ldb, double *C,
                        size_t ldc, void *ctx)
{
  (void)ctx;
  for (size_t i = 0; i < m; i++)
    for (size_t j = 0; j < n; j++) {
      float c = 0;
      for (size_t l = 0; l < k; l++) c += (float)A[i * lda + l] * (float)B[l * ldb + j];
      C[i * ldc + j] = c;
    }
}
typedef void (*fdgemm)(const char *, const char *, const int *, const int *, const int *, const double *, const double *,
                       const int *, const double *, const int *, const double *, double *, const int *);
static fdgemm sys_dgemm;
static void gemm_system(size_t m, size_t n, size_t k, const double *A, size_t lda, const double *B, size_t ldb, double *C,
                        size_t ldc, void *ctx)
{
  (void)ctx;
  int M = (int)m, N = (int)n, K = (int)k, LA = (int)lda, LB = (int)ldb, LC = (int)ldc;
  double one = 1, zero = 0;
  sys_dgemm("N", "N", &N, &M, &K, &one, B, &LB, A, &LA, &zero, C, &LC);   /* row-major C = A B is column-major C^T = B^T A^T */
}

static int ozaki(void)
{
  void *h = dlopen("libblas.so.3", RTLD_NOW);
  sys_dgemm = h ? (fdgemm)dlsym(h, "dgemm_") : NULL;
  unsigned long long tested = 0, bad[4] = {0}, ctl = 0, oz_cases = 0, cases_n = 0, neg = 0, neg_tried = 0;
  uint64_t scr_seed = 1;
  static const int RANGE[5] = {0, 6, 20, 45, 150};
  for (int c = 0; c < 300; c++) {
    size_t m = 1 + next() % 24, n = 1 + next() % 24, k = c % 10 == 9 ? 3000 : c % 10 == 8 ? 300 : 1 + next() % 60;
    if (k >= 300) { m = 1 + next() % 6; n = 1 + next() % 6; }
    if (c < 4) { m = 3; n = 2; k = 1 + (size_t)c; }   /* exact zeros of one sign: below (row 2 nonzero, so the Ozaki path runs) */
    else if (c < 16) { m = 2; n = 3; k = 3; }          /* midpoints: below */
    int R = RANGE[c % 5], base = rexp(-300, 300);
    if (c % 7 == 3) base = -1074 + 60 + R;   /* near the subnormals */
    size_t lda = k + next() % 3, ldb = n + next() % 3, ldc = n + next() % 3;
    double *A = malloc(m * lda * sizeof *A), *B = malloc(k * ldb * sizeof *B), *C0 = malloc(m * ldc * sizeof *C0);
    for (size_t i = 0; i < m * lda; i++) A[i] = val(&F64, base - rexp(0, R));
    for (size_t i = 0; i < k * ldb; i++) B[i] = val(&F64, rexp(-R, 0));
    if (c % 6 == 2 && m > 1) for (size_t l = 0; l < k; l++) A[l] = 0;   /* a zero row */
    if (c % 6 == 4) for (size_t l = 0; l + 1 < k; l += 2) { A[l + 1] = A[l]; B[(l + 1) * ldb] = -B[l * ldb]; }   /* column 0 cancels */
    if (c % 23 == 5) A[0] = NAN;   /* the fallback */
    /* beta on its own cycle (period 4), so that zero rows (c = 2 mod 6) and
       the cancelling column (c = 4 mod 6) also come with beta = 0, where an
       exact zero's sign shows */
    double beta = (c % 4 == 0) ? 0 : val(&F64, rexp(-2, 2));
    if (c < 4) {   /* rows of +0 and -0 against a column of positives and one of negatives: every zero product of one
                      sign, so each exact zero's sign is fixed (+0, -0, -0, +0) and only the direct path gets it; row 2
                      stays random: a matrix of zeros alone falls back to crgemm whole, and would test nothing here */
      for (size_t l = 0; l < k; l++) {
        A[l] = 0.0;
        A[lda + l] = -0.0;
        B[l * ldb] = fabs(val(&F64, 0));
        B[l * ldb + 1] = -fabs(val(&F64, 0));
      }
      beta = 0;
    } else if (c < 16) {   /* column 0: a + h, exactly halfway between two binary64 values; columns 1 and 2: just above and
                              below it, by a product far down. Odd c: a subnormal x plus 2^-1075, half the subnormal
                              quantum, which exists only as a product: A's row [x 2^10, 2^-1074, 2^-1074] against
                              [2^-10, 1/2, 0 or +-2^-30] (a narrow range, so the Ozaki path runs, not the fallback) */
      int sub = c & 1;
      for (size_t i = 0; i < 2; i++) {
        double a = sub ? ldexp(val(&F64, -1030), 10) : val(&F64, rexp(-200, 200));
        int ea = ilogb(a);
        A[i * lda] = a;
        A[i * lda + 1] = sub ? ldexp(1, -1074) : ldexp(1, ea - 53) * (a < 0 ? -1 : 1);
        A[i * lda + 2] = sub ? ldexp(1, -1074) : ldexp(1, ea - 133);
      }
      for (size_t j = 0; j < 3; j++) {
        B[j] = sub ? ldexp(1, -10) : 1;
        B[ldb + j] = sub ? 0.5 : 1;
        B[2 * ldb + j] = j == 0 ? 0 : (j == 1 ? 1 : -1) * (sub ? ldexp(1, -30) : 1);
      }
      beta = 0;
    }
    for (size_t i = 0; i < m * ldc; i++) C0[i] = val(&F64, base - R);
    cases_n++;
    for (int md = 0; md < 4; md++) {
      double *want = malloc(m * ldc * sizeof *want);
      memcpy(want, C0, m * ldc * sizeof *want);
      crgemm(m, n, k, A, lda, B, ldb, beta, want, ldc, md);
      crsum_dgemm G[4] = {NULL, gemm_scrambled, sys_dgemm ? gemm_system : NULL, gemm_plain};
      for (int g = 0; g < 4; g++) {
        if (g == 2 && !sys_dgemm) continue;
        unsigned long long calls = 0;
        void *ctx = g == 1 ? (void *)&scr_seed : g == 3 ? (void *)&calls : NULL;
        double *got = malloc(m * ldc * sizeof *got);
        memcpy(got, C0, m * ldc * sizeof *got);
        crgemm_oz(m, n, k, A, lda, B, ldb, beta, got, ldc, md, G[g], ctx);
        for (size_t i = 0; i < m; i++)
          for (size_t j = 0; j < n; j++) {
            double gv = got[i * ldc + j], wv = want[i * ldc + j];
            tested++;
            bad[g] += !same(gv, wv);
            int moved = (tested * 0x9e3779b97f4a7c15ULL >> 58) == 3 || tested == 1;
            ctl += !same(moved ? (isnan(gv) ? 0 : gv == INFINITY ? DBL_MAX : nextafter(gv, INFINITY)) : gv, wv);
          }
        if (g == 3 && md == 0 && calls) oz_cases++;
        free(got);
      }
      if (md == 0) {   /* negative control: a GEMM in binary32 */
        double *got = malloc(m * ldc * sizeof *got);
        memcpy(got, C0, m * ldc * sizeof *got);
        crgemm_oz(m, n, k, A, lda, B, ldb, beta, got, ldc, md, gemm_sloppy, NULL);
        for (size_t i = 0; i < m; i++)
          for (size_t j = 0; j < n; j++) { neg_tried++; neg += !same(got[i * ldc + j], want[i * ldc + j]); }
        free(got);
      }
      free(want);
    }
    free(A); free(B); free(C0);
  }
  printf("ozaki      %llu elements against crgemm: internal GEMM %llu differ, scrambled %llu, system BLAS %s, counting %llu (control: %llu differ)\n",
         tested, bad[0], bad[1], sys_dgemm ? (bad[2] ? "DIFFER" : "0 differ") : "not found (skipped)", bad[3], ctl);
  if (sys_dgemm && bad[2]) printf("    system BLAS: %llu differ\n", bad[2]);
  printf("ozaki      the Ozaki path ran on %llu of %llu cases (the rest fell back: NaN, or too wide)\n", oz_cases, cases_n);
  printf("ozaki      a binary32 GEMM differs on %llu of %llu elements %s\n", neg, neg_tried,
         neg ? "(as it must)" : "NEGATIVE CONTROL FAILED: it must differ");
  if (h) dlclose(h);
  if (!ctl || !neg || oz_cases < cases_n / 2) return 2;
  return bad[0] || bad[1] || bad[2] || bad[3] ? 1 : 0;
}

int main(void)
{
  int r = 0;
  printf("1, 2: against MPFR's mpfr_sum in four modes, and the same bits in any order or split\n");
  r = kit_worst(r, section(&F64, 0));
  r = kit_worst(r, section(&F64, 1));
  r = kit_worst(r, section(&F32, 0));
  r = kit_worst(r, section(&F32, 1));
  printf("3: threads\n");
  r = kit_worst(r, threads());
  printf("4: negative control\n");
  r = kit_worst(r, negative());
  printf("5: matrix products\n");
  r = kit_worst(r, matrices());
  printf("6: the Ozaki scheme\n");
  r = kit_worst(r, ozaki());
  return kit_verdict(r, "every sum and dot product is correctly rounded, in any order, and every control differs");
}
