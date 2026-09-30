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
   Every run has a control: a result moved by an ulp in one case in 64,
   and in at least one. The last line is the verdict. */
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
  return kit_verdict(r, "every sum and dot product is correctly rounded, in any order, and every control differs");
}
