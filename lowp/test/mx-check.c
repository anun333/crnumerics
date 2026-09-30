/* mx-check: lowp's MX functions (lowp.h, lowp/MX.md) against the kit's
   exact reference (kit_mx_ref1, kit_mx_ref2), in four rounding modes, for
   every element type and function.

   Blocks can't be enumerated, so they are built with purpose:
     values   every element value that isn't NaN or infinite, at every
              scale (FP8: every eighth scale and the 16 at each end),
              grouped by sign and by whether the function's exact result
              is finite, so that a domain error doesn't hide its
              neighbours
     errors   one element whose result is NaN or infinite among finite
              ones: the whole block must be NaN
     specials a NaN scale; FP8 blocks holding a NaN or an infinity; zeros
     random   512 blocks per type, function and mode (pairs: 1024)
   and every block's output is compared element by element, scale too.
   Also: q = p and y = x in place give the same blocks; modes lowp doesn't
   take are refused with nothing written; a negative control (E2M1 exp
   against MPFR's exp2) must differ. Every run has a control: one element
   moved (the scale, in a NaN block) in about one block in 64, and in at
   least one. The last line is the verdict. */
#include <fenv.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "kit.h"
#include "lowp.h"

#define K LOWP_MX_K
static const mpfr_rnd_t RND[4] = {MPFR_RNDN, MPFR_RNDU, MPFR_RNDD, MPFR_RNDZ};
static const kit_fmt TYPES[5] = {KIT_E5M2, KIT_E4M3, KIT_E3M2, KIT_E2M3, KIT_E2M1};
static const char *TNAME[5] = {"e5m2", "e4m3", "e3m2", "e2m3", "e2m1"};

typedef int (*mx1)(const uint8_t *, const uint8_t *, uint8_t *, uint8_t *, size_t, int);
typedef int (*mx2)(const uint8_t *, const uint8_t *, const uint8_t *, const uint8_t *, uint8_t *, uint8_t *, size_t, int);
static const struct { int t; const char *name; mx1 f1; mx2 f2; } L[] = {
#define LOWP_MX1(t, f) {0, #f, lowp_mx_##t##_##f, 0},
#define LOWP_MX2(t, f) {0, #f, 0, lowp_mx_##t##_##f},
#include "lowp-mx-list.h"
};
enum { NL = sizeof L / sizeof *L, PER = NL / 5 };   /* the list is PER functions for each type, in order */

static kit_mpfr1 ref1(const char *n)
{
  for (int i = 0; i < kit_nfns1; i++) if (!strcmp(kit_fns1[i].name, n)) return kit_fns1[i].ref;
  return 0;
}
static kit_mpfr2 ref2(const char *n)
{
  for (int i = 0; i < kit_nfns2; i++) if (!strcmp(kit_fns2[i].name, n)) return kit_fns2[i].ref;
  return 0;
}

static uint64_t rng = 0x2545f4914f6cdd1dULL;
static uint64_t next(void)
{
  uint64_t z = (rng += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

/* a block to try: elements and scale (and the second argument's) */
typedef struct { uint8_t p[K], x, p2[K], x2; } blk;
static blk *B;
static size_t nb, cap;
static void add(const blk *b)
{
  if (nb == cap) B = realloc(B, (cap = cap ? 2 * cap : 4096) * sizeof *B);
  B[nb++] = *b;
}

/* is the exact result at this element finite (and the element a number)? */
static int finite_at(kit_fmt f, kit_mpfr1 r1, uint8_t c, int x)
{
  double d = kit_decode(f, c);
  if (isnan(d) || isinf(d)) return 0;
  mpfr_t v, z;
  mpfr_inits2(64, v, z, (mpfr_ptr)0);
  mpfr_set_d(v, d, MPFR_RNDN);
  mpfr_mul_2si(v, v, x - 127, MPFR_RNDN);
  r1(z, v, MPFR_RNDZ);
  int ok = !mpfr_nan_p(z) && !mpfr_inf_p(z);
  mpfr_clears(v, z, (mpfr_ptr)0);
  return ok;
}

/* the blocks for type f and a one-argument function */
static void blocks1(kit_fmt f, kit_mpfr1 r1)
{
  int bits = kit_info(f)->bits, n = 1 << bits;
  nb = 0;
  for (int x = 0; x < 255; x++) {
    if (bits == 8 && x % 8 && x > 15 && x < 239) continue;
    for (int sg = 0; sg < 2; sg++) {
      uint8_t ok[256], bad[256];
      int no = 0, nbad = 0;
      for (int c = 0; c < n / 2; c++) {
        uint8_t code = (uint8_t)(c | sg << (bits - 1));
        double d = kit_decode(f, code);
        if (isnan(d) || isinf(d)) continue;
        if (finite_at(f, r1, code, x)) ok[no++] = code; else bad[nbad++] = code;
      }
      for (int i = 0; i < no; i += K) {   /* values */
        blk b = {.x = (uint8_t)x};
        for (int j = 0; j < K; j++) b.p[j] = ok[(i + j) % no];
        add(&b);
      }
      for (int i = 0; i < nbad && no; i += 1 + nbad / 4) {   /* errors */
        blk b = {.x = (uint8_t)x};
        for (int j = 0; j < K; j++) b.p[j] = ok[j % no];
        b.p[(unsigned)i % K] = bad[i];
        add(&b);
      }
    }
  }
  /* specials */
  blk b = {.x = 0xff};
  for (int j = 0; j < K; j++) b.p[j] = (uint8_t)(j % n);
  add(&b);
  memset(&b, 0, sizeof b);
  b.x = 127;
  add(&b);   /* zeros */
  if (bits == 8) {   /* a NaN, or an infinity, among numbers */
    for (int c = 0; c < 256; c++) {
      double d = kit_decode(f, (uint64_t)c);
      if (!isnan(d) && !isinf(d)) continue;
      for (int j = 0; j < K; j++) b.p[j] = (uint8_t)(0x30 + j);
      b.p[7] = (uint8_t)c;
      add(&b);
    }
  }
  for (int i = 0; i < 512; i++) {   /* random */
    uint64_t r = next();
    b.x = (uint8_t)(r % 255);
    for (int j = 0; j < K; j++) b.p[j] = (uint8_t)(next() % (uint64_t)n);
    add(&b);
  }
}

/* the blocks for type f and a two-argument function: every value against
   a rotation of every value, at a few pairs of scales, and random */
static void blocks2(kit_fmt f)
{
  int bits = kit_info(f)->bits, n = 1 << bits;
  static const int XS[] = {0, 1, 60, 120, 126, 127, 128, 134, 200, 253, 254};
  enum { NX = sizeof XS / sizeof *XS };
  nb = 0;
  uint8_t num[256];
  int nn = 0;
  for (int c = 0; c < n; c++) {
    double d = kit_decode(f, (uint64_t)c);
    if (!isnan(d) && !isinf(d)) num[nn++] = (uint8_t)c;
  }
  for (int a = 0; a < NX; a++)
    for (int bx = 0; bx < NX; bx++)
      for (int i = 0; i < nn; i += K)
        for (int rot = 1; rot < nn; rot += nn / 4 + 1) {
          blk b = {.x = (uint8_t)XS[a], .x2 = (uint8_t)XS[bx]};
          for (int j = 0; j < K; j++) { b.p[j] = num[(i + j) % nn]; b.p2[j] = num[(i + j + rot) % nn]; }
          add(&b);
        }
  for (int i = 0; i < 1024; i++) {
    blk b;
    b.x = (uint8_t)(next() % 255);
    b.x2 = (uint8_t)(next() % 255);
    for (int j = 0; j < K; j++) { b.p[j] = (uint8_t)(next() % (uint64_t)n); b.p2[j] = (uint8_t)(next() % (uint64_t)n); }
    add(&b);
  }
  blk b = {.x = 0xff, .x2 = 127};
  add(&b);
}

/* lowp against the kit on every block: one element per comparison, and
   the scale; the control moves one output of a block now and then */
static void run(kit_fmt f, int li, kit_mpfr1 r1, kit_mpfr2 r2, int m, kit_tally *run, kit_tally *ctl)
{
  kit_tally a = {.args = 1}, c = {.args = 1};
  unsigned long long first = ~0ULL;
#pragma omp parallel
  {
    kit_tally ta = {.args = 1}, tc = {.args = 1};
    unsigned long long tf = ~0ULL;
#pragma omp for schedule(dynamic, 16)
    for (size_t k = 0; k < nb; k++) {
      const blk *b = &B[k];
      uint8_t q[K], y;
      uint64_t p64[K], p2[K], w[K];
      int wy;
      if (L[li].f1) L[li].f1(b->p, &b->x, q, &y, 1, m);
      else L[li].f2(b->p, &b->x, b->p2, &b->x2, q, &y, 1, m);
      for (int j = 0; j < K; j++) { p64[j] = b->p[j]; p2[j] = b->p2[j]; }
      if (r1) kit_mx_ref1(f, r1, K, p64, b->x, RND[m], w, &wy);
      else kit_mx_ref2(f, r2, K, p64, b->x, p2, b->x2, RND[m], w, &wy);
      int moved = k == nb / 2 || (k * 0x9e3779b97f4a7c15ULL >> 58) == 5;
      for (int j = 0; j <= K; j++) {   /* j = K: the scale */
        uint64_t got = j < K ? q[j] : y, want = j < K ? w[j] : (uint64_t)wy, cg = got;
        if (moved && j == (wy == 0xff ? K : 3)) cg = j < K ? kit_perturb(f, got) : got ^ 1;
        unsigned long long id = k * (K + 1) + (unsigned)j;
        ta.tested++; tc.tested++;
        if (got != want) { ta.differ++; if (id < tf) { tf = id; ta.first_x = k; ta.first_y = (uint64_t)j; ta.first_got = got; ta.first_want = want; } }
        if (cg != want) tc.differ++;
      }
    }
#pragma omp critical
    {
      a.tested += ta.tested; a.differ += ta.differ; c.tested += tc.tested; c.differ += tc.differ;
      if (tf < first) { first = tf; a.first_x = ta.first_x; a.first_y = ta.first_y; a.first_got = ta.first_got; a.first_want = ta.first_want; }
    }
  }
  a.first_rnd = RND[m];
  if (!run->differ && a.differ) { *run = (kit_tally){run->tested, 0, 1, a.first_rnd, a.first_x, a.first_y, a.first_got, a.first_want}; }
  run->tested += a.tested;
  run->differ += a.differ;
  ctl->tested += c.tested;
  ctl->differ += c.differ;
}

static int report(const char *what, kit_tally t, kit_tally c)
{
  int r = !t.tested || !c.differ ? 2 : t.differ != 0;
  printf("%-30s %12llu tested, %llu differ (control: %llu differ)", what, t.tested, t.differ, c.differ);
  if (t.differ)
    printf("\n    first: %s, block %llu, %s %llu: got 0x%02llx, want 0x%02llx", kit_rnd_name(t.first_rnd),
           (unsigned long long)t.first_x, t.first_y == K ? "the scale" : "element", (unsigned long long)t.first_y,
           (unsigned long long)t.first_got, (unsigned long long)t.first_want);
  if (!c.differ) printf("  VOID: the control did not differ");
  printf("\n");
  return r;
}

int main(void)
{
  if (NL != 5 * PER || strcmp(L[15].name, "exp")) {
    printf("VERDICT: VOID: lowp-mx-list.h is not the same functions for each type, exp 16th\n");
    return 2;
  }
  int r = 0;
  printf("every type, function and mode: lowp against the kit's exact block\n");
  for (int t = 0; t < 5; t++)
    for (int i = 0; i < PER; i++) {
      int li = t * PER + i;
      kit_mpfr1 r1 = L[li].f1 ? ref1(L[li].name) : 0;
      kit_mpfr2 r2 = L[li].f2 ? ref2(L[li].name) : 0;
      if (!r1 && !r2) { printf("%s: no reference\nVERDICT: VOID\n", L[li].name); return 2; }
      kit_tally a = {0}, c = {0};
      if (r1) blocks1(TYPES[t], r1); else blocks2(TYPES[t]);
      for (int m = 0; m < 4; m++) run(TYPES[t], li, r1, r2, m, &a, &c);
      char what[64];
      snprintf(what, sizeof what, "%s %s", TNAME[t], L[li].name);
      r = kit_worst(r, report(what, a, c));
    }
  /* in place */
  int same = 1;
  for (int t = 0; t < 5; t++) {
    int li = t * PER + 15;   /* exp */
    blocks1(TYPES[t], ref1("exp"));
    for (size_t k = 0; k < nb && k < 64; k++) {
      uint8_t q[K], y, p[K], x = B[k].x;
      memcpy(p, B[k].p, K);
      L[li].f1(B[k].p, &B[k].x, q, &y, 1, 0);
      L[li].f1(p, &x, p, &x, 1, 0);
      same &= !memcmp(p, q, K) && x == y;
    }
  }
  printf("%-30s %s\n", "in place (q = p, y = x)", same ? "the same blocks" : "DIFFERENT");
  /* refused modes */
  uint8_t in[K] = {0x38}, out[K], xs = 127, ys = 0xaa;
  memset(out, 0xaa, K);
  int refused = lowp_mx_e4m3_exp(in, &xs, out, &ys, 1, 4) == -1 && lowp_mx_e2m1_exp(in, &xs, out, &ys, 1, -1) == -1 &&
                lowp_mx_e2m1_pow(in, &xs, in, &xs, out, &ys, 1, 7) == -1 && out[0] == 0xaa && ys == 0xaa;
  printf("%-30s %s\n", "modes it doesn't take", refused ? "refused, nothing written" : "NOT REFUSED");
  /* negative control: E2M1 exp judged against exp2 */
  kit_tally a = {0}, c = {0};
  blocks1(KIT_E2M1, mpfr_exp2);
  run(KIT_E2M1, 4 * PER + 15, mpfr_exp2, 0, 0, &a, &c);
  int neg = report("e2m1 exp against exp2", a, c) == 1;
  printf("  %s\n", neg ? "(differs, as it must)" : "NEGATIVE CONTROL FAILED: this must differ");
  if (!same || !refused || !neg) r = kit_worst(r, 2);
  if (!same) r = 1;
  return kit_verdict(r, "every MX block is the exact block, in every mode, and every control differs");
}
