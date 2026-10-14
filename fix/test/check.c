/* check: crfix (crfix.h), 2026-10-14. Every result is compared with the specification computed by MPFR
   (crfix-ref.c), saturation included, in each of the three containers, every fout from 0 to 31 and every mode.

     1  the nine one-argument functions: every int8 input and every 16th int16 input, fin 0, 4, 7, 8, 15, 16, 20,
        24, 31 (check all: every 16-bit input and every fin, and each approximated function's enclosure checked
        to hold the exact value: the bound, on every input)
     2  div: every pair of int8 inputs at fa, fb in {0, 3, 7} (check all: also 0 .. 15), and random int16 pairs
        and formats
     3  32-bit inputs: random x of the whole int32 range, random fin, each function
     4  crfix_table against crfix_eval; the domain edges (log of 0 and below, sqrt below 0, rsqrt and recip of 0,
        division by 0); invalid parameters give 0, and crfix_table -1
     5  how many results the slow path decides (the 64-bit enclosure could not), each against the specification
     6  the affine formats (crfix_eval_affine, crfix_table_affine): every int8 input of random TFLite-like
        configurations (input scales random binary32 values, zero points, the standard sigmoid and tanh outputs
        and random ones), every mode, and int16 inputs; invalid scales refused
     7  softmax (crfix_softmax): random int8 vectors of 1 to 64 elements and some int16, c from TFLite-like beta and
        scales, the standard output (1/256, -128) and random ones, every mode; all equal (exact ties: 1/2 and 1/4),
        one dominant (the others vanish), and one element (saturating); invalid parameters refused
   The control: each result made one larger on every 101st case must differ there, and for each function the
   specification's rounding down and to nearest must differ somewhere (else a comparison blind to the mode would
   pass). Built with CRFIX_PORTABLE, the
   same check runs the portable 128-bit code. The last line is the verdict. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpfr.h>
#include "crfix.h"
#include "crfix-internal.h"
#include "crfix-ref.h"

static int ALL;
static const char *NAME[] = {"exp", "exp2", "log", "log2", "sigmoid", "tanh", "sqrt", "rsqrt", "recip"};
static uint64_t rng = 0x9e3779b97f4a7c15ULL;
static uint64_t next(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; }

typedef struct { long cases, wrong, control, control_seen, modes; } tally;
static void add(tally *t, const tally *u) { t->cases += u->cases; t->wrong += u->wrong; t->control += u->control; t->control_seen += u->control_seen; t->modes += u->modes; }

/* one function at x 2^-fin, every fout, mode and container */
static void one(int fn, int32_t x, int fin, tally *t)
{
  int64_t ref[128];
  crfix_ref1(fn, x, fin, ref);
  for (int i = 0; i < 128; i++)
    for (int bout = 8; bout <= 32; bout *= 2) {
      int32_t r = crfix_ref_sat(ref[i], bout), l = crfix_eval(fn, x, fin, bout, i / 4, i % 4);
      t->cases++;
      if (l != r) { if (t->wrong++ < 3) printf("  %s x %d fin %d fout %d mode %d bout %d: %d, not %d\n", NAME[fn], (int)x, fin, i / 4, i % 4, bout, (int)l, (int)r); }
      if ((t->cases % 101) == 0) { t->control_seen++; t->control += (int64_t)l + 1 != r; }
      if (i % 4 == CRFIX_DOWN) t->modes += ref[i] != ref[i - CRFIX_DOWN + CRFIX_NEAREST];   /* down against nearest */
    }
}

static int sec1(void)
{
  static const int fins[] = {0, 4, 7, 8, 15, 16, 20, 24, 31};
  long bound_bad = 0, enclosures = 0;
  int r = 0;
  for (int fn = 0; fn < CRFIX_NFN; fn++) {
    tally t = {0};
    int nf = ALL ? 32 : 9;
#pragma omp parallel
    {
      tally u = {0};
      long bb = 0, en = 0;
#pragma omp for schedule(dynamic, 16) collapse(2)
      for (int xi = -32768; xi <= 32767; xi++)
        for (int k = 0; k < nf; k++) {
          int fin = ALL ? k : fins[k];
          if (!ALL && !(xi >= -128 && xi <= 127) && (xi & 15)) continue;
          one(fn, xi, fin, &u);
          if (ALL && fn < 6) {   /* the bound */
            crfix_iv e;
            crfix_enclose_internal(fn, xi, fin, &e);
            if (e.kind == CRFIX_IV_EXACT || e.kind == CRFIX_IV_OPEN) {
              en++;
              bb += !crfix_ref_inside(fn, xi, fin, e.lo.hi, e.lo.lo, e.hi.hi, e.hi.lo, e.kind == CRFIX_IV_EXACT);
            }
          }
        }
#pragma omp critical
      { add(&t, &u); bound_bad += bb; enclosures += en; }
    }
    printf("%-8s %ld results, %ld not the specification's (must be 0); control: %ld of %ld one larger differ (must "
           "be all), %ld where rounding down and to nearest differ (must be > 0)\n", NAME[fn], t.cases, t.wrong, t.control,
           t.control_seen, t.modes);
    if (t.wrong) r = 1;
    else if (t.control != t.control_seen || !t.cases || !t.modes) r = r ? r : 2;
  }
  if (ALL) {
    printf("the bound: %ld enclosures of exp, exp2, log, log2, sigmoid and tanh, %ld missing the exact value (must be 0)\n",
           enclosures, bound_bad);
    if (bound_bad) r = 1;
    if (!enclosures) r = r ? r : 2;
  }
  return r;
}

static int sec2(void)
{
  static const int fs[] = {0, 3, 7};
  tally t = {0};
  int nf = ALL ? 16 : 3;
#pragma omp parallel
  {
    tally u = {0};
#pragma omp for schedule(dynamic, 4) collapse(2)
    for (int a = -128; a <= 127; a++)
      for (int b = -128; b <= 127; b++)
        for (int i = 0; i < nf; i++)
          for (int j = 0; j < nf; j++) {
            int fa = ALL ? i : fs[i], fb = ALL ? j : fs[j];
            for (int fout = 0; fout < 32; fout += ALL ? 1 : 5)
              for (int mode = 0; mode < 4; mode++) {
                int64_t q = crfix_ref_div(a, fa, b, fb, fout, mode);
                for (int bout = 8; bout <= 32; bout *= 2) {
                  int32_t r = crfix_ref_sat(q, bout), l = crfix_div(a, fa, b, fb, bout, fout, mode);
                  u.cases++;
                  if (l != r && u.wrong++ < 3) printf("  div %d/%d by %d/%d fout %d mode %d bout %d: %d, not %d\n", a, fa, b, fb, fout, mode, bout, (int)l, (int)r);
                  if ((u.cases % 101) == 0) { u.control_seen++; u.control += (int64_t)l + 1 != r; }
                }
              }
          }
#pragma omp critical
    add(&t, &u);
  }
  long rand_cases = 0, rand_wrong = 0;
  for (int k = 0; k < (ALL ? 400000 : 40000); k++) {
    int32_t a = (int16_t)next(), b = (int16_t)next();
    int fa = (int)(next() % 32), fb = (int)(next() % 32), fout = (int)(next() % 32), mode = (int)(next() % 4), bout = 8 << (next() % 3);
    rand_cases++;
    if (crfix_div(a, fa, b, fb, bout, fout, mode) != crfix_ref_sat(crfix_ref_div(a, fa, b, fb, fout, mode), bout)) rand_wrong++;
  }
  printf("div: %ld int8 pairs' results, %ld wrong; %ld random int16 pairs and formats, %ld wrong (must be 0); control: %ld of %ld differ\n",
         t.cases, t.wrong, rand_cases, rand_wrong, t.control, t.control_seen);
  return t.wrong || rand_wrong ? 1 : t.control != t.control_seen ? 2 : 0;
}

static int sec3(void)
{
  long cases = 0, wrong = 0;
  int n = ALL ? 20000 : 2000;
#pragma omp parallel for reduction(+ : cases, wrong) schedule(dynamic, 8)
  for (int k = 0; k < CRFIX_NFN * n; k++) {
    uint64_t h = (uint64_t)k * 0x9e3779b97f4a7c15ULL;
    h ^= h >> 29; h *= 0xbf58476d1ce4e5b9ULL; h ^= h >> 32;
    int fn = k % CRFIX_NFN, fin = (int)(h % 32);
    int32_t x = (int32_t)(uint32_t)(h >> 16);
    int64_t ref[128];
    crfix_ref1(fn, x, fin, ref);
    for (int i = 0; i < 128; i++)
      for (int bout = 8; bout <= 32; bout *= 2) {
        cases++;
        if (crfix_eval(fn, x, fin, bout, i / 4, i % 4) != crfix_ref_sat(ref[i], bout)) {
#pragma omp critical
          if (wrong < 3) printf("  32-bit: %s x %d fin %d fout %d mode %d bout %d\n", NAME[fn], (int)x, fin, i / 4, i % 4, bout);
          wrong++;
        }
      }
  }
  printf("32-bit inputs: %ld results at random x, %ld wrong (must be 0)\n", cases, wrong);
  return wrong ? 1 : !cases ? 2 : 0;
}

static int sec4(void)
{
  int bad = 0;
  int32_t tab[256];
  for (int k = 0; k < 200; k++) {
    int fn = (int)(next() % CRFIX_NFN), fin = (int)(next() % 32), fout = (int)(next() % 32), mode = (int)(next() % 4), bout = 8 << (next() % 3);
    if (crfix_table(fn, fin, bout, fout, mode, tab)) bad++;
    for (int x = -128; x < 128; x++) bad += tab[x + 128] != crfix_eval(fn, x, fin, bout, fout, mode);
  }
  /* the domain edges, as crfix.h states them */
  bad += crfix_log(0, 3, 16, 8, 0) != -32768 || crfix_log2(-5, 3, 8, 2, 1) != -128 || crfix_sqrt(-1, 0, 16, 4, 0) != 0 ||
         crfix_rsqrt(0, 2, 16, 4, 0) != 32767 || crfix_rsqrt(-3, 2, 32, 4, 0) != 2147483647 ||
         crfix_recip(0, 5, 8, 1, 2) != 127 || crfix_div(3, 0, 0, 0, 16, 0, 0) != 32767 ||
         crfix_div(-3, 0, 0, 0, 16, 0, 0) != -32768 || crfix_div(0, 0, 0, 0, 16, 0, 0) != 0;
  /* invalid parameters */
  bad += crfix_exp(1, 32, 16, 0, 0) != 0 || crfix_exp(1, 0, 12, 0, 0) != 0 || crfix_exp(1, 0, 16, -1, 0) != 0 ||
         crfix_exp(1, 0, 16, 0, 4) != 0 || crfix_div(1, 0, 1, 40, 16, 0, 0) != 0 || crfix_eval(99, 1, 0, 16, 0, 0) != 0 ||
         crfix_table(99, 0, 16, 0, 0, tab) != -1 || crfix_table(0, 0, 9, 0, 0, tab) != -1;
  bad += strcmp(crfix_version(), CRFIX_VERSION) != 0;
  printf("tables, domain edges, invalid parameters, version: %d wrong (must be 0)\n", bad);
  return bad ? 1 : 0;
}

/* the slow path: results the 64-bit enclosure leaves undecided, decided by crfix-mp.c, each against the
   specification (section 1 compares them too, among all the others) */
static int sec5(void)
{
  long undecided = 0, wrong = 0;
#pragma omp parallel for reduction(+ : undecided, wrong) schedule(dynamic, 64) collapse(2)
  for (int fn = 0; fn < 6; fn++)
    for (int x = -32768; x <= 32767; x++)
      for (int fin = 0; fin < 32; fin += ALL ? 1 : 5) {
        if (!ALL && (x & 7)) continue;
        crfix_iv e;
        crfix_enclose_internal(fn, x, fin, &e);
        if (e.kind != CRFIX_IV_OPEN) continue;
        int64_t ref[128];
        int have = 0;
        for (int i = 0; i < 128; i++) {
          int64_t q;
          if (crfix_round_internal(&e, i / 4, i % 4, &q)) continue;
          undecided++;
          if (!have) { crfix_ref1(fn, x, fin, ref); have = 1; }
          for (int bout = 8; bout <= 32; bout *= 2) wrong += crfix_eval(fn, x, fin, bout, i / 4, i % 4) != crfix_ref_sat(ref[i], bout);
        }
      }
  printf("the slow path decides %ld results here (the 64-bit enclosure straddled a boundary), %ld wrong (must be 0)\n",
         undecided, wrong);
  return wrong ? 1 : 0;
}

/* 6: the affine formats */
static int sec6(void)
{
  long cases = 0, wrong = 0, controls = 0, control_seen = 0;
  int n = ALL ? 400 : 40;
#pragma omp parallel for reduction(+ : cases, wrong, controls, control_seen) schedule(dynamic, 1)
  for (int k = 0; k < n * 6; k++) {
    int fn = k % 6;
    uint64_t h = (uint64_t)k * 0x9e3779b97f4a7c15ULL + 12345;
    h ^= h >> 31; h *= 0xbf58476d1ce4e5b9ULL; h ^= h >> 29;
    crfix_affine in = {0, 0, 8}, out = {0, 0, 8};
    uint32_t sb = (uint32_t)(127 - 12 + (int)(h % 16)) << 23 | (uint32_t)((h >> 8) & 0x7fffff);
    float fs;
    memcpy(&fs, &sb, sizeof fs);
    in.scale = fs;
    in.zero_point = (int)((h >> 32) % 256) - 128;
    int kind = (int)((h >> 40) % 3);
    if (kind == 0 && fn == CRFIX_SIGMOID) { out.scale = 1.0 / 256; out.zero_point = -128; }   /* TFLite's LOGISTIC */
    else if (kind == 0 && fn == CRFIX_TANH) { out.scale = 1.0 / 128; out.zero_point = 0; }    /* and TANH */
    else {
      uint32_t ob = (uint32_t)(127 - 10 + (int)((h >> 44) % 12)) << 23 | (uint32_t)((h >> 13) & 0x7fffff);
      float fo;
      memcpy(&fo, &ob, sizeof fo);
      out.scale = fo;
      out.zero_point = (int)((h >> 52) % 256) - 128;
    }
    out.bits = 8 << ((h >> 60) % 3);
    for (int mode = 0; mode < 4; mode++) {
      int32_t t[256];
      crfix_table_affine(fn, &in, &out, mode, t);
      for (int q = -128; q < 128; q++) {
        int32_t r = crfix_ref_sat(crfix_ref_affine(fn, q, in.scale, in.zero_point, out.scale, out.zero_point, mode), out.bits);
        cases++;
        if (t[q + 128] != r) {
#pragma omp critical
          if (wrong < 3) printf("  affine fn %d q %d in %a %d out %a %d bits %d mode %d: %d, not %d\n", fn, q, in.scale, in.zero_point, out.scale, out.zero_point, out.bits, mode, (int)t[q + 128], (int)r);
          wrong++;
        }
        if (cases % 101 == 0) { control_seen++; controls += (int64_t)t[q + 128] + 1 != r; }
      }
      for (int j = 0; j < 8; j++) {   /* int16 inputs */
        int32_t q = (int16_t)(h >> (j * 8)) * (j + 1);
        int32_t r = crfix_ref_sat(crfix_ref_affine(fn, q, in.scale, in.zero_point, out.scale, out.zero_point, mode), out.bits);
        cases++;
        wrong += crfix_eval_affine(fn, q, &in, &out, mode) != r;
      }
    }
  }
  crfix_affine good = {0.5, 0, 8}, zero = {0.0, 0, 8}, negv = {-0.5, 0, 8}, tiny = {0x1p-1060, 0, 8}, wide = {0.5, 0, 16};
  int32_t t[256];
  int refused = crfix_eval_affine(CRFIX_EXP, 1, &zero, &good, 0) == 0 && crfix_eval_affine(CRFIX_EXP, 1, &good, &negv, 0) == 0 &&
                crfix_eval_affine(CRFIX_EXP, 1, &tiny, &good, 0) == 0 && crfix_eval_affine(CRFIX_SQRT, 1, &good, &good, 0) == 0 &&
                crfix_table_affine(CRFIX_EXP, &wide, &good, 0, t) == -1 && crfix_table_affine(CRFIX_EXP, &good, &good, 4, t) == -1;
  printf("affine: %ld results of random TFLite-like configurations, %ld not the specification's (must be 0); invalid "
         "scales and parameters refused: %s; control: %ld of %ld one larger differ\n", cases, wrong, refused ? "yes" : "NO",
         controls, control_seen);
  return wrong || !refused ? 1 : controls != control_seen || !cases ? 2 : 0;
}

static int sec7(void)
{
  long cases = 0, wrong = 0, vectors = 0;
  int n_vec = ALL ? 3000 : 300;
#pragma omp parallel for reduction(+ : cases, wrong, vectors) schedule(dynamic, 4)
  for (int k = 0; k < n_vec; k++) {
    uint64_t h = (uint64_t)k * 0x9e3779b97f4a7c15ULL + 777;
    h ^= h >> 31; h *= 0xbf58476d1ce4e5b9ULL; h ^= h >> 29;
    size_t n = 1 + h % 64;
    int32_t q[64], y[64];
    int64_t ref[64];
    int kind = (int)((h >> 8) % 6);
    uint64_t g = h;
    for (size_t i = 0; i < n; i++) {
      g = g * 6364136223846793005ULL + 1442695040888963407ULL;
      q[i] = kind == 1 ? 17 : kind == 2 ? (i == 3 ? 127 : -128 + (int32_t)(g >> 59)) : kind == 5 ? (int16_t)(g >> 40) : (int32_t)(g >> 56) - 128;
    }
    if (kind == 1) n = (h >> 12) % 2 ? 2 : 4;   /* equal: 1/2 or 1/4 exactly */
    if (kind == 3) n = 1;
    uint32_t sb = (uint32_t)(127 - 10 + (int)((h >> 20) % 12)) << 23 | (uint32_t)((h >> 24) & 0x7fffff);
    float fs;
    memcpy(&fs, &sb, sizeof fs);
    double c = fs;
    if (kind == 5) c = c / 256;   /* int16 logits: a smaller step */
    crfix_affine out = {1.0 / 256, -128, 8};
    if ((h >> 50) % 3 == 0) {
      uint32_t ob = (uint32_t)(127 - 9 + (int)((h >> 52) % 8)) << 23 | (uint32_t)((h >> 30) & 0x7fffff);
      float fo;
      memcpy(&fo, &ob, sizeof fo);
      out.scale = fo; out.zero_point = (int)((h >> 40) % 256) - 128; out.bits = 8 << ((h >> 60) % 3);
    }
    for (int mode = 0; mode < 4; mode++) {
      crfix_softmax(n, q, c, &out, mode, y);
      crfix_ref_softmax(n, q, c, out.scale, out.zero_point, mode, ref);
      for (size_t i = 0; i < n; i++) {
        int32_t r = crfix_ref_sat(ref[i], out.bits);
        cases++;
        if (y[i] != r) {
#pragma omp critical
          if (wrong < 3) printf("  softmax n %zu kind %d c %a mode %d i %zu: %d, not %d\n", n, kind, c, mode, i, (int)y[i], (int)r);
          wrong++;
        }
      }
    }
    vectors++;
  }
  int32_t q2[2] = {5, 5}, y2[2];
  crfix_affine bad = {0, 0, 8}, good = {1.0 / 256, -128, 8};
  int refused = crfix_softmax(0, q2, 1.0, &good, 0, y2) == -1 && crfix_softmax(2, q2, -1.0, &good, 0, y2) == -1 &&
                crfix_softmax(2, q2, 1.0, &bad, 0, y2) == -1 && crfix_softmax(2, q2, 1.0, &good, 7, y2) == -1;
  int ties = !crfix_softmax(2, q2, 1.0, &good, CRFIX_NEAREST, y2) && y2[0] == 0 && y2[1] == 0;   /* 1/2 -> 128 - 128 */
  printf("softmax: %ld vectors, %ld results, %ld not the specification's (must be 0); 1/2 exactly: %s; invalid refused: %s\n",
         vectors, cases, wrong, ties ? "yes" : "NO", refused ? "yes" : "NO");
  return wrong || !refused || !ties ? 1 : !cases ? 2 : 0;
}

static int worst(int a, int b) { return a == 1 || b == 1 ? 1 : a > b ? a : b; }
int main(int argc, char **argv)
{
  ALL = argc > 1 && !strcmp(argv[1], "all");
#ifdef CRFIX_PORTABLE
  printf("crfix %s, the portable 128-bit code; %s\n", crfix_version(), ALL ? "every 16-bit input" : "a sample");
#else
  printf("crfix %s, native 128-bit arithmetic where the compiler has it; %s\n", crfix_version(), ALL ? "every 16-bit input" : "a sample");
#endif
  int r = 0;
  r = worst(r, sec1());
  r = worst(r, sec2());
  r = worst(r, sec3());
  r = worst(r, sec4());
  r = worst(r, sec5());
  r = worst(r, sec6());
  r = worst(r, sec7());
  if (r == 0) printf("VERDICT: IDENTICAL crfix gives the specification's result on every case checked, and every control differs\n");
  else if (r == 1) printf("VERDICT: DIFFERS: a result is not the specification's, or an enclosure misses the exact value\n");
  else printf("VERDICT: VOID: a section tested nothing, or its control did not differ\n");
  return r;
}
