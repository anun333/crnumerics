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
     5  how many results the exceptions table decides (the enclosure could not), and how many of those the
        enclosure's middle would get wrong without it
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
      if ((t->cases % 101) == 0) { t->control_seen++; t->control += l + 1 != r; }
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
                  if ((u.cases % 101) == 0) { u.control_seen++; u.control += l + 1 != r; }
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

/* how much the exceptions table carries: results the enclosure leaves undecided, and how many of them the
   enclosure's middle (the fallback for 32-bit inputs) would round wrongly */
static int sec5(void)
{
  long undecided = 0, middle_wrong = 0;
#pragma omp parallel for reduction(+ : undecided, middle_wrong) schedule(dynamic, 64) collapse(2)
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
          crfix_iv m = e;
          m.kind = CRFIX_IV_EXACT;
          uint64_t lo = e.lo.lo + e.hi.lo, hi = e.lo.hi + e.hi.hi + (lo < e.lo.lo);
          m.lo.lo = lo >> 1 | hi << 63; m.lo.hi = (uint64_t)((int64_t)hi >> 1);
          m.hi = m.lo;
          crfix_round_internal(&m, i / 4, i % 4, &q);
          middle_wrong += q != ref[i];
        }
      }
  printf("the exceptions table (%d entries) decides %ld results here; without it, the enclosure's middle would round %ld of them wrongly\n",
         crfix_nexc_internal(), undecided, middle_wrong);
  return 0;
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
  if (r == 0) printf("VERDICT: IDENTICAL crfix gives the specification's result on every case checked, and every control differs\n");
  else if (r == 1) printf("VERDICT: DIFFERS: a result is not the specification's, or an enclosure misses the exact value\n");
  else printf("VERDICT: VOID: a section tested nothing, or its control did not differ\n");
  return r;
}
