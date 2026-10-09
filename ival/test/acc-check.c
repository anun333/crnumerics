/* acc-check.c: ival's accurate mode (ival_acc_f, ival.c) against its tight mode (ival_f, checked against MPFR by
   check.c): every result must hold the tight one and lie within one ulp of it at each end (IEEE 1788's "accurate"),
   empty exactly when that is, for all 32 functions on special and random intervals of every magnitude and width.

   Run with IVAL_CRMVEC naming crmvec's libmvec.so.1 (make ival-acc-check CRMVEC=...), and it must load: a run in
   which no result differs from the tight one is VOID, since the vector path cannot have run (its widened ends differ
   from the tight ones almost everywhere). Without IVAL_CRMVEC the accurate mode is the tight one, and the check says
   so and requires equality. In place, and the caller's rounding mode kept, as for the tight mode. */
#include <fenv.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ival.h"

typedef void (*f1)(const double *, const double *, double *, double *, size_t);
static const struct { const char *name; f1 tight, acc; } L[] = {
#define IVAL_F(f) { #f, ival_##f, ival_acc_##f },
#include "ival-list.h"
};
enum { NL = sizeof L / sizeof L[0] };

static uint64_t rs = 0x6a09e667f3bcc909ULL;
static uint64_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }
static double anyd(void) { for (;;) { uint64_t u = rnd(); double d; memcpy(&d, &u, 8); if (isfinite(d)) return d; } }
static int same(double a, double b) { return (a != a && b != b) || a == b; }
static const double SP[] = { -INFINITY, -DBL_MAX, -1e300, -1e10, -100, -10, -3.5, -2, -1.5, -1, -0.75, -0.5, -1e-5, -1e-300,
                             -DBL_TRUE_MIN, 0, DBL_TRUE_MIN, 1e-300, 1e-5, 0.5, 0.75, 1, 1.5, 2, 3.5, 10, 100, 1e10, 1e300,
                             DBL_MAX, INFINITY };
enum { NSP = sizeof SP / sizeof SP[0] };

int main(void)
{
  const char *lib = getenv("IVAL_CRMVEC");
  int with = lib && *lib;
  enum { NR = 1 << 15 };
  static double lo[NSP * NSP + NR + 4], hi[NSP * NSP + NR + 4], tl[NSP * NSP + NR + 4], th[NSP * NSP + NR + 4],
      al[NSP * NSP + NR + 4], ah[NSP * NSP + NR + 4];
  int n = 0;
  for (int i = 0; i < NSP; i++) for (int j = i; j < NSP; j++) { lo[n] = SP[i]; hi[n] = SP[j]; n++; }
  for (int k = 0; k < NR; k++) {
    double a, b;
    switch (k % 4) {
      case 0: a = anyd(); b = anyd(); break;                                     /* any */
      case 1: a = ((double)(rnd() >> 11) * 0x1p-53 - 0.5) * 40; b = a + (double)(rnd() >> 11) * 0x1p-53 * 3; break;   /* to width 3 */
      case 2: a = ((double)(rnd() >> 11) * 0x1p-53 - 0.5) * 40; b = a * (1 + (double)(rnd() >> 11) * 0x1p-63); break;  /* narrow */
      default: a = ((double)(rnd() >> 11) * 0x1p-53 - 0.5) * 4; b = a; break;   /* points */
    }
    if (b < a) { double t = a; a = b; b = t; }
    lo[n] = a; hi[n] = b; n++;
  }
  lo[n] = NAN; hi[n] = NAN; n++; lo[n] = 2; hi[n] = 1; n++;
  long checked = 0, bad = 0, differ = 0;
  char first[400] = "";
  for (int f = 0; f < NL; f++) {
    L[f].tight(lo, hi, tl, th, n);
    L[f].acc(lo, hi, al, ah, n);
    long fd = 0;
    for (int k = 0; k < n; k++) {
      checked++;
      int et = tl[k] != tl[k], ea = al[k] != al[k];
      int ok = et == ea;
      if (ok && !et) {
        ok = al[k] <= tl[k] && ah[k] >= th[k];   /* holds the tight interval */
        ok = ok && al[k] >= nextafter(tl[k], -INFINITY) && ah[k] <= nextafter(th[k], INFINITY);   /* one ulp at most */
        if (!with) ok = ok && same(al[k], tl[k]) && same(ah[k], th[k]);
      }
      fd += !same(al[k], tl[k]) || !same(ah[k], th[k]);
      if (!ok && !bad++)
        snprintf(first, sizeof first, " (first: %s [%a, %a]: accurate [%a, %a], tight [%a, %a])", L[f].name, lo[k], hi[k], al[k], ah[k], tl[k], th[k]);
    }
    differ += fd;
    printf("%-8s %7d intervals, %6ld results looser than the tight ones\n", L[f].name, n, fd);
  }
  /* in place, and the rounding mode left alone */
  double a[4] = { -1, 0.5, 2, 3 }, b[4] = { 1, 1.5, 3, 4 }, l[4], h[4];
  ival_acc_exp(a, b, l, h, 4);
  fesetround(FE_UPWARD);
  ival_acc_exp(a, b, a, b, 4);
  int inplace = !memcmp(a, l, sizeof a) && !memcmp(b, h, sizeof b) && fegetround() == FE_UPWARD;
  fesetround(FE_TONEAREST);
  if (!inplace) { bad++; snprintf(first, sizeof first, " (in place or the rounding mode: DIFFERENT)"); }
  if (with && !differ) { printf("VERDICT: VOID (IVAL_CRMVEC=%s, but no result differs from the tight mode: the vector path did not run)\n", lib); return 2; }
  if (!bad)
    printf("VERDICT: IDENTICAL (%ld results accurate: each holds the tight one, one ulp wider at most; %s)\n", checked,
           with ? "through crmvec" : "NO CRMVEC: the accurate mode is the tight mode, and equal to it");
  else
    printf("VERDICT: DIFFERS (%ld of %ld)%s\n", bad, checked, first);
  return bad != 0;
}
