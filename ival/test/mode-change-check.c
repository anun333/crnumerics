/* mode-change-check.c: ival-scalar.h in a caller built as most are, without -frounding-math (make check builds
   this file without crnumerics' FP flags). Such a compiler takes floating-point operations to be pure, and the
   header finds the rounding mode by arithmetic (ival1__mode), so nothing in C stops it from reusing one call's mode
   check, or moving it out of a loop, across a change of mode: the operation after the change would then run the
   other mode's formula. The barrier in ival1__mode is volatile asm so that it cannot; this checks that every
   inline result equals the library's when the mode changes between calls in one function, and in a loop whose
   body changes it. Planted bug 47 (a check that always answers "upward", as one made upward and reused would) must
   fail it. 2026-10-10. */
#include <fenv.h>
#include <stdio.h>
#include "ival.h"
#include "ival-scalar.h"

static long checked, bad;
static char first[256];
static void cmp(const char *what, double l, double h, double rl, double rh)
{
  checked++;
  if (l != rl || h != rh) {
    if (!bad++) snprintf(first, sizeof first, " (first: %s gives [%a, %a], the library [%a, %a])", what, l, h, rl, rh);
  }
}

/* a sum and a product in one function, the mode changed between calls: none may reuse an earlier call's check */
__attribute__((noinline)) static void twice(double a, double b, double c, double d, double *r)
{
  ival1_add(a, a, b, b, &r[0], &r[1]);
  fesetround(FE_DOWNWARD);
  ival1_add(a, a, b, b, &r[2], &r[3]);
  fesetround(FE_UPWARD);
  ival1_mul(c, c, d, d, &r[4], &r[5]);
  fesetround(FE_DOWNWARD);
  ival1_mul(c, c, d, d, &r[6], &r[7]);
  fesetround(FE_TONEAREST);
}

/* a loop whose body changes the mode: the check, the same computation each time round, must not leave the loop */
__attribute__((noinline)) static void loop(const double *a, const double *b, double *l, double *h, int n)
{
  for (int k = 0; k < n; k++) {
    fesetround(k & 1 ? FE_DOWNWARD : FE_UPWARD);
    ival1_mul(a[k], a[k], b[k], b[k], &l[k], &h[k]);
    ival1_scale(a[k], b[k], b[k] + 1, &l[n + k], &h[n + k]);
  }
  fesetround(FE_TONEAREST);
}

int main(void)
{
  volatile double va = 1.0, vb = 0x1p-60, vc = 1.0 + 0x1p-30, vd = 3.0 + 0x1p-29;   /* a + b and c * d inexact */
  double a = va, b = vb, c = vc, d = vd, r[8], rl, rh, pl, ph;
  fesetround(FE_UPWARD);   /* the first call's check reads upward */
  twice(a, b, c, d, r);
  ival_add(&a, &a, &b, &b, &rl, &rh, 1);
  ival_mul(&c, &c, &d, &d, &pl, &ph, 1);
  cmp("an add, upward", r[0], r[1], rl, rh);
  cmp("an add after a change to downward", r[2], r[3], rl, rh);
  cmp("a product after a change back to upward", r[4], r[5], pl, ph);
  cmp("a product after a change to downward", r[6], r[7], pl, ph);

  enum { N = 64 };
  double al[N], bl[N], l[2 * N], h[2 * N], ml[N], mh[N], sl[N], sh[N], bu[N];
  for (int k = 0; k < N; k++) { al[k] = (k % 3 == 2 ? -1.0 : 1.0) + k * 0x1p-30; bl[k] = 3.0 + k * 0x1p-29; bu[k] = bl[k] + 1; }
  fesetround(FE_UPWARD);   /* so that a check moved out of the loop reads upward */
  loop(al, bl, l, h, N);
  ival_mul(al, al, bl, bl, ml, mh, N);
  ival_mul(al, al, bl, bu, sl, sh, N);
  for (int k = 0; k < N; k++) {
    cmp(k & 1 ? "a product in the loop, downward" : "a product in the loop, upward", l[k], h[k], ml[k], mh[k]);
    cmp(k & 1 ? "a scale in the loop, downward" : "a scale in the loop, upward", l[N + k], h[N + k], sl[k], sh[k]);
  }
  printf("ival-scalar.h with the mode changed between calls, built without -frounding-math: %ld of %ld results the library's%s\n",
         checked - bad, checked, first);
  if (bad) printf("VERDICT: DIFFERS (%ld)\n", bad);
  else printf("VERDICT: IDENTICAL\n");
  return bad != 0;
}
