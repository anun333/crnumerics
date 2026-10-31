/* roundeven-check.c: kit/roundeven.h's kit_roundeven against rint in round-to-nearest (which rounds ties to even),
   on every halfway and near-halfway value from 2^-2 to 2^53, every power of two, signed zeros, infinities and a
   NaN, and 2^24 random bit patterns. Control: round-half-away (C's round), which must differ on the ties. */
#include <fenv.h>
#include <math.h>
#include <stdio.h>
#include "roundeven.h"
#include "kit.h"

static uint64_t rng = 0x9e3779b97f4a7c15ULL;
static uint64_t next(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; }

static int same(double a, double b)
{
  if (isnan(a) && isnan(b)) return 1;
  uint64_t x, y; memcpy(&x, &a, 8); memcpy(&y, &b, 8); return x == y;
}

int main(void)
{
  fesetround(FE_TONEAREST);
  long n = 0, bad = 0, ctl = 0;
  double special[] = {0.0, -0.0, INFINITY, -INFINITY, NAN, 0x1p-1074, 0.5, -0.5, 1.5, -1.5, 2.5, -2.5};
  for (unsigned i = 0; i < sizeof special / sizeof *special; i++, n++) bad += !same(kit_roundeven(special[i]), rint(special[i]));
  for (int e = -2; e <= 53; e++)
    for (int s = 0; s < 2; s++)
      for (long k = 1; k <= 4096; k++) {
        double h = ldexp((double)k + 0.5, e - 12), v[3] = {h, nextafter(h, 0), nextafter(h, INFINITY)};
        for (int j = 0; j < 3; j++, n++) {
          double x = s ? -v[j] : v[j];
          bad += !same(kit_roundeven(x), rint(x));
          ctl += !same(round(x), rint(x));
        }
      }
  for (int e = -1074; e <= 1023; e++, n++) bad += !same(kit_roundeven(ldexp(1, e)), rint(ldexp(1, e)));
  for (long i = 0; i < 1L << 24; i++, n++) {
    uint64_t u = next(); double x; memcpy(&x, &u, 8);
    bad += !same(kit_roundeven(x), rint(x));
  }
  printf("roundeven: %ld values, %ld differ from rint in round-to-nearest (control, round-half-away: %ld differ)\n", n, bad, ctl);
  return kit_verdict(bad ? 1 : ctl ? 0 : 2, "kit_roundeven is rint in round-to-nearest on every value tried, and the control differs");
}
