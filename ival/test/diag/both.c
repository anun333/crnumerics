/* diagnosis, macOS CI only (to be removed): sqrt(0.5) through ival-rev.c's both() pattern, built with the
   library's flags: rounded down and up must differ (0x1.6a09e667f3bccp-1, ...bcdp-1) */
#include <fenv.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
static inline uint64_t rd_fpcr(void) { uint64_t r; __asm__ volatile("mrs %0, fpcr" : "=r"(r)); return r; }
static inline void wr_fpcr(uint64_t r) { __asm__ volatile("msr fpcr, %0" : : "r"(r)); }
static inline void set_round(int m) { uint64_t c = rd_fpcr(), d = (c & ~0xc00000ull) | (uint64_t)(unsigned)m; if (d != c) wr_fpcr(d); }
static double sqrt1(double x) { return sqrt(x); }
__attribute__((noinline)) static void both(double (*g)(double), double x, double *d, double *u)
{
  volatile double v = x;
  set_round(FE_DOWNWARD); *d = g(v);
  set_round(FE_UPWARD); *u = g(v);
  set_round(FE_TONEAREST);
}
int main(void)
{
  double d, u;
  printf("FE_DOWNWARD %#x FE_UPWARD %#x FE_TONEAREST %#x\n", FE_DOWNWARD, FE_UPWARD, FE_TONEAREST);
  both(sqrt1, 0.5, &d, &u);
  printf("both(sqrt): down %a up %a %s\n", d, u, d != u ? "(differ: right)" : "(EQUAL: the modes were not applied)");
  volatile double v = 0.5; double d2, u2;
  fesetround(FE_DOWNWARD); d2 = sqrt(v); fesetround(FE_UPWARD); u2 = sqrt(v); fesetround(FE_TONEAREST);
  printf("fesetround + sqrt: down %a up %a\n", d2, u2);
  volatile double a = 1.0, b = 0x1p-60; double s1, s2;
  set_round(FE_DOWNWARD); s1 = a + b; set_round(FE_UPWARD); s2 = a + b; set_round(FE_TONEAREST);
  printf("msr + add: down %a up %a\n", s1, s2);
  return d == u;
}
