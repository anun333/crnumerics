/* env-check.c: ival leaves the caller's floating-point state as it was (ival.h). ival saves and restores only the SSE
   unit's MXCSR on x86-64 (ival-eft.h) and runs its CORE-MATH objects' fenv calls on MXCSR (ival-fenv.c), so this
   checks what a caller can see: for every rounding mode, with no flag raised and with all raised, and on x86-64 with
   flush-to-zero and denormals-are-zero on and off, each kind of entry point is called on inputs that raise flags
   inside (overflow, underflow, invalid, division by zero, inexact), and afterwards fegetround and fetestexcept must
   give what they gave before, and on x86-64 MXCSR and the x87 control word and status flags must be unchanged.
   Its controls must fail it: planted bug 42 (the flags raised inside left for the caller), and on x86-64 a build
   without ival-fenv.c's renames, where CORE-MATH's pow raises underflow through glibc's feraiseexcept, which sets it
   in the x87 status word that the MXCSR restore does not reach (the subnormal, inexact powers below). */
#include <fenv.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#if defined(__x86_64__)
#include <immintrin.h>
#endif
#include "ival.h"

typedef struct { int round, flags; unsigned csr, cw, sw; } state;
static state read_state(void)
{
  state s = { fegetround(), fetestexcept(FE_ALL_EXCEPT), 0, 0, 0 };
#if defined(__x86_64__)
  unsigned short cw, sw;
  __asm__ volatile("fnstcw %0" : "=m"(cw));
  __asm__ volatile("fnstsw %0" : "=m"(sw));
  s.csr = _mm_getcsr(); s.cw = cw; s.sw = sw & 0x3fu;
#endif
  return s;
}
enum { N = 6 };
static const double LO[N] = { 700, -1e308, -INFINITY, 0, 1e-300, -3 }, HI[N] = { 1000, 1e308, INFINITY, 0, 2e-300, 1e300 };
static double zl[N], zh[N], wl[N], wh[N];
static const double PX[N] = { 0.5, 0.5, 0.75, 2, 0.5, 0.25 }, PY[N] = { 1060.3, 1061.1, 2500.5, -1070.6, 1073.9, 530.25 },
                    PZ[N] = { 1070.7, 1062.3, 2600.5, -1060.2, 1074.0, 535.25 };   /* subnormal, inexact powers */
static const char *NAME[] = { "add", "mul", "div", "exp", "cos", "tan", "pow", "tgamma", "acc_exp", "powrev1", "sinrev",
                              "sqrt", "mid", "text", "pow underflow" };
enum { NF = sizeof NAME / sizeof NAME[0] };
static void call(int f)
{
  switch (f) {
    case 0: ival_add(LO, HI, LO, HI, zl, zh, N); break;
    case 1: ival_mul(LO, HI, LO, HI, zl, zh, N); break;
    case 2: ival_div(LO, HI, HI, LO, zl, zh, N); break;   /* [0, 0] divisors and reversed ones too */
    case 3: ival_exp(LO, HI, zl, zh, N); break;
    case 4: ival_cos(LO, HI, zl, zh, N); break;
    case 5: ival_tan(LO, HI, zl, zh, N); break;
    case 6: ival_pow(LO, HI, LO, HI, zl, zh, N); break;
    case 7: ival_tgamma(LO, HI, zl, zh, N); break;
    case 8: ival_acc_exp(LO, HI, zl, zh, N); break;
    case 9: ival_powrev1(LO, HI, LO, HI, LO, HI, zl, zh, N); break;
    case 10: ival_sinrev(LO, HI, LO, HI, zl, zh, N); break;
    case 11: ival_sqrt(LO, HI, zl, zh, N); break;
    case 12: ival_mid(LO, HI, wl, N); (void)wh; break;
    case 13: { static const char *t[1] = { "[0.1, 1e400]" }; unsigned char st[1]; ival_text(t, zl, zh, st, 1); break; }
    default: ival_pow(PX, PX, PY, PZ, zl, zh, N); break;   /* CORE-MATH's pow raises underflow with feraiseexcept */
  }
}
int main(void)
{
  static const int MODES[4] = { FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO };
  long checked = 0, bad = 0;
  char first[300] = "";
  for (int m = 0; m < 4; m++)
    for (int fl = 0; fl < 2; fl++)
      for (int fz = 0; fz < 2; fz++)
        for (int f = 0; f < NF; f++) {
#if !defined(__x86_64__)
          if (fz) continue;
#endif
          fesetround(MODES[m]);
          feclearexcept(FE_ALL_EXCEPT);
          if (fl) feraiseexcept(FE_ALL_EXCEPT);
#if defined(__x86_64__)
          _mm_setcsr(fz ? _mm_getcsr() | 0x8040u : _mm_getcsr() & ~0x8040u);
#endif
          state a = read_state();
          call(f);
          state b = read_state();
          checked++;
          if (memcmp(&a, &b, sizeof a)) {
            if (!bad++) snprintf(first, sizeof first, " (first: %s, mode %d, flags %s, flush %d: round %#x -> %#x, flags %#x -> %#x, "
                                 "mxcsr %#x -> %#x, x87 cw %#x -> %#x, sw %#x -> %#x)", NAME[f], m, fl ? "all" : "none", fz,
                                 a.round, b.round, a.flags, b.flags, a.csr, b.csr, a.cw, b.cw, a.sw, b.sw);
          }
        }
  fesetround(FE_TONEAREST); feclearexcept(FE_ALL_EXCEPT);
#if defined(__x86_64__)
  _mm_setcsr(_mm_getcsr() & ~0x8040u);
#endif
  if (!bad) printf("VERDICT: IDENTICAL (%ld calls: each left the rounding mode, the flags%s as the caller had them)\n", checked,
#if defined(__x86_64__)
                   ", MXCSR (flush modes included) and the x87 control and status words"
#else
                   ""
#endif
                   );
  else printf("VERDICT: DIFFERS (%ld of %ld)%s\n", bad, checked, first);
  return bad != 0;
}
