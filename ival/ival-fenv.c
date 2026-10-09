/* ival-fenv.c: on x86-64, the floating-point environment calls of ival's own CORE-MATH objects, on MXCSR alone. The
   build renames their references (objcopy --redefine-syms=ival/ival-fenv.syms) to these, hidden in the library.
   ival sets the rounding mode in MXCSR only (ival-eft.h), and glibc's fegetround reads the x87 control word instead,
   so CORE-MATH's cos, tan and pow, which ask fegetround, would round as if to nearest in ival's directed passes; and
   glibc's feraiseexcept raises some flags in the x87 status word, which ival's MXCSR restore would leave behind for
   the caller. Here both read and write MXCSR, whose flag bits are the FE_ values and whose rounding-control bits are
   the FE_ rounding values shifted left 3. feholdexcept keeps MXCSR in the first bytes of the fenv_t, which only
   feupdateenv reads back. Nothing here outside x86-64. */
#if defined(__x86_64__)
#include <fenv.h>
#include <string.h>
#include <immintrin.h>
#define HIDDEN __attribute__((visibility("hidden")))
#define FLAGS(e) ((unsigned)(e) & FE_ALL_EXCEPT)
HIDDEN int ival_cm_fegetround(void)
{
#if IVAL_PLANT_ARITH == 41   /* 41: glibc's, which reads the x87 unit's mode */
  return fegetround();
#endif
  return (int)((_mm_getcsr() >> 3) & 0xc00u);
}
HIDDEN int ival_cm_feraiseexcept(int e) { _mm_setcsr(_mm_getcsr() | FLAGS(e)); return 0; }
HIDDEN int ival_cm_fetestexcept(int e) { return (int)(_mm_getcsr() & FLAGS(e)); }
HIDDEN int ival_cm_feclearexcept(int e) { _mm_setcsr(_mm_getcsr() & ~FLAGS(e)); return 0; }
HIDDEN int ival_cm_fegetexceptflag(fexcept_t *f, int e) { *f = (fexcept_t)(_mm_getcsr() & FLAGS(e)); return 0; }
HIDDEN int ival_cm_fesetexceptflag(const fexcept_t *f, int e)
{
  _mm_setcsr((_mm_getcsr() & ~FLAGS(e)) | (*f & FLAGS(e)));
  return 0;
}
HIDDEN int ival_cm_feholdexcept(fenv_t *env)   /* save, clear the flags, mask every exception */
{
  unsigned c = _mm_getcsr();
  memcpy(env, &c, sizeof c);
  _mm_setcsr((c & ~FLAGS(FE_ALL_EXCEPT)) | 0x1f80u);
  return 0;
}
HIDDEN int ival_cm_feupdateenv(const fenv_t *env)   /* restore, then raise what was raised meanwhile */
{
  unsigned c, raised = _mm_getcsr() & FLAGS(FE_ALL_EXCEPT);
  memcpy(&c, env, sizeof c);
  _mm_setcsr(c | raised);
  return 0;
}
#else
typedef int ival_fenv_unused;   /* an empty translation unit is not ISO C */
#endif
