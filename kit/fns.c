/* fns.c: the functions the libraries here provide, by name, with their
   correctly rounded MPFR references (kit.h). The one-argument list is
   CORE-MATH's binary16 and bfloat16 list (crmvec-f16-list.h), sincos
   aside; the two-argument one likewise. Where IEEE 754 defines a function
   differently from MPFR, the reference is a wrapper. */
#include <string.h>
#include "kit.h"

/* lgamma: MPFR's takes a sign argument */
int kit_mpfr_lgamma(mpfr_ptr y, mpfr_srcptr x, mpfr_rnd_t r)
{
  int s;
  return mpfr_lgamma(y, &s, x, r);
}

/* rSqrt(-0) is -inf in IEEE 754 (MPFR's rec_sqrt gives +inf) */
int kit_mpfr_rsqrt(mpfr_ptr y, mpfr_srcptr x, mpfr_rnd_t r)
{
  if (mpfr_zero_p(x) && mpfr_signbit(x)) { mpfr_set_inf(y, -1); return 0; }
  return mpfr_rec_sqrt(y, x, r);
}

const kit_fn1 kit_fns1[] = {
  {"acos", mpfr_acos},       {"acosh", mpfr_acosh},     {"acospi", mpfr_acospi},   {"asin", mpfr_asin},
  {"asinh", mpfr_asinh},     {"asinpi", mpfr_asinpi},   {"atan", mpfr_atan},       {"atanh", mpfr_atanh},
  {"atanpi", mpfr_atanpi},   {"cbrt", mpfr_cbrt},       {"cos", mpfr_cos},         {"cosh", mpfr_cosh},
  {"cospi", mpfr_cospi},     {"erf", mpfr_erf},         {"erfc", mpfr_erfc},       {"exp", mpfr_exp},
  {"exp10", mpfr_exp10},     {"exp10m1", mpfr_exp10m1}, {"exp2", mpfr_exp2},       {"exp2m1", mpfr_exp2m1},
  {"expm1", mpfr_expm1},     {"lgamma", kit_mpfr_lgamma}, {"log", mpfr_log},       {"log10", mpfr_log10},
  {"log10p1", mpfr_log10p1}, {"log1p", mpfr_log1p},     {"log2", mpfr_log2},       {"log2p1", mpfr_log2p1},
  {"rsqrt", kit_mpfr_rsqrt}, {"sin", mpfr_sin},         {"sinh", mpfr_sinh},       {"sinpi", mpfr_sinpi},
  {"sqrt", mpfr_sqrt},       {"tan", mpfr_tan},         {"tanh", mpfr_tanh},       {"tanpi", mpfr_tanpi},
  {"tgamma", mpfr_gamma},
};
const int kit_nfns1 = sizeof kit_fns1 / sizeof *kit_fns1;

const kit_fn2 kit_fns2[] = {
  {"atan2", mpfr_atan2}, {"atan2pi", mpfr_atan2pi}, {"hypot", mpfr_hypot}, {"pow", mpfr_pow},
};
const int kit_nfns2 = sizeof kit_fns2 / sizeof *kit_fns2;
