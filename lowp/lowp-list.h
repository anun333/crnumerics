/* lowp's functions, by shape (lowp.h). Define LOWP_F1 (one argument) and
   LOWP_F2 (two) before including. The same list, in the same order, as
   the kit's kit_fns1 and kit_fns2 (kit/fns.c). */
LOWP_F1(acos) LOWP_F1(acosh) LOWP_F1(acospi) LOWP_F1(asin) LOWP_F1(asinh) LOWP_F1(asinpi) LOWP_F1(atan)
LOWP_F1(atanh) LOWP_F1(atanpi) LOWP_F1(cbrt) LOWP_F1(cos) LOWP_F1(cosh) LOWP_F1(cospi) LOWP_F1(erf)
LOWP_F1(erfc) LOWP_F1(exp) LOWP_F1(exp10) LOWP_F1(exp10m1) LOWP_F1(exp2) LOWP_F1(exp2m1) LOWP_F1(expm1)
LOWP_F1(lgamma) LOWP_F1(log) LOWP_F1(log10) LOWP_F1(log10p1) LOWP_F1(log1p) LOWP_F1(log2) LOWP_F1(log2p1)
LOWP_F1(rsqrt) LOWP_F1(sin) LOWP_F1(sinh) LOWP_F1(sinpi) LOWP_F1(sqrt) LOWP_F1(tan) LOWP_F1(tanh)
LOWP_F1(tanpi) LOWP_F1(tgamma)
LOWP_F2(atan2) LOWP_F2(atan2pi) LOWP_F2(hypot) LOWP_F2(pow)
#undef LOWP_F1
#undef LOWP_F2
