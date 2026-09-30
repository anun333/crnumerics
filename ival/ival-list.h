/* ival's functions (ival.h): define IVAL_F(f) (one interval) and/or
   IVAL_F2(f) (two) before including; an undefined one lists nothing.
   CORE-MATH's binary64 functions but lgamma and tgamma (not monotone on the
   negatives: to come); of the two-argument ones, hypot so far (atan2 and
   pow to come). */
#ifndef IVAL_F
#define IVAL_F(f)
#endif
#ifndef IVAL_F2
#define IVAL_F2(f)
#endif
IVAL_F(acos) IVAL_F(acosh) IVAL_F(acospi) IVAL_F(asin) IVAL_F(asinh) IVAL_F(asinpi) IVAL_F(atan) IVAL_F(atanh)
IVAL_F(atanpi) IVAL_F(cbrt) IVAL_F(cos) IVAL_F(cosh) IVAL_F(cospi) IVAL_F(erf) IVAL_F(erfc) IVAL_F(exp)
IVAL_F(exp10) IVAL_F(exp2) IVAL_F(expm1) IVAL_F(log) IVAL_F(log10) IVAL_F(log1p) IVAL_F(log2) IVAL_F(rsqrt)
IVAL_F(sin) IVAL_F(sinh) IVAL_F(sinpi) IVAL_F(sqrt) IVAL_F(tan) IVAL_F(tanh) IVAL_F(tanpi)
IVAL_F2(hypot)
#undef IVAL_F
#undef IVAL_F2
