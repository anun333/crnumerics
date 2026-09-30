/* crmvec's half-precision (IEEE binary16) and bfloat16 functions, by shape:
   CORE-MATH's 43 in each format but compound (whose MPFR reference needs
   MPFR 4.3). Define H1 (one argument), H2 (two), HSC (sincos) before
   including. Added 2026-09-27. */
H1(acos) H1(acosh) H1(acospi) H1(asin) H1(asinh) H1(asinpi) H1(atan) H1(atanh) H1(atanpi)
H1(cbrt) H1(cos) H1(cosh) H1(cospi) H1(erf) H1(erfc) H1(exp) H1(exp10) H1(exp10m1) H1(exp2)
H1(exp2m1) H1(expm1) H1(lgamma) H1(log) H1(log10) H1(log10p1) H1(log1p) H1(log2) H1(log2p1)
H1(rsqrt) H1(sin) H1(sinh) H1(sinpi) H1(sqrt) H1(tan) H1(tanh) H1(tanpi) H1(tgamma)
H2(atan2) H2(atan2pi) H2(hypot) H2(pow)
HSC(sincos)
#undef H1
#undef H2
#undef HSC
