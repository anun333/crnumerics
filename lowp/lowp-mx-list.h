/* lowp's MX functions (lowp.h, lowp/MX.md), for every element type. Define
   LOWP_MX1(t, f) (one argument) and LOWP_MX2(t, f) (two) before
   including. The one-argument list is lowp-list.h's without exp10m1,
   exp2m1, log10p1 and log2p1, which CORE-MATH has no binary64 version of
   (the MX functions round CORE-MATH's binary64 results to odd). */
#define LOWP_MX_ALL(t)                                                                                         \
  LOWP_MX1(t, acos) LOWP_MX1(t, acosh) LOWP_MX1(t, acospi) LOWP_MX1(t, asin) LOWP_MX1(t, asinh)               \
  LOWP_MX1(t, asinpi) LOWP_MX1(t, atan) LOWP_MX1(t, atanh) LOWP_MX1(t, atanpi) LOWP_MX1(t, cbrt)               \
  LOWP_MX1(t, cos) LOWP_MX1(t, cosh) LOWP_MX1(t, cospi) LOWP_MX1(t, erf) LOWP_MX1(t, erfc) LOWP_MX1(t, exp)     \
  LOWP_MX1(t, exp10) LOWP_MX1(t, exp2) LOWP_MX1(t, expm1) LOWP_MX1(t, lgamma) LOWP_MX1(t, log)                 \
  LOWP_MX1(t, log10) LOWP_MX1(t, log1p) LOWP_MX1(t, log2) LOWP_MX1(t, rsqrt) LOWP_MX1(t, sin)                  \
  LOWP_MX1(t, sinh) LOWP_MX1(t, sinpi) LOWP_MX1(t, sqrt) LOWP_MX1(t, tan) LOWP_MX1(t, tanh)                    \
  LOWP_MX1(t, tanpi) LOWP_MX1(t, tgamma)                                                                       \
  LOWP_MX2(t, atan2) LOWP_MX2(t, atan2pi) LOWP_MX2(t, hypot) LOWP_MX2(t, pow)
LOWP_MX_ALL(e5m2) LOWP_MX_ALL(e4m3) LOWP_MX_ALL(e3m2) LOWP_MX_ALL(e2m3) LOWP_MX_ALL(e2m1) LOWP_MX_ALL(int8)
#undef LOWP_MX_ALL
#undef LOWP_MX1
#undef LOWP_MX2
