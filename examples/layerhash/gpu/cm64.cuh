/* cm64.cuh: CORE-MATH's binary64 exp, log1p, erfc and rsqrt as CUDA device
   code (d_exp::cr_exp_dev, ...), from the copies devify.sh makes (exp.cuh,
   ...): the C library and builtin calls they make, mapped to CUDA's. */
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <errno.h>
#include <fenv.h>
#undef errno
__device__ int crdev_errno;
#define errno crdev_errno
#define fegetround() FE_TONEAREST
#define feraiseexcept(x) ((void)0)
#define fetestexcept(x) 0
#define feclearexcept(x) ((void)0)
#define __builtin_copysign copysign
#define __builtin_ctzll(x) (__ffsll((long long)(x)) - 1)
#define __builtin_clzll __clzll
#define __builtin_fabs fabs
#define __builtin_fma __fma_rn
#define __builtin_round round
#define __builtin_roundeven rint
#define __builtin_floor floor
#define __builtin_ldexp ldexp
#define __builtin_sqrt sqrt
#define __builtin_nan(s) __longlong_as_double(0x7ff8000000000000LL)
#define __builtin_expect(x, y) (x)
#define rsqrt cr_rsqrt_wrapper_unused
#define cr_exp cr_exp_dev
#define cr_log1p cr_log1p_dev
#define cr_erfc cr_erfc_dev
#define cr_rsqrt cr_rsqrt_dev
/* HIP (rocm/run-rocm.sh): clang compiles these __device__ bodies in both of
   its passes, and both keep the x86 host's macros,
   so CORE-MATH would take its SSE paths (_mm_getcsr, _mm_set_sd) in device
   code; hidden there, it takes its portable ones (nvcc on the GB10's aarch64
   host never saw them) */
#if defined(__HIP__)
#pragma push_macro("__x86_64__")
#pragma push_macro("__i386__")
#pragma push_macro("__SSE2__")
#pragma push_macro("__SSE4_1__")
#pragma push_macro("__AVX__")
#pragma push_macro("__AVX2__")
#pragma push_macro("__FMA__")
#undef __x86_64__
#undef __i386__
#undef __SSE2__
#undef __SSE4_1__
#undef __AVX__
#undef __AVX2__
#undef __FMA__
#endif
namespace d_exp {
#include "exp.cuh"
}
namespace d_log1p {
#include "log1p.cuh"
}
namespace d_erfc {
#include "erfc.cuh"
}
/* rsqrt.c reads the rounding mode from the host CPU's control register
   (x86's MXCSR, aarch64's FPCR) when compiled for those; in the device pass
   the host-architecture macros are hidden, so it takes its fegetround path,
   which is FE_TONEAREST here */
#if defined(__CUDA_ARCH__)
#pragma push_macro("__aarch64__")
#pragma push_macro("__x86_64__")
#undef __aarch64__
#undef __x86_64__
#endif
namespace d_rsqrt {
#include "rsqrt.cuh"
}
#if defined(__CUDA_ARCH__)
#pragma pop_macro("__aarch64__")
#pragma pop_macro("__x86_64__")
#endif
#if defined(__HIP__)
#pragma pop_macro("__x86_64__")
#pragma pop_macro("__i386__")
#pragma pop_macro("__SSE2__")
#pragma pop_macro("__SSE4_1__")
#pragma pop_macro("__AVX__")
#pragma pop_macro("__AVX2__")
#pragma pop_macro("__FMA__")
#endif
#undef cr_exp
#undef cr_log1p
#undef cr_erfc
#undef cr_rsqrt
#undef rsqrt
