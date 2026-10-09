/* Correctly-rounded cubic root of binary16 value.

Copyright (c) 2025-2026 Maxence Ponsardin and Paul Zimmermann

This file is part of the CORE-MATH project
(https://core-math.gitlabpages.inria.fr/).

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include <stdint.h>
#include <math.h> // only used during performance tests

// Warning: clang also defines __GNUC__
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic ignored "-Wunknown-pragmas"
#endif

#pragma STDC FENV_ACCESS ON


typedef union {_Float16 f; uint16_t u;} b16u16_u;
typedef union {float f; uint32_t u;} b32u32_u;

// T1[i] approximates (1+i/2^6)^(1/3)
static const float T1[] = {
  0x1p+0, 0x1.015392p+0, 0x1.02a3aep+0, 0x1.03f068p+0, 0x1.0539d6p+0,
  0x1.06800ep+0, 0x1.07c324p+0, 0x1.090328p+0, 0x1.0a403p+0, 0x1.0b7a4cp+0,
  0x1.0cb18cp+0, 0x1.0de602p+0, 0x1.0f17bcp+0, 0x1.1046ccp+0, 0x1.11733ep+0,
  0x1.129d22p+0, 0x1.13c484p+0, 0x1.14e974p+0, 0x1.160bfcp+0, 0x1.172c2ap+0,
  0x1.184a0ap+0, 0x1.1965a8p+0, 0x1.1a7f0ep+0, 0x1.1b9648p+0, 0x1.1cab62p+0,
  0x1.1dbe62p+0, 0x1.1ecf56p+0, 0x1.1fde46p+0, 0x1.20eb3cp+0, 0x1.21f64p+0,
  0x1.22ff5cp+0, 0x1.240698p+0, 0x1.250bfep+0, 0x1.260f94p+0, 0x1.271164p+0,
  0x1.281172p+0, 0x1.290fcap+0, 0x1.2a0c72p+0, 0x1.2b077p+0, 0x1.2c00cap+0,
  0x1.2cf888p+0, 0x1.2deeb2p+0, 0x1.2ee34ep+0, 0x1.2fd66p+0, 0x1.30c7fp+0,
  0x1.31b802p+0, 0x1.32a6ap+0, 0x1.3393cap+0, 0x1.347f8ap+0, 0x1.3569e4p+0,
  0x1.3652dep+0, 0x1.373a7ap+0, 0x1.3820cp+0, 0x1.3905b4p+0, 0x1.39e95cp+0,
  0x1.3acbbcp+0, 0x1.3bacd6p+0, 0x1.3c8cb2p+0, 0x1.3d6b54p+0, 0x1.3e48bep+0,
  0x1.3f24f6p+0, 0x1.4p+0, 0x1.40d9ep+0, 0x1.41b298p+0,
};

_Float16 cr_cbrtf16(_Float16 x){
  b16u16_u t = {.f = x};
  b32u32_u xf = {.f = x};
  int e = xf.u >> 23; // exponent, biased by 127

  // check for special cases
  if (e == 0xff || e == 0x1ff) // NaN or Inf
    /* the expression x+x yields qNaN for x=sNaN and raises invalid,
       it returns qNaN for x=qNaN,
       it returns +Inf for x=+Inf,
       and returns -Inf for x=-Inf */
    return x + x;
  if (xf.f == 0) return x; // +/-0

  // check for exact cases
  static const uint16_t tm[] =
    {0x00, 0xff, 0xff, 0xff,
     0x33, 0x1c, 0x32, 0xff,
     0xff, 0xff, 0xff, 0x00,
     0xff, 0xff, 0xff, 0x10};
  if (tm[(xf.u >> 19) % 16] == (xf.u >> 13) % 64) { // exact cases (not supported by cbrtf)
    int expo = (xf.u & 0x7fffffff) >> 23;
    static const int te[] =
      {0, 0, 0, 0,
       1, 2, 0, 0,
       0, 0, 0, 1,
       0, 0, 0, 0};
    static const uint16_t tf[] =
      {0x3c00, 0, 0, 0,
       0x3d80, 0x3f00, 0x3c80, 0,
       0, 0, 0, 0x3e00,
       0, 0, 0, 0x3d00};
    if (te[(xf.u >> 19) % 16] == (expo + 2) % 3) {
      t.u = (t.u & 0x8000) + ((unsigned)((expo - 127 - te[(xf.u >> 19) % 16]) / 3) << 10) + tf[(xf.u >> 19) % 16];
      return t.f;
    }
  }

  static const float S[] = {1.0, -1.0};
  b32u32_u sgn = {.f = S[xf.u>>31]};
  xf.u &= 0x7fffffffu;
  e = (e & 0xff) + 2;
  // T0[i] approximates 2^(i/3)
  static const double T0[] = {1.0, 0x1.428a3p+0, 0x1.965fecp+0};
  uint32_t k = (e / 3) - 43;
  sgn.u += k << 23;
  int i = e % 3, j = (xf.u >> 17) & 0x3f;
  b32u32_u x1 = {.u = 0x3f800000u | (xf.u & 0xfe0000u)};
  xf.u = 0x3f800000u | (xf.u & 0x1ffffu);
  float x2 = xf.f - 1.0f;
  // now x = 2^(3k) * 2^i * (x1 + x2) with x1 = 1+j/2^6
  double r = x2 / x1.f;
  r = 1.0 + r * (0x1.5553a2p-2 - 0x1.c1a552p-4 * r);
  return (double) sgn.f * T0[i] * (double) T1[j] * r;
}

// dummy function since GNU libc does not provide it
_Float16 cbrtf16 (_Float16 x) {
  return (_Float16) cbrtf ((float) x);
}
