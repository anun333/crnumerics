/* crfix-ref.h: crfix.h's specification computed with MPFR, for the exceptions generator and the check
   (2026-10-14). A result is given before saturation: INT64_MAX and INT64_MIN stand for values past every
   container. */
#ifndef CRFIX_REF_H
#define CRFIX_REF_H
#include <stdint.h>

/* a one-argument function (CRFIX_EXP .. CRFIX_RECIP) at x 2^-fin: out[fout * 4 + mode] for fout 0 .. 31 and every
   mode */
void crfix_ref1(int fn, int32_t x, int fin, int64_t out[128]);
/* (a 2^-fa) / (b 2^-fb) */
int64_t crfix_ref_div(int32_t a, int fa, int32_t b, int fb, int fout, int mode);
/* round(f(sin (q - zin)) / sout) + zout, before saturation */
int64_t crfix_ref_affine(int fn, int32_t q, double sin, int32_t zin, double sout, int32_t zout, int mode);
/* saturated to a container of bout bits */
int32_t crfix_ref_sat(int64_t q, int bout);
/* 1 if the exact value of the function at x 2^-fin lies strictly inside (lo, hi) 2^-64 (signed Q64.64 words, as
   crfix-internal.h's enclosures), or equals lo when exact is set; 0 if not */
int crfix_ref_inside(int fn, int32_t x, int fin, uint64_t lo_hi, uint64_t lo_lo, uint64_t hi_hi, uint64_t hi_lo,
                     int exact);
#endif
