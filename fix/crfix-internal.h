/* crfix-internal.h: crfix.c's enclosures and slow path, for its check (not installed; the symbols
   are hidden in the shared library). */
#ifndef CRFIX_INTERNAL_H
#define CRFIX_INTERNAL_H
#include <stddef.h>
#include <stdint.h>
#include "crfix.h"

/* a 128-bit word pair; as a signed number, two's complement */
typedef struct { uint64_t hi, lo; } crfix_u128;
/* an enclosure of a function's exact value v, in signed Q64.64 (units of 2^-64):
     CRFIX_IV_EXACT  v = lo = hi exactly
     CRFIX_IV_OPEN   lo < v < hi
     CRFIX_IV_BIG    v >= 2^32 (every container saturates)
     CRFIX_IV_DOMAIN not a number: the function's domain-edge value instead (crfix.h) */
enum { CRFIX_IV_EXACT, CRFIX_IV_OPEN, CRFIX_IV_BIG, CRFIX_IV_DOMAIN };
typedef struct { int kind; crfix_u128 lo, hi; } crfix_iv;

/* the enclosure of a function of the six approximated ones (CRFIX_EXP .. CRFIX_TANH) at x 2^-fin */
#if defined(__GNUC__) && !defined(_WIN32)
__attribute__((visibility("hidden")))
#endif
void crfix_enclose_internal(int fn, int32_t x, int fin, crfix_iv *e);
/* its rounding to fout fractional bits in mode: 1 and the result (before saturation) in *q, or 0 when the enclosure
   straddles a boundary */
#if defined(__GNUC__) && !defined(_WIN32)
__attribute__((visibility("hidden")))
#endif
int crfix_round_internal(const crfix_iv *e, int fout, int mode, int64_t *q);
/* the fast path of an affine format (crfix.h's crfix_eval_affine): 1 and the result before saturation, or 0 when
   undecided (the slow path's then); its scales valid */
#if defined(__GNUC__) && !defined(_WIN32)
__attribute__((visibility("hidden")))
#endif
int crfix_affine_fast_internal(int fn, int32_t q, const crfix_affine *in, const crfix_affine *out, int mode, int64_t *r);
/* softmax's fast path: 1 and every result before saturation, or 0 when one is undecided (the slow path's then) */
#if defined(__GNUC__) && !defined(_WIN32)
__attribute__((visibility("hidden")))
#endif
int crfix_softmax_fast_internal(size_t n, const int32_t *q, double c, const crfix_affine *out, int mode, int64_t *r);
/* the slow path (crfix-mp.c): f((-1)^neg N 2^e) / (so 2^eo) + z, N = nhi 2^64 + nlo, rounded in mode, before
   saturation (INT64_MAX and INT64_MIN past every container); f one of the six approximated functions, v in its domain */
#if defined(__GNUC__) && !defined(_WIN32)
__attribute__((visibility("hidden")))
#endif
int64_t crfix_mp_internal(int fn, int neg, uint64_t nhi, uint64_t nlo, int e, uint64_t so, int eo, int64_t z, int mode);
/* softmax (crfix-mp.c): out[i] = round(p_i / (so 2^eo)) + z before saturation, c = cm 2^ce; 0, or -1 */
#if defined(__GNUC__) && !defined(_WIN32)
__attribute__((visibility("hidden")))
#endif
int crfix_softmax_internal(size_t n, const int32_t *q, uint64_t cm, int ce, uint64_t so, int eo, int64_t z, int mode,
                           int64_t *out);
#endif
