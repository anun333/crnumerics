/* crfix-internal.h: crfix.c's enclosures, for its exceptions generator and its check (not installed; the symbols
   are hidden in the shared library). */
#ifndef CRFIX_INTERNAL_H
#define CRFIX_INTERNAL_H
#include <stdint.h>

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
/* the exceptions table's size */
#if defined(__GNUC__) && !defined(_WIN32)
__attribute__((visibility("hidden")))
#endif
int crfix_nexc_internal(void);
#endif
