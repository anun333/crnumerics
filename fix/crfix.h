/* crfix.h: correctly rounded fixed-point functions, in integer arithmetic only (2026-10-13; ROADMAP.md, item 3).

   The specification. A fixed-point number is an integer x and a count of fractional bits F: its value is x 2^-F.
   A function's result is a value in an output format, an integer container of B bits (8, 16 or 32) with its own
   fractional bits Fo: the exact result times 2^Fo, rounded once to an integer in the mode asked for, then saturated
   to the container, [-2^(B-1), 2^(B-1) - 1]. F and Fo are 0 to 31.

     mode   CRFIX_NEAREST (ties to even), CRFIX_UP (toward +infinity), CRFIX_DOWN (toward -infinity), CRFIX_ZERO
            (crsum.h's names and numbers)

   Functions of one argument, x with fin fractional bits:
     crfix_exp      e^v               crfix_log      ln v           crfix_sigmoid   1 / (1 + e^-v)
     crfix_exp2     2^v               crfix_log2     log2 v         crfix_tanh      tanh v
     crfix_sqrt     sqrt v            crfix_rsqrt    1 / sqrt v     crfix_recip     1 / v
   and crfix_div(a, fa, b, fb, ...) = (a 2^-fa) / (b 2^-fb).

   Where the exact result is not a number, each function gives the value at the edge of its domain, saturated:
   log and log2 of v <= 0, the most negative value (log 0 = -infinity); sqrt of v < 0, 0; rsqrt of v <= 0 and recip
   of 0, the largest; a / 0, the largest for a > 0 and the most negative for a < 0, and 0 / 0 = 0. Fixed point has
   no NaN, so a kernel gets a number it can saturate through.

   Invalid parameters (F or Fo outside 0 to 31, B not 8, 16 or 32, an unknown mode) give 0.

   What is proven: for every input x of 16 bits or fewer (-32768 to 32767, so every int8 and int16), every fin and
   fout from 0 to 31, every container and every mode, each function's result is the specification's, checked
   against MPFR on every case (fix/test/check.c, the all mode). sqrt, rsqrt, recip and div are computed exactly in
   integers, and are checked exhaustively on 16-bit inputs (div on every int8 pair, and on samples of int16 pairs).
   exp, exp2, log, log2, sigmoid and tanh decide their rounding from a 64-bit enclosure of the exact value; the rare
   input whose enclosure straddles a rounding boundary (83 of all the 16-bit cases) goes to a slow path of interval
   arithmetic in 320-bit fixed point, correct by construction (crfix-mp.c). So 32-bit inputs are correctly rounded
   too (checked on samples: the fast path's enclosure bounds were proven on every 16-bit input, and the slow path
   against MPFR on samples of every kind).

   How: exp and exp2 reduce to 2^f on [0, 1), a 64-entry table and a degree-7 polynomial in 64-bit fixed point;
   log and log2 to ln(1 + s) for |s| < 2^-7 by a 64-entry table of reciprocals, and its series to s^9; sigmoid and
   tanh from exp(-|v|) and exp(-2|v|) and a division whose rounding direction keeps the enclosure. Integer
   arithmetic only: uint64_t, and unsigned __int128 where the compiler has one (portable code otherwise, which the
   check also runs). No floating-point operation anywhere, so it runs where there is no FPU. */
#ifndef CRFIX_H
#define CRFIX_H
#include <stddef.h>
#include <stdint.h>

#define CRFIX_VERSION "0.7.0"
enum { CRFIX_NEAREST = 0, CRFIX_UP = 1, CRFIX_DOWN = 2, CRFIX_ZERO = 3 };
/* function numbers, for crfix_eval and crfix_table */
enum { CRFIX_EXP = 0, CRFIX_EXP2, CRFIX_LOG, CRFIX_LOG2, CRFIX_SIGMOID, CRFIX_TANH, CRFIX_SQRT, CRFIX_RSQRT, CRFIX_RECIP,
       CRFIX_NFN };
#ifdef __cplusplus
extern "C" {
#endif
int32_t crfix_exp(int32_t x, int fin, int bout, int fout, int mode);
int32_t crfix_exp2(int32_t x, int fin, int bout, int fout, int mode);
int32_t crfix_log(int32_t x, int fin, int bout, int fout, int mode);
int32_t crfix_log2(int32_t x, int fin, int bout, int fout, int mode);
int32_t crfix_sigmoid(int32_t x, int fin, int bout, int fout, int mode);
int32_t crfix_tanh(int32_t x, int fin, int bout, int fout, int mode);
int32_t crfix_sqrt(int32_t x, int fin, int bout, int fout, int mode);
int32_t crfix_rsqrt(int32_t x, int fin, int bout, int fout, int mode);
int32_t crfix_recip(int32_t x, int fin, int bout, int fout, int mode);
int32_t crfix_div(int32_t a, int fa, int32_t b, int fb, int bout, int fout, int mode);
/* one of the above by number */
int32_t crfix_eval(int fn, int32_t x, int fin, int bout, int fout, int mode);
/* an int8 kernel's lookup table: table[x + 128] = the function at x, for x = -128 .. 127; 0, or -1 for invalid
   parameters (nothing written) */
int crfix_table(int fn, int fin, int bout, int fout, int mode, int32_t table[256]);

/* affine formats (2026-10-13), TFLite's quantization: a value is scale (q - zero_point), scale a positive normal
   binary64 (a float32 scale converts exactly), read from its bits with no floating-point operation. The result
   is round(f(scale_in (q - zero_point_in)) / scale_out) + zero_point_out in mode, saturated to out->bits (8, 16
   or 32, signed). exp, exp2, log, log2, sigmoid and tanh (log and log2 of a value <= 0: the most negative); invalid
   parameters give 0. The fixed-point functions' 64-bit enclosures decide nearly every call, in about 0.1 us; the
   rest go to the slow path, interval arithmetic in 320-bit fixed point (crfix-mp.c), correct by construction for
   any scales from about 2^-190 to 2^190. crfix_table_affine: in->bits must be 8; table[q + 128] for q = -128 .. 127;
   0, or -1. */
typedef struct { double scale; int32_t zero_point; int bits; } crfix_affine;
int32_t crfix_eval_affine(int fn, int32_t q, const crfix_affine *in, const crfix_affine *out, int mode);
int crfix_table_affine(int fn, const crfix_affine *in, const crfix_affine *out, int mode, int32_t table[256]);
/* softmax (2026-10-13): y[i] = round(p_i / out->scale) + out->zero_point, saturated to out->bits, with
   p_i = e^(c (q[i] - max q)) / sum_j e^(c (q[j] - max q)), the exact softmax of c q; c > 0 a binary64, beta times the
   logits' scale as TFLite has it (the zero point cancels). 64-bit enclosures decide nearly every vector, in about
   0.1 us an element; a vector with an undecided element goes to the slow path, correct by construction for any n
   and c: each distinct difference's exponential once (an int8 vector has at most 256), then exact sums of interval
   bounds; 0, or -1 for invalid parameters or want of memory. */
int crfix_softmax(size_t n, const int32_t *q, double c, const crfix_affine *out, int mode, int32_t *y);
const char *crfix_version(void);
#ifdef __cplusplus
}
#endif
#endif
