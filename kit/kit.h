/* kit.h: the checking kit shared by the reproducible-numerics libraries
   (kit/README.md). Three things every library here needs, done once:

   - Formats: binary64, binary32, binary16, bfloat16, the OCP 8-bit
     formats E5M2 and E4M3, and the OCP MX element formats E2M3, E3M2
     (6 bits) and E2M1 (4 bits), as bit encodings with exact decoding.
   - A reference: the correctly rounded result of a function in a format
     and rounding mode, through MPFR (correctly rounded at any precision),
     with the format's exponent range, subnormals and overflow applied.
   - Runs and verdicts: every input of an 8- or 16-bit format, or a sample
     of a wider one, compared bit for bit with the reference. Every run
     also runs its control, the same check on a candidate made wrong by one
     ulp on some inputs, which must differ; a check whose control passes
     is blind, and says so.

   The verdict convention, as in crmvec: IDENTICAL (exit 0), DIFFERS
   (exit 1), VOID (exit 2: nothing was tested, the control did not
   differ, or the check could not run). A verdict line is the last line a
   check prints. */
#ifndef KIT_H
#define KIT_H
#include <stdint.h>
#include <mpfr.h>

/* KIT_INT8 is OCP MX's INT8 element (MX v1.0, 5.3.4): two's complement
   with an implicit 2^-6, so k/64 for k in -128 ... 127. Its magnitudes are
   those of a float format with 1 exponent bit, 6 trailing bits and bias 1
   (below), which is how it rounds; only the encoding differs. -2 (0x80)
   decodes, but rounding never gives it: the spec leaves it unused for
   symmetry, and overflow clamps to +-127/64. */
typedef enum { KIT_B64, KIT_B32, KIT_B16, KIT_BF16, KIT_E5M2, KIT_E4M3, KIT_E2M3, KIT_E3M2, KIT_E2M1,
               KIT_INT8, KIT_NFMT } kit_fmt;

typedef struct {
  const char *name;
  int bits, ebits, mbits, bias;   /* encoding width, exponent and trailing significand bits, bias */
  int has_inf;                    /* 0 for E4M3: its top exponent holds finite numbers and one NaN */
  int has_nan;                    /* 0 for the MX element formats: every encoding is a number */
} kit_fmtinfo;

/* what a format without NaN gives for a NaN result: no encoding (the MX
   formats leave NaN to the block's shared scale) */
#define KIT_NONE (~0ULL)

const kit_fmtinfo *kit_info(kit_fmt f);

/* E4M3 has no infinity. An overflow gives NaN (the default), or with this
   set the largest finite value of the right sign (OCP's saturating mode).
   The MX element formats, with neither infinity nor NaN, always saturate. */
extern int kit_e4m3_saturate;
/* E5M2's saturating mode (OFP8 1.0, Table 3): an infinity, exact or from an
   overflow away from zero, becomes the largest finite value of its sign
   instead (the default, 0, keeps the infinity) */
extern int kit_e5m2_saturate;

int kit_isnan(kit_fmt f, uint64_t bits);
/* the value of an encoding, exactly: every format here embeds in binary64 */
double kit_decode(kit_fmt f, uint64_t bits);
/* the encoding of a value the format represents exactly (NaN: the format's
   canonical quiet NaN, or KIT_NONE without one); aborts on a value it can't
   represent, a bug */
uint64_t kit_encode(kit_fmt f, double v);
/* the largest finite value */
double kit_max(kit_fmt f);

/* x rounded to the format in mode rnd: correctly rounded, with the format's
   subnormals, and overflow to infinity (E4M3: NaN or saturation; the MX
   element formats: saturation) */
uint64_t kit_round(kit_fmt f, mpfr_srcptr x, mpfr_rnd_t rnd);
/* the same for y already rounded in mode rnd to the format's precision
   (mbits + 1), in MPFR's default range, with ternary value t (y changes) */
uint64_t kit_round_t(kit_fmt f, mpfr_ptr y, int t, mpfr_rnd_t rnd);


/* the correctly rounded value of an MPFR function at an input of the
   format. One IEEE rule replaces MPFR's conventions: an input that is a
   signaling NaN gives a NaN (MPFR has no signaling NaNs, and gives
   pow(x, 0) = 1 for every NaN x). A function whose IEEE result differs
   from MPFR's elsewhere, such as rSqrt(-0) = -inf, takes a wrapper.
   These and kit_round work in MPFR's default exponent range, which they
   leave as they found it; they don't depend on the C rounding mode, and
   are safe to call from several threads when MPFR is built thread-safe. */
typedef int (*kit_mpfr1)(mpfr_ptr, mpfr_srcptr, mpfr_rnd_t);
typedef int (*kit_mpfr2)(mpfr_ptr, mpfr_srcptr, mpfr_srcptr, mpfr_rnd_t);
uint64_t kit_ref1(kit_fmt f, kit_mpfr1 fn, uint64_t x, mpfr_rnd_t rnd);
uint64_t kit_ref2(kit_fmt f, kit_mpfr2 fn, uint64_t x, uint64_t y, mpfr_rnd_t rnd);

/* MX blocks (lowp/MX.md): k elements p of type f (E5M2, E4M3, E3M2, E2M3,
   E2M1) under the E8M0 scale byte x. The correctly rounded function on
   the block: elements q and scale byte *y, from the exact results; a NaN
   block has scale 0xff and elements 0. The two-argument form takes a
   second block (p2, x2) of the same type. */
int kit_mx_emax(kit_fmt f);
void kit_mx_ref1(kit_fmt f, kit_mpfr1 fn, int k, const uint64_t *p, int x, mpfr_rnd_t rnd, uint64_t *q, int *y);
void kit_mx_ref2(kit_fmt f, kit_mpfr2 fn, int k, const uint64_t *p, int x, const uint64_t *p2, int x2, mpfr_rnd_t rnd,
                 uint64_t *q, int *y);

/* the function under test, on encodings. It runs in the C rounding mode
   that matches rnd (the runner sets it on every thread), from several
   threads at once. */
typedef uint64_t (*kit_cand1)(uint64_t x);
typedef uint64_t (*kit_cand2)(uint64_t x, uint64_t y);

typedef struct {
  unsigned long long tested, differ;
  int args;                                           /* 1 or 2 */
  mpfr_rnd_t first_rnd;                               /* the first difference found: */
  uint64_t first_x, first_y, first_got, first_want;   /* mode, input(s), result, reference */
} kit_tally;
/* sum over runs (say, the four rounding modes): counts add, the first
   difference is the earliest run's */
void kit_tally_add(kit_tally *sum, kit_tally t);

/* One-argument runs: every input (formats of 32 bits or fewer; mind
   the time MPFR takes on 2^32), or the format's edge values and then n
   inputs drawn with every exponent equally likely. The control is
   computed in the same pass, against the same reference: the candidate's
   result moved by one ulp on about one input in 64, and on at least one. A
   mode without a C
   equivalent (MPFR_RNDA) tests nothing, so its run is void. */
kit_tally kit_exhaust1(kit_fmt f, kit_cand1 cand, kit_mpfr1 ref, mpfr_rnd_t rnd, kit_tally *control);
kit_tally kit_sample1(kit_fmt f, kit_cand1 cand, kit_mpfr1 ref, mpfr_rnd_t rnd, unsigned long long n,
                      uint64_t seed, kit_tally *control);
/* two-argument runs: n pairs drawn as above (every pair, for formats of 8
   bits or fewer with n = 0) */
kit_tally kit_sample2(kit_fmt f, kit_cand2 cand, kit_mpfr2 ref, mpfr_rnd_t rnd, unsigned long long n,
                      uint64_t seed, kit_tally *control);
/* the one-ulp move the control applies (a NaN becomes zero, since it has
   no neighbour and a control that leaves it alone can't differ; infinity
   becomes the largest finite value) */
uint64_t kit_perturb(kit_fmt f, uint64_t bits);

/* NaNs compare equal to NaNs (the default), or bit for bit with this set */
extern int kit_nan_bits;
int kit_same(kit_fmt f, uint64_t a, uint64_t b);

/* one line for a run and its control; returns 0 identical, 1 differs,
   2 void (nothing tested, or the control did not differ) */
int kit_report(const char *what, kit_fmt f, kit_tally run, kit_tally control);
/* two results combined: differs over void over identical */
int kit_worst(int a, int b);
/* the verdict line over every report, given their combined result;
   returns it, for the exit status */
int kit_verdict(int result, const char *identical_text);

/* the functions the libraries here provide (CORE-MATH's list for binary16
   and bfloat16, sincos aside), in alphabetical order, with their MPFR
   references; the IEEE wrappers are exported for other uses */
typedef struct { const char *name; kit_mpfr1 ref; } kit_fn1;
typedef struct { const char *name; kit_mpfr2 ref; } kit_fn2;
extern const kit_fn1 kit_fns1[];
extern const int kit_nfns1;
extern const kit_fn2 kit_fns2[];
extern const int kit_nfns2;
int kit_mpfr_lgamma(mpfr_ptr y, mpfr_srcptr x, mpfr_rnd_t r);
int kit_mpfr_rsqrt(mpfr_ptr y, mpfr_srcptr x, mpfr_rnd_t r);

const char *kit_rnd_name(mpfr_rnd_t rnd);
int kit_fenv_of(mpfr_rnd_t rnd);   /* FE_TONEAREST ... for a C candidate */

/* reductions (sum.c): the correctly rounded sum of x[0 .. n-1], or with y
   the sum of the exact products x[i] y[i], rounded to f (KIT_B64 or
   KIT_B32) in rnd, by MPFR's mpfr_sum; and a random permutation of 0 .. n-1 */
double kit_sum_ref(kit_fmt f, const double *x, const double *y, size_t n, mpfr_rnd_t rnd);
void kit_shuffle(uint64_t seed, size_t *idx, size_t n);

#endif
