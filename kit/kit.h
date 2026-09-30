/* kit.h: the checking kit shared by the reproducible-numerics libraries
   (numerics/README.md). Three things every library here needs, done once:

   - Formats: binary64, binary32, binary16, bfloat16, and the OCP 8-bit
     formats E5M2 and E4M3, as bit encodings with exact decoding.
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

typedef enum { KIT_B64, KIT_B32, KIT_B16, KIT_BF16, KIT_E5M2, KIT_E4M3, KIT_NFMT } kit_fmt;

typedef struct {
  const char *name;
  int bits, ebits, mbits, bias;   /* encoding width, exponent and trailing significand bits, bias */
  int has_inf;                    /* 0 for E4M3: its top exponent holds finite numbers and one NaN */
} kit_fmtinfo;

const kit_fmtinfo *kit_info(kit_fmt f);

/* E4M3 has no infinity. An overflow gives NaN (the default), or with this
   set the largest finite value of the right sign (OCP's saturating mode). */
extern int kit_e4m3_saturate;

int kit_isnan(kit_fmt f, uint64_t bits);
/* the value of an encoding, exactly: every format here embeds in binary64 */
double kit_decode(kit_fmt f, uint64_t bits);
/* the encoding of a value the format represents exactly (NaN: the format's
   canonical quiet NaN); aborts on a value it can't represent, a bug */
uint64_t kit_encode(kit_fmt f, double v);
/* the largest finite value */
double kit_max(kit_fmt f);

/* x rounded to the format in mode rnd: correctly rounded, with the format's
   subnormals, and overflow to infinity (E4M3: NaN or saturation) */
uint64_t kit_round(kit_fmt f, mpfr_srcptr x, mpfr_rnd_t rnd);

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
   result moved by one ulp on about one input in 64. A mode without a C
   equivalent (MPFR_RNDA) tests nothing, so its run is void. */
kit_tally kit_exhaust1(kit_fmt f, kit_cand1 cand, kit_mpfr1 ref, mpfr_rnd_t rnd, kit_tally *control);
kit_tally kit_sample1(kit_fmt f, kit_cand1 cand, kit_mpfr1 ref, mpfr_rnd_t rnd, unsigned long long n,
                      uint64_t seed, kit_tally *control);
/* two-argument runs: n pairs drawn as above (every pair, for 8-bit
   formats with n = 0) */
kit_tally kit_sample2(kit_fmt f, kit_cand2 cand, kit_mpfr2 ref, mpfr_rnd_t rnd, unsigned long long n,
                      uint64_t seed, kit_tally *control);
/* the one-ulp move the control applies (NaN stays NaN, infinity becomes
   the largest finite value) */
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

const char *kit_rnd_name(mpfr_rnd_t rnd);
int kit_fenv_of(mpfr_rnd_t rnd);   /* FE_TONEAREST ... for a C candidate */

#endif
