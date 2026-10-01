/* crnn-ref.h: MPFR references for crnn's one-argument functions
   (crnn-ref.c), round to nearest only, and a count of the inputs that
   needed more than 512 bits */
#ifndef CRNN_REF_H
#define CRNN_REF_H
#include <mpfr.h>
int crnn_mpfr_sigmoid(mpfr_ptr y, mpfr_srcptr x, mpfr_rnd_t rnd);
int crnn_mpfr_silu(mpfr_ptr y, mpfr_srcptr x, mpfr_rnd_t rnd);
int crnn_mpfr_gelu(mpfr_ptr y, mpfr_srcptr x, mpfr_rnd_t rnd);
int crnn_mpfr_softplus(mpfr_ptr y, mpfr_srcptr x, mpfr_rnd_t rnd);
extern unsigned long crnn_ref_long;
#endif
