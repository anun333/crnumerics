#ifndef CRDIST_REF_H
#define CRDIST_REF_H
#include <mpfr.h>
int crdist_mpfr_erfinv(mpfr_ptr y, mpfr_srcptr x, mpfr_rnd_t rnd);
int crdist_mpfr_erfcinv(mpfr_ptr y, mpfr_srcptr x, mpfr_rnd_t rnd);
int crdist_mpfr_probit(mpfr_ptr y, mpfr_srcptr x, mpfr_rnd_t rnd);
int crdist_mpfr_ndtr(mpfr_ptr y, mpfr_srcptr x, mpfr_rnd_t rnd);
int crdist_mpfr_log_ndtr(mpfr_ptr y, mpfr_srcptr x, mpfr_rnd_t rnd);
int crdist_mpfr_normpdf(mpfr_ptr y, mpfr_srcptr x, mpfr_rnd_t rnd);
int crdist_mpfr_logit(mpfr_ptr y, mpfr_srcptr x, mpfr_rnd_t rnd);
void crdist_mpfr_value(int f, mpfr_ptr t, mpfr_srcptr x);   /* ndtr .. logit (3 .. 6) at t's precision, for the bound check */
/* the equation G(r s) = v whose positive root r is |f(x)| (crdist-ref.c) */
typedef struct { int g, scaled; mpfr_srcptr v; } crdist_eqn;
int crdist_eqn_at(crdist_eqn *e, mpfr_ptr v, int f, float x);
int crdist_cmp(const crdist_eqn *e, mpfr_srcptr m);   /* the sign of r - m, m >= 0 */
extern unsigned long crdist_ref_long;   /* comparisons decided only past 512 bits */
#endif
