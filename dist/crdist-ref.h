#ifndef CRDIST_REF_H
#define CRDIST_REF_H
#include <mpfr.h>
int crdist_mpfr_erfinv(mpfr_ptr y, mpfr_srcptr x, mpfr_rnd_t rnd);
int crdist_mpfr_erfcinv(mpfr_ptr y, mpfr_srcptr x, mpfr_rnd_t rnd);
int crdist_mpfr_probit(mpfr_ptr y, mpfr_srcptr x, mpfr_rnd_t rnd);
/* the equation G(r s) = v whose positive root r is |f(x)| (crdist-ref.c) */
typedef struct { int g, scaled; mpfr_srcptr v; } crdist_eqn;
int crdist_eqn_at(crdist_eqn *e, mpfr_ptr v, int f, float x);
int crdist_cmp(const crdist_eqn *e, mpfr_srcptr m);   /* the sign of r - m, m >= 0 */
extern unsigned long crdist_ref_long;   /* comparisons decided only past 512 bits */
#endif
