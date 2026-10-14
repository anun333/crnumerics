/* crlin-internal.h: what crlin.c and band.c share (2026-10-13); not installed. */
#ifndef CRLIN_INTERNAL_H
#define CRLIN_INTERNAL_H
#include <fenv.h>
#include <math.h>

#define ENV_ENTER fenv_t env_; fegetenv(&env_); fesetenv(FE_DFL_ENV)
#define ENV_LEAVE fesetenv(&env_)
/* threads (2026-10-13): rows of the O(n^2) work, and the columns of the inverse, go to OpenMP threads. Each row is
   computed the same way whichever thread takes it (the exact sums rounded once, the plain loops in the same order), so
   the bits don't depend on how many threads there are. Each thread sets the default floating-point environment itself
   and puts its own back: a pool's worker keeps the mode it was created with, which ENV_ENTER on the calling thread
   doesn't reach, and the exact sums and the bounds' upward steps need round-to-nearest without flushing. */
#define PAR_ENTER fenv_t tenv_; fegetenv(&tenv_); fesetenv(FE_DFL_ENV)
#define PAR_LEAVE fesetenv(&tenv_)
#define PAR_MIN 64   /* below this many rows, one thread */

static inline double up(double t) { return nextafter(t, INFINITY); }   /* above any value t (rounded to nearest) stands for */
#endif
