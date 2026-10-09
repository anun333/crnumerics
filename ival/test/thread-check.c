/* thread-check.c: ival from several threads at once. Eight threads, released together by a barrier, make the
   process's first calls at the same moment (so the accurate mode's one-time crmvec load and the arithmetic's CPU check
   are raced), then call the arithmetic, the functions in both modes and a reverse operation, 20 times each, on the
   same inputs. Each thread keeps its first round's results and checks the later rounds against them; after the
   threads are joined, one thread computes the same calls alone, and every thread's results must equal those bit for
   bit. Each thread runs with its own rounding mode set (the four in turn), which ival must leave as it was. Built
   with -fsanitize=thread it also reports any data race (make ival-thread-tsan). With IVAL_CRMVEC set the accurate
   mode runs through crmvec. */
#include <fenv.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ival.h"

enum { N = 4096, T = 8, REPS = 20, NOUT = 6 };
typedef double block[NOUT][2][N];
static double lo[N], hi[N], lo2[N], hi2[N];
static block first[T];
static pthread_barrier_t start;
static int drift[T], moded[T];

static void work(block out)
{
  ival_mul(lo, hi, lo2, hi2, out[0][0], out[0][1], N);
  ival_div(lo, hi, lo2, hi2, out[1][0], out[1][1], N);
  ival_exp(lo, hi, out[2][0], out[2][1], N);
  ival_acc_exp(lo, hi, out[3][0], out[3][1], N);
  ival_acc_sin(lo, hi, out[4][0], out[4][1], N);
  ival_sqrrev(lo2, hi2, lo, hi, out[5][0], out[5][1], N);
}
static void *thread(void *arg)
{
  int t = (int)(intptr_t)arg;
  static const int modes[4] = { FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO };
  fesetround(modes[t % 4]);
  double (*out)[2][N] = malloc(sizeof(block));
  pthread_barrier_wait(&start);   /* the process's first calls, all at once */
  work(first[t]);
  for (int r = 1; r < REPS; r++) {
    work(out);
    if (memcmp(out, first[t], sizeof(block))) drift[t]++;
  }
  moded[t] = fegetround() != modes[t % 4];
  free(out);
  return NULL;
}
int main(void)
{
  uint64_t s = 0x9e3779b97f4a7c15ULL;
  for (int k = 0; k < N; k++) {
    s ^= s << 13; s ^= s >> 7; s ^= s << 17;
    double x = ((double)(s >> 11) * 0x1p-53 - 0.5) * 20;
    lo[k] = x; hi[k] = x + fabs(x) * 1e-3 * (double)((s >> 3) & 1);
    lo2[k] = 0.5 + (double)(s & 0xffff) * 0x1p-16; hi2[k] = lo2[k] * 1.25;
  }
  pthread_t th[T];
  pthread_barrier_init(&start, NULL, T);
  for (int t = 0; t < T; t++) pthread_create(&th[t], NULL, thread, (void *)(intptr_t)t);
  for (int t = 0; t < T; t++) pthread_join(th[t], NULL);
  static block ref;   /* one thread, alone, after */
  work(ref);
  int differ = 0, drifts = 0, modes = 0;
  for (int t = 0; t < T; t++) { differ += memcmp(first[t], ref, sizeof ref) != 0; drifts += drift[t]; modes += moded[t]; }
  const char *lib = getenv("IVAL_CRMVEC");
  if (!differ && !drifts && !modes)
    printf("VERDICT: IDENTICAL (%d threads, first calls at once, %d rounds of %d operations each, every result equal "
           "to one thread's alone, every thread's rounding mode kept; accurate mode %s)\n", T, REPS, NOUT,
           lib && *lib ? "through crmvec" : "without crmvec");
  else
    printf("VERDICT: DIFFERS (%d threads' first results differ from one thread's, %d later rounds drifted, %d threads' "
           "rounding modes changed)\n", differ, drifts, modes);
  return differ || drifts || modes;
}
