/* run.c: the kit's runs, controls and verdicts (kit.h). */
#include "kit.h"
#include <fenv.h>
#include <limits.h>
#include <stdio.h>

const char *kit_rnd_name(mpfr_rnd_t rnd)
{
  switch (rnd) {
  case MPFR_RNDN: return "nearest";
  case MPFR_RNDU: return "up";
  case MPFR_RNDD: return "down";
  case MPFR_RNDZ: return "zero";
  default: return "away";
  }
}

int kit_fenv_of(mpfr_rnd_t rnd)
{
  switch (rnd) {
  case MPFR_RNDN: return FE_TONEAREST;
  case MPFR_RNDU: return FE_UPWARD;
  case MPFR_RNDD: return FE_DOWNWARD;
  case MPFR_RNDZ: return FE_TOWARDZERO;
  default: return -1;   /* C has no rounding away from zero */
  }
}

static uint64_t mix(uint64_t z)   /* splitmix64's finalizer */
{
  z += 0x9e3779b97f4a7c15ULL;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

/* the format's edge values, both signs: zero, the smallest and largest
   subnormal, the smallest normal, the largest finite value, 1 and its
   neighbours, 1/2, 2, the infinities and NaNs (quiet and signaling) */
static int edges(kit_fmt f, uint64_t *e)
{
  const kit_fmtinfo *i = kit_info(f);
  uint64_t s = 1ULL << (i->bits - 1), m1 = 1ULL << i->mbits, one = (uint64_t)i->bias << i->mbits;
  uint64_t v[] = {0, 1, m1 - 1, m1, kit_encode(f, kit_max(f)), one, one - 1, one + 1, one - m1, one + m1};
  int n = 0;
  for (unsigned k = 0; k < sizeof v / sizeof *v; k++) { e[n++] = v[k]; e[n++] = v[k] | s; }
  uint64_t top = ((1ULL << i->ebits) - 1) << i->mbits;
  if (i->has_inf) {
    uint64_t q = top | 1ULL << (i->mbits - 1);
    e[n++] = top; e[n++] = top | s; e[n++] = q; e[n++] = q | s; e[n++] = top | 1;
  } else {
    e[n++] = top | (m1 - 1); e[n++] = top | (m1 - 1) | s;
  }
  return n;
}

/* a random encoding with every exponent equally likely */
static uint64_t draw(kit_fmt f, uint64_t seed, uint64_t k)
{
  const kit_fmtinfo *i = kit_info(f);
  uint64_t a = mix(seed ^ mix(2 * k)), b = mix(seed ^ mix(2 * k + 1));
  uint64_t e = (a >> 1) % (1ULL << i->ebits), m = b & ((1ULL << i->mbits) - 1);
  return (a & 1) << (i->bits - 1) | e << i->mbits | m;
}

/* the inputs of a run, by index */
typedef struct {
  kit_fmt f;
  int args, all8;          /* all8: every pair of an 8-bit format */
  unsigned long long exhaust, n;
  uint64_t seed;
  int ne;
  uint64_t e[32];
} inputs;

static unsigned long long count(const inputs *in)
{
  if (in->exhaust) return in->exhaust;
  if (in->all8) return 1ULL << 16;
  return (in->args == 1 ? in->ne : in->ne * in->ne) + in->n;
}

static void input(const inputs *in, unsigned long long k, uint64_t *x, uint64_t *y)
{
  unsigned long long ne = in->args == 1 ? in->ne : in->ne * in->ne;
  if (in->exhaust) *x = k;
  else if (in->all8) { *x = k >> 8; *y = k & 255; }
  else if (k < ne && in->args == 1) *x = in->e[k];
  else if (k < ne) { *x = in->e[k / in->ne]; *y = in->e[k % in->ne]; }
  else {
    *x = draw(in->f, in->seed, 2 * (k - ne));
    if (in->args == 2) *y = draw(in->f, in->seed, 2 * (k - ne) + 1);
  }
}

/* a tally and the index of its first difference */
typedef struct { kit_tally t; unsigned long long first; } acc;

static void note(acc *a, unsigned long long k, int differ, uint64_t x, uint64_t y, uint64_t got, uint64_t want)
{
  a->t.tested++;
  if (!differ) return;
  a->t.differ++;
  if (k < a->first) { a->first = k; a->t.first_x = x; a->t.first_y = y; a->t.first_got = got; a->t.first_want = want; }
}

static void merge(acc *into, const acc *from)
{
  unsigned long long tested = into->t.tested + from->t.tested, differ = into->t.differ + from->t.differ;
  if (from->first < into->first) *into = *from;
  into->t.tested = tested;
  into->t.differ = differ;
}

void kit_tally_add(kit_tally *sum, kit_tally t)
{
  unsigned long long tested = sum->tested + t.tested, differ = sum->differ + t.differ;
  if (!sum->differ && t.differ) *sum = t;
  if (!sum->args) sum->args = t.args;
  sum->tested = tested;
  sum->differ = differ;
}

/* the candidate on every input, against the reference and its control;
   the control's candidate is this one moved by an ulp on about one input
   in 64 */
static kit_tally run(const inputs *in, kit_cand1 c1, kit_cand2 c2, kit_mpfr1 r1, kit_mpfr2 r2,
                     mpfr_rnd_t rnd, kit_tally *control)
{
  acc all = {{0, 0, in->args, rnd, 0, 0, 0, 0}, ULLONG_MAX}, ctl = all;
  int mode = kit_fenv_of(rnd);
  unsigned long long n = mode < 0 ? 0 : count(in);
  kit_fmt f = in->f;
  /* several threads only when MPFR keeps its state per thread */
#pragma omp parallel if (mpfr_buildopt_tls_p())
  {
    acc a = all, c = ctl;
    int old = fegetround();
    if (n) fesetround(mode);
#pragma omp for schedule(dynamic, 1024)
    for (unsigned long long k = 0; k < n; k++) {
      uint64_t x, y = 0;
      input(in, k, &x, &y);
      uint64_t got = in->args == 1 ? c1(x) : c2(x, y);
      uint64_t want = in->args == 1 ? kit_ref1(f, r1, x, rnd) : kit_ref2(f, r2, x, y, rnd);
      note(&a, k, !kit_same(f, got, want), x, y, got, want);
      uint64_t moved = mix(k ^ 0x6a09e667f3bcc908ULL) % 64 ? got : kit_perturb(f, got);
      note(&c, k, !kit_same(f, moved, want), x, y, moved, want);
    }
    fesetround(old);
    mpfr_free_cache2(MPFR_FREE_LOCAL_CACHE);
#pragma omp critical
    {
      merge(&all, &a);
      merge(&ctl, &c);
    }
  }
  *control = ctl.t;
  return all.t;
}

kit_tally kit_exhaust1(kit_fmt f, kit_cand1 cand, kit_mpfr1 ref, mpfr_rnd_t rnd, kit_tally *control)
{
  inputs in = {.f = f, .args = 1, .exhaust = kit_info(f)->bits <= 32 ? 1ULL << kit_info(f)->bits : 0};
  if (!in.exhaust) {   /* binary64: nothing tested, so the run is void */
    kit_tally none = {.args = 1, .first_rnd = rnd};
    *control = none;
    return none;
  }
  return run(&in, cand, 0, ref, 0, rnd, control);
}

kit_tally kit_sample1(kit_fmt f, kit_cand1 cand, kit_mpfr1 ref, mpfr_rnd_t rnd, unsigned long long n,
                      uint64_t seed, kit_tally *control)
{
  inputs in = {.f = f, .args = 1, .n = n, .seed = seed};
  in.ne = edges(f, in.e);
  return run(&in, cand, 0, ref, 0, rnd, control);
}

kit_tally kit_sample2(kit_fmt f, kit_cand2 cand, kit_mpfr2 ref, mpfr_rnd_t rnd, unsigned long long n,
                      uint64_t seed, kit_tally *control)
{
  inputs in = {.f = f, .args = 2, .n = n, .seed = seed, .all8 = n == 0 && kit_info(f)->bits == 8};
  in.ne = edges(f, in.e);
  return run(&in, 0, cand, 0, ref, rnd, control);
}

static void show(char *s, size_t len, kit_fmt f, uint64_t b)
{
  int digits = kit_info(f)->bits / 4;
  snprintf(s, len, "0x%0*llx (%.17g)", digits, (unsigned long long)b, kit_decode(f, b));
}

int kit_report(const char *what, kit_fmt f, kit_tally run, kit_tally control)
{
  int r = !run.tested || !control.differ ? 2 : run.differ != 0;
  printf("%-30s %12llu tested, %llu differ (control: %llu differ)", what, run.tested, run.differ, control.differ);
  if (run.differ) {
    char x[64], y[64], got[64], want[64];
    show(x, sizeof x, f, run.first_x);
    show(y, sizeof y, f, run.first_y);
    show(got, sizeof got, f, run.first_got);
    show(want, sizeof want, f, run.first_want);
    printf("\n    first: %s at %s%s%s: got %s, want %s", kit_rnd_name(run.first_rnd), x,
           run.args == 2 ? ", " : "", run.args == 2 ? y : "", got, want);
  }
  if (!run.tested) printf("  VOID: nothing tested");
  else if (!control.differ) printf("  VOID: the control did not differ, so this check is blind");
  printf("\n");
  return r;
}

int kit_worst(int a, int b) { return a == 1 || b == 1 ? 1 : a == 2 || b == 2 ? 2 : 0; }

int kit_verdict(int result, const char *identical_text)
{
  if (result == 0) printf("VERDICT: IDENTICAL %s\n", identical_text);
  else if (result == 1) printf("VERDICT: DIFFERS: a run above differs from its reference\n");
  else printf("VERDICT: VOID: a run above tested nothing, or its control did not differ\n");
  return result;
}
