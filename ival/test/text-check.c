/* text-check.c: ival_text and ival_nums (ival-text.c) against MPFR.

   Random literals of every form: inf-sup intervals of decimal numbers (1 to 40 digits, exponents to +-340, so
   subnormals and overflow come up), hexadecimal ones, singletons, half-lines, rationals p/q, and the uncertain form
   with and without a radius, one-sided, with "??", with an exponent. The reference reads each number with
   mpfr_strtofr at 4096 bits rounding the way the bound goes, then rounds that to binary64 the same way: a decimal of
   this size that is not a binary64 value is far further than 2^-4096 from one, so the two roundings agree with the
   exact one. The uncertain form's ends are m - r and m + r as exact integers times a power of ten, rounded once (rounding
   m and r first is wrong when their sum is a binary64 value: the first version of this reference did, 42 failures). A rational p/q (up to 25
   digits each) must be the tightest interval, exactly as MPFR divides the two integers.
   Literals that are not ones (garbage, ends out of order, an infinite singleton, decorations) must give the empty
   interval and status 1. */
#include <float.h>
#include <math.h>
#include <mpfr.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ival.h"

static uint64_t rs = 0x5851f42d4c957f2dULL;
static uint64_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }
static int same(double a, double b) { return (a != a && b != b) || a == b; }

static mpfr_t A, B, C;
static double rd(const char *s, int up)   /* s rounded down or up to binary64 */
{
  mpfr_strtofr(A, s, NULL, 0, up ? MPFR_RNDU : MPFR_RNDD);
  return mpfr_get_d(A, up ? MPFR_RNDU : MPFR_RNDD) + 0.0;
}
static void digits(char *out, int n, int lead)   /* n random digits, the first nonzero when lead */
{
  for (int i = 0; i < n; i++) out[i] = (char)('0' + rnd() % 10);
  if (lead && n) out[0] = (char)('1' + rnd() % 9);
  out[n] = 0;
}
static void decimal(char *out)   /* a random decimal number */
{
  char d[48], f[48];
  digits(d, 1 + (int)(rnd() % 20), 0); digits(f, (int)(rnd() % 20), 0);
  int e = (int)(rnd() % 681) - 340;
  sprintf(out, "%s%s%s%se%d", rnd() % 2 ? "-" : "", d, *f ? "." : "", f, e);
}

static long checked, bad;
static char first[600];
static void want(const char *s, double zl, double zh, int st, double rl, double rh, int rst)
{
  checked++;
  if (same(zl, rl) && same(zh, rh) && st == rst) return;
  if (!bad++) snprintf(first, sizeof first, " (first: \"%s\" = [%a, %a] status %d, want [%a, %a] status %d)", s, zl, zh, st, rl, rh, rst);
}

int main(void)
{
  mpfr_init2(A, 4096); mpfr_init2(B, 4096); mpfr_init2(C, 4096);
  enum { N = 60000 };
  static char buf[N][200];
  static const char *s[N];
  static double rl[N], rh[N], zl[N], zh[N];
  static int rst[N], kind[N], loose[N];
  static unsigned char st[N];
  for (int i = 0; i < N; i++) {
    char x[96], y[96], m[48], f[48], r[24];
    int k = (int)(rnd() % 8);
    kind[i] = k; loose[i] = 0; rst[i] = 0;
    switch (k) {
      case 0: case 1: {   /* [x, y], ordered by value */
        decimal(x); decimal(y);
        double a = rd(x, 0), b = rd(y, 1);
        if (a > b) { char t[96]; strcpy(t, x); strcpy(x, y); strcpy(y, t); a = rd(x, 0); b = rd(y, 1); }
        sprintf(buf[i], "[%s, %s]", x, y);
        rl[i] = a; rh[i] = b;
        if (a == INFINITY || b == -INFINITY || a > b) { rl[i] = rh[i] = NAN; rst[i] = 1; }
        else if (rd(x, 1) > rd(y, 0)) rst[i] = 2;
        break;
      }
      case 2: {           /* [x] */
        decimal(x);
        sprintf(buf[i], "[ %s ]", x);
        rl[i] = rd(x, 0); rh[i] = rd(x, 1);
        if (rl[i] == INFINITY || rh[i] == -INFINITY) { rl[i] = rh[i] = NAN; rst[i] = 1; }
        break;
      }
      case 3: {           /* half-lines and hexadecimal */
        double v = ldexp((double)(rnd() >> 11), (int)(rnd() % 2000) - 1100);
        if (rnd() % 2) v = -v;
        sprintf(x, "%a", v);
        if (rnd() % 2) { sprintf(buf[i], "[%s,]", x); rl[i] = v + 0.0; rh[i] = INFINITY; }
        else { sprintf(buf[i], "[-Inf, %s]", x); rl[i] = -INFINITY; rh[i] = v + 0.0; }
        break;
      }
      case 4: {           /* a rational p/q */
        char p[32], q[32];
        digits(p, 1 + (int)(rnd() % 25), 1); digits(q, 1 + (int)(rnd() % 25), 1);
        int neg = rnd() % 2;
        sprintf(buf[i], "[%s%s/%s]", neg ? "-" : "", p, q);
        mpfr_strtofr(B, p, NULL, 10, MPFR_RNDN); mpfr_strtofr(C, q, NULL, 10, MPFR_RNDN);   /* exact: integers */
        if (neg) mpfr_neg(B, B, MPFR_RNDN);
        mpfr_div(A, B, C, MPFR_RNDD); rl[i] = mpfr_get_d(A, MPFR_RNDD) + 0.0;
        mpfr_div(A, B, C, MPFR_RNDU); rh[i] = mpfr_get_d(A, MPFR_RNDU) + 0.0;
        loose[i] = !(rd(p, 0) == rd(p, 1) && rd(q, 0) == rd(q, 1));
        break;
      }
      case 5: case 6: {   /* the uncertain form */
        digits(m, 1 + (int)(rnd() % 18), 0); digits(f, (int)(rnd() % 8), 0);
        int side = (int)(rnd() % 3), inf = rnd() % 10 == 0, nr = (int)(rnd() % 4), e = (int)(rnd() % 61) - 30;
        digits(r, nr, 0);
        int neg = rnd() % 2;
        sprintf(buf[i], "%s%s%s%s?%s%s", neg ? "-" : "", m, *f ? "." : "", f, inf ? "?" : r, side == 1 ? "u" : side == 2 ? "d" : "");
        if (e) sprintf(buf[i] + strlen(buf[i]), "e%d", e);
        /* the ends exactly: N = M - R and M + R as integers (M, R m's and r's digits, a half unit as 10 M -+ 5), then
           N 10^k rounded the bound's way in one step (an exact product for k >= 0, a division for k < 0) */
        char md[128], rdg[64];
        sprintf(md, "%s%s", m, f);
        int k = e - (int)strlen(f);
        if (nr) strcpy(rdg, r); else { strcat(md, "0"); strcpy(rdg, "5"); k--; }
        mpfr_strtofr(B, md, NULL, 10, MPFR_RNDN); mpfr_strtofr(C, rdg, NULL, 10, MPFR_RNDN);   /* exact integers */
        if (neg) mpfr_neg(B, B, MPFR_RNDN);
        for (int up = 0; up < 2; up++) {
          mpfr_rnd_t md2 = up ? MPFR_RNDU : MPFR_RNDD;
          mpfr_t N, T;
          mpfr_inits2(4096, N, T, (mpfr_ptr)0);
          int sidem = (up && side == 2) || (!up && side == 1);   /* that end is m itself */
          if (sidem) mpfr_set(N, B, MPFR_RNDN); else if (up) mpfr_add(N, B, C, MPFR_RNDN); else mpfr_sub(N, B, C, MPFR_RNDN);
          mpfr_set_ui(T, 10, MPFR_RNDN); mpfr_pow_ui(T, T, (unsigned)abs(k), MPFR_RNDN);   /* exact: k is small */
          if (k >= 0) mpfr_mul(A, N, T, md2); else mpfr_div(A, N, T, md2);
          double v = mpfr_get_d(A, md2) + 0.0;
          if (inf && !sidem) v = up ? INFINITY : -INFINITY;
          if (up) rh[i] = v; else rl[i] = v;
          mpfr_clears(N, T, (mpfr_ptr)0);
        }
        break;
      }
      default: {          /* not literals */
        static const char *badl[] = { "[1, 2", "1, 2]", "[2, 1]", "[+inf]", "[-inf]", "[inf, inf]", "[1, 2]_com", "[ nai ]",
                                      "[1 0, 2]", "[0x1p, 2]", "[1/0]", "hello", "", "[1,2,3]", "3.5?1x", "?1", "[1e, 2]" };
        strcpy(buf[i], badl[rnd() % (sizeof badl / sizeof badl[0])]);
        rl[i] = rh[i] = NAN; rst[i] = 1;
      }
    }
    s[i] = buf[i];
  }
  ival_text(s, zl, zh, st, N);
  long loosened = 0;
  for (int i = 0; i < N; i++) {
    loosened += kind[i] == 4 && loose[i];
    want(s[i], zl[i], zh[i], st[i], rl[i], rh[i], rst[i]);
  }
  /* nums */
  double l[6] = { 1, -INFINITY, INFINITY, NAN, 2, -0.0 }, u[6] = { 2, INFINITY, INFINITY, 1, 1, 0.0 }, nl[6], nh[6];
  unsigned char ns[6];
  ival_nums(l, u, nl, nh, ns, 6);
  double wl[6] = { 1, -INFINITY, NAN, NAN, NAN, 0 }, wh[6] = { 2, INFINITY, NAN, NAN, NAN, 0 };
  int ws[6] = { 0, 0, 1, 1, 1, 0 };
  for (int i = 0; i < 6; i++) want("nums", nl[i], nh[i], ns[i], wl[i], wh[i], ws[i]);
  if (!bad)
    printf("VERDICT: IDENTICAL (%ld literals as MPFR reads them, %ld of them rationals whose p or q is not a binary64 value)\n", checked, loosened);
  else
    printf("VERDICT: DIFFERS (%ld of %ld)%s\n", bad, checked, first);
  return bad != 0;
}
