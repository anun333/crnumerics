/* ival-text.c: IEEE 1788.1's interval constructors, numsToInterval and textToInterval (ival.h), 2026-10-09.

   ival_nums: [l, u] when l <= u, l is not +inf and u is not -inf (NaN fails that), else the empty interval.
   ival_text reads 1788's interval literals:
     - inf-sup: "[l, u]", "[x]" (the tightest interval around x), "[l,]" and "[,u]" (an omitted end infinite), "[,]"
       and "[entire]", "[]" and "[empty]"; l and u decimal, hexadecimal ("0x1.8p-3"), "inf" or "infinity" with a sign,
       or a rational "p/q" of decimal integers; case and spaces inside the brackets free;
     - uncertain: "m?r", m a decimal number and r a radius in units of m's last digit ("3.56?1" is [3.55, 3.57]): no r
       means half a unit, "??" an infinite radius; then "u" or "d" for one side only, then an exponent applying to
       both ("2.500?5e+27").
   The bounds are the exact values rounded outward. Decimals and hexadecimals go through strtod with the rounding mode
   set down or up (C's strtod rounds in the current mode, as glibc's does); the uncertain form's ends are computed
   exactly in decimal first. A rational p/q is rounded exactly too: from a quotient within a few ulps, the doubles d
   beside it are tested by d q <= p in big integers (p and q of up to 400 digits). A literal that is not one, or whose lower end exceeds
   its upper, gives the empty interval and status 1 (1788's UndefinedOperation); bounds that may be in either order
   after rounding give the interval and status 2 (PossiblyUndefinedOperation); status may be NULL. */
#include <ctype.h>
#include <fenv.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include "ival.h"
#include "ival-eft.h"

#define ENTER fenv_t env; fegetenv(&env); fesetround(FE_TONEAREST); flush_off();
#define LEAVE fesetenv(&env);

void ival_nums(const double *l, const double *u, double *lo, double *hi, unsigned char *status, size_t n)
{
  for (size_t i = 0; i < n; i++) {
    int bad = empty(l[i], u[i]);
    lo[i] = bad ? NAN : canon(l[i]); hi[i] = bad ? NAN : canon(u[i]);
    if (status) status[i] = (unsigned char)bad;
  }
}

/* s[0..len) equals word, ignoring case */
static int word(const char *s, size_t len, const char *w) { return strlen(w) == len && !strncasecmp(s, w, len); }
/* a decimal or hexadecimal number, exactly the whole of s[0..len), rounded down (up = 0) or up */
static int number(const char *s, size_t len, int up, double *v)
{
  char buf[512];
  if (len == 0 || len >= sizeof buf) return 0;
  memcpy(buf, s, len); buf[len] = 0;
  const char *b = buf + (buf[0] == '+' || buf[0] == '-');
  if (word(b, strlen(b), "inf") || word(b, strlen(b), "infinity")) { *v = buf[0] == '-' ? -INFINITY : INFINITY; return 1; }
  /* digits, a point, an exponent: what strtod reads, and nothing it reads as inf or nan */
  for (const char *c = b; *c; c++)
    if (!(isxdigit((unsigned char)*c) || *c == '.' || *c == 'x' || *c == 'X' || *c == 'p' || *c == 'P' || *c == '+' || *c == '-'))
      return 0;
  if (!isdigit((unsigned char)b[0]) && b[0] != '.') return 0;
  char *end;
  fesetround(up ? FE_UPWARD : FE_DOWNWARD);
  volatile double r = strtod(buf, &end);
  fesetround(FE_TONEAREST);
  if (*end) return 0;
  *v = r;
  return 1;
}
/* ---- rationals p/q, rounded exactly: the double d is at most |p|/|q| exactly when d |q| <= |p|, compared as big
   integers (base 10^9, d = M 2^E with M an integer, a power of two moved to whichever side keeps it whole) ---- */
enum { LIMBS = 160 };   /* 1440 decimal digits: the products below for p and q of up to 400 digits */
typedef struct { uint32_t d[LIMBS]; int n; } big;
static int big_dec(const char *s, size_t len, big *b)   /* digits only */
{
  b->n = 0;
  if (len > 400) return 0;
  for (size_t k = 0; k < len; k++) {   /* b = b * 10 + digit */
    uint64_t c = (uint64_t)(s[k] - '0');
    for (int i = 0; i < b->n; i++) { uint64_t t = (uint64_t)b->d[i] * 10 + c; b->d[i] = (uint32_t)(t % 1000000000u); c = t / 1000000000u; }
    if (c) b->d[b->n++] = (uint32_t)c;
  }
  return 1;
}
static void big_mul(big *b, uint32_t m)   /* b *= m, m < 2^31 */
{
  uint64_t c = 0;
  for (int i = 0; i < b->n; i++) { uint64_t t = (uint64_t)b->d[i] * m + c; b->d[i] = (uint32_t)(t % 1000000000u); c = t / 1000000000u; }
  while (c && b->n < LIMBS) { b->d[b->n++] = (uint32_t)(c % 1000000000u); c /= 1000000000u; }
}
static void big_pow2(big *b, int e) { for (; e >= 30; e -= 30) big_mul(b, 1u << 30); if (e > 0) big_mul(b, 1u << e); }
static int big_cmp(const big *a, const big *b)
{
  int an = a->n, bn = b->n;
  while (an && !a->d[an - 1]) an--;
  while (bn && !b->d[bn - 1]) bn--;
  if (an != bn) return an > bn ? 1 : -1;
  for (int i = an - 1; i >= 0; i--) if (a->d[i] != b->d[i]) return a->d[i] > b->d[i] ? 1 : -1;
  return 0;
}
/* the sign of d q - p, for d >= 0 finite */
static int cmp_dq(double d, const big *q, const big *p)
{
  if (d == 0) return p->n ? -1 : 0;
  int e;
  double m = frexp(d, &e);
  uint64_t M = (uint64_t)ldexp(m, 53);
  e -= 53;
  big l = *q, r = *p;
  big_mul(&l, (uint32_t)(M >> 31)); big_mul(&l, 1u << 31);   /* q M, in two steps (M < 2^53) */
  big lo = *q;
  big_mul(&lo, (uint32_t)(M & 0x7fffffffu));
  /* l = q (M >> 31) 2^31, then add q (M & (2^31 - 1)) */
  uint64_t c = 0;
  int n = l.n > lo.n ? l.n : lo.n;
  for (int i = 0; i < n; i++) {
    uint64_t t = (uint64_t)(i < l.n ? l.d[i] : 0) + (i < lo.n ? lo.d[i] : 0) + c;
    l.d[i] = (uint32_t)(t % 1000000000u); c = t / 1000000000u;
  }
  l.n = n;
  if (c && l.n < LIMBS) l.d[l.n++] = (uint32_t)c;
  if (e >= 0) big_pow2(&l, e); else big_pow2(&r, -e);
  return big_cmp(&l, &r);
}
/* |p| / |q| rounded down and up, exactly; 0 if q is 0 or too long */
static int ratio(const char *ps, size_t pl, const char *qs, size_t ql, double *d, double *u)
{
  big p, q;
  if (!big_dec(ps, pl, &p) || !big_dec(qs, ql, &q) || !big_cmp(&q, &(big){ .n = 0 })) return 0;
  double pd, pu, qd, qu;
  number(ps, pl, 0, &pd); number(ps, pl, 1, &pu); number(qs, ql, 0, &qd); number(qs, ql, 1, &qu);
  double y = div_r(pd, qu, 0);   /* below the quotient, within a few ulps */
  if (isinf(y)) y = DBL_MAX;
  while (y > 0 && cmp_dq(y, &q, &p) > 0) y = nextafter(y, 0);                         /* down to d q <= p */
  for (double t; (t = nextafter(y, INFINITY)) <= DBL_MAX && cmp_dq(t, &q, &p) <= 0;) y = t;   /* the largest such */
  *d = y;
  if (cmp_dq(y, &q, &p) == 0) *u = y;
  else *u = nextafter(y, INFINITY);   /* past DBL_MAX: +inf */
  return 1;
}
/* a number or a rational p/q (decimal integers, a sign on p), rounded down or up */
static int bound(const char *s, size_t len, int up, double *v)
{
  while (len && isspace((unsigned char)*s)) s++, len--;
  while (len && isspace((unsigned char)s[len - 1])) len--;
  const char *slash = memchr(s, '/', len);
  if (!slash) return number(s, len, up, v);
  int neg = len && s[0] == '-';
  if (len && (s[0] == '-' || s[0] == '+')) s++, len--, slash = memchr(s, '/', len);
  size_t pl = (size_t)(slash - s), ql = len - pl - 1;
  if (!pl || !ql) return 0;
  for (size_t k = 0; k < len; k++) if (k != pl && !isdigit((unsigned char)s[k])) return 0;
  double d, u;
  if (!ratio(s, pl, slash + 1, ql, &d, &u)) return 0;
#if IVAL_PLANT_ARITH == 28   /* 28: p and q rounded outward and divided outward, as the first version did */
  { double pd, pu, qd, qu; number(s, pl, 0, &pd); number(s, pl, 1, &pu); number(slash + 1, ql, 0, &qd); number(slash + 1, ql, 1, &qu);
    d = div_r(pd, qu, 0); u = div_r(pu, qd, 1); }
#endif
  *v = neg ? -(up ? d : u) : (up ? u : d);
  return 1;
}

/* ---- the uncertain form: decimal digit strings, signs apart ---- */
/* a + b or |a - b| of digit strings (no sign), into out; returns the sign of a - b for sub */
static int dsub(const char *a, const char *b, char *out)
{
#if IVAL_PLANT_ARITH != 27   /* 27: leading zeros kept, so "000" counts as longer than "5" */
  while (a[0] == '0' && a[1]) a++;   /* leading zeros off, so the longer is the larger */
  while (b[0] == '0' && b[1]) b++;
#endif
  size_t la = strlen(a), lb = strlen(b);
  int cmp = la != lb ? (la > lb ? 1 : -1) : strcmp(a, b);
  if (cmp < 0) { const char *t = a; a = b; b = t; size_t u = la; la = lb; lb = u; }
  int borrow = 0;
  out[la] = 0;
  for (size_t i = 0; i < la; i++) {
    int d = a[la - 1 - i] - '0' - borrow - (i < lb ? b[lb - 1 - i] - '0' : 0);
    borrow = d < 0; out[la - 1 - i] = (char)('0' + d + 10 * borrow);
  }
  return cmp > 0 ? 1 : cmp < 0 ? -1 : 0;
}
static void dadd(const char *a, const char *b, char *out)
{
  size_t la = strlen(a), lb = strlen(b), l = (la > lb ? la : lb) + 1;
  int carry = 0;
  out[l] = 0;
  for (size_t i = 0; i < l; i++) {
    int d = carry + (i < la ? a[la - 1 - i] - '0' : 0) + (i < lb ? b[lb - 1 - i] - '0' : 0);
    carry = d / 10; out[l - 1 - i] = (char)('0' + d % 10);
  }
}
/* sign * digits * 10^e rounded down or up */
static double scaled(int sign, const char *digits, long e, int up)
{
  char *buf = malloc(strlen(digits) + 32);
  sprintf(buf, "%s%se%ld", sign < 0 ? "-" : "", digits, e);
  double v;
  number(buf, strlen(buf), up, &v);
  free(buf);
  return v;
}
static int uncertain(const char *s, double *lo, double *hi)
{
  /* m: sign, digits, point, digits */
  int sign = 1;
  if (*s == '+' || *s == '-') sign = *s++ == '-' ? -1 : 1;
  size_t cap = strlen(s) + 4;
  char *m = malloc(cap), *r = malloc(cap), *x = malloc(2 * cap + 4), *y = malloc(2 * cap + 4);
  size_t ml = 0, rl = 0;
  long dec = 0;
  int ok = 0, seen = 0, inf = 0, side = 0;
  for (; isdigit((unsigned char)*s); s++) m[ml++] = *s, seen = 1;
  if (*s == '.') for (s++; isdigit((unsigned char)*s); s++) m[ml++] = *s, dec++, seen = 1;
  m[ml] = 0;
  if (!seen || *s++ != '?') goto out;
  if (*s == '?') inf = 1, s++;
  else for (; isdigit((unsigned char)*s); s++) r[rl++] = *s;
  r[rl] = 0;
  if (*s == 'u' || *s == 'd') side = *s++ == 'u' ? 1 : -1;
  long e = 0;
  if (*s == 'e' || *s == 'E') {
    char *end;
    e = strtol(s + 1, &end, 10);
    if (end == s + 1 || e > 100000 || e < -100000) goto out;
    s = end;
  }
  if (*s) goto out;
  if (inf) {
    double v = scaled(sign, m, e - dec, 0), w = scaled(sign, m, e - dec, 1);
    *lo = side > 0 ? v : -INFINITY; *hi = side < 0 ? w : INFINITY;
    ok = 1; goto out;
  }
  if (!rl) { strcat(m, "0"); dec++; strcpy(r, "5"); }   /* half a unit of the last digit */
  /* [m - r, m + r] times 10^(e - dec), with sign: the lower end is sign m - r */
  long sc = e - dec;
  if (sign > 0) {
    int s1 = dsub(m, r, x); dadd(m, r, y);
    *lo = side > 0 ? scaled(1, m, sc, 0) : scaled(s1, x, sc, 0);
    *hi = side < 0 ? scaled(1, m, sc, 1) : scaled(1, y, sc, 1);
  } else {   /* -m - r and -m + r */
    dadd(m, r, x); int s2 = dsub(r, m, y);
    *lo = side > 0 ? scaled(-1, m, sc, 0) : scaled(-1, x, sc, 0);
    *hi = side < 0 ? scaled(-1, m, sc, 1) : scaled(s2, y, sc, 1);
  }
  ok = 1;
out:
  free(m); free(r); free(x); free(y);
  return ok;
}

/* one literal: 0 valid, 1 not, 2 valid with the order of its ends undecided */
static int text1(const char *s, double *lo, double *hi)
{
  *lo = *hi = NAN;
  while (isspace((unsigned char)*s)) s++;
  size_t len = strlen(s);
  while (len && isspace((unsigned char)s[len - 1])) len--;
  if (!len) return 1;
  if (s[0] != '[') {
    char *t = malloc(len + 1);
    memcpy(t, s, len); t[len] = 0;
    int ok = uncertain(t, lo, hi);
    free(t);
    if (!ok) { *lo = *hi = NAN; return 1; }
    return 0;
  }
  if (s[len - 1] != ']') return 1;
  const char *in = s + 1;
  size_t il = len - 2;
  while (il && isspace((unsigned char)*in)) in++, il--;
  while (il && isspace((unsigned char)in[il - 1])) il--;
  if (!il || word(in, il, "empty")) return 0;
  if (word(in, il, "entire")) { *lo = -INFINITY; *hi = INFINITY; return 0; }
  const char *comma = memchr(in, ',', il);
  double l, u, lu, ud;
  if (!comma) {   /* [x]: x finite, the tightest interval around it */
    if (!bound(in, il, 0, &l) || !bound(in, il, 1, &u)) return 1;
    if (l == INFINITY || u == -INFINITY) return 1;   /* x itself infinite */
    *lo = canon(l); *hi = canon(u);
    return 0;
  }
  size_t ll = (size_t)(comma - in), ul = il - ll - 1;
  const char *us = comma + 1;
  int lempty = 1, uempty = 1;
  for (size_t k = 0; k < ll; k++) if (!isspace((unsigned char)in[k])) lempty = 0;
  for (size_t k = 0; k < ul; k++) if (!isspace((unsigned char)us[k])) uempty = 0;
  if (lempty) l = lu = -INFINITY; else if (!bound(in, ll, 0, &l) || !bound(in, ll, 1, &lu)) return 1;
  if (uempty) u = ud = INFINITY; else if (!bound(us, ul, 1, &u) || !bound(us, ul, 0, &ud)) return 1;
  if (l == INFINITY || u == -INFINITY || l > u) return 1;
  *lo = canon(l); *hi = canon(u);
  return lu > ud ? 2 : 0;   /* the exact ends may be in either order */
}
void ival_text(const char *const *s, double *lo, double *hi, unsigned char *status, size_t n)
{
  ENTER
  for (size_t i = 0; i < n; i++) {
    int st = text1(s[i], &lo[i], &hi[i]);
    if (st == 1) lo[i] = hi[i] = NAN;
    if (status) status[i] = (unsigned char)st;
  }
  LEAVE
}
