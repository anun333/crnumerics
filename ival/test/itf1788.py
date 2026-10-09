#!/usr/bin/env python3
"""itf1788.py: IEEE 1788's ITF1788 test suite (bare intervals) as a C program that runs it on ival.

Usage: itf1788.py ITL_DIR > itf1788-check.c   (ITL_DIR: the itl folder of a clone of ITF1788, on GitHub)

Reads every testcase whose name has no "_dec" (decorations are not in ival), and of its statements those whose
operation ival has. An interval literal's bounds are binary64 numbers rounded to nearest, as ITF1788's own plugins
write them (the C++ one as I<double>(a, b), its numbers C++ literals): [-2.0, -0.1] is the interval from -2 to the
double nearest -0.1, and the expected results assume that. (Rounding the bounds outward instead, as text-to-interval
would, was the first try: 96 tests then failed, each on a decimal bound.) A statement
"op args = tight <= accurate" wants the tight result; ival claims the tightest, so the program counts a result that
is only within the accurate one as a failure, reported apart. Every other operation is counted, by name, as not in
ival yet: those counts are the part of 1788 still to do.

The program runs each operation's tests twice, all in one call (the vector path) and one at a time, which must
agree, and prints a VERDICT line."""
import math, re, sys
from fractions import Fraction
from pathlib import Path

INF = float('inf')

# ITL name -> (ival function, kind, number of interval operands). Kinds: I an interval, N a number, Z a number whose
# zero sign 1788.1 fixes (inf -0, sup +0; ITF1788's own plugins compare the other numbers with ==), M midRad's two
# numbers, B a boolean, X isMember (a number and an interval to a boolean), O overlap's state, P an interval and an
# int to an interval (pown)
OPS = {'neg': ('neg', 'I', 1), 'sqr': ('sqr', 'I', 1), 'recip': ('recip', 'I', 1), 'sqrt': ('sqrt', 'I', 1),
       'exp': ('exp', 'I', 1), 'exp2': ('exp2', 'I', 1), 'exp10': ('exp10', 'I', 1), 'expm1': ('expm1', 'I', 1),
       'log': ('log', 'I', 1), 'log2': ('log2', 'I', 1), 'log10': ('log10', 'I', 1), 'logp1': ('log1p', 'I', 1),
       'sin': ('sin', 'I', 1), 'cos': ('cos', 'I', 1), 'tan': ('tan', 'I', 1), 'asin': ('asin', 'I', 1),
       'acos': ('acos', 'I', 1), 'atan': ('atan', 'I', 1), 'sinh': ('sinh', 'I', 1), 'cosh': ('cosh', 'I', 1),
       'tanh': ('tanh', 'I', 1), 'asinh': ('asinh', 'I', 1), 'acosh': ('acosh', 'I', 1), 'atanh': ('atanh', 'I', 1),
       'cbrt': ('cbrt', 'I', 1), 'add': ('add', 'I', 2), 'sub': ('sub', 'I', 2), 'mul': ('mul', 'I', 2),
       'div': ('div', 'I', 2), 'pow': ('pow', 'I', 2), 'atan2': ('atan2', 'I', 2), 'hypot': ('hypot', 'I', 2),
       'fma': ('fma', 'I', 3),
       'pos': ('pos', 'I', 1), 'abs': ('abs', 'I', 1), 'sign': ('sign', 'I', 1), 'ceil': ('ceil', 'I', 1),
       'floor': ('floor', 'I', 1), 'trunc': ('trunc', 'I', 1), 'roundTiesToAway': ('round', 'I', 1),
       'roundTiesToEven': ('roundeven', 'I', 1), 'min': ('min', 'I', 2), 'max': ('max', 'I', 2),
       'intersection': ('intersect', 'I', 2), 'convexHull': ('hull', 'I', 2),
       'cancelMinus': ('cancelminus', 'I', 2), 'cancelPlus': ('cancelplus', 'I', 2),
       'inf': ('inf', 'Z', 1), 'sup': ('sup', 'Z', 1), 'mid': ('mid', 'N', 1), 'wid': ('wid', 'N', 1),
       'rad': ('rad', 'N', 1), 'mag': ('mag', 'N', 1), 'mig': ('mig', 'N', 1), 'midRad': ('midrad', 'M', 1),
       'isEmpty': ('isempty', 'B', 1), 'isEntire': ('isentire', 'B', 1), 'isSingleton': ('issingleton', 'B', 1),
       'isCommonInterval': ('iscommon', 'B', 1), 'isMember': ('ismember', 'X', 1), 'equal': ('equal', 'B', 2),
       'subset': ('subset', 'B', 2), 'less': ('less', 'B', 2), 'precedes': ('precedes', 'B', 2),
       'interior': ('interior', 'B', 2), 'strictLess': ('strictless', 'B', 2),
       'strictPrecedes': ('strictprecedes', 'B', 2), 'disjoint': ('disjoint', 'B', 2), 'overlap': ('overlap', 'O', 2), 'pown': ('pown', 'P', 1)}
# overlap's states, in ival.h's enum ival_overlap order
STATES = ['bothEmpty', 'firstEmpty', 'secondEmpty', 'before', 'meets', 'overlaps', 'starts', 'containedBy',
          'finishes', 'equals', 'finishedBy', 'contains', 'startedBy', 'overlappedBy', 'metBy', 'after']


def exact(s):
    """the exact value of an ITL number, a Fraction or +-inf"""
    t = s.strip().lower().replace('_', '')
    neg = t.startswith('-')
    t = t.lstrip('+-')
    if t in ('infinity', 'inf'):
        return -INF if neg else INF
    if t.startswith('0x'):
        m = re.fullmatch(r'0x([0-9a-f]*)(?:\.([0-9a-f]*))?(?:p([+-]?\d+))?', t)
        if not m:
            raise ValueError(s)
        ip, fp, ex = m.group(1) or '', m.group(2) or '', int(m.group(3) or 0)
        v = Fraction(int(ip + fp or '0', 16), 16 ** len(fp)) * Fraction(2) ** ex
    else:
        v = Fraction(t)
    return -v if neg else v


def nearest(s):
    """an ITL number rounded to the nearest binary64, as a C or C++ literal is (exactly, with fractions)"""
    if s.strip().lower() == 'nan':
        return math.nan
    v = exact(s)
    if isinstance(v, float):
        return v
    try:
        f = float(v)                  # correctly rounded (int / int)
        return -0.0 if f == 0 and s.strip().startswith('-') else f
    except OverflowError:
        return INF if v > 0 else -INF


def interval(tok):
    """an ITL interval literal as (lo, hi); None if it is not one ival can take"""
    m = re.fullmatch(r'\[\s*([^\],]*?)\s*(?:,\s*([^\]]*?)\s*)?\]', tok)
    if not m:
        return None
    a, b = m.group(1).lower(), m.group(2)
    if b is None:
        if a == 'empty':
            return (math.nan, math.nan)
        if a == 'entire':
            return (-INF, INF)
        if a == 'nai':
            return None
        b = a
    return (nearest(a) + 0.0, nearest(b) + 0.0)   # interval ends: zeros unsigned


def c(x):
    if math.isnan(x):
        return 'NAN'
    if math.isinf(x):
        return 'INFINITY' if x > 0 else '-INFINITY'
    return x.hex()


def tests(itl):
    """(file, testcase, ITL op, operands, tight, accurate, line) for every bare statement"""
    for f in sorted(Path(itl).glob('*.itl')):
        src = f.read_text()
        # comments out, keeping line numbers
        src = re.sub(r'/\*.*?\*/', lambda m: '\n' * m.group(0).count('\n'), src, flags=re.S)
        src = re.sub(r'//[^\n]*', '', src)
        for m in re.finditer(r'testcase\s+(\S+)\s*\{(.*?)\}', src, flags=re.S):
            name, body = m.group(1), m.group(2)
            if '_dec' in name:
                continue
            pos = m.start(2)
            for st in body.split(';'):
                lead = len(st) - len(st.lstrip())
                stl = src.count('\n', 0, pos + lead) + 1
                pos += len(st) + 1
                st = ' '.join(st.split())
                if not st:
                    continue
                st = re.sub(r'\s+signal\s+\w+$', '', st)   # exceptions: ival raises none
                lhs, _, rhs = st.partition(' = ')
                toks = re.findall(r'\[[^\]]*\]|\S+', lhs)
                res = re.findall(r'\[[^\]]*\]|<=|\S+', rhs)
                yield f.name, name, toks[0], toks[1:], res, stl


def number(tok):
    try:
        return nearest(tok)
    except (ValueError, ZeroDivisionError):
        return None


def main():
    itl = sys.argv[1]
    rows, skipped, odd = [], {}, []
    for fname, case, op, args, res, line in tests(itl):
        where = f'{fname}:{line} {case}'
        if op not in OPS:
            skipped[op] = skipped.get(op, 0) + 1
            continue
        fn, kind, k = OPS[op]
        x = 0.0
        if kind == 'X':                                   # isMember: the number first
            x = number(args[0]) if args else None
            args = args[1:]
        if kind == 'P':                                   # pown: the int last
            try:
                x = float(int(args[-1])) if args and -2**31 <= int(args[-1]) < 2**31 else None
            except ValueError:
                x = None
            args = args[:-1]
        ivs = [interval(a) for a in args]
        ok = len(ivs) == k and None not in ivs and x is not None
        tight = acc = None
        code = 0
        if kind in ('I', 'P'):
            tight = interval(res[0]) if res else None
            acc = interval(res[2]) if len(res) == 3 and res[1] == '<=' else tight
            ok = ok and tight is not None and acc is not None and len(res) in (1, 3)
        elif kind in ('N', 'Z', 'M'):
            nums = [number(r) for r in res]
            ok = ok and len(nums) == (2 if kind == 'M' else 1) and None not in nums
            if ok:
                tight = acc = (nums[0], nums[-1])
        elif kind in ('B', 'X'):
            ok = ok and len(res) == 1 and res[0] in ('true', 'false')
            code = int(ok and res[0] == 'true')
        else:
            ok = ok and len(res) == 1 and res[0] in STATES
            code = STATES.index(res[0]) if ok else 0
        if not ok:
            odd.append(f'{where}: {op} {" ".join(args)} = {" ".join(res)}')
            continue
        tight = tight or (0.0, 0.0)
        acc = acc or tight
        rows.append((fn, kind, k, ivs, x, tight, acc, code, where))
    fns = sorted({(r[0], r[1], r[2]) for r in rows})
    idx = {f: i for i, (f, _, _) in enumerate(fns)}
    out = []
    w = out.append
    w('/* generated by ival/test/itf1788.py from ITF1788\'s itl files: do not edit */')
    w('#include <math.h>\n#include <stdio.h>\n#include <stdlib.h>\n#include <string.h>\n#include "ival.h"')
    w('struct t { int f; double a[6], x, tl, th, al, ah; int code; const char *where; };')
    w('static const char *const NAME[] = { ' + ', '.join(f'"{f}"' for f, _, _ in fns) + ' };')
    w('static const char KIND[] = "' + ''.join(kd for _, kd, _ in fns) + '";')
    w('static const int ARITY[] = { ' + ', '.join(str(k) for _, _, k in fns) + ' };')
    w('static const struct t T[] = {')
    for fn, kind, k, ivs, x, tight, acc, code, where in rows:
        a = [v for iv in ivs for v in iv] + [0.0] * (6 - 2 * k)
        w(f'  {{ {idx[fn]}, {{ {", ".join(c(v) for v in a)} }}, {c(x)}, {c(tight[0])}, {c(tight[1])}, {c(acc[0])}, '
          f'{c(acc[1])}, {code}, "{where}" }},')
    w('};')
    w('enum { NT = sizeof T / sizeof T[0], NF = sizeof NAME / sizeof NAME[0] };')
    w('static void call(int f, const double *a0, const double *a1, const double *b0, const double *b1,')
    w('                 const double *c0, const double *c1, const double *x, double *zl, double *zh,')
    w('                 unsigned char *r, size_t n)\n{')
    w('  int *pw = malloc((n ? n : 1) * sizeof *pw);\n  for (size_t i = 0; i < n; i++) pw[i] = (int)x[i];\n  switch (f) {')
    for f, kind, k in fns:
        i = idx[f]
        if kind == 'I':
            args = ['a0, a1', 'a0, a1, b0, b1', 'a0, a1, b0, b1, c0, c1'][k - 1] + ', zl, zh'
        elif kind in ('N', 'Z'):
            args = 'a0, a1, zl'
        elif kind == 'M':
            args = 'a0, a1, zl, zh'
        elif kind == 'X':
            args = 'x, a0, a1, r'
        elif kind == 'P':
            args = 'a0, a1, pw, zl, zh'
        else:
            args = ['a0, a1', 'a0, a1, b0, b1'][k - 1] + ', r'
        w(f'    case {i}: ival_{f}({args}, n); break;')
    w('  }\n  free(pw);\n  (void)b0; (void)b1; (void)c0; (void)c1; (void)x; (void)zh; (void)r;\n}')
    skip = ', '.join(f'{op} {n}' for op, n in sorted(skipped.items(), key=lambda x: (-x[1], x[0])))
    w(f'static const char SKIPPED[] = "{skip}";')
    w(f'static const int NSKIPPED = {sum(skipped.values())}, NODD = {len(odd)};')
    w(RUNNER)
    print('\n'.join(out))
    for o in odd[:20]:
        print('unread:', o, file=sys.stderr)


RUNNER = r'''
static int same(double a, double b) { return (a != a && b != b) || a == b; }
/* inf and sup must match to the sign of a zero (inf of [0, 1] is -0) */
static int samenum(double a, double b) { return same(a, b) && (a != a || signbit(a) == signbit(b)); }
/* one result against its test: 0 right, 1 only accurate (holds the tight interval, inside the accurate one), 2 wrong */
static int verdict(const struct t *t, char kind, double zl, double zh, int r)
{
  if (kind == 'B' || kind == 'X' || kind == 'O') return r == t->code ? 0 : 2;
  if (kind == 'Z') return samenum(zl, t->tl) ? 0 : 2;
  if (kind == 'N') return same(zl, t->tl) ? 0 : 2;
  if (kind == 'M') return same(zl, t->tl) && same(zh, t->th) ? 0 : 2;
  if (same(zl, t->tl) && same(zh, t->th)) return 0;
  return zl == zl && t->tl == t->tl && zl <= t->tl && zh >= t->th && zl >= t->al && zh <= t->ah ? 1 : 2;
}
int main(void)
{
  static double v[7][NT], zl[NT], zh[NT];
  static unsigned char r[NT];
  int per[NF] = { 0 }, bad = 0, accurate = 0, split = 0, shown = 0;
  for (int f = 0; f < NF; f++) {
    int n = 0, idx[NT];
    for (int i = 0; i < NT; i++)
      if (T[i].f == f) { for (int j = 0; j < 6; j++) v[j][n] = T[i].a[j]; v[6][n] = T[i].x; idx[n++] = i; }
    per[f] = n;
    call(f, v[0], v[1], v[2], v[3], v[4], v[5], v[6], zl, zh, r, n);   /* all at once */
    for (int k = 0; k < n; k++) {                                      /* and one at a time */
      double a[7], sl = 0, sh = 0; unsigned char sr = 0;
      for (int j = 0; j < 7; j++) a[j] = v[j][k];
      call(f, &a[0], &a[1], &a[2], &a[3], &a[4], &a[5], &a[6], &sl, &sh, &sr, 1);
      const struct t *t = &T[idx[k]];
      char kind = KIND[f];
      int num = kind == 'N' || kind == 'Z';
      int once = kind == 'I' || kind == 'P' || num || kind == 'M' ? memcmp(&sl, &zl[k], 8) || (!num && memcmp(&sh, &zh[k], 8)) : sr != r[k];
      if (once) split++;
      int vd = verdict(t, kind, zl[k], zh[k], r[k]);
      if (!vd) continue;
      if (vd == 1) accurate++; else bad++;
      if (shown++ < 12) {
        printf("%s %s:", vd == 1 ? "NOT TIGHTEST" : "WRONG", NAME[f]);
        if (kind == 'X') printf(" %a", t->x);
        for (int j = 0; j < 2 * ARITY[f]; j += 2) printf(" [%a, %a]", t->a[j], t->a[j + 1]);
        if (kind == 'P') printf(" %d", (int)t->x);
        if (kind == 'B' || kind == 'X' || kind == 'O') printf(" = %d, want %d (%s)\n", r[k], t->code, t->where);
        else if (kind == 'N' || kind == 'Z') printf(" = %a, want %a (%s)\n", zl[k], t->tl, t->where);
        else printf(" = [%a, %a], want [%a, %a] (%s)\n", zl[k], zh[k], t->tl, t->th, t->where);
      }
    }
  }
  printf("ITF1788 bare tests ival runs, by operation:");
  for (int f = 0; f < NF; f++) printf(" %s %d", NAME[f], per[f]);
  printf("\nnot in ival yet (%d): %s\n", NSKIPPED, SKIPPED);
  if (NODD) printf("statements the converter could not read: %d\n", NODD);
  if (!bad && !accurate && !split && !NODD && NT > 0)
    printf("VERDICT: IDENTICAL (%d ITF1788 tests, every result the tight one, all at once and one at a time)\n", NT);
  else
    printf("VERDICT: DIFFERS (%d wrong, %d only accurate, %d differ between all at once and one at a time, %d unread, of %d)\n",
           bad, accurate, split, NODD, NT);
  return bad || accurate || split || NODD || NT == 0;
}'''


if __name__ == '__main__':
    main()
