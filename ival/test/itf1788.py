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

# ITL name -> (ival function, number of interval operands)
OPS = {'neg': ('neg', 1), 'sqr': ('sqr', 1), 'recip': ('recip', 1), 'sqrt': ('sqrt', 1), 'exp': ('exp', 1),
       'exp2': ('exp2', 1), 'exp10': ('exp10', 1), 'expm1': ('expm1', 1), 'log': ('log', 1), 'log2': ('log2', 1),
       'log10': ('log10', 1), 'logp1': ('log1p', 1), 'sin': ('sin', 1), 'cos': ('cos', 1), 'tan': ('tan', 1),
       'asin': ('asin', 1), 'acos': ('acos', 1), 'atan': ('atan', 1), 'sinh': ('sinh', 1), 'cosh': ('cosh', 1),
       'tanh': ('tanh', 1), 'asinh': ('asinh', 1), 'acosh': ('acosh', 1), 'atanh': ('atanh', 1), 'cbrt': ('cbrt', 1),
       'add': ('add', 2), 'sub': ('sub', 2), 'mul': ('mul', 2), 'div': ('div', 2), 'pow': ('pow', 2),
       'atan2': ('atan2', 2), 'hypot': ('hypot', 2), 'fma': ('fma', 3)}


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
    v = exact(s)
    if isinstance(v, float):
        return v
    try:
        return float(v) + 0.0         # correctly rounded (int / int); zeros unsigned
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
    return (nearest(a), nearest(b))


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


def main():
    itl = sys.argv[1]
    rows, skipped, odd = [], {}, []
    for fname, case, op, args, res, line in tests(itl):
        where = f'{fname}:{line} {case}'
        if op not in OPS:
            skipped[op] = skipped.get(op, 0) + 1
            continue
        fn, k = OPS[op]
        ivs = [interval(a) for a in args]
        tight = interval(res[0]) if res else None
        acc = interval(res[2]) if len(res) == 3 and res[1] == '<=' else tight
        if len(ivs) != k or None in ivs or tight is None or acc is None or len(res) not in (1, 3):
            odd.append(f'{where}: {op} {" ".join(args)} = {" ".join(res)}')
            continue
        rows.append((fn, k, ivs, tight, acc, where))
    fns = sorted({(r[0], r[1]) for r in rows})
    out = []
    w = out.append
    w('/* generated by ival/test/itf1788.py from ITF1788\'s itl files: do not edit */')
    w('#include <math.h>\n#include <stdio.h>\n#include <stdlib.h>\n#include <string.h>\n#include "ival.h"')
    w('struct t { int f; double a[6], tl, th, al, ah; const char *where; };')
    w('static const char *const NAME[] = { ' + ', '.join(f'"{f}"' for f, _ in fns) + ' };')
    w('static const int ARITY[] = { ' + ', '.join(str(k) for _, k in fns) + ' };')
    w('static const struct t T[] = {')
    idx = {f: i for i, (f, _) in enumerate(fns)}
    for fn, k, ivs, tight, acc, where in rows:
        a = [x for iv in ivs for x in iv] + [0.0] * (6 - 2 * k)
        w(f'  {{ {idx[fn]}, {{ {", ".join(c(x) for x in a)} }}, {c(tight[0])}, {c(tight[1])}, {c(acc[0])}, '
          f'{c(acc[1])}, "{where}" }},')
    w('};')
    w('enum { NT = sizeof T / sizeof T[0], NF = sizeof NAME / sizeof NAME[0] };')
    w('static void call(int f, const double *a0, const double *a1, const double *b0, const double *b1,')
    w('                 const double *c0, const double *c1, double *zl, double *zh, size_t n)\n{\n  switch (f) {')
    for f, k in fns:
        if k == 1:
            w(f'    case {idx[f]}: ival_{f}(a0, a1, zl, zh, n); break;')
        elif k == 2:
            w(f'    case {idx[f]}: ival_{f}(a0, a1, b0, b1, zl, zh, n); break;')
        else:
            w(f'    case {idx[f]}: ival_{f}(a0, a1, b0, b1, c0, c1, zl, zh, n); break;')
    w('  }\n  (void)b0; (void)b1; (void)c0; (void)c1;\n}')
    skip = ', '.join(f'{op} {n}' for op, n in sorted(skipped.items(), key=lambda x: (-x[1], x[0])))
    w(f'static const char SKIPPED[] = "{skip}";')
    w(f'static const int NSKIPPED = {sum(skipped.values())}, NODD = {len(odd)};')
    w(r'''
static int same(double a, double b) { return (a != a && b != b) || a == b; }
int main(void)
{
  static double v[6][NT], zl[NT], zh[NT], sl[NT], sh[NT];
  int per[NF] = { 0 }, bad = 0, accurate = 0, split = 0, shown = 0;
  for (int f = 0; f < NF; f++) {
    int n = 0, idx[NT];
    for (int i = 0; i < NT; i++) if (T[i].f == f) { for (int j = 0; j < 6; j++) v[j][n] = T[i].a[j]; idx[n++] = i; }
    per[f] = n;
    call(f, v[0], v[1], v[2], v[3], v[4], v[5], zl, zh, n);   /* all at once */
    for (int k = 0; k < n; k++) {                                     /* and one at a time */
      double a[6]; for (int j = 0; j < 6; j++) a[j] = v[j][k];
      call(f, &a[0], &a[1], &a[2], &a[3], &a[4], &a[5], &sl[k], &sh[k], 1);
      const struct t *t = &T[idx[k]];
      if (!same(sl[k], zl[k]) || !same(sh[k], zh[k])) split++;
      if (same(zl[k], t->tl) && same(zh[k], t->th)) continue;
      /* only accurate: it holds the tight result and lies within the accurate one (an empty result is neither) */
      int acc = zl[k] == zl[k] && t->tl == t->tl && zl[k] <= t->tl && zh[k] >= t->th && zl[k] >= t->al && zh[k] <= t->ah;
      if (acc) accurate++; else bad++;
      if (shown++ < 12) {
        printf("%s %s:", acc ? "NOT TIGHTEST" : "WRONG", NAME[f]);
        for (int j = 0; j < 2 * ARITY[f]; j += 2) printf(" [%a, %a]", t->a[j], t->a[j + 1]);
        printf(" = [%a, %a], want [%a, %a] (%s)\n", zl[k], zh[k], t->tl, t->th, t->where);
      }
    }
  }
  printf("ITF1788 bare tests ival runs, by operation:");
  for (int f = 0; f < NF; f++) printf(" %s %d", NAME[f], per[f]);
  printf("\nnot in ival yet (%d): %s\n", NSKIPPED, SKIPPED);
  if (NODD) printf("statements the converter could not read: %d\n", NODD);
  if (!bad && !accurate && !split && NT > 0)
    printf("VERDICT: IDENTICAL (%d ITF1788 tests, every result the tight one, all at once and one at a time)\n", NT);
  else
    printf("VERDICT: DIFFERS (%d wrong, %d only accurate, %d differ between all at once and one at a time, of %d)\n",
           bad, accurate, split, NT);
  return bad || accurate || split || NT == 0;
}''')
    print('\n'.join(out))
    for o in odd[:20]:
        print('unread:', o, file=sys.stderr)


if __name__ == '__main__':
    main()
