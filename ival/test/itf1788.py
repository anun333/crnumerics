#!/usr/bin/env python3
"""itf1788.py: IEEE 1788's ITF1788 test suite (bare intervals) as a C program that runs it on ival.

Usage: itf1788.py ITL_DIR > itf1788-check.c   (ITL_DIR: the itl folder of a clone of ITF1788, on GitHub)

Reads every testcase whose name has no "_dec" (decorations are not in ival), and of its statements those whose
operation ival has. An interval literal's bounds are binary64 numbers rounded to nearest, as ITF1788's own plugins
write them (the C++ one as I<double>(a, b), its numbers C++ literals): [-2.0, -0.1] is the interval from -2 to the
double nearest -0.1, and the expected results assume that. (Rounding the bounds outward instead, as text-to-interval
would, was the first try: 96 tests then failed, each on a decimal bound.) A statement
"op args = tight <= accurate" wants the tight result, except where ITF1788's own expectation is not the tightest:
those few are corrected, each with a proof run at generation (CORRECTIONS: exact fractions, or mpmath at 400 bits
for the ones about pi, whose tests are left out when mpmath is not installed); ival claims the tightest, so the program counts a result that
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
# int to an interval (pown), Q two intervals to two (mulRevToPair), T a string and U two numbers to an interval and a
# status (textToInterval, numsToInterval). mulrev2 is mulRev's two-operand form, X whole
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
       'strictPrecedes': ('strictprecedes', 'B', 2), 'disjoint': ('disjoint', 'B', 2), 'overlap': ('overlap', 'O', 2), 'pown': ('pown', 'P', 1),
       'mulRevToPair': ('mulrevpair', 'Q', 2), 'mulRev': ('mulrev2', 'I', 2), 'mulRevTen': ('mulrev', 'I', 3),
       'sqrRev': ('sqrrev1', 'I', 1), 'sqrRevBin': ('sqrrev', 'I', 2), 'absRev': ('absrev1', 'I', 1),
       'absRevBin': ('absrev', 'I', 2), 'coshRev': ('coshrev1', 'I', 1), 'coshRevBin': ('coshrev', 'I', 2),
       'pownRev': ('pownrev1', 'P', 1), 'pownRevBin': ('pownrev', 'P', 2), 'rootn': ('rootn', 'P', 1),
       'b-textToInterval': ('text', 'T', 0), 'b-numsToInterval': ('nums', 'U', 0),
       'sinRev': ('sinrev1', 'I', 1), 'sinRevBin': ('sinrev', 'I', 2), 'cosRev': ('cosrev1', 'I', 1),
       'cosRevBin': ('cosrev', 'I', 2), 'tanRev': ('tanrev1', 'I', 1), 'tanRevBin': ('tanrev', 'I', 2),
       'powRev1': ('powrev1', 'I', 3), 'powRev2': ('powrev2', 'I', 3)}
# the constructors' status for each 1788 signal
SIGNALS = {'': 0, 'UndefinedOperation': 1, 'PossiblyUndefinedOperation': 2}
# the calls that are not ival_<name>(operands..., results, n): the one-operand reverse forms take X whole
CALLS = {'mulrev2': 'ival_mulrev(a0, a1, b0, b1, ninf, pinf, zl, zh, n)',
         'sqrrev1': 'ival_sqrrev(a0, a1, ninf, pinf, zl, zh, n)', 'absrev1': 'ival_absrev(a0, a1, ninf, pinf, zl, zh, n)',
         'coshrev1': 'ival_coshrev(a0, a1, ninf, pinf, zl, zh, n)',
         'sinrev1': 'ival_sinrev(a0, a1, ninf, pinf, zl, zh, n)', 'cosrev1': 'ival_cosrev(a0, a1, ninf, pinf, zl, zh, n)',
         'tanrev1': 'ival_tanrev(a0, a1, ninf, pinf, zl, zh, n)',
         'pownrev1': 'ival_pownrev(a0, a1, ninf, pinf, pw, zl, zh, n)',
         'pownrev': 'ival_pownrev(a0, a1, b0, b1, pw, zl, zh, n)'}
# overlap's states, in ival.h's enum ival_overlap order
STATES = ['bothEmpty', 'firstEmpty', 'secondEmpty', 'before', 'meets', 'overlaps', 'starts', 'containedBy',
          'finishes', 'equals', 'finishedBy', 'contains', 'startedBy', 'overlappedBy', 'metBy', 'after']


# Expected results in ITF1788 that are not the tightest, each with its proof, run by exact arithmetic whenever the
# program is generated (a failing proof stops the generation). Key: file:line; value: (the tight interval, proof).
def _root7_proof():
    bd, be = Fraction(float.fromhex('0x1.588cea3f093bdp+153')), Fraction(float.fromhex('0x1.588cea3f093bep+153'))
    return bd ** 7 <= Fraction(2) ** 1074 < be ** 7   # 2^(1074/7) lies in [bd, be): bd is the root rounded down


def _mp():   # mpmath at 400 bits, for the proofs about pi; without it those corrections are not trusted
    import mpmath
    mpmath.mp.prec = 400
    return mpmath


def _rd(mp, x):   # the largest double at most x
    f = float(x)
    while mp.mpf(f) > x:
        f = math.nextafter(f, -INF)
    while mp.mpf(math.nextafter(f, INF)) <= x:
        f = math.nextafter(f, INF)
    return f


def _tight(lo, hi, ends):   # [lo, hi] is the exact ends that ends(mp) gives, rounded out
    mp = _mp()
    a, b = ends(mp)
    return float.fromhex(lo) == _rd(mp, a) and float.fromhex(hi) == -_rd(mp, -b)


def _tightc(lo, hi, ends):
    return ((lo, hi), lambda: _tight(lo, hi, ends))


def _log2_proof():   # (1/4)^(-1/2) = 2 exactly: 2^2 = 1/(1/4)
    return Fraction(2) ** 2 == 1 / Fraction(1, 4)


_D = lambda mp: mp.acos(1 - mp.mpf(2) ** -53)   # cos d = 1 - 2^-53: how far from an extremum the set reaches
_TA = lambda mp: mp.atan(mp.mpf(float.fromhex('0x1.d02967c31cdb4p+53')))
_TB = lambda mp: mp.atan(mp.mpf(float.fromhex('0x1.d02967c31cdb5p+53')))
_SA = lambda mp: mp.atan(mp.mpf(float.fromhex('0x1.72cece675d1fcp-52')))
_SB = lambda mp: mp.atan(mp.mpf(float.fromhex('0x1.72cece675d1fdp-52')))
CORRECTIONS = {
    # cosRev [-1, -1] within [3.14, 3.15]: {pi}, so [pi down, pi up]; ITF1788 has the upper end an ulp out
    'libieeep1788_rev.itl:633': _tightc('0x1.921fb54442d18p+1', '0x1.921fb54442d19p+1', lambda mp: (mp.pi, mp.pi)),
    # cosRev [-1, -(1 - 2^-53)] near pi and -pi: [pi - d, pi + d]
    'libieeep1788_rev.itl:642': _tightc('0x1.921fb52442d18p+1', '0x1.921fb56442d19p+1',
                                        lambda mp: (mp.pi - _D(mp), mp.pi + _D(mp))),
    'libieeep1788_rev.itl:643': _tightc('-0x1.921fb56442d19p+1', '-0x1.921fb52442d18p+1',
                                        lambda mp: (-mp.pi - _D(mp), -mp.pi + _D(mp))),
    # sinRev [1 - 2^-53, 1] near pi/2: [pi/2 - d, pi/2 + d]
    'libieeep1788_rev.itl:555': _tightc('0x1.921fb50442d18p+0', '0x1.921fb58442d19p+0',
                                        lambda mp: (mp.pi / 2 - _D(mp), mp.pi / 2 + _D(mp))),
    # tanRev of a narrow C near tan's pole, within [-1.5708, 1.5708]: the pieces k = -1 and 0
    'libieeep1788_rev.itl:711': _tightc('-0x1.921fb54442d19p+0', '0x1.921fb54442d19p+0',
                                        lambda mp: (_TA(mp) - mp.pi, _TB(mp))),
    # tanRev of a narrow C near 0, within [-3.15, 3.15]: the pieces k = -1 to 1
    'libieeep1788_rev.itl:713': _tightc('-0x1.921fb54442d18p+1', '0x1.921fb54442d1ap+1',
                                        lambda mp: (_SA(mp) - mp.pi, _SB(mp) + mp.pi)),
    # pownRev [0, 2^-1074] -7: {x > 0 : x^-7 <= 2^-1074} = [2^(1074/7), inf]; ITF1788 wants the lower end ...bc
    'libieeep1788_rev.itl:276': (('0x1.588cea3f093bdp+153', 'infinity'), _root7_proof),
    'libieeep1788_rev.itl:277': (('-infinity', '-0x1.588cea3f093bdp+153'), _root7_proof),
    # powRev2 [1/4, 1/2] (and [1/4, 1]) [2, inf]: x = 1 gives 1, outside C; for x < 1, x^y >= 2 exactly when
    # y <= ln 2 / ln x, which is greatest at the least x, -1/2 at x = 1/4, and unbounded below. So [-inf, -1/2];
    # ITF1788 has [entire] and [-inf, 0] (its [2, 4] neighbours have -1/2)
    'pow_rev.itl:609': (('-infinity', '-0.5'), _log2_proof),
    'pow_rev.itl:642': (('-infinity', '-0.5'), _log2_proof),
}


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
        for m in re.finditer(r'testcase\s+(\S+)\s*\{', src):
            depth, j = 1, m.end()                         # the body to the matching brace (lists use braces too)
            while depth and j < len(src):
                depth += {'{': 1, '}': -1}.get(src[j], 0)
                j += 1
            name, body, body_start = m.group(1), src[m.end():j - 1], m.end()
            if '_dec' in name:
                continue
            pos = body_start
            for st in body.split(';'):
                lead = len(st) - len(st.lstrip())
                stl = src.count('\n', 0, pos + lead) + 1
                pos += len(st) + 1
                st = ' '.join(st.split())
                if not st:
                    continue
                m2 = re.search(r'\s+signal\s+(\w+)$', st)
                signal = m2.group(1) if m2 else ''
                if m2:
                    st = st[:m2.start()]
                lhs, sep, rhs = st.rpartition(' = ')
                if not sep:
                    lhs, rhs = st, ''
                toks = re.findall(r'"[^"]*"|\{[^}]*\}|\[[^\]]*\]|\S+', lhs)
                res = re.findall(r'\[[^\]]*\]|<=|\S+', rhs)
                yield f.name, name, toks[0], toks[1:], res, stl, signal


def number(tok):
    try:
        return nearest(tok)
    except (ValueError, ZeroDivisionError):
        return None


# 1788.1's reductions are crsum's (crsum/crsum.h): sum, sumAbs, sumSquare and dot over lists of numbers
REDUCTIONS = {'sum_nearest': 0, 'sum_abs_nearest': 1, 'sum_sqr_nearest': 2, 'dot_nearest': 3}


def main():
    itl = sys.argv[1]
    rows, skipped, odd, corrected, reds, unproved = [], {}, [], [], [], []
    for fname, case, op, args, res, line, signal in tests(itl):
        where = f'{fname}:{line} {case}'
        if op in REDUCTIONS:
            lists = [[number(v) for v in a.strip('{}').split(',')] for a in args]
            want = number(res[0]) if len(res) == 1 else None
            if want is None or any(None in l for l in lists) or len(lists) != (2 if op == 'dot_nearest' else 1):
                odd.append(f'{where}: {op} {" ".join(args)} = {" ".join(res)}')
            else:
                reds.append((REDUCTIONS[op], lists, want, where))
            continue
        if op not in OPS:
            skipped[op] = skipped.get(op, 0) + 1
            continue
        fn, kind, k = OPS[op]
        x = 0.0
        text = None
        if kind == 'X':                                   # isMember: the number first
            x = number(args[0]) if args else None
            args = args[1:]
        if kind == 'T':                                   # a string literal
            text = args[0][1:-1] if len(args) == 1 and args[0].startswith('"') else None
            x = 0.0 if text is not None else None
            args = []
        if kind == 'U':                                   # two numbers, kept in the first interval's slots
            nums2 = [number(a) for a in args]
            x = 0.0 if len(nums2) == 2 and None not in nums2 else None
            args = []
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
        key = f'{fname}:{line}'
        if key in CORRECTIONS:
            (lo_s, hi_s), proof = CORRECTIONS[key]
            try:
                proved = proof()
            except ImportError:   # mpmath absent: the test is left out, not trusted either way
                unproved.append(key)
                continue
            if not proved:
                sys.exit(f'itf1788.py: the proof for the correction at {key} fails')
            res = [f'[{lo_s}, {hi_s}]']
            corrected.append(key)
        if kind in ('T', 'U'):
            tight = interval(res[0]) if len(res) == 1 else None
            ok = ok and tight is not None and signal in SIGNALS
            acc = tight
            code = SIGNALS.get(signal, 0)
            if ok and kind == 'U':
                ivs = [tuple(nums2)]
        elif kind == 'Q':
            pair = [interval(r) for r in res]
            ok = ok and len(pair) == 2 and None not in pair
            if ok:
                tight = acc = (pair[0][0], pair[0][1], pair[1][0], pair[1][1])
        elif kind in ('I', 'P'):
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
        rows.append((fn, kind, k if kind != 'U' else 1, ivs, x, tight, acc, code, where, text))
    fns = sorted({(r[0], r[1], r[2]) for r in rows})
    idx = {f: i for i, (f, _, _) in enumerate(fns)}
    out = []
    w = out.append
    w('/* generated by ival/test/itf1788.py from ITF1788\'s itl files: do not edit */')
    w('#include <math.h>\n#include <stdio.h>\n#include <stdlib.h>\n#include <string.h>\n#include "ival.h"')
    w('struct t { int f; double a[6], x, tl, th, al, ah; int code; const char *where, *text; };')
    w('static const char *const NAME[] = { ' + ', '.join(f'"{f}"' for f, _, _ in fns) + ' };')
    w('static const char KIND[] = "' + ''.join(kd for _, kd, _ in fns) + '";')
    w('static const int ARITY[] = { ' + ', '.join(str(k) for _, _, k in fns) + ' };')
    w('static const struct t T[] = {')
    for fn, kind, k, ivs, x, tight, acc, code, where, text in rows:
        a = [v for iv in ivs for v in iv]
        a += [0.0] * (6 - len(a))
        lit = 'NULL' if text is None else '"' + text.replace('\\', '\\\\').replace('"', '\\"') + '"'
        if kind == 'Q':                                   # the pair: tl th the first, al ah the second
            acc = (tight[2], tight[3])
        w(f'  {{ {idx[fn]}, {{ {", ".join(c(v) for v in a)} }}, {c(x)}, {c(tight[0])}, {c(tight[1])}, {c(acc[0])}, '
          f'{c(acc[1])}, {code}, "{where}", {lit} }},')
    w('};')
    w('enum { NT = sizeof T / sizeof T[0], NF = sizeof NAME / sizeof NAME[0] };')
    w('static void call(int f, const double *a0, const double *a1, const double *b0, const double *b1,')
    w('                 const double *c0, const double *c1, const double *x, double *zl, double *zh, double *z2l,')
    w('                 double *z2h, const char *const *strs,')
    w('                 unsigned char *r, size_t n)\n{')
    w('  int *pw = malloc((n ? n : 1) * sizeof *pw);\n  for (size_t i = 0; i < n; i++) pw[i] = (int)x[i];')
    w('  double *ninf = malloc((n ? n : 1) * 8), *pinf = malloc((n ? n : 1) * 8);')
    w('  for (size_t i = 0; i < n; i++) { ninf[i] = -INFINITY; pinf[i] = INFINITY; }')
    w('  switch (f) {')
    for f, kind, k in fns:
        i = idx[f]
        if f in CALLS:
            w(f'    case {i}: {CALLS[f]}; break;')
            continue
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
        elif kind == 'Q':
            args = 'a0, a1, b0, b1, zl, zh, z2l, z2h'
        elif kind == 'T':
            args = 'strs, zl, zh, r'
        elif kind == 'U':
            args = 'a0, a1, zl, zh, r'
        else:
            args = ['a0, a1', 'a0, a1, b0, b1'][k - 1] + ', r'
        w(f'    case {i}: ival_{f}({args}, n); break;')
    w('  }\n  free(pw); free(ninf); free(pinf);\n  (void)b0; (void)b1; (void)c0; (void)c1; (void)x; (void)zh; (void)z2l; (void)z2h; (void)r; (void)strs;\n}')
    skip = ', '.join(f'{op} {n}' for op, n in sorted(skipped.items(), key=lambda x: (-x[1], x[0])))
    w(f'static const char SKIPPED[] = "{skip}";')
    w(f'static const int NSKIPPED = {sum(skipped.values())}, NODD = {len(odd)};')
    w(f'static const char CORRECTED[] = "{", ".join(corrected)}";')
    w(f'static const int NCORRECTED = {len(corrected)};')
    w(f'static const char UNPROVED[] = "{", ".join(unproved)}";')
    w(f'static const int NUNPROVED = {len(unproved)};')
    w('#include "crsum.h"')
    for j, (op, lists, want, where) in enumerate(reds):
        for li, l in enumerate(lists):
            w(f'static const double R{j}_{li}[] = {{ {", ".join(c(v) for v in l)} }};')
    w('struct red { int op; const double *a, *b; size_t n; double want; const char *where; };')
    w('static const struct red RED[] = {')
    for j, (op, lists, want, where) in enumerate(reds):
        b = f'R{j}_1' if len(lists) == 2 else f'R{j}_0'
        w(f'  {{ {op}, R{j}_0, {b}, {len(lists[0])}, {c(want)}, "{where}" }},')
    w('  { -1, 0, 0, 0, 0, 0 } };')
    w(f'enum {{ NRED = {len(reds)} }};')
    w(RUNNER)
    print('\n'.join(out))
    for o in odd[:20]:
        print('unread:', o, file=sys.stderr)


RUNNER = r'''
static int same(double a, double b) { return (a != a && b != b) || a == b; }
/* inf and sup must match to the sign of a zero (inf of [0, 1] is -0) */
static int samenum(double a, double b) { return same(a, b) && (a != a || signbit(a) == signbit(b)); }
/* one result against its test: 0 right, 1 only accurate (holds the tight interval, inside the accurate one), 2 wrong */
static int verdict(const struct t *t, char kind, double zl, double zh, double z2l, double z2h, int r)
{
  if (kind == 'Q') return same(zl, t->tl) && same(zh, t->th) && same(z2l, t->al) && same(z2h, t->ah) ? 0 : 2;
  if (kind == 'T' || kind == 'U') return same(zl, t->tl) && same(zh, t->th) && r == t->code ? 0 : 2;
  if (kind == 'B' || kind == 'X' || kind == 'O') return r == t->code ? 0 : 2;
  if (kind == 'Z') return samenum(zl, t->tl) ? 0 : 2;
  if (kind == 'N') return same(zl, t->tl) ? 0 : 2;
  if (kind == 'M') return same(zl, t->tl) && same(zh, t->th) ? 0 : 2;
  if (same(zl, t->tl) && same(zh, t->th)) return 0;
  return zl == zl && t->tl == t->tl && zl <= t->tl && zh >= t->th && zl >= t->al && zh <= t->ah ? 1 : 2;
}
int main(void)
{
  static double v[7][NT], zl[NT], zh[NT], z2l[NT], z2h[NT];
  static unsigned char r[NT];
  static const char *sv[NT];
  int per[NF] = { 0 }, bad = 0, accurate = 0, split = 0, shown = 0;
  for (int f = 0; f < NF; f++) {
    int n = 0, idx[NT];
    for (int i = 0; i < NT; i++)
      if (T[i].f == f) { for (int j = 0; j < 6; j++) v[j][n] = T[i].a[j]; v[6][n] = T[i].x; sv[n] = T[i].text; idx[n++] = i; }
    per[f] = n;
    call(f, v[0], v[1], v[2], v[3], v[4], v[5], v[6], zl, zh, z2l, z2h, sv, r, n);   /* all at once */
    for (int k = 0; k < n; k++) {                                      /* and one at a time */
      double a[7], sl = 0, sh = 0, s2l = 0, s2h = 0; unsigned char sr = 0;
      for (int j = 0; j < 7; j++) a[j] = v[j][k];
      call(f, &a[0], &a[1], &a[2], &a[3], &a[4], &a[5], &a[6], &sl, &sh, &s2l, &s2h, &sv[k], &sr, 1);
      const struct t *t = &T[idx[k]];
      char kind = KIND[f];
      int num = kind == 'N' || kind == 'Z';
      int once = kind == 'I' || kind == 'P' || kind == 'Q' || kind == 'T' || kind == 'U' || num || kind == 'M' ? memcmp(&sl, &zl[k], 8) || (!num && memcmp(&sh, &zh[k], 8)) : sr != r[k];
      if (kind == 'Q') once = once || memcmp(&s2l, &z2l[k], 8) || memcmp(&s2h, &z2h[k], 8);
      if (once) split++;
      int vd = verdict(t, kind, zl[k], zh[k], z2l[k], z2h[k], r[k]);
      if (!vd) continue;
      if (vd == 1) accurate++; else bad++;
      if (shown++ < 12) {
        printf("%s %s:", vd == 1 ? "NOT TIGHTEST" : "WRONG", NAME[f]);
        if (kind == 'X') printf(" %a", t->x);
        if (kind == 'T') printf(" \"%s\"", t->text);
        for (int j = 0; j < 2 * ARITY[f]; j += 2) printf(" [%a, %a]", t->a[j], t->a[j + 1]);
        if (kind == 'P') printf(" %d", (int)t->x);
        if (kind == 'B' || kind == 'X' || kind == 'O') printf(" = %d, want %d (%s)\n", r[k], t->code, t->where);
        else if (kind == 'N' || kind == 'Z') printf(" = %a, want %a (%s)\n", zl[k], t->tl, t->where);
        else if (kind == 'T' || kind == 'U') printf(" = [%a, %a] status %d, want [%a, %a] status %d (%s)\n", zl[k], zh[k], r[k], t->tl, t->th, t->code, t->where);
        else if (kind == 'Q') printf(" = [%a, %a] [%a, %a], want [%a, %a] [%a, %a] (%s)\n", zl[k], zh[k], z2l[k], z2h[k], t->tl, t->th, t->al, t->ah, t->where);
        else printf(" = [%a, %a], want [%a, %a] (%s)\n", zl[k], zh[k], t->tl, t->th, t->where);
      }
    }
  }
  /* the reductions, by crsum */
  for (int j = 0; j < NRED; j++) {
    const struct red *q = &RED[j];
    double ab[64], got;
    if (q->op == 1) { for (size_t i = 0; i < q->n; i++) ab[i] = fabs(q->a[i]); got = crsum(ab, q->n, CRSUM_NEAREST); }
    else if (q->op == 0) got = crsum(q->a, q->n, CRSUM_NEAREST);
    else got = crdot(q->a, q->b, q->n, CRSUM_NEAREST);
    if (!same(got, q->want)) { bad++; if (shown++ < 12) printf("WRONG reduction %d: %a, want %a (%s)\n", q->op, got, q->want, q->where); }
  }
  printf("ITF1788 bare tests ival runs, by operation:");
  for (int f = 0; f < NF; f++) printf(" %s %d", NAME[f], per[f]);
  printf(" reductions (crsum) %d", NRED);
  printf("\nnot in ival yet (%d): %s\n", NSKIPPED, SKIPPED);
  if (NODD) printf("statements the converter could not read: %d\n", NODD);
  if (NCORRECTED) printf("expected results corrected, each proved not the tightest (itf1788.py, CORRECTIONS): %d: %s\n", NCORRECTED, CORRECTED);
  if (NUNPROVED) printf("tests left out, their corrections' proofs needing mpmath: %d: %s\n", NUNPROVED, UNPROVED);
  if (!bad && !accurate && !split && !NODD && NT > 0)
    printf("VERDICT: IDENTICAL (%d ITF1788 tests, every result the tight one, all at once and one at a time)\n", NT + NRED);
  else
    printf("VERDICT: DIFFERS (%d wrong, %d only accurate, %d differ between all at once and one at a time, %d unread, of %d)\n",
           bad, accurate, split, NODD, NT + NRED);
  return bad || accurate || split || NODD || NT == 0;
}'''


if __name__ == '__main__':
    main()
