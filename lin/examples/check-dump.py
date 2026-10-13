"""check-dump.py FILE...: each file from cublas.c's DUMP (n, A, b, lo, hi as binary64): the exact solution of A x = b,
in fractions (Gauss-Jordan on the binary64 values taken exactly), must lie in [lo, hi] in every component; and the
control, each enclosure shrunk by one ulp at one end, must miss it somewhere. For small n (60 takes seconds).
Prints one line per file and a verdict; exit 0 when every file holds and its control differs. 2026-10-13."""
import math
import struct
import sys
from fractions import Fraction


def read(path):
    raw = open(path, "rb").read()
    v = struct.unpack("<%dd" % (len(raw) // 8), raw)
    n = int(v[0])
    A = [list(v[1 + i * n:1 + (i + 1) * n]) for i in range(n)]
    o = 1 + n * n
    return n, A, list(v[o:o + n]), list(v[o + n:o + 2 * n]), list(v[o + 2 * n:o + 3 * n])


def exact(n, A, b):
    M = [[Fraction(a) for a in row] + [Fraction(c)] for row, c in zip(A, b)]
    for k in range(n):
        p = max(range(k, n), key=lambda i: abs(M[i][k]))
        M[k], M[p] = M[p], M[k]
        for i in range(n):
            if i != k and M[i][k]:
                f = M[i][k] / M[k][k]
                M[i] = [a - f * c for a, c in zip(M[i], M[k])]
    return [M[i][n] / M[i][i] for i in range(n)]


ok = True
for path in sys.argv[1:]:
    n, A, b, lo, hi = read(path)
    x = exact(n, A, b)
    miss = sum(not (Fraction(lo[i]) <= x[i] <= Fraction(hi[i])) for i in range(n))
    shrunk = sum(not (Fraction(math.nextafter(lo[i], math.inf)) <= x[i] <= Fraction(hi[i])) or
                 not (Fraction(lo[i]) <= x[i] <= Fraction(math.nextafter(hi[i], -math.inf))) for i in range(n))
    print(f"{path}: n = {n}, {miss} components outside (must be 0); control: {shrunk} miss when shrunk by an ulp")
    ok &= miss == 0 and shrunk > 0
print("VERDICT: IDENTICAL every enclosure holds the exact solution" if ok else "VERDICT: DIFFERS")
sys.exit(0 if ok else 1)
