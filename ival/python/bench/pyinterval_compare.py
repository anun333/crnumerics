"""ival's binding against pyinterval (CRlibm), the other correctly rounded interval library for Python, on the same
inputs: whether each result is tight (the exact value, from mpmath at 300 bits, inside; the two ends adjacent
doubles), whether the two give the same bounds, and the time per interval: one at a time, and as one NumPy array.

pyinterval 1.2.0 (2017) builds only from source, and its dependency crlibm 1.0.3 imports distutils.command.upload,
which Python 3.12 lacks and setuptools 75 dropped. It builds with setuptools 74 and no build isolation:
    python3 -m venv v && v/bin/pip install "setuptools==74.1.3" numpy mpmath
    v/bin/pip install --no-build-isolation crlibm pyinterval
    IVAL_LIBRARY=.../libival.so PYTHONPATH=ival/python/src v/bin/python ival/python/bench/pyinterval_compare.py
"""
import math
import platform
import random
import sys
import time

import mpmath
import numpy as np
from interval import imath, interval

import ival

mpmath.mp.prec = 300
rng = random.Random(1009)


def adjacent(lo, hi):
    return lo == hi or math.nextafter(lo, math.inf) == hi


def ulps_out(lo, hi, tlo, thi):   # how many doubles (lo, hi) reaches beyond the tight (tlo, thi)
    n = 0
    while lo < tlo:
        tlo, n = math.nextafter(tlo, -math.inf), n + 1
    m = 0
    while hi > thi:
        thi, m = math.nextafter(thi, math.inf), m + 1
    return max(n, m)


CASES = [("exp", mpmath.exp, lambda: rng.uniform(-700, 700)), ("log", mpmath.log, lambda: math.exp(rng.uniform(-700, 700))),
         ("sin", mpmath.sin, lambda: rng.uniform(-1e6, 1e6)), ("cos", mpmath.cos, lambda: rng.uniform(-1e6, 1e6)),
         ("tan", mpmath.tan, lambda: rng.uniform(-100, 100)), ("atan", mpmath.atan, lambda: rng.uniform(-1e3, 1e3)),
         ("sqrt", mpmath.sqrt, lambda: rng.uniform(0, 1e300)), ("sinh", mpmath.sinh, lambda: rng.uniform(-700, 700)),
         ("cosh", mpmath.cosh, lambda: rng.uniform(-700, 700)), ("tanh", mpmath.tanh, lambda: rng.uniform(-20, 20)),
         ("expm1", mpmath.expm1, lambda: rng.uniform(-30, 700)), ("log1p", mpmath.log1p, lambda: rng.uniform(-0.999, 1e6))]
N = 2000
def cpu():
    try:
        with open("/proc/cpuinfo") as f:
            return next(l.split(":", 1)[1].strip() for l in f if l.startswith(("model name", "CPU part")))
    except (OSError, StopIteration):
        return platform.processor() or "?"


print(f"# {platform.machine()} {cpu()}, python {platform.python_version()}, numpy {np.__version__}, "
      f"ival {ival.version()}; {N} points per function")
print("function\tpyinterval not tight\tival not tight\tbounds differ\tpyinterval's most ulps beyond tight")
for name, ref, gen in CASES:
    xs = [gen() for _ in range(N)]
    P = [getattr(imath, name)(interval[x])[0] for x in xs]
    I = getattr(ival, name)(ival.Interval(np.array(xs)))
    bad_p = bad_i = diff = worst = 0
    for x, (pl, ph), il, ih in zip(xs, P, I.lo, I.hi):
        e = ref(mpmath.mpf(x))
        bad_p += not (mpmath.mpf(pl) <= e <= mpmath.mpf(ph) and adjacent(pl, ph))
        bad_i += not (mpmath.mpf(il) <= e <= mpmath.mpf(ih) and adjacent(il, ih))
        diff += (pl, ph) != (il, ih)
        worst = max(worst, ulps_out(pl, ph, il, ih))
    print(f"{name}\t{bad_p}\t{bad_i}\t{diff}\t{worst}")

xs = [rng.uniform(-10, 10) for _ in range(20000)]


def best(f, rep=5):   # the least of rep timed passes after one untimed
    f()
    t = min(_timed(f) for _ in range(rep))
    if t <= 0:
        print("VOID: a timing of zero")
        sys.exit(1)
    return t


def _timed(f):
    t0 = time.perf_counter()
    f()
    return time.perf_counter() - t0


lo = np.array(xs)
print("\noperation\tpyinterval ns\tival one at a time ns\tival array of 20000 ns   (per interval)")
for name in ("exp", "sin", "atan"):
    pf, f = getattr(imath, name), getattr(ival, name)
    X = ival.Interval(lo, lo + 0.5)
    tp = best(lambda: [pf(interval[x, x + 0.5]) for x in xs])
    ts = best(lambda: [f(ival.Interval(x, x + 0.5)) for x in xs])
    ta = best(lambda: f(X), 20)
    print(f"{name}\t{tp / len(xs) * 1e9:.0f}\t{ts / len(xs) * 1e9:.0f}\t{ta / len(xs) * 1e9:.1f}")
A, B = ival.Interval(lo, lo + 0.5), ival.Interval(lo - 1, lo)
tp = best(lambda: [interval[x, x + 0.5] * interval[x - 1, x] + 2.0 for x in xs])
ts = best(lambda: [ival.Interval(x, x + 0.5) * ival.Interval(x - 1, x) + 2.0 for x in xs])
ta = best(lambda: A * B + 2.0, 20)
print(f"a*b+2\t{tp / len(xs) * 1e9:.0f}\t{ts / len(xs) * 1e9:.0f}\t{ta / len(xs) * 1e9:.1f}")
with open("/proc/loadavg") as fh:
    print("# load", fh.read().strip())
