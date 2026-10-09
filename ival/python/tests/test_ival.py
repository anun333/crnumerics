"""Tests of ival's Python binding: results tight against mpmath (300 bits) and exact fractions, arrays elementwise
the same as one interval at a time, the constructors, empty intervals, the reverse operations and the accurate mode.
Run with libival on the linker's path or IVAL_LIBRARY naming it: python3 -m pytest ival/python/tests"""
import math
import random
from fractions import Fraction

import mpmath
import numpy as np
import pytest

import ival

mpmath.mp.prec = 300
rng = random.Random(20261009)


def adjacent(lo, hi):
    """lo and hi the same double or neighbours: the tightest interval around a value not a double"""
    return lo == hi or math.nextafter(lo, math.inf) == hi


def brackets(lo, hi, exact):
    return mpmath.mpf(lo) <= exact <= mpmath.mpf(hi)


def test_version():
    assert isinstance(ival.version(), str) and ival.version().count(".") == 2


FUNCS = [("exp", mpmath.exp, lambda: rng.uniform(-700, 700)), ("log", mpmath.log, lambda: math.exp(rng.uniform(-700, 700))),
         ("sin", mpmath.sin, lambda: rng.uniform(-1e6, 1e6)), ("cos", mpmath.cos, lambda: rng.uniform(-1e6, 1e6)),
         ("tan", mpmath.tan, lambda: rng.uniform(-100, 100)), ("atan", mpmath.atan, lambda: rng.uniform(-1e3, 1e3)),
         ("sinh", mpmath.sinh, lambda: rng.uniform(-700, 700)), ("tanh", mpmath.tanh, lambda: rng.uniform(-20, 20)),
         ("cbrt", lambda x: mpmath.sign(x) * mpmath.cbrt(abs(x)), lambda: rng.uniform(-1e9, 1e9)),   # the real root
         ("exp2", lambda x: mpmath.power(2, x), lambda: rng.uniform(-1000, 1000)),
         ("log1p", mpmath.log1p, lambda: rng.uniform(-0.999, 1e6)), ("erf", mpmath.erf, lambda: rng.uniform(-6, 6)),
         ("asinh", mpmath.asinh, lambda: rng.uniform(-1e6, 1e6)), ("sqrt", mpmath.sqrt, lambda: rng.uniform(0, 1e300))]


@pytest.mark.parametrize("name,ref,gen", FUNCS, ids=[f[0] for f in FUNCS])
def test_points_tight(name, ref, gen):
    """f of a point [x, x]: the exact f(x) inside, the ends adjacent (each the exact value rounded down and up)"""
    xs = np.array([gen() for _ in range(300)])
    r = getattr(ival, name)(ival.Interval(xs))
    for x, lo, hi in zip(xs, r.lo, r.hi):
        exact = ref(mpmath.mpf(float(x)))
        assert brackets(lo, hi, exact), (name, x, lo, hi)
        assert adjacent(lo, hi), (name, x, lo, hi)


def test_arith_exact():
    """the four operations against exact fractions: inside, and the ends adjacent"""
    for _ in range(2000):
        a, b = rng.uniform(-1e6, 1e6), rng.uniform(-1e6, 1e6)
        for op, ex in (("__add__", Fraction(a) + Fraction(b)), ("__sub__", Fraction(a) - Fraction(b)),
                       ("__mul__", Fraction(a) * Fraction(b)), ("__truediv__", Fraction(a) / Fraction(b))):
            r = getattr(ival.Interval(a), op)(b)
            assert Fraction(float(r.lo)) <= ex <= Fraction(float(r.hi)), (op, a, b)
            assert adjacent(float(r.lo), float(r.hi)), (op, a, b)


def test_point_one_plus_point_two():
    r = ival.Interval(0.1) + 0.2
    assert float(r.lo) == 0.3 and float(r.hi) == math.nextafter(0.3, 1)


def test_text_rounds_outward():
    t = ival.Interval.from_text("[0.1, 0.2]")
    assert Fraction(float(t.lo)) < Fraction("0.1") and adjacent(float(t.lo), 0.1)
    assert Fraction(float(t.hi)) > Fraction("0.2") or float(t.hi) == 0.2
    assert Fraction("0.2") <= Fraction(float(t.hi)) and adjacent(0.2, float(t.hi)) or float(t.hi) == 0.2
    e = ival.Interval.from_text(["[empty]", "[entire]", "[1, 2]"])
    assert e.shape == (3,) and e.is_empty()[0] and e.is_entire()[1] and float(e.lo[2]) == 1.0


def test_arrays_match_one_at_a_time():
    lo = np.array([rng.uniform(-10, 10) for _ in range(500)])
    hi = lo + np.array([rng.uniform(0, 3) for _ in range(500)])
    X = ival.Interval(lo, hi)
    for f in (ival.sin, ival.exp, ival.atan, ival.sqr, ival.cosh):
        R = f(X)
        for k in range(0, 500, 37):
            r = f(ival.Interval(lo[k], hi[k]))
            assert (float(r.lo), float(r.hi)) == (R.lo[k], R.hi[k])
    Y = X * 2 + 1                                          # broadcasting a number
    assert Y.shape == (500,)
    Z = X + ival.Interval(lo[:1], hi[:1])                  # a one-interval array broadcast
    assert Z.shape == (500,)


def test_empty():
    e = ival.Interval(2, 1)
    assert e.is_empty()
    for r in (e + 1, ival.exp(e), e * ival.Interval(1, 2), ival.sqrt(ival.Interval(-3, -1))):
        assert r.is_empty()
    assert not ival.Interval(0, 1).is_empty()


def test_domain_and_poles():
    assert float(ival.log(ival.Interval(-1, 1)).lo) == -math.inf
    t = ival.tan(ival.Interval(1, 2))                      # across pi/2
    assert float(t.lo) == -math.inf and float(t.hi) == math.inf
    assert float((ival.Interval(-2, 3) ** 2).lo) == 0.0     # an even power straddling 0


def test_pown_tight():
    for _ in range(300):
        x, p = rng.uniform(-50, 50), rng.choice([2, 3, 4, 5, 7, 11, -1, -2, -3])
        r = ival.pown(ival.Interval(x), p)
        exact = mpmath.power(mpmath.mpf(x), p)
        assert brackets(float(r.lo), float(r.hi), exact) and adjacent(float(r.lo), float(r.hi)), (x, p)


def test_reverse():
    r = ival.sqrrev(ival.Interval(4, 9))
    assert (float(r.lo), float(r.hi)) == (-3.0, 3.0)
    r = ival.sqrrev(ival.Interval(4, 9), ival.Interval(0, 10))
    assert (float(r.lo), float(r.hi)) == (2.0, 3.0)
    r = ival.powrev2(ival.Interval(0.25, 0.5), ival.Interval(2, math.inf))   # ITF1788.jl#8's case
    assert (float(r.lo), float(r.hi)) == (-math.inf, -0.5)
    r = ival.mulrev(ival.Interval(2, 4), ival.Interval(8, 8))
    assert (float(r.lo), float(r.hi)) == (2.0, 4.0)


def test_accurate_mode():
    """one ulp at most outside the tight result (equal to it without crmvec)"""
    xs = np.array([rng.uniform(-50, 50) for _ in range(400)])
    X = ival.Interval(xs, xs + 0.5)
    for name in ("exp", "sin", "log1p", "atan"):
        T, A = getattr(ival, name)(X), getattr(ival, "acc_" + name)(X)
        ok = ~np.isnan(T.lo)
        assert np.all(A.lo[ok] <= T.lo[ok]) and np.all(A.hi[ok] >= T.hi[ok])
        assert np.all(np.nextafter(T.lo[ok], -np.inf) <= A.lo[ok]) and np.all(A.hi[ok] <= np.nextafter(T.hi[ok], np.inf))


def test_measures_and_membership():
    x = ival.Interval(1, 3)
    assert x.mid == 2.0 and x.wid == 2.0 and x.rad == 1.0 and x.mag == 3.0 and x.mig == 1.0
    assert 2.5 in x and 3.5 not in x
    assert ival.Interval(1.5, 2).subset(x) and not x.subset(ival.Interval(1.5, 2))
    assert float((x & ival.Interval(2, 5)).lo) == 2.0 and float((x | ival.Interval(5, 6)).hi) == 6.0


def _ok(r, exact):   # the interval r contains the exact number (a Fraction), and its ends are adjacent doubles
    return Fraction(float(r.lo)) <= exact <= Fraction(float(r.hi)) and adjacent(float(r.lo), float(r.hi))


def test_numbers_round_outward():
    """a number that is not a double becomes the two doubles around it, not the nearest one"""
    from decimal import Decimal
    for v in (2**53 + 1, -(2**60) - 3, 10**20 + 7, Decimal("0.1"), Fraction(1, 3)):
        assert _ok(ival.Interval(v), Fraction(v)), v
    r = ival.Interval(Fraction(1, 3), 2**53 + 1)
    assert Fraction(float(r.lo)) <= Fraction(1, 3) and Fraction(float(r.hi)) >= 2**53 + 1
    assert ival.Interval(2**53 + 3, 2**53 + 1).is_empty()      # lo > hi though both round to 2^53 + 2
    big = ival.Interval(10**400)                                # beyond the largest double
    assert float(big.lo) == np.finfo(np.float64).max and float(big.hi) == math.inf
    a = np.array([2**53 + 1, 2**62 + 1, -(2**63) + 1, 5], dtype=np.int64)
    r = ival.Interval(a)
    for k, v in enumerate(a.tolist()):
        assert Fraction(float(r.lo[k])) <= v <= Fraction(float(r.hi[k])), v
    if np.finfo(np.longdouble).nmant > 52:                      # x86's 80-bit long double
        x = np.longdouble(1) / 3
        assert _ok(ival.Interval(x), Fraction(*x.as_integer_ratio()))
        r = ival.Interval(np.array([x, -x]))
        assert _ok(r[1], -Fraction(*x.as_integer_ratio()))
    assert (float(ival.Interval(0.5).lo), float(ival.Interval(7).hi)) == (0.5, 7.0)   # exact numbers stay points
    r = ival.Interval(1, 2) + Fraction(1, 3)                                          # an operand too
    assert Fraction(r.lo) <= Fraction(4, 3) and Fraction(r.hi) >= Fraction(7, 3)
    assert ival.contains(ival.Interval(0, 2**53), 2**53 + 1) is False and (2**53 + 1) in ival.Interval(2**53 + 1)
    assert not ival.contains(ival.Interval(0, 1), np.array([2**60 + 1])).any()


def test_one_at_a_time_matches_arrays():
    """an Interval of one interval (Python floats) gives the bounds of the array path, through every kind of call"""
    xs = [rng.uniform(-3, 3) for _ in range(200)]
    X = ival.Interval(np.array(xs), np.array(xs) + 0.25)
    one = [ival.Interval(x, x + 0.25) for x in xs]
    assert all(type(o.lo) is float for o in one)
    calls = [(lambda A: ival.exp(A)), (lambda A: A * A - 1 / A), (lambda A: ival.pown(A, 3)), (lambda A: A ** -2),
             (lambda A: ival.atan2(A, A + 1)), (lambda A: ival.sqrrev(A, ival.Interval(0, 4))),
             (lambda A: ival.fma(A, A, 1)), (lambda A: ival.powrev1(ival.Interval(2, 3), A + 4)),
             (lambda A: ival.rootn(A, 3)), (lambda A: A & ival.Interval(0, 1))]
    for f in calls:
        R = f(X)
        for k in range(0, 200, 7):
            r = f(one[k])
            assert type(r.lo) is float
            assert np.array_equal([r.lo, r.hi], [R.lo[k], R.hi[k]], equal_nan=True), k
    assert [o.mid for o in one[:5]] == list(X.mid[:5]) and [o.is_empty() for o in one[:5]] == [False] * 5
    assert ival.Interval(1, 2).subset(ival.Interval(0, 3)) is True and ival.contains(ival.Interval(1, 2), 1.5) is True
    assert type(X[3].lo) is float and X[3].equal(one[3])
