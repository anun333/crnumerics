"""Tests of crlin's Python binding: enclosures against the exact rational solution (fractions), singular systems
raising NotVerified, both modes, numpy's threaded BLAS, and a user's inverse. Run with libcrlin on the linker's path
or CRLIN_LIBRARY naming it: python3 -m pytest lin/python/tests"""
from fractions import Fraction
import random

import numpy as np
import pytest

import crlin

rng = np.random.default_rng(20261013)


def exact_solution(A, b):
    """Gauss-Jordan in fractions: the binary64 entries taken exactly"""
    n = len(b)
    M = [[Fraction(float(A[i, j])) for j in range(n)] + [Fraction(float(b[i]))] for i in range(n)]
    for k in range(n):
        p = next(i for i in range(k, n) if M[i][k] != 0)
        M[k], M[p] = M[p], M[k]
        for i in range(n):
            if i != k and M[i][k] != 0:
                f = M[i][k] / M[k][k]
                M[i] = [a - f * c for a, c in zip(M[i], M[k])]
    return [M[i][n] / M[i][i] for i in range(n)]


@pytest.mark.parametrize("mode", ["classical", "exact"])
def test_holds_exact_solution(mode):
    for n in (1, 2, 5, 12, 25):
        for _ in range(4):
            A = rng.standard_normal((n, n)) * 2.0 ** rng.integers(-20, 20)
            b = rng.standard_normal(n)
            lo, hi = crlin.solve(A, b, mode=mode)
            x = exact_solution(A, b)
            for i in range(n):
                assert Fraction(float(lo[i])) <= x[i] <= Fraction(float(hi[i])), (n, i)
            assert np.all((hi - lo) <= 4 * np.spacing(np.maximum(abs(lo), abs(hi))))   # an ulp or two wide


def test_singular_raises():
    A = np.ones((4, 4)); b = np.arange(4.0)
    with pytest.raises(crlin.NotVerified):
        crlin.solve(A, b)
    A = np.array([[1.0, 2.0], [2.0, 4.0]])
    with pytest.raises(crlin.NotVerified):
        crlin.solve(A, np.array([1.0, 1.0]), inverse=np.eye(2))   # numpy is not asked; crlin still refuses


def test_large_threaded():
    n = 600
    A = rng.standard_normal((n, n)); b = rng.standard_normal(n)
    lo, hi = crlin.solve(A, b)       # numpy's BLAS, with its threads
    x = np.linalg.solve(A, b)
    assert np.all(lo <= hi) and np.all(np.abs(x - (lo + hi) / 2) <= 1e-9 * np.abs(x).max())


def test_bad_inverse_never_wrong():
    n = 8
    A = rng.standard_normal((n, n)); b = rng.standard_normal(n)
    x = exact_solution(A, b)
    for s in (1e-12, 1e-3, 0.3):
        R = np.linalg.inv(A) * (1 + s * rng.standard_normal((n, n)))
        try:
            lo, hi = crlin.solve(A, b, inverse=R)
        except crlin.NotVerified:
            continue
        assert all(Fraction(float(lo[i])) <= x[i] <= Fraction(float(hi[i])) for i in range(n))


def test_version():
    assert crlin.version().count(".") == 2


@pytest.mark.parametrize("mode", ["classical", "exact"])
def test_several_right_hand_sides(mode):
    n, k = 9, 4
    A = rng.standard_normal((n, n)); B = rng.standard_normal((n, k))
    lo, hi = crlin.solve(A, B, mode=mode)
    assert lo.shape == (n, k)
    for j in range(k):
        l1, h1 = crlin.solve(A, B[:, j], mode=mode)
        assert np.array_equal(l1, lo[:, j]) and np.array_equal(h1, hi[:, j])
        x = exact_solution(A, B[:, j])
        assert all(Fraction(float(lo[i, j])) <= x[i] <= Fraction(float(hi[i, j])) for i in range(n))


def test_one_column_fails():
    A = np.eye(4) * 0.5; B = np.ones((4, 3)); B[:, 2] = 1e308   # the third column's solution overflows
    with pytest.raises(crlin.NotVerified) as e:
        crlin.solve(A, B)
    assert e.value.columns == [2] and np.all(e.value.lo[:, :2] <= 2) and np.all(e.value.hi[:, :2] >= 2)
