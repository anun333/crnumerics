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


def pd_exact(A, t=0.0):
    """A - t I positive definite, exactly: symmetric elimination in fractions, every pivot positive"""
    n = A.shape[0]
    M = [[Fraction(float(A[i, j])) - (Fraction(t) if i == j else 0) for j in range(n)] for i in range(n)]
    for k in range(n):
        if M[k][k] <= 0:
            return False
        for i in range(k + 1, n):
            f = M[i][k] / M[k][k]
            M[i] = [a - f * c for a, c in zip(M[i], M[k])]
    return True


@pytest.mark.parametrize("mode", ["classical", "exact"])
def test_spd(mode):
    for n in (1, 3, 8, 20):
        B = rng.standard_normal((n + 2, n))
        A = B.T @ B
        A = (A + A.T) / 2                                   # symmetric to the bit
        lam = crlin.spd(A, mode=mode)
        assert lam > 0 and pd_exact(A, lam), n
        assert lam >= np.linalg.eigvalsh(A)[0] / 8, n       # near the smallest (numpy's, unverified, as a guide)
    # a factor from LAPACK, shifted
    A = rng.standard_normal((60, 60)); A = A @ A.T + np.eye(60); A = (A + A.T) / 2
    lmin = np.linalg.eigvalsh(A)[0]
    lam = crlin.spd(A, mode=mode, factor=(np.linalg.cholesky(A - 0.9 * lmin * np.eye(60)), 0.9 * lmin))
    assert 0.85 * lmin <= lam <= lmin and pd_exact(A, lam)


@pytest.mark.parametrize("mode", ["classical", "exact"])
def test_spd_refuses(mode):
    B = rng.integers(-3, 4, (4, 6)).astype(float)
    with pytest.raises(crlin.NotVerified) as e:              # singular: rank 4 of 6
        crlin.spd(B.T @ B, mode=mode)
    assert not e.value.lambda_lo > 0
    with pytest.raises(crlin.NotVerified):                   # indefinite, its diagonal positive
        crlin.spd(np.array([[1.0, 2.0], [2.0, 1.0]]), mode=mode)
    with pytest.raises(ValueError):                          # not symmetric
        crlin.spd(np.array([[2.0, 1.0], [1.5, 2.0]]), mode=mode)
    with pytest.raises(crlin.NotVerified):                   # a factor that is not one
        A = np.eye(5) * 2
        crlin.spd(A, mode=mode, factor=(rng.standard_normal((5, 5)), 0.0))


@pytest.mark.parametrize("mode", ["classical", "exact"])
def test_solve_interval(mode):
    import itertools
    for n in (1, 2):
        for _ in range(5):
            Am = rng.standard_normal((n, n)) + 3 * np.eye(n); bm = rng.standard_normal(n)
            rA = abs(Am) * 2.0 ** -20; rb = abs(bm) * 2.0 ** -20
            lo, hi = crlin.solve_interval(Am - rA, Am + rA, bm - rb, bm + rb, mode=mode)
            ends = [(Am - rA)[i, j] for i in range(n) for j in range(n)], [(Am + rA)[i, j] for i in range(n) for j in range(n)]
            for v in itertools.product((0, 1), repeat=n * n + n):   # every vertex, solved in fractions
                A = np.array([ends[v[q]][q] for q in range(n * n)]).reshape(n, n)
                b = np.array([(bm - rb, bm + rb)[v[n * n + i]][i] for i in range(n)])
                x = exact_solution(A, b)
                for i in range(n):
                    assert Fraction(float(lo[i])) <= x[i] <= Fraction(float(hi[i]))
    # holding a singular matrix: [[1, [0.5, 2]], [1, 1]] holds a12 = 1
    with pytest.raises(crlin.NotVerified):
        crlin.solve_interval([[1, 0.5], [1, 1]], [[1, 2], [1, 1]], [1, 1], [1, 1], mode=mode)
    with pytest.raises(ValueError):
        crlin.solve_interval([[2.0]], [[1.0]], [1.0], [1.0], mode=mode)


def test_extended():
    # Hilbert matrices, rounded to binary64: past n = 12 only the extended mode proves them
    for n in (14, 16, 20):
        A = np.array([[1.0 / (i + j + 1) for j in range(n)] for i in range(n)]); b = np.ones(n)
        with pytest.raises(crlin.NotVerified):
            crlin.solve(A, b, mode="exact")
        lo, hi = crlin.solve(A, b, mode="extended")
        x = exact_solution(A, b)
        for i in range(n):
            assert Fraction(float(lo[i])) <= x[i] <= Fraction(float(hi[i]))
        assert np.all((hi - lo) <= 4 * np.spacing(np.maximum(abs(lo), abs(hi))))
    # determinant 1, condition number 1.6e25 (exactly, from the integer inverse): A^30, Fibonacci-like
    A = np.array([[1.0, 1.0], [1.0, 2.0]])
    M = np.eye(2)
    for _ in range(30):
        M = M @ A                                            # integers below 2^53, exact
    lo, hi = crlin.solve(M, np.array([1.0, 0.0]), mode="extended")
    x = exact_solution(M, np.array([1.0, 0.0]))
    assert all(Fraction(float(lo[i])) <= x[i] <= Fraction(float(hi[i])) for i in range(2))
    with pytest.raises(ValueError):
        crlin.spd(np.eye(2), mode="extended")


def lstsq_exact(A, b):
    """the least-squares (m >= n) or minimum-norm (m < n) solution, in fractions"""
    m, n = A.shape
    Aq = [[Fraction(float(A[i, j])) for j in range(n)] for i in range(m)]
    bq = [Fraction(float(v)) for v in b]
    if m >= n:
        G = [[sum(Aq[l][i] * Aq[l][j] for l in range(m)) for j in range(n)] for i in range(n)]
        c = [sum(Aq[l][i] * bq[l] for l in range(m)) for i in range(n)]
        return solve_fractions(G, c)
    G = [[sum(Aq[i][l] * Aq[j][l] for l in range(n)) for j in range(m)] for i in range(m)]
    y = solve_fractions(G, bq)
    return [sum(Aq[i][j] * y[i] for i in range(m)) for j in range(n)]


def solve_fractions(G, c):
    k = len(c)
    M = [row[:] + [c[i]] for i, row in enumerate(G)]
    for col in range(k):
        p = next(i for i in range(col, k) if M[i][col] != 0)
        M[col], M[p] = M[p], M[col]
        for i in range(k):
            if i != col and M[i][col] != 0:
                f = M[i][col] / M[col][col]
                M[i] = [a - f * q for a, q in zip(M[i], M[col])]
    return [M[i][k] / M[i][i] for i in range(k)]


@pytest.mark.parametrize("mode", ["classical", "exact", "extended"])
def test_lstsq(mode):
    for m, n in ((6, 3), (10, 4), (3, 5), (5, 5)):
        A = rng.standard_normal((m, n)); b = rng.standard_normal(m)
        lo, hi = crlin.lstsq(A, b, mode=mode)
        x = lstsq_exact(A, b)
        for j in range(n):
            assert Fraction(float(lo[j])) <= x[j] <= Fraction(float(hi[j]))
        assert np.all((hi - lo) <= 4 * np.spacing(np.maximum(abs(lo), abs(hi))))
    A = rng.standard_normal((7, 3)); A[:, 2] = A[:, 0]          # rank-deficient
    with pytest.raises(crlin.NotVerified):
        crlin.lstsq(A, np.ones(7), mode=mode)


def test_lstsq_polynomial_fit():
    # a degree-9 fit on [0, 1] at 30 points: the Vandermonde matrix's condition number is 3.5e6 (numpy's), its normal
    # equations' about 1e13
    t = np.linspace(0, 1, 30); A = np.vander(t, 10); b = np.cos(3 * t)
    lo, hi = crlin.lstsq(A, b)
    x = lstsq_exact(A, b)
    assert all(Fraction(float(lo[j])) <= x[j] <= Fraction(float(hi[j])) for j in range(10))
    assert np.all((hi - lo) <= 4 * np.spacing(np.maximum(abs(lo), abs(hi))))


def below_exact(A, t):
    """the number of eigenvalues of A below t: negative pivots of A - t I in fractions (None at a zero pivot)"""
    n = A.shape[0]
    M = [[Fraction(float(A[i, j])) - (Fraction(t) if i == j else 0) for j in range(n)] for i in range(n)]
    neg = 0
    for k in range(n):
        if M[k][k] == 0:
            return None
        neg += M[k][k] < 0
        for i in range(k + 1, n):
            f = M[i][k] / M[k][k]
            M[i] = [a - f * q for a, q in zip(M[i], M[k])]
    return neg


@pytest.mark.parametrize("mode", ["classical", "exact"])
def test_eig_sym(mode):
    for n in (1, 4, 9):
        B = rng.standard_normal((n, n)); A = B + B.T
        lo, hi = crlin.eig_sym(A, mode=mode)
        for k in range(n):
            b1, b2 = below_exact(A, lo[k]), below_exact(-A, -hi[k])
            assert b1 is not None and b2 is not None and b1 <= k and b2 <= n - 1 - k
        assert np.all(hi - lo <= 1e-12 * np.abs(A).max() * n)
    A = rng.standard_normal((200, 200)); A = A + A.T          # larger: against numpy's (unverified) as a guide
    lo, hi = crlin.eig_sym(A, mode=mode)
    w = np.linalg.eigvalsh(A)
    assert np.all(lo <= w) and np.all(w <= hi) and np.all(hi - lo <= 1e-10 * np.abs(w).max())
    with pytest.raises(ValueError):
        crlin.eig_sym(np.array([[1.0, 2.0], [2.5, 1.0]]), mode=mode)
