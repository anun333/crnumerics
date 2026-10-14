"""crlin: verified linear systems over NumPy arrays, from crnumerics' C library libcrlin. solve(A, b) returns two
arrays lo and hi with lo <= x* <= hi for the exact solution x* of A x = b (the binary64 values of A and b taken
exactly), or raises NotVerified when the proof does not go through (A singular, or too ill-conditioned for
binary64). It never returns an enclosure that misses x*.

    import numpy as np, crlin
    A = np.random.default_rng(1).standard_normal((500, 500)); b = np.ones(500)
    lo, hi = crlin.solve(A, b)            # each [lo[i], hi[i]] holds the exact x*_i, typically an ulp wide
    lam = crlin.spd(A.T @ A)              # A^T A proved positive definite, every eigenvalue >= lam > 0

The approximate inverse comes from numpy.linalg.inv and the products from numpy's matmul, threaded or not: every
bound is computed from exact sums or with a priori error bounds in round-to-nearest, never by switching the
rounding mode, so a threaded BLAS cannot break it. mode="classical" (the default) bounds I - R A from two products;
mode="exact" from the exact product (slower, and asks nothing of the BLAS but binary64 arithmetic); mode="extended"
carries R and the solution as pairs of doubles, for systems too ill-conditioned for the others (condition numbers
to about 1e30; several times slower than "exact").

The library is libcrlin.so.0: found by the environment variable CRLIN_LIBRARY, then a copy inside this package,
then the system's linker paths.
"""
import ctypes
import ctypes.util
import os

import numpy as np

__all__ = ["solve", "solve_interval", "lstsq", "eig_sym", "spd", "solve_banded", "solve_sparse", "NotVerified", "version",
           "library_path"]


class NotVerified(ArithmeticError):
    """The system could not be proved: A singular, or too ill-conditioned for binary64. With several right-hand
    sides, columns lists those that failed, and lo and hi hold the enclosures of the others (and -inf, +inf). From
    spd, lambda_lo is the lower bound that could be proved (not positive, or -inf)."""

    def __init__(self, msg, columns=None, lo=None, hi=None, lambda_lo=None):
        super().__init__(msg)
        self.columns, self.lo, self.hi, self.lambda_lo = columns, lo, hi, lambda_lo


def _find():
    env = os.environ.get("CRLIN_LIBRARY")
    if env:
        return env
    for name in ("libcrlin.so.0", "libcrlin.0.dylib"):
        here = os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
        if os.path.exists(here):
            return here
    for name in ("libcrlin.so.0", "libcrlin.so", "libcrlin.0.dylib", "libcrlin.dylib"):
        try:
            ctypes.CDLL(name)
            return name
        except OSError:
            pass
    found = ctypes.util.find_library("crlin")
    if found:
        return found
    raise OSError("crlin: libcrlin.so not found; build crnumerics' libcrlin or set CRLIN_LIBRARY")


library_path = _find()
_lib = ctypes.CDLL(library_path)
_P = ctypes.POINTER(ctypes.c_double)
_GEMM = ctypes.CFUNCTYPE(None, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, _P, ctypes.c_size_t, _P,
                         ctypes.c_size_t, _P, ctypes.c_size_t, ctypes.c_void_p)
_lib.crlin_solve.argtypes = [ctypes.c_size_t, _P, ctypes.c_size_t, _P, _P, ctypes.c_size_t, _P, _P, ctypes.c_int,
                             _GEMM, ctypes.c_void_p]
_lib.crlin_solve.restype = ctypes.c_int
_lib.crlin_solve_many.argtypes = [ctypes.c_size_t, ctypes.c_size_t, _P, ctypes.c_size_t, _P, ctypes.c_size_t, _P,
                                  ctypes.c_size_t, _P, _P, ctypes.c_size_t, ctypes.c_int, _GEMM, ctypes.c_void_p]
_lib.crlin_solve_many.restype = ctypes.c_int
_lib.crlin_solve_interval.argtypes = [ctypes.c_size_t, ctypes.c_size_t, _P, _P, ctypes.c_size_t, _P, _P, ctypes.c_size_t,
                                      _P, ctypes.c_size_t, _P, _P, ctypes.c_size_t, ctypes.c_int, _GEMM, ctypes.c_void_p]
_lib.crlin_solve_interval.restype = ctypes.c_int
_lib.crlin_lstsq.argtypes = [ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, _P, ctypes.c_size_t, _P, ctypes.c_size_t,
                             ctypes.c_double, _P, ctypes.c_size_t, _P, _P, ctypes.c_size_t, ctypes.c_int, _GEMM,
                             ctypes.c_void_p]
_lib.crlin_lstsq.restype = ctypes.c_int
_lib.crlin_eig_sym.argtypes = [ctypes.c_size_t, _P, ctypes.c_size_t, _P, ctypes.c_size_t, _P, _P, _P, ctypes.c_int, _GEMM,
                               ctypes.c_void_p]
_lib.crlin_eig_sym.restype = ctypes.c_int
_SZ = ctypes.POINTER(ctypes.c_size_t)
_lib.crlin_solve_band.argtypes = [ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, _P, ctypes.c_size_t, _P, _P, _P,
                                  ctypes.c_int]
_lib.crlin_solve_band.restype = ctypes.c_int
_lib.crlin_solve_sparse.argtypes = [ctypes.c_size_t, _SZ, _SZ, _P, _P, _P, _P, ctypes.c_int]
_lib.crlin_solve_sparse.restype = ctypes.c_int
_lib.crlin_spd.argtypes = [ctypes.c_size_t, _P, ctypes.c_size_t, _P, ctypes.c_size_t, ctypes.c_double,
                           ctypes.POINTER(ctypes.c_double), ctypes.c_int, _GEMM, ctypes.c_void_p]
_lib.crlin_spd.restype = ctypes.c_int
_lib.crlin_version.restype = ctypes.c_char_p
_MODES = {"exact": 0, "classical": 1, "extended": 2}


@_GEMM
def _numpy_gemm(m, n, k, A, lda, B, ldb, C, ldc, ctx):
    a = np.ctypeslib.as_array(A, shape=(m, lda))[:, :k]
    b = np.ctypeslib.as_array(B, shape=(k, ldb))[:, :n]
    c = np.ctypeslib.as_array(C, shape=(m, ldc))[:, :n]
    np.matmul(a, b, out=c)


def version():
    """The C library's version."""
    return _lib.crlin_version().decode()


def solve(A, b, mode="classical", inverse=None):
    """lo, hi with lo <= x* <= hi componentwise for the exact solution of A x = b (A square; b a vector, or a matrix
    of several right-hand sides, n by k, solved together), or NotVerified. inverse: an approximate inverse of A to
    use instead of numpy.linalg.inv(A); the proof holds for any, only its success depends on it being close."""
    A = np.require(A, dtype=np.float64, requirements="C")
    b = np.require(b, dtype=np.float64, requirements="C")
    n = A.shape[0]
    if A.ndim != 2 or A.shape != (n, n) or b.shape[:1] != (n,) or b.ndim > 2:
        raise ValueError("crlin.solve: A must be n by n and b of length n, or n by k")
    if mode not in _MODES:
        raise ValueError(f"crlin.solve: mode is 'classical', 'exact' or 'extended', not {mode!r}")
    if not (np.isfinite(A).all() and np.isfinite(b).all()):
        raise ValueError("crlin.solve: A and b must be finite")
    if inverse is None:
        try:
            inverse = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            if mode != "extended":
                raise NotVerified("crlin.solve: numpy found A singular") from None
            # extended: Rump's iteration starts from crlin's own inverse, which may still get through
    R = None if inverse is None else np.require(inverse, dtype=np.float64, requirements="C")
    k = 1 if b.ndim == 1 else b.shape[1]
    lo = np.empty(b.shape)
    hi = np.empty(b.shape)
    p = lambda a: a.ctypes.data_as(_P)
    s = _lib.crlin_solve_many(n, k, p(A), n, p(b), k, None if R is None else p(R), n, p(lo), p(hi), k, _MODES[mode],
                              _numpy_gemm, None)
    if s != 0:
        bad = [j for j in range(k) if np.isinf((lo if b.ndim == 2 else lo[:, None])[:, j]).all()]
        raise NotVerified("crlin.solve: not proved (A singular, or too ill-conditioned for binary64)" +
                          (f"; columns {bad}" if b.ndim == 2 else ""), bad if b.ndim == 2 else None, lo, hi)
    return lo, hi


def solve_interval(A_lo, A_hi, b_lo, b_hi, mode="classical", inverse=None):
    """lo, hi enclosing the solution of A x = b for every A with A_lo <= A <= A_hi and every b with b_lo <= b <= b_hi
    (each entry an interval; b a vector or n by k), which also proves every such A nonsingular; or NotVerified (an
    [A] holding a singular matrix never verifies). inverse: an approximate inverse of the midpoint, else numpy's."""
    Al, Ah, bl, bh = (np.require(a, dtype=np.float64, requirements="C") for a in (A_lo, A_hi, b_lo, b_hi))
    n = Al.shape[0]
    if Al.ndim != 2 or Al.shape != (n, n) or Ah.shape != Al.shape or bl.shape != bh.shape or bl.shape[:1] != (n,) \
            or bl.ndim > 2:
        raise ValueError("crlin.solve_interval: A_lo and A_hi must be n by n, b_lo and b_hi of length n or n by k")
    if mode not in ("classical", "exact"):
        raise ValueError(f"crlin.solve_interval: mode is 'classical' or 'exact', not {mode!r}")
    if not all(np.isfinite(a).all() for a in (Al, Ah, bl, bh)) or not ((Al <= Ah).all() and (bl <= bh).all()):
        raise ValueError("crlin.solve_interval: the bounds must be finite, each lo <= hi")
    if inverse is None:
        try:
            inverse = np.linalg.inv(Al / 2 + Ah / 2)
        except np.linalg.LinAlgError:
            raise NotVerified("crlin.solve_interval: numpy found the midpoint singular") from None
    R = np.require(inverse, dtype=np.float64, requirements="C")
    k = 1 if bl.ndim == 1 else bl.shape[1]
    lo = np.empty(bl.shape)
    hi = np.empty(bl.shape)
    p = lambda a: a.ctypes.data_as(_P)
    s = _lib.crlin_solve_interval(n, k, p(Al), p(Ah), n, p(bl), p(bh), k, p(R), n, p(lo), p(hi), k, _MODES[mode],
                                  _numpy_gemm, None)
    if s != 0:
        bad = [j for j in range(k) if np.isinf((lo if bl.ndim == 2 else lo[:, None])[:, j]).all()]
        raise NotVerified("crlin.solve_interval: not proved (an A in the intervals may be singular, or the radii are "
                          "too large)" + (f"; columns {bad}" if bl.ndim == 2 else ""), bad if bl.ndim == 2 else None,
                          lo, hi)
    return lo, hi


def lstsq(A, b, mode="classical"):
    """lo, hi enclosing the exact least-squares solution of A x = b (A m by n, m >= n: the x minimizing ||A x - b||_2,
    A proved of full column rank), or for m < n the exact minimum-norm solution (A proved of full row rank); b a
    vector or m by k. NotVerified when A is rank-deficient or too ill-conditioned (mode="extended" goes further).
    Through the augmented system of order m + n, so its cost is a solve of that order: heavy when m is much larger
    than n. Its scaling comes from numpy's singular values, its inverse from numpy."""
    A = np.require(A, dtype=np.float64, requirements="C")
    b = np.require(b, dtype=np.float64, requirements="C")
    if A.ndim != 2 or b.shape[:1] != A.shape[:1] or b.ndim > 2:
        raise ValueError("crlin.lstsq: A must be m by n and b of length m, or m by k")
    if mode not in _MODES:
        raise ValueError(f"crlin.lstsq: mode is 'classical', 'exact' or 'extended', not {mode!r}")
    if not (np.isfinite(A).all() and np.isfinite(b).all()):
        raise ValueError("crlin.lstsq: A and b must be finite")
    m, n = A.shape
    alpha, R = 0.0, None
    try:
        smin = np.linalg.svd(A, compute_uv=False)[-1]
        if smin > 0:
            alpha = float(2.0 ** np.floor(np.log2(smin / np.sqrt(2))))   # a power of two near sigma_min / sqrt 2
            N, p = m + n, max(m, n)
            K = np.zeros((N, N))
            K[:p, :p] = alpha * np.eye(p)
            if m >= n:
                K[:m, m:] = A; K[m:, :m] = A.T
            else:
                K[:n, n:] = A.T; K[n:, :n] = A
            R = np.linalg.inv(K)
    except np.linalg.LinAlgError:
        alpha, R = 0.0, None                                 # crlin's own choice and inverse
    k = 1 if b.ndim == 1 else b.shape[1]
    lo = np.empty((n,) if b.ndim == 1 else (n, k))
    hi = np.empty(lo.shape)
    p = lambda a: a.ctypes.data_as(_P)
    s = _lib.crlin_lstsq(m, n, k, p(A), n, p(b), k, alpha, None if R is None else p(R), m + n, p(lo), p(hi), k,
                         _MODES[mode], _numpy_gemm, None)
    if s < 0:
        raise ValueError("crlin.lstsq: refused by the library")
    if s != 0:
        raise NotVerified("crlin.lstsq: not proved (A rank-deficient, or too ill-conditioned)", None, lo, hi)
    return lo, hi


def eig_sym(A, mode="classical", eigh=None):
    """lo, hi with lo[k] <= lambda_k <= hi[k] for the k-th smallest eigenvalue of the symmetric matrix A, every one;
    or NotVerified. Each enclosure is about n u ||A|| wide, absolutely: tight for eigenvalues near ||A||, wide relative
    to small ones. eigh: (d, V), approximate eigenvalues and eigenvectors (columns), else numpy.linalg.eigh(A)'s; the
    proof holds for any, only the width depends on them."""
    A = np.require(A, dtype=np.float64, requirements="C")
    n = A.shape[0]
    if A.ndim != 2 or A.shape != (n, n):
        raise ValueError("crlin.eig_sym: A must be n by n")
    if mode not in ("classical", "exact"):
        raise ValueError(f"crlin.eig_sym: mode is 'classical' or 'exact', not {mode!r}")
    if not np.isfinite(A).all() or not (A == A.T).all():
        raise ValueError("crlin.eig_sym: A must be finite and symmetric")
    d, V = np.linalg.eigh(A) if eigh is None else eigh
    d = np.require(d, dtype=np.float64, requirements="C")
    V = np.require(V, dtype=np.float64, requirements="C")
    if d.shape != (n,) or V.shape != (n, n):
        raise ValueError("crlin.eig_sym: eigh must be (d, V), d of length n and V n by n")
    lo = np.empty(n)
    hi = np.empty(n)
    p = lambda a: a.ctypes.data_as(_P)
    s = _lib.crlin_eig_sym(n, p(A), n, p(V), n, p(d), p(lo), p(hi), _MODES[mode], _numpy_gemm, None)
    if s < 0:
        raise ValueError("crlin.eig_sym: refused by the library")
    if s != 0:
        raise NotVerified("crlin.eig_sym: not proved (the eigenvectors too far from orthogonal)", None, lo, hi)
    return lo, hi


def spd(A, mode="classical", factor=None):
    """A lower bound lam > 0 on every eigenvalue of the symmetric matrix A, which proves A positive definite, or
    NotVerified (its lambda_lo the bound that could be proved). factor: (L, c), an approximate Cholesky factor of
    A - c I (numpy.linalg.cholesky(A - c * np.eye(n)), say), from anywhere: the proof holds for any, and lam is near
    A's smallest eigenvalue when c is. None: the C library factors A itself, aiming at about half the smallest
    eigenvalue (unblocked, threaded: for large A, a factor from LAPACK is faster)."""
    A = np.require(A, dtype=np.float64, requirements="C")
    n = A.shape[0]
    if A.ndim != 2 or A.shape != (n, n):
        raise ValueError("crlin.spd: A must be n by n")
    if mode not in ("classical", "exact"):
        raise ValueError(f"crlin.spd: mode is 'classical' or 'exact', not {mode!r}")
    if not np.isfinite(A).all() or not (A == A.T).all():
        raise ValueError("crlin.spd: A must be finite and symmetric")
    p = lambda a: a.ctypes.data_as(_P)
    lam = ctypes.c_double()
    if factor is None:
        s = _lib.crlin_spd(n, p(A), n, None, n, 0.0, ctypes.byref(lam), _MODES[mode], _numpy_gemm, None)
    else:
        L = np.require(factor[0], dtype=np.float64, requirements="C")
        if L.shape != (n, n) or not np.isfinite(L).all():
            raise ValueError("crlin.spd: the factor must be n by n and finite")
        s = _lib.crlin_spd(n, p(A), n, p(L), n, float(factor[1]), ctypes.byref(lam), _MODES[mode], _numpy_gemm, None)
    if s < 0:
        raise ValueError("crlin.spd: refused by the library")
    if s != 0:
        raise NotVerified("crlin.spd: not proved positive definite", lambda_lo=lam.value)
    return lam.value


def _band_result(s, lo, hi, what):
    if s < 0:
        raise ValueError(f"crlin.{what}: refused by the library (bad arguments, or a band past a gigabyte)")
    if s != 0:
        raise NotVerified(f"crlin.{what}: not proved (A singular, or too ill-conditioned for this route)", None, lo, hi)
    return lo, hi


def solve_banded(l_and_u, ab, b, mode="classical"):
    """lo, hi enclosing the exact solution of the banded system A x = b, in scipy.linalg.solve_banded's layout: (kl, ku)
    and ab with ab[ku + i - j, j] = A[i, j]. No inverse is formed: a proven lower bound on the smallest eigenvalue of A
    (symmetric positive definite) or of A^T A bounds ||x* - x~||_2 by the exact residual, so every component's
    enclosure has the same half-width (an ulp or less when A is well-conditioned). The A^T A route, for the others,
    reaches condition numbers of about 1e8. mode: "classical" (the bound's entries with a priori error bounds) or
    "exact" (each an exact sum; slower on wide bands)."""
    kl, ku = (int(v) for v in l_and_u)
    ab = np.asarray(ab, dtype=np.float64)
    b = np.require(b, dtype=np.float64, requirements="C")
    n = b.shape[0] if b.ndim == 1 else -1
    if n < 1 or kl < 0 or ku < 0 or ab.shape != (kl + ku + 1, n):
        raise ValueError("crlin.solve_banded: ab must be (kl + ku + 1) by n and b of length n")
    if mode not in ("classical", "exact"):
        raise ValueError(f"crlin.solve_banded: mode is 'classical' or 'exact', not {mode!r}")
    if not (np.isfinite(ab).all() and np.isfinite(b).all()):
        raise ValueError("crlin.solve_banded: ab and b must be finite")
    # to row-major band storage: AB[i, j - i + kl] = A[i, j] = ab[ku + i - j, j]
    AB = np.zeros((n, kl + ku + 1))
    for d in range(-kl, ku + 1):       # d = j - i
        i0, i1 = max(0, -d), min(n, n - d)
        if i1 > i0:
            AB[i0:i1, d + kl] = ab[ku - d, i0 + d:i1 + d]
    lo = np.empty(n)
    hi = np.empty(n)
    p = lambda a: a.ctypes.data_as(_P)
    s = _lib.crlin_solve_band(n, kl, ku, p(AB), kl + ku + 1, p(b), p(lo), p(hi), _MODES[mode])
    return _band_result(s, lo, hi, "solve_banded")


def solve_sparse(A, b, mode="classical"):
    """lo, hi enclosing the exact solution of the sparse system A x = b: A a scipy.sparse matrix (any format), or a
    triple (indptr, indices, data) of compressed rows. Reordered by reverse Cuthill-McKee into a band and solved as
    solve_banded solves it (crlin.h's crlin_solve_sparse): its cost grows as n times the band's width squared, fine for
    meshes of tens of thousands of unknowns."""
    if isinstance(A, tuple):
        indptr, indices, data = A
    else:
        A = A.tocsr()
        A.sum_duplicates()
        indptr, indices, data = A.indptr, A.indices, A.data
    indptr = np.require(indptr, dtype=np.uintp, requirements="C")
    indices = np.require(indices, dtype=np.uintp, requirements="C")
    data = np.require(data, dtype=np.float64, requirements="C")
    b = np.require(b, dtype=np.float64, requirements="C")
    n = b.shape[0] if b.ndim == 1 else -1
    if n < 1 or indptr.shape != (n + 1,) or indices.shape != data.shape:
        raise ValueError("crlin.solve_sparse: A must be n by n in compressed rows, b of length n")
    if mode not in ("classical", "exact"):
        raise ValueError(f"crlin.solve_sparse: mode is 'classical' or 'exact', not {mode!r}")
    lo = np.empty(n)
    hi = np.empty(n)
    p = lambda a: a.ctypes.data_as(_P)
    q = lambda a: a.ctypes.data_as(_SZ)
    s = _lib.crlin_solve_sparse(n, q(indptr), q(indices), p(data), p(b), p(lo), p(hi), _MODES[mode])
    return _band_result(s, lo, hi, "solve_sparse")
