"""crlin: verified linear systems over NumPy arrays, from crnumerics' C library libcrlin. solve(A, b) returns two
arrays lo and hi with lo <= x* <= hi for the exact solution x* of A x = b (the binary64 values of A and b taken
exactly), or raises NotVerified when the proof does not go through (A singular, or too ill-conditioned for
binary64). It never returns an enclosure that misses x*.

    import numpy as np, crlin
    A = np.random.default_rng(1).standard_normal((500, 500)); b = np.ones(500)
    lo, hi = crlin.solve(A, b)            # each [lo[i], hi[i]] holds the exact x*_i, typically an ulp wide

The approximate inverse comes from numpy.linalg.inv and the products from numpy's matmul, threaded or not: every
bound is computed from exact sums or with a priori error bounds in round-to-nearest, never by switching the
rounding mode, so a threaded BLAS cannot break it. mode="classical" (the default) bounds I - R A from two products;
mode="exact" from the exact product (slower, and asks nothing of the BLAS but binary64 arithmetic).

The library is libcrlin.so.0: found by the environment variable CRLIN_LIBRARY, then a copy inside this package,
then the system's linker paths.
"""
import ctypes
import ctypes.util
import os

import numpy as np

__all__ = ["solve", "NotVerified", "version", "library_path"]


class NotVerified(ArithmeticError):
    """The system could not be proved: A singular, or too ill-conditioned for binary64."""


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
_lib.crlin_version.restype = ctypes.c_char_p
_MODES = {"exact": 0, "classical": 1}


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
    """lo, hi with lo <= x* <= hi componentwise for the exact solution of A x = b (A square, b a vector), or
    NotVerified. inverse: an approximate inverse of A to use instead of numpy.linalg.inv(A); the proof holds for any,
    only its success depends on it being close."""
    A = np.require(A, dtype=np.float64, requirements="C")
    b = np.require(b, dtype=np.float64, requirements="C")
    n = A.shape[0]
    if A.ndim != 2 or A.shape != (n, n) or b.shape != (n,):
        raise ValueError("crlin.solve: A must be n by n and b of length n")
    if mode not in _MODES:
        raise ValueError(f"crlin.solve: mode is 'classical' or 'exact', not {mode!r}")
    if not (np.isfinite(A).all() and np.isfinite(b).all()):
        raise ValueError("crlin.solve: A and b must be finite")
    if inverse is None:
        try:
            inverse = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            raise NotVerified("crlin.solve: numpy found A singular") from None
    R = np.require(inverse, dtype=np.float64, requirements="C")
    lo = np.empty(n)
    hi = np.empty(n)
    p = lambda a: a.ctypes.data_as(_P)
    s = _lib.crlin_solve(n, p(A), n, p(b), p(R), n, p(lo), p(hi), _MODES[mode], _numpy_gemm, None)
    if s != 0:
        raise NotVerified("crlin.solve: not proved (A singular, or too ill-conditioned for binary64)")
    return lo, hi
