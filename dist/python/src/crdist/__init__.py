"""crdist for Python: erfinv, erfcinv and probit (the standard normal quantile) over NumPy arrays, correctly rounded
to nearest in binary32, binary16 and bfloat16, from crnumerics' C library libcrdist. Every input of every function
in each format was checked against MPFR; the results are the same bits on every machine, in any floating-point
environment.

    import numpy as np, crdist
    crdist.probit(np.float32(0.975))                 # 1.959964 (R's qnorm, SciPy's ndtri), the float32 nearest
    crdist.erfinv(np.array([0.1, 0.5], np.float32)) # elementwise
    crdist.probit(np.float16(0.9))                   # binary16 in, binary16 out, rounded once
    u = np.random.default_rng(1).random(10, dtype=np.float32)
    z = crdist.probit(u)                             # normal draws, the same everywhere for the same u
    crdist.log_ndtr(np.float32(-40))                 # log Phi(-40) = -804.6084, where Phi underflows
    s = crdist.Stream(seed=42)                       # Philox4x32-10: the same draws on every machine
    g = s.gamma(1000, shape=2.5)

The format follows the input: float16 arrays use binary16, ml_dtypes' bfloat16 arrays use bfloat16 (or pass bfloat16
bit patterns as uint16 to erfinv_bf16, erfcinv_bf16 and probit_bf16), float32 uses binary32, and anything else is
converted to float32 first (rounded to nearest), so a float64 0.1 becomes float32's 0.1 before the function sees it.

The library is libcrdist.so.0: found by the environment variable CRDIST_LIBRARY, then a copy inside this package
(a wheel's), then the system's linker paths (crnumerics' `make install`), then pkg-config's libdir for crdist.
"""
import ctypes
import ctypes.util
import os
import subprocess

import numpy as np

__all__ = ["erfinv", "erfcinv", "probit", "ndtr", "log_ndtr", "normpdf", "logit", "erfinv_bf16", "erfcinv_bf16",
           "probit_bf16", "Stream", "version", "library_path"]


def _find():
    env = os.environ.get("CRDIST_LIBRARY")
    if env:
        return env
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libcrdist.so.0")   # a wheel's own copy
    if os.path.exists(here):
        return here
    for name in ("libcrdist.so.0", "libcrdist.so"):
        try:
            ctypes.CDLL(name)
            return name
        except OSError:
            pass
    found = ctypes.util.find_library("crdist")
    if found:
        return found
    try:
        libdir = subprocess.run(["pkg-config", "--variable=libdir", "crdist"], capture_output=True, text=True,
                                check=True).stdout.strip()
        for name in ("libcrdist.so.0", "libcrdist.so"):
            path = os.path.join(libdir, name)
            if os.path.exists(path):
                return path
    except (OSError, subprocess.CalledProcessError):
        pass
    raise OSError("crdist: libcrdist.so not found; install crnumerics (make install) or set CRDIST_LIBRARY")


library_path = _find()
_lib = ctypes.CDLL(library_path)
_lib.crdist_version.restype = ctypes.c_char_p
_lib.crdist_version.argtypes = []
_SIG = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
for _f in ("erfinv", "erfcinv", "probit", "ndtr", "log_ndtr", "normpdf", "logit"):
    for _name in ("crdist_%sf" % _f, "crdist_%s_f16" % _f, "crdist_%s_bf16" % _f):
        getattr(_lib, _name).restype = None
        getattr(_lib, _name).argtypes = _SIG


def version():
    """The C library's version (CRDIST_VERSION of the crdist.h it was built from)."""
    return _lib.crdist_version().decode()


def _call(name, x, dtype):
    a = np.require(x, dtype=dtype, requirements="C")   # keeps a 0-d array 0-d
    y = np.empty_like(a)
    getattr(_lib, name)(ctypes.c_void_p(y.ctypes.data), ctypes.c_void_p(a.ctypes.data), a.size)
    return y if y.ndim else y[()]


def _dispatch(f, x):
    dt = getattr(x, "dtype", None)
    if dt == np.float16:
        return _call("crdist_%s_f16" % f, x, np.float16)
    if dt is not None and dt.name == "bfloat16":   # ml_dtypes' type: two bytes, bfloat16's bits
        a = np.require(x, requirements="C")
        return _call("crdist_%s_bf16" % f, a.view(np.uint16), np.uint16).view(dt)
    return _call("crdist_%sf" % f, x, np.float32)


def erfinv(x):
    """The y with erf(y) = x, correctly rounded (-1 <= x <= 1; erfinv(+-1) = +-inf, NaN outside)."""
    return _dispatch("erfinv", x)


def erfcinv(x):
    """The y with erfc(y) = x, correctly rounded (0 <= x <= 2; erfcinv(0) = inf, erfcinv(2) = -inf, NaN outside)."""
    return _dispatch("erfcinv", x)


def probit(p):
    """The y with Phi(y) = p, Phi the standard normal distribution function, correctly rounded (0 <= p <= 1;
    probit(0) = -inf, probit(1) = inf, NaN outside): R's qnorm, SciPy's ndtri."""
    return _dispatch("probit", p)


def ndtr(x):
    """Phi(x), the standard normal distribution function, correctly rounded (SciPy's ndtr, R's pnorm); the
    survival function is ndtr(-x)."""
    return _dispatch("ndtr", x)


def log_ndtr(x):
    """log Phi(x), correctly rounded, also where Phi underflows (SciPy's log_ndtr)."""
    return _dispatch("log_ndtr", x)


def normpdf(x):
    """The standard normal density e^(-x^2/2)/sqrt(2 pi), correctly rounded."""
    return _dispatch("normpdf", x)


def logit(p):
    """log(p/(1 - p)), correctly rounded (0 <= p <= 1; logit(0) = -inf, logit(1) = inf, NaN outside)."""
    return _dispatch("logit", p)


def erfinv_bf16(bits):
    """erfinv on bfloat16 numbers given as their bit patterns (uint16), returning bit patterns."""
    return _call("crdist_erfinv_bf16", bits, np.uint16)


def erfcinv_bf16(bits):
    """erfcinv on bfloat16 bit patterns (uint16)."""
    return _call("crdist_erfcinv_bf16", bits, np.uint16)


def probit_bf16(bits):
    """probit on bfloat16 bit patterns (uint16)."""
    return _call("crdist_probit_bf16", bits, np.uint16)


class _CStream(ctypes.Structure):
    _fields_ = [("seed", ctypes.c_uint64), ("substream", ctypes.c_uint64), ("index", ctypes.c_uint64)]


for _name in ("crdist_uniformf", "crdist_normalf", "crdist_exponentialf"):
    getattr(_lib, _name).argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(_CStream)]
_lib.crdist_words.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(_CStream)]
for _name in ("crdist_gammaf", "crdist_chisquaredf", "crdist_student_tf"):
    getattr(_lib, _name).argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_double, ctypes.POINTER(_CStream)]
_lib.crdist_betaf.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_double, ctypes.c_double, ctypes.POINTER(_CStream)]


class Stream:
    """A Philox4x32-10 stream (seed, substream), read from index on: the same draws on every machine (crdist.h's
    specification; each sampler's bits are fixed by it). Each call takes the words it needs and moves index past
    them; streams with different substreams never overlap.

        s = crdist.Stream(seed=42)
        z = s.normal(1000)                  # float32
        g = s.gamma(1000, shape=2.5)
        crdist.Stream(42).normal(1000)      # the same z again
    """

    def __init__(self, seed, substream=0, index=0):
        self._s = _CStream(seed, substream, index)

    seed = property(lambda self: self._s.seed)
    substream = property(lambda self: self._s.substream)
    index = property(lambda self: self._s.index)

    def _run(self, name, n, *args, dtype=np.float32):
        y = np.empty(int(n), dtype)
        getattr(_lib, name)(ctypes.c_void_p(y.ctypes.data), y.size, *args, ctypes.byref(self._s))
        return y

    def words(self, n):
        """the stream's next n 32-bit words"""
        return self._run("crdist_words", n, dtype=np.uint32)

    def uniform(self, n):
        """(2k + 1) 2^-24 for k the word's top 23 bits: 2^23 values in (0, 1), symmetric about 1/2"""
        return self._run("crdist_uniformf", n)

    def normal(self, n):
        """standard normal: probit of a binary32 in (0, 1/2], the sign from the word's top bit"""
        return self._run("crdist_normalf", n)

    def exponential(self, n):
        return self._run("crdist_exponentialf", n)

    def gamma(self, n, shape):
        return self._run("crdist_gammaf", n, float(shape))

    def beta(self, n, a, b):
        return self._run("crdist_betaf", n, float(a), float(b))

    def chisquared(self, n, k):
        return self._run("crdist_chisquaredf", n, float(k))

    def student_t(self, n, nu):
        return self._run("crdist_student_tf", n, float(nu))
