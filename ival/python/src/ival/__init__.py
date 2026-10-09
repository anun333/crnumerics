"""ival for Python: tight IEEE 1788.1 intervals over NumPy arrays, from crnumerics' C library libival.

An Interval holds one interval, as two Python floats, or an array of them, as two NumPy arrays of float64 broadcast
like NumPy. Every result is the tightest binary64 interval around the exact one: the functions are CORE-MATH's,
correctly rounded in every rounding mode. The empty interval is [nan, nan].

    import ival
    x = ival.Interval(1, 2)
    y = ival.exp(x) * x + 1                          # [e + 1, 2 e^2 + 1], each end rounded outward
    t = ival.Interval.from_text("[0.1, 0.2]")       # decimal ends rounded outward, not to nearest
    s = ival.sin(ival.Interval([0, 1, 2], [0.5, 4, 3]))   # three intervals at once

A number that is not a double (a large int, a Fraction, a Decimal, a long double) becomes the two doubles around it,
never the nearest one, wherever it is given: as an end, as an operand, or as a member to test.

The library is libival.so.0 (crnumerics' `make install`): found by the environment variable IVAL_LIBRARY, then the
system's linker paths, then pkg-config's libdir for ival.
"""
import ctypes
import ctypes.util
import math
import numbers
import os
import subprocess
from decimal import Decimal
from fractions import Fraction

import numpy as np

__all__ = ["Interval", "version", "library_path"]

_N = ctypes.c_size_t
_c = ctypes.c_double
_ref = ctypes.byref
_ONE = ctypes.c_size_t(1)
_INF = math.inf
_NAN = math.nan


def _find():
    env = os.environ.get("IVAL_LIBRARY")
    if env:
        return env
    for name in ("libival.so.0", "libival.so"):
        try:
            ctypes.CDLL(name)
            return name
        except OSError:
            pass
    found = ctypes.util.find_library("ival")
    if found:
        return found
    try:
        libdir = subprocess.run(["pkg-config", "--variable=libdir", "ival"], capture_output=True, text=True,
                                check=True).stdout.strip()
        for name in ("libival.so.0", "libival.so"):
            path = os.path.join(libdir, name)
            if os.path.exists(path):
                return path
    except (OSError, subprocess.CalledProcessError):
        pass
    raise OSError("ival: libival.so not found; install crnumerics' ival (make install) or set IVAL_LIBRARY")


library_path = _find()
_lib = ctypes.CDLL(library_path)
_lib.ival_version.restype = ctypes.c_char_p
_lib.ival_version.argtypes = []


def version():
    """The C library's version (IVAL_VERSION of the ival.h it was built from)."""
    return _lib.ival_version().decode()


def _arr(x):   # float64, C-contiguous, keeping a 0-d array 0-d (np.ascontiguousarray makes it 1-d)
    return np.require(x, dtype=np.float64, requirements="C")


def _p(a):   # the array's address (the array stays alive through the call: the caller holds it)
    return ctypes.c_void_p(a.ctypes.data)


# ---- numbers to doubles, rounded outward
def _round(v, up):
    """the real number v as a double: v itself when it is one, else the double just above it (up) or below it"""
    if type(v) is float:
        return v
    if isinstance(v, np.integer):
        v = int(v)                       # NumPy compares an int64 with a float in float64, inexactly
    elif isinstance(v, (np.bool_, bool)) or not isinstance(v, (numbers.Real, Decimal)):
        raise TypeError(f"ival: an interval's end must be a real number, not {type(v).__name__}")
    try:
        f = float(v)                     # to nearest
    except OverflowError:                # an int or a Fraction beyond the largest double
        f = _INF if v > 0 else -_INF
    if f != f or f == v:                 # NaN, or exact (the comparisons are exact for int, Fraction, Decimal, long double)
        return f
    if up:
        return f if f > v else math.nextafter(f, _INF)
    return f if f < v else math.nextafter(f, -_INF)


def _exact(v):   # v as a Fraction; None for an infinity or NaN
    if isinstance(v, np.integer):
        v = int(v)
    try:
        return Fraction(*v.as_integer_ratio())
    except (OverflowError, ValueError, AttributeError):
        return None


def _ends(x, up):
    """an array of numbers as float64, each rounded outward (up or down) when it is not a double; and a mask of
    the ones rounded (None when none can be)"""
    a = np.asarray(x)
    k, size = a.dtype.kind, a.dtype.itemsize
    if a.dtype == np.float64 or (k in "iuf" and size <= 4):
        return _arr(a), None                                # exact in float64
    if k == "f":                                            # a long double: compared with its rounding exactly
        f = a.astype(np.float64)
        moved = f < a if up else f > a
        return _arr(np.where(moved, np.nextafter(f, _INF if up else -_INF), f)), moved
    if k in "iu":
        f = a.astype(np.float64)
        moved = (a > 2**53) | (a < -2**53) if k == "i" else a > 2**53
        if moved.any():
            f[moved] = [_round(int(v), up) for v in a[moved]]
        return _arr(f), moved
    if k == "O":
        f = np.array([_round(v, up) for v in a.ravel()], dtype=np.float64).reshape(a.shape)
        return _arr(f), np.array([type(v) is not float for v in a.ravel()], dtype=bool).reshape(a.shape)
    if k == "b":
        raise TypeError("ival: an interval's end must be a real number, not a bool")
    raise TypeError(f"ival: an interval's end must be a real number, not {a.dtype}"
                    + (" (text goes through Interval.from_text)" if k in "US" else ""))


def _same(*arrs):
    """the arrays broadcast to one shape, as C float64 arrays; when they are arrays of one shape and floats (an
    array and a number), the arrays pass as they are and each float is filled out, cheaper than broadcasting"""
    s = next((a.shape for a in arrs if type(a) is np.ndarray), None)
    if s is not None and all(type(a) is float or (type(a) is np.ndarray and a.shape == s) for a in arrs):
        return [np.full(s, a) if type(a) is float else _arr(a) for a in arrs]
    return [_arr(a) for a in np.broadcast_arrays(*arrs)]


def _mk(lo, hi):
    obj = _new(Interval)
    obj.lo, obj.hi = lo, hi
    return obj


def _wrap(lo, hi):   # an array result; one interval is kept as two floats
    return _mk(float(lo), float(hi)) if lo.ndim == 0 else _mk(lo, hi)


def _one(lo, hi):
    """[lo, hi] for two doubles, as ival_nums makes it: empty unless lo <= hi, lo != +inf, hi != -inf; zeros +0"""
    if lo <= hi and lo != _INF and hi != -_INF:
        return _mk(0.0 if lo == 0 else lo, 0.0 if hi == 0 else hi)
    return _mk(_NAN, _NAN)


class Interval:
    """One interval, or a NumPy-shaped array of them: lo and hi are two floats, or two float64 arrays of one shape;
    [nan, nan] is empty. Interval(a) is the smallest interval holding the number a (the point [a, a] when a is a
    double); Interval(lo, hi) is the smallest holding [lo, hi], empty unless lo <= hi, lo != +inf and hi != -inf
    (1788's numsToInterval). Decimal ends are best given as text (from_text) or as Decimal: 0.1 as a float is
    already rounded to nearest."""

    __slots__ = ("lo", "hi")
    __array_priority__ = 1000   # so that ndarray + Interval comes here

    def __init__(self, lo, hi=None):
        point = hi is None
        if point:
            hi = lo
        if type(lo) is int and -2**53 <= lo <= 2**53:   # exact as a double
            lo = float(lo)
        if type(hi) is int and -2**53 <= hi <= 2**53:
            hi = float(hi)
        if type(lo) is float and type(hi) is float:   # as ival_nums does it, without the call
            if lo <= hi and lo != _INF and hi != -_INF:
                self.lo, self.hi = 0.0 if lo == 0 else lo, 0.0 if hi == 0 else hi
            else:
                self.lo = self.hi = _NAN
            return
        if not (isinstance(lo, np.ndarray) or isinstance(hi, np.ndarray)) and np.ndim(lo) == 0 and np.ndim(hi) == 0:
            l, h = _round(lo, False), _round(hi, True)
            if (type(lo) is not float or type(hi) is not float) and not point:
                ql, qh = _exact(lo), _exact(hi)   # rounded: lo > hi is decided on the numbers themselves
                if ql is not None and qh is not None and ql > qh:
                    l = h = _NAN
            r = _one(l, h)
            self.lo, self.hi = r.lo, r.hi
            return
        a, b = np.broadcast_arrays(np.asarray(lo), np.asarray(hi))
        lo_d, ml = _ends(a, False)
        hi_d, mh = _ends(b, True)
        if (ml is not None or mh is not None) and not point:
            moved = (ml if ml is not None else False) | (mh if mh is not None else False)
            lo_d, hi_d = lo_d.copy(), hi_d.copy()
            for k in np.flatnonzero(moved):
                ql, qh = _exact(a.flat[k]), _exact(b.flat[k])
                if ql is not None and qh is not None and ql > qh:
                    lo_d.flat[k] = hi_d.flat[k] = _NAN
        out_lo, out_hi, st = np.empty_like(lo_d), np.empty_like(lo_d), np.empty(lo_d.shape, dtype=np.uint8)
        _lib.ival_nums(_p(lo_d), _p(hi_d), _p(out_lo), _p(out_hi), _p(st), lo_d.size)
        if out_lo.ndim == 0:
            out_lo, out_hi = float(out_lo), float(out_hi)
        self.lo, self.hi = out_lo, out_hi

    @classmethod
    def from_text(cls, s):
        """1788's textToInterval: "[0.1, 0.2]", "[1]", "0.1" (a point), "[entire]", "[empty]", "1.5?3" ...; each
        decimal end rounded outward. s may be a string or an array of strings."""
        strs = np.asarray(s, dtype=object)
        flat = [str(t).encode() for t in strs.ravel()]
        n = len(flat)
        arr = (ctypes.c_char_p * n)(*flat)
        lo, hi, st = np.empty(n), np.empty(n), np.empty(n, dtype=np.uint8)
        _lib.ival_text(arr, _p(lo), _p(hi), _p(st), n)
        return _wrap(lo.reshape(strs.shape), hi.reshape(strs.shape))

    @classmethod
    def empty(cls, shape=()):
        return _mk(_NAN, _NAN) if shape == () else _mk(np.full(shape, np.nan), np.full(shape, np.nan))

    @classmethod
    def entire(cls, shape=()):
        return _mk(-_INF, _INF) if shape == () else _mk(np.full(shape, -np.inf), np.full(shape, np.inf))

    # ---- shape and access
    @property
    def shape(self):
        return np.shape(self.lo)

    def __len__(self):
        return len(self.lo)

    def __getitem__(self, k):
        return _wrap(_arr(self.lo[k]), _arr(self.hi[k]))

    def __iter__(self):
        for k in range(len(self)):
            yield self[k]

    def __repr__(self):
        if type(self.lo) is float:
            return f"Interval({self.lo!r}, {self.hi!r})" if self.lo == self.lo else "Interval.empty()"
        return f"Interval(lo={self.lo!r}, hi={self.hi!r})"

    # ---- arithmetic (tight; a number is the smallest interval holding it)
    def __add__(self, o): return _bin(_lib.ival_add, self, o)
    def __radd__(self, o): return _bin(_lib.ival_add, o, self)
    def __sub__(self, o): return _bin(_lib.ival_sub, self, o)
    def __rsub__(self, o): return _bin(_lib.ival_sub, o, self)
    def __mul__(self, o): return _bin(_lib.ival_mul, self, o)
    def __rmul__(self, o): return _bin(_lib.ival_mul, o, self)
    def __truediv__(self, o): return _bin(_lib.ival_div, self, o)
    def __rtruediv__(self, o): return _bin(_lib.ival_div, o, self)
    def __neg__(self): return _un(_lib.ival_neg, self)
    def __pos__(self): return self
    def __abs__(self): return _un(_lib.ival_abs, self)

    def __pow__(self, p):
        if isinstance(p, (int, np.integer)) or (isinstance(p, np.ndarray) and p.dtype.kind in "iu"):
            return pown(self, p)
        return pow(self, p)

    # ---- the 1788 measures and tests
    @property
    def inf(self): return _num(_lib.ival_inf, self)
    @property
    def sup(self): return _num(_lib.ival_sup, self)
    @property
    def mid(self): return _num(_lib.ival_mid, self)
    @property
    def wid(self): return _num(_lib.ival_wid, self)
    @property
    def rad(self): return _num(_lib.ival_rad, self)
    @property
    def mag(self): return _num(_lib.ival_mag, self)
    @property
    def mig(self): return _num(_lib.ival_mig, self)

    def is_empty(self): return _bool1(_lib.ival_isempty, self)
    def is_entire(self): return _bool1(_lib.ival_isentire, self)
    def is_singleton(self): return _bool1(_lib.ival_issingleton, self)

    def __contains__(self, x):
        """x in X: the number x lies in X (in every interval of an array)"""
        r = contains(self, x)
        return bool(r) if np.ndim(r) == 0 else bool(np.all(r))

    def subset(self, o): return _bool2(_lib.ival_subset, self, o)
    def interior(self, o): return _bool2(_lib.ival_interior, self, o)
    def disjoint(self, o): return _bool2(_lib.ival_disjoint, self, o)
    def equal(self, o): return _bool2(_lib.ival_equal, self, o)

    def __and__(self, o): return _bin(_lib.ival_intersect, self, o)
    def __or__(self, o): return _bin(_lib.ival_hull, self, o)


_new = object.__new__


def _as_interval(x):
    if isinstance(x, Interval):
        return x
    if type(x) is int and -2**53 <= x <= 2**53:
        x = float(x)
    if type(x) is float:   # a double is the point [x, x]; an infinity or NaN the empty interval
        return _mk(0.0 if x == 0 else x, 0.0 if x == 0 else x) if -_INF < x < _INF else _mk(_NAN, _NAN)
    return Interval(x)


# Each operation below has two paths: one interval (two floats, passed by reference, with no NumPy: a call costs
# about a microsecond) and arrays (broadcast, one call over all of them).
def _out(shape):
    return np.empty(shape), np.empty(shape)


def _un(f, x):
    x = _as_interval(x)
    if type(x.lo) is float:
        zl, zh = _c(), _c()
        f(_ref(_c(x.lo)), _ref(_c(x.hi)), _ref(zl), _ref(zh), _ONE)
        return _mk(zl.value, zh.value)
    lo, hi = _arr(x.lo), _arr(x.hi)
    zl, zh = _out(lo.shape)
    f(_p(lo), _p(hi), _p(zl), _p(zh), _N(lo.size))
    return _mk(zl, zh)


def _bin(f, x, y):
    x, y = _as_interval(x), _as_interval(y)
    if type(x.lo) is float and type(y.lo) is float:
        zl, zh = _c(), _c()
        f(_ref(_c(x.lo)), _ref(_c(x.hi)), _ref(_c(y.lo)), _ref(_c(y.hi)), _ref(zl), _ref(zh), _ONE)
        return _mk(zl.value, zh.value)
    al, ah, bl, bh = _same(x.lo, x.hi, y.lo, y.hi)
    zl, zh = _out(al.shape)
    f(_p(al), _p(ah), _p(bl), _p(bh), _p(zl), _p(zh), _N(al.size))
    return _wrap(zl, zh)


def _num(f, x):
    if type(x.lo) is float:
        y = _c()
        f(_ref(_c(x.lo)), _ref(_c(x.hi)), _ref(y), _ONE)
        return y.value
    lo, hi = _arr(x.lo), _arr(x.hi)
    y = np.empty(lo.shape)
    f(_p(lo), _p(hi), _p(y), _N(lo.size))
    return y


def _bool1(f, x):
    if type(x.lo) is float:
        r = ctypes.c_ubyte()
        f(_ref(_c(x.lo)), _ref(_c(x.hi)), _ref(r), _ONE)
        return bool(r.value)
    lo, hi = _arr(x.lo), _arr(x.hi)
    r = np.empty(lo.shape, dtype=np.uint8)
    f(_p(lo), _p(hi), _p(r), _N(lo.size))
    return r.astype(bool)


def _bool2(f, x, y):
    x, y = _as_interval(x), _as_interval(y)
    if type(x.lo) is float and type(y.lo) is float:
        r = ctypes.c_ubyte()
        f(_ref(_c(x.lo)), _ref(_c(x.hi)), _ref(_c(y.lo)), _ref(_c(y.hi)), _ref(r), _ONE)
        return bool(r.value)
    al, ah, bl, bh = _same(x.lo, x.hi, y.lo, y.hi)
    r = np.empty(al.shape, dtype=np.uint8)
    f(_p(al), _p(ah), _p(bl), _p(bh), _p(r), _N(al.size))
    return r.astype(bool) if r.ndim else bool(r)


def contains(x, v):
    """1788's isMember: whether the real number v lies in X, elementwise (an infinity never does)"""
    x = _as_interval(x)
    if np.ndim(v) == 0 and type(x.lo) is float:
        if type(v) is not float:   # compared exactly, not rounded
            q = _exact(v)
            return q is not None and x.lo <= q <= x.hi
        r = ctypes.c_ubyte()
        _lib.ival_ismember(_ref(_c(v)), _ref(_c(x.lo)), _ref(_c(x.hi)), _ref(r), _ONE)
        return bool(r.value)
    a = np.asarray(v)
    if not (a.dtype == np.float64 or (a.dtype.kind in "iuf" and a.dtype.itemsize <= 4)):
        va, lo, hi = np.broadcast_arrays(a, x.lo, x.hi)
        r = np.array([contains(_mk(float(l), float(h)), w) for w, l, h in zip(va.ravel(), lo.ravel(), hi.ravel())])
        return r.reshape(va.shape) if va.ndim else bool(r[0])
    v, lo, hi = (_arr(b) for b in np.broadcast_arrays(a, x.lo, x.hi))
    r = np.empty(v.shape, dtype=np.uint8)
    _lib.ival_ismember(_p(v), _p(lo), _p(hi), _p(r), _N(v.size))
    return r.astype(bool) if r.ndim else bool(r)


# ---- the functions, each the tightest interval around f over X (within f's domain)
_UNARY = ["acos", "acosh", "acospi", "asin", "asinh", "asinpi", "atan", "atanh", "atanpi", "cbrt", "cos", "cosh",
          "cospi", "erf", "erfc", "exp", "exp10", "exp2", "expm1", "log", "log10", "log1p", "log2", "rsqrt", "sin",
          "sinh", "sinpi", "sqrt", "tan", "tanh", "tanpi", "tgamma", "sqr", "recip", "sign", "ceil", "floor",
          "trunc", "round", "roundeven"]
_BINARY = ["atan2", "hypot", "pow", "min", "max", "cancelminus", "cancelplus"]


def _make_unary(name):
    cf = getattr(_lib, "ival_" + name)

    def f(x):
        return _un(cf, x)
    f.__name__ = name
    f.__doc__ = f"ival_{name}: the tightest interval around {name}(x) for x in X, elementwise"
    return f


def _make_binary(name):
    cf = getattr(_lib, "ival_" + name)

    def f(x, y):
        return _bin(cf, x, y)
    f.__name__ = name
    f.__doc__ = f"ival_{name}: the tightest interval around {name}(x, y) for x in X, y in Y, elementwise"
    return f


for _n in _UNARY:
    globals()[_n] = _make_unary(_n)
    __all__.append(_n)
for _n in _BINARY:
    globals()[_n] = _make_binary(_n)
    __all__.append(_n)
for _n in _UNARY[:32]:   # the 32 functions (ival-list.h) have the accurate mode: one ulp at most, at vector speed
    globals()["acc_" + _n] = _make_unary("acc_" + _n)
    __all__.append("acc_" + _n)
for _n in ("atan2", "hypot", "pow"):
    globals()["acc_" + _n] = _make_binary("acc_" + _n)
    __all__.append("acc_" + _n)


def _int1(p):   # an integer exponent or root as a C int
    if type(p) is int and -2**31 <= p < 2**31:
        return p
    if isinstance(p, (np.bool_, bool)) or not isinstance(p, (numbers.Integral, np.integer)):
        if not (isinstance(p, (numbers.Real, np.floating)) and float(p).is_integer()):
            raise TypeError(f"ival: the power must be an integer, not {p!r}")
    p = int(p)
    if not -2**31 <= p < 2**31:
        raise OverflowError(f"ival: the power {p} is beyond a C int")
    return p


def _ints(p):   # an array of them
    a = np.asarray(p)
    if a.dtype.kind not in "iu" and not (a.dtype.kind == "f" and np.all(a == np.floor(a))):
        raise TypeError(f"ival: the powers must be integers, not {a.dtype}")
    if a.size and (a.min() < -2**31 or a.max() >= 2**31):
        raise OverflowError("ival: a power beyond a C int")
    return a.astype(np.intc)


def _pow_int(f, x, p):
    x = _as_interval(x)
    if type(x.lo) is float and (type(p) is int or np.ndim(p) == 0):
        zl, zh = _c(), _c()
        f(_ref(_c(x.lo)), _ref(_c(x.hi)), _ref(ctypes.c_int(_int1(p))), _ref(zl), _ref(zh), _ONE)
        return _mk(zl.value, zh.value)
    lo, hi, pp = np.broadcast_arrays(x.lo, x.hi, _ints(p))
    lo, hi = _arr(lo), _arr(hi)
    pp = np.require(pp, dtype=np.intc, requirements="C")
    zl, zh = _out(lo.shape)
    f(_p(lo), _p(hi), _p(pp), _p(zl), _p(zh), _N(lo.size))
    return _wrap(zl, zh)


def pown(x, p):
    """x^p for integer p, elementwise (1788's pown): tight; a negative power of [0, 0] is empty"""
    return _pow_int(_lib.ival_pown, x, p)


def rootn(x, q):
    """the real q-th root, elementwise (1788.1's rootn)"""
    return _pow_int(_lib.ival_rootn, x, q)


def fma(a, b, c):
    """a * b + c with one rounding of each end, elementwise"""
    a, b, c = _as_interval(a), _as_interval(b), _as_interval(c)
    if type(a.lo) is float and type(b.lo) is float and type(c.lo) is float:
        zl, zh = _c(), _c()
        _lib.ival_fma(*(_ref(_c(v)) for v in (a.lo, a.hi, b.lo, b.hi, c.lo, c.hi)), _ref(zl), _ref(zh), _ONE)
        return _mk(zl.value, zh.value)
    arrs = _same(a.lo, a.hi, b.lo, b.hi, c.lo, c.hi)
    zl, zh = _out(arrs[0].shape)
    _lib.ival_fma(*(_p(v) for v in arrs), _p(zl), _p(zh), _N(arrs[0].size))
    return _wrap(zl, zh)


# ---- reverse operations: the tightest interval around {x in X : f(x) in C}
def _make_rev(name):
    cf = getattr(_lib, "ival_" + name)

    def f(c, x=None):
        c = _as_interval(c)
        x = _mk(-_INF, _INF) if x is None else _as_interval(x)
        if type(c.lo) is float and type(x.lo) is float:
            zl, zh = _c(), _c()
            cf(_ref(_c(c.lo)), _ref(_c(c.hi)), _ref(_c(x.lo)), _ref(_c(x.hi)), _ref(zl), _ref(zh), _ONE)
            return _mk(zl.value, zh.value)
        cl, ch, xl, xh = _same(c.lo, c.hi, x.lo, x.hi)
        zl, zh = _out(cl.shape)
        cf(_p(cl), _p(ch), _p(xl), _p(xh), _p(zl), _p(zh), _N(cl.size))
        return _wrap(zl, zh)
    f.__name__ = name
    f.__doc__ = f"ival_{name}(C, X): the tightest interval around {{x in X : f(x) in C}} (X the whole line if omitted)"
    return f


for _n in ("sqrrev", "absrev", "coshrev", "sinrev", "cosrev", "tanrev"):
    globals()[_n] = _make_rev(_n)
    __all__.append(_n)


def pownrev(c, p, x=None):
    """the tightest interval around {x in X : x^p in C}"""
    c = _as_interval(c)
    x = _mk(-_INF, _INF) if x is None else _as_interval(x)
    if type(c.lo) is float and type(x.lo) is float and (type(p) is int or np.ndim(p) == 0):
        zl, zh = _c(), _c()
        _lib.ival_pownrev(_ref(_c(c.lo)), _ref(_c(c.hi)), _ref(_c(x.lo)), _ref(_c(x.hi)),
                          _ref(ctypes.c_int(_int1(p))), _ref(zl), _ref(zh), _ONE)
        return _mk(zl.value, zh.value)
    cl, ch, xl, xh, pp = np.broadcast_arrays(c.lo, c.hi, x.lo, x.hi, _ints(p))
    cl, ch, xl, xh = _arr(cl), _arr(ch), _arr(xl), _arr(xh)
    pp = np.require(pp, dtype=np.intc, requirements="C")
    zl, zh = _out(cl.shape)
    _lib.ival_pownrev(_p(cl), _p(ch), _p(xl), _p(xh), _p(pp), _p(zl), _p(zh), _N(cl.size))
    return _wrap(zl, zh)


def _rev3(f, b, c, x):
    b, c = _as_interval(b), _as_interval(c)
    x = _mk(-_INF, _INF) if x is None else _as_interval(x)
    if type(b.lo) is float and type(c.lo) is float and type(x.lo) is float:
        zl, zh = _c(), _c()
        f(*(_ref(_c(v)) for v in (b.lo, b.hi, c.lo, c.hi, x.lo, x.hi)), _ref(zl), _ref(zh), _ONE)
        return _mk(zl.value, zh.value)
    arrs = _same(b.lo, b.hi, c.lo, c.hi, x.lo, x.hi)
    zl, zh = _out(arrs[0].shape)
    f(*(_p(v) for v in arrs), _p(zl), _p(zh), _N(arrs[0].size))
    return _wrap(zl, zh)


def mulrev(b, c, x=None):
    """the tightest interval around {x in X : x b in C for some b in B}"""
    return _rev3(_lib.ival_mulrev, b, c, x)


def powrev1(b, c, x=None):
    """the tightest interval around {x in X : x^y in C for some y in B}"""
    return _rev3(_lib.ival_powrev1, b, c, x)


def powrev2(a, c, y=None):
    """the tightest interval around {y in Y : x^y in C for some x in A}"""
    return _rev3(_lib.ival_powrev2, a, c, y)


__all__ += ["pown", "rootn", "fma", "contains", "pownrev", "mulrev", "powrev1", "powrev2"]
