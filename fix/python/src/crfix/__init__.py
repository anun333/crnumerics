"""crfix: correctly rounded fixed-point functions on NumPy integer arrays, from crnumerics' C library libcrfix.

A fixed-point value is x * 2**-f (f fractional bits). Each function's exact value is rounded once, in the mode asked
for, to an integer of `bits` bits with fout fractional bits, then saturated:

    import numpy as np, crfix
    x = np.arange(-128, 128, dtype=np.int8)
    y = crfix.sigmoid(x, fin=4, fout=7, bits=8)           # sigmoid(x / 16), as round(sigmoid * 128) in int8
    t = crfix.table("tanh", fin=5, fout=7, bits=8)        # an int8 kernel's lookup table: t[x + 128]

TFLite's affine formats, a value being scale * (q - zero_point), with any scale:

    inp, out = crfix.Affine(0.0478, 3), crfix.Affine(1 / 256, -128)
    t = crfix.table_affine("sigmoid", inp, out)          # a LOGISTIC kernel's table, every entry correctly rounded
    p = crfix.softmax(logits, c=1.0 * 0.0478, out=out)    # the exact softmax of c * logits, rounded into out

The functions are exp, exp2, log, log2, sigmoid, tanh, sqrt, rsqrt, recip, silu (v sigmoid(v)) and gelu (v Phi(v),
the exact GELU), and div; the affine formats take all but sqrt, rsqrt and recip. A container is 8, 16 or 32 bits,
signed, or "u8" and "u16" (unsigned: TFLite's uint8). Modes: "nearest" (ties to even), "up", "down", "zero". Domain edges, as libcrfix states them: log of a
value <= 0 gives the most negative result; sqrt of a negative value 0; rsqrt of a value <= 0 and recip of 0 the
largest; a / 0 by a's sign, 0 / 0 gives 0. The library uses integer arithmetic only.

The library is libcrfix.so.0: found by the environment variable CRFIX_LIBRARY, then a copy inside this package, then
the system's linker paths.
"""
import ctypes
import ctypes.util
import os
from typing import NamedTuple

import numpy as np

__all__ = ["exp", "exp2", "log", "log2", "sigmoid", "tanh", "sqrt", "rsqrt", "recip", "silu", "gelu", "div", "eval",
           "table", "Affine", "eval_affine", "table_affine", "softmax", "layernorm", "FUNCTIONS", "AFFINE_FUNCTIONS", "MODES",
           "UNSIGNED", "version", "library_path"]

FUNCTIONS = ("exp", "exp2", "log", "log2", "sigmoid", "tanh", "sqrt", "rsqrt", "recip", "silu", "gelu")   # C's numbers
AFFINE_FUNCTIONS = ("exp", "exp2", "log", "log2", "sigmoid", "tanh", "silu", "gelu")
UNSIGNED = 256   # or'ed into a bit count: unsigned 8 and 16 bits
MODES = {"nearest": 0, "up": 1, "down": 2, "zero": 3}


def _find():
    env = os.environ.get("CRFIX_LIBRARY")
    if env:
        return env
    for name in ("libcrfix.so.0", "libcrfix.0.dylib"):
        here = os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
        if os.path.exists(here):
            return here
    for name in ("libcrfix.so.0", "libcrfix.so", "libcrfix.0.dylib", "libcrfix.dylib"):
        try:
            ctypes.CDLL(name)
            return name
        except OSError:
            pass
    found = ctypes.util.find_library("crfix")
    if found:
        return found
    raise OSError("crfix: libcrfix.so not found; build crnumerics' libcrfix or set CRFIX_LIBRARY")


library_path = _find()
_lib = ctypes.CDLL(library_path)
_I32 = ctypes.POINTER(ctypes.c_int32)
_int = ctypes.c_int


class _Affine(ctypes.Structure):
    _fields_ = [("scale", ctypes.c_double), ("zero_point", ctypes.c_int32), ("bits", ctypes.c_int)]


_lib.crfix_version.restype = ctypes.c_char_p
_lib.crfix_eval.argtypes = [_int, ctypes.c_int32, _int, _int, _int, _int]
_lib.crfix_eval.restype = ctypes.c_int32
_lib.crfix_div.argtypes = [ctypes.c_int32, _int, ctypes.c_int32, _int, _int, _int, _int]
_lib.crfix_div.restype = ctypes.c_int32
_lib.crfix_eval_affine.argtypes = [_int, ctypes.c_int32, ctypes.POINTER(_Affine), ctypes.POINTER(_Affine), _int]
_lib.crfix_eval_affine.restype = ctypes.c_int32
_lib.crfix_table.argtypes = [_int, _int, _int, _int, _int, _I32]
_lib.crfix_table.restype = _int
_lib.crfix_table_affine.argtypes = [_int, ctypes.POINTER(_Affine), ctypes.POINTER(_Affine), _int, _I32]
_lib.crfix_table_affine.restype = _int
_lib.crfix_softmax.argtypes = [ctypes.c_size_t, _I32, ctypes.c_double, ctypes.POINTER(_Affine), _int, _I32]
_lib.crfix_softmax.restype = _int
if hasattr(_lib, "crfix_layernorm"):   # after libcrfix 0.8.0
    _lib.crfix_layernorm.argtypes = [ctypes.c_size_t, _I32, _int, ctypes.c_int32, _int, _I32, _int, _I32, _int, _int, _int,
                                     _int, _I32]
    _lib.crfix_layernorm.restype = _int
# the array functions came after libcrfix 0.7.0: with an older library, a loop over the scalar ones
_ARRAYS = hasattr(_lib, "crfix_eval_n")
if _ARRAYS:
    _lib.crfix_eval_n.argtypes = [_int, ctypes.c_size_t, _I32, _int, _int, _int, _int, _I32]
    _lib.crfix_eval_n.restype = _int
    _lib.crfix_eval_affine_n.argtypes = [_int, ctypes.c_size_t, _I32, ctypes.POINTER(_Affine), ctypes.POINTER(_Affine),
                                         _int, _I32]
    _lib.crfix_eval_affine_n.restype = _int
    _lib.crfix_div_n.argtypes = [ctypes.c_size_t, _I32, _int, _I32, _int, _int, _int, _int, _I32]
    _lib.crfix_div_n.restype = _int


def version():
    """libcrfix's version string"""
    return _lib.crfix_version().decode()


class Affine(NamedTuple):
    """TFLite's quantization: a value is scale * (q - zero_point), scale a positive normal float (a float32 scale
    converts exactly), q of `bits` bits (8, 16 or 32, signed, or "u8" and "u16")"""
    scale: float
    zero_point: int = 0
    bits: object = 8


def _fn(fn, affine=False):
    names = AFFINE_FUNCTIONS if affine else FUNCTIONS
    if isinstance(fn, str) and fn in names:
        return FUNCTIONS.index(fn)
    raise ValueError("crfix: function %r; one of %s" % (fn, ", ".join(names)))


def _bits(b):
    """a container: 8, 16, 32, or "u8", "u16" (and their numbers, 8 | UNSIGNED and 16 | UNSIGNED)"""
    if b in ("u8", "u16"):
        return int(b[1:]) | UNSIGNED
    if b in (8, 16, 32, 8 | UNSIGNED, 16 | UNSIGNED):
        return int(b)
    raise ValueError("crfix: bits %r; 8, 16, 32, \"u8\" or \"u16\"" % (b,))


def _mode(mode):
    if mode in MODES:
        return MODES[mode]
    raise ValueError("crfix: mode %r; one of %s" % (mode, ", ".join(MODES)))


def _ints(x, name="x"):
    """x as a contiguous int32 array (a copy), refusing floats and values outside int32"""
    a = np.asarray(x)
    if a.dtype.kind == "b" or a.dtype.kind not in "iu":
        raise TypeError("crfix: %s must hold integers (fixed-point values), not %s" % (name, a.dtype))
    if a.size and (a.min() < -2 ** 31 or a.max() > 2 ** 31 - 1):
        raise ValueError("crfix: %s outside int32" % name)
    return np.array(a, dtype=np.int32, order="C")   # a copy; a 0-d array stays one


def _ptr(a):
    return a.ctypes.data_as(_I32)


def _shape(y, a):
    return int(y[()]) if a.ndim == 0 else y


def _aff(a, name):
    if not isinstance(a, Affine):
        a = Affine(*a)
    return _Affine(float(a.scale), int(a.zero_point), _bits(a.bits))


def eval(fn, x, fin, fout, bits=32, mode="nearest"):
    """fn (a name in FUNCTIONS) of x * 2**-fin, rounded to fout fractional bits in mode and saturated to `bits` (8, 16
    or 32): an int32 array of x's shape, or an int for a scalar x"""
    f, m, a, bits = _fn(fn), _mode(mode), _ints(x), _bits(bits)
    y = np.empty_like(a)
    if _ARRAYS:
        if _lib.crfix_eval_n(f, a.size, _ptr(a), fin, bits, fout, m, _ptr(y)):
            raise ValueError("crfix: invalid parameters (fin and fout 0 to 31, bits 8, 16 or 32)")
    else:
        if not (0 <= fin < 32 and 0 <= fout < 32):
            raise ValueError("crfix: invalid parameters (fin and fout 0 to 31, bits 8, 16 or 32)")
        y.reshape(-1)[:] = [_lib.crfix_eval(f, int(v), fin, bits, fout, m) for v in a.reshape(-1)]
    return _shape(y, a)


def _one(name):
    def f(x, fin, fout, bits=32, mode="nearest"):
        return eval(name, x, fin, fout, bits, mode)
    f.__name__ = name
    f.__doc__ = "%s of x * 2**-fin, correctly rounded to fout fractional bits and saturated to `bits`; see eval" % name
    return f


exp, exp2, log, log2, sigmoid, tanh, sqrt, rsqrt, recip, silu, gelu = (_one(n) for n in FUNCTIONS)


def div(a, fa, b, fb, fout, bits=32, mode="nearest"):
    """(a * 2**-fa) / (b * 2**-fb), correctly rounded to fout fractional bits and saturated; a and b broadcast"""
    m, bits = _mode(mode), _bits(bits)
    A, B = np.broadcast_arrays(_ints(a, "a"), _ints(b, "b"))
    A, B = np.array(A, order="C"), np.array(B, order="C")
    y = np.empty_like(A)
    if _ARRAYS:
        if _lib.crfix_div_n(A.size, _ptr(A), fa, _ptr(B), fb, bits, fout, m, _ptr(y)):
            raise ValueError("crfix: invalid parameters (fa, fb and fout 0 to 31, bits 8, 16 or 32)")
    else:
        if not (0 <= fa < 32 and 0 <= fb < 32 and 0 <= fout < 32):
            raise ValueError("crfix: invalid parameters (fa, fb and fout 0 to 31, bits 8, 16 or 32)")
        y.reshape(-1)[:] = [_lib.crfix_div(int(u), fa, int(v), fb, bits, fout, m)
                            for u, v in zip(A.reshape(-1), B.reshape(-1))]
    return _shape(y, A)


def table(fn, fin, fout, bits=8, mode="nearest"):
    """an int8 kernel's lookup table: an int32 array t of 256 with t[x + 128] = fn at x, x = -128 .. 127"""
    t = np.empty(256, dtype=np.int32)
    if _lib.crfix_table(_fn(fn), fin, _bits(bits), fout, _mode(mode), _ptr(t)):
        raise ValueError("crfix: invalid parameters (fin and fout 0 to 31, bits 8, 16 or 32)")
    return t


def eval_affine(fn, q, inp, out, mode="nearest"):
    """round(fn(inp.scale * (q - inp.zero_point)) / out.scale) + out.zero_point in mode, saturated to out.bits; fn
    one of AFFINE_FUNCTIONS; q any int32 values"""
    f, m, a = _fn(fn, True), _mode(mode), _ints(q, "q")
    i, o = _aff(inp, "inp"), _aff(out, "out")
    y = np.empty_like(a)
    if _ARRAYS:
        if _lib.crfix_eval_affine_n(f, a.size, _ptr(a), ctypes.byref(i), ctypes.byref(o), m, _ptr(y)):
            raise ValueError("crfix: invalid parameters (scales positive normal floats, out.bits 8, 16 or 32)")
    else:
        y.reshape(-1)[:] = [_lib.crfix_eval_affine(f, int(v), ctypes.byref(i), ctypes.byref(o), m) for v in a.reshape(-1)]
        if a.size and not (i.scale > 0 and o.scale > 0):
            raise ValueError("crfix: invalid parameters (scales positive normal floats, out.bits 8, 16 or 32)")
    return _shape(y, a)


def table_affine(fn, inp, out, mode="nearest"):
    """an 8-bit kernel's lookup table in affine formats: inp.bits 8, t[q + 128] for q = -128 .. 127, or "u8", t[q] for
    q = 0 .. 255"""
    t = np.empty(256, dtype=np.int32)
    if _lib.crfix_table_affine(_fn(fn, True), ctypes.byref(_aff(inp, "inp")), ctypes.byref(_aff(out, "out")),
                               _mode(mode), _ptr(t)):
        raise ValueError("crfix: invalid parameters (inp.bits 8, scales positive normal floats, out.bits 8, 16 or 32)")
    return t


def softmax(q, c, out=Affine(1 / 256, -128, 8), mode="nearest"):
    """the exact softmax of c * q over q's last axis, each p rounded to round(p / out.scale) + out.zero_point in mode
    and saturated to out.bits; c > 0 is beta times the logits' scale, as TFLite has it (their zero point cancels)"""
    a, m, o = _ints(q, "q"), _mode(mode), _aff(out, "out")
    if a.ndim == 0 or a.shape[-1] == 0:
        raise ValueError("crfix: softmax of an empty vector")
    y = np.empty_like(a)
    rows, ys = a.reshape(-1, a.shape[-1]), y.reshape(-1, a.shape[-1])
    for r in range(rows.shape[0]):
        row, yr = np.ascontiguousarray(rows[r]), np.empty(rows.shape[1], dtype=np.int32)
        if _lib.crfix_softmax(row.size, _ptr(row), float(c), ctypes.byref(o), m, _ptr(yr)):
            raise ValueError("crfix: invalid parameters (c and out.scale positive normal floats, out.bits 8, 16 or 32)")
        ys[r] = yr
    return y


def layernorm(x, fin, fout, eps=0, fe=0, gamma=None, fg=0, beta=None, fb=0, bits=32, mode="nearest"):
    """layer normalization over x's last axis: y = gamma (v - mean) / sqrt(var + eps) + beta with v = x 2**-fin, the
    exact mean and population variance of v, eps = eps 2**-fe, gamma = gamma 2**-fg and beta = beta 2**-fb (integer
    arrays of the last axis's length, or None for 1 and 0); each y rounded once to fout fractional bits in mode and
    saturated. x within -32768 .. 32768 and at most 65536 a row. No variance and eps 0: y = beta"""
    if not hasattr(_lib, "crfix_layernorm"):
        raise OSError("crfix: %s has no crfix_layernorm (it came after libcrfix 0.8.0)" % library_path)
    a, m, bits = _ints(x), _mode(mode), _bits(bits)
    if a.ndim == 0 or a.shape[-1] == 0:
        raise ValueError("crfix: layernorm of an empty vector")
    n = a.shape[-1]
    g = None if gamma is None else _ints(gamma, "gamma").reshape(-1)
    b = None if beta is None else _ints(beta, "beta").reshape(-1)
    if (g is not None and g.size != n) or (b is not None and b.size != n):
        raise ValueError("crfix: gamma and beta must have the last axis's length, %d" % n)
    y = np.empty_like(a)
    rows, ys = a.reshape(-1, n), y.reshape(-1, n)
    for r in range(rows.shape[0]):
        row, yr = np.ascontiguousarray(rows[r]), np.empty(n, dtype=np.int32)
        if _lib.crfix_layernorm(n, _ptr(row), fin, int(eps), fe, None if g is None else _ptr(g), fg,
                                None if b is None else _ptr(b), fb, bits, fout, m, _ptr(yr)):
            raise ValueError("crfix: invalid parameters (x within -32768 .. 32768, n <= 65536, eps >= 0, the fractional "
                             "bits 0 to 31, bits 8, 16, 32, u8 or u16)")
        ys[r] = yr
    return y

