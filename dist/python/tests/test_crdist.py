"""Tests of crdist's Python binding: results correctly rounded against mpmath (300 bits, then rounded once to the
format), the format following the input's dtype, shapes, scalars and non-contiguous arrays, the special values,
and the floating-point state left as it was. crdist-check proves the C library on every input; these test that
the binding hands the right inputs to the right functions and the results back. Run with libcrdist on the
linker's path or CRDIST_LIBRARY naming it: python3 -m pytest dist/python/tests"""
import random

import mpmath
import numpy as np
import pytest

import crdist

mpmath.mp.prec = 300
rng = random.Random(20261013)


def exact(f, x):
    x = mpmath.mpf(float(x))
    if f == "erfinv":
        return mpmath.erfinv(x)
    if f == "erfcinv":
        return mpmath.erfinv(1 - x)
    return mpmath.sqrt(2) * mpmath.erfinv(2 * x - 1)


def rounded(v, bits):
    """v rounded once to `bits` significant bits, to nearest (normal numbers only)"""
    with mpmath.workprec(bits):
        return float(+v)


def draw(f, dtype, n, lo_exp):
    """n numbers of the dtype inside f's domain, away from its ends, with |y| in the format's normal range"""
    out = []
    top = 1.0 if f != "erfcinv" else 2.0
    while len(out) < n:
        x = dtype(rng.uniform(0, top) if rng.random() < 0.5 else 2.0 ** rng.uniform(lo_exp, -1))
        if 0 < x < top and x != 1 and not (f == "probit" and x == 0.5) and abs(exact(f, x)) > 2.0 ** -13:
            out.append(x)
    return np.array(out, dtype)


@pytest.mark.parametrize("f", ["erfinv", "erfcinv", "probit"])
def test_float32_correctly_rounded(f):
    x = draw(f, np.float32, 400, -100)
    y = getattr(crdist, f)(x)
    assert y.dtype == np.float32
    for a, b in zip(x, y):
        assert float(b) == rounded(exact(f, a), 24), (f, float(a))


@pytest.mark.parametrize("f", ["erfinv", "erfcinv", "probit"])
def test_float16_correctly_rounded(f):
    x = draw(f, np.float16, 300, -13)
    y = getattr(crdist, f)(x)
    assert y.dtype == np.float16
    for a, b in zip(x, y):
        assert float(b) == rounded(exact(f, a), 11), (f, float(a))


def test_bfloat16_bits():
    # 0.5 is 0x3f00; erfinv(0.5) = 0.4769..., bfloat16's nearest 0.4765625 = 0x3ef4
    assert int(crdist.erfinv_bf16(np.uint16(0x3f00))) == 0x3ef4
    bits = np.array([0x3f00, 0x3f40, 0x0001], np.uint16)   # 0.5, 0.75, the smallest subnormal
    y = crdist.probit_bf16(bits)
    assert y.dtype == np.uint16 and y.shape == (3,)
    f32 = (y.astype(np.uint32) << 16).view(np.float32)
    assert f32[0] == 0 and 0.67 < f32[1] < 0.68 and f32[2] < -13


def test_bfloat16_dtype():
    ml = pytest.importorskip("ml_dtypes")
    x = np.array([0.5, 0.25], ml.bfloat16)
    y = crdist.erfinv(x)
    assert y.dtype == ml.bfloat16
    assert int(y.view(np.uint16)[0]) == 0x3ef4


def test_special_values():
    inf, nan = np.inf, np.nan
    e = crdist.erfinv(np.array([-1, 1, 0, -0.0, 2, nan], np.float32))
    assert e[0] == -inf and e[1] == inf and e[2] == 0 and np.signbit(e[3]) and np.isnan(e[4]) and np.isnan(e[5])
    c = crdist.erfcinv(np.array([0, 2, 1, -1, 3], np.float32))
    assert c[0] == inf and c[1] == -inf and c[2] == 0 and not np.signbit(c[2]) and np.isnan(c[3]) and np.isnan(c[4])
    p = crdist.probit(np.array([0, 1, 0.5, -0.5, 1.5], np.float32))
    assert p[0] == -inf and p[1] == inf and p[2] == 0 and np.isnan(p[3]) and np.isnan(p[4])


def test_shapes_scalars_and_conversion():
    x = np.linspace(0.05, 0.95, 12, dtype=np.float32).reshape(3, 4)
    y = crdist.probit(x)
    assert y.shape == (3, 4) and y.dtype == np.float32
    assert np.array_equal(crdist.probit(x.T), y.T)             # non-contiguous: elementwise the same
    assert np.array_equal(crdist.probit(x[:, ::2]), y[:, ::2])
    s = crdist.probit(np.float32(0.975))
    assert np.ndim(s) == 0 and float(s) == rounded(exact("probit", np.float32(0.975)), 24)
    # a float64 or a list becomes float32 first (rounded to nearest), and the result is float32
    assert crdist.probit([0.975]).dtype == np.float32
    assert crdist.probit(np.float64(0.975)) == s
    assert crdist.erfinv(0.5) == crdist.erfinv(np.float32(0.5))


def test_environment_left_alone():
    # the library sets round-to-nearest for itself and restores the caller's state; NumPy's error state is Python's
    before = np.geterr()
    crdist.erfinv(np.array([0.5, 2.0], np.float32))
    assert np.geterr() == before


def test_version():
    assert crdist.version().count(".") == 2
