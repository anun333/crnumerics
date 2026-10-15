"""Tests of crdistr, crdist's Python binding: results correctly rounded against mpmath (300 bits, then rounded once to the
format), the format following the input's dtype, shapes, scalars and non-contiguous arrays, the special values,
and the floating-point state left as it was. crdist-check proves the C library on every input; these test that
the binding hands the right inputs to the right functions and the results back. Run with libcrdist on the
linker's path or CRDIST_LIBRARY naming it: python3 -m pytest dist/python/tests"""
import random

import mpmath
import numpy as np
import pytest

import crdistr

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
    y = getattr(crdistr, f)(x)
    assert y.dtype == np.float32
    for a, b in zip(x, y):
        assert float(b) == rounded(exact(f, a), 24), (f, float(a))


@pytest.mark.parametrize("f", ["erfinv", "erfcinv", "probit"])
def test_float16_correctly_rounded(f):
    x = draw(f, np.float16, 300, -13)
    y = getattr(crdistr, f)(x)
    assert y.dtype == np.float16
    for a, b in zip(x, y):
        assert float(b) == rounded(exact(f, a), 11), (f, float(a))


def test_bfloat16_bits():
    # 0.5 is 0x3f00; erfinv(0.5) = 0.4769..., bfloat16's nearest 0.4765625 = 0x3ef4
    assert int(crdistr.erfinv_bf16(np.uint16(0x3f00))) == 0x3ef4
    bits = np.array([0x3f00, 0x3f40, 0x0001], np.uint16)   # 0.5, 0.75, the smallest subnormal
    y = crdistr.probit_bf16(bits)
    assert y.dtype == np.uint16 and y.shape == (3,)
    f32 = (y.astype(np.uint32) << 16).view(np.float32)
    assert f32[0] == 0 and 0.67 < f32[1] < 0.68 and f32[2] < -13


def test_bfloat16_dtype():
    ml = pytest.importorskip("ml_dtypes")
    x = np.array([0.5, 0.25], ml.bfloat16)
    y = crdistr.erfinv(x)
    assert y.dtype == ml.bfloat16
    assert int(y.view(np.uint16)[0]) == 0x3ef4


def test_special_values():
    inf, nan = np.inf, np.nan
    e = crdistr.erfinv(np.array([-1, 1, 0, -0.0, 2, nan], np.float32))
    assert e[0] == -inf and e[1] == inf and e[2] == 0 and np.signbit(e[3]) and np.isnan(e[4]) and np.isnan(e[5])
    c = crdistr.erfcinv(np.array([0, 2, 1, -1, 3], np.float32))
    assert c[0] == inf and c[1] == -inf and c[2] == 0 and not np.signbit(c[2]) and np.isnan(c[3]) and np.isnan(c[4])
    p = crdistr.probit(np.array([0, 1, 0.5, -0.5, 1.5], np.float32))
    assert p[0] == -inf and p[1] == inf and p[2] == 0 and np.isnan(p[3]) and np.isnan(p[4])


def test_shapes_scalars_and_conversion():
    x = np.linspace(0.05, 0.95, 12, dtype=np.float32).reshape(3, 4)
    y = crdistr.probit(x)
    assert y.shape == (3, 4) and y.dtype == np.float32
    assert np.array_equal(crdistr.probit(x.T), y.T)             # non-contiguous: elementwise the same
    assert np.array_equal(crdistr.probit(x[:, ::2]), y[:, ::2])
    s = crdistr.probit(np.float32(0.975))
    assert np.ndim(s) == 0 and float(s) == rounded(exact("probit", np.float32(0.975)), 24)
    # a float64 or a list becomes float32 first (rounded to nearest), and the result is float32
    assert crdistr.probit([0.975]).dtype == np.float32
    assert crdistr.probit(np.float64(0.975)) == s
    assert crdistr.erfinv(0.5) == crdistr.erfinv(np.float32(0.5))


def test_environment_left_alone():
    # the library sets round-to-nearest for itself and restores the caller's state; NumPy's error state is Python's
    before = np.geterr()
    crdistr.erfinv(np.array([0.5, 2.0], np.float32))
    assert np.geterr() == before


def test_version():
    assert crdistr.version().count(".") == 2


def exact_fn(f, x):
    x = mpmath.mpf(float(x))
    if f == "ndtr":
        return mpmath.ncdf(x)
    if f == "log_ndtr":
        return mpmath.log(mpmath.ncdf(x))
    if f == "normpdf":
        return mpmath.npdf(x)
    return mpmath.log(x / (1 - x))


@pytest.mark.parametrize("f", ["ndtr", "log_ndtr", "normpdf", "logit"])
def test_functions_of_x_correctly_rounded(f):
    out = []
    while len(out) < 300:
        if f == "logit":
            x = np.float32(rng.uniform(0, 1) if rng.random() < 0.5 else 2.0 ** rng.uniform(-60, -1))
            ok = 0 < x < 1 and x != 0.5
        else:
            x = np.float32(rng.uniform(-14, 6) if rng.random() < 0.7 else -(10 ** rng.uniform(1, 30)))
            ok = abs(float(exact_fn(f, x))) > 2.0 ** -120 if f != "log_ndtr" else x < 5
        if ok:
            out.append(x)
    x = np.array(out, np.float32)
    y = getattr(crdistr, f)(x)
    assert y.dtype == np.float32
    for a, b in zip(x, y):
        v = exact_fn(f, a)
        if abs(v) > 3e38:
            assert np.isinf(b), (f, float(a))
        else:
            assert float(b) == rounded(v, 24), (f, float(a))


def test_log_ndtr_far():
    assert float(crdistr.log_ndtr(np.float32(-40))) == rounded(mpmath.log(mpmath.ncdf(-40)), 24)
    assert crdistr.log_ndtr(np.float32(-1e20)) == -np.inf   # -5e39, past binary32
    assert crdistr.ndtr(np.float32(-40)) == 0


def fnv(y):
    h = 0xcbf29ce484222325
    for u in np.asarray(y, np.float32).view(np.uint32).tolist():
        for b in range(4):
            h ^= (u >> (8 * b)) & 0xff
            h = (h * 0x100000001b3) % 2 ** 64
    return h


def test_stream_fingerprints():
    # crdist-check's recorded fingerprints (seed 1, substream 0, index 0, 4096 draws): the binding gives the C bits
    assert fnv(crdistr.Stream(1).normal(4096)) == 0xe0b080312afb39d4
    assert fnv(crdistr.Stream(1).gamma(4096, 2.5)) == 0xc6a5cb829535e89e
    assert fnv(crdistr.Stream(1).beta(4096, 0.01, 0.02)) == 0xa972487992e10f5b


def test_stream_index_and_substreams():
    s = crdistr.Stream(7, substream=3)
    a = s.words(10)
    assert s.index == 10 and a.dtype == np.uint32
    assert np.array_equal(crdistr.Stream(7, 3, 4).words(6), a[4:])          # cut by index
    assert not np.array_equal(crdistr.Stream(7, 4).words(10), a)            # another substream
    t = crdistr.Stream(9)
    t.normal(5)
    assert t.index == 5                                                    # a normal takes one word
    g = crdistr.Stream(9).gamma(3, -1.0)
    assert np.isnan(g).all()
    # Random123's known answer for Philox4x32-10 at counter 0, key 0
    assert crdistr.Stream(0).words(4).tolist() == [0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8]


def fnv64(y):
    h = 0xcbf29ce484222325
    for v in np.asarray(y, np.int64).view(np.uint64).tolist():
        for b in range(8):
            h ^= (v >> (8 * b)) & 0xff
            h = (h * 0x100000001b3) % 2 ** 64
    return h


def test_discrete():
    # crdist-check's recorded fingerprints (stream (1, 0, 0), 4096 draws)
    assert fnv64(crdistr.Stream(1).poisson(4096, 3)) == 0xf651042b777b9148
    assert fnv64(crdistr.Stream(1).binomial(4096, 1000, 0.3)) == 0xf7ddf5e327c935ad
    s = crdistr.Stream(5)
    k = s.binomial(100, 0, 0.3)
    assert k.dtype == np.int64 and (k == 0).all() and s.index == 0     # no trials, no words
    assert (s.poisson(3, -1) == -1).all() and s.index == 0             # outside the range: -1, no words
    assert abs(crdistr.Stream(9).poisson(20000, 4.5).mean() - 4.5) < 0.1


def fnv_f64(y):
    h = 0xcbf29ce484222325
    for v in np.asarray(y, np.float64).view(np.uint64).tolist():
        for b in range(8):
            h ^= (v >> (8 * b)) & 0xff
            h = (h * 0x100000001b3) % 2 ** 64
    return h


def test_binary64_samplers():
    # crdist-check's recorded fingerprints (stream (1, 0, 0), 4096 draws)
    assert fnv_f64(crdistr.Stream(1).uniform(4096, dtype=np.float64)) == 0x9aa59947648adde1
    assert fnv_f64(crdistr.Stream(1).exponential(4096, dtype=np.float64)) == 0x63806eeb79eaa165
    # two words a draw, and the uniform is (2m + 1) 2^-53 for m the top 52 bits of the two words, exactly
    s = crdistr.Stream(11, 2)
    u = s.uniform(100, dtype=np.float64)
    assert s.index == 200 and u.dtype == np.float64
    w = crdistr.Stream(11, 2).words(200).astype(np.uint64)
    m = (w[0::2] << np.uint64(20)) | (w[1::2] >> np.uint64(12))
    assert np.array_equal(u, (2 * m.astype(np.float64) + 1) * 2.0 ** -53)
    # the exponential is -log of the same uniform, rounded once: within an ulp of the C library's log
    e = crdistr.Stream(11, 2).exponential(100, dtype=np.float64)
    assert (np.abs(e + np.log(u)) <= np.spacing(e)).all()
    assert crdistr.Stream(1).uniform(3).dtype == np.float32           # the default is unchanged
    with pytest.raises(ValueError):
        crdistr.Stream(1).uniform(3, dtype=np.float16)
