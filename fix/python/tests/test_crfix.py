"""Tests of crfix's Python binding, against a reference independent of the library and of MPFR: Python's decimal
(exp and ln correctly rounded), widened by a generous error bound and refined until the rounding is decided, and
exact fractions and integer square roots for sqrt, rsqrt, recip and div and for every exact case (ties included).
Run with libcrfix on the linker's path or CRFIX_LIBRARY naming it: python3 -m pytest fix/python/tests"""
from decimal import Decimal, localcontext, MAX_EMAX, MIN_EMIN
from fractions import Fraction
import math
import re
import subprocess
import sys

import numpy as np
import pytest

import crfix

rng = np.random.default_rng(20261013)
MODES = ("nearest", "up", "down", "zero")


def _round(t, mode):
    """an exact Fraction rounded in mode"""
    f = math.floor(t)
    if mode == "down":
        return f
    if mode == "up":
        return math.ceil(t)
    if mode == "zero":
        return math.trunc(t)
    r = t - f
    return f if r < Fraction(1, 2) else f + 1 if r > Fraction(1, 2) else f + (f & 1)


def _round_iv(lo, hi, mode):
    """t in the open (lo, hi), t irrational, rounded in mode; None when (lo, hi) leaves it open"""
    fl, fh = math.floor(lo), math.floor(hi)
    if mode == "nearest":
        a, b = math.floor(lo + Fraction(1, 2)), math.floor(hi + Fraction(1, 2))
        return a if a == b else None
    if fl != fh:
        return None
    return {"down": fl, "up": fl + 1, "zero": fl if fl >= 0 else fl + 1}[mode]


def _sat(r, bits):
    lim = 1 << (bits - 1)
    return max(-lim, min(lim - 1, r))


def _sqrt_exact(R):
    """sqrt of a Fraction R >= 0 as a Fraction when rational, else None"""
    a, b = math.isqrt(R.numerator), math.isqrt(R.denominator)
    return Fraction(a, b) if a * a == R.numerator and b * b == R.denominator else None


def _approx(fn, v, P):
    """fn(v) for a Fraction v, at P digits: the value and a bound on its error (generous: a few units of the last
    digit, times 1 + |v| for what an argument's rounding does to exp). Relative for exp and exp2, whose values can be
    10^-4000; sigmoid and tanh through their complement to 1, sigmoid(-|v|) and 1 - tanh|v| = 2 e / (1 + e) with
    e = e^-2|v|, relative too, then subtracted from 1 exactly (sigmoid(11926) is below 1 by 10^-5180)"""
    with localcontext() as ctx:
        ctx.prec, ctx.Emax, ctx.Emin = P, MAX_EMAX, MIN_EMIN
        V = Decimal(v.numerator) / Decimal(v.denominator)
        k = (1 + abs(V)) * Decimal(10) ** (6 - P)
        if fn in ("exp", "exp2"):
            d = V.exp() if fn == "exp" else (V * Decimal(2).ln()).exp()
            return Fraction(d), Fraction(d * k)
        if fn in ("log", "log2"):
            d = V.ln() if fn == "log" else V.ln() / Decimal(2).ln()
            return Fraction(d), Fraction((abs(d) + 1) * k)
        if fn == "sigmoid":
            c = 1 / (1 + abs(V).exp())
            return (Fraction(c) if V < 0 else 1 - Fraction(c)), Fraction(c * k)
        e = (-2 * abs(V)).exp()
        c = 2 * e / (1 + e)
        t = 1 - Fraction(c)
        return (-t if V < 0 else t), Fraction(c * k)


def _exact(fn, v):
    """fn(v) as a Fraction when it is rational (exp(0), exp2 of an integer, log(1), log2 of a power of 2,
    sigmoid(0), tanh(0)), else None"""
    if fn == "exp" and v == 0:
        return Fraction(1)
    if fn == "exp2" and v.denominator == 1:
        return Fraction(2) ** int(v)
    if fn == "log" and v == 1:
        return Fraction(0)
    if fn == "log2" and v > 0 and v.numerator & (v.numerator - 1) == 0 and v.denominator & (v.denominator - 1) == 0:
        return Fraction(v.numerator.bit_length() - v.denominator.bit_length())
    if fn == "sigmoid" and v == 0:
        return Fraction(1, 2)
    if fn == "tanh" and v == 0:
        return Fraction(0)
    return None


def _through(fn, v, scale, z, mode, bits):
    """round(fn(v) * scale) + z in mode, saturated to bits, for a transcendental fn: an enclosure wholly past the
    container saturates however it rounds (exp(11926) has 5,180 digits)"""
    e, lim = _exact(fn, v), 1 << (bits - 1)
    if e is not None:
        return _sat(_round(e * scale, mode) + z, bits)
    P = 40
    while P < 5000:
        d, err = _approx(fn, v, P)
        lo, hi = (d - err) * scale, (d + err) * scale   # rounded before z is added: toward zero is the value's
        if lo + z >= lim or hi + z <= -lim - 1:
            return _sat(lim if lo + z >= lim else -lim - 1, bits)
        r = _round_iv(lo, hi, mode)
        if r is not None:
            return _sat(r + z, bits)
        P *= 2
    raise AssertionError("reference undecided: %s %s" % (fn, v))


def reference(fn, x, fin, fout, bits, mode):
    """crfix.h's specification of fn at x 2^-fin"""
    v, s, lim = Fraction(x, 1 << fin), Fraction(1 << fout), 1 << (bits - 1)
    if fn in ("log", "log2") and v <= 0:
        return -lim
    if fn == "sqrt":
        if v < 0:
            return 0
        R = v * s * s
        e = _sqrt_exact(R)
        if e is not None:
            return _sat(_round(e, mode), bits)
        k = 64   # sqrt(R) between isqrt(floor(R 4^k)) / 2^k and isqrt(ceil(R 4^k)) / 2^k + 2^-k, strictly
        while True:
            lo = Fraction(math.isqrt(math.floor(R * 4 ** k)), 2 ** k)
            hi = Fraction(math.isqrt(math.ceil(R * 4 ** k)) + 1, 2 ** k)
            r = _round_iv(lo, hi, mode)
            if r is not None:
                return _sat(r, bits)
            k *= 2
    if fn == "rsqrt":
        if v <= 0:
            return lim - 1
        R = s * s / v
        e = _sqrt_exact(R)
        if e is not None:
            return _sat(_round(e, mode), bits)
        k = 64
        while True:
            lo = Fraction(math.isqrt(math.floor(R * 4 ** k)), 2 ** k)
            hi = Fraction(math.isqrt(math.ceil(R * 4 ** k)) + 1, 2 ** k)
            r = _round_iv(lo, hi, mode)
            if r is not None:
                return _sat(r, bits)
            k *= 2
    if fn == "recip":
        return lim - 1 if v == 0 else _sat(_round(s / v, mode), bits)
    return _through(fn, v, s, 0, mode, bits)


def test_functions_against_decimal():
    """every int8 input and some int16 ones, every function and mode, at several formats; and the reference must
    tell the modes apart (a control: else a comparison blind to the mode would pass)"""
    xs = np.concatenate([np.arange(-128, 128), rng.integers(-32768, 32768, 40)]).astype(np.int32)
    differ = 0
    for fn in crfix.FUNCTIONS:
        for fin, fout, bits in ((0, 0, 8), (4, 7, 8), (7, 15, 16), (3, 12, 32), (15, 14, 16), (12, 30, 32)):
            for mode in MODES:
                got = crfix.eval(fn, xs, fin, fout, bits, mode)
                ref = [reference(fn, int(x), fin, fout, bits, mode) for x in xs]
                assert list(got) == ref, (fn, fin, fout, bits, mode,
                                          [(int(x), int(g), r) for x, g, r in zip(xs, got, ref) if g != r][:5])
            differ += int(np.sum(crfix.eval(fn, xs, fin, fout, bits, "down") != crfix.eval(fn, xs, fin, fout, bits)))
    assert differ > 0


def test_named_functions_and_scalars():
    assert crfix.exp(0, 4, 14, 16) == 16384 and isinstance(crfix.exp(0, 4, 14, 16), int)
    assert crfix.sqrt(2, 0, 14, 16) == 23170 and crfix.sqrt(2, 0, 14, 16, "up") == 23171
    assert crfix.sqrt(9, 2, 0, 8) == 2   # sqrt(9/4) = 3/2, a tie: to even
    assert crfix.log2(8, 0, 0, 8) == 3 and crfix.sigmoid(0, 3, 7, 8) == 64
    x = np.arange(-6, 6, dtype=np.int16).reshape(3, 4)
    y = crfix.tanh(x, 2, 7, 8)
    assert y.shape == (3, 4) and y.dtype == np.int32
    assert list(y.reshape(-1)) == [reference("tanh", int(v), 2, 7, 8, "nearest") for v in x.reshape(-1)]


def test_domain_edges():
    assert crfix.log(0, 3, 8, 16) == -32768 and crfix.log2(-5, 3, 2, 8, "up") == -128
    assert crfix.sqrt(-1, 0, 4, 16) == 0 and crfix.rsqrt(0, 2, 4, 16) == 32767 and crfix.recip(0, 5, 1, 8) == 127
    assert list(crfix.div([3, -3, 0], 0, 0, 0, 0, 16)) == [32767, -32768, 0]


def test_div_exact():
    a, b = rng.integers(-2 ** 20, 2 ** 20, 2000), rng.integers(-300, 300, 2000)
    for fa, fb, fout, bits in ((0, 0, 0, 8), (3, 7, 12, 16), (12, 0, 20, 32)):
        for mode in MODES:
            got = crfix.div(a, fa, b, fb, fout, bits, mode)
            lim = 1 << (bits - 1)
            for u, v, g in zip(a, b, got):
                u, v = int(u), int(v)
                if v == 0:
                    r = lim - 1 if u > 0 else -lim if u < 0 else 0
                else:
                    r = _sat(_round(Fraction(u, 1 << fa) / Fraction(v, 1 << fb) * (1 << fout), mode), bits)
                assert g == r, (u, fa, v, fb, fout, bits, mode)
    assert list(crfix.div(np.arange(1, 4), 0, 2, 0, 1, 8)) == [1, 2, 3]   # b broadcast


def test_table_equals_eval():
    xs = np.arange(-128, 128)
    for fn in crfix.FUNCTIONS:
        for mode in MODES:
            assert np.array_equal(crfix.table(fn, 5, 7, 8, mode), crfix.eval(fn, xs, 5, 7, 8, mode))


def _f32():
    return float(np.float32(2.0 ** rng.uniform(-10, 1)))


def test_affine_against_decimal():
    """random TFLite-like configurations, float32 scales; every int8 input and some 16-bit ones; every mode"""
    for fn in crfix.FUNCTIONS[:6]:
        for k in range(4):
            inp = crfix.Affine(_f32(), int(rng.integers(-128, 128)))
            if k == 0 and fn == "sigmoid":
                out = crfix.Affine(1 / 256, -128, 8)
            elif k == 0 and fn == "tanh":
                out = crfix.Affine(1 / 128, 0, 8)
            else:
                out = crfix.Affine(_f32(), int(rng.integers(-128, 128)), int(rng.choice([8, 16, 32])))
            qs = np.concatenate([np.arange(-128, 128), rng.integers(-32768, 32768, 16)])
            for mode in MODES:
                got = crfix.eval_affine(fn, qs, inp, out, mode)
                lim = 1 << (out.bits - 1)
                for q, g in zip(qs, got):
                    v = Fraction(inp.scale) * (int(q) - inp.zero_point)
                    if fn in ("log", "log2") and v <= 0:
                        r = -lim
                    else:
                        r = _through(fn, v, 1 / Fraction(out.scale), out.zero_point, mode, out.bits)
                    assert g == r, (fn, int(q), inp, out, mode)
                assert np.array_equal(crfix.table_affine(fn, inp, out, mode), got[:256])


def _softmax_ref(q, c, out, mode):
    q = [int(v) for v in q]
    m, n, lim = max(q), len(q), 1 << (out.bits - 1)
    inv = 1 / Fraction(out.scale)
    if all(v == m for v in q):   # 1/n exactly
        return [_sat(_round(Fraction(1, n) * inv, mode) + out.zero_point, out.bits)] * n
    res, c = [], Fraction(c)
    big = max(abs(c * (v - m)) for v in q)
    for i in range(n):
        P = 40
        while True:
            with localcontext() as ctx:
                ctx.prec, ctx.Emax, ctx.Emin = P, MAX_EMAX, MIN_EMIN
                C = Decimal(c.numerator) / Decimal(c.denominator)
                e = [(C * (v - m)).exp() for v in q]
                p = e[i] / sum(e)
                err = (p + 1) * (1 + Decimal(float(big))) * n * Decimal(10) ** (6 - P)
            r = _round_iv((Fraction(p) - Fraction(err)) * inv, (Fraction(p) + Fraction(err)) * inv, mode)
            if r is not None:
                res.append(_sat(r + out.zero_point, out.bits))
                break
            P *= 2
            assert P < 5000
    return res


def test_softmax_against_decimal():
    for k in range(30):
        n = int(rng.integers(1, 40))
        q = rng.integers(-128, 128, n)
        c = _f32() if k % 3 else 1.0 * _f32() / 8
        out = crfix.Affine(1 / 256, -128, 8) if k % 2 else crfix.Affine(_f32() / 64, int(rng.integers(-128, 128)), 16)
        for mode in MODES:
            assert list(crfix.softmax(q, c, out, mode)) == _softmax_ref(q, c, out, mode), (k, mode)
    assert list(crfix.softmax([5, 5], 1.0)) == [0, 0]   # 1/2 exactly: 128 - 128
    rows = rng.integers(-128, 128, (3, 7))
    y = crfix.softmax(rows, 0.1)
    assert y.shape == (3, 7) and all(np.array_equal(y[r], crfix.softmax(rows[r], 0.1)) for r in range(3))


def test_refusals():
    with pytest.raises(TypeError):
        crfix.exp(np.array([0.5]), 0, 0)
    with pytest.raises(ValueError):
        crfix.exp([2 ** 31], 0, 0)
    with pytest.raises(ValueError):
        crfix.eval("cos", [1], 0, 0)
    with pytest.raises(ValueError):
        crfix.exp([1], 0, 0, mode="even")
    with pytest.raises(ValueError):
        crfix.exp([1], 32, 0)
    with pytest.raises(ValueError):
        crfix.exp([1], 0, 0, bits=12)
    with pytest.raises(ValueError):
        crfix.eval_affine("sqrt", [1], crfix.Affine(0.5), crfix.Affine(0.5))
    with pytest.raises(ValueError):
        crfix.table_affine("exp", crfix.Affine(0.5, 0, 16), crfix.Affine(0.5))
    with pytest.raises(ValueError):
        crfix.softmax([1, 2], -1.0)


def test_version():
    v = crfix.version()
    assert v.count(".") == 2 and all(p.isdigit() for p in v.split("."))


def _cli(*args):
    return subprocess.run([sys.executable, "-m", "crfix", *args], capture_output=True, text=True)


def test_command_line():
    r = _cli("table", "tanh", "--fin", "5", "--fout", "7")
    assert r.returncode == 0 and [int(v) for v in r.stdout.split()] == list(crfix.table("tanh", 5, 7))
    r = _cli("table", "exp", "--fin", "4", "--fout", "3", "--bits", "16", "--mode", "up", "--format", "c", "--name", "e")
    assert r.returncode == 0 and "static const int16_t e[256] = {" in r.stdout
    body = r.stdout[r.stdout.index("{") + 1:r.stdout.index("}")]
    assert [int(v) for v in re.findall(r"-?\d+", body)] == list(crfix.table("exp", 4, 3, 16, "up"))
    r = _cli("table-affine", "sigmoid", "--in", "0.0478,3", "--out", "0x1p-8,-128", "--f32", "--format", "csv")
    want = crfix.table_affine("sigmoid", crfix.Affine(float(np.float32(0.0478)), 3), crfix.Affine(1 / 256, -128))
    lines = r.stdout.split()
    assert r.returncode == 0 and lines[0] == "x,value" and [int(l.split(",")[1]) for l in lines[1:]] == list(want)
    assert [int(l.split(",")[0]) for l in lines[1:]] == list(range(-128, 128))
    assert _cli("table", "exp", "--fin", "40", "--fout", "3").returncode == 2   # invalid: refused, nothing printed
    assert _cli("table-affine", "sqrt", "--in", "0.5", "--out", "0.5").returncode == 2

