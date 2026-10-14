"""Tests of crfix's Python binding, against a reference independent of the library and of MPFR: Python's decimal
(exp and ln correctly rounded), widened by a generous error bound and refined until the rounding is decided, and
exact fractions and integer square roots for sqrt, rsqrt, recip and div and for every exact case (ties included).
Run with libcrfix on the linker's path or CRFIX_LIBRARY naming it: python3 -m pytest fix/python/tests"""
from decimal import Decimal, localcontext, MAX_EMAX, MIN_EMIN
from fractions import Fraction
import functools
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
    """t in the open (lo, hi), t irrational, rounded in mode; None when (lo, hi) leaves it open. An end on an integer
    is not t, so the floor is decided when no integer lies strictly inside (GELU at large v is just below v)"""
    fl, fh = math.floor(lo), math.ceil(hi) - 1
    if mode == "nearest":
        a, b = math.floor(lo + Fraction(1, 2)), math.ceil(hi + Fraction(1, 2)) - 1
        return a if a == b else None
    if fl != fh:
        return None
    return {"down": fl, "up": fl + 1, "zero": fl if fl >= 0 else fl + 1}[mode]


def _sat(r, bits):
    if bits in ("u8", "u16"):
        return max(0, min((1 << int(bits[1:])) - 1, r))
    lim = 1 << (bits - 1)
    return max(-lim, min(lim - 1, r))


def _sqrt_exact(R):
    """sqrt of a Fraction R >= 0 as a Fraction when rational, else None"""
    a, b = math.isqrt(R.numerator), math.isqrt(R.denominator)
    return Fraction(a, b) if a * a == R.numerator and b * b == R.denominator else None


@functools.lru_cache(maxsize=None)
def _pi(prec):
    """pi to prec digits, by the series in the decimal module's documentation"""
    with localcontext() as ctx:
        ctx.prec = prec + 2
        lasts, t, s, n, na, d, da = 0, Decimal(3), Decimal(3), 1, 0, 0, 24
        while s != lasts:
            lasts = s
            n, na = n + na, na + 8
            d, da = d + da, da + 32
            t = (t * n) / d
            s += t
        return s


def _phi(v, P):
    """Phi(v), the standard normal distribution function, and a bound on its error. |v| > 40: within 10^-340 of 0
    or 1 (e^-800 / 40 < 10^-349). Else erf(|v| / sqrt 2) = |v| sqrt(2 / pi) e^(-v^2 / 2) sum_n v^2n / (2n + 1)!!, a
    series of positive terms, at 400 more digits than P (1 - erf near v = -40 is 10^-349), until a term is below the
    sum's last digits and every later ratio is under 1/2"""
    if abs(v) > 40:
        return (Fraction(1) if v > 0 else Fraction(0)), Fraction(1, 10 ** 340)
    W = P + 400
    with localcontext() as ctx:
        ctx.prec, ctx.Emax, ctx.Emin = W, MAX_EMAX, MIN_EMIN
        V = Decimal(v.numerator) / Decimal(v.denominator)
        u = V * V
        t = s = Decimal(1)
        n = 0
        while True:
            n += 1
            t = t * u / (2 * n + 1)
            s += t
            if 2 * n + 3 > 2 * u and t < s * Decimal(10) ** (-W - 5):
                break
        erf = abs(V) * (2 / _pi(W)).sqrt() * (-u / 2).exp() * s
        phi = (1 + erf) / 2 if V > 0 else (1 - erf) / 2
        return Fraction(phi), Fraction(n + 10) * Fraction(10) ** (10 - W)


@functools.lru_cache(maxsize=None)
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
        if fn == "silu":   # v sigmoid(v): v - v c or v c, the error relative to |v| c
            c = Fraction(1 / (1 + abs(V).exp()))
            return (v * c if v < 0 else v - v * c), abs(v) * c * Fraction(k)
        if fn == "gelu":
            if abs(v) > 40:   # just below v, or just above 0: an interval on that side alone
                h = abs(v) / Fraction(2 * 10 ** 340)
                return (v - h if v > 0 else -h), h
            phi, perr = _phi(v, P)
            return v * phi, abs(v) * perr
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
    if fn in ("tanh", "silu", "gelu") and v == 0:
        return Fraction(0)
    return None


def _through(fn, v, scale, z, mode, bits):
    """round(fn(v) * scale) + z in mode, saturated to bits, for a transcendental fn: an enclosure wholly past the
    container saturates however it rounds (exp(11926) has 5,180 digits)"""
    e = _exact(fn, v)
    low, high = (0, (1 << int(bits[1:])) - 1) if bits in ("u8", "u16") else (-(1 << (bits - 1)), (1 << (bits - 1)) - 1)
    if e is not None:
        return _sat(_round(e * scale, mode) + z, bits)
    P = 40
    while P < 5000:
        d, err = _approx(fn, v, P)
        lo, hi = (d - err) * scale, (d + err) * scale   # rounded before z is added: toward zero is the value's
        if lo + z > high or hi + z < low:
            return high if lo + z > high else low
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


def test_unsigned_containers():
    """u8 and u16: the 32-bit result clamped to [0, 2^B - 1]; and TFLite's uint8 logistic, input and output"""
    xs = np.arange(-128, 128)
    for fn in crfix.FUNCTIONS:
        for bits, top in (("u8", 255), ("u16", 65535)):
            for mode in MODES:
                assert np.array_equal(crfix.eval(fn, xs, 5, 7, bits, mode), np.clip(crfix.eval(fn, xs, 5, 7, 32, mode), 0, top))
    inp, out = crfix.Affine(0.0625, 128, "u8"), crfix.Affine(1 / 256, 0, "u8")
    for fn in ("sigmoid", "gelu", "silu"):
        for mode in MODES:
            t = crfix.table_affine(fn, inp, out, mode)
            assert list(t) == list(crfix.eval_affine(fn, np.arange(256), inp, out, mode))
            for q in range(256):
                v = Fraction(inp.scale) * (q - inp.zero_point)
                assert t[q] == _through(fn, v, 1 / Fraction(out.scale), out.zero_point, mode, "u8"), (fn, q, mode)


def test_named_functions_and_scalars():
    assert crfix.exp(0, 4, 14, 16) == 16384 and isinstance(crfix.exp(0, 4, 14, 16), int)
    assert crfix.sqrt(2, 0, 14, 16) == 23170 and crfix.sqrt(2, 0, 14, 16, "up") == 23171
    assert crfix.sqrt(9, 2, 0, 8) == 2   # sqrt(9/4) = 3/2, a tie: to even
    assert crfix.log2(8, 0, 0, 8) == 3 and crfix.sigmoid(0, 3, 7, 8) == 64
    assert crfix.gelu(0, 4, 7, 8) == 0 and crfix.silu(0, 4, 7, 8) == 0 and crfix.gelu(1 << 20, 16, 7, 32) == 16 << 7
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


def _layernorm_ref(x, fin, fout, eps, fe, gamma, fg, beta, fb, bits, mode):
    """exact: T = a / sqrt(D) + b with D = var + eps, a = gamma (v - mean) 2^fout, b = beta 2^fout (fractions); floor(T)
    the largest m with m - b <= a / sqrt(D), decided by comparing squares (a float estimate only starts the search)"""
    n = len(x)
    v = [Fraction(int(t), 1 << fin) for t in x]
    mean = sum(v) / n
    D = sum((t - mean) ** 2 for t in v) / n + Fraction(eps, 1 << fe)
    out = []
    for i in range(n):
        a = (Fraction(int(gamma[i]), 1 << fg) if gamma is not None else 1) * (v[i] - mean) * (1 << fout)
        b = (Fraction(int(beta[i]), 1 << fb) if beta is not None else Fraction(0)) * (1 << fout)
        if D == 0:
            a = Fraction(0)
            D = Fraction(1)

        def ge(L):   # a / sqrt(D) >= L, exactly
            if a >= 0:
                return L <= 0 or L * L * D <= a * a
            return L < 0 and L * L * D >= a * a

        def eq(L):   # a / sqrt(D) == L
            return (L >= 0) == (a >= 0) and L * L * D == a * a

        est = math.floor(float(a) / math.sqrt(float(D)) + float(b)) if D else 0
        m = est
        while not ge(m - b):
            m -= 1
        while ge(m + 1 - b):
            m += 1
        exact = eq(m - b)
        if mode == "down":
            r = m
        elif mode == "up":
            r = m + (not exact)
        elif mode == "zero":
            r = m if ge(-b) else m + (not exact)
        else:
            h = Fraction(2 * m + 1, 2) - b
            r = m if not ge(h) else m + 1 if not eq(h) else m + (m & 1)
        out.append(_sat(r, bits))
    return out


def test_layernorm_against_fractions():
    for k in range(60):
        n = int(rng.integers(1, 40))
        x = rng.integers(-128, 128, n) if k % 3 else rng.integers(-32768, 32769, n)
        if k % 7 == 0:
            x = np.full(n, 5)
        fin, fout, fe, fg, fb = (int(t) for t in rng.integers(0, 16, 5))
        eps = 0 if k % 4 == 0 else int(rng.integers(0, 1 << 16))
        gamma = rng.integers(-300, 300, n) if k % 2 else None
        beta = rng.integers(-300, 300, n) if k % 5 else None
        bits = (8, 16, 32, "u8", "u16")[k % 5]
        for mode in MODES:
            got = crfix.layernorm(x, fin, fout, eps, fe, gamma, fg, beta, fb, bits, mode)
            assert list(got) == _layernorm_ref(x, fin, fout, eps, fe, gamma, fg, beta, fb, bits, mode), (k, mode)
    assert list(crfix.layernorm([1, 3], 0, 4, bits=16)) == [-16, 16]   # -1 and 1 exactly
    assert list(crfix.layernorm([1, 3], 0, 0, beta=[1, 1], fb=1, bits=16)) == [0, 2]   # -1/2 and 3/2: ties to even
    rows = rng.integers(-128, 128, (3, 9))
    y = crfix.layernorm(rows, 3, 7, eps=1, fe=4, bits=8)
    assert y.shape == (3, 9) and all(np.array_equal(y[r], crfix.layernorm(rows[r], 3, 7, eps=1, fe=4, bits=8)) for r in range(3))
    with pytest.raises(ValueError):
        crfix.layernorm([40000], 0, 0)
    with pytest.raises(ValueError):
        crfix.layernorm([1, 2], 0, 0, gamma=[1])

