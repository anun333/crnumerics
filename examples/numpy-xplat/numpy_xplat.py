#!/usr/bin/env python3
"""numpy_xplat.py OUT [--save TEST,...]: does the same NumPy, with the same seed, give the same numbers on every
platform? (2026-10-14.) Writes OUT/report.json: the platform, Python, NumPy and its
build and CPU features, and per test a sha256 of the result's bytes. With --save, also OUT/arrays/TEST.npy for the
tests named, to compare element by element where hashes differ.

  control.*  Generator random, integers, permutation, and the ufunc and libm inputs: exact arithmetic on PCG64's
             integers, or one IEEE operation at a time, so the same everywhere unless this harness is wrong. (The first
             version took Generator.uniform as a control and built the inputs with it: uniform differed on macOS
             arm64, its low + range * r fused into one FMA there (to be confirmed), and so did every input. uniform is
             a test now, gen.uniform.)
  gen.*      numpy.random.Generator(PCG64(SEED)), every distribution (a fresh generator per test), N draws
  legacy.*   numpy.random.RandomState(SEED), the API most existing code still calls
  ufunc.*    NumPy's own vectorized functions (np.exp, np.log, ...) on fixed float64 and float32 inputs
  libm.*     Python's math module on the same float64 inputs: the platform's C library, one call at a time
  crdist.*   crnumerics' crdist samplers (binary32, pip install crdistr), where crdistr is installed: the same seed
             and draws for comparison, built only on correctly rounded functions and a written specification
The inputs of ufunc and libm are a + (b - a) r with r from Generator.random, the product and the sum separate NumPy
operations, which no compiler can fuse (control.inputs_* shows them the same everywhere)."""
import hashlib
import json
import math
import os
import platform
import struct
import sys

import numpy as np

SEED, N, NL = 20261014, 1_000_000, 200_000


def h(a):
    a = np.ascontiguousarray(a)
    r = {"sha256": hashlib.sha256(a.tobytes()).hexdigest()[:16], "dtype": str(a.dtype), "shape": list(a.shape)}
    if a.dtype.kind in "iu":
        # the values, whatever the width: the legacy API's integer draws are C long, 32 bits on Windows and 64 elsewhere,
        # so their bytes differ where the values do not. For a little-endian int64 array this equals sha256.
        r["values"] = hashlib.sha256(a.astype("<i8").tobytes()).hexdigest()[:16]
    return r


def gen():
    return np.random.Generator(np.random.PCG64(SEED))


def tests():
    t = {}
    # the control
    t["gen.uniform"] = lambda: gen().uniform(-3.0, 7.0, N)
    t["control.random"] = lambda: gen().random(N)
    t["control.random_f32"] = lambda: gen().random(N, dtype=np.float32)
    t["control.integers"] = lambda: gen().integers(0, 10**9, N)
    t["control.permutation"] = lambda: gen().permutation(N)
    # Generator
    G = {
        "standard_normal": lambda g: g.standard_normal(N),
        "standard_normal_f32": lambda g: g.standard_normal(N, dtype=np.float32),
        "standard_exponential": lambda g: g.standard_exponential(N),
        "standard_exponential_f32": lambda g: g.standard_exponential(N, dtype=np.float32),
        "standard_exponential_inv": lambda g: g.standard_exponential(N, method="inv"),
        "standard_gamma_0.5": lambda g: g.standard_gamma(0.5, N),
        "standard_gamma_2.5": lambda g: g.standard_gamma(2.5, N),
        "standard_gamma_2.5_f32": lambda g: g.standard_gamma(2.5, N, dtype=np.float32),
        "gamma_30": lambda g: g.gamma(30.0, 1.0, N),
        "beta_0.3_2.2": lambda g: g.beta(0.3, 2.2, N),
        "beta_5_5": lambda g: g.beta(5.0, 5.0, N),
        "chisquare_4": lambda g: g.chisquare(4.0, N),
        "noncentral_chisquare": lambda g: g.noncentral_chisquare(3.0, 2.5, N),
        "f_5_7": lambda g: g.f(5.0, 7.0, N),
        "noncentral_f": lambda g: g.noncentral_f(5.0, 7.0, 2.0, N),
        "standard_t_3": lambda g: g.standard_t(3.0, N),
        "standard_cauchy": lambda g: g.standard_cauchy(N),
        "lognormal": lambda g: g.lognormal(0.0, 1.0, N),
        "logistic": lambda g: g.logistic(0.0, 1.0, N),
        "gumbel": lambda g: g.gumbel(0.0, 1.0, N),
        "laplace": lambda g: g.laplace(0.0, 1.0, N),
        "weibull_1.7": lambda g: g.weibull(1.7, N),
        "pareto_2.5": lambda g: g.pareto(2.5, N),
        "power_3.2": lambda g: g.power(3.2, N),
        "rayleigh": lambda g: g.rayleigh(1.0, N),
        "vonmises": lambda g: g.vonmises(1.0, 2.5, N),
        "wald": lambda g: g.wald(1.0, 2.0, N),
        "triangular": lambda g: g.triangular(0.0, 1.0, 3.0, N),
        "dirichlet": lambda g: g.dirichlet([0.5, 2.0, 3.0], N // 3),
        "multivariate_normal_svd": lambda g: g.multivariate_normal([0, 1, 2], [[2, .5, .3], [.5, 1, .2], [.3, .2, 3]], N // 3),
        "multivariate_normal_cholesky": lambda g: g.multivariate_normal([0, 1, 2], [[2, .5, .3], [.5, 1, .2], [.3, .2, 3]], N // 3, method="cholesky"),
        "poisson_3.5": lambda g: g.poisson(3.5, N),
        "poisson_150": lambda g: g.poisson(150.0, N),
        "binomial_40_0.3": lambda g: g.binomial(40, 0.3, N),
        "binomial_2000_0.4": lambda g: g.binomial(2000, 0.4, N),
        "negative_binomial": lambda g: g.negative_binomial(5, 0.3, N),
        "geometric": lambda g: g.geometric(0.2, N),
        "hypergeometric": lambda g: g.hypergeometric(50, 30, 20, N),
        "logseries": lambda g: g.logseries(0.9, N),
        "zipf_2.5": lambda g: g.zipf(2.5, N),
    }
    for k, f in G.items():
        t["gen." + k] = (lambda f: lambda: f(gen()))(f)
    # the legacy API
    L = {
        "standard_normal": lambda r: r.standard_normal(N),
        "normal": lambda r: r.normal(3.0, 2.0, N),
        "standard_exponential": lambda r: r.standard_exponential(N),
        "standard_gamma_2.5": lambda r: r.standard_gamma(2.5, N),
        "gamma_0.5": lambda r: r.gamma(0.5, 1.0, N),
        "beta_0.3_2.2": lambda r: r.beta(0.3, 2.2, N),
        "chisquare_4": lambda r: r.chisquare(4.0, N),
        "standard_t_3": lambda r: r.standard_t(3.0, N),
        "lognormal": lambda r: r.lognormal(0.0, 1.0, N),
        "standard_cauchy": lambda r: r.standard_cauchy(N),
        "vonmises": lambda r: r.vonmises(1.0, 2.5, N),
        "poisson_3.5": lambda r: r.poisson(3.5, N),
        "poisson_150": lambda r: r.poisson(150.0, N),
        "binomial_40_0.3": lambda r: r.binomial(40, 0.3, N),
        "multivariate_normal": lambda r: r.multivariate_normal([0, 1, 2], [[2, .5, .3], [.5, 1, .2], [.3, .2, 3]], N // 3),
        "rand": lambda r: r.rand(N),
    }
    for k, f in L.items():
        t["legacy." + k] = (lambda f: lambda: f(np.random.RandomState(SEED)))(f)
    # NumPy's vectorized functions, and the platform's libm through math, on the same inputs
    def u(a, b):   # one IEEE operation at a time: np.multiply, then np.add
        return np.add(np.multiply(gen().random(N), float(b) - float(a)), float(a))
    U = {
        "exp": (np.exp, math.exp, (-700, 700)), "exp_small": (np.exp, math.exp, (-5, 5)),
        "log": (np.log, math.log, (1e-3, 1e3)), "log2": (np.log2, math.log2, (1e-3, 1e3)),
        "log10": (np.log10, math.log10, (1e-3, 1e3)), "log1p": (np.log1p, math.log1p, (-0.9, 10)),
        "expm1": (np.expm1, math.expm1, (-5, 5)), "exp2": (np.exp2, lambda v: 2.0 ** v, (-50, 50)),
        "sin": (np.sin, math.sin, (-100, 100)), "cos": (np.cos, math.cos, (-100, 100)),
        "tan": (np.tan, math.tan, (-1.5, 1.5)), "arcsin": (np.arcsin, math.asin, (-1, 1)),
        "arctan": (np.arctan, math.atan, (-50, 50)), "sinh": (np.sinh, math.sinh, (-10, 10)),
        "cosh": (np.cosh, math.cosh, (-10, 10)), "tanh": (np.tanh, math.tanh, (-10, 10)),
        "cbrt": (np.cbrt, math.cbrt if hasattr(math, "cbrt") else None, (-1e3, 1e3)),
    }
    for a, b in sorted({v[2] for v in U.values()}):
        t["control.inputs_%g_%g" % (a, b)] = (lambda a, b: lambda: u(a, b))(a, b)
    for k, (nf, mf, (a, b)) in U.items():
        t["ufunc.%s_f64" % k] = (lambda nf, a, b: lambda: nf(u(a, b)))(nf, a, b)
        t["ufunc.%s_f32" % k] = (lambda nf, a, b: lambda: nf(u(a, b).astype(np.float32)))(nf, a, b)
        if mf is not None:
            t["libm.%s" % k] = (lambda mf, a, b: lambda: np.array([mf(v) for v in u(a, b)[:NL].tolist()]))(mf, a, b)
    def v(a, b):   # a second, independent input stream for pow's exponents
        return np.add(np.multiply(gen().random(2 * N)[N:], float(b) - float(a)), float(a))
    t["control.inputs_pow_exponents"] = lambda: v(-5, 5)
    t["ufunc.power_f64"] = lambda: np.power(u(0, 10), v(-5, 5))
    t["libm.pow"] = lambda: np.array([math.pow(x, y) for x, y in zip(u(0, 10)[:NL].tolist(), v(-5, 5)[:NL].tolist())])
    try:
        import crdistr
    except ImportError:
        return t
    C = {
        "uniform": lambda s: s.uniform(N), "normal": lambda s: s.normal(N), "exponential": lambda s: s.exponential(N),
        "gamma_2.5": lambda s: s.gamma(N, 2.5), "gamma_0.5": lambda s: s.gamma(N, 0.5), "beta_0.3_2.2": lambda s: s.beta(N, 0.3, 2.2),
        "chisquared_4": lambda s: s.chisquared(N, 4.0), "student_t_3": lambda s: s.student_t(N, 3.0),
        "poisson_3.5": lambda s: s.poisson(N, 3.5), "poisson_150": lambda s: s.poisson(N, 150.0),
        "binomial_40_0.3": lambda s: s.binomial(N, 40, 0.3),
    }
    for k, f in C.items():
        t["crdist." + k] = (lambda f: lambda: f(crdistr.Stream(SEED)))(f)
    return t


def _libc():
    """glibc's name and version from Python, or musl's from its loader (Python reports nothing for musl)"""
    lib = list(platform.libc_ver())
    if not lib[0] and sys.platform.startswith("linux"):
        import glob, subprocess
        for ld in glob.glob("/lib/ld-musl-*.so.1"):
            r = subprocess.run([ld], capture_output=True, text=True)
            ver = [w for w in (r.stderr or "").split("\n") if w.startswith("Version")]
            lib = ["musl", ver[0].split()[-1] if ver else ""]
    return lib


def _crdistr_version():
    try:
        import crdistr
        return crdistr.version()
    except Exception:
        return None


def main():
    out = sys.argv[1]
    save = set(sys.argv[3].split(",")) if len(sys.argv) > 3 and sys.argv[2] == "--save" else set()
    os.makedirs(os.path.join(out, "arrays"), exist_ok=True)
    try:
        from numpy._core._multiarray_umath import __cpu_features__ as feats
    except ImportError:
        feats = {}
    cfg = getattr(np.__config__, "CONFIG", {})
    rep = {
        "platform": platform.platform(), "machine": platform.machine(), "python": sys.version.split()[0],
        "numpy": np.__version__, "crdistr": _crdistr_version(), "libc": _libc(), "mac": platform.mac_ver()[0],
        "windows": list(platform.win32_ver()),
        "cpu_features": sorted(k for k, v in feats.items() if v),
        "simd": cfg.get("SIMD Extensions", {}), "compilers": {k: v.get("version") for k, v in cfg.get("Compilers", {}).items() if isinstance(v, dict)},
        "seed": SEED, "n": N, "tests": {},
    }
    for name, f in tests().items():
        a = f()
        rep["tests"][name] = h(a)
        if name in save or "all" in save:
            np.save(os.path.join(out, "arrays", name + ".npy"), a)
    with open(os.path.join(out, "report.json"), "w") as fh:
        json.dump(rep, fh, indent=1, sort_keys=True)
    print("%s numpy %s python %s: %d tests" % (platform.machine(), np.__version__, rep["python"], len(rep["tests"])))


if __name__ == "__main__":
    main()
