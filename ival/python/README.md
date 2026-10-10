# ival for Python

Interval arithmetic over NumPy arrays: every result is the tightest
interval of doubles around the exact one (IEEE 1788.1). It calls
crnumerics' C library `libival`, whose functions are CORE-MATH's,
correctly rounded.

```python
import ival, numpy as np

x = ival.Interval(1, 2)
ival.exp(x) * x + 1                # Interval(3.718281828459045, 15.7781121978613)
ival.Interval(0.1) + 0.2           # Interval(0.3, 0.30000000000000004): the exact sum inside
ival.Interval.from_text("[0.1]")   # Interval(0.09999999999999999, 0.1): the decimal 0.1 inside
X = ival.Interval(np.linspace(0, 1, 10**6), np.linspace(0, 1, 10**6) + 1e-3)
ival.sin(X)                        # a million intervals in one call
```

## Installing

On Linux x86-64 and aarch64 (glibc 2.26 or later), the wheels on PyPI
carry the library, so nothing else needs installing. The package is
`crival` on PyPI; the module is `ival`:

```sh
pip install crival
python3 -c 'import ival; print(ival.exp(ival.Interval(1)))'
```

Elsewhere, or to use a libival you installed, install `libival` first,
from the root of crnumerics, then the package:

```sh
make install-ival PREFIX=$HOME/.local     # or /usr/local; installs libival.so.0 and ival.pc
pip install ./ival/python                 # numpy is the one dependency
```

`ival/python/build-wheel.sh` builds a wheel that carries the library,
for the glibc it was built on and newer. The wheels on PyPI are built in
the manylinux_2_28 containers by `.github/workflows/wheels.yml`.

The binding finds the library through `IVAL_LIBRARY` (a path), then a
copy inside the package (a wheel's), then the system's linker paths, then `pkg-config --variable=libdir ival` (for a
prefix like `$HOME/.local`, `PKG_CONFIG_PATH` must include its
`lib/pkgconfig`).
`ival.library_path` says which it loaded, and `ival.version()` the
library's version. It works with libival 0.1.0 and later; 0.2.0 fixes sinRev, cosRev
and tanRev, which in 0.1.0 could miss a set. Python 3.9 or
later; tested on Linux x86-64 and aarch64.

## What there is

- **`Interval(lo, hi)`**, or `Interval(a)` for the smallest interval
  holding `a`. One interval holds two Python floats; an array of them two
  float64 arrays, broadcast as NumPy does. The empty interval is
  `[nan, nan]`; `Interval(lo, hi)` is empty unless `lo <= hi`
  (1788's numsToInterval). `Interval.from_text("[0.1, 0.2]")` reads 1788's
  text forms, a string or an array of strings, rounding each decimal
  outward.
- **Numbers that are not doubles round outward.** A Python int beyond
  2^53, a `Fraction`, a `Decimal`, a long double, an int64 array: each
  becomes the two doubles around it, never the nearest one, as an end,
  as an operand (`x + Fraction(1, 3)`) or as a member to test
  (`2**53 + 1 in x`). A float is taken as the double it is, so `0.1` is
  already rounded: give decimals as text or `Decimal`.
- **Arithmetic:** `+ - * /`, unary `-`, `abs`, `**` (an integer power
  is `pown`, tight; any other `pow`), `&` (intersection), `|` (hull).
- **40 functions:** `exp exp2 exp10 expm1 log log2 log10 log1p sqrt cbrt
  rsqrt sin cos tan sinpi cospi tanpi asin acos atan asinpi acospi atanpi
  sinh cosh tanh asinh acosh atanh erf erfc tgamma sqr recip sign ceil
  floor trunc round roundeven`, and `atan2 hypot pow min max cancelminus
  cancelplus pown rootn fma`.
- **The accurate mode:** `acc_exp`, `acc_sin`, ... (the first 32 above,
  and `acc_atan2`, `acc_hypot`, `acc_pow`) are at most one ulp wider than
  the tight result, and faster on arrays, when the environment variable
  `IVAL_CRMVEC` names crmvec's `libmvec.so.1` (x86-64 with AVX2 and FMA).
  Without it they give the tight result.
- **Reverse operations** (the tightest interval around
  `{x in X : f(x) in C}`, X the whole line if omitted): `sqrrev absrev
  coshrev sinrev cosrev tanrev pownrev mulrev powrev1 powrev2`.
- **Measures and tests:** `.inf .sup .mid .wid .rad .mag .mig`,
  `is_empty() is_entire() is_singleton()`, `subset interior disjoint
  equal`, `contains(X, v)` and `v in X`.

The rounding mode is the library's business: it sets what it needs and
restores the caller's, so nothing here changes how the rest of a program
rounds. A call releases the GIL, and the library is thread-safe.

## Speed, and pyinterval

pyinterval (on CRlibm) was Python's correctly rounded interval library.
Its last release was in March 2017, and `pip install pyinterval` fails on
Python 3.12: its crlibm imports `distutils.command.upload`, which Python
3.12 lacks and setuptools 75 dropped. Built with setuptools 74 and no
build isolation, it and this binding were run on the same inputs by
`bench/pyinterval_compare.py`, on an AMD Ryzen 5 PRO 5650U, 2026-10-09:

- **Bounds:** on 2,000 points each of exp, log, sin, cos, tan, atan,
  sqrt, sinh, cosh, expm1 and log1p, both tight and the same, bound for
  bound. pyinterval's tanh is a quotient of sinh and cosh, every bound
  rounded outward: valid, but not tight on any of the 2,000 points, up to
  4 doubles too wide.
- **Time per interval,** for intervals of width 0.5 (and 1, for `b`):

  | | pyinterval | ival, one at a time | ival, an array of 20,000 |
  |---|---:|---:|---:|
  | exp | 11.0 µs | 1.8 µs | 25 ns |
  | sin | 11.7 µs | 1.8 µs | 141 ns |
  | atan | 10.8 µs | 1.7 µs | 31 ns |
  | `a*b + 2` | 22.2 µs | 4.5 µs | 20 ns |

  One interval at a time costs about a microsecond of ctypes per call;
  arrays pay it once per call.

## Tests

`make ival-python-check` at the root of crnumerics runs `tests/` against
the library just built (it needs pytest and mpmath): bounds against
mpmath at 300 bits and exact fractions, arrays against one interval at
a time, outward rounding of numbers that are not doubles, the reverse
operations and the accurate mode.
