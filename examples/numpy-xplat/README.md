# NumPy's random draws across platforms

Does the same NumPy, with the same seed, give the same numbers on every
platform? `numpy_xplat.py OUT` writes a report of 136 tests, each a hash of
10^6 values:
- every `Generator` distribution, and the legacy `RandomState`;
- NumPy's own `exp`, `log`, ... in float64 and float32;
- the platform's C library through Python's `math`;
- crdist's samplers, where `crdistr` is installed.

The controls (`random`, `integers`, `permutation`, and every function's
inputs) must be the same everywhere. `compare.py DIR` groups the hosts
test by test; `diff.py` compares saved arrays element by element.
`.github/workflows/numpy-xplat.yml` runs it on GitHub's Windows and Ubuntu
runners.

NumPy's own policy guarantees a stream only on the same build and the same
machine, so a difference is not a NumPy bug. The script measures how often
that applies, and why.

## Results, 2026-10-14

NumPy 2.5.4 from PyPI's wheels on eleven platforms: Linux on glibc 2.38,
2.39 and 2.41 (x86-64 with and without AVX-512, aarch64), Alpine (musl
1.2.6), macOS on Apple M1, and Windows on x86-64 and Arm64 (GitHub's
runners). Each ran the script twice, with identical reports.

| tests | differ somewhere | the same on all eleven |
|---|---|---|
| `Generator`, 41 | 27 | the integer distributions, `standard_normal`, `standard_cauchy`, `triangular`, `wald`, `multivariate_normal` by Cholesky |
| legacy `RandomState`, 16 | 12 | `binomial`, `poisson`, `rand` |
| NumPy's own functions, 35 | 31 | `sin` and `cos` in float32, `tanh` |
| the C library through `math`, 18 | 18 | none |
| **crdist's samplers, 11** | **0** | all eleven |

The differences are in the last bits: macOS and Windows on Arm differ
from glibc on 9 to 30% of the draws of `uniform`, `gamma`, `beta` and the
legacy `normal`, by at most 7.3e-14. They come from four causes, each
tested:
- **The C library.** None is correctly rounded, and each is wrong on
  other inputs. Against mpmath, `expm1` is wrong on 8.6% of calls on
  glibc and musl, 13.7% on Windows and 0.13% on macOS. Within glibc the
  version decides, not the CPU.
- **FMA contraction.** NumPy's macOS and Windows-on-Arm builds fuse
  `uniform`'s `-3 + 10 r` into one rounding (20,000 of 20,000 sampled
  values), and the samplers built on such expressions follow. Linux on
  Arm and Windows on x86-64 do not fuse.
- **CPU dispatch.** GitHub's `ubuntu-24.04` runner gave the same results
  three times, then landed on a CPU with AVX-512, and 28 tests changed:
  NumPy's `exp`, `log`, `expm1`, `tan`, `cbrt` and others, and
  `multivariate_normal`. Same workflow, image and C library.
- **LAPACK,** for `multivariate_normal`'s SVD: five groups by CPU family.

The legacy API's integer draws are C `long`, int32 on Windows: other
bytes, the same values. The report hashes integer results as int64 too,
and `compare.py` groups by that.

crdistr has no Windows or musl wheel yet. The workflow builds `libcrdist`
as a DLL with MSYS2's Clang, and the Alpine run built it with GCC. One
flag was needed on x86-64: CORE-MATH's `exp.c` and `erfc.c` call libm's
`roundeven` there, which glibc has and neither musl nor MinGW-w64 does.
`-msse4.1` makes it one instruction, which is exact either way.
