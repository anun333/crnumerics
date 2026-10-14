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
