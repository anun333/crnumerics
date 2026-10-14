# History

**2026-10-14: crfix's layer normalization.** `crfix_layernorm` computes
y_i = γ_i(v_i − mean)/√(var + ε) + β_i, each rounded once, with no
approximation anywhere:
- **How:** z² = A²·2^w/M for integers A and M, so every rounding decision
  is a comparison of integer products below 2^320. An integer square root
  brackets each value, and a binary search settles it.
- **Inputs:** within the int16 range, with up to 65,536 values a vector.
- **Checked** against GMP rationals in the C check, and against exact
  fractions in Python's.
- **Cost:** 165 ns an element. `crfix.layernorm` in Python works over the
  last axis.

**2026-10-14: crfix gets SiLU, GELU and unsigned containers.**
`crfix_silu` is v·sigmoid(v). `crfix_gelu` is the exact v·Φ(v), not the
tanh approximation:
- **Below |v| = 8:** erf's series of positive terms, with e^(−v²/2) kept
  as a power of two times a 61-bit mantissa.
- **At |v| ≥ 8:** erfc's asymptotic series, whose remainder is bounded by
  the next term.
- **Both functions:** they lie strictly between v/2 and max(v, 0), and the
  enclosures are clipped to that range. The slow path has both, and the
  affine formats take both.

`8 | CRFIX_UNSIGNED` and `16 | CRFIX_UNSIGNED` are unsigned containers,
TFLite's uint8 among them, for results and for affine tables' inputs.

The check now calls the slow path directly for every approximated
function, since the new fast paths leave it no cases in the sample. The
Python tests check SiLU and GELU against `decimal`, with GELU's erf
computed from its series in the tests themselves. GELU takes 0.8 µs a
call, ten times sigmoid; SiLU takes 77 ns. The digest over eleven
functions is d7a3b901584298d7 on all four builds, and the Cortex-M0 build
is 50 KB of code, still without floating point.

**2026-10-14: crnumerics 0.8.0** (tag `v0.8.0`): crfix's fast paths for
the affine formats and softmax, its arrays and Python package, two fixes
to 0.7.0's affine results, and crfix on a Cortex-M0 (the entries below).
The candidate passed `make check` (26 verdicts, crfix's check with the
arrays and its install check among them) and the install checks on
cfarm421 and cfarm424. It also passed CI, whose crfix-mcu job gave the
same digest on the Cortex-M0. The tag's archive is the commit file for
file (302 files). **crfix 0.8.0 on PyPI**, its first upload: wheels for
x86-64 and aarch64 Linux and Apple Silicon macOS, each tested in a clean
venv before the upload, and the sdist. Installed from PyPI into a clean
venv, it loads its own library and passes the binding's 10 tests.
crlin, crdistr and crival were not uploaded, their code unchanged.

**2026-10-13: crfix for Python, and array functions; a rounding fix.**
`crfix_eval_n`, `crfix_eval_affine_n` and `crfix_div_n` handle an array
in one call. The Python package `crfix` (`fix/python`) takes NumPy integer
arrays: the nine functions, division, tables, TFLite's affine formats and
softmax. Its tests use a second reference independent of MPFR, Python's
`decimal` and exact fractions. That reference found a bug in 0.7.0: with
an output zero point, CRFIX_ZERO rounded y + zero point toward 0 rather
than y, as crfix.h specifies. The MPFR reference had made the same
mistake, so the C check passed. It is fixed in the fast path, the slow
path and the reference. Only affine and softmax results in that mode
change; the digest's other sections are identical, mode by mode. A wheel
workflow, `crfix-wheels.yml`, follows crlin's.

**2026-10-13: crfix softmax gets a fast path.** Each exponential is the
affine exp's 64-bit enclosure, the sum's bounds are 128-bit sums, and
each p_i's bounds come from a division rounded each way. A vector with
an undecided element goes to the slow path. 0.09 to 0.1 µs an element,
from 1.2 to 3.4, with the same results. The check now compares the fast
path's own results with MPFR, adds edge vectors, and adds two vectors
built so that one bound's last unit decides; it also gains the
one-larger control.

**2026-10-13: crfix's affine formats get a fast path; a slow-path fix.**
`crfix_eval_affine` and `crfix_table_affine` first try the fixed-point
functions' 64-bit enclosures. The input's magnitude goes in as Q64.64,
and each end is divided outward by the output scale. Only what the
enclosure leaves undecided takes the slow path. An int8 table takes 13 to
26 µs instead of 0.9 to 2.4 ms, with the same results. The check now
compares the fast path's own decisions with MPFR, on the random
configurations (all 245,760 int8 results decided) and on new edge ones.
The edge cases found a bug in 0.7.0's slow path: it saturated exp and
exp2 values from 2^62 up, whatever the output scale. With output scales
above 2^30 that is wrong: exp(46) at scale 2^50 is 84,342. Such a value
is now kept as 2^g·2^k. The MPFR reference had its own slip, also fixed:
it took the sum with the output zero point as exact when it rounded. The
fixed-point functions' results are unchanged (the digest), and so is the
Cortex-M0 build's lack of floating point.

**2026-10-13, after 0.7.0: crfix on a microcontroller.** `make
crfix-cross-check` builds crfix for a Cortex-M0. It finds no
floating-point routine, libm call or post-ARMv6-M instruction, each count
with its control. A digest of 23 million results matches the host's on
32-bit soft-float ARM Linux and on the M0 build running bare metal under
qemu. A CI job runs it.

**2026-10-13: crnumerics 0.7.0** (tag `v0.7.0`): crfix, with its slow path,
affine formats and softmax; crlin's banded and sparse systems; crsum's
faster short sums (the entries below). The candidate passed `make check`
(crfix's check and install check among its 28 verdicts) and the install
checks on cfarm421 and cfarm424, and CI; the tag's archive is the commit
file for file (289 files). **crlin 0.7.0 on PyPI**, wheels for x86-64 and
aarch64 Linux and Apple Silicon macOS, each tested in a clean venv before
the upload; installed from PyPI into a clean venv, it loads its own library
and passes the binding's 24 tests. crdistr and crival were not uploaded,
their code unchanged; crfix has no Python package.

**2026-10-13: crfix softmax.** `crfix_softmax`: the exact softmax of c·q,
rounded once into the output's affine format, correct by construction
through the slow path. Checked against MPFR on random vectors, exact ties
included. 1 ms for 1000 int8 logits.

**2026-10-13: crfix's slow path, and TFLite's affine formats.**
- **The slow path** (`fix/crfix-mp.c`): interval arithmetic in 320-bit
  fixed point, integers only, correct by construction. It decides the rare
  results the 64-bit enclosure straddles, and replaces the exceptions table
  and its generator, so 32-bit inputs are correctly rounded too. Alone,
  against MPFR, it matched 1,152,000 random cases of every function and
  mode.
- **Affine formats:** `crfix_eval_affine` and `crfix_table_affine` take
  value = scale (q − zero point), the scale any binary64, read from its
  bits. Correctly rounded int8 tables for TFLite's LOGISTIC, TANH, EXP and
  LOG take about 2 ms each. Checked against MPFR on random TFLite-like
  configurations.

**2026-10-13: crlin solves banded and sparse systems** (`crlin_solve_band`,
`crlin_solve_sparse`; Python `crlin.solve_banded` and `crlin.solve_sparse`).
- **How:** no inverse. A proven lower bound on the smallest eigenvalue of
  A (if it is SPD) or of A^T A, from `crlin_spd`'s argument on the band,
  bounds ‖x* − x̃‖₂ by the exact residual. Sparse matrices are reordered by
  reverse Cuthill–McKee.
- **Checked:** against exact rational solutions, both modes; singular
  systems never verify.
- **Speed:** a 200 × 200 grid's Laplacian verifies in 3.7 s at an ulp
  (classical mode). For a sparse solver that is slow: scipy's `spsolve`
  is about 10 times faster, unverified.

**2026-10-13: crsum rounds short sums three times faster.** `crsum_round`
(and `crdot`, `crsum`) settled, copied and scanned all 134 limbs of the
accumulator and took the result a bit at a time: 624 of a three-term dot
product's 782 ns. It now works only on the limbs that hold the value (the
carries stop two limbs above the highest), and reads the result from a
96-bit window. A short dot product takes about 245 ns. The results are
unchanged: crsum's check against MPFR passes, in the build that settles
every 3 terms too, and so do crlin's and crnn's.

**2026-10-13, after 0.6.0: crfix, correctly rounded fixed-point functions**
(`fix/`, ROADMAP item 3): exp, exp2, log, log2, sigmoid, tanh, sqrt, rsqrt,
reciprocal and division on fixed-point numbers, any fractional bits from 0
to 31 in and out, int8, int16 or int32 containers, four rounding modes,
saturated. Integer arithmetic only, native 128-bit or portable.
- **Proven** against MPFR on every 16-bit input, every fin, fout, mode
  and container, on cfarm421 (x86-64) and cfarm424 (aarch64), in both the
  native and the portable build: 805,306,368 results per function, and
  6.4 billion int8-pair divisions. The quick check in `make check` covers
  every int8 input and every 16th int16 input.
- **The reference had a bug:** its first all-inputs run reported 3
  random divisions wrong. The reference's "past every container" test was
  in unscaled units, and its conversion wrapped; the library was right.
- **The enclosures:** 9,875,714 of them, every one holding its exact
  value. 83 results they leave undecided, all of exp, are in a generated
  table.
- **sigmoid and tanh near 0** first rounded 188 random cases wrongly: their
  values sit v³/48 or v³/3 below a dyadic boundary, less than 2^−64 for
  tiny v. A series path fixed it.
- **The reference** first computed sigmoid as (1 + tanh(v/2))/2, which
  keeps no accuracy at e^−32768, and rounded from closed ends.
- 12 to 61 ns a call natively, 18 to 101 portably.

**2026-10-13: crnumerics 0.6.0** (tag `v0.6.0`): crlin, first released
(the entries below). The candidate passed `make check` and the install
checks of ival, crdist and crlin on cfarm421 and cfarm424, and CI; the
tag's archive is the commit file for file (275 files). **crlin 0.6.0 on
PyPI**, its first upload, with wheels for x86-64 and aarch64 Linux and
Apple Silicon macOS, each tested in a clean venv before the upload;
installed from PyPI into a clean venv, it loads its own library and
passes the binding's 22 tests. crdistr and crival were not uploaded, their
code unchanged.

**2026-10-13, after 0.5.0: crlin on a GPU.** `lin/examples/cublas.c`
(`make crlin-cublas`) gives crlin cuBLAS's DGEMM, in pedantic math mode so
no emulated binary64 slips in, and cuSOLVER's inverse.
- **Correct:** on cfarm107's NVIDIA GB10, small systems in all three modes
  held the exact solution, checked in fractions.
- **Not faster there:** sizes to n = 8000 verified at one ulp, but GB10's
  binary64 runs at about 150 GFLOP/s, and the same machine's CPU was 2.7
  to 7.3 times faster.

**2026-10-13, after 0.5.0: crlin encloses symmetric eigenvalues.**
`crlin_eig_sym` (and `crlin.eig_sym`) encloses every eigenvalue of a
symmetric matrix, from approximate eigenpairs (LAPACK's, or crlin's own
Jacobi method).
- **How:** G = Vᵀ V − I and R = A V − V diag(d) are enclosed, each entry
  one exact product; then Weyl puts Vᵀ A V's eigenvalues near d, and
  Ostrowski carries them to A.
- **Checked:** exactly, by Sylvester's inertia in rationals: none of 1,073
  outside.
- **Widths:** about n·u·‖A‖ absolute in the exact mode, and a thousand
  times that in the classical mode at n = 800.

**2026-10-13, after 0.5.0: crlin least squares, and a poor R improved.**
- **Least squares:** `crlin_lstsq` (and `crlin.lstsq`) encloses the exact
  least-squares solution for m ≥ n, or the minimum-norm one for m < n,
  through the augmented system of order m + n, exact in binary64 with its
  scaling α a power of two. Checked against the normal equations solved in
  rationals; rank-deficient A never verifies.
- **A poor R:** with nearly dependent columns (condition near 5×10^8), 7
  of 17 least-squares problems failed at every α. LU's inverse of the
  augmented matrix had left |I − R A|'s row sums near 10, though its
  condition number was about 10^9. So now, when the bound's row sum
  reaches 1/2, R is improved once to fl(inv(fl(R A))·R), and all 17
  verify. A system whose first R passes is unchanged, bit for bit; the
  Hilbert matrices gained a size in the exact mode (to n = 12).

**2026-10-13, after 0.5.0: crlin for very ill-conditioned systems.**
`CRLIN_EXTENDED` (and `mode="extended"`) carries R and x̃ as pairs of
doubles, R from Rump's iteration for extremely ill-conditioned matrices,
every product exact. Hilbert matrices verify to n = 24 (to 11 before), and
integer matrices of determinant 1 to condition numbers past 10^31; none
wrong in any mode. It costs 1.2 to 1.5 times the exact mode when one round
suffices, 6 to 8 times on ill-conditioned systems. The box iteration and
the final sums moved into one function the modes share, with point results
bit for bit as before.

**2026-10-13, after 0.5.0: crlin on interval data.**
`crlin_solve_interval` (and `crlin.solve_interval`) encloses the solutions
of every system inside interval bounds on A and b, and proves every A in
them nonsingular.
- **How:** the solve's proof on the midpoints. The residual's box is the
  exact range over the intervals, and the bound on |I − R A| gains
  |R|·rad(A). Point data's results are unchanged, bit for bit.
- **Checked:** against every vertex's exact solution for n ≤ 3, and samples
  beyond; the enclosures were at most 1.33 times the hull's width.
- **A slip in the test:** its first singular cases included 2-by-2
  matrices that were not singular, since the "last row = row 0 + row 1"
  construction makes row 1 its own sum; 5 of them verified, correctly.

**2026-10-13, after 0.5.0: crlin proves positive definiteness.**
`crlin_spd` (and `crlin.spd`) proves a symmetric A positive definite, with a
lower bound on its smallest eigenvalue: A = L Lᵀ + c I + E for any L and c,
so λ_min(A) ≥ c − ‖E‖₂, with E enclosed as in the solve. Its own factor
aims c at half the smallest eigenvalue; a factor from LAPACK is faster.
- **Checked exactly:** A − λ_lo I is positive definite by elimination in
  rationals for every bound; none of 78 matrices that are not positive
  definite verified.
- **Badly scaled matrices** (rows and columns over 60 binades) failed, 34
  of 40, since their smallest eigenvalue lies below n·u·max A_ii; a second
  try on D A D, scaled exactly by powers of two, proves them all.
- **A miscount caught:** the README first said Hilbert matrices verify to
  n = 12. They verify to 11; 12 and 13 fail, and 14, rounded to binary64,
  is not positive definite at all.

**2026-10-13, after 0.5.0: crlin installed, and threaded.**
- **Installed** by `make install` (Linux): `libcrlin.so.0`, `libcrlin.a`,
  `crlin.h` and `crlin.pc`. The library exports `crlin_` names only, with
  its copy of crsum made local to it. `make check` builds against an
  installed copy, and pointed at another library its check says DIFFERS.
  A wheel (`lin/python/build-wheel.sh`, `crlin-wheels.yml`) waits on a PyPI
  publisher.
- **Threads:** the work outside the GEMM (the exact residuals, the box, the
  bound's entries, the inverse) is split by rows among OpenMP threads, with
  the same bits for any number. On a 64-core EPYC at 32 threads, the
  classical mode's own part took 342 ms at n = 3200 against 734 on one
  thread with the same BLAS.
- **A pool's mode:** each thread sets the default floating-point mode
  itself, since a pool's threads keep the mode they were created with. The
  test of that first used well-conditioned systems only, and passed a build
  that never set it. The final enclosure there is the same ulp around x*
  whatever R was, so nearly singular systems are now the witnesses.

**2026-10-13, after 0.5.0: crlin, several right-hand sides.**
`crlin_solve_many` (and `crlin.solve` with b an n-by-k matrix): R and the
bound on |I − R A| shared, and in the classical mode the columns batched
as matrix products, the residual kept exact. At n = 400, 50 columns went
from 580 ms to 105 (47 times numpy's solve); a single column, now through
exact matrix-vector residuals, to 6 to 12 times. Two fixes found on the way:
- **The box iteration** stopped at the first failing row and left the
  others stale. Computing every row verifies more (Hilbert to n = 11, not
  8) and is never wrong.
- **One overflowing column** failed every column in the batched path. The C
  test of it had accepted infinite enclosures as "kept"; it now asks for
  finite, tight ones, and the Python test found it first.

The factorization in place of the inverse was measured and set aside: the
method needs R itself.

**2026-10-13, after 0.5.0: crlin, verified linear systems** (`lin/`,
ROADMAP item 11). `crlin_solve` encloses the exact solution of A x = b
(Rump's method), or refuses; it never switches the rounding mode, so a
threaded BLAS can't break it. Two modes for I − R A: the exact product
(crgemm_oz) or two binary64 products with an a priori bound whose absolute
term assumes flush-to-zero may be on in a BLAS's worker threads (its
first version assumed gradual underflow; a subnormal entry now sends the
solve to the exact product, since denormals-are-zero would break the
bound). Checked against exact rational solutions (fraction-free
Gauss–Jordan in GMP, itself checked against A x = b): 516 systems, 0
misses, the widest 3.9e-16; shrunk by an ulp, 506 miss (the control); 100
exactly singular systems, none verified; Hilbert matrices never wrong;
spoiled inverses never wrong. On one thread the classical mode takes 8 to
17 times numpy's unverified `solve`, Arb's `arb_mat_solve` 113 to 470
times; crlin's enclosures are an ulp wide, Arb's 25 times wider. Python:
`crlin.solve` (`lin/python`).

**2026-10-13: crnumerics 0.5.0** (tag `v0.5.0`): Poisson and binomial,
macOS, `crdist-check hash-all` (below). The candidate passed `make check`
and both install checks on cfarm421 and cfarm424 and CI; the tag's archive
is the commit file for file. **crdistr 0.5.0 on PyPI** with wheels for
x86-64 and aarch64 Linux and Apple Silicon macOS, each tested in a clean
venv before the upload; installed from PyPI into a clean venv, it passes
the binding's tests. Its summary now names every function.

**2026-10-13, after 0.4.1: crdist finished for now: Poisson, binomial,
macOS, and the proof carried to other builds.**
- **Poisson and binomial** (`int64_t`): inversion for small means, then
  Hormann's PTRS (Poisson, as NumPy) and BTRS (binomial, as TensorFlow and
  JAX), specified in `crdist.h` with log k! as a table plus Stirling's
  series (CORE-MATH's lgamma writes signgam, so not it). `crdist-check`: 27
  cases (lam to 1e15, n to 2^53, parameters out of range), 221,184 draws
  against the specification computed with MPFR, 0 differ; chi-squared
  tests against the exact probabilities and mean tests, their controls
  failing; fingerprints. In Python, `Stream.poisson` and `Stream.binomial`.
- **macOS:** crdist builds with Apple clang (a dylib, no objcopy, no
  OpenMP); CI's macos job runs its check and the binding, and
  `crdistr-wheels.yml` builds an Apple Silicon wheel. The check and the
  generators now link only the four CORE-MATH files crdist calls:
  `libcm.a`'s binary16 files need `exp10f`, which macOS's libm lacks.
- **`crdist-check hash-all`:** every output of every function on all 2^32
  inputs, hashed, against the digests of the builds `crdist-check all`
  proved (42432d0 and 85ef764, the same digests from both). Matched on
  board5 (gcc 13), cfarm424 (aarch64, 21 s on 32 threads) and cfarm104
  (macOS on an M1, Apple clang 21, 6 min on one thread): the proof of every
  input holds there too. Its first version kept 8 MB of thread-local
  buffers, which come out of every OpenMP thread's stack, and crashed on
  aarch64; the buffers are allocated now.
- **The wheel workflow's release check** named one of the four CORE-MATH
  files crdist carries (`log`); it names all four now.

**2026-10-13: crnumerics 0.4.1** (tag `v0.4.1`): crdistr (below). The C
code is 0.4.0's but for the version strings. **crdistr 0.4.1 is on PyPI**
(https://pypi.org/project/crdistr/), uploaded by `crdistr-wheels.yml`
through trusted publishing: manylinux_2_28 wheels for x86-64 and aarch64,
each tested in a clean venv first, and the source package. Installed from
PyPI into a clean venv, it passes the binding's tests. Its one-line summary
there still names only erfinv, erfcinv and probit; `pyproject.toml` is
corrected for the next upload.

**2026-10-13, after 0.4.0: crdistr.** The Python package is crdistr on
PyPI and as a module (`import crdistr`; it was `import crdist`), and its
workflow is `crdistr-wheels.yml`, the name PyPI's trusted publisher was set
up for. 0.4.0's upload had been refused (`invalid-publisher`) because the
workflow was then `crdist-wheels.yml`. A module named crdist could also
have met the unrelated "crdist" project on PyPI. The C library is still
crdist. `build-wheel.sh` now removes the egg-info whatever the
distribution's name; it had left one behind under the earlier name.

**2026-10-13: crnumerics 0.4.0** (tag `v0.4.0`). crdist's 16-bit formats,
`make install` and Python binding, `ndtr`, `log_ndtr`, `normpdf` and `logit`,
and the samplers (the entries below). ival, crsum, crnn and lowp unchanged.
The candidate passed `make check` and both install checks on cfarm421 and
cfarm424 and CI; the tag's archive is the commit file for file and passed
the same on cfarm424. The Python package's wheels (`crdistr`) were built
and tested; its first upload to PyPI waits on the trusted publisher's
settings.

**2026-10-13, after 0.3.0: the normal distribution's functions of x, and
samplers.**
- **`ndtr`, `log_ndtr`, `normpdf`, `logit`:** correctly rounded in binary32,
  binary16 and bfloat16, from CORE-MATH's `erfc`, `exp`, `log` and `log1p`
  within 2^-50. Below -30 `log_ndtr` is -x^2/2 plus the asymptotic series,
  kept as a pair with a rounding test of its own: -x^2/2 is often exactly
  a binary32 midpoint (x^2 has 48 bits), and with one binary64 value the
  table needed 32,069 entries for it; with the pair, 23. (Its first version
  refused every result past binary32's range, 533 million inputs: a sign
  taken with copysign where a product was meant. The generator's count
  showed it, and the run was stopped.) Every one of the 2^32 inputs of each
  checked against MPFR on cfarm421 (x86-64, 13 min on 64 threads) and
  cfarm424 (aarch64, 27 min on 32): 0 differ; the tables generated there
  byte-identical to the committed ones. The binary64 bound on every input
  (bound-all, cfarm421): within 2^-50 everywhere, none beyond 2^-51.
- **Samplers:** uniform, normal, exponential, gamma (Marsaglia and Tsang),
  beta, chi-squared and Student t from a Philox4x32-10 stream (seed,
  substream, index), their bits fixed by a specification in `crdist.h`:
  Philox words, crdist's probit, CORE-MATH's `log` and `exp` in binary64,
  IEEE arithmetic. `crdist-check` reproduces Random123's known answers,
  recomputes the specification with MPFR (69,632 draws, 0 differ), tests
  the distributions (KS against MPFR's exact distribution functions, with
  failing controls), and records fingerprints of fixed streams; CI's arm64
  job gives the same. In Python as `crdist.Stream`.

**2026-10-13, after 0.3.0: crdist in binary16 and bfloat16, installed, and
from Python.**
- **binary16 and bfloat16:** `crdist_erfinv_f16` and `_bf16`, likewise for
  `erfcinv` and `probit`, on bit patterns, correctly rounded in the format
  from the same binary64 values with a rounding test per format; no input of
  either format is undecided, so the tables are empty. `crdist-check` tries
  all 2^16 inputs of each against MPFR, also under round-upward with
  flush-to-zero: 0 differ. Rounding the binary32 result again would have
  been wrong on 5 inputs.
- **Installed:** `make install` puts `libcrdist.so.0` (the soname now
  versioned), `libcrdist.a`, `crdist.h` and `crdist.pc` under `PREFIX`
  (Linux); `make check` builds against an installed copy through
  pkg-config, shared, static and C++; `crdist_version()`.
- **From Python:** `dist/python`, the module `crdist` over NumPy arrays,
  the format following the dtype; `make crdist-python-check` against mpmath,
  in CI on x86-64 and arm64. A planted error of 2^-30 in probit's sqrt 2
  failed its float32 test. Its PyPI name is not chosen ("crdist" is taken).

**2026-10-13: crnumerics 0.3.0** (tag `v0.3.0`). crdist: binary32 `erfinv`,
`erfcinv` and `probit`, correctly rounded and checked on every input (below).
ival, crsum, crnn and lowp unchanged; ival's soname is still `libival.so.0`,
and no crival upload, as its results did not change. The candidate passed
`make check` and ival's install check on cfarm421 (x86-64) and cfarm424
(aarch64) and CI; the tag's archive is the commit file for file.

**2026-10-12 to 13, after 0.2.2: crdist, the inverse normal distribution,
correctly rounded** (`dist/`). `crdist_erfinvf`,
`crdist_erfcinvf` and `crdist_probitf` (the standard normal quantile, R's
`qnorm`, SciPy's `ndtri`) in binary32, correctly rounded to nearest on
every input; CORE-MATH and RLIBM have no inverse of erf, and the best vendor
accuracy found is Intel IPP's, within an ulp. JAX's `random.normal` is
`sqrt(2) * erf_inv(uniform)` (jax e0f38af4d2), the first user to look at.
- **Computed** in binary64: up to 1/2, erfinv(x) = x P(x^2) (degree 13); in the
  tail, the inverse of erfc as six degree-14 pieces in s = sqrt(-log q), q
  down to 2^-149. The polynomials come from `dist/fit.py` (mpmath, Chebyshev
  nodes), and CORE-MATH's `log` is the only function called. Within 2^-50;
  rounded to binary32 when that can't round two ways, else a table made
  from all 2^32 inputs with MPFR: 20, 52 and 46 entries, byte-identical when
  generated on x86-64 with gcc 13 and 14 and on aarch64 with gcc 14. On 3 of
  them, plainly rounding the binary64 value would be wrong. 36 to 54 ns an
  element (Ryzen 5 PRO 5650U).
- **The references:** MPFR has no inverse of erf, so `crdist-ref.c` compares
  the root with exact midpoints, by erf, erfc or erfc(m/sqrt 2) under
  directed roundings from 64 bits up; a bisection with no start agrees.
- **Proven on every input,** `crdist-check all` on cfarm421 (x86-64) and
  cfarm424 (aarch64), gcc 14.2.0: 3 x 2^32 inputs, 0 differ, and the kit's
  control 67,111,000 differ for each function (2 h 27 min on 64 threads; 6 h
  6 min on 32). The sampled check runs in `make check`. The binary64 bound,
  `crdist-check bound-all` on cfarm421: within 2^-50 on every input inside
  each domain; only 51 inputs, all of `probit`, lie beyond 2^-51.
- **The first version,** the same evening, started from Winitzki's
  approximation and took three to five Newton steps through CORE-MATH's erf
  or erfc and exp: within 2^-51 and a 51-entry table, identical on both
  hosts, but 290 to 880 ns an element. Its generator measured the start at
  2.26e-3 of the result in the tail, not the 2e-3 its comment had assumed.
  The polynomials replaced it before its exhaustive runs finished.

**2026-10-12: crnumerics 0.2.2** (tag `v0.2.2`). CORE-MATH ea7a421 inside ival
and the kit, with the same results, and the bfloat16 references right under
GCC 13 on AArch64.
- **CORE-MATH ea7a421:** ten of the carried functions updated (below); old
  against new, the same results and flags on 686,469,600 calls on x86-64 and
  aarch64, every worst case included.
- **GCC 13 on AArch64:** the vendored bfloat16 functions, the self-test's
  references, built right with the fp16 extension on (below).
- **ival from Julia:** `make ival-julia-ccall` and `make ival-julia-threads`
  (below).

**2026-10-12, in 0.2.2: CORE-MATH ea7a421.** CORE-MATH
ea7a421's cosh, cospi, exp, exp10, exp2, log10, log2, sinh, sinpi and tanpi
(upstream's source-coverage work since 040ee48, then 10-12's sinh type fix,
log2 patch and tanpi change). Old (040ee48) against new in all four rounding
modes: the same results and flags on 686,469,600 calls on x86-64 and on
aarch64, among them every input of the ten functions' worst-case files and its
negative (1,922,435). 81a899d's sinh, which narrowed a variable to `int`, gave
the same results as ea7a421's on the same calls: it cost time, not results.
All 131 carried files equal ea7a421.

**2026-10-11, after 0.2.1: the bfloat16 references under GCC 13 on AArch64.**
Built by GCC 13 with AArch64's fp16 extension (`-march=native` on most
current cores, so Ubuntu 24.04's default compiler on such a machine), 17 of
the vendored CORE-MATH bfloat16 functions were wrong and the kit's self-test
said so (cos(0) = 0.0078125; cospi on 124,424 of its 262,144 inputs and
modes). GCC loads a `__bf16` constant with FMOV's half-precision immediate,
which encodes binary16, so 1.0 arrives as 0x3c00: GCC PR111867, fixed in
GCC 14 and never backported. The Makefile now compiles `core-math/bf16/`
with `core-math/gcc13-bf16-fix.h`, which turns fp16 off for those files
under GCC before 14, and GCC 13 then loads the constant from memory. On
cfarm424 (Debian gcc 13.3.0, `-march=armv8.2-a+fp16`) the self-test went
from DIFFERS to IDENTICAL; gcc 13 without fp16 and gcc 14 with it were
IDENTICAL before and after. The arm64 CI job now builds the self-test with
fp16 too. Outside `core-math/bf16/` only the self-test uses `__bf16`, and its
own bfloat16 checks passed in the faulty build.

**2026-10-11, after 0.2.1:** `make ival-julia-ccall` (`ival/julia/ccall-check.jl`).
ival called from Julia through `ccall`: on 4,096 intervals each of exp, exp2,
exp10, tanh, cbrt, asinh, acosh and atanh, every bound is the same as
IntervalArithmetic.jl's `:correct` (1.0.12, Julia 1.11.7), and Julia is in
round-to-nearest before and after every call (libc's `fegetround`, and an
addition that only nearest rounds to 1.0). ival changes the rounding mode
only inside its own C code, so nothing LLVM compiles for Julia runs in
another mode. The script first checks that its checks can fail.
`make ival-julia-threads` (`ival/julia/threads-check.jl`) does the same from
several Julia threads at once: with 3 and with 7 worker threads, 25,600
calls of 4,096 intervals gave every bound of `:correct`, while a sentinel
on another thread checked its own rounding mode 86 to 152 million times
and always found round-to-nearest. The mode is per thread (MXCSR, FPCR).

**2026-10-09 to 10: crnumerics 0.2.1** (tag `v0.2.1`). ival on macOS (Apple
Silicon).
- **macOS:** `make` builds `libival.a` and `libival.dylib` with Apple
  clang; a macos-14 CI job runs every ival check, ITF1788 (7,451 tests
  tight) and the Python binding. On macOS the build needs no objcopy (Apple's
  linker hides the CORE-MATH symbols itself), runs one check without
  OpenMP, and gives the thread check its own barrier.
- **`make install` on macOS:** `libival.<version>.dylib` with its links,
  its install name LIBDIR's, and `ival.pc`; the install check asks
  `otool` and `nm` there. Python wheels for macOS 11 and later.
- **Two faults the port found, both giving wrong intervals on macOS only:**
  - sqrRev's square roots rounded down and up came out to nearest, so a
    double just beyond √c was kept: Apple clang 15, which keeps no
    floating-point order on AArch64, moved the inlined roots past the
    rounding-mode changes. Each root is now stored through a volatile
    before the next change.
  - text was read to nearest, not outward: macOS's `strtod` ignores the
    rounding mode, which the parser relied on. Numbers are now rounded by
    deciding exactly, with big integers, on which side of the exact value
    each neighbouring double lies; no C library's rounding is relied on.

**2026-10-09, after 0.1.0: crnumerics 0.2.0** (tag `v0.2.0`). ival one
interval at a time; a fix to the periodic reverses; Python.
- **Fixed: sinRev, cosRev and tanRev could miss a set,** returning the
  empty set or too little, when a crossing of C's end lay within a few
  ulps of X's end. 0.1.0 gave the empty set for cosRev of
  [sin 0.5, sin 1.5] in [0.5 + π/2, π − 1.6 + π/2], where the set is
  about [2.0708, 3.0708]. Two faults: whether f at X's end lies above C
  was read from f rounded down (all three now read it rounded up; it
  broke sinRev and cosRev), and the crossing's estimate could fall a
  period short (it broke all three). Found by IBEX's tests; rev-check
  now puts X's ends on and next to the crossings, and planted bugs 43 and
  44 put each fault back.
- **A call on one interval** costs 20 to 60 ns, not 165 to 186: on x86-64
  ival saves and sets MXCSR alone (CORE-MATH's objects get their fenv
  calls renamed to MXCSR versions, since glibc's read the x87 unit); on
  aarch64 FPCR and FPSR directly; small arrays take the scalar code.
  `env-check` shows every call leaving the caller's rounding mode, flags,
  MXCSR and x87 words as they were.
- **`ival-scalar.h`,** installed: the arithmetic on one interval, inline
  and bit for bit the library's. To nearest by error-free
  transformations; rounding upward, as Gaol-style interval classes keep
  it, one operation a bound (4.5 ns an add, 7.7 a product); in any other
  mode, the library. `ival1_scale` multiplies a point by an interval in
  two products. The mode check sits behind a volatile barrier, and
  `mode-change-check`, built without `-frounding-math` as most callers
  are, changes the mode between calls (planted bug 47 fails it).
- **Integer powers** without pow when a double-double product decides
  them (2 ≤ p ≤ 64, error at most p·2^−100): pown 3 to 6 times faster,
  pownRev 2.
- **Against IntervalArithmetic.jl** (`make ival-julia-compare`): the same
  bounds bit for bit as its `:correct` rounding; exp2 and tanh 50 to 100
  times faster.
- **Python** (`ival/python`): a binding over NumPy arrays (ctypes). Numbers
  that are not doubles round outward. 27 tests, in `make
  ival-python-check` and CI. Against pyinterval it gives the same tight
  bounds on 11 functions, and is 6 times faster one interval at a time.
  `build-wheel.sh` makes a wheel that carries `libival.so.0`. On PyPI as
  `crival` (`pip install crival`, then `import ival`), wheels for Linux
  x86-64 and aarch64.
- **IBEX** (outside this repository): an `INTERVAL_LIB=ival` backend
  passes 61 of IBEX's 62 tests. The one failure asserts bit-identical
  Jacobians along two evaluation orders, and ival is one ulp tighter on
  one. On IBEX's 264 solver benchmarks it takes 28% less time than filib
  on aarch64 and 23% more than Gaol on x86-64 (ROADMAP item 4).

**2026-10-08 to 09.** ival to a full IEEE 1788.1 library;
**crnumerics 0.1.0** (tag `v0.1.0`, the first).
- **Arithmetic:** add, sub, mul, div, neg, sqr, recip, sqrt and fma, the
  tightest. Error-free transformations are the reference, and arrays run
  with the rounding mode set per block; four lanes at a time with AVX2
  and FMA, bit for bit. The caller's flush modes are off inside.
- **The rest of 1788.1:**
  - the roundings, abs, sign, min, max, the set operations, cancel,
    the numbers (mid, rad, ...), the predicates and overlap;
  - pown for every real x, rootn;
  - mulRev and mulRevToPair, and the reverse operations of sqr, abs,
    cosh, pown, sin, cos and tan, with powRev1 and powRev2;
  - numsToInterval and textToInterval, with every bound exact;
  - the reductions, through crsum.
- **The functions** switch the rounding mode per block, not per interval
  (exp 6 times faster, sin and cos 3 to 4). The accurate mode
  (`ival_acc_f`, one ulp at most) runs through crmvec's vector functions
  for all 32 functions and atan2, hypot and pow.
- **Checks:**
  - ITF1788, IEEE 1788's test suite, through a converter: 7,451 tests of
    ival's operations, every result tight. Expectations that are not the
    tightest are corrected, each with a proof.
  - The periodic reverses against an MPFR reference.
  - Thread safety, checked.
  - A CI job under ASan, UBSan and TSan.
  - Comparisons with MPFI, Boost.Interval, filib++ and libieeep1788.
- **CORE-MATH 040ee48:** sin.c, exp.c, log/log.c, f16/cbrtf16.c and
  bf16/acos_bf16.c.
- **`make install`:** `libival.so.0`, `libival.a`, `ival.h`, `ival.pc`,
  checked by building against the installed copy. The version is
  `IVAL_VERSION` in `ival.h`.

**2026-10-06.** CORE-MATH e78b460: cospi.c, sinpi.c, sin.c, pow/pow.c,
pow/pow.h and f16/cbrtf16.c. Upstream fixed the cbrtf16 finding (398b235).

**2026-10-05.** README: an overview, with each library's section moved
unchanged to its directory's `README.md`.

**2026-10-02.** ival's `tgamma`.
- **ival:** `ival_tgamma` over any interval, the tightest binary64
  enclosure:
  - the poles counted exactly; the extrema from a table of 185
    (`ival/gen-tgamma.py`), each between its binary64 neighbours and
    rounded both ways;
  - past −184, where every binary64 value has |Γ| < 2^−1074, the extremum
    rounds to ±0.
- **ival's check:** a reference for tgamma that finds the extrema by
  bisection on digamma's sign in MPFR, with signs from lgamma (Γ underflows
  MPFR's own exponent range past n = 180).
  - It caught one bug: [k − 1, k] beyond 2^50, both ends poles, had given
    [+∞, +∞]. That is now [+0, +∞].
  - A planted error, an extremum's value an ulp low, is caught. An
    extremum misplaced by one neighbour cannot show: Γ is flat there.
  - `IVAL_ONLY=f` runs one function alone.
- **crsum:** `OZ8_KC` settable at compile time, and `bench-ozk` for m × n ×
  k shapes. Blocking along k measured, no clear win (ROADMAP item 1).

**2026-10-01.** The Ozaki scheme on int8 units.
- **crsum:** `crgemm_oz8`, the same exact scheme as `crgemm_oz` with 7-bit
  slices on integer dot-product instructions instead of binary64 GEMMs:
  every slice product an exact int32 (k in chunks of 32,768), the sums
  added in int64, and `crgemm_oz`'s rounding step, now shared. Checked
  against `crgemm` bit for bit through the internal kernel (AVX512-VNNI on
  cfarm151, Arm SDOT on cfarm424, plain C on the laptop), plain C, a
  scrambled order and a counting GEMM; a GEMM dropping one term differs.
  With VNNI it takes half `crgemm_oz`'s time at n = 256.
- **crnn:** the five one-argument functions in binary16 and bfloat16,
  correctly rounded on all 2^16 inputs of each (checked against MPFR,
  and under round-upward with flush-to-zero). binary16 needs no exception
  table; bfloat16 needs 256 entries each for SiLU and GELU, and the check
  catches their removal.
- **crnn on an NVIDIA GPU:** its five functions (all 2^32 inputs each) and
  its composites (crsum and crnn compiled as CUDA device code, 2000
  vectors) give the CPU's bits on a GB10.
- **crgemm_oz8's kernels:** AVX2 for x86 without VNNI (bytes widened to
  16 bits, `vpmaddwd`), 19 times plain C; the x86 kernels blocked;
  `CRSUM_I8_KERNEL` to pick one. Two planted kernel bugs (a wrong column
  in an edge block, the k tail dropped) were caught by the check. Then an
  Arm I8MM kernel (`smmla`), 1.4 to 1.8 times SDOT on the GB10's cores.
- **The x86 int8 kernels over panels of B's rows** (about 16 KB, L1):
  `crgemm_oz8` at 1024³ 28% faster with VNNI on cfarm151 and 11% with AVX2
  on the laptop (19% and none at 512³). The check got cases with several
  panels; a planted panel bug is caught by them and by nothing before.
- **The SDOT kernel blocked** like the others (two rows by four, over the
  same panels): `crgemm_oz8` 1.4 to 1.9 times faster on a Neoverse N1.
- **The I8MM kernel reworked** (16-byte loads, four accumulators, the
  panels): `crgemm_oz8` 1.16 to 1.45 times faster on the GB10.
- **crnn through crmvec:** an optional vector path (`CRNN_CRMVEC`) for
  sigmoid, SiLU, GELU and softplus, 2.5 to 3.5 times faster, the same bits
  on all 2^32 inputs of each (hashed both ways), because the proof needs
  only correct rounding.
- **crsum_add 3 times faster** on short blocks: thread-local bins and a
  touched-groups mask instead of allocating, clearing and scanning 4096
  bins per call; crnn's composites gain with it.
- **crblas:** `dgemm_` and `dgemm_64_` through `crgemm_oz`, for programs
  that already call a BLAS. Julia's products, through its
  libblastrampoline with Julia's own OpenBLAS inside, become the exact
  products rounded once (`make julia-check`); libblastrampoline's probes
  needed `isamax`, `zdotc`, `cdotc` and `sdot` too.

**2026-09-30.** The first day, in one burst.
- **Beginnings:** the checking kit and repro-scan, first as a folder of
  [crmvec](https://github.com/anun333/crmvec), then this repository, with
  their history.
- **lowp:** correctly rounded FP8 math (E4M3, E5M2), proven on every input
  and pair, and MX block functions, correct by construction. The rules
  were then checked against OCP's OFP8 1.0 and MX v1.0 specifications:
  E4M3's saturating infinity and the MX scale, saturation and ties are the
  specifications'; the all-zero block, the scale clamp and the whole-block
  NaN rule are choices they leave open. E5M2's saturating mode, which OFP8
  requires, was added. MXINT8 elements followed, and their first check
  found a scale bug that every earlier element type had hidden, in lowp
  and in the kit's reference alike (a nonzero result too small to
  represent taken for a zero): fixed in both.
- **ival:** 31 interval functions, then `hypot`, `atan2` (across the branch
  cut) and `pow` on boxes.
- **crsum:** correctly rounded sums and dot products, the same bits in any
  order; binned for speed (1.2 times a naive binary64 sum); matrix
  products; and `crgemm_oz`, exact matrix products through any binary64
  BLAS (the Ozaki scheme), checked through netlib's BLAS and through
  threaded OpenBLAS on x86-64 and arm64.
- **crnn** (the same evening): neural-network primitives with one answer.
  sigmoid, SiLU, GELU, softplus and rsqrt are correctly rounded in
  binary32, through a binary64 fast path and a table of the 531 inputs it
  can't decide. logsumexp, softmax, layernorm and rmsnorm are specified bit
  for bit on crsum. Its own checks corrected it three times before it was
  committed:
  - SiLU and GELU at tiny inputs sit exactly on a binary32 midpoint in
    binary64. The first table had 16.7 million entries each; breaking the
    tie analytically left 24 and 15.
  - logsumexp took log of the rounded sum, 9.5 million ulp off when the sum
    is 1 plus terms far below it. It now takes log1p of the exact sum less
    1.
  - layernorm subtracted a rounded mean, 30 ulp off for a large mean and a
    small spread. The mean is now carried in two parts.
- **Found along the way, for other projects:** crmvec's binary16 `cbrt`
  called the C library's `cbrtf` (fixed in crmvec); glibc's `libm` gives
  other bits on CPUs without FMA; Microsoft's MX reference implementation
  picks a scale one step too large just below a power of two.

Every check here has a control that must fail, and was trusted only after
deliberately planted bugs showed it could see them. The planted bugs that
passed are part of the record: some changed no result and proved nothing,
and some showed a gap in the test cases, which was then closed. README.md
has the details, library by library.
