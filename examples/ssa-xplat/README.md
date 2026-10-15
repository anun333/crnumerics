# Stochastic biochemistry across platforms

Do the stochastic simulators of biochemistry give the same trajectories,
from the same model and the same seed, on every platform?
`ssa_xplat.py OUT --save` runs four classic models, eight trajectories
each, sampled at fixed times:
- Gillespie's dimerization and decay;
- Lotka-Volterra predator and prey, which can go extinct;
- Schlögl's bistable model, where one different event can switch the
  branch;
- constitutive gene expression, with bursts.

The simulators are the ones that are installed: libRoadRunner's Gillespie
integrator, COPASI's direct method (through basico), and GillesPy2's NumPy
and C++ solvers. Beside them runs a reference built on crdist: the
Gillespie direct method with waiting times from `crdist_exponential` (the
correctly rounded −log of a 52-bit uniform from Philox4x32-10), the
reaction chosen with `crdist_uniform`, the total propensity summed
exactly (`math.fsum`), and every other step one IEEE operation. It calls
no C math library, so it should give the same bits everywhere.

The report has the same format as [../numpy-xplat](../numpy-xplat), so its
`compare.py` groups the platforms test by test. `split.py BASE OTHER...`
shows where each trajectory first splits from BASE's, as a share of the
run. `.github/workflows/ssa-xplat.yml` runs it on GitHub's Windows and
Ubuntu runners.

Each run is reproducible on its own machine. Whether it is reproducible
across machines is what this measures.

## Results, 2026-10-15

Eleven reports: Linux on glibc 2.38, 2.39 and 2.41 (x86-64 with and
without AVX-512, aarch64), Alpine (musl 1.2.6), macOS on Apple M1, and
Windows on x86-64 and Arm64 (GitHub's runners). Each simulator ran where it
has a wheel. Each platform ran the script twice; where the two reports
differed, the cause was COPASI (below).

**Stochastic simulation: the same everywhere.** Every simulator gave the
same sampled trajectories on every platform it ran on: libRoadRunner,
COPASI, GillesPy2's NumPy and C++ solvers, and the crdist reference. So
did libRoadRunner's first 2,000 event times at full precision. In the
direct method the sequence of reactions depends only on the counts and the
selection draw. A C math library that rounds a `log` differently moves an
event time by an ulp, not the reactions, and samples at fixed times
almost never see it. The crdist reference is the same by construction:
its log is correctly rounded and it calls no C math library.

**The same models as ODEs: different on x86 and Arm.** libRoadRunner
(CVODE), COPASI (LSODA) and GillesPy2's SciPy solver each give one result
on x86-64 and another on Arm. Most differences are 10^-14 to 10^-11 of a
species' range. The two LSODA solvers differ by 1.9 × 10^-6 on the gene
expression model, the size of the solver's tolerance rather than of a
rounding, which suggests a step that was taken one way on one machine and
the other way on the other. These models use no exp or log, so the
compiled arithmetic decides, not the math library. It is the build, not
the CPU: on Windows for Arm, COPASI agrees with x86 while SciPy's solver
agrees with macOS.

**COPASI's ODE results do not repeat.** COPASI 4.48.309 (the
python-copasi wheel) gave different deterministic results for the same
model in two runs of the same script on macOS, Windows x86-64 and Windows
Arm64, up to 3 × 10^-10 of a species' range. On Linux it repeats, unless an
earlier model was removed in the same process: then 8 cycles of loading,
running and removing the same model gave 5 different results. Valgrind
reports 162 uses of uninitialised values, all inside the link-matrix QR
factorization (`CLinkMatrix::build` calling the LAPACK bundled in the
module, `dgeqp3`), every one of them in LAPACK's block-size query
(`ilaenv`, half in the string comparison it calls). If that query answers
differently, the factorization takes another path and its last bits
change. That is the likely cause; it is not proven.

**Problems running the simulators:**
- GillesPy2 1.8.3's C++ solver does not compile with Alpine's GCC or with
  MinGW on Windows: `model.h` uses `uint8_t` without including
  `<cstdint>`.
- libRoadRunner 2.10.0's Linux wheel needs `libpython3.12.so.1.0` and does
  not say where to find it, so it fails to load on a Python whose shared
  library is not on the default search path (uv's standalone Python), until
  `LD_LIBRARY_PATH` names it. On Arm Linux the newest wheel is 2.7.0,
  which requires NumPy 1.26 (`numpy ~=1.26`).
