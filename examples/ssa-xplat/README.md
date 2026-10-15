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
running and removing the same model gave 5 different results. The
differences are far inside the solver's tolerance, which is all an ODE
solver promises: COPASI closed our report (bug 3356) as invalid because
"the numerical differences are far beyond the requested tolerances", and
under that contract they are right.

What decides the last bits is where COPASI's objects sit in memory. With
the memory layout held fixed (address randomisation off, a fixed clock,
and fixed seeds for Python's string hashing and for the hash salt of the
XML parser bundled in the module), the 8 cycles repeat exactly in every
process; adding only seeded random allocations between cycles gives a
different sequence for each seed, again repeatable. Across all these runs
there were eight distinct time series, differing in the last bits, as
when the same terms are added in an order set by object addresses (not
traced to the line). On macOS and Windows the allocator varies the layout
from run to run, which would explain why fresh runs differ there.

*Correction (2026-10-15):* this section first named uninitialised values
in LAPACK's block-size query (`ilaenv`, which valgrind reports) as the
likely cause. They cannot be: in the bundled LAPACK, `ilaenv` compares the
routine name with a declared length of 1 against two- and three-letter
names, so no comparison can succeed, and it returns its default whatever
the uninitialised bytes hold.

**Problems running the simulators:**
- GillesPy2 1.8.3's C++ solver does not compile with Alpine's GCC or with
  MinGW on Windows: `model.h` uses `uint8_t` without including
  `<cstdint>`.
- libRoadRunner 2.10.0's Linux wheel needs `libpython3.12.so.1.0` and does
  not say where to find it, so it fails to load on a Python whose shared
  library is not on the default search path (uv's standalone Python), until
  `LD_LIBRARY_PATH` names it. On Arm Linux the newest wheel is 2.7.0,
  which requires NumPy 1.26 (`numpy ~=1.26`).
