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
