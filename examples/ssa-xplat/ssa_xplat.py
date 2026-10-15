#!/usr/bin/env python3
"""ssa_xplat.py OUT [--save] [--only SIM,...]: do the stochastic simulators of biochemistry give the same trajectories,
from the same model and seed, on every platform? (2026-10-15.)

Four classic models (below), each run as several trajectories, sampled at fixed times, by each simulator that is
installed: libRoadRunner's Gillespie integrator, COPASI's direct method (through basico), GillesPy2's NumPy and C++
SSA solvers, and a reference built on crdist (crdistr): the Gillespie direct method from Philox draws, a correctly
rounded log, Python's float operations and math.fsum, so no C math library and no contraction anywhere. Writes
OUT/report.json: the platform, the versions, and per (simulator, model) a hash of the int64 array (trajectory,
sample, species). With --save, also OUT/arrays/NAME.npy, for element-by-element comparison (where does each
trajectory first split from another platform's?). Same report format as ../numpy-xplat, so its compare.py and
diff.py read it. A simulator that is not installed is left out of the report, not counted as a difference.
"""
import hashlib
import json
import math
import os
import platform
import sys

import numpy as np

SEED, NTRAJ = 20261015, 8

# model: species with initial counts, fixed (boundary) species, reactions (rate constant, reactants, products),
# end time and number of samples. Propensity (mass action, stochastic): c * prod_s x_s (x_s - 1) ... / n_s!
MODELS = {
    # Gillespie's dimerization-decay (Gillespie 2001, the tau-leaping paper), counts scaled down
    "dimer_decay": dict(
        species={"S1": 2000, "S2": 0, "S3": 0}, fixed={},
        reactions=[(1.0, {"S1": 1}, {}), (0.002, {"S1": 2}, {"S2": 1}), (0.5, {"S2": 1}, {"S1": 2}),
                   (0.04, {"S2": 1}, {"S3": 1})],
        t_end=10.0, samples=101),
    # Lotka-Volterra predator and prey: oscillates, and can go extinct
    "lotka_volterra": dict(
        species={"X": 50, "Y": 100}, fixed={},
        reactions=[(1.0, {"X": 1}, {"X": 2}), (0.005, {"X": 1, "Y": 1}, {"Y": 2}), (0.6, {"Y": 1}, {})],
        t_end=30.0, samples=301),
    # Schlogl's model: bistable (about 85 and 560), so one different event can switch the branch
    "schlogl": dict(
        species={"X": 250}, fixed={"A": 100000, "B": 200000},
        reactions=[(3e-7, {"A": 1, "X": 2}, {"X": 3, "A": 1}), (1e-4, {"X": 3}, {"X": 2, "A": 1}),
                   (1e-3, {"B": 1}, {"X": 1, "B": 1}), (3.5, {"X": 1}, {"B": 1})],
        t_end=10.0, samples=101),
    # constitutive gene expression: mRNA and protein, with bursts
    "gene_expression": dict(
        species={"M": 0, "P": 0}, fixed={"G": 1},
        reactions=[(10.0, {"G": 1}, {"G": 1, "M": 1}), (1.0, {"M": 1}, {}), (5.0, {"M": 1}, {"M": 1, "P": 1}),
                   (0.05, {"P": 1}, {})],
        t_end=100.0, samples=201),
}


def h(a):
    a = np.ascontiguousarray(a)
    r = {"sha256": hashlib.sha256(a.tobytes()).hexdigest()[:16], "dtype": str(a.dtype), "shape": list(a.shape)}
    if a.dtype.kind in "iu":
        r["values"] = hashlib.sha256(a.astype("<i8").tobytes()).hexdigest()[:16]
    return r


def times(m):
    return [m["t_end"] * k / (m["samples"] - 1) for k in range(m["samples"])]


def law(c, reactants):
    """the propensity as a formula: c * x * (x - 1) / 2 ..., with c the rate constant's name"""
    f = [c]
    for s, n in sorted(reactants.items()):
        for k in range(n):
            f.append(s if k == 0 else "(%s - %d)" % (s, k))
        if n > 1:
            f.append("/ %d" % math.factorial(n))
    return " * ".join(f).replace("* /", "/")


def mathml(c_name, reactants):
    """the same propensity as MathML: times(c, x, (x - 1), ...) divided by n!"""
    terms = ["<ci>%s</ci>" % c_name]
    div = 1
    for s, n in sorted(reactants.items()):
        for k in range(n):
            terms.append("<ci>%s</ci>" % s if k == 0 else
                         "<apply><minus/><ci>%s</ci><cn type=\"integer\">%d</cn></apply>" % (s, k))
        div *= math.factorial(n)
    prod = "<apply><times/>%s</apply>" % "".join(terms) if len(terms) > 1 else terms[0]
    if div > 1:
        prod = "<apply><divide/>%s<cn type=\"integer\">%d</cn></apply>" % (prod, div)
    return '<math xmlns="http://www.w3.org/1998/Math/MathML">%s</math>' % prod


def sbml(name, m):
    """SBML Level 3 Version 1: amounts only (hasOnlySubstanceUnits), volume 1, the propensity as the kinetic law"""
    sp = []
    for s, x in m["species"].items():
        sp.append('<species id="%s" compartment="cell" initialAmount="%d" hasOnlySubstanceUnits="true" '
                  'boundaryCondition="false" constant="false"/>' % (s, x))
    for s, x in m["fixed"].items():
        sp.append('<species id="%s" compartment="cell" initialAmount="%d" hasOnlySubstanceUnits="true" '
                  'boundaryCondition="true" constant="true"/>' % (s, x))
    rx = []
    for j, (c, rea, pro) in enumerate(m["reactions"]):
        refs = lambda d: "".join('<speciesReference species="%s" stoichiometry="%d" constant="true"/>' % (s, n)
                                 for s, n in sorted(d.items()))
        rx.append('<reaction id="R%d" reversible="false">%s%s<kineticLaw>%s<listOfLocalParameters>'
                  '<localParameter id="c" value="%r"/></listOfLocalParameters></kineticLaw></reaction>'
                  % (j, "<listOfReactants>%s</listOfReactants>" % refs(rea) if rea else "",
                     "<listOfProducts>%s</listOfProducts>" % refs(pro) if pro else "", mathml("c", rea), c))
    return ('<?xml version="1.0" encoding="UTF-8"?>\n'
            '<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">'
            '<model id="%s" substanceUnits="item" timeUnits="second" extentUnits="item">'
            '<listOfCompartments><compartment id="cell" size="1" spatialDimensions="3" constant="true"/>'
            '</listOfCompartments><listOfSpecies>%s</listOfSpecies><listOfReactions>%s</listOfReactions>'
            '</model></sbml>' % (name, "".join(sp), "".join(rx)))


# --- the reference: crdist's Philox draws, a correctly rounded log, IEEE operations, math.fsum ---
def ssa_crdist(m, traj):
    import crdistr
    sp = list(m["species"]) + list(m["fixed"])
    x = dict(m["species"], **m["fixed"])
    tw = crdistr.Stream(SEED, substream=2 * traj)        # waiting times: -log U (binary64, correctly rounded)
    sel = crdistr.Stream(SEED, substream=2 * traj + 1)   # which reaction: U (binary64)
    E, U, k = [], [], 0
    out, ts, t, i = [], times(m), 0.0, 0
    while True:
        a = []
        for c, rea, _ in m["reactions"]:
            hh = 1
            for s, n in rea.items():
                for q in range(n):
                    hh *= x[s] - q
                hh //= math.factorial(n)
            a.append(c * float(hh) if hh > 0 else 0.0)   # one rounding; hh is an exact integer
        a0 = math.fsum(a)                                  # the exact sum, rounded once (no libm)
        if k == len(E):
            E = tw.exponential(4096, dtype=np.float64).tolist()
            U = sel.uniform(4096, dtype=np.float64).tolist()
            k = 0
        tn = t + E[k] / a0 if a0 > 0 else math.inf
        while i < len(ts) and ts[i] < tn:
            out.append([x[s] for s in m["species"]])
            i += 1
        if i == len(ts):
            break
        r, acc, j = U[k] * a0, 0.0, len(a) - 1
        for q, aq in enumerate(a):                          # the first reaction whose running sum exceeds r
            acc += aq
            if acc > r and aq > 0:
                j = q
                break
        else:
            j = max(q for q, aq in enumerate(a) if aq > 0)
        c, rea, pro = m["reactions"][j]
        for s, n in rea.items():
            if s in m["species"]:
                x[s] -= n
        for s, n in pro.items():
            if s in m["species"]:
                x[s] += n
        t, k = tn, k + 1
    return np.array(out, np.int64)


# --- the simulators ---
def run_roadrunner(name, m):
    import roadrunner
    r = roadrunner.RoadRunner(sbml(name, m))
    r.setIntegrator("gillespie")
    res = []
    for traj in range(NTRAJ):
        r.resetAll()
        r.integrator.seed = SEED + traj
        r.integrator.variable_step_size = False
        sel = ["[%s]" % s if False else s for s in m["species"]]
        y = r.simulate(0, m["t_end"], m["samples"], ["time"] + sel)
        res.append(np.rint(np.asarray(y)[:, 1:]).astype(np.int64))
    return np.array(res)


def run_copasi(name, m):
    import basico
    dm = basico.load_model_from_string(sbml(name, m))
    res = []
    for traj in range(NTRAJ):
        tc = basico.run_time_course(model=dm, method="stochastic", duration=m["t_end"],
                                    intervals=m["samples"] - 1, use_seed=True, seed=SEED + traj, use_numbers=True)
        res.append(np.rint(tc[list(m["species"])].to_numpy()).astype(np.int64))
    basico.remove_datamodel(dm)
    return np.array(res)


def gmodel(name, m):
    """GillesPy2's mass action stops at two reactant molecules, so every reaction gets the propensity as a formula,
    and the fixed species are constants in it (parameters), not species"""
    import gillespy2
    g = gillespy2.Model(name=name)
    for s, x0 in m["species"].items():
        g.add_species(gillespy2.Species(name=s, initial_value=x0, mode="discrete"))
    for s, x0 in m["fixed"].items():
        g.add_parameter(gillespy2.Parameter(name=s, expression=str(x0)))
    for j, (c, rea, pro) in enumerate(m["reactions"]):
        g.add_parameter(gillespy2.Parameter(name="c%d" % j, expression=repr(c)))
        keep = lambda d: {s: n for s, n in d.items() if s in m["species"]}
        g.add_reaction(gillespy2.Reaction(name="R%d" % j, reactants=keep(rea), products=keep(pro),
                                          propensity_function=law("c%d" % j, rea)))
    g.timespan(np.array(times(m)))
    return g


def run_gillespy2(name, m, solver):
    import sysconfig
    import gillespy2
    # its C++ solver builds with SCons found on PATH: put this Python's scripts directory there
    os.environ["PATH"] = os.pathsep.join([sysconfig.get_path("scripts"), os.path.dirname(sys.executable),
                                          os.environ.get("PATH", "")])
    g = gmodel(name, m)
    sol = {"numpy": gillespy2.NumPySSASolver, "c": gillespy2.SSACSolver}[solver]
    res = []
    for traj in range(NTRAJ):
        r = g.run(solver=sol, seed=SEED + traj, number_of_trajectories=1)
        res.append(np.stack([np.asarray(r[0][s]) for s in m["species"]], axis=1).astype(np.int64))
    return np.array(res)


SIMS = {
    "crdist": lambda n, m: np.array([ssa_crdist(m, t) for t in range(NTRAJ)]),
    "roadrunner": run_roadrunner,
    "copasi": run_copasi,
    "gillespy2_numpy": lambda n, m: run_gillespy2(n, m, "numpy"),
    "gillespy2_c": lambda n, m: run_gillespy2(n, m, "c"),
}


def _libc():
    import glob
    if glob.glob("/lib/ld-musl-*.so.1"):
        v = ""
        for p in glob.glob("/lib/ld-musl-*.so.1"):
            import subprocess
            o = subprocess.run([p], capture_output=True, text=True).stderr
            if "Version" in o:
                v = o.split("Version")[1].split()[0]
        return ["musl", v]
    return list(platform.libc_ver())


def versions():
    v = {}
    for mod, attr in [("roadrunner", "__version__"), ("basico", "__version__"), ("gillespy2", "__version__"),
                      ("crdistr", "version")]:
        try:
            x = getattr(__import__(mod), attr)
            v[mod] = x() if callable(x) else x
        except Exception as e:
            v[mod] = None
    try:
        import COPASI
        v["COPASI"] = COPASI.CVersion.VERSION.getVersion()
    except Exception:
        v["COPASI"] = None
    return v


def main():
    out = sys.argv[1]
    save = "--save" in sys.argv
    only = sys.argv[sys.argv.index("--only") + 1].split(",") if "--only" in sys.argv else list(SIMS)
    os.makedirs(os.path.join(out, "arrays"), exist_ok=True)
    rep = {"platform": platform.platform(), "machine": platform.machine(), "python": platform.python_version(),
           "libc": _libc(), "mac": platform.mac_ver()[0], "windows": list(platform.win32_ver()),
           "numpy": np.__version__, "versions": versions(), "seed": SEED, "ntraj": NTRAJ, "tests": {}, "errors": {}}
    # the control: the model definitions and their SBML, the same text everywhere
    rep["tests"]["control.models"] = h(np.frombuffer(json.dumps(MODELS, sort_keys=True).encode()
                                                     + "".join(sbml(n, m) for n, m in MODELS.items()).encode(),
                                                     np.uint8))
    for sim in only:
        for name, m in MODELS.items():
            key = "%s.%s" % (sim, name)
            try:
                a = SIMS[sim](name, m)
            except ImportError as e:
                rep["errors"][key] = "not installed: %s" % e
                break
            except Exception as e:
                rep["errors"][key] = "%s: %s" % (type(e).__name__, e)
                continue
            rep["tests"][key] = h(a)
            if save:
                np.save(os.path.join(out, "arrays", key + ".npy"), a)
            print("%-28s %s %s" % (key, a.shape, rep["tests"][key]["values"]), flush=True)
    json.dump(rep, open(os.path.join(out, "report.json"), "w"), indent=1, sort_keys=True)
    for k, e in rep["errors"].items():
        print("error %s: %s" % (k, e))


if __name__ == "__main__":
    main()
