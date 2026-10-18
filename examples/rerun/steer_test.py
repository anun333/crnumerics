#!/usr/bin/env python3
"""Does the probe read something the model uses? Steer the residual stream
along the probe's direction (rerun dump --steer) on held-out sequences, by
several multiples of the stream's typical size at that layer, and measure
the model's own next-token entropy; then the same along random directions
of the same layer. A probe that reads a quantity the model computes with
should move the entropy, in the direction it predicts, more than random
directions do. With exact replay, every change is the steering's.

usage: steer_test.py MODEL_DIR SEQS PROBE.json PART_PREFIX FROM TO [NRANDOM]
PART_PREFIX: a dump of the same sequences with the probe's layer (for the
stream's typical size); FROM, TO: the held-out sequences; up to 4 runs at a
time.
"""
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FRACS = (-0.4, -0.2, 0.0, 0.2, 0.4)


def run_all(jobs):
    procs = []
    for cmd in jobs:
        procs.append(subprocess.Popen(["nice", "-n", "19"] + cmd, stdout=subprocess.DEVNULL))
        if len(procs) == 4:
            for p in procs:
                if p.wait():
                    sys.exit("a dump failed")
            procs = []
    for p in procs:
        if p.wait():
            sys.exit("a dump failed")


def main():
    mdir, seqs, probe_path, part, a, b = sys.argv[1:7]
    nrand = int(sys.argv[7]) if len(sys.argv) > 7 else 2
    probe = json.load(open(probe_path))
    L, D = probe["layer"], len(probe["w"])
    X = np.fromfile(f"{part}.L{L}.f32", dtype="<f4").reshape(-1, D)
    size = float(np.median(np.linalg.norm(X.astype(np.float64), axis=1)))
    out = os.path.dirname(os.path.abspath(probe_path))
    dirs = {"probe": probe_path}
    rng = np.random.default_rng(1)
    for k in range(nrand):
        v = rng.standard_normal(D)
        p = os.path.join(out, f"random{k}.json")
        json.dump({"name": f"random direction {k}", "layer": L, "bias": 0.0, "w": [float(f"{x:.9g}") for x in v.astype(np.float32)]},
                  open(p, "w"))
        dirs[f"random{k}"] = p
    jobs, keys = [], []
    for name, path in dirs.items():
        for f in FRACS:
            if f == 0 and name != "probe":
                continue
            pre = os.path.join(out, f"steer-{name}-{f:+.2f}")
            cmd = [os.path.join(HERE, "rerun"), "dump", mdir, seqs, pre, "--from", a, "--to", b]
            if f != 0:
                cmd += ["--steer", path, repr(f * size)]
            jobs.append(cmd)
            keys.append((name, f, pre))
    run_all(jobs)
    base = np.fromfile(os.path.join(out, "steer-probe-+0.00.ent.f64"), dtype="<f8")
    print(f"layer {L}; the residual stream's median size there {size:.4g}; {len(base)} held-out tokens; "
          f"mean entropy unsteered {base.mean():.4f} nats")
    print("direction    " + "  ".join(f"{f:+.1f}x" for f in FRACS) + "   (change in mean entropy, nats)")
    for name in dirs:
        row = []
        for f in FRACS:
            pre = os.path.join(out, f"steer-{name}-{f:+.2f}") if (f != 0 or name == "probe") else None
            e = np.fromfile(f"{pre}.ent.f64", dtype="<f8") if pre else base
            row.append(f"{e.mean() - base.mean():+.4f}")
        print(f"{name:12s} " + "  ".join(f"{r:>6s}" for r in row))


if __name__ == "__main__":
    main()
