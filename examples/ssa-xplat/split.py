#!/usr/bin/env python3
"""split.py BASE OTHER...: where do the trajectories split? For each (simulator, model) array saved by
ssa_xplat.py --save in BASE/arrays and in each OTHER/arrays, and for each trajectory, the first sample at which the
species counts differ from BASE's, as a fraction of the run: "same" if never. 2026-10-15."""
import os
import sys

import numpy as np


def first_split(a, b):
    """per trajectory, the first differing sample index, or None"""
    out = []
    for i in range(a.shape[0]):
        d = np.nonzero((a[i] != b[i]).any(axis=1))[0]
        out.append(int(d[0]) if len(d) else None)
    return out


def main():
    base, others = sys.argv[1], sys.argv[2:]
    tests = sorted(f[:-4] for f in os.listdir(os.path.join(base, "arrays")) if f.endswith(".npy"))
    names = [os.path.basename(o.rstrip("/")) for o in others]
    print("%-32s %s" % ("first split from " + os.path.basename(base.rstrip("/")) + " (% of the run, per trajectory)",
                        "  ".join("%-22s" % n for n in names)))
    for t in tests:
        a = np.load(os.path.join(base, "arrays", t + ".npy"))
        cells = []
        for o in others:
            p = os.path.join(o, "arrays", t + ".npy")
            if not os.path.exists(p):
                cells.append("absent")
                continue
            b = np.load(p)
            if b.shape != a.shape:
                cells.append("shape %s" % (b.shape,))
                continue
            s = first_split(a, b)
            if all(x is None for x in s):
                cells.append("same")
            else:
                n = a.shape[1] - 1
                cells.append("%d/%d split, at %s" % (sum(x is not None for x in s), len(s),
                                                    ",".join("-" if x is None else "%d%%" % round(100 * x / n) for x in s)))
        print("%-32s %s" % (t, "  ".join("%-22s" % c for c in cells)))


main()
