#!/usr/bin/env python3
"""Turn gpt2sae's output directory into readout.py's .npz format, so that
compare.py compares the two kinds of run.

usage: pack.py OUTDIR OUT.npz
"""
import platform
import sys

import numpy as np

LAYERS = (2, 6, 10)


def main():
    d, out = sys.argv[1], sys.argv[2]
    meta = dict(line.rstrip("\n").split("\t", 1) for line in open(f"{d}/meta.txt"))
    a = {"resid6": np.fromfile(f"{d}/resid6.f32", dtype="<f4").reshape(10, 128, 768)}
    for L in LAYERS:
        a[f"cnt{L}"] = np.fromfile(f"{d}/cnt{L}.i32", dtype="<i4")
        a[f"idx{L}"] = np.fromfile(f"{d}/idx{L}.i32", dtype="<i4")
        a[f"val{L}"] = np.fromfile(f"{d}/val{L}.f32", dtype="<f4")
        a[f"top{L}"] = np.fromfile(f"{d}/top{L}.i32", dtype="<i4").reshape(-1, 10)
        a[f"rsha{L}"] = np.array(meta[f"rsha{L}"])
    a["meta"] = np.array(repr({"torch": f"none (gpt2sae {meta['mode']})", "machine": platform.machine(),
                               "system": platform.system(), "threads": int(meta["threads"]), "batch": "-",
                               "seconds": float(meta["seconds"]), "compiler": meta["compiler"],
                               "flags": meta["flags"]}))
    np.savez_compressed(out, **a)


if __name__ == "__main__":
    main()
