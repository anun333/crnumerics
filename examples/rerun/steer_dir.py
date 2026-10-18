#!/usr/bin/env python3
"""A steering direction from two prompt variants (contrastive activation addition, Rimsky et al. 2023): the mean
over questions of (variant B's residual stream minus variant A's) at each prompt's last token, after one layer, from
two `rerun batch ... --layers L --max 1` dumps of the same questions. Written as a rerun probe file (its direction
is what --steer adds, scaled to ALPHA); prints the direction's length, the residual stream's median length there,
and how consistent the per-question differences are (their mean cosine with the direction).

usage: steer_dir.py A_PREFIX B_PREFIX LAYER OUT.json
"""
import json
import sys

import numpy as np


def main():
    a, b, L, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    ma = dict(l.rstrip("\n").split("\t", 1) for l in open(f"{a}.meta.txt") if "\t" in l)
    D, n = int(ma["D"]), int(ma["prompts"])
    XA = np.fromfile(f"{a}.L{L}.f32", dtype="<f4").reshape(n, D).astype(np.float64)
    XB = np.fromfile(f"{b}.L{L}.f32", dtype="<f4").reshape(n, D).astype(np.float64)
    d = XB - XA
    v = d.mean(0)
    cos = (d @ v) / (np.linalg.norm(d, axis=1) * np.linalg.norm(v))
    print(f"{n} questions, layer {L}: direction length {np.linalg.norm(v):.3f}; residual median length "
          f"{np.median(np.linalg.norm(XA, axis=1)):.3f}; per-question differences' cosine with it: mean {cos.mean():.3f}, "
          f"min {cos.min():.3f}")
    json.dump({"name": f"B minus A, layer {L} (steer_dir.py)", "layer": L, "bias": 0.0,
               "w": [float(f"{x:.9g}") for x in v.astype(np.float32)]}, open(out, "w"))
    print(f"written {out}; its length {np.linalg.norm(v):.6g} is ALPHA 1 in the units --steer uses")


if __name__ == "__main__":
    main()
