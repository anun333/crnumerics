#!/usr/bin/env python3
"""Train a linear probe on rerun's dumps: from the residual stream after a
middle layer, predict the entropy of the model's own next-token
distribution (how unsure the model is about to be), and say how well it
does on sequences it was not trained on. The labels come from rerun, so they
are the same bits on every machine; the probe is ridge regression (binary64,
numpy) on standardized inputs, written out as a probe file rerun reads
(w and bias in the raw residual stream's units).

usage: probe_train.py LAYER OUT.json PART [PART ...]
PART is a dump prefix (rerun dump ... PART --layers ...); the parts' sequences
are taken in order and the last 20% held out.
"""
import hashlib
import json
import sys

import numpy as np


def main():
    layer, out, parts = int(sys.argv[1]), sys.argv[2], sys.argv[3:]
    meta = [dict(l.rstrip("\n").split("\t", 1) for l in open(f"{p}.meta.txt") if "\t" in l) for p in parts]
    D = int(meta[0]["D"])
    X = np.concatenate([np.fromfile(f"{p}.L{layer}.f32", dtype="<f4").reshape(-1, D) for p in parts]).astype(np.float64)
    y = np.concatenate([np.fromfile(f"{p}.ent.f64", dtype="<f8") for p in parts])
    n = len(y)
    cut = int(n * 0.8) // 128 * 128  # whole sequences of 128
    Xtr, ytr, Xte, yte = X[:cut], y[:cut], X[cut:], y[cut:]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Z = (Xtr - mu) / sd
    lam = 10.0
    A = Z.T @ Z + lam * np.eye(D)
    beta = np.linalg.solve(A, Z.T @ (ytr - ytr.mean()))
    w = beta / sd
    b = ytr.mean() - mu @ w

    def score(Xs, ys):
        p = Xs @ w + b
        r2 = 1 - np.sum((ys - p) ** 2) / np.sum((ys - ys.mean()) ** 2)
        return r2, np.corrcoef(p, ys)[0, 1]

    r2tr, rtr = score(Xtr, ytr)
    r2te, rte = score(Xte, yte)
    w32 = w.astype(np.float32)
    probe = {"name": f"next-token entropy from layer {layer}", "layer": layer, "bias": float(np.float32(b)),
             "w": [float(f"{v:.9g}") for v in w32]}
    data_sha = hashlib.sha256(b"".join(open(f"{p}.ent.f64", "rb").read() for p in parts)).hexdigest()
    with open(out, "w") as f:
        json.dump(probe, f)
    print(f"layer {layer}: {n} tokens ({cut} train, {n - cut} held out); ridge {lam} on standardized inputs; "
          f"R^2 train {r2tr:.3f}, held out {r2te:.3f}; correlation held out {rte:.3f}; "
          f"entropy mean {y.mean():.3f} nats (sd {y.std():.3f}); labels sha256 {data_sha[:16]}")


if __name__ == "__main__":
    main()
