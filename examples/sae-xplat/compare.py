#!/usr/bin/env python3
"""Compare readout.py's files: for every pair against the first, per layer,
whether the residual stream and the features are the same bits, how many (token, feature)
pairs are active in one run and not the other, how large those features
were where they fired, how far the features active in both differ, and how
many tokens' top feature and top 10 changed.

usage: compare.py NAME=FILE.npz [NAME=FILE.npz ...]
"""
import ast
import hashlib
import sys

import numpy as np

LAYERS = (2, 6, 10)


def sets(z, L):
    cnt, idx, val = z[f"cnt{L}"], z[f"idx{L}"], z[f"val{L}"]
    off = np.concatenate([[0], np.cumsum(cnt)])
    return [(idx[off[t]:off[t + 1]], val[off[t]:off[t + 1]]) for t in range(len(cnt))]


def main():
    runs = [a.split("=", 1) for a in sys.argv[1:]]
    data = {n: np.load(f) for n, f in runs}
    base = runs[0][0]
    for L in LAYERS:
        classes = {}
        for n, _ in runs:
            classes.setdefault(data[n][f"rsha{L}"].item(), []).append(n)
        feats = {}
        for n, _ in runs:
            h = hashlib.sha256()
            for k in ("cnt", "idx", "val", "top"):
                h.update(np.ascontiguousarray(data[n][f"{k}{L}"]).tobytes())
            feats.setdefault(h.hexdigest(), []).append(n)
        print(f"== layer {L}: {len(classes)} distinct residual stream(s): "
              + " | ".join("{" + ", ".join(v) + "}" for v in classes.values())
              + f"; {len(feats)} distinct feature set(s): " + " | ".join("{" + ", ".join(v) + "}" for v in feats.values()))
        if len(classes) == 1 and len(feats) == 1:
            continue
        b = sets(data[base], L)
        bt = data[base][f"top{L}"]
        total = sum(len(i) for i, _ in b)
        for n, _ in runs[1:]:
            if data[n][f"rsha{L}"].item() == data[base][f"rsha{L}"].item() and \
                    all(np.array_equal(data[n][f"{k}{L}"], data[base][f"{k}{L}"]) for k in ("cnt", "idx", "val", "top")):
                continue
            o = sets(data[n], L)
            flips, flipmax, tokflip, both = 0, 0.0, 0, 0.0
            for (ia, va), (ib, vb) in zip(b, o):
                da = dict(zip(ia.tolist(), va.tolist()))
                db = dict(zip(ib.tolist(), vb.tolist()))
                only = set(da) ^ set(db)
                if only:
                    tokflip += 1
                    flips += len(only)
                    flipmax = max(flipmax, max(da.get(k, db.get(k, 0.0)) for k in only))
                for k in set(da) & set(db):
                    both = max(both, abs(da[k] - db[k]))
            ot = data[n][f"top{L}"]
            top1 = int(np.sum(bt[:, 0] != ot[:, 0]))
            top10 = int(np.sum(np.any(bt != ot, axis=1)))
            print(f"  {base} vs {n}: {flips} of {total} active (token, feature) pairs flip on or off, in {tokflip} of "
                  f"{len(b)} tokens, the largest {flipmax:.3g}; active in both: values differ by up to {both:.3g}; "
                  f"top feature changed in {top1} tokens, top 10 (in order) in {top10}")
    r = {n: data[n]["resid6"] for n, _ in runs if data[n]["resid6"].size}
    if base in r:
        for n in r:
            if n != base:
                d = np.abs(r[n] - r[base])
                print(f"  layer 6 residual stream, first 10 sequences, {base} vs {n}: max |diff| {d.max():.3g} "
                      f"(values up to {np.abs(r[base]).max():.3g})")
    for n, _ in runs:
        m = ast.literal_eval(data[n]["meta"].item())
        print(f"  {n}: torch {m['torch']}, {m['machine']} {m['system']}, threads {m['threads']}, batch {m['batch']}, "
              f"{m['seconds']} s")


if __name__ == "__main__":
    main()
