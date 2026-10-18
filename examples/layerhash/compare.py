#!/usr/bin/env python3
"""Compare layerhash runs from several machines: for each build, which
machines share a chain hash, and where the first pair that doesn't parts
(the first prompt, position and layer whose hash differs), whether the
generated tokens differ, and how far apart the logits after each prompt are.

usage: compare.py NAME=OUTBASE [NAME=OUTBASE ...]
"""
import sys

import numpy as np


def read_layers(path):
    rows = []
    for line in open(path).read().split("\n")[1:]:
        if line:
            p, pos, layer, h = line.split("\t")
            rows.append(((int(p), int(pos), int(layer)), h))
    return rows


def read_tokens(path):
    out = {}
    for line in open(path).read().split("\n")[1:]:
        if line:
            p, g, t, _ = line.split("\t")
            out.setdefault(int(p), []).append(int(t))
    return out


def main():
    runs = dict(a.split("=", 1) for a in sys.argv[1:])
    names = list(runs)
    for mode in ("cr", "tree", "stock"):
        chains = {}
        for n in names:
            chains[n] = open(f"{runs[n]}/{mode}.chain").read().split()[1]
        classes = {}
        for n in names:
            classes.setdefault(chains[n], []).append(n)
        print(f"== {mode}: {len(classes)} distinct chain hash(es): "
              + " | ".join("{" + ", ".join(v) + "}" for v in classes.values()))
        if len(classes) == 1:
            continue
        base = names[0]
        bl = read_layers(f"{runs[base]}/{mode}/layers.tsv")
        bt = read_tokens(f"{runs[base]}/{mode}/tokens.tsv")
        for n in names[1:]:
            if chains[n] == chains[base]:
                continue
            nl = read_layers(f"{runs[n]}/{mode}/layers.tsv")
            first = next((a[0] for a, b in zip(bl, nl) if a != b), None)
            ndiff = sum(a != b for a, b in zip(bl, nl))
            nt = read_tokens(f"{runs[n]}/{mode}/tokens.tsv")
            tok = [p for p in bt if bt[p] != nt.get(p)]
            dl = []
            for p in sorted(bt):
                a = np.fromfile(f"{runs[base]}/{mode}/logits-{p}.f32", dtype="<f4")
                b = np.fromfile(f"{runs[n]}/{mode}/logits-{p}.f32", dtype="<f4")
                dl.append(float(np.max(np.abs(a - b))))
            print(f"  {base} vs {n}: first difference at prompt {first[0]}, position {first[1]}, "
                  f"layer {first[2]}; {ndiff} of {len(bl)} layer hashes differ; "
                  f"logits after the prompts differ by up to {max(dl):.3g}; "
                  f"generated tokens differ in prompts {tok if tok else 'none'}")
    # the builds against each other on one machine
    for n in names:
        tok = {m: read_tokens(f"{runs[n]}/{m}/tokens.tsv") for m in ("cr", "tree", "stock")}
        same_ct = tok["cr"] == tok["tree"]
        same_cs = tok["cr"] == tok["stock"]
        print(f"  on {n}: tokens cr = tree: {same_ct}; cr = stock: {same_cs}")


if __name__ == "__main__":
    main()
