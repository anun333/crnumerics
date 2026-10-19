#!/usr/bin/env python3
"""Spend the votes where the model is unsure. From one model's greedy answers (gsm8k_score.py's correct.txt), its
residual stream at each question's last prompt token (`rerun batch ... --layers L --max 1`) and K seeded samples of
the first N questions (`rerun eval --samples K`): a logistic probe of "greedy will be wrong", fit on the questions
after the first N (lambda by 4-fold cross-validation), ranks the first N; voting on the probe's least confident
fraction f, greedy elsewhere, gives an accuracy at a cost (greedy 1 a question, a vote K more). Compared at the same
cost with votes on a random fraction (the exact expectation) and with a gate needing no probe: two samples, and three
more only when they disagree (the answer: their common value, or the majority of all K).

usage: gsm8k_gate.py MODEL_DIR Q.tsv GREEDY_PREFIX SAMPLES_PREFIX DUMP_PREFIX LAYER [TRAIN_GREEDY TRAIN_DUMP]
with TRAIN_GREEDY and TRAIN_DUMP (another question set's correct.txt prefix and dump prefix), the probe is fit on
that set instead, and ranks all N questions that have samples.
"""
import csv
import sys
from collections import Counter

import numpy as np
from tokenizers import Tokenizer

from gsm8k_score import number
from popqa_probe import auroc, fit_logistic


def majority(vals):
    c = Counter(v for v in vals if v is not None)
    return max(c.items(), key=lambda kv: (kv[1], -vals.index(kv[0])))[0] if c else None


def main():
    mdir, qtsv, gpre, spre, dpre, L = sys.argv[1:7]
    L = int(L)
    tok = Tokenizer.from_file(f"{mdir}/tokenizer.json")
    rows = list(csv.DictReader(open(qtsv, encoding="utf-8"), delimiter="\t"))
    gold = np.array([float(r["gold"]) for r in rows])
    g = np.array([int(x) for x in open(f"{gpre}.correct.txt").read().split()])
    sm = dict(l.rstrip("\n").split("\t", 1) for l in open(f"{spre}.meta.txt") if "\t" in l)
    K = int(sm["samples"])
    souts = [list(map(int, l.split())) for l in open(f"{spre}.ids.txt")]
    N = len(souts) // K
    vals = [[number(tok.decode(souts[i * K + k]))[0] for k in range(K)] for i in range(N)]
    vote = np.array([int(majority(v) == gold[i]) for i, v in enumerate(vals)])
    # the agreement gate: two samples; if they agree, their value; else the majority of all K
    agree = np.array([vals[i][0] is not None and vals[i][0] == vals[i][1] for i in range(N)])
    gate2 = np.array([int((vals[i][0] if agree[i] else majority(vals[i])) == gold[i]) for i in range(N)])
    cost2 = np.where(agree, 2, K).mean()
    D = int(dict(l.rstrip("\n").split("\t", 1) for l in open(f"{dpre}.meta.txt") if "\t" in l)["D"])
    X = np.fromfile(f"{dpre}.L{L}.f32", dtype="<f4").reshape(-1, D).astype(np.float64)
    if len(sys.argv) > 8:  # fit on another set's questions, rank these
        gt = np.array([int(x) for x in open(f"{sys.argv[7]}.correct.txt").read().split()])
        Xt = np.fromfile(f"{sys.argv[8]}.L{L}.f32", dtype="<f4").reshape(-1, D).astype(np.float64)
        X = np.vstack([X[:N], Xt])
        g = np.concatenate([g[:N], gt])
    tr = np.arange(N, len(g))
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Z = (X - mu) / sd
    folds = np.array_split(tr, 4)
    best = None
    for lam in [100.0, 1000.0, 10000.0, 100000.0]:
        sc = []
        for k in range(4):
            fi = np.concatenate([folds[j] for j in range(4) if j != k])
            w, b = fit_logistic(Z[fi], g[fi], lam)
            sc.append(auroc(Z[folds[k]] @ w + b, g[folds[k]]))
        if best is None or np.mean(sc) > best[0]:
            best = (np.mean(sc), lam)
    w, b = fit_logistic(Z[tr], g[tr], best[1])
    s = Z[:N] @ w + b
    print(f"probe after layer {L}, fit on {len(tr)} questions (lambda {best[1]:g}, cross-validated AUROC {best[0]:.3f}); "
          f"on the first {N}: AUROC {auroc(s, g[:N]):.3f} for 'greedy is right'")
    print(f"greedy {g[:N].mean():.1%} (cost 1), majority of {K} everywhere {vote.mean():.1%} (cost {1 + K})")
    order = np.argsort(s, kind="mergesort")  # least confident first
    print(f"{'voted':>6} {'cost':>5} {'probe gate':>10} {'random gate':>11}")
    for f in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
        m = int(round(f * N))
        pick = np.zeros(N, bool)
        pick[order[:m]] = True
        acc = np.where(pick, vote, g[:N]).mean()
        rnd = g[:N].mean() + (m / N) * (vote.mean() - g[:N].mean())
        print(f"{f:6.0%} {1 + K * m / N:5.2f} {acc:10.1%} {rnd:11.1%}")
    print(f"no probe: two samples, {K - 2} more when they disagree ({(~agree).mean():.0%} of questions): "
          f"{gate2.mean():.1%} at cost {cost2:.2f} (no greedy run)")


if __name__ == "__main__":
    main()
