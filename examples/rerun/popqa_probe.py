#!/usr/bin/env python3
"""Does the model know whether it knows? From `rerun batch` on PopQA
questions (popqa_prep.py): score each greedy answer (correct if it contains
one of the question's accepted answers, case-insensitive, up to the first
newline), then fit a logistic probe on the residual stream at the question's
last token to predict correctness, per layer, and compare it on held-out
questions (AUROC) with two baselines: the model's own confidence (minus the
entropy of its first answer token) and the subject's popularity (log page
views). Writes the best layer's probe as a rerun probe file (its value is the
probe's logit: above 0, "knows").

usage: popqa_probe.py TOKENIZER.json Q.tsv BATCH_PREFIX OUT_PROBE.json
"""
import csv
import json
import sys

import numpy as np
from tokenizers import Tokenizer


def auroc(score, y):
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    # ties: average ranks
    s = score[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j + 2) / 2
        i = j + 1
    pos = y == 1
    n1, n0 = pos.sum(), (~pos).sum()
    return (ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def fit_logistic(X, y, lam, iters=25):
    """L2-regularized logistic regression by Newton's method (binary64)."""
    n, d = X.shape
    Xb = np.hstack([X, np.ones((n, 1))])
    w = np.zeros(d + 1)
    R = lam * np.eye(d + 1)
    R[-1, -1] = 0
    for _ in range(iters):
        z = Xb @ w
        p = 1 / (1 + np.exp(-z))
        g = Xb.T @ (p - y) + R @ w
        H = (Xb * (p * (1 - p))[:, None]).T @ Xb + R
        step = np.linalg.solve(H, g)
        w -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return w[:-1], w[-1]


def main():
    tok = Tokenizer.from_file(sys.argv[1])
    rows = list(csv.DictReader(open(sys.argv[2], encoding="utf-8"), delimiter="\t"))
    pre, out = sys.argv[3], sys.argv[4]
    outs = [list(map(int, l.split())) for l in open(f"{pre}.ids.txt")]
    meta = dict(l.rstrip("\n").split("\t", 1) for l in open(f"{pre}.meta.txt") if "\t" in l)
    D = int(meta["D"])
    n = len(rows)
    assert len(outs) == n, (len(outs), n)
    ans = [tok.decode(o).split("\n")[0].strip() for o in outs]
    y = np.array([any(a.lower() in t.lower() for a in json.loads(r["answers"])) for r, t in zip(rows, ans)], float)
    ent = np.fromfile(f"{pre}.ent.f64", dtype="<f8")
    pop = np.log10(np.array([max(int(r["s_pop"]), 1) for r in rows], float))
    rng = np.random.default_rng(0)
    idx = rng.permutation(n)
    te, tr = idx[: n // 4], idx[n // 4:]
    print(f"{n} questions, {int(y.sum())} answered correctly ({y.mean():.1%}); held out {len(te)}")
    q = np.quantile(pop, [0, 0.25, 0.5, 0.75, 1])
    for a, b in zip(q[:-1], q[1:]):
        m = (pop >= a) & (pop <= b)
        print(f"  popularity 10^{a:.1f} to 10^{b:.1f}: {y[m].mean():.1%} correct")
    print(f"baseline, the model's own confidence (minus first-token entropy): AUROC {auroc(-ent[te], y[te]):.3f}")
    print(f"baseline, the subject's popularity: AUROC {auroc(pop[te], y[te]):.3f}")
    # the question type alone: some types are easy by default (capitals), others hard (producers)
    prop = np.array([r["prop"] for r in rows])
    acc = {p: (y[tr][prop[tr] == p].mean() if (prop[tr] == p).any() else y[tr].mean()) for p in set(prop)}
    print(f"baseline, the question type alone (its training accuracy): AUROC {auroc(np.array([acc[p] for p in prop[te]]), y[te]):.3f}")
    best = None
    for L in [int(x) for x in meta.get("layers", "").split(",") if x] or [4, 8, 12, 16]:
        try:
            X = np.fromfile(f"{pre}.L{L}.f32", dtype="<f4").reshape(n, D).astype(np.float64)
        except FileNotFoundError:
            continue
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        Z = (X - mu) / sd
        # lambda by 4-fold cross-validation on the training questions
        folds = np.array_split(tr, 4)
        lams = [10.0, 100.0, 1000.0, 10000.0]
        cv = []
        for lam in lams:
            sc = []
            for k in range(4):
                va = folds[k]
                fi = np.concatenate([folds[j] for j in range(4) if j != k])
                w, b = fit_logistic(Z[fi], y[fi], lam)
                sc.append(auroc(Z[va] @ w + b, y[va]))
            cv.append(np.mean(sc))
        lam = lams[int(np.argmax(cv))]
        w, b = fit_logistic(Z[tr], y[tr], lam)
        a = auroc(Z[te] @ w + b, y[te])
        print(f"probe after layer {L}: held-out AUROC {a:.3f} (lambda {lam:g}, cross-validated AUROC {max(cv):.3f})")
        if best is None or max(cv) > best[0]:
            best = (max(cv), L, w / sd, b - (mu / sd) @ w, a)
    _, L, w, b, a = best
    # within each question type (held out, types with at least 3 right and 3 wrong): probe against confidence
    X = np.fromfile(f"{pre}.L{L}.f32", dtype="<f4").reshape(n, D).astype(np.float64)
    sc = X @ w + b
    wins = []
    for p in sorted(set(prop)):
        m = prop[te] == p
        yy = y[te][m]
        if yy.sum() >= 3 and (1 - yy).sum() >= 3:
            ap, ae = auroc(sc[te][m], yy), auroc(-ent[te][m], yy)
            wins.append(ap > ae)
            print(f"  within {p}: probe {ap:.3f}, confidence {ae:.3f} ({int(yy.sum())} right of {len(yy)})")
    print(f"within types, the probe beats the model's confidence in {sum(wins)} of {len(wins)}")
    w32 = w.astype(np.float32)
    json.dump({"name": f"knows the answer (PopQA, layer {L})", "layer": L, "bias": float(np.float32(b)),
               "w": [float(f"{v:.9g}") for v in w32]}, open(out, "w"))
    print(f"chosen by cross-validation: layer {L}, held-out AUROC {a:.3f}; written {out}")


if __name__ == "__main__":
    main()
