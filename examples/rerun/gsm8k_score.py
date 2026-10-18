#!/usr/bin/env python3
"""Score `rerun eval` answers to gsm8k_prep.py prompts. An answer's number: the first number after its last
"Answer:", or, without one, the last number in it (commas and $ dropped); right if it equals the gold number. With K
samples a question, also the majority answer (ties to the answer seen first). Writes OUT.correct.txt: a 0/1 a
question (the majority with K > 1), for paired comparisons (gsm8k_compare.py).

usage: gsm8k_score.py MODEL_DIR Q.tsv EVAL_PREFIX [--samples K]
"""
import csv
import re
import sys
from collections import Counter

from tokenizers import Tokenizer

NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def number(text):
    t = text.replace("$", "")
    i = t.rfind("Answer:")
    found = NUM.findall(t[i + 7:]) if i >= 0 else []
    if found:
        s = found[0]
    else:
        allnums = NUM.findall(t)
        if not allnums:
            return None, i >= 0
        s = allnums[-1]
    s = s.replace(",", "").rstrip(".")
    try:
        return float(s), i >= 0
    except ValueError:
        return None, i >= 0


def main():
    mdir, qpath, pre = sys.argv[1], sys.argv[2], sys.argv[3]
    K = int(sys.argv[sys.argv.index("--samples") + 1]) if "--samples" in sys.argv else 1
    tok = Tokenizer.from_file(f"{mdir}/tokenizer.json")
    rows = list(csv.DictReader(open(qpath, encoding="utf-8"), delimiter="\t"))
    outs = [list(map(int, l.split())) for l in open(f"{pre}.ids.txt")]
    meta = dict(l.rstrip("\n").split("\t", 1) for l in open(f"{pre}.meta.txt") if "\t" in l)
    maxn = int(meta["max"])
    assert len(outs) == len(rows) * K, (len(outs), len(rows), K)
    right, fmt, cut, maj, lens = 0, 0, 0, [], []
    for i, r in enumerate(rows):
        gold = float(r["gold"])
        votes = []
        for k in range(K):
            o = outs[i * K + k]
            lens.append(len(o))
            cut += len(o) == maxn
            v, has = number(tok.decode(o))
            fmt += has
            right += v is not None and v == gold
            votes.append(v)
        c = Counter(v for v in votes if v is not None)
        best = max(c.items(), key=lambda kv: (kv[1], -votes.index(kv[0])))[0] if c else None
        maj.append(int(best is not None and best == gold))
    n = len(rows)
    print(f"{n} questions x {K}: {right / (n * K):.1%} right per answer ({right} of {n * K}); "
          f"'Answer:' line in {fmt / (n * K):.1%}; cut at {maxn} tokens: {cut}; mean length {sum(lens) / len(lens):.0f} tokens")
    if K > 1:
        print(f"majority of {K}: {sum(maj) / n:.1%} right ({sum(maj)} of {n})")
    open(f"{pre}.correct.txt", "w").write("".join(f"{x}\n" for x in maj))


if __name__ == "__main__":
    main()
