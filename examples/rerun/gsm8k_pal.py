#!/usr/bin/env python3
"""Program-aided GSM8K: from `rerun eval` answers to gsm8k_prep.py --pal prompts, take each answer's first ```python
block (or the whole answer when it has none), run every program in one throwaway container (docker, no network, a
read-only file system, 256 MB, 1 CPU; each program in its own `python -I` process with a 5-second limit), and score
the last number it printed against the gold answer. Writes OUT.correct.txt (0/1 a question) like gsm8k_score.py, and
OUT.pal.tsv (each program's exit, the number it printed, right or not).

usage: gsm8k_pal.py MODEL_DIR Q.tsv EVAL_PREFIX [--image python:3.12-alpine]
"""
import csv
import json
import re
import subprocess
import sys

from tokenizers import Tokenizer

NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
RUNNER = r'''
import json, subprocess, sys
progs = json.load(sys.stdin)
out = []
for code in progs:
    try:
        r = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=5)
        out.append([r.returncode, r.stdout[-2000:]])
    except subprocess.TimeoutExpired:
        out.append([-9, ""])
json.dump(out, sys.stdout)
'''


def code_of(text):
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    if m:
        return m.group(1)
    m = re.search(r"```(?:python|py)?\s*\n(.*)", text, re.S)  # an unclosed block (cut at the limit)
    return m.group(1) if m else text


def main():
    mdir, qtsv, pre = sys.argv[1], sys.argv[2], sys.argv[3]
    image = sys.argv[sys.argv.index("--image") + 1] if "--image" in sys.argv else "python:3.12-alpine"
    tok = Tokenizer.from_file(f"{mdir}/tokenizer.json")
    rows = list(csv.DictReader(open(qtsv, encoding="utf-8"), delimiter="\t"))
    outs = [list(map(int, l.split())) for l in open(f"{pre}.ids.txt")]
    assert len(outs) == len(rows), (len(outs), len(rows))
    progs = [code_of(tok.decode(o)) for o in outs]
    r = subprocess.run(["docker", "run", "--rm", "-i", "--network", "none", "--read-only", "--memory", "256m", "--cpus", "1",
                        "--pids-limit", "64", image, "python", "-I", "-c", RUNNER], input=json.dumps(progs),
                       capture_output=True, text=True, timeout=3600)
    if r.returncode:
        sys.exit(f"the container failed: {r.stderr[-500:]}")
    res = json.loads(r.stdout)
    assert len(res) == len(rows)
    right, ran, correct = 0, 0, []
    with open(f"{pre}.pal.tsv", "w") as f:
        f.write("row\texit\tprinted\tgold\tright\n")
        for row, (rc, so) in zip(rows, res):
            nums = NUM.findall(so.replace("$", ""))
            v = None
            if nums:
                try:
                    v = float(nums[-1].replace(",", "").rstrip("."))
                except ValueError:
                    v = None
            ok = int(v is not None and abs(v - float(row["gold"])) < 1e-6)
            right += ok
            ran += rc == 0
            correct.append(ok)
            f.write(f"{row['row']}\t{rc}\t{v}\t{row['gold']}\t{ok}\n")
    n = len(rows)
    print(f"{n} questions: {right / n:.1%} right ({right}); programs that ran cleanly {ran / n:.1%}")
    open(f"{pre}.correct.txt", "w").write("".join(f"{x}\n" for x in correct))


if __name__ == "__main__":
    main()
