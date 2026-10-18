#!/usr/bin/env python3
"""Questions for a knowledge probe: N PopQA questions (Mallen et al. 2023,
"When Not to Trust Language Models"; entity questions from Wikidata with
the subject's popularity), sampled with a fixed seed, each after three fixed
examples, as token ids for `rerun batch`; and a TSV of what the answer must
contain and the subject's popularity, for scoring.

usage: popqa_prep.py TOKENIZER.json test.tsv N OUT_PREFIX
writes OUT_PREFIX.prompts (token ids, a line each) and OUT_PREFIX.tsv
"""
import csv
import json
import random
import sys

from tokenizers import Tokenizer

SHOTS = [  # three fixed examples in PopQA's form; any that is a PopQA row is left out of the sample
    ("What is the capital of Ghana?", "Accra"),
    ("Who was the director of Jaws?", "Steven Spielberg"),
    ("What genre is Metallica?", "heavy metal"),
]


def main():
    tok = Tokenizer.from_file(sys.argv[1])
    rows = list(csv.DictReader(open(sys.argv[2], encoding="utf-8"), delimiter="\t"))
    n, out = int(sys.argv[3]), sys.argv[4]
    shot_q = {q for q, _ in SHOTS}
    rows = [r for r in rows if r["question"] not in shot_q]
    random.Random(0).shuffle(rows)
    rows = rows[:n]
    prefix = "".join(f"Q: {q}\nA: {a}\n\n" for q, a in SHOTS)
    with open(f"{out}.prompts", "w") as fp, open(f"{out}.tsv", "w") as ft:
        ft.write("id\ts_pop\tprop\tquestion\tanswers\n")
        for r in rows:
            ids = tok.encode(prefix + f"Q: {r['question']}\nA:").ids
            fp.write(" ".join(map(str, ids)) + "\n")
            ft.write(f"{r['id']}\t{r['s_pop']}\t{r['prop']}\t{r['question']}\t{json.dumps(json.loads(r['possible_answers']))}\n")
    print(f"{len(rows)} questions; prompt lengths up to {max(len(tok.encode(prefix + 'Q: ' + r['question'] + chr(10) + 'A:').ids) for r in rows)} tokens")


if __name__ == "__main__":
    main()
