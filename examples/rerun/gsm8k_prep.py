#!/usr/bin/env python3
"""GSM8K questions (Cobbe et al. 2021, "Training Verifiers to Solve Math Word Problems"; MIT license) as chat prompts
for `rerun eval`: each question in a fixed instruction, through the model's own chat template (transformers'
apply_chat_template, from the model's tokenizer_config.json), as token ids. The split's questions in order (N 0), or N
of them drawn with a fixed seed (a dev set from train).

usage: gsm8k_prep.py MODEL_DIR SPLIT.parquet N OUT_PREFIX [--think]
writes OUT_PREFIX.prompts (token ids, a line each) and OUT_PREFIX.tsv (row, gold answer, question)
--think: the template's thinking mode where it has one (Qwen3: enable_thinking); off otherwise
"""
import random
import sys

import pyarrow.parquet as pq
from transformers import AutoTokenizer

INSTRUCTION = ("Solve the following math problem step by step. The last line of your response should be of the form "
               "\"Answer: N\" where N is the final number.\n\n{q}")


def main():
    mdir, path, n, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    think = "--think" in sys.argv[5:]
    tok = AutoTokenizer.from_pretrained(mdir)
    t = pq.read_table(path).to_pydict()
    rows = list(range(len(t["question"])))
    if n:
        random.Random(0).shuffle(rows)
        rows = sorted(rows[:n])
    kw = {"enable_thinking": think} if "enable_thinking" in (tok.chat_template or "") else {}
    lens = []
    with open(f"{out}.prompts", "w") as fp, open(f"{out}.tsv", "w") as ft:
        ft.write("row\tgold\tquestion\n")
        for r in rows:
            q, a = t["question"][r], t["answer"][r]
            gold = a.split("####")[-1].strip().replace(",", "")
            ids = tok.apply_chat_template([{"role": "user", "content": INSTRUCTION.format(q=q)}], add_generation_prompt=True,
                                          tokenize=True, **kw)
            if isinstance(ids, dict):
                ids = ids["input_ids"]
            fp.write(" ".join(map(str, ids)) + "\n")
            ft.write(f"{r}\t{gold}\t{' '.join(q.split())}\n")
            lens.append(len(ids))
    print(f"{len(rows)} questions; prompts {min(lens)} to {max(lens)} tokens")
    print("the first, as text:", repr(tok.decode(ids if len(rows) == 1 else
                                                [int(x) for x in open(f'{out}.prompts').readline().split()])))


if __name__ == "__main__":
    main()
