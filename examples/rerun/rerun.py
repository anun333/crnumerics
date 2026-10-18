#!/usr/bin/env python3
"""Text in, text out, for rerun (rerun.c), which works on token ids: the
model's own tokenizer (tokenizer.json, through the tokenizers library) turns
the prompt into ids and the output back into text. Tokenizing is integer
work; the receipt covers the ids, and the text this adds to it
("prompt_text", "output_text") is for reading, not verified.

usage: rerun.py gen MODEL_DIR "prompt" [--max N] [--temp T] [--top-p P] [--seed S] [--threads N] [--out RECEIPT.json]
       rerun.py verify RECEIPT.json [MODEL_DIR] [--threads N]
       rerun.py show RECEIPT.json MODEL_DIR
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
BIN = os.path.join(HERE, "rerun")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("gen", "verify", "show"))
    ap.add_argument("a")
    ap.add_argument("b", nargs="?")
    ap.add_argument("--max", type=int, default=64)
    ap.add_argument("--temp", default="0")
    ap.add_argument("--top-p", default="1")
    ap.add_argument("--seed", default="0")
    ap.add_argument("--threads", default="1")
    ap.add_argument("--out", default="receipt.json")
    a = ap.parse_args()
    if a.cmd == "verify":
        sys.exit(subprocess.call([BIN, "verify", a.a] + ([a.b] if a.b else []) + ["--threads", a.threads]))
    if a.cmd == "show":
        tok = Tokenizer.from_file(os.path.join(a.b, "tokenizer.json"))
        r = json.load(open(a.a))
        print(tok.decode(r["prompt_ids"]) + "│" + tok.decode(r["output_ids"]))
        return
    tok = Tokenizer.from_file(os.path.join(a.a, "tokenizer.json"))
    ids = tok.encode(a.b).ids
    with tempfile.NamedTemporaryFile("w", suffix=".ids", delete=False) as f:
        f.write(" ".join(map(str, ids)))
        idfile = f.name
    try:
        rc = subprocess.call([BIN, "gen", a.a, idfile, str(a.max), a.out, "--temp", a.temp, "--top-p", a.top_p,
                              "--seed", a.seed, "--threads", a.threads], stdout=subprocess.DEVNULL)
    finally:
        os.unlink(idfile)
    if rc:
        sys.exit(rc)
    r = json.load(open(a.out))
    r["prompt_text"] = a.b
    r["output_text"] = tok.decode(r["output_ids"])
    json.dump(r, open(a.out, "w"), indent=1)
    print(a.b + r["output_text"])
    print(f"\n[receipt: {a.out}; chain {r['chain_sha256'][:16]}; {len(r['output_ids'])} tokens]", file=sys.stderr)


if __name__ == "__main__":
    main()
