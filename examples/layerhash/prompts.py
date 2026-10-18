#!/usr/bin/env python3
"""Tokenize the prompts with SmolLM2's tokenizer, so the C program needs no
tokenizer: writes prompts.tsv (id, tab, token ids) next to this file.

usage: prompts.py TOKENIZER.json
"""
import os
import sys

from tokenizers import Tokenizer

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n",
    "In 1905, Albert Einstein published",
    "The mitochondria is the",
    "Once upon a time, in a small village,",
    "The derivative of sin(x) is",
    "Water boils at a temperature of",
    "The quick brown fox",
]


def main():
    tok = Tokenizer.from_file(sys.argv[1])
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "prompts.tsv"), "w") as f:
        for i, p in enumerate(PROMPTS):
            ids = tok.encode(p).ids
            f.write(f"{i}\t{' '.join(map(str, ids))}\n")
            print(i, len(ids), repr(p))


if __name__ == "__main__":
    main()
