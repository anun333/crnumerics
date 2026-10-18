#!/usr/bin/env python3
"""Cut a public-domain text into 200 sequences of 128 GPT-2 tokens (the
context the SAEs were trained on), so every machine reads the same token ids:
writes tokens.txt next to this file, one sequence a line.

usage: prep.py TOKENIZER.json pg1342.txt
pg1342.txt is Project Gutenberg's Pride and Prejudice
(https://www.gutenberg.org/cache/epub/1342/pg1342.txt); the text from
"Chapter I." to the end marker is used.
"""
import os
import sys

from tokenizers import Tokenizer

NSEQ, CTX = 200, 128


def main():
    tok = Tokenizer.from_file(sys.argv[1])
    text = open(sys.argv[2], encoding="utf-8").read()
    start = text.index("Chapter I.]")
    end = text.index("*** END OF THE PROJECT GUTENBERG EBOOK")
    ids = tok.encode(text[start:end]).ids
    assert len(ids) >= NSEQ * CTX, len(ids)
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "tokens.txt"), "w") as f:
        for s in range(NSEQ):
            f.write(" ".join(map(str, ids[s * CTX:(s + 1) * CTX])) + "\n")
    print(f"{NSEQ} sequences of {CTX} tokens, from {len(ids)}")


if __name__ == "__main__":
    main()
