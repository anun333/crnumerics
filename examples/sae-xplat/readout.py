#!/usr/bin/env python3
"""Interpretability readouts of GPT-2 small, the way they are usually
computed: the residual stream from Hugging Face transformers on PyTorch's
CPU backend, in binary32, and the features of Joseph Bloom's residual-stream
sparse autoencoders at layers 2, 6 and 10 (ReLU((x - b_dec) W_enc + b_enc)),
on the 200 sequences of 128 tokens in tokens.txt. The residual stream is
centred first (its mean over the 768 dimensions subtracted), as
TransformerLens, which the autoencoders were trained through, centres it.

usage: THREADS=1 BATCH=1 readout.py DIR OUT.npz
DIR is fetch.sh's. OUT gets, per layer, every token's active features (index
and value), its top 10 features, and a SHA-256 of the residual stream; and
layer 6's residual stream for the first ten sequences. compare.py compares
two or more of these files.
"""
import hashlib
import os
import platform
import sys
import time

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import GPT2Model

LAYERS = (2, 6, 10)


def main():
    d, out = sys.argv[1], sys.argv[2]
    threads = int(os.environ.get("THREADS", "1"))
    batch = int(os.environ.get("BATCH", "1"))
    torch.set_num_threads(threads)
    here = os.path.dirname(os.path.abspath(__file__))
    toks = torch.tensor([[int(t) for t in line.split()] for line in open(os.path.join(here, "tokens.txt"))])
    model = GPT2Model.from_pretrained(os.path.join(d, "gpt2"), torch_dtype=torch.float32).eval()
    saes = {L: load_file(os.path.join(d, "sae", f"l{L}.safetensors")) for L in LAYERS}
    rec = {L: {"cnt": [], "idx": [], "val": [], "top": []} for L in LAYERS}
    rsha = {L: hashlib.sha256() for L in LAYERS}
    keep = []
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, toks.shape[0], batch):
            hs = model(toks[i:i + batch], output_hidden_states=True).hidden_states
            for L in LAYERS:
                x = hs[L]  # the input to block L: TransformerLens's hook_resid_pre
                x = x - x.mean(-1, keepdim=True)
                rsha[L].update(x.contiguous().numpy().tobytes())
                if L == 6 and i < 10:
                    keep.append(x.numpy().copy())
                s = saes[L]
                f = torch.relu((x - s["b_dec"]) @ s["W_enc"] + s["b_enc"]).reshape(-1, s["W_enc"].shape[1]).numpy()
                for row in f:
                    nz = np.flatnonzero(row)
                    v = row[nz]
                    rec[L]["cnt"].append(len(nz))
                    rec[L]["idx"].append(nz.astype(np.int32))
                    rec[L]["val"].append(v.astype(np.float32))
                    order = np.lexsort((nz, -v))[:10]  # by value, ties by index
                    t = np.full(10, -1, np.int32)
                    t[:len(order)] = nz[order]
                    rec[L]["top"].append(t)
    secs = time.time() - t0
    arrays = {"resid6": np.concatenate(keep)[:10] if keep else np.zeros(0)}
    for L in LAYERS:
        arrays[f"cnt{L}"] = np.array(rec[L]["cnt"], np.int32)
        arrays[f"idx{L}"] = np.concatenate(rec[L]["idx"])
        arrays[f"val{L}"] = np.concatenate(rec[L]["val"])
        arrays[f"top{L}"] = np.stack(rec[L]["top"])
        arrays[f"rsha{L}"] = np.array(rsha[L].hexdigest())
    meta = {"threads": threads, "batch": batch, "seconds": round(secs, 1), "torch": torch.__version__,
            "numpy": np.__version__, "python": platform.python_version(), "machine": platform.machine(),
            "system": platform.system(), "processor": platform.processor(), "node": platform.node(),
            "torch_config": torch.__config__.show()}
    arrays["meta"] = np.array(repr(meta))
    np.savez_compressed(out, **arrays)
    print(f"{out}: {secs:.1f} s, threads {threads}, batch {batch}; "
          + ", ".join(f"layer {L} resid {arrays[f'rsha{L}'].item()[:12]} L0 {arrays[f'cnt{L}'].mean():.1f}" for L in LAYERS))


if __name__ == "__main__":
    main()
