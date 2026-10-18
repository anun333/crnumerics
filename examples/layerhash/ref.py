#!/usr/bin/env python3
"""Check layerhash's arithmetic against Hugging Face transformers in binary32:
the same prompts, the same greedy steps. Reports, per prompt, whether the
generated tokens agree and how far the logits after the prompt are apart,
and for prompt 0 how far the residual stream is apart after each layer.
transformers sums in another order and computes RoPE in binary32, so small
differences are expected; different tokens or large ones mean a bug.

usage: ref.py MODEL_DIR PROMPTS.tsv NGEN OUTDIR
MODEL_DIR holds config.json and model.safetensors; OUTDIR is a layerhash run.
"""
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM


def main():
    mdir, ptsv, ngen, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    torch.set_num_threads(4)
    model = AutoModelForCausalLM.from_pretrained(mdir, torch_dtype=torch.float32)
    model.eval()
    ours = {}
    for line in open(f"{out}/tokens.tsv").read().split("\n")[1:]:
        if line:
            p, g, t, _ = line.split("\t")
            ours.setdefault(int(p), []).append(int(t))
    worst = 0.0
    bad = 0
    for line in open(ptsv):
        pid, ids = line.rstrip("\n").split("\t")
        pid, ids = int(pid), [int(x) for x in ids.split()]
        with torch.no_grad():
            o = model(torch.tensor([ids]), output_hidden_states=True, use_cache=True)
            logits = o.logits[0, -1].numpy()
            past = o.past_key_values
            gen = []
            nxt = int(np.argmax(logits))
            for g in range(ngen):
                gen.append(nxt)
                if g + 1 == ngen:
                    break
                o2 = model(torch.tensor([[nxt]]), past_key_values=past, use_cache=True)
                past = o2.past_key_values
                nxt = int(torch.argmax(o2.logits[0, -1]))
        ol = np.fromfile(f"{out}/logits-{pid}.f32", dtype="<f4")
        d = float(np.max(np.abs(ol - logits)))
        worst = max(worst, d)
        same = ours.get(pid) == gen
        first = next((i for i, (a, b) in enumerate(zip(ours.get(pid, []), gen)) if a != b), None)
        bad += not same
        print(f"prompt {pid}: logits max |diff| {d:.3g} (max |logit| {np.max(np.abs(logits)):.3g}); "
              f"tokens {'same' if same else f'differ from step {first}'}")
        if pid == 0:
            hs = o.hidden_states  # embeddings, then after each layer; the last is after the final norm
            r = np.fromfile(f"{out}/resid-0.f32", dtype="<f4").reshape(len(ids), 31, 576)
            for layer in range(0, 30):
                ref = hs[layer][0].numpy()
                dd = float(np.max(np.abs(r[:, layer, :] - ref)))
                scale = float(np.max(np.abs(ref)))
                if layer in (0, 1, 2, 5, 10, 20, 29):
                    print(f"  prompt 0, after layer {layer}: max |diff| {dd:.3g} (max |value| {scale:.3g})")
    print(f"VERDICT: {'AGREE' if bad == 0 and worst < 1e-2 else 'CHECK'} "
          f"(largest logit difference {worst:.3g}; prompts with different tokens: {bad})")


if __name__ == "__main__":
    main()
