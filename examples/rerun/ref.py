#!/usr/bin/env python3
"""Check rerun against Hugging Face transformers in binary32: the same prompt
ids, greedy; whether the generated tokens agree and how far apart the logits
after the prompt are.

usage: ref.py MODEL_DIR RECEIPT.json LOGITS.f32   (from rerun gen ... --logits LOGITS.f32, greedy)
"""
import json
import os
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM


def main():
    mdir, rec, lg = sys.argv[1:4]
    r = json.load(open(rec))
    torch.set_num_threads(int(os.environ.get("THREADS", "4")))
    try:
        model = AutoModelForCausalLM.from_pretrained(mdir, dtype=torch.float32).eval()
    except (ValueError, KeyError):  # a multimodal checkpoint (Qwen3.5): its text through the image-text class
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(mdir, dtype=torch.float32).eval()
    ids = r["prompt_ids"]
    with torch.no_grad():
        out = model(torch.tensor([ids]), use_cache=True)
        logits = out.logits[0, -1].numpy()
        past, gen, nxt = out.past_key_values, [], int(np.argmax(logits))
        for g in range(len(r["output_ids"])):
            gen.append(nxt)
            if g + 1 == len(r["output_ids"]):
                break
            o2 = model(torch.tensor([[nxt]]), past_key_values=past, use_cache=True)
            past, nxt = o2.past_key_values, int(torch.argmax(o2.logits[0, -1]))
    ours = np.fromfile(lg, dtype="<f4")
    d = float(np.max(np.abs(ours - logits)))
    first = next((i for i, (a, b) in enumerate(zip(gen, r["output_ids"])) if a != b), None)
    print(f"logits after the prompt: max |diff| {d:.3g} (max |logit| {np.max(np.abs(logits)):.3g}); "
          f"tokens: {'the same' if first is None else f'differ from step {first}'} ({len(gen)} steps)")
    print(f"VERDICT: {'AGREE' if first is None and d < 1e-2 else 'CHECK'}")


if __name__ == "__main__":
    main()
