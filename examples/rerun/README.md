# rerun: text from a language model, with a receipt anyone can check

`rerun` generates text with a language model and writes a **receipt**: the
model files by sha256, the settings, the prompt's and the output's token
ids, and a SHA-256 of the residual stream after every layer at every
position. `rerun verify` computes everything again, on whatever machine it
runs on, and says `VERIFIED`, or names the first position and layer that
differ.

That works because the model runs on [layerhash](../layerhash)'s
fixed-order contract, which gives the same bits on every machine we have
tried (six CPUs of three instruction sets, and a GPU): every dot product in
one order, no fusion the compiler chooses, RMS norm, softmax and SiLU from
crnn, RoPE's angles from CORE-MATH. Sampling takes its uniforms from
crdist's counter-based stream (Philox4x32-10) keyed by the seed, and adds up
probabilities in a fixed order, so sampled text repeats too. Threads split a
matrix's rows, so their number changes only the time.

```
make                                                   # rerun
python3 rerun.py gen MODEL_DIR "Once upon a time" --max 64 --temp 0.8 --top-p 0.95 --seed 1 --out r.json
python3 rerun.py verify r.json MODEL_DIR               # VERIFIED, or where it parts
./rerun gen MODEL_DIR PROMPT_IDS 64 r.json [--temp T --top-p P --seed S --threads N]   # ids in, ids out
```

Models: Llama-family (`LlamaForCausalLM`, without `rope_scaling`: SmolLM2
and others) and OLMo 2 (`Olmo2ForCausalLM`), from a directory with
`config.json`, the safetensors file or shards (bfloat16, float16 or
float32), and `tokenizer.json` for `rerun.py`. Tokenizing is integer work:
the receipt covers the token ids, and the text `rerun.py` adds to it is for
reading.

**Checks.** On SmolLM2-135M, rerun's 1,612 layer hashes for layerhash's
first prompt (52 positions, 31 each) equal layerhash's `tree` build's, with
4 threads. A receipt verifies; the same receipt re-indented by another JSON
writer verifies; a receipt with one layer hash changed is reported at that
position and layer; one with a changed output token is refused.

## Results, 2026-10-18: OLMo 2 1B, made on x86, verified on Arm

[OLMo 2 1B](https://huggingface.co/allenai/OLMo-2-0425-1B) (Ai2's fully
open model: weights, data and code; revision `a1847dff`, float32, two
shards). On an AMD EPYC 7773X (x86-64, GCC 14.2, 16 threads), against
Hugging Face transformers in binary32: the same 48 greedy tokens, the logits
after the prompt within 3.4e-5 (logits up to 22.6). Two receipts from it, for
the prompt *"Language models can be checked by running them again on another
machine, because"*:

- greedy, 48 tokens: *"they are not stored in the cloud. This is a huge
  advantage over the cloud, where the data is stored in the cloud..."*;
- sampled (temperature 0.8, top-p 0.95, seed 42), 64 tokens: *"the model
  uses the same architecture. If the model is not working, then the problem
  will be obvious. A neural network model can be changed by altering the
  layers or the weights, and retraining..."*.

Both `VERIFIED` on two AArch64 machines, a Neoverse-N1 (GCC 14.2) and a
Cortex-X925 (GCC 13.3), with 8 threads each: every one of the 1,037 and
1,309 layer hashes (61 and 77 positions, 17 each), every token, and the
chain hashes `2eabf0b2...` and `bfb61af5...`. A token step took about
0.22 s with 16 threads on the EPYC (shared with other users' jobs) and
0.064 s with 8 on the Cortex-X925; nothing here is tuned.

What the model says about itself is its own; the point is that anyone can
check that it said it.

## Probes: readouts of the model's state in the receipt

`rerun gen ... --probe PROBE.json` adds a linear probe's reading of the
residual stream after one layer, at every position, to the receipt (the
probe file by sha256), in the contract's order; `rerun verify ... --probe
PROBE.json` checks every value bit for bit. `--steer FILE ALPHA` adds ALPHA
times the unit vector along a probe's direction to the stream after its
layer (recorded, and replayed by verify). `rerun dump` runs sequences
teacher-forced and writes the residual stream after chosen layers and the
entropy of the model's own next-token distribution at every position, for
training probes (`probe_train.py`) and testing them (`steer_test.py`).

**A probe of the model's coming uncertainty** (SmolLM2-135M, 2026-10-18).
Labels: the entropy of the model's next-token distribution, computed on the
contract (crnn's softmax, CORE-MATH's log, a fixed-order sum), so they are
the same bits everywhere. 200 sequences of 128 tokens of *Pride and
Prejudice*; a ridge-regression probe on the residual stream after layers 8,
15 and 22 of 30, trained on the first 160 sequences and scored on the other
40:

| after layer | variance explained (held out) | correlation (held out) |
|---|---|---|
| 8 | 0.47 | 0.69 |
| 15 | 0.62 | 0.79 |
| 22 | 0.76 | 0.87 |

**Is it reading something the model uses?** Steering along the layer-15
probe's direction on 20 held-out sequences (2,560 tokens), by multiples of
the residual stream's median size there (146.7), against 8 random
directions:

| change in mean entropy (nats; 3.25 unsteered) | −0.4× | −0.2× | +0.2× | +0.4× |
|---|---|---|---|---|
| probe's direction | −0.52 | −0.29 | +0.39 | +0.79 |
| random directions, range | −0.03 to +0.51 | −0.02 to +0.27 | −0.14 to +0.14 | −0.17 to +0.39 |

Any push raises the entropy a little (the even part, (+0.4× + −0.4×)/2:
0.14 for the probe, 0.08 to 0.25 for the random directions). The part that
changes with the push's sign, (+0.4× − −0.4×)/2, is +0.66 for the probe and
0.05 to 0.34 in size (mean 0.13, either sign) for the random directions: the probe's direction
moves the model's uncertainty the way the probe predicts, more than any of
eight random directions does. Its readout overstates the effect: at +0.2×,
the probe itself would predict +9.3 nats; the model moves +0.39. With exact
replay, every one of these differences is the steering's alone.

**In a receipt.** For *"The year the first person walked on the moon
was"*, SmolLM2 answers *" 1969."*; the layer-15 probe's reading falls from
9.8 at the first token to 0.9 at the last prompt token, before the answer.
The receipt verifies with the probe file, and on another machine
(Neoverse-N1, made on a Ryzen); it is refused without the probe file, with
another probe file, or with one value changed by 0.001.
