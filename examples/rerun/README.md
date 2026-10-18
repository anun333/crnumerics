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
and others), OLMo 2 (`Olmo2ForCausalLM`) and BitNet b1.58
(`BitNetForCausalLM`), from a directory with
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

## Does the model know whether it knows? (OLMo 2 1B, PopQA)

`rerun batch` answers many prompts with one model load: greedy, stopping at
given tokens, keeping the residual stream after chosen layers at each
prompt's last token and the entropy of the first answer token. A prefix all
prompts share is computed once (the cache holds the same bits a fresh run
would: answers, entropies and residuals byte-identical with `--no-share`);
`--samples K` adds K seeded samples per prompt.

2,000 questions from PopQA (Mallen et al. 2023: entity questions from
Wikidata with the subject's popularity; `popqa_prep.py`, three fixed
examples before each), answered by OLMo 2 1B on cfarm421 (53 minutes, 16
threads): 18.0% correct (an accepted answer in the reply's first line).
`popqa_probe.py` fits a logistic probe on the residual stream at the
question's last token, before any answer, to predict whether the answer
will be right, and scores it on 500 held-out questions:

| signal | AUROC |
|---|---|
| the subject's popularity | 0.64 |
| the question's type alone (capitals 73% right, film producers 4%) | 0.81 |
| the model's own confidence (minus the first answer token's entropy) | 0.88 |
| probe after layer 4 / 8 / 12 / 16 of 16 | 0.83 / 0.88 / 0.89 / 0.90 |

Halfway through the network, before it has answered, the model's state says
whether it is about to be right about as well as its final confidence does.
But not better: much of either signal is the question's type, and within a
type the model's own confidence beats the layer-12 probe in 6 of 11 types,
the probe it in 3, with 2 ties (3 to 18 right answers per type in the held-out
set, so these are noisy). A linear
probe here gives an earlier reading of what the model's output will show,
not knowledge the output hides.

## BitNet b1.58 2B: the integer route

[BitNet b1.58 2B4T](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T)
(Microsoft, MIT; revision `04c3b9ad`) has ternary weights (−1, 0, 1, four
to a byte, one scale a matrix) and quantizes each token's activations to 8
bits before every linear layer (scale 127 / max |x|, rounded to nearest,
ties to even), with a sub-layer RMS norm before `o_proj` and `down_proj` and
a squared ReLU. rerun computes each of those layers as an exact int32 sum of
int8 times ternary values, so their order does not matter, then scales it
once; the rest (norms, attention, the tied bfloat16 embedding) is the
contract's binary32. A token step on the 2B model took about 0.2 s with 4
threads on a Ryzen laptop, 0.37 s with 16 on a shared EPYC.

**Against transformers** (binary32, on cfarm421): the same 48 greedy tokens;
the logits after the prompt up to 0.48 apart (logits up to 19.6). Layer by
layer: after layer 1 the residual stream agrees to 5e-7 of its range, after
layer 2 the largest difference jumps to 2.1 while the median stays at 3e-5,
and by layer 15 the largest is 4e-3 of the range. That is the quantization
at work: a tiny difference upstream moves an activation across a rounding
boundary, and the layer's output moves by a whole step. Integer matrix
products alone do not make BitNet portable; its activation rounding
amplifies any difference in the binary32 parts before it. With those parts
fixed too, it is:

**Made on x86, verified on Arm.** For the prompt *"Language models can be
checked by running them again on another machine, because"*, a greedy
receipt (48 tokens: *"they are not dependent on the hardware of the machine
they were trained on..."*) and a sampled one (temperature 0.8, top-p 0.95,
seed 42, 64 tokens: *"they are transitive. Now, let's consider a scenario
where we have two language models, A and B..."*), made on the EPYC, both
`VERIFIED` on a Neoverse-N1 and a Cortex-X925, every layer at every
position (62 and 78 positions, 31 hashes each).

## BitNet in integers only (`--int`, contract `int/v0`)

`rerun gen ... --int` runs BitNet b1.58 with no floating-point arithmetic on
the data: the residual stream, q, k, v and the attention and MLP outputs are
int64 with 24 fractional bits, every product and sum is exact (int64 and
__int128), and every rounding is written out (to nearest, ties to even). The
model's constants become integers once, at load: each bfloat16 is an integer
times a power of two, read from its bits; the RoPE tables are CORE-MATH's
correctly rounded cos and sin, scaled by 2^30 and rounded. The one function,
the softmax's exp, is crfix's (integer-only, correctly rounded). So it
needs no IEEE 754 at all, and integer arithmetic is the same on every
machine. Greedy only, so far.

The norms cost nothing in this form: BitNet quantizes every norm's output to
8 bits by dividing by its largest value, so the norm's 1/rms cancels from the
codes, which are round(127 x_i g_i / max |x g|), an exact rational rounding;
the layer's output carries r max|x g| / 127, with r = 1/sqrt(mean x^2 + 1e-5)
from an exact integer square root.

**Against the floating-point contract** (cfarm421, 16 prompts, 48 greedy
tokens each): 7 prompts give the same 48 tokens, and the other 9 part at
steps 10 to 38, each at a near-tie into an equally fluent continuation
(*"composed mainly of hydrogen and helium"* against *"made mostly of
hydrogen and helium"*): two fixed specifications of one model. The integer
path took 19.5 s for 48 tokens against 17.7 s (16 threads).

**Made on x86, verified on Arm and macOS.** The 16 integer receipts (and two
float ones), made on an AMD EPYC (Linux, GCC 14), all `VERIFIED` on a
Neoverse-N1 and a Cortex-X925 (Linux, GCC 14 and 13) and an Apple M1 (macOS,
Apple clang 21): every layer's int64 residual stream at every position, and
every token.
