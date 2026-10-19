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
and others), Qwen3 (`Qwen3ForCausalLM`), Qwen3.5's language model
(`Qwen3_5ForConditionalGeneration`, its text alone), OLMo 2
(`Olmo2ForCausalLM`) and BitNet b1.58 (`BitNetForCausalLM`), from a
directory with
`config.json`, the safetensors file or shards (bfloat16, float16 or
float32), and `tokenizer.json` for `rerun.py`. Tokenizing is integer work:
the receipt covers the token ids, and the text `rerun.py` adds to it is for
reading.

**Checks.** On SmolLM2-135M, rerun's 1,612 layer hashes for layerhash's
first prompt (52 positions, 31 each) equal layerhash's `tree` build's, with
4 threads. A receipt verifies; the same receipt re-indented by another JSON
writer verifies; a receipt with one layer hash changed is reported at that
position and layer; one with a changed output token is refused.

**Qwen3.** Qwen3-0.6B against transformers in binary32 on a GSM8K chat
prompt: the same 48 greedy tokens, the logits after the prompt within
3.2e-5 (logits up to 23.8); its receipt verifies with other thread counts.

**Qwen3.5.** Three of every four layers are Gated DeltaNet, a linear
attention with a recurrent state: a 4-tap causal convolution, q and k
scaled to unit length, and for each of 16 heads a 128 x 128 state decayed
by exp(g) and updated by the delta rule; every fourth layer is full
attention with an output gate, RoPE on the first 64 of each head's 256
dimensions, and norms that scale by 1 + w. On the contract the state's
sums go in dotf's order (row by row, the same bits), and exp, softplus,
sigmoid and 1/sqrt come from CORE-MATH and crnn. Qwen3.5-0.8B against
transformers 5.20.1 in binary32 on two chat prompts: the same greedy
tokens (10 and 48), the logits after the prompt within 7.2e-5 and 3.3e-5
(logits up to 19.4 and 23.7); the receipts verify with 1 and 3 threads,
and in `eval` each answer is `gen`'s in 1, 2 or 3 lanes (a lane's state
is zeroed when it starts a prompt). `batch` runs every prompt from
position 0 for it (a shared prefix would need the state copied).

## Many prompts side by side: `rerun eval`

```
./rerun eval MODEL_DIR PROMPTS OUT --lanes 32 --max 512 --threads 32      # greedy
./rerun eval ... --samples 5 --temp 0.6 --top-p 0.95 --seed 1            # 5 seeded samples a prompt
./rerun eval ... --mix OTHER_DIR 0.5        # every weight a + 0.5 (a - b), b from OTHER_DIR
./rerun eval ... --repeat 6-9               # layers 6 to 9 run a second time
./rerun eval ... --steer DIR.json ALPHA     # ALPHA times the file's unit direction, after its layer
```

`eval` answers every prompt (a line of token ids) and runs up to `--lanes`
of them side by side: a lane is one answer with its own cache, and each
block of work holds every lane's next token (or a new prompt's tokens), so a
weight row is used for all of them while it is in cache. A token's
arithmetic is its own row's (matrix products), its own lane's (attention) or
elementwise, so lanes change nothing but the time: each answer, and the
chain hash over its layer hashes, is what `gen` gives for that prompt alone
(`OUT.chain.txt` has the hash `gen` would put in its receipt). On SmolLM2,
six prompts in 1, 3 or 6 lanes give `gen`'s tokens and chain hashes, greedy
and sampled; the same for Qwen3-0.6B in 4 lanes.

The point is comparing a model with a modified one on many questions.
Every answer repeats bit for bit, on any machine, so the difference between
two runs on the same questions is the modification's, question by question,
not the noise of the run; what remains uncertain is the sample of questions,
which [`gsm8k_compare.py`](gsm8k_compare.py) measures with McNemar's exact
test. The modifications need no training: weights moved along the
difference between two published checkpoints (`--mix`), layers repeated
(`--repeat`), a direction added to the residual stream (`--steer`, from
[`steer_dir.py`](steer_dir.py)'s contrast of two prompt variants), and
majority votes over seeded samples. [`gsm8k_prep.py`](gsm8k_prep.py) makes
GSM8K prompts with the model's own chat template;
[`gsm8k_score.py`](gsm8k_score.py) scores the answers.

## Reasoning without training? GSM8K on OLMo 2 1B and Qwen3, 2026-10-18

OLMo-2-0425-1B-Instruct, greedy, through its chat template, 512 tokens:
79.0% on 500 questions drawn from GSM8K's train split (dev, for choosing;
OLMo's first RL stage trained on GSM data) and 70.2% on the 1,319 test
questions. Each modification against it on the same questions (McNemar's
exact test; only-base-right / only-modified-right in brackets):

| change | dev | test |
|---|---|---|
| weights + 0.5 (Instruct - DPO), `--mix`, best of four on dev | +1.2 (30/36, p 0.54) | -1.1 (100/86, p 0.34) |
| control: weights - 0.5 (Instruct - DPO) | -2.6 (38/25, p 0.13) | -2.6 (123/89, p 0.023) |
| "Think carefully, and check each step" in the prompt | +2.4 (25/37, p 0.16) | -0.6 (103/95, p 0.62) |
| steering at layer 8 along that prompt's contrast, half / full length | -0.6 / -20.0 (p 1e-18) | |
| layers 6 to 9 run twice, `--repeat 6-9` | -12.8 (88/24, p 9e-10) | |
| majority of 5 samples (temperature 0.6), first 200 dev | +5.0 (4/14, p 0.031) | |

The other three extrapolations (along Instruct - DPO or Instruct - RLVR1)
were within 1.6 points on dev. Nothing that changes the model gained on
the test set; large changes did harm (steered at full length, the model
writes organised working with wrong arithmetic: 840 - 825 = 25). The
control moved as expected on both splits, so the setup resolves a change
of 2.6 points; and the gainless extrapolation still flipped 186 of the
1,319 test answers, about as many each way. The instruction's dev gain was
mostly format: scored on the `Answer:` line alone it would have been +9.6.
The gains came from compute at answer time: the vote above, and Qwen3's
thinking mode on the first 100 dev questions (Qwen3-0.6B 67% non-thinking,
72% thinking with 2,048 tokens, 77% with 4,096, at 6.5 times the tokens;
Qwen3-1.7B 84% and 90%). Every answer is bit-exact across builds, lane
counts and machines: the first 20 dev answers' chain hashes are the same
from two builds, in 1, 20 and 32 lanes, on x86-64 and AArch64.

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
machine. Sampling is in integers too (below).

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

**Sampling in integers.** With `--temp` above 0 the next token is drawn in
integers as well. Every token's score is an exact int128 sum (the final
norm's input, times its weights, against the token's embedding row); from
it, (logit - max logit) / temperature in Q16, with the norm's 1/rms from the
same integer square root and the temperature rounded once to
Tq = temp 2^16; crfix's exp turns that into Q30 weights; top-p keeps the
smallest set of the largest weights (ties to the lower id) whose sum s
satisfies s 2^30 >= round(top_p 2^30) total, compared exactly; and one
64-bit draw u from the seeded Philox stream picks the first token whose
running sum exceeds floor(u s / 2^64). `--logits` writes the integer path's
logits after the prompt (binary32, rounded from the exact scores), for
checking only.

Checked on cfarm421 with the 16 prompts. The temperature means the same as
on the floating-point contract: after 4 prompts the two paths' logits lie on
a line of slope 0.998 to 1.005 (over the top 1000 tokens), and their
distributions at temperature 0.8 are 0.001 to 0.013 apart in total
variation, although single logits differ by up to 1.1 (BitNet's 8-bit
rounding again). At temperature 1e-4 all 16 prompts give the greedy
receipts' 48 tokens. At temperature 0.8 and top-p 0.95 (seed = the prompt's
number) every sample leaves greedy within its first 8 tokens (*"The capital
of France is Paris, known for its iconic landmarks such as the Eiffel Tower
and Notre Dame Cathedral"*), at 20.8 s for 48 tokens against greedy's
19.1 s (16 threads); what sampling adds per token is an exp and a sort
over the vocabulary's 128,256 tokens. The 32 sampled receipts (and 8
one-token receipts from the logits check) all `VERIFIED` on the Neoverse-N1,
the Cortex-X925 and the Apple M1; `verify` draws every token again from the
receipt's seed, so the sampling itself is checked, not only the layers.
