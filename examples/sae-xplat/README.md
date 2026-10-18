# Interpretability readouts across machines

Interpretability work reads a model's internals: the residual stream, and
the features a sparse autoencoder (SAE) finds in it. A claim like "feature
1234 fires on token 17" is made on one machine. Is it the same on another?

`readout.py` computes the readouts the usual way: GPT-2 small through
Hugging Face transformers on PyTorch's CPU backend in binary32, and the
features of Joseph Bloom's residual-stream SAEs at layers 2, 6 and 10
(24,576 features each, ReLU((x − b_dec) W_enc + b_enc), the residual stream
centred as TransformerLens centres it), on 200 sequences of 128 tokens of
*Pride and Prejudice* (`tokens.txt`, by `prep.py`): 25,600 tokens.
`gpt2sae.c` computes the same thing with nothing left to the machine: every
dot product in [layerhash](../layerhash)'s fixed order (`gpt2sae-tree`, or
`gpt2sae-treefma` with each step one fma), LayerNorm and softmax from crnn,
GELU's tanh form from CORE-MATH's correctly rounded tanh, the centring mean
from crsum. `compare.py` compares any of these runs.

```
./fetch.sh DIR                              # GPT-2 small, six SAEs (pinned, sha256-checked)
THREADS=1 BATCH=1 python3 readout.py DIR OUT.npz            # SAES=oai: the TopK ones; DEVICE=cuda
make && ./gpt2sae-tree DIR OUTDIR 4 && python3 pack.py OUTDIR OUT.npz   # ... 4 oai: the TopK ones
python3 compare.py A=OUT_A.npz B=OUT_B.npz ...
```

## Results, 2026-10-18

Five machines: two x86-64 Linux (AMD Ryzen 5 PRO 5650U, AMD EPYC 7773X,
both Zen 3), two AArch64 Linux (Neoverse-N1, Cortex-X925) and an Apple M1;
PyTorch 2.10 and 2.14.1 (CPU wheels) and 2.8.0 (macOS). `readout.py` ran
with 1 thread and batch 1, 1 thread and batch 16, and 4 threads and batch
16 on each.

**PyTorch: four different results on five machines.** The residual stream
and the features are the same bits on the two x86 machines (PyTorch 2.10
and 2.14.1), and different on each of the other three, at every layer,
the two AArch64 machines included (same PyTorch, different CPUs). On any
one machine, thread count and batch size changed nothing. Against the x86
pair:

| | layer 2 | layer 6 | layer 10 |
|---|---|---|---|
| active (token, feature) pairs | 832,118 | 1,329,260 | 1,485,303 |
| pairs that switch on or off, Neoverse-N1 / Cortex-X925 / M1 | 0 / 2 / 3 | 1 / 1 / 3 | 0 / 0 / 4 |
| the largest activation among them | 4.2e-6 | 3.5e-6 | 4.9e-6 |
| features active on both: largest difference | 2.5e-4 | 9.5e-4 | 1.1e-3 |
| tokens whose top feature changed | 0 | 0 | 0 |
| tokens whose top 10 changed order | 1 | 1 to 3 | 3 |

The residual stream at layer 6 differs by up to 3.4e-3 (M1) on values up to
2,900.

**The fixed-order build: one result on five machines.** `gpt2sae-tree`, and
`gpt2sae-treefma` too, give the same residual stream at the three layers and
the same features (every index, every value, every top 10) on all five, with
4 threads each. The fixed-order build differs from PyTorch's by as much as
PyTorch's builds differ from each other (on the Ryzen: 1 to 4 pairs switch
on or off per layer, all below 1e-5; residual stream within 7.3e-4): it is
GPT-2, computed in one specified order.

**What it means.** The differences are small: no token's strongest feature
moved, and only features at the edge of firing, with activations under
1e-5, switched on or off. An analysis that thresholds features or ranks
them can still change: a few tokens' top 10 reordered, and a feature near
zero can appear on one machine and not another. More to the point, a
readout from one machine cannot be checked bit for bit on another: what
another lab, or the model's own audit, reproduces is "close", never "the
same". With the order fixed, it is the same, and any difference is a
difference in the model or the input, not in the machine.

`compare.py` was checked against a planted change (one feature removed from
one token): it reports exactly that one.

## Round two: TopK autoencoders, and a GPU

**TopK.** OpenAI's TopK autoencoders for GPT-2 small (32,768 features, the
32 largest kept; on the residual stream after layers 2, 6 and 10,
normalized as OpenAI's code normalizes it), the same text, the same five
CPUs. PyTorch again gives four results (the same four classes), and the
fixed-order build again one. As few pairs differ (0 to 4 per layer), but
those that do are not small: where the 32nd and 33rd pre-activations are
nearly tied, one machine keeps one feature and another keeps the other, so
a feature with an activation of 0.53 to 0.90 is present on one machine and
absent on another (at each of the three layers, against the x86 pair). A threshold at
zero lets only near-zero features flip; a top-k cut lets features of any
size flip.

**A GPU.** PyTorch 2.14.1 with CUDA 13 on an NVIDIA GB10 (batch 16 twice,
batch 1 once, for both kinds of autoencoder). Run to run, the same batch
gave the same bits. Batch size did not: batch 1 and batch 16 differ at
every layer, on the same GPU, with 2 to 3 near-zero features switching per
layer for Bloom's autoencoders and, for the TopK ones, a feature of
activation 0.61 at layer 2. The GPU differs from every CPU as the CPUs
differ from each other. On the CPUs, batch size changed nothing.

| what changed | PyTorch, CPU | PyTorch, GPU | fixed order |
|---|---|---|---|
| the machine (5 CPUs) | 4 results | (one GPU) | 1 result |
| threads, batch size, one CPU | same | | same (threads) |
| the batch size, one GPU | | different | |
| the same run again | same | same | same |

## What it does not show

One model, six SAEs, one text, one GPU; the fixed-order build ran on the
CPUs only (layerhash's GPU build shows the same contract on the GPU for
another model). Attribution graphs, which add a pruning threshold, were not
computed.
