# A language model, layer by layer, across machines

Does a small language model compute the same bits on every machine? And
where it doesn't, which layer is the first to differ? `layerhash.c` runs
[SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) (a
Llama-shaped model: 30 layers, grouped-query attention, RoPE, SiLU, RMS
norm) in plain C, one token at a time with a cache, and writes a SHA-256 of
the residual stream after the embedding and after every layer, for every
token. Two runs agree bit for bit if and only if their hashes do, and the
first hash that differs names the place where they part.

One file, four builds, and the same model on an NVIDIA GPU ([gpu/](gpu)):

| build | dot products | softmax, RMS norm, SiLU | RoPE's angles | flags |
|---|---|---|---|---|
| `cr` | correctly rounded ([crsum](../../crsum)) | [crnn](../../nn): correctly rounded, or a fixed sequence of correctly rounded steps on exact sums | CORE-MATH's double `pow`, `sin`, `cos` | `-O2 -ffp-contract=off -frounding-math` |
| `tree` | a fixed order: element i to partial sum i mod 16, then the sixteen combined pairwise | crnn | CORE-MATH | `-O3 -march=native -ffp-contract=off` |
| `treefma` | the same order, each step of a partial sum one `fmaf` | crnn | CORE-MATH | `-O3 -march=native -ffp-contract=off` |
| `stock` | a running sum | the system's `expf`, `1/sqrtf` | the system's `powf`, `sinf`, `cosf` | `-O3 -ffast-math -march=native` |

`stock` is how plain C inference is usually built. The other three fix the
order of every operation and take the functions from correctly rounded
code, so nothing in them is left to the machine. An `fma` is as portable as
a product and a sum: IEEE 754 rounds it once, so what a compiler must not
do is choose whether to fuse (`-ffp-contract=off`), not use one where the
source says so. The weights are the model's bfloat16 values widened to
binary32, which is exact.

The GPU builds, `gpu/layerhash-gpu-tree` and `-treefma`, follow the same
order: 16 threads a row, thread j summing the elements j mod 16 in turn
with `__fmul_rn` and `__fadd_rn` (or `__fmaf_rn`), then the same pairwise
combination; crnn's RMS norm, softmax and SiLU are crnumerics' own C,
compiled as device code (`gpu/devify-tree.py` adds `__device__` and
nothing else); `nvcc -fmad=false`.

## Running it

```
python3 prompts.py tokenizer.json          # writes prompts.tsv (already here)
make                                       # layerhash-cr, -tree, -treefma, -stock
./run.sh model.safetensors 48 OUT          # each build, 48 tokens a prompt
python3 compare.py A=OUT_A B=OUT_B ...     # which machines agree, and where they part
python3 ref.py MODEL_DIR prompts.tsv 48 OUT    # cr against Hugging Face transformers
gpu/build.sh                               # with CUDA: gpu/layerhash-gpu-tree, -treefma
```

`model.safetensors` and `tokenizer.json` are from the model's repository at
revision `93efa2f097d58c2a74874c7e644dbc9b0cee75a2` (sha256 of the weights
`80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1`; the
program prints the hash it read). Eight prompts, each run and then
continued greedily for 48 tokens: 430 token steps and 13,330 layer hashes a
build. A CPU build runs in one process, on one core.

**Is the arithmetic right?** `ref.py` runs the same prompts through
transformers 4.57 in binary32: the same 48 tokens chosen for every prompt;
the logits after each prompt within 6e-5 of transformers' (logits of 20 to
30); the residual stream equal after the embedding and, by the last layer,
within 0.004 of values up to 3,000 (transformers sums in another order and
computes RoPE in binary32). The residual stream holds a few very large
values, up to 19,000 by layer 20, as in many such models.

## Results, 2026-10-18

Six CPUs (three instruction sets, two operating systems, three compilers)
and one GPU. The chain hash is a SHA-256 over all 13,330 layer hashes in
order; equal letters are equal chains.

| machine | processor | system, compiler | `cr` | `tree` | `treefma` | `stock` |
|---|---|---|---|---|---|---|
| a laptop | AMD Ryzen 5 PRO 5650U (Zen 3) | x86-64 Linux, GCC 13.3 | A | B | C | D |
| cfarm421 | AMD EPYC 7773X (Zen 3) | x86-64 Linux, GCC 14.2 | A | B | C | D |
| cfarm424 | Neoverse-N1 | AArch64 Linux, GCC 14.2 | A | B | C | E |
| cfarm107 | Cortex-X925 | AArch64 Linux, GCC 13.3 | A | B | C | F |
| cfarm104 | Apple M1 | arm64 macOS 26, Apple clang 21 | A | B | C | G |
| cfarm400 | Loongson 3C5000L | LoongArch64 Linux, GCC 14.2 | A | B | C | H |
| cfarm107 | NVIDIA GB10 (GPU) | CUDA 13, nvcc `-fmad=false` | | B | C | |

A = `cc6e6480…`, B = `efdeee68…`, C = `56b1f54d…`; D to H, five different
`stock` chains. The GCC Compile Farm's RISC-V board was left out of this
run for time.

**`cr`, `tree`, `treefma`: one hash each, on every machine, and the GPU
too.** Every one of the 13,330 layer hashes, the logits and the 384
generated tokens are the same on the six CPUs; for `tree` and `treefma`,
on the GPU as well. Each `stock` chain, run again on its own machine, was
the same as before.

**`stock`: five different results on six machines.** The two x86 machines
agree (both Zen 3, both `-march=native`, though built by GCC 13 and 14);
every other machine is its own class, the two Arm Linux machines included
(Neoverse-N1 and Cortex-X925). Each of the others parts from the x86 pair
at the first transformer layer of the first token: only the embedding
lookups (430 of the 13,330 hashes) agree. The logits after the prompts
differ from the x86 pair's by up to 1.1e-4, which here never changed a
greedy choice: every build chose the same 48 tokens for every prompt on
every machine. A difference that size can change a choice where two tokens
are nearly tied, and every token after it; eight short prompts met no such
tie.

**What it costs.** One core, each build alone, seconds for the 430 steps:

| machine | `stock` | `tree` | `treefma` | `tree` / `stock` |
|---|---:|---:|---:|---:|
| Ryzen 5 PRO 5650U | 13.5 | 14.3 | 14.0 | 1.06 |
| EPYC 7773X | 18.9 | 20.4 | 21.5 | 1.08 |
| Neoverse-N1 | 21.8 | 23.9 | 24.6 | 1.10 |
| Cortex-X925 | 7.6 | 7.7 | 7.6 | 1.01 |
| Apple M1 | 5.2 | 5.9 | 5.8 | 1.13 |
| Loongson 3C5000L | 31.4 | 26.8 | 29.3 | 0.85 |

The fixed order cost at most 13% here, and on the Loongson it was faster.
A token step reads every weight once (540 MB here), so these builds spend
much of their time waiting on memory, and sixteen partial sums in a fixed
order vectorize as well as the reordering `-ffast-math` allows. The first
version of the loop indexed `i % 16`, which the compilers did not
vectorize, and cost 2 to 5 times `stock`; `-frounding-math` on the file
(kept on the library sources) also stopped GCC vectorizing `fmaf`. Correct
rounding of every dot product (`cr`) cost 14 to 45 times `stock`,
unoptimized. The GPU builds take 24 s: they copy the residual stream back
after every layer and compute each RMS norm and softmax in one thread,
which is for checking, not speed.

## What it does not show

One small model, eight prompts, greedy decoding, one GPU. `stock` is one
plain C loop, not an optimized inference engine; llama.cpp and others make
their own choices. The point is narrower: with the order fixed and the
functions correctly rounded, the result stopped depending on the machine,
CPU or GPU, at little cost, and the first layer where an unfixed build
parts can be read off.
