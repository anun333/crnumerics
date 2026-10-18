# A language model, layer by layer, across machines

Does a small language model compute the same bits on every machine? And
where it doesn't, which layer is the first to differ? `layerhash.c` runs
[SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) (a
Llama-shaped model: 30 layers, grouped-query attention, RoPE, SiLU, RMS
norm) in plain C, one token at a time with a cache, and writes a SHA-256 of
the residual stream after the embedding and after every layer, for every
token. Two runs agree bit for bit if and only if their hashes do, and the
first hash that differs names the place where they part.

One file, three builds:

| build | dot products | softmax, RMS norm, SiLU | RoPE's angles | flags |
|---|---|---|---|---|
| `cr` | correctly rounded ([crsum](../../crsum)) | [crnn](../../nn): correctly rounded, or a fixed sequence of correctly rounded steps on exact sums | CORE-MATH's double `pow`, `sin`, `cos` | `-O2 -ffp-contract=off` |
| `tree` | a fixed order: 16 interleaved partial sums, combined pairwise | crnn | CORE-MATH | `-O2 -ffp-contract=off` |
| `stock` | a running sum | the system's `expf`, `1/sqrtf` | the system's `powf`, `sinf`, `cosf` | `-O3 -ffast-math -march=native` |

`stock` is how plain C inference is usually built. `cr` and `tree` fix the
order of every operation and take the functions from correctly rounded
code, so nothing in them is left to the machine. The weights are the
model's bfloat16 values widened to binary32, which is exact.

## Running it

```
python3 prompts.py tokenizer.json          # writes prompts.tsv (already here)
make                                       # layerhash-cr, -tree, -stock
./run.sh model.safetensors 48 OUT          # the three builds, 48 tokens a prompt
python3 compare.py A=OUT_A B=OUT_B ...     # which machines agree, and where they part
python3 ref.py MODEL_DIR prompts.tsv 48 OUT    # cr against Hugging Face transformers
```

`model.safetensors` and `tokenizer.json` are from the model's repository at
revision `93efa2f097d58c2a74874c7e644dbc9b0cee75a2` (sha256 of the weights
`80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1`; the
program prints the hash it read). Eight prompts, each run and then
continued greedily for 48 tokens: 430 token steps and 13,330 layer hashes a
build. A build that runs in one process, on one core.

**Is the arithmetic right?** `ref.py` runs the same prompts through
transformers 4.57 in binary32: the same 48 tokens chosen for every prompt;
the logits after each prompt within 6e-5 of transformers' (logits of 20 to
30); the residual stream equal after the embedding and, by the last layer,
within 0.004 of values up to 3,000 (transformers sums in another order and
computes RoPE in binary32). The residual stream holds a few very large
values, up to 19,000 by layer 20, as in many such models.

## Results, 2026-10-18

Six machines: three instruction sets, two operating systems, three
compilers. The chain hash is a SHA-256 over all 13,330 layer hashes in
order; equal letters are equal chains.

| machine | CPU | system, compiler | `cr` | `tree` | `stock` |
|---|---|---|---|---|---|
| a laptop | AMD Ryzen 5 PRO 5650U (Zen 3) | x86-64 Linux, GCC 13.3 | A | B | C |
| cfarm421 | AMD EPYC 7773X (Zen 3) | x86-64 Linux, GCC 14.2 | A | B | C |
| cfarm424 | Neoverse-N1 | AArch64 Linux, GCC 14.2 | A | B | D |
| cfarm107 | Cortex-X925 | AArch64 Linux, GCC 13.3 | A | B | E |
| cfarm104 | Apple M1 | arm64 macOS 26, Apple clang 21 | A | B | F |
| cfarm400 | Loongson 3C5000L | LoongArch64 Linux, GCC 14.2 | A | B | G |

A = `cc6e6480…`, B = `efdeee68…`; C to G, five different `stock` chains.
The GCC Compile Farm's RISC-V board was left out of this run for time.

**`cr` and `tree`: one hash on every machine.** Every one of the 13,330
layer hashes, the logits and the tokens are the same on all six machines.
The fixed-order build is still the same when compiled with `-O3
-march=native` (on the Ryzen: same hash, 42 s instead of 52 s).

**`stock`: five different results on six machines.** The two x86 machines
agree (both Zen 3, both `-march=native`, though built by GCC 13 and 14);
every other machine is its own class, the two Arm Linux machines
included (Neoverse-N1 and Cortex-X925). Each of the others parts from the
x86 pair at the first transformer layer of the first token: only the
embedding lookups (430 of the 13,330 hashes) agree. The logits after the
prompts differ from the x86 pair's by up to 1.1e-4, which here never changed
a greedy choice: every build chose the same 48 tokens for every prompt on
every machine. A difference that size can change a choice where two tokens
are nearly tied, and every token after it; eight short prompts met no such
tie.

**What it costs.** Single core, seconds for the 430 steps:

| machine | `cr` | `tree` | `stock` |
|---|---:|---:|---:|
| Ryzen 5 PRO 5650U | 340 | 52 | 15.5 |
| EPYC 7773X | 332 | 41 | 15.7 |
| Neoverse-N1 | 388 | 61 | 27.9 |
| Cortex-X925 | 153 | 18.5 | 8.5 |
| Apple M1 | 238 | 27 | 5.3 |
| Loongson 3C5000L | 675 | 111 | 30.5 |

The fixed order costs 2.2 to 5.2 times `stock` here, and correct rounding
of every dot product 14 to 45 times; neither has been optimized
(the tree's inner loop indexes `i % 16`).

## What it does not show

One small model, eight prompts, greedy decoding, CPUs only. `stock` is one
plain C loop, not an optimized inference engine; llama.cpp and others make
their own choices. The point is narrower: with the order fixed and the
functions correctly rounded, the result stopped depending on the machine,
and the first layer where an unfixed build parts can be read off.
