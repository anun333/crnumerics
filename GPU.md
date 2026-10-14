# GPUs

What runs on GPUs so far: experiments from 2026-10-01, moved here from
[ROADMAP.md](ROADMAP.md) on 2026-10-14. The GPU code is not in this
repository yet. A script builds this repository's own sources as device
code, by marking them `__device__`. Bringing that code in is the next step.

**The machines:**
- **NVIDIA:** the GCC Compile Farm's cfarm107 to cfarm109 (GB10 and Jetson
  Thor), running CUDA 13 for our account.
- **AMD:** a Radeon Vega integrated GPU (gfx90c) through ROCm 5.7's HIP.

## On the NVIDIA GB10

- **CORE-MATH's `expf` and `logf`,** compiled as CUDA device code with
  nothing changed but `__device__` on each definition, give the CPU's bits
  on all 2^32 inputs. CUDA's own `expf` and `logf` differ from the correctly
  rounded result on 160 million and 73 million inputs.
- **CORE-MATH's binary64 `exp`, `log1p`, `erfc` and `rsqrt`:** the CPU's
  bits on 2^30 inputs each.
- **crnn's five one-argument functions:** the CPU's bits on all 2^32
  inputs of each, from `crnn-fast.h` and the exception table, unchanged but
  for `__device__`.
- **crnn's composites,** with `crsum.c` and `crnn.c` compiled as device
  code and one GPU thread per vector: logsumexp, softmax, layernorm and
  rmsnorm give the CPU's bits on 2,000 vectors (4 million elements, the
  hard cases included).
- **The composites at GPU speed:** for softmax and logsumexp, a block per
  vector takes the exact sum by integer bins in shared memory, then
  crsum's rounding. They give the CPU's bits at 0.25 ns an element, 34
  times one CPU thread; layernorm and rmsnorm too.
- **The Ozaki scheme on tensor cores:** `crgemm_oz8` with its int8 GEMMs
  through cuBLAS gives the CPU's bits, a correctly rounded DGEMM. 1024³
  takes 0.35 s, against 2.0 s on one of the GB10's Arm cores
  (Cortex-X925/A725); the tensor cores' part is 4% of it.
- **The Ozaki scheme end to end on the GPU,** with the slicing, the sums
  and the rounding as kernels calling `crsum.c`'s own code: the CPU's bits,
  a correctly rounded DGEMM at 2.6 times cuBLAS's own DGEMM time at 4096³.
- **crlin with cuBLAS:** `lin/examples/cublas.c` (HISTORY.md,
  2026-10-13).

## On an AMD Radeon Vega (gfx90c)

The same CUDA sources compiled by hipcc, with a header that maps CUDA's
runtime names. CUDA's 32-lane warp operations are emulated in each half of
AMD's 64-lane wavefront. All of these give the CPU's bits:
- CORE-MATH's `expf` and `logf` on all 2^32 inputs;
- its binary64 `exp`, `log1p`, `erfc` and `rsqrt` on 2^30 inputs each;
- crnn's five functions on all 2^32 inputs each;
- crnn's composites.

Not the int8 GEMMs: rocBLAS has no kernels for that GPU, and running its
gfx900 ones there hung the GPU.

## Next

- the GPU code into this repository, built and checked as the CPU code is;
- a discrete AMD GPU, for the int8 GEMMs;
- then Intel.
