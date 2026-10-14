# crnumerics

Floating-point results that are the same bits on every machine, with
proofs anyone can rerun.

Correctly rounded functions give one right answer, so they agree
everywhere; [crmvec](https://github.com/anun333/crmvec) does that for the
vector functions compilers call. But most differences between machines
come from elsewhere: the order of a sum, fast approximations inside larger
operations, formats with no standard math library, and tools that can't
tell you where a difference comes from. The libraries here close those
gaps one at a time ([ROADMAP.md](ROADMAP.md) has the order):

| | What it gives you | Who it's for |
|---|---|---|
| **crsum** ([`crsum/`](crsum/README.md)) | correctly rounded sums, dot products and matrix products in binary64 and binary32: the same bits in any order, split or thread count; `crgemm_oz`, exact matrix products through any binary64 BLAS | simulation, training and inference, finance: anyone who needs a reduction to come out the same twice |
| **crnn** ([`nn/`](nn/README.md)) | neural-network primitives with one answer: `sigmoid`, `silu`, `gelu`, `softplus` and `rsqrt` correctly rounded in binary32, binary16 and bfloat16, and `logsumexp`, `softmax`, `layernorm` and `rmsnorm` specified bit for bit on top of crsum | inference and training that must give the same output on every machine |
| **crdist** ([`dist/`](dist/README.md)) | the normal distribution correctly rounded in binary32, binary16 and bfloat16, proven on every input: `erfinv`, `erfcinv`, `probit` (`qnorm`/`ndtri`), `ndtr`, `log_ndtr`, `normpdf`, `logit`; and samplers (normal, gamma, beta, chi-squared, t, ...) from a Philox stream with the same bits on every machine; installed with pkg-config, and `pip install crdistr` | sampling and statistics that must give the same draws on every machine |
| **crlin** ([`lin/`](lin/README.md)) | verified linear algebra: an enclosure of the exact solution of A x = b, about an ulp wide, at about 10 times an unverified solve, safe with a threaded BLAS; or a refusal, never a wrong answer. Also interval data, least squares, very ill-conditioned systems, symmetric eigenvalues and proofs of positive definiteness; installed with pkg-config, and `pip install crlin` | anyone who needs a dense solve they can trust to the last bit |
| **crfix** ([`fix/`](fix/README.md)) | exp, exp2, log, log2, sigmoid, tanh, sqrt, rsqrt, reciprocal and division on fixed-point numbers (int8, int16, int32, any fractional bits), correctly rounded and saturated, in integer arithmetic only, proven on every 16-bit input; SiLU, the exact GELU and layer normalization; int8 and uint8 lookup tables, TFLite's affine formats and softmax; no floating point at all, so the same bits on a Cortex-M0; installed with pkg-config, and `pip install crfix` | integer-only inference kernels and microcontrollers that must give the same bits everywhere |
| **lowp** ([`lowp/`](lowp/README.md)) | correctly rounded math for FP8 (E4M3, E5M2) and OCP MX blocks (MXFP8, MXFP6, MXFP4, MXINT8), proven on every input or correct by construction | low-precision machine learning; hardware and emulator writers |
| **ival** ([`ival/`](ival/README.md)) | the tightest binary64 interval enclosures of 32 elementary functions, and of `atan2`, `hypot` and `pow` on boxes | verified and interval computing |
| **repro-scan, repro-diff** ([`tools/`](tools/README.md)) | what in a binary or a build makes its results machine-dependent; a program run under changed conditions (threads, flush-to-zero, an older CPU) and its output compared | anyone chasing a result that changes between machines |
| **the checking kit** ([`kit/`](kit/README.md)) | number formats down to FP4, correctly rounded references through MPFR, runs with controls and verdicts | building checks like these |

Each is checked the way crmvec is: against answers it did not make, with
controls, and with deliberately planted bugs that the checks must catch.

## Build and check

```
make          # the kit and every library, and their checks
make check    # every check: the kit, repro-scan, lowp, lowp's MX, ival, crsum, crnn, crdist, crlin, crfix
```

Needs gcc 13 or later (for `_Float16` and `__bf16`), MPFR 4.2 or later, and
OpenMP. repro-scan needs Python 3 and binutils; its tests also use clang and
the aarch64 and riscv64 cross compilers, and skip, saying so, the cases
whose compiler is missing. CI runs both on x86-64, natively on arm64 and
on macOS (Apple Silicon), and runs crfix on a Cortex-M0 under qemu
(`.github/workflows/check.yml`).

`make install` installs ival, crdist, crlin and crfix (ival alone on
macOS). Each comes as a shared library with its soname and links, a
static one, its headers and a `.pc` file for pkg-config, under `PREFIX`
(default `/usr/local`), with `LIBDIR`, `INCLUDEDIR` and `DESTDIR` as
usual. Each version is in its header (`IVAL_VERSION` in `ival/ival.h`,
and so on). A program then builds with
`cc prog.c $(pkg-config --cflags --libs crfix)`. `make check` installs each
into a staging directory and builds a C program (shared and static) and a
C++ one against it that way.

From Python: `pip install crival` (ival), `crdistr` (crdist), `crlin` and
`crfix`. Each wheel carries its library, for x86-64 and aarch64 Linux and
Apple Silicon macOS.

Every check ends in a verdict line, as crmvec's do: `IDENTICAL` (exit 0),
`DIFFERS` (exit 1), or `VOID` (exit 2: nothing was tested, a control did not
fail, or the check could not run). repro-scan's own verdicts are `CLEAN`,
`FLAGGED` and `VOID`, with the same exit codes.

## Documentation

Each library's page, in its directory, has its interface, how it works, and
how it is checked: [the checking kit](kit/README.md), [lowp](lowp/README.md),
[ival](ival/README.md), [crsum](crsum/README.md), [crnn](nn/README.md), [crdist](dist/README.md), [crlin](lin/README.md), [crfix](fix/README.md),
[repro-scan and repro-diff](tools/README.md). What comes next is in
[ROADMAP.md](ROADMAP.md), how it got here in [HISTORY.md](HISTORY.md), and
the experiments on GPUs in [GPU.md](GPU.md).
