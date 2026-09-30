# Branch log: claude/numerics-groundwork

For the maintainer who merges this branch: what was done, why, how it
was checked, and what was found along the way. Written 2026-09-30 in a
cloud session (x86-64, 4 cores, gcc 13.3, MPFR 4.2.1, glibc 2.39).

## What this branch is

Step zero of [ROADMAP.md](ROADMAP.md): the groundwork that a family of
reproducible numerics libraries shares. It branches from main at 5bf6f82
and is independent of the review branch (`claude/beautiful-ramanujan-4gvzzd`).
Neither branch touches the other's files: this one adds `numerics/` and
`.github/workflows/numerics.yml`, and changes nothing else. The two can
be merged in either order.

Added:
- `numerics/kit/`: the checking kit (`kit.h`, `fmt.c`, `run.c`) and its
  self-test (`test/selftest.c`);
- `numerics/tools/`: repro-scan and its tests (`test/run-tests`);
- `numerics/lowp/`: correctly rounded FP8 and MX math (Phase 1, below);
- `numerics/ival/`: interval functions (Phase 1, below);
- `numerics/Makefile`: `make` and `make check`, which ends in verdicts
  like crmvec's;
- `numerics/README.md`, `numerics/ROADMAP.md`, this log;
- `.github/workflows/numerics.yml`: `make check` on x86-64 and natively
  on arm64.

## Before merging

- If crmvec is published through the export script's file list (as the
  review branch's log describes), add `numerics/` and
  `.github/workflows/numerics.yml` to it.
- Nothing left unproven in CI: the first run of `numerics.yml`
  (run 36653948093, 2026-09-30) passed both jobs.
  - **arm64 (Neoverse N2):** the kit's self-test is IDENTICAL, so its
    rounding checks hold against AArch64's hardware conversions too.
    repro-scan's tests ran 22 of 22 cases, none skipped, with the x86
    cases cross-compiled.
  - **x86-64:** both verdicts IDENTICAL, 22 of 22 cases.

## Decisions, and why

- **A subdirectory of crmvec, for now, not a new repository.** It shares
  CORE-MATH's sources (the self-test checks the kit against them) and
  crmvec's CI and conventions. What ties it to crmvec is small:
  - `../f16`, `../bf16` and a dozen of CORE-MATH's files at the top;
  - `../crmvec-f16-list.h`.

  Splitting it out later is a matter of vendoring those.
- **Encodings, not C types.** Every format is a `uint64_t` encoding with
  exact decoding to `double`. One code path serves six formats, including
  FP8, which has no C type.
- **The reference's recipe:** compute at the format's precision in MPFR's
  wide exponent range, then `mpfr_check_range` and `mpfr_subnormalize`.
  This is MPFR's documented recipe for a narrower range. crmvec's
  `f16check` sets the range before computing instead; both are valid.
  This one also keeps every input in range, whatever the format.
- **E4M3 overflow:**
  - toward zero: ±448;
  - away from zero: NaN, or ±448 with `kit_e4m3_saturate`;
  - an exact infinity (an infinite input or a pole): NaN, or ±448 when
    saturating, in every direction.

  **Unverified:** that the OCP 8-bit specification says the same for an
  infinity in saturating mode. The rule is in one place (`finish()` in
  `fmt.c`) if it needs changing.
- **The control inside every run**, rather than as separate runs: every
  result is moved by one ulp on one input in 64 and compared with the same
  reference. It costs nothing extra, and nobody can forget to run it.
- **Combining verdicts:** DIFFERS over VOID over IDENTICAL. A proven
  difference is the more useful thing to report.
- **repro-scan in Python on binutils**, not a C ELF parser: no
  dependencies beyond what a build machine has, and `llvm-objdump` as the
  fallback for another ISA.
- **FLAG versus info:**
  - FLAG: what gives different bits on another machine of the same kind,
    or in another build, as a matter of course;
  - info: AArch64's and RISC-V's estimates (exact within the ISA), FMA
    (not a difference by itself), libm calls, FP-environment writes,
    dispatch and OpenMP regions.

  `--strict` counts info findings as flags.
- **A separate workflow file, and this log under `numerics/`,** so as not
  to collide with the review branch's changes to `check.yml` and its
  `BRANCH-LOG.md` at the top.

## How it was checked

- `make -C numerics check`: both verdicts IDENTICAL.
  - The kit's self-test takes 12 s on four cores. It checks the kit
    against the hardware's conversions and OCP's tables, a brute-force
    rounding search, and CORE-MATH (every binary16 and bfloat16 input of
    37 functions, samples of 4 more and of binary32 and binary64, the
    8-bit formats through binary64). All in four rounding modes, with a
    negative control.
  - repro-scan's tests: 22 of 22 cases on x86-64, AArch64 and RISC-V,
    and the control fails as it must.
- **The kit, under ASan and UBSan:** clean (50 s).
- **Planted bugs, one at a time.** Each was caught (DIFFERS, VOID or an
  abort):
  - eleven in the kit. The self-test's brute-force part alone caught
    every rounding bug. The one that survived changed nothing, and its
    code was removed.
  - four in repro-scan.
- **repro-scan on real libraries** (numerics/README.md has the details):
  - crmvec's `libmvec.so.1`, built from this branch's base: CLEAN;
  - glibc 2.39's `libmvec.so.1`: FLAGGED, 79 estimate instructions;
  - glibc 2.39's `libm.so.6`: FLAGGED, 39 x87 instructions.

## Found along the way, for crmvec

1. **crmvec's binary16 `cbrt` depends on the platform's libm.**
   - **What:** CORE-MATH's `cr_cbrtf16` (`f16/cbrtf16.c`) returns
     `cbrtf((float)x)` for most inputs, which is the C library's `cbrtf`,
     not CORE-MATH's. repro-scan's libm finding led to it.
   - **The kit's measurement** (every input, four modes):
     - correct with glibc 2.39's `cbrtf`, with CORE-MATH's, and with
       `(float)cbrt(double)`;
     - wrong on 3 inputs with a `cbrtf` one ulp off on some inputs, as a
       libm within 1 ulp may be. The first: 0x23c9, to nearest, gives
       0x33ed instead of 0x33ee.
   - **Suggested fix (untested):** build `f16/cbrtf16.c` with
     `-Dcbrtf=cr_cbrtf` (crmvec already builds CORE-MATH's `cbrtf.c`),
     then run `f16check` and a scan.
   - **Upstream:** whether to tell CORE-MATH is the owner's call.
   - `cr_cbrt_bf16` doesn't call libm.
2. **Dead code in `libmvec.so.1`.**
   - **What:** each of CORE-MATH's binary16 and bfloat16 files also
     defines a stand-in under the bare name (`sinf16`, `acos_bf16`, …)
     that calls libm. crmvec links them in, hidden, but never exports or
     calls them. They are the source of every other libm import.
   - **Suggested fix (untested):** `-ffunction-sections
     -Wl,--gc-sections` should drop them.
3. **A data point for crmvec's README:** glibc's own `libmvec.so.1` uses
   reciprocal estimates (`rcpps`, `vrcp14pd` and others), whose bits
   differ between x86 vendors. The same glibc can give different answers
   on different machines.

## Phase 1, first library: lowp (FP8 math), 2026-09-30

Added `numerics/lowp/`: correctly rounded E4M3 and E5M2 math (README.md,
"lowp"). The kit gained `kit/fns.c`, the function list with its MPFR
references, shared by the self-test, lowp's generator and lowp's check.

Decisions:
- **Tables for the one-argument functions, committed** (`lowp-tables.h`,
  280 KB of text). A lookup gives the same bits on every machine by
  construction, and users need no MPFR. `make check` regenerates the
  tables and compares them byte for byte, so a stale copy fails.
- **CORE-MATH's binary64 functions for the two-argument ones.** Tables of
  every pair would be megabytes, and the exhaustive check proves the
  binary64 route instead. lowp carries its own copy of `pow`, `atan2`,
  `atan2pi` and `hypot`, made local to it (`objcopy --localize-hidden`), so
  it links beside CORE-MATH or crmvec without a clash.
- **An explicit mode argument**, rather than the C rounding mode: no
  hidden state changes a result. The caller's rounding mode and flags are
  restored.
- **The kit's control now replaces a NaN with zero** (it left NaN alone
  before). An 8-bit run moves only about 4 results, and for `acosh` all
  four were NaN, so the control was blind (VOID). Every earlier verdict
  still holds.

Checked:
- `make check` gives four IDENTICAL verdicts in 29 s: the kit, repro-scan,
  the tables' freshness, and lowp's check;
- lowp's check covers every input and pair, four modes, three
  configurations, against MPFR and against a second path through
  CORE-MATH's binary64 functions;
- clean under ASan and UBSan;
- ten planted bugs: nine caught. The tenth, the two-argument functions
  left in round-to-nearest, changes no result on any pair (README.md says
  why), and the code keeps the matching mode anyway.

## Phase 1: FP6, FP4 and MX, 2026-09-30

- **The kit:** E2M3, E3M2 (FP6) and E2M1 (FP4) are the first formats with
  neither infinity nor NaN. Overflow saturates, and a NaN result has no
  encoding (`KIT_NONE`). A new self-test part G counts what exhaustive and
  every-pair runs hand over, since a comparison can't see an input left
  out. The control now always moves at least one result: a 16-input run
  had moved none. Six planted bugs, each caught, two after adding the
  check that could see them.
- **`lowp/MX.md`:** what a correctly rounded function on an MX block is.
  - **Its sources:** OCP's specification couldn't be read (opencompute.org
    is blocked here), so the rules follow the reference implementation,
    microsoft/microxcaling at 7bc41952de39, each marked to be checked
    against the spec. The decision it needs from the owner: have someone
    check those rules against the spec's text.
  - **The scale is taken from the exact results.** The reference's PyTorch
    path takes log2 in the tensor's own type. The largest binary32 below
    2^k then gets scale exponent k instead of k−1, for every k from 4 to
    127 tried: worth telling them, the owner's call.
- **`lowp/mx.c`:** the MX functions for 5 element types and 37 functions.
  - **Correct by construction:** CORE-MATH's binary64 results rounded down
    and up give the result rounded to odd, which rounds again correctly,
    and the one nearer zero gives the scale's exponent exactly.
  - **The cost:** liblowp now carries 36 of CORE-MATH's binary64 functions
    (747 KB).
  - **Checked:** `lowp/test/mx-check.c` builds blocks with purpose against
    the kit's exact reference (`kit/mx.c`). IDENTICAL, in 27 s. Ten planted
    bugs, each caught. Clean under ASan and UBSan.

## Phase 1: ival (interval functions), 2026-09-30

`numerics/ival/`: interval versions of 31 of CORE-MATH's binary64
functions, each the tightest enclosure.

Decisions:
- **No high-precision π in the library.** `sin`, `cos` and `tan` are cut
  into pieces shorter than π, and the exact signs of the derivative at the
  pieces' ends say what lies inside. `sinpi`, `cospi` and `tanpi` count
  their integer and half-integer points exactly. The check's reference
  does use π to 2,200 bits, so the two agree by different routes.
- **Domains as IEEE 1788 has them:** the interval is intersected with the
  domain, open at poles (log at 0, atanh at ±1). An empty result is
  [NaN, NaN].
- **Scalar CORE-MATH for now,** switching the rounding mode per bound.
  Vector code for the directed modes comes later.

Checked:
- IDENTICAL on about 33,000 intervals per function, with no sampled
  interior point outside, clean under ASan and UBSan;
- five bugs found by the check in the first version, all fixed (README.md
  lists them);
- eleven planted bugs, each caught, one after adding the intervals that
  could see it.

## Phase 1: repro-scan's options scan, and repro-diff, 2026-09-30

- **repro-scan reads `compile_commands.json`:**
  - FLAG: fast-math options, `-mrecip`, x87 arithmetic, `-march=native`
    and `-ffp-contract=fast`;
  - noted: contraction not pinned off.
- **`numerics/tools/repro-diff`** runs a program as given and under
  changed conditions (a repeat, 1 and N threads, flush-to-zero by an
  `LD_PRELOAD` shim, an older CPU under qemu), and compares output, exit
  status and named files byte for byte.
- **Its first trial:** glibc 2.39's `exp`, `sin`, `pow`, `cos`, `atan` and
  `log` give other bits on a CPU without FMA, for 10 to 138 of 200,000
  arguments each (README.md has the example); a CORE-MATH build gives the
  same bits. A data point for crmvec's README too.
- **Checked:** 39 cases in `tools/test/run-tests`, each with its exact
  expected result; eight planted bugs, each caught.
- **One test error on the way:** `target_clones("arch=haswell")`
  dispatches on the CPU model, not its features, so a test program
  never took its FMA path. The test now uses `target_clones("fma")`.
- **CI:** both jobs install qemu-user, so the `cpu` condition runs there
  too (skipped where it can't).

## What's left

The rest of ROADMAP.md, Phase 1:
- MX: the rules against OCP's text, INT8 elements, packed FP4;
- bfloat16 vector functions;
- ival: `lgamma`, `tgamma`, two-argument functions, binary32, vector
  code;
- repro-scan for GPU code and Python wheels; more repro-diff conditions.

Also not done: the OCP check of E4M3's infinity rule, above (lowp inherits
it).
