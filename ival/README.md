# ival: interval functions and arithmetic

`ival/ival.h` gives interval versions of 32 of CORE-MATH's binary64
functions: `ival_f(lo, hi, ylo, yhi, n)` maps each interval
[lo[i], hi[i]] to the smallest binary64 interval that contains f over it.
That is the least value rounded down and the greatest rounded up, each
correctly. For example, `ival_sin` on [1, 2] gives [sin 1 rounded down, 1],
since the maximum at π/2 lies inside. The rules:
- an interval is intersected with the function's domain first: log on
  [−1, 4] is log on [0, 4], which is [−∞, log 4 rounded up];
- an empty intersection gives the empty interval [NaN, NaN];
- infinite ends are allowed;
- a pole inside gives an infinite end;
- the C rounding mode and flags are left as they were;
- flush-to-zero and denormals-are-zero, which a `-ffast-math` program
  starts with, are turned off for the call and given back (2026-10-09;
  before that, a caller with them set got wrong bounds near the
  subnormals).

The functions: the monotone ones (`exp`, `log`, `atan`, `erf`, `sqrt`,
`acos` and 18 more), `cosh`, the periodic `sin`, `cos`, `tan`, `sinpi`,
`cospi`, `tanpi`, and `tgamma` (2026-10-02). Of the two-argument functions, `hypot` and `atan2`
(2026-09-30), which take a box, X × Y, in C's argument order:
- **`ival_hypot(xlo, xhi, ylo, yhi, zlo, zhi, n)`:** its bounds are at the
  least and greatest magnitudes, since hypot grows with |x| and |y|;
- **`ival_atan2(ylo, yhi, xlo, xhi, zlo, zhi, n)`:** over the box minus the
  origin, where atan2 is undefined (the origin alone is empty). A box with
  points at x < 0 on both sides of the axis gives the whole range, from
  −π rounded down to π rounded up; otherwise the bounds are at its corners.
  Zeros are unsigned, as in IEEE 1788: y = 0 at x < 0 is π;
- **`ival_pow(xlo, xhi, ylo, yhi, zlo, zhi, n)`:** IEEE 1788's `pow`, whose
  domain is x > 0, and x = 0 with y > 0 (negative bases are for `pown` and
  `rootn`). It is monotone in each argument with the other fixed, so its
  bounds are at the corners; at the domain's edge C's `pow` gives the
  limits from inside (`pow(+0, y)` is 0, 1 or +∞ as y is positive, zero or
  negative). x = 0 alone gives [0, 0] when y reaches above 0, and the
  empty interval otherwise.

Not yet: `lgamma` (on the negatives its segments run on to 2^52, past any
table), binary32.

**How it works.** Each bound is a CORE-MATH value, computed rounding down
or up, at the point of the interval where f is least or greatest:
- **monotone functions:** an end;
- **`cosh`:** 1 when 0 is inside;
- **`sinpi`, `cospi`, `tanpi`:** their extrema and poles are at integers
  and half-integers, and ival counts them exactly;
- **`sin`, `cos`, `tan`:** the interval is cut into pieces shorter than
  half a period, and the sign of the derivative at each piece's ends
  says whether the piece holds an extremum or a pole.
  - Those signs are exact, computed rounding to nearest: rounding down
    would turn sin of 2^−1074 into 0.
  - Beyond 2^54, neighbouring binary64 values are further apart than π.
    There the parity of the sign changes counts the one or two critical
    points.
- **`tgamma`:** its poles are the integers 0, −1, −2, …
  - A pole alone gives the empty interval. A pole strictly inside gives
    [−∞, +∞], since Γ changes sign across each pole.
  - Otherwise the interval lies in one segment, (−n−1, −n), or in [0, ∞).
    The bounds are its ends' values, with a pole at an end giving that
    side's infinity. When the segment's one extremum lies inside, its
    value replaces the lower bound (Γ > 0) or the upper one (Γ < 0).
  - No extremum is a binary64 value. `ival/tgamma-table.h`
    (`gen-tgamma.py`, mpmath at 400 bits) gives each one as the two
    binary64 values around it, with its value rounded down and up. That
    covers the minimum on the positives and the segments down to −184.
  - Below −184, every binary64 value has |Γ| < 2^−1074. The extremum's
    value then rounds to +0 or −0, and that bound holds anyway.

**How it is checked** (`ival/test/check.c`, under a minute on three cores,
most of it `tgamma`'s reference; `IVAL_ONLY=f` runs one function):
- **The reference** finds the bounds its own way: MPFR at both ends in
  both directions, without assuming monotonicity. It adds each
  function's critical points and poles inside the interval, located with
  π to 2,200 bits and exact counting. For `tgamma` it uses no table:
  - the poles come from exact counting;
  - each extremum comes from bisecting on digamma's sign, which is exact
    at any precision because MPFR rounds correctly;
  - its value is evaluated at 400 bits;
  - signs come from MPFR's lgamma, since Γ itself underflows MPFR's
    exponent range far out.
- **The intervals, per function:**
  - points;
  - 2^14 random intervals of every width and magnitude;
  - intervals around every kind of critical point, near and far (the
    binary64 value nearest a multiple of π/2 included; for `tgamma`,
    every extremum's two neighbours and every pole down to −200, across
    the table's end and past 2^52 and 2^53);
  - the domain's edges;
  - infinite and empty ends.
- **A check with no reference:** the exact function at 8 points inside
  each interval must lie within it.
- **Controls:** each run's control, a negative control (`exp` against
  `exp2`), and a test in place.
- **The flush modes:** every function again with flush-to-zero and
  denormals-are-zero set, bit for bit the same, and the modes still set
  afterwards. Without the fix, up to 7,937 intervals of a function
  differed.
- **`hypot`, `atan2` and `pow` on boxes:** 69,376 boxes (the intervals
  above paired at random, the specials against each other, boxes on and
  across both axes and around x = 1 at every scale), against a reference
  that evaluates MPFR at every pair of candidate points (the ends; zeros
  inside; for atan2's y, +0 and −0, the value on the axis and the limit
  from below; for pow, x = 1 and y = 0, within its domain written out
  again), and 8 exact points in each box. Negative controls must differ:
  the corners alone (hypot: 14,585 boxes; atan2: 6,948), and for pow, x's
  reach to 0 forgotten (5,552). Planted bugs, each caught: for hypot, a
  zero crossing ignored and the upper bound rounded down; for atan2, the
  cut ignored, signed zeros kept, and π rounded down; for pow, the x = 0
  case, the domain's cut at 0, and the diagonal corners only. (The first
  try at that last plant changed no result, since an inner loop still ran
  every corner: a plant counts only once it changes an output.)

All IDENTICAL: about 33,000 intervals and up to 150,000 points per
function. Clean under ASan and UBSan.

**The check found five bugs** in the first version, all fixed:
- the pieces of `sin`, `cos` and `tan` where neighbours are further apart
  than π;
- `sinpi`, `cospi` and `tanpi` beyond 2^52;
- a point interval of `cos` at 0;
- `cos` on [0, 2^−1074], where rounding down made sin's sign 0;
- the reference evaluating `tanpi` at a pole on the interval's end.

Eleven bugs were then planted, one at a time, and each was caught. One of
them, −0 not replaced by +0 at an end, first needed new test intervals
[−0, x]: rsqrt(−0) is −∞, but rsqrt over [−0, 4] reaches +∞.

## Installing and versions

`make install` (crnumerics' Makefile; `PREFIX`, `LIBDIR`, `INCLUDEDIR`,
`DESTDIR`) installs `libival.so.<version>` with the links `libival.so.0`
and `libival.so`, `libival.a`, `ival.h`, `ival-list.h`, `ival-scalar.h`
(below) and `ival.pc`.
- **The version** is `IVAL_VERSION` in `ival.h`, 0.1.0 so far.
  `ival_version()` gives the library's, to compare with the header's.
- **The soname** is `libival.so.<first number>`. While that number is 0,
  any release may change the interface.
- **Exports:** only `ival_` names. CORE-MATH's functions inside it are
  hidden, so a program's own copy of them does not clash.
- **C++:** `ival.h` declares everything `extern "C"`.
- **Linking statically:** `libival.a` needs `-ldl -lm` after it, which
  `pkg-config --static --libs ival` gives.

`ival/test/install-check.c`, in `make check`, uses the library the way a
user would. It installs into `build/stage`, builds a C program through
pkg-config (shared and static) and a C++ one, and checks:
- the versions agree, and two results come out tight;
- the soname is right, and every export is an `ival_` name;
- the static build runs without `libival.so`.

Each of these fails on a wrong library, and on a static build that is
missing or that loads `libival.so`.

## Python

`ival/python` is a binding over NumPy arrays (ctypes, nothing to
compile): `pip install ./ival/python` once libival is installed. Its
README has the details, and a comparison with pyinterval.

## One interval at a time

Some callers work interval by interval: a C++ interval class (the backends
of the solver IBEX), or a binding called once per element. Two things make
that cheap (2026-10-09).

**The library's calls.** A call on one interval used to cost 164 to 186 ns.
glibc's `fegetenv`/`fesetenv` (122 ns on x86-64) and `fesetround` (146
there and back) were most of it: they handle the x87 unit, which ival never
uses.
- **Now, on x86-64:** ival saves, sets and restores MXCSR alone, writing it
  only when the value changes.
- **ival's own CORE-MATH objects** have their fenv calls renamed to MXCSR
  versions (`ival-fenv.c`). Both halves are needed:
  - glibc's `fegetround` reads the x87 mode, and cos, tan and pow ask it.
    Without the rename, the check fails.
  - glibc's `feraiseexcept` sets overflow, underflow and inexact in the x87
    status word. Without the rename, pow's underflow reached the caller,
    and the environment check fails.
- **On aarch64:** FPCR and FPSR directly. glibc's calls use the same
  registers, so no rename is needed.
- **Up to 4 intervals,** the arithmetic runs the scalar code, which is exact
  without changing the rounding mode.

`ival/test/env-check.c` (in `make check`) calls each kind of entry point in
every rounding mode, with no flag raised and with all raised, and with the
flush modes on and off. Afterwards the mode, the flags, MXCSR and the x87
control and status words must be as the caller had them. The control,
planted bug 42, keeps the flags raised inside and fails it.

**`ival-scalar.h`** (installed with `ival.h`) has the arithmetic inline for
one interval: `ival1_add`, `_sub`, `_mul`, `_div`, `_sqr`, `_sqrt`,
`_recip`, `_neg`, and `ival1_scale(d, ...)`, the point [d, d] times an
interval in two products (2026-10-10; a mixed-sign `_mul` takes four).
Plain C99 and C++11.
- **The same results as the library, bit for bit.**
- **Two modes inline.** Rounding upward (as Gaol-style classes keep it),
  each bound is one operation, the lower one negated on negated operands.
  To nearest, each bound is the result to nearest stepped one double
  toward its exact residual (TwoSum, fma).
- **Any other mode, or the flush modes on:** the library. Each function
  checks; on x86-64 the check is three operations rather than a read of
  MXCSR, which was half an add's cost on Zen 3. On aarch64 it reads FPCR.
- **Callers built without `-frounding-math`,** as most are: the check sits
  behind a volatile barrier, so a compiler that takes floating-point
  operations to be pure cannot reuse it across a change of mode or move it
  out of a loop. `mode-change-check` (in `make check`, built without
  crnumerics' FP flags) changes the mode between calls and in a loop;
  planted bug 47, a check stuck on "upward", fails 66 of its 132 results.
- **Flags** are raised as ordinary arithmetic raises them.
- **Checked:** arith-check compares it with the library on every result,
  to nearest, rounding up, rounding down and with the flush modes set, and
  `ival1_scale` with `ival_mul` of the point (planted bug 46, the ends of a
  negative scale swapped, differs on 205,502). It found a
  bug in the first version: `neg`, being exact, skipped the check, and with
  denormals-are-zero its comparisons read a subnormal end as 0. A control
  whose check never hands over differs on 210,021.

ns per operation; x86-64 is board5 (Ryzen 5 PRO 5650U, load under 1), aarch64
is cfarm424 (Neoverse N1):

| | add | mul | div | sqrt | exp |
|---|---|---|---|---|---|
| x86-64: the library, one interval, before | 164 | 168 | 166 | | 186 |
| x86-64: the library, one interval | 17 | 47 | 29 | | 54 |
| x86-64: `ival-scalar.h` | 10 | 17 | 16 | 10 | |
| x86-64: `ival-scalar.h` with `-mfma` | 7.6 | 12 | 13 | 8.2 | |
| x86-64: the library, arrays | 0.7 | 1.3 | 1.4 | | 22 |
| aarch64: the library, one interval | 64 | 133 | 98 | | 222 |
| aarch64: `ival-scalar.h` | 10 | 15 | 22 | 11 | |

To nearest, an inline add is about 113 instructions: the empty tests, the
mode check and two exact sums. Rounding upward it is the empty tests, the
mode check and two additions.

## Threads

Every function may be called from any number of threads at once. The
rounding mode and flush modes ival sets are the calling thread's own, and
are given back. Two pieces of state are shared, both set once:
- the arithmetic's check for AVX2 and FMA, stored atomically;
- the accurate mode's load of crmvec. One thread claims it, and the
  others wait until it is done, so every call after it gives the same bits.

`ival/test/thread-check.c` checks this. Eight threads make the process's
first calls together, each with a different rounding mode set, then run 20
rounds of the arithmetic, the functions in both modes and a reverse
operation. Every result must equal one thread's alone, and every thread's
mode must be kept. `make check` runs it. `make ival-thread-tsan
[CRMVEC=...]` runs it under ThreadSanitizer, which reports no race. That
build reported the race in the accurate mode's first version of the load,
where every first caller wrote the table, when that version was planted
back.

## The accurate mode

`ival_acc_f` (2026-10-09), for each of the 32 functions and for atan2,
hypot and pow, gives IEEE 1788's "accurate" result: each bound at most one
ulp outside the tightest one. It runs at vector speed through crmvec,
whose vector functions are correctly rounded to nearest. Moved one ulp
outward, such a value bounds the exact one, and lies at most one ulp
beyond the tight bound, which is that value or its neighbour. Each end is
then held to the function's range where that is exact (exp ≥ 0,
|tanh| ≤ 1, hypot ≥ 0, ...).

crmvec's `libmvec.so.1` is loaded at first use from the path the
environment variable `IVAL_CRMVEC` names, on CPUs with AVX2 and FMA.
crnn loads it the same way, and nothing is needed at build time.

How each function runs:
- **four intervals at a time throughout:**
  - the monotone functions and cosh;
  - sin, cos and tan on intervals narrower than 2.5, where the slopes at
    the ends (from crmvec's cos and sin) say whether an extremum or a pole
    lies inside;
  - hypot, at the least and greatest magnitudes of each side.
- **by record and replay:** wide sin, cos and tan, tgamma, atan2 and pow.
  - The tight mode's own bound logic (the pieces, the exact counting,
    tgamma's table of extrema, the box corners) never chooses its points
    by the values it gets back.
  - So it runs once to record the points it asks for. crmvec evaluates
    all of them, four at a time, and the logic runs again on those values
    moved outward.
  - One value provider serves both modes. The tight mode's results are
    unchanged by it, as check.c shows.
- **the tight path, which is accurate too:**
  - the pi functions: crmvec's vector sinpi and cospi take 17.7 ns a
    value, and through them the accurate mode measured slower than the
    tight one;
  - sqrt, which is fast already;
  - the whole call, without crmvec.

`ival/test/acc-check.c` checks the accurate mode.
- **What it checks:** every result must hold the tight one (itself checked
  against MPFR) and be at most one ulp wider at each end.
- **Coverage:** 1,164,310 results, the box functions included.
- **Void runs:** with crmvec, a function whose results never differ from
  the tight mode would make the run void, since its vector path cannot
  have run. Without crmvec, the two modes must be equal.
- **Running it:** `make ival-acc-check CRMVEC=/path/to/libmvec.so.1` runs
  it through crmvec; `make check` runs it without.

**Cost**, ns per interval for 4,096 narrow intervals (`make
build/ival-fn-bench`, board5, load 3, 2026-10-09):

| function | tight | accurate |
|---|---|---|
| exp | 25 | 11 |
| log | 49 | 12 |
| atan | 33 | 20 |
| sin | 171 | 25 |
| cos | 186 | 35 |
| tan | 271 | 31 |
| cosh | 54 | 19 |
| tgamma | 276 | 194 |
| atan2 | 200 | 155 |
| pow | 586 | 141 |
| hypot | 50 | 11 |

On wide intervals, cos drops from 239 to 145 ns.

The tight mode itself switches the rounding mode once per block of 64
intervals: one pass rounding to nearest for the domain and sin's pieces,
then every lower bound rounding down, then every upper bound rounding up.
Before that change (same date) it switched per interval, and exp took
133 ns, sin 592.

## How it compares

`make ival-compare` (`ival/test/compare.cpp`) runs ival and four other
interval libraries on the same 4,096 narrow intervals. It reports ns per
operation, and checks every result against ival's tight one, which
ival's own checks prove against MPFR. A result is tight, wider by some
ulps, or misses the tight one, in which case it is not an enclosure.

cfarm421 (EPYC 7773X), one core, GCC 14, 2026-10-09:

| library | add | mul | div | exp | log | sin | atan |
|---|---|---|---|---|---|---|---|
| ival tight | 1.0 | 1.6 | 1.6 | 24 | 51 | 192 | 33 |
| ival accurate (crmvec) | 1.0 | 1.6 | 1.6 | 12 | 12 | 29 | 23 |
| filib++ 3.0.2 | 2.2 | 3.7 | 4.1 | 35 | 19 | 35 | 28 |
| Boost.Interval 1.83 | 44 | 128 | 44 | 49 | 50 | 396 | 138 |
| MPFI 1.5.4 (53 bits) | 67 | 93 | 82 | 2,058 | 2,680 | 4,365 | 5,260 |
| libieeep1788 | 294 | 347 | 338 | 2,216 | 2,836 | 8,249 | 5,345 |

The widths:
- **ival tight, MPFI and libieeep1788:** every result tight.
- **ival accurate:** about 75% of the function results are wider, by one
  ulp at an end at most.
- **filib++:** arithmetic tight. Every function result is wider, by up to
  16 (exp, log), 29 (sin) and 36 (atan) ulps.
- **Boost.Interval:** arithmetic tight. Its transcendental policy,
  rounded_transc_std, calls the C library's functions with the rounding
  mode set down and up. Its documentation says those functions must honour
  the mode, "unfortunately ... rarely the case". glibc's do not promise
  it. With them, 61 of the exp results, 190 of sin and 3,095 of atan miss the true
  value (for example atan of [−10, −10] comes back as a point beside
  it), and log's are up to one ulp wider.

The command used, with the other libraries' paths: `make ival-compare
COMPARE="-DHAVE_MPFI -DHAVE_BOOST -DHAVE_FILIB -DHAVE_P1788"
COMPARE_INC="-I <prefix>/include -I <libieeep1788> -I <its cmake
build>" COMPARE_LIBS="<prefix>/lib/libprim.a <prefix>/lib/libmpfi.a -lmpfr
-lgmp"`, with IVAL_CRMVEC naming crmvec's libmvec.so.1.

**Against Julia's IntervalArithmetic.jl**, `make ival-julia-compare`
(`ival/julia/`) runs the same 4,096 narrow intervals through ival and
through IntervalArithmetic.jl's default rounding, `:correct`.
- **How `:correct` gets its bounds:** CRlibm where it has the function,
  MPFR otherwise.
- **Results:** both claim the tightest enclosure, and every bound agrees,
  bit for bit.
- **Times:** ns per interval on one EPYC 7773X core (cfarm421, load 14,
  2026-10-09), with IntervalArithmetic 1.0.12 on Julia 1.11.7. Each is the
  best of three runs, each run the least of 7 passes.

| | exp (CRlibm) | exp2 (MPFR) | tanh (MPFR) |
|---|---|---|---|
| IntervalArithmetic.jl, `:correct` | 70 | 3,130 | 3,016 |
| ival, tight | 30 | 31 | 60 |

## Arithmetic

`ival_add`, `ival_sub`, `ival_mul`, `ival_div` (two intervals),
`ival_neg`, `ival_sqr`, `ival_recip` (one) and `ival_fma` (three) give
the tightest binary64 interval around the exact result, under IEEE 1788's
rules (2026-10-08 and 09):
- empty operands give the empty interval;
- at interval ends 0 × ∞ is 0;
- division goes by where 0 lies in the divisor: B = [0, 0] is empty,
  B = [0, b] with A < 0 gives [−∞, a/b], and 0 strictly inside B gives
  the whole line;
- zero ends come out as +0.

The C rounding mode, the flags, and flush-to-zero and
denormals-are-zero (which a `-ffast-math` program starts with) are
given back as they were.

**How it works.** Arrays go in blocks of 256: every lower end of a block
is computed rounding down, then every upper end rounding up. That is two
rounding-mode changes a block, none per operation. On x86-64 with AVX2
and FMA the passes take four lanes at a time; elsewhere they are plain C
that the compiler vectorizes. The lanes the passes do not cover (empty
operands, divisors touching 0) go to the scalar code.

That scalar code is a second algorithm. It rounds to nearest, and the
exact rounding error from an error-free transformation (TwoSum, the fma
residual of a product or quotient) says whether each end moves one ulp.
Near underflow, where those errors stop being exact, it rounds down or up.
It was planned as the vector code too, until measuring showed it 3 to 13
times slower than switching the mode per block.

**Cost**, ns per interval for n = 1024 (`make build/ival-arith-bench`,
one core, the least of 7 passes, 2026-10-09). "Unrounded" is the ends
rounded to nearest: no enclosure, the floor any method pays.

| machine | operation | unrounded | ival | scalar reference |
|---|---|---|---|---|
| EPYC 7773X (AVX2) | add | 1.05 | 1.30 | 11.1 |
| | mul | 1.61 | 2.09 | 75.6 |
| | div | 5.61 | 1.85 | 23.0 |
| Neoverse N1 (portable C) | add | 1.08 | 3.54 | 18.7 |
| | mul | 4.06 | 10.3 | 79.4 |
| | div | 7.16 | 9.63 | 42.9 |

Division beats its unrounded form on the EPYC because the four-lane passes
divide each end once, where the unrounded loop divides all four corners.

**How it is checked** (`ival/test/arith-check.c`, a few seconds):
- **The reference** is MPFR, finding each result its own way rather than
  by 1788's case tables:
  - add and sub: the ends' sums rounded down and up;
  - mul and sqr: the exact corner products, least and greatest;
  - div: every quotient of ends, plus the limits where the divisor
    reaches 0;
  - fma: the least exact product plus C's lower end, at 4,400 bits.
- **The inputs:** every pair of 24 special ends (zeros, infinities,
  DBL_MAX, subnormals, and 2^−969 and 2^−960, where the scalar
  transformations stop being exact), 2^15 random intervals of every
  magnitude and width, and empty ones. That is 352,144 pairs, or triples
  for fma.
- **Points inside:** exact results at points inside the operands must
  lie in the result.
- **The paths:** the default, the portable passes and the scalar
  reference must agree bit for bit (7.4 million results), also with the
  flush modes set and with the result written over an operand.
- **The negative control:** ends merely rounded to nearest must differ
  from the reference.
- **Planted bugs:** fifteen, one at a time, each caught.

All of it passes on board5 (GCC 13, Clang 18), the EPYC and the N1 (GCC
14, Clang 19).

## The other 1788.1 operations

`ival/ival-1788.c` (2026-10-09) has the rest of 1788.1's basic
operations, with the same conventions:
- **one interval to one:** `pos`, `abs`, `sign`, and the integer
  roundings `ceil`, `floor`, `trunc`, `round` (1788's roundTiesToAway)
  and `roundeven` (roundTiesToEven);
- **two to one:** `min` and `max`, `intersect`, `hull` (convexHull), and
  `cancelminus` and `cancelplus`. cancelMinus(A, B) is the tightest Z
  with B + Z holding A. It exists when A is at least as wide as B,
  compared exactly (two differences, each exact as a TwoSum pair, since
  rounding to nearest is monotone); otherwise the result is the whole
  line;
- **numbers:**
  - `inf` and `sup`: −0 for inf at a zero end, as 1788.1 says, and +∞ and
    −∞ for the empty interval;
  - `mid`: rounded to nearest, and ±DBL_MAX for a half-line;
  - `wid` and `rad`: rounded up;
  - `mag`, `mig`, and `midrad` for mid and rad together;
- **booleans** (one byte each): `isempty`, `isentire`, `issingleton`,
  `iscommon`, `ismember`, `equal`, `subset`, `less`, `precedes`,
  `interior`, `strictless`, `strictprecedes`, `disjoint`;
- **`overlap`:** which of the sixteen states two intervals are in (an
  `enum ival_overlap`);
- **`mulrevpair` and `mulrev`:** division with gaps, 1788's
  mulRevToPair and mulRev. The set {x : x·b ∈ C for some b ∈ B} comes as
  at most two intervals, the first before the second. mulrev gives the
  hull of that set's intersection with X.
  - An end of the set is a quotient of ends, kept rounded both ways. The
    hull takes the outer roundings.
  - Meeting X is decided exactly on the inner roundings: a real q is below
    a binary64 x exactly when q rounded down is.
  - An end that is 0 only as a limit (an infinite divisor) is not in the
    set.
- **reverse operations** (`ival/ival-rev.c`): `sqrrev`, `absrev`,
  `coshrev` and `pownrev` give the tightest interval around
  {x ∈ X : f(x) ∈ C}. With X the whole line they are 1788's one-argument
  forms.
  - The set comes as pieces whose ends are the inverse function at C's
    ends: sqrt, the identity, acosh, or a root.
  - Those ends are rounded both ways and intersected with X exactly, as
    for mulrev.
  - No correctly rounded n-th root exists here, so the root is found by
    search over the doubles. CORE-MATH's pow is correctly rounded, so
    whether a double lies below the root is decided exactly: y^p ≤ c
    exactly when y^p rounded up is. The largest such double is the root
    rounded down.
- **`sinrev`, `cosrev` and `tanrev`:** the periodic reverses. Each end
  is the first point of the set past X's end, rounded:
  - the first crossing of C's nearer end is computed in double-double, an
    inverse function plus multiples of the period, to a few ulps (below
    2^48; beyond, X's end is scanned from directly);
  - from a few ulps before it, the doubles and the gaps between them are
    tested exactly. A double is in the set when f there, rounded down
    and up, lies within C. A gap holds a point of the set when f's range
    over it meets C: f's values at the gap's ends, and the extrema or
    poles inside, counted by the parity of the slopes' signs (at most one
    in a gap shorter than half a period, every value in one two periods
    long);
- **`powrev1` and `powrev2`:** 1788's powRev1 and powRev2, the tightest
  interval around {x ∈ X : x^y ∈ C for some y ∈ B} and around
  {y ∈ Y : x^y ∈ C for some x ∈ A}, on pow's domain (x > 0, and x = 0
  with y > 0).
  - The set is at most four pieces: x = 0, and the exponents above 0,
    below 0 and at 0 (for powrev2, the bases above 1, below 1 and at 1).
  - Over one side the intervals {x : x^y ∈ C} move continuously with y,
    so their union is an interval. Its ends are c^(1/y) (for powrev2,
    log_x(c)) at an end of C and an end of the side, chosen by whether c
    is above or below 1.
  - An end at an excluded point is a limit and the piece is open there:
    y → 0 or x → 1 runs the power to 0 or ∞; y → ±∞, x → 0 and x → ∞
    take c^(1/y) to 1 and log_x(c) to 0.
  - Any other end is found by search as pownrev's roots are: whether a
    double t is at most c^(1/y) is whether t^y ≤ c, which CORE-MATH's pow
    rounded up decides exactly.
  - 0.8 µs an interval for powrev1, 1.6 µs for powrev2 (board5): each end
    costs about five correctly rounded pows.
- **`rootn(x, q)`** (1788.1 recommends it): the same root, used forward;
  over the whole line for odd q, over x ≥ 0 for even q, and with a pole
  at 0 for negative q;
- **`pown(x, p)`** (in `ival.c`, an `int` power per interval): x^p for
  every real x. Its bounds are CORE-MATH's pow at the ends, rounded down
  and up. A negative power has a pole at 0: [0, 0] alone is empty, and an
  interval that reaches 0 reaches the infinity on that side.

`ival/test/1788-check.c` checks them in under a second:
- **MPFR:** the rounding operations (mid, wid, rad and the cancel pair)
  are checked against it;
- **by definition, element by element:** the exact operations;
- **against overlap's state, computed another way:** the predicates
  equal, subset and disjoint;
- **overlap itself:** against its converse, overlap(B, A);
- **the interval pairs:** every pair of 24 special ends, plus random and
  subnormal intervals, 4,931,076 results in all;
- **pown:** against MPFR's pow_si at the ends, adding 0 and the limits
  at the pole where the interval reaches them, for 22 powers (among them
  0, 1000, −1000 and the int extremes) on every interval;
- **mulrev:** three checks:
  - containment by exact rationals: c/b in X must be in the result;
  - against the pair: with X the whole line, the result is the hull of
    mulrevpair's two intervals;
  - tightness at the ends, by an oracle that decides membership another
    way: for a point d (each finite end of the pair, the doubles beside
    it, and random points), d is in the set exactly when d·B, computed
    exactly, meets C. mulrev on X = [d, d] must agree. Before this oracle,
    a planted bug that tested meeting X on the outer roundings passed
    every other check and ITF1788;
- **the negative control:** the midpoint taken as the sum of the halves
  must differ. It differs 621 times, because the halves round twice
  below 2^−1021;
- **planted bugs:** eight (two in pown, two in mulrev), each caught by
  this check, and all but the mulrev meeting test by ITF1788 too;
- **sanitizers and Clang:** clean under ASan and UBSan, and passes with
  Clang.

**The constructors** (`ival/ival-text.c`): `ival_nums` makes [l, u] (1788's
numsToInterval), and `ival_text` reads 1788's interval literals, each
bound the exact value rounded outward:
- inf-sup literals: "[1, 2]", "[0.1]", "[1/3, 2/3]", "[0x1.8p-3,]",
  "[entire]";
- uncertain literals: "3.56?1", "2.5?u", "0.0??", "2.500?5e+27".

How the bounds are computed:
- decimal and hexadecimal numbers go through strtod with the rounding mode
  set down or up;
- an uncertain literal's ends are computed exactly in decimal first;
- a rational p/q (up to 400 digits each) is decided exactly: from a
  quotient within a few ulps, each double d beside it is tested by
  d·q ≤ p in big integers.

An optional status gives 1788's signals: 1 for a string that is not a
literal (the empty interval), 2 when the ends may be in either order
after rounding.

`ival/test/text-check.c` checks 60,006 random literals of every form
against MPFR in under a second:
- MPFR reads each number at 4,096 bits, rounded the bound's way;
- for an uncertain literal, it forms m ± r exactly as an integer times a
  power of ten, then rounds once;
- invalid literals must give status 1;
- the first version rounded rationals as p and q each rounded outward.
  That was up to 3 ulps wide, and this check found it. Planted back, it
  passes ITF1788.

**The reductions**, 1788.1's correctly rounded sum, dot product,
sumSquare and sumAbs over numbers, are crsum's (`crsum/crsum.h`): `crsum`
and `crdot` in any of the four rounding directions, sumSquare being
`crdot(x, x, ...)`.

`ival/test/rev-check.c` checks the reverse operations and rootn in a few
seconds:
- **a point oracle.** For a point X = [d, d], the result must be [d, d]
  exactly when f(d) ∈ C, decided in exact arithmetic:
  - sqr, abs and cosh in MPFR at 2,200 bits;
  - pown by d^|p| exactly, and for a negative power by multiplying C's
    ends rather than dividing.
- **the points:** each finite end of the result over the whole line, the
  doubles beside it, and random points. C runs over every pair of 26
  special ends and random intervals, with 21 powers. That is 403,851
  results; 127,633 points are members and 242,618 are not.
- **rootn:** each bound is proved by exact powers to be the double below
  the root with the next one above it.
- **the negative control:** bounds moved one double outward must fail
  that proof.
- **sinrev, cosrev and tanrev:** checked against a reference built
  another way. The set's pieces come from asin, acos and atan plus
  multiples of the period, in MPFR at 2,200 bits; the first and last
  pieces meeting X are found by floor and ceil, and rounded once. It runs
  6,000 random C and X for each function, X from near 0 to 2^1000.
  - It found a bug in the first version: a tan gap two periods long or
    more holds every value. The first version counted its poles as 1 or 2
    by parity, a double too high where neighbouring doubles are 2^482
    apart.
  - ITF1788 has no such test, and misses it.
- **powrev1 and powrev2:** checked against the forward image of pow,
  the other way round from `ival-rev.c`.
  - A double is in the set when the image of B (or A) under pow at it
    meets C.
  - An open gap (a, b) meets the set when the image of the box does. For
    each y the powers of the x in (a, b) form an open interval from a^y
    to b^y. These move continuously with y, so they cover the open
    interval from their least end to their greatest, which lie at the
    ends of y's range because pow is monotone in y.
  - Each power is compared with a double exactly: MPFR's pow rounded to
    64 bits, and its ternary when it equals the double.
  - A result is then proved the tightest: nothing of the set in X below
    it or above it, and something in X within one double of each end.
  - Of 98,304 results, 69,459 are nonempty. X is the whole line, random,
    or within a few doubles of the whole-line result's ends. Lower ends
    moved one double outward must fail the proof, and do 36,773 times.
  - A point check alone would not do: an end rounded inward past an
    irrational end leaves no double between them, so every point agrees.
    Planted bug 39 is that.
- **planted bugs:** nine, each caught. One is the bug the first version
  had: GCC computed an inlined sqrt once for both rounding modes. One
  more, an open end on the mirror's wrong side, passes ITF1788. For
  powrev:
  - 37, the search's test on the power rounded to nearest;
  - 38, the negative side's ends chosen as the positive side's;
  - 39, an end rounded down both ways;
  - 40, a limit taken as reached.

Ten of ITF1788's expected results are not the tightest, and the
converter corrects them, each with a proof run whenever it generates the
program:
- **pownRev of [0, 2^−1074] (and its negative) with power −7:** ITF1788
  wants the lower end 0x1.588cea3f093bcp+153. The root 2^(1074/7) lies
  between 0x1.588cea3f093bdp+153 and the next double, which is ival's
  answer. Proved in exact fractions.
- **six of the trigonometric reverses:** ITF1788 wants them one or two
  ulps wider than the exact ends rounded outward, for example cosRev of
  [−1, −1] within [3.14, 3.15], which is {π}. Proved with mpmath at
  400 bits. Without mpmath those tests are left out and listed, not
  trusted.
- **powRev2 of [1/4, 1/2] and [1/4, 1] with C = [2, ∞]:** ITF1788 wants
  [entire] and [−∞, 0], where both sets are (−∞, −1/2]:
  - x = 1 gives 1, outside C;
  - for x < 1, x^y ≥ 2 exactly when y ≤ ln 2 / ln x. That is greatest at
    the least x, where it is −1/2 since (1/4)^(−1/2) = 2.
  - Their neighbours with C = [2, 4] end at −1/2 as well. The fact used,
    2² = 1/(1/4), is checked in exact fractions.

## IEEE 1788's test suite

`make itf1788-check ITF1788=/path/to/ITF1788` runs ITF1788, the public
test suite for IEEE 1788 (on GitHub; commit b6ee1e2 checked), on
ival. Its tests come from libieeep1788, MPFI, C-XSC and FI_LIB.
`ival/test/itf1788.py` converts the files of a clone into a C program,
and none of them is copied here. `ITF1788_ITL=` names the folder of `.itl`
files directly: CI uses JuliaIntervals' ITF1788.jl, whose `src/itl` holds
the same files with two of their tests corrected (`midRad [nai]`, and
`wid [0, 0]` written as 0, not −0). CI also runs `make ival-sanitize`, all
of ival's checks under ASan and UBSan, and the thread check under
ThreadSanitizer. Decorated tests (`_dec`) are left out,
because ival has no decorations.

Of the bare tests, the 7,451 for operations ival has all pass with the
tight result (2026-10-09). That covers the arithmetic, fma, pow, atan2,
hypot, 24 of the one-argument functions and the operations above,
powRev1 and powRev2 included. The
program runs them all in one call and one at a time, and the two must
agree. For the constructors it also checks the status against the
expected signal (UndefinedOperation 1, PossiblyUndefinedOperation 2).

Numbers must match to the sign of a zero only for inf and sup, where
1788.1 fixes it. ITF1788's own plugins compare the other numbers with ==,
and its MPFI tests write the width of [0, 0] as −0.

The program also lists, with counts, the operations it skipped (439
tests):
- the decorated forms of the constructors, and intervalPart;
- functions 1788.1 does not have: csc, sec, cot and their kin, from
  libieeep1788's own tests.

A literal's bounds are binary64 values rounded to nearest, as ITF1788's
own plugins write them. The first version of the converter rounded them
outward, and 96 tests failed, each on a decimal bound.
