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

## IEEE 1788's test suite

`make itf1788-check ITF1788=/path/to/ITF1788` runs ITF1788, the public
test suite for IEEE 1788 (on GitHub; commit b6ee1e2 checked), on
ival. Its tests come from libieeep1788, MPFI, C-XSC and FI_LIB.
`ival/test/itf1788.py` converts the files of a clone into a C program,
and none of them is copied here. Decorated tests (`_dec`) are left out,
because ival has no decorations.

Of the bare tests, the 6,209 for operations ival has all pass with the
tight result (2026-10-09). That covers the arithmetic, fma, pow, atan2,
hypot, 24 of the one-argument functions and the operations above. The
program runs them all in one call and one at a time, and the two must
agree.

Numbers must match to the sign of a zero only for inf and sup, where
1788.1 fixes it. ITF1788's own plugins compare the other numbers with ==,
and its MPFI tests write the width of [0, 0] as −0.

The program also lists, with counts, the operations it skipped (2,182
tests). Those are the part of 1788 ival does not have yet: the other
reverse operations, rootn, textToInterval and numsToInterval, the
reductions, and functions 1788.1 does not require (csc, sec, cot and
their kin).

A literal's bounds are binary64 values rounded to nearest, as ITF1788's
own plugins write them. The first version of the converter rounded them
outward, and 96 tests failed, each on a decimal bound.
