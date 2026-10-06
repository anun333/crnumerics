# ival: interval functions

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
- the C rounding mode and flags are left as they were.

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
