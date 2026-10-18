#!/usr/bin/env python3
"""Paired comparison of two scored runs on the same questions (gsm8k_score.py's OUT.correct.txt): the questions only
A gets right (b) and only B gets right (c), and McNemar's exact two-sided p (binomial, b + c trials at 1/2). Every
run being bit-exact repeatable, b and c are the change itself, not run-to-run noise; what p measures is the sample
of questions.

usage: gsm8k_compare.py A.correct.txt B.correct.txt
"""
import sys
from math import comb


def main():
    a = [int(x) for x in open(sys.argv[1]).read().split()]
    b = [int(x) for x in open(sys.argv[2]).read().split()]
    assert len(a) == len(b), (len(a), len(b))
    only_a = sum(x and not y for x, y in zip(a, b))
    only_b = sum(y and not x for x, y in zip(a, b))
    m = only_a + only_b
    k = min(only_a, only_b)
    p = min(1.0, 2 * sum(comb(m, i) for i in range(k + 1)) / 2 ** m) if m else 1.0
    n = len(a)
    print(f"{n} questions: A {sum(a) / n:.1%}, B {sum(b) / n:.1%} ({(sum(b) - sum(a)) / n:+.1%}); only A right {only_a}, "
          f"only B right {only_b}; McNemar exact p = {p:.3g}")


if __name__ == "__main__":
    main()
