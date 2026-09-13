# -*- coding: utf-8 -*-
"""Split the partial-input results by shared stem, and repeat the tests with
one item per case.

About one sampled item in seven carries a case vignette copied in from a shared
stem. This script checks whether those items drive the options-only result, and
whether sampling two questions from the same case affects it. It needs only
files in this repository: features.json for the per-item flags, results.json
and controls.json for the model's answers. All three list the 800 items in the
same order.
"""
import json
import math


def binom_sf(k, n):
    """P(X >= k) for X ~ Binomial(n, 0.2), computed exactly."""
    return sum(math.comb(n, i) * 4 ** (n - i) for i in range(k, n + 1)) / 5 ** n


def fisher_two_sided(a, b, c, d):
    """Fisher's exact test on the 2x2 table [[a, b], [c, d]]."""
    n1, n2, m1 = a + b, c + d, a + c
    total = n1 + n2

    def prob(x):
        return math.comb(n1, x) * math.comb(n2, m1 - x) / math.comb(total, m1)

    observed = prob(a)
    lo, hi = max(0, m1 - n2), min(n1, m1)
    return min(1.0, sum(prob(x) for x in range(lo, hi + 1) if prob(x) <= observed * (1 + 1e-7)))


def main():
    feats = json.load(open("features.json", encoding="utf-8"))
    results = json.load(open("results.json", encoding="utf-8"))
    controls = json.load(open("controls.json", encoding="utf-8"))
    n = len(feats)

    outcomes = {}
    for cond in ("full", "options", "question"):
        rows = [r for r in results if r["condition"] == cond]
        assert len(rows) == n and all(r["gold"] == f["answer"] for r, f in zip(rows, feats)), cond
        outcomes[cond] = [bool(r["correct"]) for r in rows]
    for cond in ("shuffle", "swap"):
        rows = [r for r in controls if r["control"] == cond]
        assert len(rows) == n and all(r["bank"] == f["bank"] for r, f in zip(rows, feats)), cond
        # For swap this is the rate of choosing the inserted option, not accuracy.
        outcomes[cond] = [r["gold"] == r["pick"] for r in rows]

    shared = [bool(f["shared_stem"]) for f in feats]
    ns = sum(shared)
    no = n - ns
    print(f"items with a shared stem: {ns} of {n}\n")
    print(f"{'condition':<10}{'shared stem':>20}{'other items':>20}{'p, other vs 0.2':>17}{'Fisher p':>10}")
    for cond, hits in outcomes.items():
        a = sum(1 for h, s in zip(hits, shared) if h and s)
        c = sum(1 for h, s in zip(hits, shared) if h and not s)
        left = f"{a}/{ns} = {a / ns:.3f}"
        right = f"{c}/{no} = {c / no:.3f}"
        print(f"{cond:<10}{left:>20}{right:>20}{binom_sf(c, no):>17.1e}"
              f"{fisher_two_sided(a, ns - a, c, no - c):>10.3f}")

    groups = set()
    keep = []
    for i, f in enumerate(feats):
        g = f.get("case_group")
        if g is not None:
            if g in groups:
                continue
            groups.add(g)
        keep.append(i)
    print(f"\nkeeping one item per case: {len(keep)} of {n} items")
    for cond in ("options", "shuffle"):
        k = sum(1 for i in keep if outcomes[cond][i])
        print(f"  {cond:<8} {k}/{len(keep)} = {k / len(keep):.3f}   p = {binom_sf(k, len(keep)):.1e}")


if __name__ == "__main__":
    main()
