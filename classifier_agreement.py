# -*- coding: utf-8 -*-
"""Do the language models get right the items a text-only classifier gets right?

Reads classifier_results.json (written by option_classifier.py) and every
model's options-only answers from results.json and extended_*.json, all in
dataset.json order. Needs only files in this repository.
"""
import json
import math
import os
from collections import Counter


def fisher_two_sided(a, b, c, d):
    """Fisher's exact test on the 2x2 table [[a, b], [c, d]]."""
    n1, n2, m1 = a + b, c + d, a + c
    total = n1 + n2

    def prob(x):
        return math.comb(n1, x) * math.comb(n2, m1 - x) / math.comb(total, m1)

    observed = prob(a)
    lo, hi = max(0, m1 - n2), min(n1, m1)
    return min(1.0, sum(prob(x) for x in range(lo, hi + 1) if prob(x) <= observed * (1 + 1e-7)))


def options_runs():
    runs = {"deepseek-v4-flash": [r for r in json.load(open("results.json", encoding="utf-8"))
                                  if r["condition"] == "options"]}
    for f in sorted(os.listdir(".")):
        if f.startswith("extended_") and f.endswith(".json") and not f.endswith("_summary.json"):
            for r in json.load(open(f, encoding="utf-8")):
                if r["condition"] == "options":
                    runs.setdefault(r["model"], []).append(r)
    return runs


def main():
    clf = json.load(open("classifier_results.json", encoding="utf-8"))
    n = len(clf)
    runs = options_runs()
    for name in ("surface", "text"):
        right = [c[f"{name}_pick"] == c["gold"] for c in clf]
        k = sum(right)
        print(f"\n{name} classifier: {k}/{n} = {k / n:.3f}")
        print(f"{'model (options)':<20}{'acc, clf right':>16}{'acc, clf wrong':>16}{'Fisher p':>11}"
              f"{'same pick':>11}{'chance':>8}")
        for model, rows in runs.items():
            assert len(rows) == n and all(r["gold"] == c["gold"] for r, c in zip(rows, clf)), model
            ok = [r["pick"] == r["gold"] for r in rows]
            a = sum(1 for o, g in zip(ok, right) if o and g)
            c_ = sum(1 for o, g in zip(ok, right) if o and not g)
            p = fisher_two_sided(a, k - a, c_, (n - k) - c_)
            same = sum(r["pick"] == x[f"{name}_pick"] for r, x in zip(rows, clf)) / n
            pa = Counter(r["pick"] for r in rows)
            pb = Counter(x[f"{name}_pick"] for x in clf)
            chance = sum(pa[L] * pb[L] for L in "ABCDE") / n ** 2
            print(f"{model:<20}{a / k:>16.3f}{c_ / (n - k):>16.3f}{p:>11.1e}{same:>11.3f}{chance:>8.3f}")


if __name__ == "__main__":
    main()
