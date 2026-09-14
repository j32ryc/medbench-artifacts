# -*- coding: utf-8 -*-
"""Partial-input results on the public CMExam test set.

Reads cmexam_local.json, written by run_multi.py on the items produced by
build_cmexam.py. Every row carries cmexam_row, the question's position among the
data rows of CMExam's test_with_annotations.csv, so each answer can be matched to
the public file. Needs only files in this repository.
"""
import json
import math
from collections import Counter, defaultdict

CONDITIONS = ("full", "options", "shuffle")


def binom_sf(k: int, n: int) -> float:
    """P(X >= k) for X ~ Binomial(n, 0.2), computed exactly."""
    return sum(math.comb(n, i) * 4 ** (n - i) for i in range(k, n + 1)) / 5 ** n


def wilson(k: int, n: int, z: float = 1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def main():
    runs = defaultdict(list)
    for r in json.load(open("cmexam_local.json", encoding="utf-8")):
        runs[(r["model"], r["condition"])].append(r)
    models = list(dict.fromkeys(m for m, _ in runs))
    reference = next(iter(runs.values()))
    order = [r["cmexam_row"] for r in reference]
    for rows in runs.values():
        assert [r["cmexam_row"] for r in rows] == order, "runs cover different items"

    base = runs.get((models[0], "options")) or runs.get((models[0], "full"))
    if base:
        print(f"items: {len(base)}; answer key {dict(sorted(Counter(r['gold'] for r in base).items()))}")
    print(f"{'model':<13}{'condition':<9}{'correct':>13}{'acc':>7}{'95% CI':>17}{'p vs 0.2':>10}"
          f"{'unparsed':>9}  preferred letter")
    acc = {}
    for m in models:
        for c in CONDITIONS:
            rows = runs.get((m, c))
            if not rows:
                continue
            n = len(rows)
            k = sum(1 for r in rows if r["correct"])
            lo, hi = wilson(k, n)
            picks = Counter(r["pick"] for r in rows if r["pick"])
            fav = picks.most_common(1)[0][0] if picks else "-"
            bound = sum(1 for r in rows if r["gold"] == fav) / n
            unparsed = sum(1 for r in rows if r["pick"] is None)
            acc[(m, c)] = k / n
            p = binom_sf(k, n) if k / n > 0.2 else 1.0
            print(f"{m:<13}{c:<9}{f'{k}/{n}':>13}{k / n:>7.3f}  [{lo:.3f}, {hi:.3f}]{p:>10.1e}"
                  f"{unparsed:>9}  {fav} (always choosing it: {bound:.3f})")
        for c in ("options", "shuffle"):
            if (m, "full") in acc and (m, c) in acc and acc[(m, "full")] > 0.2:
                rho = (acc[(m, c)] - 0.2) / (acc[(m, "full")] - 0.2)
                print(f"{'':<22}rho ({c} vs full) = {rho:.1%}")

    print("\nby medical discipline:")
    for m in models:
        for c in ("options", "shuffle"):
            rows = runs.get((m, c))
            if not rows:
                continue
            per = defaultdict(lambda: [0, 0])
            for r in rows:
                per[r["bank"]][0] += r["correct"]
                per[r["bank"]][1] += 1
            cells = ", ".join(f"{b} {k / n:.3f} (n={n})" for b, (k, n) in sorted(per.items(), key=lambda kv: -kv[1][1]))
            print(f"  {m} {c}: {cells}")


if __name__ == "__main__":
    main()
