# -*- coding: utf-8 -*-
"""Exact binomial tests for every condition, the options-only condition for
each model, and a per-specialty breakdown.

Reports exact tail probabilities rather than a normal approximation: with
n=800 and p=0.2 the approximation is adequate, but the exact figure costs
nothing and removes a reviewer question. Wilson intervals are used instead of
Wald because the latter misbehaves near the boundaries and these proportions sit
close to the chance rate.
"""
from __future__ import annotations

import io
import json
import math
import os
from collections import Counter, defaultdict

CHANCE = 0.2


def binom_sf(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p), computed exactly."""
    total = 0.0
    for i in range(k, n + 1):
        total += math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i))
    return total


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return centre - half, centre + half


def summarize(name: str, k: int, n: int, out):
    acc = k / n
    lo, hi = wilson(k, n)
    if acc > CHANCE:
        p = binom_sf(k, n, CHANCE)
        tail = "above"
    else:
        p = 1 - binom_sf(k + 1, n, CHANCE)
        tail = "below"
    sigma = math.sqrt(n * CHANCE * (1 - CHANCE))
    z = (k - n * CHANCE) / sigma
    out.write(f"{name:<18}{k:>6}/{n:<5}{acc:>9.4f}  "
              f"[{lo:.4f}, {hi:.4f}]  z={z:>+6.2f}  "
              f"p({tail})={p:.3e}\n")


def preferred(rows):
    """The most frequently chosen letter, and the score from always choosing it."""
    picks = Counter(r["pick"] for r in rows if r["pick"])
    fav = picks.most_common(1)[0][0]
    return fav, sum(1 for r in rows if r["gold"] == fav) / len(rows)


def main():
    out = io.open("stats.txt", "w", encoding="utf-8")
    out.write(f"chance baseline: {CHANCE}\n")
    out.write(f"{'condition':<18}{'correct':>12}{'acc':>9}  {'95% Wilson':<20}"
              f"{'z':>9}  p\n")
    out.write("-" * 88 + "\n")

    per_bank = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    unparsed = Counter()
    options_rows = []

    if os.path.exists("results.json"):
        rows = json.load(io.open("results.json", encoding="utf-8"))
        by = defaultdict(lambda: [0, 0])
        for r in rows:
            c = by[r["condition"]]
            c[0] += r["correct"]
            c[1] += 1
            b = per_bank[r["condition"]][r["bank"]]
            b[0] += r["correct"]
            b[1] += 1
            unparsed[r["condition"]] += r["pick"] is None
        for cond in ("full", "options", "question"):
            if cond in by:
                summarize(cond, by[cond][0], by[cond][1], out)
        options_rows = [r for r in rows if r["condition"] == "options"]

    if os.path.exists("controls.json"):
        rows = json.load(io.open("controls.json", encoding="utf-8"))
        by = defaultdict(lambda: [0, 0])
        for r in rows:
            c = by[r["control"]]
            c[0] += (r["pick"] == r["gold"])
            c[1] += 1
            b = per_bank[r["control"]][r["bank"]]
            b[0] += (r["pick"] == r["gold"])
            b[1] += 1
            unparsed[r["control"]] += r["pick"] is None
        for cond in ("shuffle", "swap"):
            if cond in by:
                summarize(cond, by[cond][0], by[cond][1], out)
    else:
        out.write("(controls.json not present yet)\n")

    out.write("unparsed responses (scored as incorrect): "
              + ", ".join(f"{k} {v}" for k, v in unparsed.items()) + "\n")

    models = [("deepseek-v4-flash", options_rows)] if options_rows else []
    if os.path.exists("multi_options.json"):
        by_model = defaultdict(list)
        for r in json.load(io.open("multi_options.json", encoding="utf-8")):
            if r["condition"] == "options":
                by_model[r["model"]].append(r)
        models += list(by_model.items())
    if models:
        out.write("\n=== options condition by model ===\n")
        for model, rows in models:
            summarize(model, sum(1 for r in rows if r["correct"]), len(rows), out)
            fav, score = preferred(rows)
            n_unparsed = sum(1 for r in rows if r["pick"] is None)
            out.write(f"{'':<18}preferred letter {fav}, always choosing it scores {score:.3f}; "
                      f"unparsed {n_unparsed}\n")

    out.write("\n=== per-specialty ===\n")
    banks = sorted({b for c in per_bank.values() for b in c})
    conds = [c for c in ("full", "options", "question", "shuffle", "swap")
             if c in per_bank]
    out.write(f"{'bank':<34}" + "".join(f"{c:>11}" for c in conds) + "\n")
    out.write("-" * (34 + 11 * len(conds)) + "\n")
    for b in banks:
        row = f"{b:<34}"
        for c in conds:
            k, n = per_bank[c][b]
            row += f"{k/n:>11.3f}" if n else f"{'-':>11}"
        out.write(row + "\n")

    out.close()
    print(io.open("stats.txt", encoding="utf-8").read())


if __name__ == "__main__":
    main()
