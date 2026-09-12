# -*- coding: utf-8 -*-
"""Exact binomial tests for every condition, plus per-specialty breakdown.

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
from collections import defaultdict

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
    out.write(f"{name:<12}{k:>6}/{n:<5}{acc:>9.4f}  "
              f"[{lo:.4f}, {hi:.4f}]  z={z:>+6.2f}  "
              f"p({tail})={p:.3e}\n")


def main():
    out = io.open("stats.txt", "w", encoding="utf-8")
    out.write(f"chance baseline: {CHANCE}\n")
    out.write(f"{'condition':<12}{'correct':>12}{'acc':>9}  {'95% Wilson':<20}"
              f"{'z':>9}  p\n")
    out.write("-" * 82 + "\n")

    per_bank = defaultdict(lambda: defaultdict(lambda: [0, 0]))

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
        for cond in ("full", "options", "question"):
            if cond in by:
                summarize(cond, by[cond][0], by[cond][1], out)

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
        for cond in ("shuffle", "swap"):
            if cond in by:
                summarize(cond, by[cond][0], by[cond][1], out)
    else:
        out.write("(controls.json not present yet)\n")

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
