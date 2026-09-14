# -*- coding: utf-8 -*-
"""The same items in Chinese and in English translation, side by side.

Pairs each local model's answers on the Chinese originals (chinese_local.json,
rows carry sample_index) with its answers on the English translation of the
same items (extended_local.json, rows in dataset.json order). Differences are
tested with an exact McNemar test on the items where exactly one version was
answered correctly. Needs only files in this repository.
"""
import json
import math
from collections import defaultdict


def binom_sf(k: int, n: int) -> float:
    """P(X >= k) for X ~ Binomial(n, 0.2), computed exactly."""
    return sum(math.comb(n, i) * 4 ** (n - i) for i in range(k, n + 1)) / 5 ** n


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def load(path):
    runs = defaultdict(list)
    for r in json.load(open(path, encoding="utf-8")):
        runs[(r["model"], r["condition"])].append(r)
    return runs


def main():
    en, zh = load("extended_local.json"), load("chinese_local.json")
    print(f"{'model':<13}{'condition':<9}{'items':>6}{'English':>9}{'Chinese':>9}"
          f"{'p zh vs 0.2':>13}{'en only':>9}{'zh only':>9}{'McNemar p':>11}{'same pick':>11}")
    for (model, cond), zrows in zh.items():
        erows = en.get((model, cond))
        if not erows:
            continue
        pairs = [(erows[z["sample_index"]], z) for z in zrows]
        assert all(e["gold"] == z["gold"] and e["bank"] == z["bank"] for e, z in pairs), (model, cond)
        n = len(pairs)
        ke = sum(e["correct"] for e, _ in pairs)
        kz = sum(z["correct"] for _, z in pairs)
        b = sum(1 for e, z in pairs if e["correct"] and not z["correct"])
        c = sum(1 for e, z in pairs if z["correct"] and not e["correct"])
        same = sum(1 for e, z in pairs if e["pick"] == z["pick"]) / n
        print(f"{model:<13}{cond:<9}{n:>6}{ke / n:>9.3f}{kz / n:>9.3f}{binom_sf(kz, n):>13.1e}"
              f"{b:>9}{c:>9}{mcnemar_exact(b, c):>11.3f}{same:>11.3f}")


if __name__ == "__main__":
    main()
