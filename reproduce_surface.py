# -*- coding: utf-8 -*-
"""Recompute the surface-feature table from features.json.

Needs only files in this repository. The original items are not included, so the
numbers are computed from the per-item measurements written by
export_features.py.
"""
import json
import statistics

LETTERS = list("ABCDE")
FEATURES = [
    ("absolute", "Absolute qualifier"),
    ("catch_all", "Catch-all option"),
    ("hedge", "Hedge"),
    ("numeric", "Contains a digit"),
]


def main():
    with open("features.json", encoding="utf-8") as f:
        items = json.load(f)
    n = len(items)
    print(f"items: {n}\n")

    print(f"{'feature':<22}{'in correct':>12}{'base rate':>12}{'ratio':>8}")
    for key, name in FEATURES:
        col = "feat_" + key
        in_correct = sum(1 for x in items if x[col][x["answer"]]) / n
        base = sum(1 for x in items for k in LETTERS if x[col][k]) / (n * 5)
        print(f"{name:<22}{in_correct:>12.4f}{base:>12.4f}{in_correct / base:>8.2f}")

    longest = sum(1 for x in items if x["longest_is_answer"]) / n
    correct_len = statistics.mean(x["opt_len"][x["answer"]] for x in items)
    other_len = statistics.mean(
        x["opt_len"][k] for x in items for k in LETTERS if k != x["answer"]
    )
    print(f"\nlongest option is correct: {longest:.3f} (chance 0.200)")
    print(f"mean length, correct vs distractor: {correct_len:.1f} vs {other_len:.1f}")


if __name__ == "__main__":
    main()
