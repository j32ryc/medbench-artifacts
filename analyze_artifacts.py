# -*- coding: utf-8 -*-
"""What, if anything, distinguishes the options the partial-input baseline picks?

A partial-input baseline above chance proves a benchmark is cheatable but says
nothing about the mechanism. This looks for the mechanism with no model calls at
all, by asking whether the correct option is identifiable from surface features
a reader could exploit without knowing any medicine:

  length      the classic multiple-choice artifact -- longest option is correct
  position    letter preference, which would inflate accuracy without any signal
  absolutes   "always/never/all/only" are conventionally wrong-answer markers
  hedges      "may/usually/often" are conventionally right-answer markers
  catch-all   "all of the above" style options
  numeric     digits/units, which read as textbook-precise

Each feature is reported as: how often the correct option has it versus how
often a randomly chosen option would. A ratio near 1.0 means no signal.
"""
from __future__ import annotations

import io
import json
import re
import statistics
from collections import Counter, defaultdict

DATA = "dataset.json"
RESULTS = "results.json"
OUT = "artifacts.txt"

FEATURES = {
    "catch_all": re.compile(r"all of the above|none of the above|both .+ and |以上", re.I),
    "absolute": re.compile(r"\b(always|never|all|only|must|no|entirely|completely)\b", re.I),
    "hedge": re.compile(r"\b(may|can|usually|often|generally|sometimes|possible|likely)\b", re.I),
    "numeric": re.compile(r"\d"),
}


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def main() -> None:
    data = json.load(io.open(DATA, encoding="utf-8"))
    results = json.load(io.open(RESULTS, encoding="utf-8"))

    # results.json is written in condition-major order over the same item
    # sequence, so index within a condition maps back to dataset order.
    per_cond = defaultdict(list)
    for r in results:
        per_cond[r["condition"]].append(r)

    out = io.open(OUT, "w", encoding="utf-8")
    out.write(f"items: {len(data)}\n\n")

    # ---- 1. is the correct option simply the longest? --------------------
    longest_is_correct = 0
    shortest_is_correct = 0
    rank_positions = []
    len_correct, len_distractor = [], []
    for d in data:
        opts = d["options"]
        key = d["answer"]
        lengths = {k: len(norm(v)) for k, v in opts.items()}
        order = sorted(lengths, key=lambda k: -lengths[k])
        if order[0] == key:
            longest_is_correct += 1
        if order[-1] == key:
            shortest_is_correct += 1
        rank_positions.append(order.index(key) + 1)
        len_correct.append(lengths[key])
        len_distractor.extend(v for k, v in lengths.items() if k != key)

    n = len(data)
    out.write("=== 1. option length ===\n")
    out.write(f"  longest option is correct : {longest_is_correct}/{n} = "
              f"{longest_is_correct/n:.4f}  (chance 0.2000)\n")
    out.write(f"  shortest option is correct: {shortest_is_correct}/{n} = "
              f"{shortest_is_correct/n:.4f}  (chance 0.2000)\n")
    out.write(f"  mean length rank of correct: {statistics.mean(rank_positions):.3f} "
              f"(chance 3.000; lower = correct tends to be longer)\n")
    out.write(f"  mean chars, correct   : {statistics.mean(len_correct):.1f}\n")
    out.write(f"  mean chars, distractor: {statistics.mean(len_distractor):.1f}\n\n")

    # ---- 2. lexical features -------------------------------------------
    out.write("=== 2. lexical features (correct vs any-option baseline) ===\n")
    out.write(f"  {'feature':<12}{'in correct':>12}{'in random':>12}{'ratio':>9}\n")
    for name, rx in FEATURES.items():
        c_hit = sum(1 for d in data if rx.search(d["options"][d["answer"]]))
        all_hit = sum(1 for d in data for v in d["options"].values() if rx.search(v))
        p_correct = c_hit / n
        p_random = all_hit / (n * 5)
        ratio = p_correct / p_random if p_random else float("nan")
        out.write(f"  {name:<12}{p_correct:>12.4f}{p_random:>12.4f}{ratio:>9.2f}\n")
    out.write("\n")

    # ---- 3. what did the model actually pick when blind? ----------------
    opts_cond = per_cond.get("options", [])
    if opts_cond:
        picks = Counter(r["pick"] for r in opts_cond)
        gold = Counter(r["gold"] for r in opts_cond)
        out.write("=== 3. options-only condition ===\n")
        out.write(f"  {'letter':<8}{'model picked':>14}{'gold':>8}\n")
        for k in "ABCDE":
            out.write(f"  {k:<8}{picks.get(k,0):>14}{gold.get(k,0):>8}\n")
        acc = sum(r["correct"] for r in opts_cond) / len(opts_cond)
        out.write(f"  accuracy: {acc:.4f}\n\n")

        # If the model always guessed its favourite letter, what would it score?
        fav = picks.most_common(1)[0][0]
        fav_acc = gold.get(fav, 0) / len(opts_cond)
        out.write(f"  most-picked letter: {fav}\n")
        out.write(f"  score if it ALWAYS picked {fav}: {fav_acc:.4f}\n")
        out.write("  (if accuracy is close to this, the result may be letter bias\n"
                  "   rather than any signal read from the option text)\n\n")

    out.close()
    print(io.open(OUT, encoding="utf-8").read())


if __name__ == "__main__":
    main()
