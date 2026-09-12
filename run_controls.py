# -*- coding: utf-8 -*-
"""Two controls that decide what the options-only result actually means.

The main experiment showed 28.63% accuracy with the question removed, against a
20% chance baseline. That proves the benchmark is cheatable but leaves two
explanations open, and they have different implications:

  shuffle   Permute the options so the correct answer lands on a different
            letter, then re-ask. If accuracy survives, the model is reading the
            option *text*; if it collapses toward chance, the original result
            was entangled with letter position.

  swap      Replace the correct option with a distractor drawn from a different
            question in the same specialty, so no option is correct. The model
            is still forced to choose. If it disproportionately avoids the
            foreign option, the correct option is textually marked -- the
            artifact is a property of how correct options are written, not of
            the model's medical knowledge.

Both reuse the same items and the same prompt as the main run, so any difference
is attributable to the manipulation alone.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict

from dotenv import load_dotenv

GATEWAY_DIR = os.path.join("D:", os.sep, "Claude Pro", "ai-agent-security-gateway")
load_dotenv(os.path.join(GATEWAY_DIR, ".env"))
load_dotenv()

SYSTEM = ("You are taking a multiple-choice medical exam. Reply with exactly one "
          "letter: A, B, C, D, or E. Output nothing else. If you are unsure, "
          "still choose the single most likely option.")

LETTER = re.compile(r"\b([A-E])\b")
LETTERS = list("ABCDE")


def prompt_for(options: dict) -> str:
    body = "\n".join(f"{k}. {options[k]}" for k in LETTERS)
    return (f"{body}\n\nOne of these options is the correct answer to a "
            f"medical exam question you have not been shown. "
            f"Answer with one letter.")


def extract(reply: str):
    if not reply:
        return None
    s = reply.strip().upper()
    if len(s) == 1 and s in "ABCDE":
        return s
    m = LETTER.search(s)
    return m.group(1) if m else None


def make_shuffle(item, rng):
    """Permute options; return (new_options, new_gold_letter)."""
    texts = [item["options"][k] for k in LETTERS]
    gold_text = item["options"][item["answer"]]
    rng.shuffle(texts)
    new_opts = {LETTERS[i]: texts[i] for i in range(5)}
    new_gold = next(k for k, v in new_opts.items() if v == gold_text)
    return new_opts, new_gold


def make_swap(item, pool, rng):
    """Replace the correct option with a distractor from another item.

    Returns (new_options, foreign_letter). There is no correct answer now; the
    quantity of interest is how often the model lands on the foreign text.
    """
    donor = rng.choice(pool)
    donor_texts = [v for k, v in donor["options"].items()]
    foreign = rng.choice(donor_texts)
    new_opts = dict(item["options"])
    new_opts[item["answer"]] = foreign
    return new_opts, item["answer"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-v4-flash")
    ap.add_argument("--dataset", default="dataset.json")
    ap.add_argument("--controls", nargs="+", default=["shuffle", "swap"],
                    choices=["shuffle", "swap"])
    ap.add_argument("--seed", type=int, default=20260910)
    ap.add_argument("--out", default="controls.json")
    args = ap.parse_args()

    sys.path.insert(0, GATEWAY_DIR)
    from gateway import providers
    client = providers.make_client(args.model)
    extra = providers.completion_kwargs(args.model)

    data = json.load(open(args.dataset, encoding="utf-8"))
    by_bank = defaultdict(list)
    for d in data:
        by_bank[d["bank"]].append(d)

    rng = random.Random(args.seed)
    records = []

    for control in args.controls:
        correct = unparsed = foreign_picked = 0
        picks = Counter()
        per_bank = defaultdict(lambda: [0, 0])
        for i, item in enumerate(data, 1):
            if control == "shuffle":
                opts, gold = make_shuffle(item, rng)
            else:
                pool = [d for d in by_bank[item["bank"]] if d is not item]
                opts, gold = make_swap(item, pool, rng)

            try:
                r = client.chat.completions.create(
                    model=args.model,
                    messages=[{"role": "system", "content": SYSTEM},
                              {"role": "user", "content": prompt_for(opts)}],
                    max_tokens=2000, temperature=0, **extra)
                reply = r.choices[0].message.content or ""
            except Exception as e:
                reply = ""
                print(f"  [{control}] {i} error: {type(e).__name__}: {str(e)[:80]}")

            pick = extract(reply)
            if pick is None:
                unparsed += 1
            else:
                picks[pick] += 1

            if control == "shuffle":
                ok = (pick == gold)
                correct += ok
                per_bank[item["bank"]][0] += ok
            else:
                # `gold` here marks where the foreign text was placed.
                hit = (pick == gold)
                foreign_picked += hit
                per_bank[item["bank"]][0] += hit
            per_bank[item["bank"]][1] += 1

            records.append({"control": control, "bank": item["bank"],
                            "gold": gold, "pick": pick})
            if i % 100 == 0:
                base = correct if control == "shuffle" else foreign_picked
                print(f"  [{control}] {i}/{len(data)} rate={base/i:.3f}", flush=True)

        n = len(data)
        print(f"\n=== {control} ===", flush=True)
        if control == "shuffle":
            print(f"  accuracy after permuting options: {correct}/{n} = {correct/n:.4f}")
            print("  (compare: 0.2863 unshuffled, 0.2000 chance)")
        else:
            print(f"  picked the foreign option: {foreign_picked}/{n} = {foreign_picked/n:.4f}")
            print("  (0.2000 would mean the foreign text is indistinguishable)")
        print(f"  unparsed: {unparsed}")
        print(f"  picks   : {dict(sorted(picks.items()))}")
        for b, (c, t) in sorted(per_bank.items()):
            print(f"    {b:<34} {c:>3}/{t:<4} {c/t:.3f}")

    json.dump(records, open(args.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
