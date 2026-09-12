# -*- coding: utf-8 -*-
"""Score a model on the dataset under full-input and partial-input conditions.

Conditions
  full     question + options          -- the benchmark as intended
  options  options only, no question   -- the partial-input baseline
  question question only, no options   -- control: is the question alone enough?

The `options` condition is the experiment. If a model scores well above 20% with
the question removed, the options themselves carry the answer and the benchmark
is measuring something other than the knowledge it claims to.

The `question` condition guards against a specific misreading: a model might
score above chance on `options` simply by preferring longer or more
textbook-sounding strings, independent of any medical content. Running both
separates "the options leak" from "the question leaks".

Answers are extracted from a strict single-letter reply. A response that does
not yield a letter is recorded as `unparsed` rather than silently counted wrong,
because scoring a refusal as an error would understate the model and hide
prompt-format problems.
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

# Credentials live with the gateway project; this tool is a separate
# directory, so point dotenv at that .env explicitly rather than relying
# on cwd discovery.
load_dotenv(os.path.join("D:", os.sep, "Claude Pro",
                         "ai-agent-security-gateway", ".env"))
load_dotenv()

CONDITIONS = ("full", "options", "question")

SYSTEM = ("You are taking a multiple-choice medical exam. Reply with exactly one "
          "letter: A, B, C, D, or E. Output nothing else. If you are unsure, "
          "still choose the single most likely option.")

LETTER = re.compile(r"\b([A-E])\b")


def build_prompt(item: dict, condition: str) -> str:
    opts = "\n".join(f"{k}. {v}" for k, v in sorted(item["options"].items()))
    if condition == "full":
        return f"{item['question']}\n{opts}\n\nAnswer with one letter."
    if condition == "options":
        # No question at all -- the model must pick from the options alone.
        return (f"{opts}\n\nOne of these options is the correct answer to a "
                f"medical exam question you have not been shown. "
                f"Answer with one letter.")
    if condition == "question":
        return (f"{item['question']}\n\nAnswer with the letter (A-E) of the "
                f"correct option. The options are not shown.")
    raise ValueError(condition)


def extract(reply: str) -> str | None:
    if not reply:
        return None
    s = reply.strip().upper()
    if len(s) == 1 and s in "ABCDE":
        return s
    m = LETTER.search(s)
    return m.group(1) if m else None


def make_client(model: str):
    sys.path.insert(0, r"D:\Claude Pro\ai-agent-security-gateway")
    from gateway import providers
    return providers.make_client(model), providers.completion_kwargs(model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-v4-flash")
    ap.add_argument("--dataset", default="dataset.json")
    ap.add_argument("--conditions", nargs="+", default=["full", "options"],
                    choices=CONDITIONS)
    ap.add_argument("--limit", type=int, default=None,
                    help="questions per bank (default: all in dataset)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--sleep", type=float, default=0.0)
    args = ap.parse_args()

    data = json.load(open(args.dataset, encoding="utf-8"))
    if args.limit:
        by_bank = defaultdict(list)
        for d in data:
            by_bank[d["bank"]].append(d)
        data = [d for b in by_bank for d in by_bank[b][:args.limit]]

    client, extra = make_client(args.model)
    results = []
    print(f"model={args.model}  items={len(data)}  conditions={args.conditions}")

    for cond in args.conditions:
        correct = unparsed = 0
        picks = Counter()
        per_bank = defaultdict(lambda: [0, 0])
        for i, item in enumerate(data, 1):
            try:
                r = client.chat.completions.create(
                    model=args.model,
                    messages=[{"role": "system", "content": SYSTEM},
                              {"role": "user", "content": build_prompt(item, cond)}],
                    max_tokens=2000, temperature=0, **extra)
                reply = r.choices[0].message.content or ""
            except Exception as e:
                reply = ""
                print(f"  [{cond}] item {i} error: {type(e).__name__}: {str(e)[:90]}")
            pick = extract(reply)
            ok = (pick == item["answer"])
            if pick is None:
                unparsed += 1
            else:
                picks[pick] += 1
            correct += ok
            per_bank[item["bank"]][0] += ok
            per_bank[item["bank"]][1] += 1
            results.append({"bank": item["bank"], "condition": cond,
                            "gold": item["answer"], "pick": pick, "correct": ok})
            if i % 50 == 0:
                print(f"  [{cond}] {i}/{len(data)}  running acc={correct/i:.3f}")
            if args.sleep:
                time.sleep(args.sleep)

        n = len(data)
        print(f"\n=== {cond} ===")
        print(f"  accuracy : {correct}/{n} = {correct/n:.4f}")
        print(f"  unparsed : {unparsed}")
        print(f"  picks    : {dict(sorted(picks.items()))}")
        for b, (c, t) in sorted(per_bank.items()):
            print(f"    {b:<34} {c:>3}/{t:<4} {c/t:.3f}")

    if args.out:
        json.dump(results, open(args.out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
