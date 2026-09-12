# -*- coding: utf-8 -*-
"""Run the partial-input baseline across several models, local and hosted.

The single-model result cannot distinguish "this corpus is exploitable" from
"this model exploits it". Replicating across model families settles that: an
artifact visible to models trained on different data by different organizations
is a property of the corpus.

Local models are reached through Ollama, which needs no key and imposes no
quota, so every local model can be run on the full item set. Hosted models are
reached through the same provider abstraction used elsewhere in this project.

  python run_multi.py --models qwen2.5:7b llama3.1:8b mistral:7b
  python run_multi.py --models deepseek-v4-flash --limit 25
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict

import requests
from dotenv import load_dotenv

GATEWAY_DIR = os.path.join("D:", os.sep, "Claude Pro", "ai-agent-security-gateway")
load_dotenv(os.path.join(GATEWAY_DIR, ".env"))
load_dotenv()

OLLAMA = "http://127.0.0.1:11434"
CHANCE = 0.2
LETTERS = list("ABCDE")
LETTER_RE = re.compile(r"\b([A-E])\b")

SYSTEM = ("You are taking a multiple-choice medical exam. Reply with exactly one "
          "letter: A, B, C, D, or E. Output nothing else. If you are unsure, "
          "still choose the single most likely option.")


def is_local(model: str) -> bool:
    """Ollama tags carry a colon (qwen2.5:7b); hosted model ids do not."""
    return ":" in model


def build_prompt(item: dict, condition: str) -> str:
    opts = "\n".join(f"{k}. {item['options'][k]}" for k in LETTERS)
    if condition == "full":
        return f"{item['question']}\n{opts}\n\nAnswer with one letter."
    if condition == "options":
        return (f"{opts}\n\nOne of these options is the correct answer to a "
                f"medical exam question you have not been shown. "
                f"Answer with one letter.")
    raise ValueError(condition)


def extract(reply: str):
    if not reply:
        return None
    s = reply.strip().upper()
    if len(s) == 1 and s in "ABCDE":
        return s
    m = LETTER_RE.search(s)
    return m.group(1) if m else None


def ask_local(model: str, prompt: str) -> str:
    r = requests.post(f"{OLLAMA}/api/chat", timeout=180, json={
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": prompt}],
        "stream": False,
        # Greedy decoding, and a tight cap since a valid answer is one token.
        "options": {"temperature": 0, "num_predict": 2000},
    })
    r.raise_for_status()
    return r.json().get("message", {}).get("content", "")


def make_hosted(model: str):
    sys.path.insert(0, GATEWAY_DIR)
    from gateway import providers
    client = providers.make_client(model)
    extra = providers.completion_kwargs(model)

    def ask(prompt: str) -> str:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": prompt}],
            max_tokens=2000, temperature=0, **extra)
        return r.choices[0].message.content or ""
    return ask


def binom_sf(k: int, n: int, p: float) -> float:
    return sum(math.comb(n, i) * p**i * (1-p)**(n-i) for i in range(k, n+1))


def wilson(k: int, n: int, z: float = 1.96):
    if not n:
        return float("nan"), float("nan")
    p, d = k/n, 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return c-h, c+h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--dataset", default="dataset.json")
    ap.add_argument("--conditions", nargs="+", default=["full", "options"])
    ap.add_argument("--limit", type=int, default=None,
                    help="items per bank (default: all)")
    ap.add_argument("--out", default="multi_results.json")
    args = ap.parse_args()

    data = json.load(open(args.dataset, encoding="utf-8"))
    if args.limit:
        by = defaultdict(list)
        for d in data:
            by[d["bank"]].append(d)
        data = [d for b in by for d in by[b][:args.limit]]

    records, summary = [], []
    for model in args.models:
        ask = (lambda p, m=model: ask_local(m, p)) if is_local(model) else make_hosted(model)
        for cond in args.conditions:
            t0 = time.time()
            correct = unparsed = 0
            picks = Counter()
            for i, item in enumerate(data, 1):
                try:
                    reply = ask(build_prompt(item, cond))
                except Exception as e:
                    reply = ""
                    if i <= 3 or i % 100 == 0:
                        print(f"  [{model}/{cond}] {i} error: {type(e).__name__}: {str(e)[:80]}",
                              flush=True)
                pick = extract(reply)
                ok = (pick == item["answer"])
                correct += ok
                if pick is None:
                    unparsed += 1
                else:
                    picks[pick] += 1
                records.append({"model": model, "condition": cond,
                                "bank": item["bank"], "gold": item["answer"],
                                "pick": pick, "correct": ok})
                if i % 100 == 0:
                    print(f"  [{model}/{cond}] {i}/{len(data)} acc={correct/i:.3f}", flush=True)

            n = len(data)
            acc = correct / n
            lo, hi = wilson(correct, n)
            p = binom_sf(correct, n, CHANCE) if acc > CHANCE else 1.0
            fav = picks.most_common(1)[0][0] if picks else "-"
            fav_rate = sum(1 for d in data if d["answer"] == fav) / n if picks else 0.0
            summary.append(dict(model=model, cond=cond, correct=correct, n=n, acc=acc,
                                lo=lo, hi=hi, p=p, unparsed=unparsed, fav=fav,
                                fav_rate=fav_rate, secs=time.time()-t0))
            print(f"\n=== {model} / {cond} ===")
            print(f"  accuracy {correct}/{n} = {acc:.4f}  [{lo:.3f}, {hi:.3f}]"
                  f"  p={p:.2e}  unparsed={unparsed}  ({time.time()-t0:.0f}s)")
            print(f"  favourite letter {fav} (would score {fav_rate:.3f})\n", flush=True)

    print("\n" + "=" * 78)
    print(f"{'model':<26}{'cond':<9}{'acc':>8}{'95% CI':>18}{'p':>11}{'unparsed':>10}")
    print("-" * 78)
    for s in summary:
        ci = "[{:.3f}, {:.3f}]".format(s["lo"], s["hi"])
        print(f"{s['model']:<26}{s['cond']:<9}{s['acc']:>8.4f}{ci:>18}"
              f"{s['p']:>11.1e}{s['unparsed']:>10}")

    json.dump(records, open(args.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    json.dump(summary, open("multi_summary.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\nwrote {args.out} and multi_summary.json")


if __name__ == "__main__":
    main()
