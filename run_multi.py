# -*- coding: utf-8 -*-
"""Run the partial-input baseline across several models, local and hosted.

The single-model result cannot distinguish "this corpus is exploitable" from
"this model exploits it". Replicating across model families settles that: an
artifact visible to models trained on different data by different organizations
is a property of the corpus.

Local models (names with a colon, such as qwen2.5:7b) are reached through
Ollama, which needs no key and imposes no quota. Hosted models go through the
provider abstraction in the gateway project, which picks DeepSeek, Gemini or
Qwen (DashScope) from the model name.

Conditions, with the same prompts as run_eval.py and run_controls.py:
  full      question and options
  options   options only
  shuffle   options only, permuted. The permutations are drawn exactly as
            run_controls.py draws them over the full dataset, so every item is
            shown in the order deepseek-v4-flash saw it.

A failed call is retried and never scored. Items that still fail get a second
pass at the end of the condition; a condition with any item still missing is
not saved. After several consecutive failures (quota used up, key revoked,
network down) the run stops and exits with an error instead of recording the
failures as wrong answers. Finished conditions are written as soon as they
complete.

  python run_multi.py --models mistral:7b llama3.1:8b gemma2:9b qwen2.5:7b --conditions options --limit 25 --out multi_options.json
  python run_multi.py --models qwen3.8-max-0902 --conditions full options shuffle --out extended_results.json --append
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict

import requests
from dotenv import load_dotenv

from run_controls import make_shuffle

GATEWAY_DIR = os.path.join("D:", os.sep, "Claude Pro", "ai-agent-security-gateway")
load_dotenv(os.path.join(GATEWAY_DIR, ".env"))
load_dotenv()

OLLAMA = "http://127.0.0.1:11434"
CHANCE = 0.2
LETTERS = list("ABCDE")
LETTER_RE = re.compile(r"\b([A-E])\b")
SHUFFLE_SEED = 20260910          # run_controls.py's default seed
RETRIES = 3
MAX_CONSECUTIVE_FAILURES = 5

SYSTEM = ("You are taking a multiple-choice medical exam. Reply with exactly one "
          "letter: A, B, C, D, or E. Output nothing else. If you are unsure, "
          "still choose the single most likely option.")


def is_local(model: str) -> bool:
    """Ollama tags carry a colon (qwen2.5:7b); hosted model ids do not."""
    return ":" in model


def options_block(options: dict) -> str:
    return "\n".join(f"{k}. {options[k]}" for k in LETTERS)


def build_prompt(item: dict, condition: str, options: dict | None = None) -> str:
    if condition == "full":
        return f"{item['question']}\n{options_block(item['options'])}\n\nAnswer with one letter."
    if condition in ("options", "shuffle"):
        return (f"{options_block(options or item['options'])}\n\nOne of these options is the "
                f"correct answer to a medical exam question you have not been shown. "
                f"Answer with one letter.")
    raise ValueError(condition)


def shuffled_options(data: list) -> list:
    """(options, gold) per item, drawn in the same sequence as run_controls.py."""
    rng = random.Random(SHUFFLE_SEED)
    return [make_shuffle(item, rng) for item in data]


def extract(reply: str):
    if not reply:
        return None
    s = reply.strip().upper()
    if len(s) == 1 and s in "ABCDE":
        return s
    m = LETTER_RE.search(s)
    return m.group(1) if m else None


# Local calls must never go through a proxy. On Windows, requests also picks up
# the system proxy from the registry when no *_PROXY variable is set.
OLLAMA_SESSION = requests.Session()
OLLAMA_SESSION.trust_env = False


def ask_local(model: str, prompt: str) -> str:
    r = OLLAMA_SESSION.post(f"{OLLAMA}/api/chat", timeout=180, json={
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": prompt}],
        "stream": False,
        # Keep the model in GPU memory between items; otherwise a server that
        # unloads idle models pays several seconds of loading per question.
        "keep_alive": "30m",
        # Greedy decoding. The length cap only stops runaway output; the letter
        # is parsed from whatever comes back.
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


def call_with_retry(ask, prompt: str) -> str:
    for attempt in range(RETRIES):
        try:
            return ask(prompt)
        except Exception:
            if attempt == RETRIES - 1:
                raise
            time.sleep(2 ** (attempt + 1))


def binom_sf(k: int, n: int, p: float) -> float:
    return sum(math.comb(n, i) * p**i * (1-p)**(n-i) for i in range(k, n+1))


def wilson(k: int, n: int, z: float = 1.96):
    if not n:
        return float("nan"), float("nan")
    p, d = k/n, 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return c-h, c+h


def merge(existing: list, new: list, same) -> list:
    """Replace entries for the same model and condition, keep everything else."""
    return [e for e in existing if not any(same(e, x) for x in new[:1])] + new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--dataset", default="dataset.json")
    ap.add_argument("--conditions", nargs="+", default=["full", "options"],
                    choices=["full", "options", "shuffle"])
    ap.add_argument("--limit", type=int, default=None,
                    help="items per bank (default: all)")
    ap.add_argument("--out", default="multi_results.json")
    ap.add_argument("--append", action="store_true",
                    help="keep other models and conditions already in --out")
    ap.add_argument("--progress-every", type=int, default=100,
                    help="print running accuracy every N items")
    args = ap.parse_args()
    summary_path = os.path.splitext(args.out)[0] + "_summary.json"

    data = json.load(open(args.dataset, encoding="utf-8"))
    shuffled = shuffled_options(data)
    index = list(range(len(data)))
    if args.limit:
        by = defaultdict(list)
        for i, d in enumerate(data):
            by[d["bank"]].append(i)
        index = [i for b in by for i in by[b][:args.limit]]

    records, summary = [], []
    if args.append and os.path.exists(args.out):
        records = json.load(open(args.out, encoding="utf-8"))
    if args.append and os.path.exists(summary_path):
        summary = json.load(open(summary_path, encoding="utf-8"))

    def same_run(a, b):
        return a["model"] == b["model"] and a.get("condition", a.get("cond")) == b.get("condition", b.get("cond"))

    aborted = None
    for model in args.models:
        ask = (lambda p, m=model: ask_local(m, p)) if is_local(model) else make_hosted(model)
        for cond in args.conditions:
            t0 = time.time()
            rows, failed = {}, []
            consecutive = 0
            for pass_no, todo in ((1, index), (2, None)):
                if pass_no == 2:
                    todo, failed = failed, []
                for n_done, i in enumerate(todo, 1):
                    item = data[i]
                    options, gold = shuffled[i] if cond == "shuffle" else (item["options"], item["answer"])
                    try:
                        reply = call_with_retry(ask, build_prompt(item, cond, options))
                        consecutive = 0
                    except Exception as e:
                        failed.append(i)
                        consecutive += 1
                        print(f"  [{model}/{cond}] item {i} failed: {type(e).__name__}: {str(e)[:120]}", flush=True)
                        if consecutive >= MAX_CONSECUTIVE_FAILURES:
                            aborted = f"{model}/{cond}: {consecutive} consecutive failures"
                            break
                        continue
                    pick = extract(reply)
                    rows[i] = {"model": model, "condition": cond, "bank": item["bank"],
                               "gold": gold, "pick": pick, "correct": pick == gold}
                    if pass_no == 1 and n_done % args.progress_every == 0:
                        acc = sum(r["correct"] for r in rows.values()) / len(rows)
                        print(f"  [{model}/{cond}] {n_done}/{len(index)} acc={acc:.3f} "
                              f"({time.time() - t0:.0f}s)", flush=True)
                if aborted or not failed:
                    break
            if aborted:
                break
            if failed:
                print(f"  [{model}/{cond}] {len(failed)} items still failing; condition not saved", flush=True)
                continue

            ordered = [rows[i] for i in index]
            records = merge(records, ordered, same_run)
            json.dump(records, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

            n = len(ordered)
            correct = sum(r["correct"] for r in ordered)
            unparsed = sum(r["pick"] is None for r in ordered)
            picks = Counter(r["pick"] for r in ordered if r["pick"])
            acc = correct / n
            lo, hi = wilson(correct, n)
            p = binom_sf(correct, n, CHANCE) if acc > CHANCE else 1.0
            fav = picks.most_common(1)[0][0] if picks else "-"
            fav_rate = sum(1 for r in ordered if r["gold"] == fav) / n if picks else 0.0
            entry = dict(model=model, cond=cond, correct=correct, n=n, acc=acc, lo=lo, hi=hi,
                         p=p, unparsed=unparsed, fav=fav, fav_rate=fav_rate, secs=time.time() - t0)
            summary = merge(summary, [entry], same_run)
            json.dump(summary, open(summary_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            print(f"\n=== {model} / {cond} ===")
            print(f"  accuracy {correct}/{n} = {acc:.4f}  [{lo:.3f}, {hi:.3f}]"
                  f"  p={p:.2e}  unparsed={unparsed}  ({time.time()-t0:.0f}s)")
            print(f"  favourite letter {fav} (would score {fav_rate:.3f})\n", flush=True)
        if aborted:
            break

    print(f"\nresults in {args.out}, summary in {summary_path}")
    if aborted:
        print(f"STOPPED: {aborted}. The unfinished condition was not saved.")
        sys.exit(2)


if __name__ == "__main__":
    main()
