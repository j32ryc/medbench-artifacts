# -*- coding: utf-8 -*-
"""Check that each aligned Chinese item is the question its English item was
translated from.

Two passes with a local bilingual model (qwen2.5:7b through Ollama, temperature
0): first the whole question and options, then, for pairs the first pass
accepted, the five options alone. Pairs rejected by either pass, and a random
sample of 30 accepted pairs, are written to zh_alignment_review.txt for reading.
That file contains question text and is not distributed; only the counts are
printed. build_dataset_zh.py excludes the pairs that reading showed to differ.

Needs dataset.json, dataset_zh_aligned.json and a local Ollama server.
"""
import json
import random
import time
from collections import Counter

import requests

MODEL = "qwen2.5:7b"
LETTERS = "ABCDE"
SESSION = requests.Session()
SESSION.trust_env = False

WHOLE = ("Below are two multiple-choice questions, one in Chinese and one in English.\n"
         "Is the English one a translation of the Chinese one, with the same five options "
         "in the same order (A to E)? Minor wording differences in translation are fine.\n"
         "Reply with exactly one word: YES or NO.\n\nChinese:\n{zh}\n\nEnglish:\n{en}")
OPTIONS = ("Two lists of five answer options follow, one in Chinese and one in English.\n"
           "Does each English option translate the Chinese option with the same letter?\n"
           "Ignore small wording differences. Reply with exactly one word: YES or NO.\n\n"
           "Chinese:\n{zh}\n\nEnglish:\n{en}")


def options_text(item):
    return "\n".join(f"{k}. {item['options'][k]}" for k in LETTERS)


def whole_text(item):
    return item["question"] + "\n" + options_text(item)


def ask(prompt):
    r = SESSION.post("http://127.0.0.1:11434/api/chat", timeout=180, json={
        "model": MODEL, "messages": [{"role": "user", "content": prompt}],
        "stream": False, "keep_alive": "30m", "options": {"temperature": 0, "num_predict": 5}})
    r.raise_for_status()
    reply = r.json()["message"]["content"].strip().upper()
    return "YES" if reply.startswith("YES") else "NO"


def block(zh, en, note):
    lines = [f"[{zh['sample_index']}] {zh['bank']}  {note}",
             f"   ZH {zh['question']}", f"   EN {en['question']}"]
    lines += [f"      {k}: {zh['options'][k]}  ||  {en['options'][k]}" for k in LETTERS]
    return "\n".join(lines)


def main():
    en = json.load(open("dataset.json", encoding="utf-8"))
    aligned = json.load(open("dataset_zh_aligned.json", encoding="utf-8"))
    t0 = time.time()
    first = {z["sample_index"]: ask(WHOLE.format(zh=whole_text(z), en=whole_text(en[z["sample_index"]])))
             for z in aligned}
    second = {z["sample_index"]: ask(OPTIONS.format(zh=options_text(z), en=options_text(en[z["sample_index"]])))
              for z in aligned if first[z["sample_index"]] == "YES"}
    print(f"pairs: {len(aligned)}; whole-item check {dict(Counter(first.values()))}; "
          f"options-only check on accepted pairs {dict(Counter(second.values()))} ({time.time() - t0:.0f}s)")

    by_index = {z["sample_index"]: z for z in aligned}
    rejected = [i for i in first if first[i] == "NO" or second.get(i) == "NO"]
    accepted = [i for i in first if first[i] == "YES" and second.get(i) == "YES"]
    random.seed(1)
    sample = sorted(random.sample(accepted, min(30, len(accepted))))
    with open("zh_alignment_review.txt", "w", encoding="utf-8") as f:
        f.write(f"{len(rejected)} pairs rejected by a check, then 30 random accepted pairs\n\n")
        for i in rejected:
            f.write(block(by_index[i], en[i], "REJECTED (whole item)" if first[i] == "NO" else "REJECTED (options)") + "\n\n")
        for i in sample:
            f.write(block(by_index[i], en[i], "accepted") + "\n\n")
    excluded = sorted(z["sample_index"] for z in aligned if z["excluded"])
    print(f"pairs to read: {len(rejected)} rejected + {len(sample)} sampled, in zh_alignment_review.txt")
    print(f"currently excluded after reading: {excluded}")


if __name__ == "__main__":
    main()
