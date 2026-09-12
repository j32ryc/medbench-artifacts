# -*- coding: utf-8 -*-
"""Build a balanced evaluation set for the partial-input baseline experiment.

Method (Poliak et al. 2018; Gururangan et al. 2018): feed a model only part of
the input and see whether it still beats chance. If it does, the benchmark
contains artifacts a model can exploit without the competence the benchmark
claims to measure.

Note the asymmetry, per Feng et al. (2019): a *high* partial-input score proves
the dataset is cheatable, but a low one does not prove it is clean. Only the
positive direction supports a conclusion.

Sampling is stratified by specialty so a result cannot be an artifact of one
bank, and restricted to five-option single-answer items so chance is exactly
20% and accuracy is directly interpretable.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import zipfile

BASE = r"E:\医学大模型\2025年医师题库英文版"
ARCHIVES = ["主治医师总_已加编号.zip", "带编号的json文件（正高）.zip"]

# Eight specialties: large, overwhelmingly single-answer, and spread across
# internal medicine, surgery, nursing, diagnostics and dentistry.
BANKS = [
    "M-332-Pediatrics",
    "M-368-Nursing",
    "M-301-GeneralPractice",
    "M-304-CardiovascularMedicine",
    "M-352-ClinicalLaboratoryTesting",
    "M-317-GeneralSurgery",
    "M-353-Dentistry",
    "M-334-Ophthalmology",
]

ANS = re.compile(r"(?:reference answer|参考答案)\s*[:：]?\s*([A-Ga-g]+)", re.I)
# Options are "A. text" up to the next option letter or end of string.
OPT_SPLIT = re.compile(r"(?:^|\s)([A-G])[\.\uff0e、]\s*")
SHARED = re.compile(r"\[Shared Stem\]|\[共享题干\]|共用题干")
CJK = re.compile(r"[\u4e00-\u9fff]")


def parse_item(stem: str, label: str):
    """Split a raw item into (question_text, {letter: option_text}, answer).

    Returns None when the item cannot be scored unambiguously -- no answer, not
    exactly five options, an answer outside the options, or residual Chinese
    (which would confound a translation-quality effect with an artifact effect).
    """
    m = ANS.search(label) or ANS.search(stem)
    if not m:
        return None
    key = m.group(1).upper()
    if len(key) != 1:
        return None                      # multi-answer: chance level differs
    if CJK.search(stem) or CJK.search(label):
        return None

    parts = OPT_SPLIT.split(stem)
    if len(parts) < 3:
        return None
    question = parts[0].strip()
    options = {}
    for i in range(1, len(parts) - 1, 2):
        letter, text = parts[i], parts[i + 1].strip()
        if letter in options:            # duplicated letter: malformed item
            return None
        options[letter] = text

    if sorted(options) != list("ABCDE"):
        return None
    if key not in options:
        return None
    if not question or len(question) < 15:
        return None
    if any(len(v) < 1 for v in options.values()):
        return None
    return question, options, key


def load_bank(bank: str):
    for arc in ARCHIVES:
        path = os.path.join(BASE, arc)
        if not os.path.exists(path):
            continue
        with zipfile.ZipFile(path) as z:
            for info in z.infolist():
                if info.is_dir() or not info.filename.lower().endswith(".json"):
                    continue
                if os.path.splitext(os.path.basename(info.filename))[0] != bank:
                    continue
                return json.loads(z.read(info).decode("utf-8", errors="replace"))
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-bank", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260902)
    ap.add_argument("--out", default="dataset.json")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out, stats = [], {}
    for bank in BANKS:
        raw = load_bank(bank)
        usable = []
        for it in raw:
            if not isinstance(it, dict):
                continue
            parsed = parse_item(it.get("data", ""), it.get("label", ""))
            if parsed is None:
                continue
            q, opts, key = parsed
            usable.append({
                "bank": bank,
                "question": q,
                "options": opts,
                "answer": key,
                "shared_stem": bool(SHARED.search(it.get("data", ""))),
            })
        # Deduplicate within the bank before sampling, so the same item cannot
        # be drawn twice and inflate agreement between conditions.
        seen, uniq = set(), []
        for u in usable:
            k = re.sub(r"\s+", " ", u["question"]).strip().lower()
            if k not in seen:
                seen.add(k)
                uniq.append(u)
        rng.shuffle(uniq)
        picked = uniq[:args.per_bank]
        out.extend(picked)
        stats[bank] = {"raw": len(raw), "usable": len(uniq), "picked": len(picked)}

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    print(f"{'bank':<34}{'raw':>8}{'usable':>9}{'picked':>8}")
    print("-" * 59)
    for b, s in stats.items():
        print(f"{b:<34}{s['raw']:>8}{s['usable']:>9}{s['picked']:>8}")
    print("-" * 59)
    print(f"{'TOTAL':<34}{sum(s['raw'] for s in stats.values()):>8}"
          f"{sum(s['usable'] for s in stats.values()):>9}"
          f"{len(out):>8}")
    from collections import Counter
    print("\nanswer distribution in sample:",
          dict(sorted(Counter(o["answer"] for o in out).items())))


if __name__ == "__main__":
    main()
