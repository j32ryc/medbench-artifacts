# -*- coding: utf-8 -*-
"""Audit what the PDF-to-JSON construction pipeline dropped or distorted.

Benchmark papers report how many questions their dataset contains. They rarely
report what the extraction lost, because the intermediate artifacts are gone by
publication time. Those intermediates still exist here, so the loss is
measurable -- which is the point of this script.

Four questions, each answerable from the data alone:

  1. Silent drops. 03_extract_questions.py skips any block without a
     "参考答案" marker (`if "参考答案" not in full_text: continue`) and keeps no
     count. How many blocks were discarded, and were they真 questions?
  2. Header/footer bleed. Lines that are not question starts get appended to the
     current question, so any watermark the cleaning step missed ends up inside
     a stem.
  3. Shared stems. Case vignettes shared across several questions are copied in
     by insert_share_stem.py. Items still carrying the marker show how much of
     the corpus depends on that step having run.
  4. Option integrity. Items whose options are not a clean A-E run cannot be
     scored the way a benchmark assumes.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import zipfile
from collections import Counter, defaultdict

BASE = r"E:\医学大模型\2025年医师题库英文版"
ARCHIVES = ["主治医师总_已加编号.zip", "带编号的json文件（正高）.zip"]

ANS = re.compile(r"(?:reference answer|参考答案)\s*[:：]?\s*([A-Ga-g]+)", re.I)
OPT = re.compile(r"(?:^|\s)([A-G])[\.\uff0e、]\s")
SHARED = re.compile(r"\[Shared Stem\]|\[共享题干\]|共用题干", re.I)
CJK = re.compile(r"[\u4e00-\u9fff]")

# Watermark / navigation text the cleaning pass was meant to remove. Anything
# still present leaked into a question stem.
LEAKS = [
    ("watermark_edu", re.compile(r"浩翔教育|Haoxiang", re.I)),
    ("watermark_qq", re.compile(r"qq\s*[:：]?\s*\d{6,}", re.I)),
    ("exam_paper_hdr", re.compile(r"真题试卷|真题及精解|模拟试卷")),
    ("unit_header", re.compile(r"^Unit\s+\d|第[一二三四五六七八九十]+单元")),
    ("page_marker", re.compile(r"##########|页码\s*\d+")),
    ("platform_ad", re.compile(r"一体化职业考试学习平台|网校课程")),
]


def iter_items():
    for arc in ARCHIVES:
        path = os.path.join(BASE, arc)
        if not os.path.exists(path):
            continue
        with zipfile.ZipFile(path) as z:
            for info in z.infolist():
                if info.is_dir() or not info.filename.lower().endswith(".json"):
                    continue
                bank = os.path.splitext(os.path.basename(info.filename))[0]
                try:
                    data = json.loads(z.read(info).decode("utf-8", errors="replace"))
                except json.JSONDecodeError:
                    continue
                if isinstance(data, dict):
                    data = list(data.values())
                for it in data:
                    if isinstance(it, dict):
                        yield bank, it


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="pipeline_audit.txt")
    args = ap.parse_args()

    seen = set()
    total = 0
    c = Counter()
    leak_counts = Counter()
    leak_examples = defaultdict(list)
    opt_shapes = Counter()
    per_bank_shared = Counter()
    per_bank_total = Counter()

    for bank, it in iter_items():
        stem = it.get("data", "") or ""
        label = it.get("label", "") or ""
        key = re.sub(r"\s+", " ", stem).strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        total += 1
        per_bank_total[bank] += 1

        # --- 1. answerability
        m = ANS.search(label) or ANS.search(stem)
        if not m:
            c["no_answer_marker"] += 1
        elif len(m.group(1)) > 1:
            c["multi_answer"] += 1
        else:
            c["single_answer"] += 1

        # --- 2. leakage of page furniture into the stem
        for name, rx in LEAKS:
            if rx.search(stem):
                leak_counts[name] += 1
                c["any_leak"] += 1 if not any(
                    r.search(stem) for n, r in LEAKS if n < name) else 0
                if len(leak_examples[name]) < 3:
                    leak_examples[name].append(stem[:150])
                break

        # --- 3. shared stems
        if SHARED.search(stem):
            c["shared_stem"] += 1
            per_bank_shared[bank] += 1

        # --- 4. option integrity
        letters = OPT.findall(stem)
        uniq = sorted(set(letters))
        if not letters:
            opt_shapes["none"] += 1
        elif len(letters) != len(uniq):
            opt_shapes["duplicate_letters"] += 1
        elif uniq == list("ABCDE"):
            opt_shapes["clean_ABCDE"] += 1
        elif uniq == sorted("ABCDEFG"[:len(uniq)]):
            opt_shapes[f"clean_{len(uniq)}_options"] += 1
        else:
            opt_shapes["gapped_or_odd"] += 1

        if CJK.search(stem) or CJK.search(label):
            c["residual_cjk"] += 1

    out = io.open(args.out, "w", encoding="utf-8")
    out.write(f"unique items audited: {total}\n\n")

    out.write("=== 1. answerability ===\n")
    for k in ("single_answer", "multi_answer", "no_answer_marker"):
        out.write(f"  {k:<22}{c[k]:>8}{c[k]/total*100:>8.2f}%\n")

    out.write("\n=== 2. page furniture leaked into stems ===\n")
    out.write(f"  {'items with any leak':<22}{c['any_leak']:>8}{c['any_leak']/total*100:>8.2f}%\n")
    for name, n in leak_counts.most_common():
        out.write(f"    {name:<20}{n:>8}{n/total*100:>8.2f}%\n")
    for name, exs in leak_examples.items():
        out.write(f"\n  --- {name} examples ---\n")
        for e in exs:
            out.write(f"    {e}\n")

    out.write("\n=== 3. shared stems ===\n")
    out.write(f"  {'items with marker':<22}{c['shared_stem']:>8}{c['shared_stem']/total*100:>8.2f}%\n")
    out.write("  top banks by shared-stem share:\n")
    ranked = sorted(per_bank_shared.items(),
                    key=lambda kv: -kv[1] / max(per_bank_total[kv[0]], 1))[:10]
    for b, n in ranked:
        t = per_bank_total[b]
        out.write(f"    {b:<40}{n:>6}/{t:<6}{n/t*100:>7.2f}%\n")

    out.write("\n=== 4. option integrity ===\n")
    for k, n in opt_shapes.most_common():
        out.write(f"  {k:<22}{n:>8}{n/total*100:>8.2f}%\n")

    out.write(f"\n=== 5. residual Chinese ===\n")
    out.write(f"  {'items':<22}{c['residual_cjk']:>8}{c['residual_cjk']/total*100:>8.2f}%\n")
    out.close()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
