# -*- coding: utf-8 -*-
"""Audit what the PDF-to-JSON construction pipeline dropped or distorted.

Benchmark papers report how many questions their dataset contains. They rarely
report what the extraction lost, because the intermediate files are gone by
publication time. Those files still exist here, so the losses can be measured.

Five checks, each answerable from the data alone:

  1. Answerability. Items whose answer key can be parsed, and how many have
     more than one correct letter.
  2. Page furniture. Lines that are not question starts get appended to the
     current question, so a watermark or header the cleaning step missed ends
     up inside a stem.
  3. Shared stems. The insertion step copies a case vignette into every
     question that depends on it and marks the copy 【共享题干】. Machine
     translation renders the marker in several spellings ([Shared Question],
     [Shared question stem], [Shared topic], ...); all of them are counted.
  4. Option integrity. Items whose options are not a clean A-E run cannot be
     scored the way a benchmark assumes. Two cases are broken down further:
     repeated letters in records that hold more than one question, and items
     where the strict pattern finds no options although they are present in
     another form.
  5. Residual Chinese, a lower bound on translation failures.

Only counts are written. Item text is not, because the output file is published.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import zipfile
from collections import Counter

BASE = r"E:\医学大模型\2025年医师题库英文版"
ARCHIVES = ["主治医师总_已加编号.zip", "带编号的json文件（正高）.zip"]

ANS = re.compile(r"(?:reference answer|参考答案)\s*[:：]?\s*([A-Ga-g]+)", re.I)
# Strict option label: a capital letter, a period, then whitespace ("A. text").
OPT = re.compile(r"(?:^|\s)([A-G])[\.\uff0e、]\s")
# The same labels with other punctuation or no space after it ("A.text", "A:").
OPT_LOOSE = re.compile(r"(?:^|[\s\u3000])([A-G])\s*[\.\uff0e、:：\)）]")
OPT_PAREN = re.compile(r"[\(（]\s*([A-G])\s*[\)）]")
SHARED = re.compile(
    r"[\[【]\s*(?:shared\s+(?:questions?(?:\s+stem)?|stem|topic|title)|共享题干|共用题干)\s*[\]】]"
    r"|共用题干",
    re.I)
# Question-type header at the start of each question (single or multiple choice).
QTYPE = re.compile(r"\[\s*(?:single|multiple|multi)[\s-]*choices?(?:\s+questions?)?\s*\]", re.I)
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
                        yield arc, bank, it


def pct(n, total):
    return f"{n:>8}{n / total * 100:>8.2f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="pipeline_audit.txt")
    args = ap.parse_args()

    seen = set()
    total = 0
    c = Counter()
    leak_counts = Counter()
    opt_shapes = Counter()
    merged = Counter()
    no_opts = Counter()
    arc_total, arc_shared = Counter(), Counter()
    bank_total, bank_shared = Counter(), Counter()

    for arc, bank, it in iter_items():
        stem = it.get("data", "") or ""
        label = it.get("label", "") or ""
        key = re.sub(r"\s+", " ", stem).strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        total += 1
        arc_total[arc] += 1
        bank_total[bank] += 1

        # --- 1. answerability
        m = ANS.search(label) or ANS.search(stem)
        if not m:
            c["no_answer_marker"] += 1
        elif len(m.group(1)) > 1:
            c["multi_answer"] += 1
        else:
            c["single_answer"] += 1

        # --- 2. page furniture in the stem (counted under the first pattern that matches)
        for name, rx in LEAKS:
            if rx.search(stem):
                leak_counts[name] += 1
                c["any_leak"] += 1
                break

        # --- 3. shared stems
        shared = bool(SHARED.search(stem))
        if shared:
            c["shared_stem"] += 1
            arc_shared[arc] += 1
            bank_shared[bank] += 1

        # --- 4. option integrity
        letters = OPT.findall(stem)
        uniq = sorted(set(letters))
        if not letters:
            opt_shapes["none"] += 1
            if len(set(OPT.findall(label))) >= 4:
                no_opts["options stored in the answer field"] += 1
            elif len(set(OPT_LOOSE.findall(stem))) >= 4:
                no_opts["other punctuation or no space"] += 1
            elif len(set(OPT_PAREN.findall(stem))) >= 4:
                no_opts["letters in parentheses"] += 1
            else:
                no_opts["no option labels found"] += 1
        elif len(letters) != len(uniq):
            opt_shapes["duplicate_letters"] += 1
            if len(QTYPE.findall(stem)) >= 2:
                merged["with shared-stem marker" if shared else "without shared-stem marker"] += 1
        elif uniq == list("ABCDE"):
            opt_shapes["clean_ABCDE"] += 1
        elif uniq == sorted("ABCDEFG"[:len(uniq)]):
            opt_shapes[f"clean_{len(uniq)}_options"] += 1
        else:
            opt_shapes["gapped_or_odd"] += 1

        if CJK.search(stem) or CJK.search(label):
            c["residual_cjk"] += 1

    out = io.open(args.out, "w", encoding="utf-8")
    out.write(f"unique items audited: {total}\n")

    out.write("\n=== 1. answerability ===\n")
    for k in ("single_answer", "multi_answer", "no_answer_marker"):
        out.write(f"  {k:<40}{pct(c[k], total)}\n")

    out.write("\n=== 2. page furniture in stems ===\n")
    out.write(f"  {'items with any match':<40}{pct(c['any_leak'], total)}\n")
    for name, n in leak_counts.most_common():
        out.write(f"    {name:<38}{pct(n, total)}\n")

    out.write("\n=== 3. shared stems ===\n")
    out.write(f"  {'items with a copied vignette':<40}{pct(c['shared_stem'], total)}\n")
    for arc in ARCHIVES:
        share = arc_shared[arc] / max(arc_total[arc], 1) * 100
        out.write(f"    {arc:<36}{arc_shared[arc]:>8} of {arc_total[arc]:<8}{share:>6.2f}%\n")
    zero = [b for b in bank_total if bank_shared[b] == 0]
    out.write(f"  banks with no marked item: {len(zero)} of {len(bank_total)}\n")
    out.write("  all banks, by share of marked items:\n")
    for b in sorted(bank_total, key=lambda b: (-bank_shared[b] / bank_total[b], b)):
        share = bank_shared[b] / bank_total[b] * 100
        out.write(f"    {b:<52}{bank_shared[b]:>6} of {bank_total[b]:<6}{share:>7.2f}%\n")

    out.write("\n=== 4. option integrity ===\n")
    for k, n in opt_shapes.most_common():
        out.write(f"  {k:<40}{pct(n, total)}\n")
    out.write("  duplicate letters in records holding two or more questions:\n")
    for k in ("with shared-stem marker", "without shared-stem marker"):
        out.write(f"    {k:<38}{pct(merged[k], total)}\n")
    out.write(f"    {'total':<38}{pct(sum(merged.values()), total)}\n")
    out.write("  no options found by the strict pattern, by what the item contains:\n")
    for k in ("other punctuation or no space", "options stored in the answer field",
              "letters in parentheses", "no option labels found"):
        out.write(f"    {k:<38}{pct(no_opts[k], total)}\n")

    out.write("\n=== 5. residual Chinese ===\n")
    out.write(f"  {'items':<40}{pct(c['residual_cjk'], total)}\n")
    out.close()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
