# -*- coding: utf-8 -*-
"""Convert the CMExam test set into the item format used by run_multi.py.

CMExam (Liu et al., NeurIPS 2023 Datasets and Benchmarks Track) is a public set
of questions from the Chinese National Medical Licensing Examination with
official splits. It is released for academic research only, so no question text
is copied into this repository: download test_with_annotations.csv from
https://github.com/williamliujl/CMExam and pass its path with --csv.

Kept: questions with a single-letter answer and exactly five options A-E. Each
row records cmexam_row, its position among the CSV's data rows, and run_multi.py
copies that into every result so answers can be matched to the public file.
bank is CMExam's "Medical Discipline" annotation.
"""
import argparse
import csv
import json
import re
from collections import Counter

# "A 选项文字": a letter, then a space or punctuation. A line such as "B超检查"
# has no separator and is treated as a continuation of the previous option.
OPTION = re.compile(r"^\s*([A-H])(?:\s+|\s*[\.．、:：]\s*)(.*)$")


def parse_options(field: str):
    options = {}
    for line in field.splitlines():
        m = OPTION.match(line)
        if m:
            if m.group(1) in options:
                return None
            options[m.group(1)] = m.group(2).strip()
        elif options and line.strip():
            last = list(options)[-1]
            options[last] = f"{options[last]} {line.strip()}"
    return options


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="D:/Claude Pro/datasets/CMExam/test_with_annotations.csv")
    ap.add_argument("--out", default="D:/Claude Pro/datasets/CMExam/cmexam_test_single5.json")
    args = ap.parse_args()

    csv.field_size_limit(10 ** 8)
    with open(args.csv, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    items, dropped = [], Counter()
    for i, r in enumerate(rows):
        answer = re.sub(r"[^A-Z]", "", r["Answer"].upper())
        options = parse_options(r["Options"])
        question = r["Question"].strip()
        if len(answer) != 1:
            dropped["more than one correct letter"] += 1
        elif not options or sorted(options) != list("ABCDE") or not all(options.values()):
            dropped["options not exactly A-E"] += 1
        elif answer not in options or not question:
            dropped["answer or question missing"] += 1
        else:
            items.append({"cmexam_row": i, "bank": r["Medical Discipline"], "question": question,
                          "options": options, "answer": answer})

    json.dump(items, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"test questions: {len(rows)}; kept {len(items)}; dropped {dict(dropped)}")
    print("answer key:", dict(sorted(Counter(x["answer"] for x in items).items())))
    print("by discipline:", dict(Counter(x["bank"] for x in items).most_common()))
    print(f"wrote {args.out} (contains question text; not for distribution)")


if __name__ == "__main__":
    main()
