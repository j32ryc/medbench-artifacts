# -*- coding: utf-8 -*-
"""Recover the Chinese original of each item in dataset.json.

The items were machine-translated from Chinese text files in which each item is
a block: a question line with its options, then a 参考答案 line. The English
archive keeps the same banks in the same order, but some blocks were lost or
split differently on the way, so the two sequences are aligned rather than
zipped. The alignment keys survive translation: the answer letter and the
numbers in the question.

Each sampled English item is mapped to its Chinese block and kept only if the
answer letters agree and the Chinese options parse as A-E. Eight aligned pairs
were then removed by hand (EXCLUDED): check_zh_alignment.py lists the pairs a
local bilingual model judged to differ, and reading them showed that six were a
different question about the same case, one an unrelated question, and one
another version of the question with its options in a different order.

Writes dataset_zh_aligned.json (every aligned pair, with an "excluded" flag) and
dataset_zh.json (the pairs kept). Rows carry sample_index, the item's position
in dataset.json. Both files contain question text and are not distributed.
"""
from __future__ import annotations

import difflib
import io
import json
import os
import re
import unicodedata
import zipfile
from collections import Counter

import build_dataset as bd

ZH_DIRS = ["E:/PythonProject3/translated_ready_input",
           "E:/PythonProject3/final_corrected_output_20250415",
           "E:/PythonProject3/final_corrected_output"]
ZH_NAMES = {
    "M-332-Pediatrics": ["儿科主治医师【代码：332】题库.txt"],
    "M-368-Nursing": ["2025年《【代码：368】护理学》题库.txt"],
    "M-301-GeneralPractice": ["4全科主治医师【代码：301】题库.txt", "全科主治医师【代码：301】题库.txt"],
    "M-304-CardiovascularMedicine": ["心血管【代码：304】主治医师题库.txt"],
    "M-352-ClinicalLaboratoryTesting": ["临床医学检验【代码：352】主治医师题库.txt"],
    "M-317-GeneralSurgery": ["3普通外科【代码：317】主治医师题库.txt", "普通外科【代码：317】主治医师题库.txt"],
    "M-353-Dentistry": ["2口腔医学主治医师【代码：353】题库.txt", "口腔医学主治医师【代码：353】题库.txt"],
    "M-334-Ophthalmology": ["2025年《【代码：334】眼科学》题库.txt"],
}
# Positions in dataset.json whose aligned Chinese block is a different question.
EXCLUDED = {2, 59, 166, 381, 515, 587, 645, 652}

ANS_ZH = re.compile(r"参考答案\s*[:：]\s*([A-Ga-g]+)")
NUM = re.compile(r"\d+(?:\.\d+)?")
OPT_STRICT = re.compile(r"(?:^|\s)([A-E])[\.．、]\s*")
OPT_LENIENT = re.compile(r"(?<![A-Za-z0-9])([A-E])[\.．、]")


def nfkc(s: str) -> str:
    return unicodedata.normalize("NFKC", s)


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", nfkc(s)).strip().lower()


def nums(s: str, k: int = 8) -> tuple:
    return tuple(NUM.findall(nfkc(s)))[:k]


def split_options(text: str):
    """(question, {letter: option}) when the text ends in exactly one A-E run."""
    for rx in (OPT_STRICT, OPT_LENIENT):
        ms = list(rx.finditer(text))
        letters = [m.group(1) for m in ms]
        if len(letters) >= 5 and letters[-5:] == list("ABCDE"):
            run = ms[-5:]
            question = text[:run[0].start()].strip()
            options = {}
            for i, m in enumerate(run):
                end = run[i + 1].start() if i < 4 else len(text)
                options[m.group(1)] = text[m.end():end].strip()
            if question and all(options.values()):
                return question, options
    return None


def parse_zh(path: str) -> list:
    text = io.open(path, encoding="utf-8", errors="replace").read()
    items = []
    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        m = ANS_ZH.search(block)
        if block and m:
            items.append({"text": block[:m.start()].strip(), "answer": m.group(1).upper()})
    return items


def load_en(bank: str) -> list:
    with zipfile.ZipFile(os.path.join(bd.BASE, bd.ARCHIVES[0])) as z:
        for info in z.infolist():
            if os.path.splitext(os.path.basename(info.filename))[0] == bank:
                data = json.loads(z.read(info).decode("utf-8", errors="replace"))
                return [it for it in (data if isinstance(data, list) else data.values()) if isinstance(it, dict)]
    return []


def en_key(item: dict) -> tuple:
    m = bd.ANS.search(item.get("label", "") or "") or bd.ANS.search(item.get("data", "") or "")
    return (m.group(1).upper() if m else None, nums(item.get("data", "") or ""))


def best_alignment(en: list, bank: str):
    en_keys = [en_key(it) for it in en]
    best = None
    for folder in ZH_DIRS:
        for name in ZH_NAMES[bank]:
            path = os.path.join(folder, name)
            if not os.path.exists(path) or os.path.getsize(path) == 0:
                continue
            zh = parse_zh(path)
            matcher = difflib.SequenceMatcher(None, en_keys, [(it["answer"], nums(it["text"])) for it in zh],
                                              autojunk=False)
            mapping = {a + k: b + k for a, b, size in matcher.get_matching_blocks() for k in range(size)}
            if best is None or len(mapping) > len(best[2]):
                best = (path, zh, mapping)
    return best


def main() -> None:
    data = json.load(open("dataset.json", encoding="utf-8"))
    aligned, counts = [], Counter()
    for bank in ZH_NAMES:
        en = load_en(bank)
        path, zh, mapping = best_alignment(en, bank)
        by_text = {}
        for i, it in enumerate(en):
            by_text.setdefault(norm(it.get("data", "") or "")[:80], []).append(i)
        c = Counter()
        for j, d in enumerate(data):
            if d["bank"] != bank:
                continue
            c["sampled"] += 1
            # The archive text is question then options, so the question is a prefix
            # of it; compare at most 60 characters, fewer for a short question.
            head = norm(d["question"])[:60]
            cands = [i for key, ids in by_text.items() if key.startswith(head) for i in ids
                     if en_key(en[i])[0] == d["answer"]]
            if not cands or cands[0] not in mapping:
                c["not aligned"] += 1
                continue
            block = zh[mapping[cands[0]]]
            parsed = split_options(nfkc(block["text"]))
            if block["answer"] != d["answer"] or not parsed:
                c["not aligned"] += 1
                continue
            question, options = parsed
            aligned.append({"sample_index": j, "bank": bank, "question": question, "options": options,
                            "answer": block["answer"], "excluded": j in EXCLUDED})
            c["aligned"] += 1
        print(f"{bank:<34} sampled {c['sampled']:>3}  aligned {c['aligned']:>3}  "
              f"({len(mapping)}/{len(en)} archive items aligned; {os.path.basename(path)})")
        counts.update(c)

    kept = [{k: v for k, v in row.items() if k != "excluded"} for row in aligned if not row["excluded"]]
    json.dump(aligned, open("dataset_zh_aligned.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(kept, open("dataset_zh.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\naligned {counts['aligned']} of {counts['sampled']}; removed by hand {len(aligned) - len(kept)}; "
          f"kept {len(kept)}")
    print("wrote dataset_zh_aligned.json and dataset_zh.json (not for distribution)")


if __name__ == "__main__":
    main()
