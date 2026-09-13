# -*- coding: utf-8 -*-
"""Export per-item derived features, without the item text.

The source items come from commercially published examination-preparation
material, so redistributing them is not ours to do. Everything the paper claims
about option *form* -- length, lexical markers, which letter is keyed -- is a
function of the text rather than the text itself, so exporting those functions
lets a reader recompute every surface statistic in the paper while leaving the
questions undistributed.

Two structural fields are included as well. shared_stem says whether the item
carries a copied case vignette. case_group is the same for sampled items that
come from one case, and null otherwise. It is a hash of the numbers in the
opening of the vignette (ages, durations, lab values), because those survive
even when the vignette was machine-translated differently for each question.

What this does NOT permit is recomputing the model-based conditions, which need
the actual prompts. For those we ship the per-item model responses instead
(gold, pick, correct), which is sufficient to re-derive every accuracy figure.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
from collections import Counter

DATA = "dataset.json"
OUT = "features.json"

FEATURES = {
    "catch_all": re.compile(r"all of the above|none of the above|both .+ and |以上", re.I),
    "absolute": re.compile(r"\b(always|never|all|only|must|no|entirely|completely)\b", re.I),
    "hedge": re.compile(r"\b(may|can|usually|often|generally|sometimes|possible|likely)\b", re.I),
    "numeric": re.compile(r"\d"),
}
LETTERS = list("ABCDE")

SHARED = re.compile(
    r"[\[【]\s*(?:shared\s+(?:questions?(?:\s+stem)?|stem|topic|title)|共享题干|共用题干)\s*[\]】]"
    r"|共用题干",
    re.I)
QTYPE = re.compile(r"\[\s*(?:single|multiple|multi)[\s-]*choices?(?:\s+questions?)?\s*\]", re.I)
NUM = re.compile(r"\d+(?:\.\d+)?")


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def case_key(d: dict):
    """Bank plus the numbers in the first 300 characters of the case description,
    or None when there are fewer than three numbers to go on."""
    q = d["question"]
    m = SHARED.search(q)
    text = q[m.end():] if m else q
    t = QTYPE.search(text)
    if m and t:
        text = text[:t.start()]
    nums = NUM.findall(norm(text)[:300])
    return (d["bank"], *nums) if len(nums) >= 3 else None


def main() -> None:
    data = json.load(io.open(DATA, encoding="utf-8"))
    keys = [case_key(d) for d in data]
    counts = Counter(k for k in keys if k is not None)

    out = []
    for d, ck in zip(data, keys):
        opts = d["options"]
        lengths = {k: len(norm(opts[k])) for k in LETTERS}
        order = sorted(LETTERS, key=lambda k: -lengths[k])
        row = {
            # A stable id that cannot be inverted back to the question text.
            "id": hashlib.sha256(norm(d["question"]).encode()).hexdigest()[:16],
            "bank": d["bank"],
            "answer": d["answer"],
            "stem_len": len(norm(d["question"])),
            "opt_len": lengths,
            "longest_is_answer": order[0] == d["answer"],
            "answer_len_rank": order.index(d["answer"]) + 1,
            "shared_stem": bool(SHARED.search(d["question"])),
            "case_group": (hashlib.sha256(repr(ck).encode()).hexdigest()[:12]
                           if ck is not None and counts[ck] > 1 else None),
        }
        for name, rx in FEATURES.items():
            row[f"feat_{name}"] = {k: bool(rx.search(opts[k])) for k in LETTERS}
        out.append(row)

    json.dump(out, io.open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"wrote {OUT}: {len(out)} items, no question or option text included")
    print(f"  shared_stem: {sum(r['shared_stem'] for r in out)}, "
          f"items in a case_group: {sum(r['case_group'] is not None for r in out)}")

    # Sanity: confirm no source text leaked into the export.
    blob = io.open(OUT, encoding="utf-8").read()
    fragments = [norm(d["question"])[:40] for d in data]
    fragments += [norm(v)[:25] for d in data for v in d["options"].values() if len(norm(v)) >= 25]
    found = sum(1 for f in fragments if f in blob)
    print(f"text fragments checked: {len(fragments)}, found in the export: {found} (must be 0)")


if __name__ == "__main__":
    main()
