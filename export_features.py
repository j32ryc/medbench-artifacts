# -*- coding: utf-8 -*-
"""Export per-item derived features, without the item text.

The source items come from commercially published examination-preparation
material, so redistributing them is not ours to do. Everything the paper claims
about option *form* -- length, lexical markers, which letter is keyed -- is a
function of the text rather than the text itself, so exporting those functions
lets a reader recompute every surface statistic in the paper while leaving the
questions undistributed.

What this does NOT permit is recomputing the model-based conditions, which need
the actual prompts. For those we ship the per-item model responses instead
(gold, pick, correct), which is sufficient to re-derive every accuracy figure.
"""
from __future__ import annotations

import hashlib
import io
import json
import re

DATA = "dataset.json"
OUT = "features.json"

FEATURES = {
    "catch_all": re.compile(r"all of the above|none of the above|both .+ and |以上", re.I),
    "absolute": re.compile(r"\b(always|never|all|only|must|no|entirely|completely)\b", re.I),
    "hedge": re.compile(r"\b(may|can|usually|often|generally|sometimes|possible|likely)\b", re.I),
    "numeric": re.compile(r"\d"),
}
LETTERS = list("ABCDE")


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def main() -> None:
    data = json.load(io.open(DATA, encoding="utf-8"))
    out = []
    for d in data:
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
            "shared_stem": d.get("shared_stem", False),
        }
        for name, rx in FEATURES.items():
            row[f"feat_{name}"] = {k: bool(rx.search(opts[k])) for k in LETTERS}
        out.append(row)

    json.dump(out, io.open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"wrote {OUT}: {len(out)} items, no question or option text included")

    # Sanity: confirm no source text leaked into the export.
    blob = io.open(OUT, encoding="utf-8").read()
    leaked = [d for d in data if norm(d["question"])[:40] in blob]
    print(f"items whose stem appears in the export: {len(leaked)} (must be 0)")


if __name__ == "__main__":
    main()
