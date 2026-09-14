# -*- coding: utf-8 -*-
"""Pick the correct option from the option text alone, with no language model.

A partial-input result from an LLM could reflect what that particular model
happens to know. This asks the same question with a plain classifier trained on
other questions from the corpus. If it also beats chance on the 800 evaluation
items, the signal is in how the options are written.

Training data: every scoreable five-option item in both archives, except
  - items from the eight banks the evaluation sample was drawn from,
  - items sharing three or more identical options with an evaluation item,
    which catches the same question reappearing in another bank, and
  - duplicates by question text.

Two models, both logistic regression over single options; the highest-scoring
option in an item is the prediction:
  surface   length and word-list features, measured relative to the other
            options of the same item; no vocabulary
  text      the surface features plus word unigrams and bigrams (TF-IDF)

Neither model sees the answer letter or the option's position, so shuffling the
options cannot change a prediction. The regularization strength is chosen on
10% of the training questions before the evaluation items are scored.

Needs the source archives (like build_dataset.py) and scikit-learn. Writes
classifier_results.json: for each evaluation item, in dataset.json order, the
bank, the keyed letter, and each model's pick. No item text.
"""
from __future__ import annotations

import json
import math
import os
import random
import re
import time
import zipfile
from collections import Counter, defaultdict

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import build_dataset as bd

LETTERS = list("ABCDE")
# Same word lists as export_features.py, plus negation.
FLAGS = [
    re.compile(r"all of the above|none of the above|both .+ and |以上", re.I),
    re.compile(r"\b(always|never|all|only|must|no|entirely|completely)\b", re.I),
    re.compile(r"\b(may|can|usually|often|generally|sometimes|possible|likely)\b", re.I),
    re.compile(r"\d"),
    re.compile(r"\b(not|without|cannot|none|except)\b", re.I),
]
WORD = re.compile(r"[a-z0-9]+")
SEED = 20260914


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def load_pool() -> list:
    items = []
    for arc in bd.ARCHIVES:
        with zipfile.ZipFile(os.path.join(bd.BASE, arc)) as z:
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
                    if not isinstance(it, dict):
                        continue
                    parsed = bd.parse_item(it.get("data", ""), it.get("label", ""))
                    if parsed:
                        q, opts, key = parsed
                        items.append({"bank": bank, "question": q, "options": opts, "answer": key})
    return items


def option_features(item: dict):
    """Option texts and a dense feature row per option, relative to its item."""
    opts = [norm(item["options"][k]) for k in LETTERS]
    lens = np.array([len(o) for o in opts], dtype=float)
    words = [set(WORD.findall(o.lower())) for o in opts]
    flags = np.array([[bool(rx.search(o)) for rx in FLAGS] for o in opts], dtype=float)
    mean, sd = lens.mean(), lens.std() or 1.0
    order = sorted(range(5), key=lambda i: -lens[i])
    rows = []
    for i in range(5):
        rank = order.index(i)
        overlap = np.mean([len(words[i] & words[j]) / (len(words[i] | words[j]) or 1)
                           for j in range(5) if j != i])
        row = [math.log1p(lens[i]), (lens[i] - mean) / sd, rank / 4,
               float(rank == 0), float(rank == 4), overlap]
        row += list(flags[i])
        # A marker only this option carries stands out more than one they all share.
        row += list(flags[i] * (flags.sum(axis=0) == 1))
        rows.append(row)
    return opts, rows


def matrices(items: list):
    texts, dense, labels = [], [], []
    for it in items:
        opts, rows = option_features(it)
        texts += opts
        dense += rows
        labels += [int(k == it["answer"]) for k in LETTERS]
    return texts, np.array(dense), np.array(labels)


def picks(model, X) -> list:
    scores = model.decision_function(X).reshape(-1, 5)
    return [LETTERS[i] for i in scores.argmax(axis=1)]


def binom_sf(k: int, n: int) -> float:
    """P(X >= k) for X ~ Binomial(n, 0.2), exact."""
    return sum(math.comb(n, i) * 4 ** (n - i) for i in range(k, n + 1)) / 5 ** n


def wilson(k: int, n: int, z: float = 1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def main() -> None:
    t0 = time.time()
    test = json.load(open("dataset.json", encoding="utf-8"))
    test_banks = set(bd.BANKS)
    test_questions = {norm(d["question"]).lower() for d in test}
    by_option = defaultdict(set)
    for t, d in enumerate(test):
        for v in d["options"].values():
            by_option[norm(v).lower()].add(t)

    pool = load_pool()
    dropped = Counter()
    seen, train = set(), []
    for it in pool:
        q = norm(it["question"]).lower()
        if it["bank"] in test_banks:
            dropped["evaluation bank"] += 1
            continue
        if q in test_questions or q in seen:
            dropped["duplicate question"] += 1
            continue
        shared = Counter(t for v in it["options"].values() for t in by_option.get(norm(v).lower(), ()))
        if shared and max(shared.values()) >= 3:
            dropped["shares 3+ options with an evaluation item"] += 1
            continue
        seen.add(q)
        train.append(it)
    print(f"scoreable items in the archives: {len(pool)}; training items: {len(train)}; dropped: {dict(dropped)}")

    rng = random.Random(SEED)
    rng.shuffle(train)
    n_val = len(train) // 10
    val, fit = train[:n_val], train[n_val:]

    fit_texts, fit_dense, fit_y = matrices(fit)
    val_texts, val_dense, _ = matrices(val)
    test_texts, test_dense, _ = matrices(test)
    val_gold = [it["answer"] for it in val]
    test_gold = [d["answer"] for d in test]

    scaler = StandardScaler().fit(fit_dense)
    D = {name: sparse.csr_matrix(scaler.transform(x))
         for name, x in (("fit", fit_dense), ("val", val_dense), ("test", test_dense))}
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=200_000, sublinear_tf=True)
    T = {"fit": vec.fit_transform(fit_texts), "val": vec.transform(val_texts), "test": vec.transform(test_texts)}
    X = {"surface": D, "text": {k: sparse.hstack([T[k], D[k]]).tocsr() for k in D}}

    results = {}
    for name in ("surface", "text"):
        best = None
        for C in (0.1, 1.0, 10.0):
            m = LogisticRegression(C=C, solver="liblinear", max_iter=1000).fit(X[name]["fit"], fit_y)
            acc = np.mean([p == g for p, g in zip(picks(m, X[name]["val"]), val_gold)])
            print(f"  {name:<8} C={C:<5} validation accuracy {acc:.3f}  ({time.time() - t0:.0f}s)")
            if best is None or acc > best[0]:
                best = (acc, C, m)
        test_picks = picks(best[2], X[name]["test"])
        k = sum(p == g for p, g in zip(test_picks, test_gold))
        lo, hi = wilson(k, len(test))
        print(f"{name}: C={best[1]}, evaluation items {k}/{len(test)} = {k / len(test):.3f} "
              f"[{lo:.3f}, {hi:.3f}], p = {binom_sf(k, len(test)):.1e}")
        per_bank = defaultdict(lambda: [0, 0])
        for d, p in zip(test, test_picks):
            per_bank[d["bank"]][0] += p == d["answer"]
            per_bank[d["bank"]][1] += 1
        print("  by bank: " + ", ".join(f"{b[6:]} {c / n:.2f}" for b, (c, n) in per_bank.items()))
        results[name] = test_picks

    rows = [{"bank": d["bank"], "gold": d["answer"],
             "surface_pick": results["surface"][i], "text_pick": results["text"][i]}
            for i, d in enumerate(test)]
    json.dump(rows, open("classifier_results.json", "w", encoding="utf-8"), indent=1)

    llm = [r for r in json.load(open("results.json", encoding="utf-8")) if r["condition"] == "options"]
    if len(llm) == len(test):
        for name in ("surface", "text"):
            agree = np.mean([r["pick"] == p for r, p in zip(llm, results[name])])
            pa = Counter(r["pick"] for r in llm)
            pb = Counter(results[name])
            expected = sum(pa[k] * pb[k] for k in LETTERS) / len(test) ** 2
            both = [(r["correct"], p == d["answer"]) for r, p, d in zip(llm, results[name], test)]
            llm_given_right = np.mean([a for a, b in both if b])
            llm_given_wrong = np.mean([a for a, b in both if not b])
            print(f"deepseek-v4-flash options vs {name}: same pick {agree:.3f} (chance {expected:.3f}); "
                  f"deepseek accuracy where {name} is right {llm_given_right:.3f}, where wrong {llm_given_wrong:.3f}")
    print(f"wrote classifier_results.json ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
