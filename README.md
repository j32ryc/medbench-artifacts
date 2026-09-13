# medbench-artifacts

Code and results for the paper *What Do Medical Exam Benchmarks Measure? A
Partial-Input Analysis of a 225,554-Item Chinese Licensing Corpus*.

## Overview

Language models are often evaluated on medical licensing exams, and their
accuracy is taken as a sign of clinical knowledge. That assumes a correct answer
requires the knowledge the question was written to test.

This project checks the assumption with the partial-input method from natural
language inference ([Gururangan et al. 2018](https://arxiv.org/abs/1803.02324);
[Poliak et al. 2018](https://arxiv.org/abs/1805.01042)). The model is shown only
the answer options, with the question removed, and we measure whether it still
does better than the 20% expected from guessing.

On 800 items, `deepseek-v4-flash` did:

| Condition | Model sees | Accuracy |
|---|---|---|
| `full` | question and options | 0.781 |
| `options` | options only | 0.286 |
| `shuffle` | options only, shuffled | 0.326 |
| `question` | question only | 0.170 |
| (guessing) | | 0.200 |

In the `question` condition 101 answers had no usable letter and count as wrong.
On the other 699 the accuracy is 0.195.

Four more models, run locally on 200 items each (25 per specialty), also scored
above chance:

| Model | Developer | Items | Accuracy | p |
|---|---|---|---|---|
| gemma2:9b | Google | 200 | 0.310 | 1.5e-4 |
| deepseek-v4-flash | DeepSeek | 800 | 0.286 | 3.5e-9 |
| qwen2.5:7b | Alibaba | 200 | 0.265 | 0.016 |
| llama3.1:8b | Meta | 200 | 0.255 | 0.034 |
| mistral:7b | Mistral | 200 | 0.240 | 0.094 |

`mistral:7b` is not significant. An earlier run of the four local models, whose
per-item output was not kept, differed by at most two items per model, and in
that run `llama3.1:8b` was not significant either (p = 0.069).

Part of the effect can be seen in the option text itself. Correct options are
slightly longer than distractors, and they contain absolute words such as
*always* or *must* 1.71 times as often as options in general. These words are
rare, so that ratio rests on small counts.

Many questions in these exams share a case description with the questions next
to it. The construction pipeline copies the description into each of them. 121
of the 800 sampled items are such questions, and the result holds without them:
on the other 679 items, options-only accuracy is 0.277 (p = 9.3e-7).

## What is included

The questions come from commercially published exam-preparation material and are
not included in this repository.

- `features.json`: per-item measurements (option lengths, word-feature flags,
  keyed letter, whether a shared case description was copied in, and a group id
  for sampled items from the same case) under ids that cannot be reversed.
  Enough to recompute the surface-feature table in the paper, but not to rebuild
  a question.
- `results.json`, `controls.json`: `deepseek-v4-flash`'s answer to each item in
  each condition (key, choice, correct or not).
- `multi_options.json`: the four local models' answers in the options-only
  condition.
- `pipeline_audit.txt`: counts from the structural audit of the whole corpus.
- The extraction and evaluation code, so the corpus can be rebuilt from your own
  copy of the source material.

## Reproducing the numbers

These run without an API key, using only the files in this repository:

```
python reproduce_surface.py   # surface-feature table, from features.json
python stats.py               # accuracy, intervals and p-values for every condition and model
python shared_stem_split.py   # results with and without shared-case items
```

The following need the source items, which are not included. The model runs also
need Ollama or API credentials.

```
python build_dataset.py
python export_features.py
python analyze_artifacts.py
python pipeline_audit.py
python run_eval.py     --conditions full options question
python run_controls.py --controls shuffle swap
python run_multi.py    --models mistral:7b llama3.1:8b gemma2:9b qwen2.5:7b --conditions options --limit 25 --out multi_options.json
```

`run_multi.py` sends model names containing a colon, such as `qwen2.5:7b`, to a
local Ollama server, and other names to an API provider.

## Limitations

A partial-input score above chance shows that a dataset can be exploited. A score
at chance would not show that it is clean, because the probe may be too weak
([Feng et al. 2019](https://arxiv.org/abs/1905.05778)). The figures here are
lower bounds.

The results also do not show that the models lack medical knowledge. A model that
scores 0.781 where the options-only baseline is 0.326 clearly knows something.
The point is that the benchmark does not separate that knowledge from familiarity
with how the items are written.

The step that copies shared case descriptions depends on the header words used in
these particular books. Six of the 100 banks contain no marked items, and groups
of questions that share options rather than a case description are not handled.

## License

MIT. No exam content is included.
