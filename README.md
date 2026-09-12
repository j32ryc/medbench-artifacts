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
| `stem` | question only | 0.170 |
| (guessing) | | 0.200 |

Four more models, run locally on 200 items each, also scored above chance,
three of them significantly:

| Model | Developer | Items | Accuracy | p |
|---|---|---|---|---|
| gemma2:9b | Google | 200 | 0.310 | 1.5e-4 |
| deepseek-v4-flash | DeepSeek | 800 | 0.286 | 3.5e-9 |
| qwen2.5:7b | Alibaba | 200 | 0.265 | 0.016 |
| llama3.1:8b | Meta | 200 | 0.255 | 0.034 |
| mistral:7b | Mistral | 200 | 0.240 | 0.094 |

Part of the effect can be seen in the option text itself. Correct options are
slightly longer than distractors, and they contain absolute words such as
*always* or *must* 1.71 times as often as options in general. These words are
rare, so that ratio rests on small counts.

## What is included

The questions come from commercially published exam-preparation material and are
not included in this repository.

- `features.json`: per-item measurements (option lengths, word-feature flags,
  keyed letter) under ids that cannot be reversed. Enough to recompute the
  surface-feature table in the paper, but not to rebuild a question.
- `results.json`, `controls.json`, `multi_results.json`: each model's answer to
  each item (key, choice, correct or not). Enough to recompute the accuracy
  figures.
- The extraction and evaluation code, so the corpus can be rebuilt from your own
  copy of the source material.

## Reproducing the numbers

These run without an API key, using only the files in this repository:

```
python reproduce_surface.py   # surface-feature table, from features.json
python stats.py               # accuracy, intervals and p-values, from the result files
```

The following need the source items, which are not included. The model runs also
need Ollama or API credentials.

```
python build_dataset.py
python pipeline_audit.py
python run_eval.py     --conditions full options stem
python run_controls.py --controls shuffle swap
python run_multi.py    --models mistral:7b llama3.1:8b gemma2:9b qwen2.5:7b
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

## License

MIT. No exam content is included.
