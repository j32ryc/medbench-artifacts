# medbench-artifacts

Code and results for the paper *Answering Without the Question: Partial-Input
Baselines on Chinese Medical Qualification Exam Questions*.

## Overview

Language models are often evaluated on medical exams, and their accuracy is
taken as a sign of clinical knowledge. That assumes a correct answer requires the
knowledge the question was written to test.

This project checks the assumption with the partial-input method from natural
language inference ([Gururangan et al. 2018](https://arxiv.org/abs/1803.02324);
[Poliak et al. 2018](https://arxiv.org/abs/1805.01042)). A model is shown only the
answer options, with the question removed, and we measure whether it still does
better than the 20% expected from guessing. The questions come from preparation
materials for the Chinese attending-physician and senior-professional
qualification exams.

Seven models were run on the same 800 items (chance is 0.200):

| Model | Developer | Question and options | Options only | Options only, shuffled |
|---|---|---|---|---|
| qwen3.8-max-0902 | Alibaba | 0.898 | 0.396 | 0.383 |
| glm-5.2 | Zhipu AI | 0.838 | 0.346 | 0.311 |
| deepseek-v4-flash | DeepSeek | 0.781 | 0.286 | 0.326 |
| qwen2.5:7b | Alibaba | 0.709 | 0.270 | 0.249 |
| gemma2:9b | Google | 0.595 | 0.271 | 0.261 |
| llama3.1:8b | Meta | 0.564 | 0.271 | 0.248 |
| mistral:7b | Mistral | 0.438 | 0.226 | 0.261 |

With the options shuffled, where a preferred letter cannot help, every model is
significantly above chance. Hosted models were called with their reasoning mode
switched off.

`deepseek-v4-flash` was also run with the question only (0.170; 101 answers had
no usable letter and count as wrong, and on the other 699 the accuracy is 0.195)
and with a substitution control described in the paper.

No language model is needed to see the effect. A logistic-regression classifier
that looks only at surface features of the options (length, word overlap, a few
word lists), trained on questions from other specialties, scores 0.286 on the same
800 items. Every language model does better on the items this classifier gets
right.

Many questions in these exams share a case description with the questions next
to it, and the construction pipeline copies the description into each of them.
121 of the 800 sampled items are such questions; on the other 679,
`deepseek-v4-flash` still scores 0.277 with options only (p = 9.3e-7).

## What is included

The questions come from commercially published exam-preparation material and are
not included in this repository.

- `features.json`: per-item measurements (option lengths, word-feature flags,
  keyed letter, whether a shared case description was copied in, and a group id
  for sampled items from the same case) under ids that cannot be reversed.
- `results.json`, `controls.json`: `deepseek-v4-flash`'s answers in every
  condition.
- `extended_results.json`, `extended_glm.json`, `extended_local.json`: the other
  six models' answers, with `_summary.json` files beside them.
- `classifier_results.json`: the two classifiers' choice for each item.
- `pipeline_audit.txt`: counts from the structural audit of the whole corpus.
- `multi_options.json`: an earlier 200-item pilot of the local models, superseded
  by `extended_local.json`.
- The extraction, evaluation and analysis code.

None of the result files contain question text; each row holds only the bank, the
keyed letter, the model's letter and whether it was right.

## Reproducing the numbers

These run without an API key, using only the files in this repository:

```
python reproduce_surface.py     # surface-feature table, from features.json
python stats.py                 # accuracy, intervals and p-values for every model and condition
python shared_stem_split.py     # results with and without shared-case items
python classifier_agreement.py  # do the models succeed where the classifier does?
```

The following need the source items, which are not included. The model runs also
need Ollama or API credentials, and the classifier needs scikit-learn.

```
python build_dataset.py
python export_features.py
python analyze_artifacts.py
python pipeline_audit.py
python option_classifier.py
python run_eval.py     --conditions full options question
python run_controls.py --controls shuffle swap
python run_multi.py    --models gemma2:9b qwen2.5:7b llama3.1:8b mistral:7b --conditions options shuffle full --out extended_local.json --append
python run_multi.py    --models qwen3.8-max-0902 --conditions full options shuffle --out extended_results.json --append
python run_multi.py    --models glm-5.2 --conditions full options shuffle --out extended_glm.json --append
```

`run_multi.py` sends model names containing a colon, such as `qwen2.5:7b`, to a
local Ollama server, and picks a hosted provider from the other names.

## Limitations

A partial-input score above chance shows that a dataset can be exploited. A score
at chance would not show that it is clean, because the probe may be too weak
([Feng et al. 2019](https://arxiv.org/abs/1905.05778)). The figures here are
lower bounds.

The results also do not show that the models lack medical knowledge. A model that
scores 0.898 where its options-only score is 0.383 clearly knows something. The
point is that the benchmark does not separate that knowledge from familiarity
with how the items are written.

The step that copies shared case descriptions depends on the header words used in
these particular books. Six of the 100 banks contain no marked items, and groups
of questions that share options rather than a case description are not handled.

## License

MIT. No exam content is included.
