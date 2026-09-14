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

### Not a translation effect

The 800 items are Google translations. The Chinese originals of 727 of them were
recovered and checked, and the local models were run on them in Chinese. On the
same 727 items:

| Model | Options only, English | Options only, Chinese | Shuffled, English | Shuffled, Chinese |
|---|---|---|---|---|
| qwen2.5:7b | 0.274 | 0.303 | 0.250 | 0.281 |
| llama3.1:8b | 0.265 | 0.286 | 0.242 | 0.231 |
| gemma2:9b | 0.279 | 0.279 | 0.263 | 0.265 |

None of these English-Chinese differences is significant (exact McNemar test,
p >= 0.099). `mistral:7b` often did not answer the Chinese prompts with a letter,
so it is left out of this comparison; its answers are in `chinese_local.json`.

### A public benchmark: CMExam

The same test on the 6,405 single-answer, five-option questions of the
[CMExam](https://github.com/williamliujl/CMExam) test split, which comes from the
Chinese National Medical Licensing Examination (Liu et al., NeurIPS 2023 Datasets
and Benchmarks Track), in Chinese:

| Model | Question and options | Options only | Options only, shuffled |
|---|---|---|---|
| qwen2.5:7b | 0.812 | 0.302 | 0.296 |
| llama3.1:8b | 0.571 | 0.273 | 0.252 |
| gemma2:9b | 0.517 | 0.262 | 0.253 |

All six partial-input results are above chance (p < 1e-23). CMExam has been public
since 2023, so part of this may come from models having seen the questions during
training.

## What is included

No question text is included: not the commercial preparation material, not its
Chinese originals, and not the CMExam questions (CMExam is released for academic
research only; get it from its own repository).

- `features.json`: per-item measurements (option lengths, word-feature flags,
  keyed letter, whether a shared case description was copied in, and a group id
  for sampled items from the same case) under ids that cannot be reversed.
- `results.json`, `controls.json`: `deepseek-v4-flash`'s answers in every
  condition.
- `extended_results.json`, `extended_glm.json`, `extended_local.json`: the other
  six models' answers, with `_summary.json` files beside them.
- `classifier_results.json`: the two classifiers' choice for each item.
- `chinese_local.json`: the local models' answers on the Chinese originals. Each
  row carries `sample_index`, the item's position in the 800-item sample.
- `cmexam_local.json`: the local models' answers on CMExam. Each row carries
  `cmexam_row`, the question's position among the data rows of CMExam's
  `test_with_annotations.csv`.
- `pipeline_audit.txt`: counts from the structural audit of the whole corpus.
- `multi_options.json`: an earlier 200-item pilot of the local models, superseded
  by `extended_local.json`.
- The extraction, evaluation and analysis code.

Each row of a result file holds only identifiers, the keyed letter, the model's
letter and whether it was right.

## Reproducing the numbers

These run without an API key, using only the files in this repository:

```
python reproduce_surface.py     # surface-feature table, from features.json
python stats.py                 # accuracy, intervals and p-values on the 800 items
python shared_stem_split.py     # results with and without shared-case items
python classifier_agreement.py  # do the models succeed where the classifier does?
python compare_languages.py     # English translation versus Chinese original
python cmexam_stats.py          # CMExam results, overall and by discipline
```

The following need the source items, or for CMExam its public test file. The
model runs also need Ollama or API credentials, and the classifier needs
scikit-learn.

```
python build_dataset.py
python build_dataset_zh.py
python check_zh_alignment.py
python build_cmexam.py --csv test_with_annotations.csv --out cmexam_test_single5.json
python export_features.py
python analyze_artifacts.py
python pipeline_audit.py
python option_classifier.py
python run_eval.py     --conditions full options question
python run_controls.py --controls shuffle swap
python run_multi.py    --models gemma2:9b qwen2.5:7b llama3.1:8b mistral:7b --conditions options shuffle full --out extended_local.json --append
python run_multi.py    --models qwen3.8-max-0902 --conditions full options shuffle --out extended_results.json --append
python run_multi.py    --models glm-5.2 --conditions full options shuffle --out extended_glm.json --append
python run_multi.py    --models gemma2:9b qwen2.5:7b llama3.1:8b mistral:7b --lang zh --dataset dataset_zh.json --conditions full options shuffle --out chinese_local.json --append
python run_multi.py    --models gemma2:9b qwen2.5:7b llama3.1:8b --lang zh --dataset cmexam_test_single5.json --conditions options shuffle full --out cmexam_local.json --append
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

A model that saw a question during training may recognize it from its options
alone. This matters most for CMExam, which is public.

The step that copies shared case descriptions depends on the header words used in
these particular books. Six of the 100 banks contain no marked items, and groups
of questions that share options rather than a case description are not handled.

## License

MIT. No exam content is included.
