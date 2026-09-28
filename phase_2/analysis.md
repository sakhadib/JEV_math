# Phase 2 analysis: CompMath-MCQ (harder dataset, newer models' blind spot?)

Phase 1 (`../analysis.md`, `../baseline/baseline_analysis.md`) showed NLP2025-math was a weak benchmark — eliminable distractors, all-integer golds, 70.6% of golds the smallest option. Phase 2 repeats the exact experiment on **CompMath-MCQ** ([arXiv:2603.03334](https://arxiv.org/abs/2603.03334), March 2026): 1,527 three-option items of higher-level math (linear algebra, calculus, probability; LaTeX-formatted). Same four systems, same zero-shot direct-choice protocol, same harness discipline (fsync-per-item, resumable). All runs complete: 1,527/1,527 rows per system, zero API errors. Deterministic analysis: `analyze_phase2.py` → `stats.json`, `tables/`.

## 0. TL;DR — answering "how are the scores this high?!"

**It is not contamination — at least not for the strongest baseline.** `openai/gpt-oss-20b` was released **August 2025, seven months before this dataset existed**, and it scores 96.79% (1,478/1,527). A model cannot memorize what did not exist during its training. The uncomfortable conclusion: **CompMath-MCQ is simply easy for strong models** — a 33%-floor 3-option MCQ whose items are heavily definitional/recognition-based ("If \(A\) is orthogonal: …", "Which of the following statements … is correct?") and 23.5% of which (359/1,527) are number-substituted instances of just 117 shared question skeletons. New ≠ hard. The dataset tests *knowledge of undergraduate math facts*, and 2025–2026 models have that knowledge cold.

## 1. Headline results

| System | Accuracy | Wilson 95% CI | vs Jev (McNemar, paired n=1,527) | Parse fails | Reasoning tokens |
|---|---|---|---|---|---|
| **gpt-oss-20b** | **96.79%** (1,478/1,527) | [95.78, 97.56] | Jev loses, **p = 0.0013** (discordants 24 vs 53) | 3 | ⚠️ nonzero on **1,393/1,527 (91.2%)**, mean 66.9 |
| **space-bunny-alpha** | 95.35% (1,456/1,527) | [94.18, 96.30] | **tie, p = 0.525** (discordants 41 vs 48) | 11 | 0 |
| **jev-1.13** | 94.89% (1,449/1,527) | [93.67, 95.89] | — | 0 (typed by design) | n/a |
| **llama-3.1-8b-instruct** | 60.12% (918/1,527) | [57.64, 62.55] | Jev wins, p = 1.6e-126 (discordants 563 vs 32) | 64 | 0 |

Random floor here is 33.3% (3 options). Position bias: Jev, gpt-oss, space-bunny all clean (χ² p = 0.84 / 0.94 / 0.77); **llama is again severely biased** (χ² p = 4.5e-43) — its 60% is inflated by letter-preference, not math.

**Ten items were missed by all four systems** — the dataset's genuine hard core: compmath-00140, -00693, -00758, -00978, -01024, -01061, -01272, -01357, -01435, -01444.

## 2. The template finding

Normalizing digits to `#` collapses the 1,527 questions to 1,285 skeletons; **359 items (23.5%) share a skeleton** with at least one other item (top skeleton ×18: "The matrix \(A = \begin{pmatrix} \# & \# \\ \# & \# \end{pmatrix}\) is:"). Accuracy on template-family vs unique items:

| System | Template items | Unique items |
|---|---|---|
| gpt-oss-20b | 97.8% | 96.5% |
| space-bunny-alpha | 95.5% | 95.3% |
| jev-1.13 | 95.0% | 94.9% |
| llama-3.1-8b | 51.5% | 62.8% |

For the three strong systems, templates add ~1–2 points at most — the ease is intrinsic (recognition questions), not driven by template repetition. Curiously, templates are *harder* for llama (51.5% vs 62.8%): the most-templated family (matrix definiteness classification) is a concept it consistently gets wrong.

## 3. Jev's confidence behavior — the streak broke (and that's good science)

Phase 1's headline was "never confidently wrong": 3,993/3,993 correct at p=1.0. On CompMath-MCQ:

- **Degenerate (p=1.0) rows: 949 (62.1%)** — accuracy at p=1.0 is **99.79% (947/949)**, i.e. **Jev's first-ever confidently-wrong items: 2** (plus 2 more with `raw.confidence` ≥ 0.99 below p=1.0; total 4 at raw ≥ 0.99).
- Remaining 76 of 78 errors all carried stated doubt (confidence distribution of errors peaks at 0.5–0.7).
- **ECE 0.0128** (equal-count, 10 bins); mean top-1 0.9412 vs accuracy 0.9489 — still slightly **under**confident (gap −0.008); `raw.confidence` again more conservative (mean 0.9110).
- **Selective prediction still works:** abstaining on the 1/2/5/10% least-confident items lifts accuracy 94.89% → 95.37% / 95.79% / 96.83% / **97.82%** (1,374 retained). At p=1.0 alone, 62.1% coverage at 99.79%.

**What this means.** The harder dataset did exactly what a harder dataset should: it found Jev's first confident failures (2 of 949 certainties, 0.21%) and compressed its deferral gains (99.70% → 97.82% at 90% coverage). The confidence signal degraded gracefully rather than collapsing — a model that is 99.8% reliable when certain and honest when not remains the operationally unique offering in this comparison, since no chat baseline returns any confidence under this protocol.

## 4. gpt-oss-20b: same protocol violation, bigger win

gpt-oss again ignored the no-reasoning instruction (hidden reasoning tokens on 91.2% of items, mean 66.9 tokens) and again won on raw accuracy (96.79%, McNemar p = 0.0013 vs Jev). Its parse discipline *improved* on this dataset (3 empty answers vs 95 in phase 1). The phase-1 conclusion stands: its accuracy belongs to the "hidden-CoT reasoner" category, not to the single-judgment category shared by Jev, space-bunny, and llama. Within the true zero-shot single-pass category, the ranking is **space-bunny 95.35% ≈ Jev 94.89% (tie) > llama 60.12%**.

## 5. Efficiency

| System | Wall-clock (1,527 items) | Latency p50 | Total cost |
|---|---|---|---|
| jev-1.13 | **297 s** | 793 ms | $0.0251 |
| llama-3.1-8b | 463 s | 1,147 ms | $0.0052 |
| gpt-oss-20b | 1,069 s | 1,536 ms | $0.0252 |
| space-bunny-alpha | 1,292 s | 1,395 ms | $0.00 |

Jev remains the fastest and its latency flat (p50 793 ms, same as phase 1's 794 ms despite LaTeX input).

## 6. Cross-phase summary

| Metric | Phase 1: NLP2025-math (7,473 × 4-opt) | Phase 2: CompMath-MCQ (1,527 × 3-opt) |
|---|---|---|
| Jev accuracy | 96.94% | 94.89% (−2.0 pp) |
| Jev accuracy at p=1.0 | 100.00% (3,993/3,993) | 99.79% (947/949) |
| Jev deferral, 90% coverage | 99.70% | 97.82% |
| Best baseline | gpt-oss 98.30% (beats Jev, p=7.5e-9) | gpt-oss 96.79% (beats Jev, p=0.0013) |
| space-bunny vs Jev | 97.11% vs 96.92%, tie (p=0.51) | 95.35% vs 94.89%, tie (p=0.53) |
| llama-3.1-8b | 40.09% + position bias | 60.12% + position bias |

**The story for the paper is now consistent across two datasets:** on raw zero-shot accuracy, Jev sits level with a free frontier stealth model and one to two points behind a reasoning model that cheats the no-reasoning protocol; its differentiators are the typed guarantee (0 parse failures across 9,000 total items vs 43–95 per chat model), the calibrated confidence that supports deferral, and flat sub-second latency at cent-scale cost. Dataset choice moves everyone's absolute numbers but not this ordering.

## 7. Threats specific to phase 2

- **Contamination is possible for Jev and space-bunny** (both postdate or could postdate March 2026) and **impossible for gpt-oss-20b and llama-3.1** (Aug 2025 / Dec 2024). The gpt-oss score is therefore the cleanest evidence that the dataset is intrinsically easy; Jev's and space-bunny's scores cannot be fully cleared of contamination suspicion. Not testable from this data.
- **n = 1,527 is 5× smaller** than phase 1; Wilson intervals are ±1.1–1.4 pp. The Jev-vs-space-bunny tie would need ~10× more items to resolve at this effect size.
- **Two confidently-wrong Jev items** is too few to characterize; they are flagged, not analyzed (ids in `stats.json` — both are among the items worth reading for the paper's error appendix).
- The lm-eval-harness packaging of CompMath-MCQ may predate the arXiv paper; the March 2026 date is the paper's, not necessarily the data's first public appearance.

## Reproduction

```bash
./venv/bin/python phase_2/run_jev.py          # Jev run — already executed
./venv/bin/python phase_2/run_baseline.py     # 3 chat baselines — already executed
./venv/bin/python phase_2/analyze_phase2.py   # regenerates stats.json + tables/, byte-identical
```
