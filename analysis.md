# Analysis: Jev-1.13 on NLP2025-math

All numbers below are produced by `analyze.py` (deterministic; seeds fixed) from `results.jsonl` (7,473 rows, the run of record, unmodified). Machine-readable tables are in `tables/`, figures in `figures/`, per-item data in `derived.csv`, all scalar results in `stats.json`.

## 0. Summary of findings

- **The most surprising finding: Jev is never confidently wrong.** Of 229 errors, **zero** occurred on items where the model assigned probability 1.0 to its answer (0/229, 0.0% of errors), and zero where the secondary `raw.confidence` signal was ≥ 0.99. On the 3,993 items (53.4% of the dataset) where Jev returned a degenerate {1.0, 0, 0, 0} distribution, accuracy was **exactly 100%** (3,993/3,993). Maximum confidence is a perfect correctness signal on this dataset.
- **The model is *under*confident on average.** Mean top-1 probability 0.9455 vs accuracy 0.9694 — a −2.39-point gap (and −4.40 points for `raw.confidence`, mean 0.9253). Where it expresses doubt, it is still usually right.
- **Accuracy is 96.94% (7,244/7,473)** [Wilson 95% CI: 96.52%, 97.30%], at a total cost of **$0.1296** and **8.3 minutes** of wall-clock — 14.98 items/s, i.e. the run sat exactly on the harness's 15 req/s pacing cap, not on a model limit.
- **Latency is constant** (~800 ms, sd 81 ms) and **independent of input length** (Pearson r = −0.015, p = 0.19, n = 7,471) and **identical for correct and incorrect answers** (medians 793.9 vs 795.2 ms; Mann–Whitney U = 830,586, p = 0.97). This is consistent with the "single judgment, no generation" System One framing.
- **Abstention works extremely well:** deferring the 10% least-confident items raises accuracy from 96.94% to **99.70%** on the retained 6,726 items. A threshold at top-1 = 1.0 alone covers 53.4% of the dataset at 100% accuracy.
- **The returned probabilities are coarse but real:** only 114 distinct values, all consistent with 2-decimal rounding (smallest non-zero = 0.01), and 53.4% of rows are fully degenerate. Yet the distribution sums to 1 within rounding on all but 22 rows (0.29%), never assigns 0 to the gold option (0/7,473), and is better calibrated (ECE 0.0239) than the separate `raw.confidence` field (ECE 0.0440).
- **Errors cluster where you'd expect intuition to fail:** tightly-spaced options (93.8% accuracy in the lowest option-spread quartile vs 98.7% in the highest), mid-ranked gold values (91.2% when the gold is neither the smallest nor largest option), more numeric literals (92.9% top quartile), longer questions (94.6% top quartile). 71.2% of errors (163/229) picked the numerically *nearest* distractor to the gold value — arithmetic slips, not wrong approaches.
- **No position bias** (χ²(3) = 0.30, p = 0.960) and no mid-run drift (split-half 96.60% vs 97.27%, two-proportion z = −1.68, p = 0.093).

**If you read nothing else:** Jev-1.13 answered 7,473 math word problems as single 800-millisecond multiple-choice judgments for 13 cents total, got 96.94% right, and — most unusually for an ML system — its maximum-confidence state has *never* been observed wrong on this dataset: 3,993/3,993 items at probability 1.0 were correct. Its errors are predominantly near-miss arithmetic slips on multi-step problems, and its uncertainty is informative enough that a trivial deferral rule pushes effective accuracy to 99.7% at 90% coverage.

## 1. Data integrity and analysis population

`results.jsonl` contains 7,473 lines; **0 failed to parse**, **0 duplicate ids**, **0 rows with `status != "ok"`** — no rows were excluded for any reason. The retry distribution shows 7,471 rows completed on attempt 1 and 2 rows needed a second attempt (0.027% retry rate). A **single resolved model id** (`typesafe/jev-1.13-20260917`) covers all rows — no mid-run model change. Schema completeness is perfect (0 missing/null for all 21 documented fields). All `gold` ∈ {0,1,2,3}, all rows have exactly 4 choices, and there are **0 duplicate-option rows**, so argmax accuracy is unambiguous everywhere. Timestamps span 2026-09-27T22:43:47Z → 22:52:06Z (499 s wall-clock).

**Internal-consistency caveat (8 rows).** Recomputing fields from `probs` disagrees with the recorded fields on 8 rows (8 `predicted_index` mismatches, 8 `predicted_value`, 7 `correct`; 0 for confidence/margin/entropy). These are exact-probability *ties* (possible because values are 2-decimal): the harness recorded the API's own `choice` field (with the API's tie-break), while the analysis re-derives argmax with A-before-D tie-breaking. The recorded headline was 96.92% (7,243/7,473); the recomputed value is 96.94% (7,244/7,473). **All downstream numbers use recomputed values**, so the analysis headline differs from `experiment.md`'s by one item (+0.01 pp). Both are reported here; nothing was silently repaired.

**Analysis population: N = 7,473** for everything below.

**What this means.** The run of record is clean: complete, single-model, essentially retry-free, and every derived quantity is internally consistent except 8 tie rows whose handling is now explicit. The one-item headline shift is a tie-break artifact, not a data problem.

## 2. Headline performance and baselines

| Metric | Value | n |
|---|---|---|
| **Accuracy** | **96.94%** [Wilson 95% CI 96.52%–97.30%] | 7,244/7,473 |
| Uniform random (analytic) | 25% | — |
| Always predict A | 25.10% | 7,473 |
| Always predict B | 24.70% | 7,473 |
| Always predict C | 25.05% | 7,473 |
| Always predict D | 25.14% (best fixed letter) | 7,473 |
| Accuracy excluding duplicate-option rows | identical (no such rows) | 7,473 |

(`tables/headline.csv`.) Gold letters are near-uniform (A 25.1%, B 24.7%, C 25.1%, D 25.1%), so all trivial baselines sit at ≈25%.

**What this means.** Jev clears the trivial baselines by ~72 points. That is expected for any competent model; the meaningful comparisons — against chat LLMs on identical items — are explicitly out of scope (§10). One dataset property worth flagging: **every gold value is an integer** (7,473/7,473), and distractors are frequently absurd in magnitude, which makes elimination strategies available (see §7.1 and §10).

## 3. The nature of the returned probabilities

This section gates the calibration analysis, so it is reported first.

**Quantization.** Across all 29,892 assigned probabilities (7,473 rows × 4 options) there are only **114 distinct values**, every one consistent with 2-decimal rounding; the smallest non-zero value is 0.01. The distribution of values is in `figures/prob_values_hist.png` (log-count). The probabilities are **not** full-precision floats — they are percentages.

**Degeneracy.** 53.43% of rows (3,993/7,473) have top-1 probability exactly 1.0; 65.06% (4,863/7,473) are ≥ 0.99. 84.99% of rows (6,352/7,473) assign exactly 0.0 to at least one option. Distribution of non-zero-option count per row: 1 option — 3,993 rows; 2 — 1,512; 3 — 847; 4 — 1,121.

**Normalization.** `sum(probs)` ranges from 0.99 to 1.00; 22 rows (0.29%) deviate from 1 by more than 1e-6 — all consistent with independent 2-decimal rounding of each option.

**The two confidence signals.** `raw.answers.answer.confidence` is a second scalar, present on all rows, with 94 distinct values in [0.03, 1.00]. It **disagrees with top-1(probs) on 3,980 rows (53.3%)** — always downward (mean 0.9253 vs 0.9455) — yet the two are nearly collinear (Pearson r = 0.9993; Spearman ρ = 0.9299). `raw.confidence` also takes values *below* the smallest possible top-1 in a 4-way distribution (down to 0.03, vs top-1 minimum 0.27), so it is not simply the max of the returned distribution; it appears to be a separate, more conservative self-assessment. Both are carried through §4–§5.

**Verdict.** The returned distribution is **coarse but informative**: quantized to whole percents, majority-degenerate, but normalized and never assign 0 to the gold option (0/7,473 rows — so NLL is finite and the distribution has real support). It supports calibration analysis, with the caveat that resolution below 0.01 does not exist and that 53% of rows carry no distributional information at all. The `raw.confidence` field is a distinct, more conservative signal — and §4 shows it is the *worse*-calibrated of the two.

**What this means.** Jev's "calibrated probabilities" are percentages, not floats, and most of the time the model commits fully. That limits what calibration analysis can resolve — but what remains is genuinely usable, and the existence of two redundant-but-different confidence signals is itself a finding about the API's behavior.

## 4. Calibration

Caveat carried from §3: with 53% of mass at exactly 1.0 and 2-decimal quantization, ECE values below ~0.02 are near the floor of what the binning can detect. Numbers below are real but should not be over-interpreted past the second decimal.

| Score | Binning | ECE | MCE |
|---|---|---|---|
| top-1(probs) | equal-count (10 bins) | **0.0239** | 0.0904 |
| top-1(probs) | equal-width (10 bins) | 0.0241 | 0.2867 |
| raw.confidence | equal-count (10 bins) | 0.0440 | 0.2129 |
| raw.confidence | equal-width (10 bins) | 0.0440 | 0.4271 |

Reliability diagrams: `figures/reliability_top1_{eqcount,eqwidth,zoom}.png`, same for `rawconf`; per-bin tables in `tables/calibration_*.csv`. Equal-count bins degenerate (most bins sit at ~1.0), so the **0.9–1.0 zoomed diagrams** are the informative ones; bin counts are annotated on every plot. The zoom shows the only meaningful miscalibration: items with top-1 in [0.9, 1.0) are *more* accurate than their stated confidence.

- **Overconfidence gap (mean confidence − accuracy): −0.0239 for top-1; −0.0440 for raw.confidence.** The sign is negative — the model is **underconfident** on average. raw.confidence is roughly twice as underconfident.
- **Brier score (multiclass, one-hot vs 4-option distribution): 0.0471.** Binary Brier on (top-1, correct): 0.0230, decomposed (equal-count bins) into reliability 0.00181, resolution 0.00688, uncertainty 0.0297.
- **NLL of the gold option: 0.0941** with clipping ε = 1e-3. The ε matters less than usual here because **zero rows assigned the gold option probability 0.0** (0/7,473) — the model's distribution never rules out the right answer entirely, even on the items it gets wrong.
- Confidence histograms split by correctness: `figures/conf_hist_correct_incorrect.png` (log count) — errors concentrate in the 0.3–0.8 band; corrects pile up at 1.0.

**What this means.** By the numbers, top-1(probs) is the better-calibrated score and `raw.confidence` is a strictly more pessimistic version of it — for thresholding and quoting confidence, use the distribution, not the scalar. The headline calibration fact is not ECE's magnitude but its *sign*: Jev's residual uncertainty is honest and then some. Combined with §5's zero confidently-wrong rate, the picture is a model whose probability 1.0 means "certain and right" and whose sub-1.0 probabilities understate its accuracy.

## 5. Selective prediction and deferral

Risk–coverage curves for all four candidate scores are in `figures/risk_coverage.png` (log-error y-axis); summary in `tables/selective.csv`.

| Score | AURC | AUROC [95% bootstrap CI, 2,000 resamples] |
|---|---|---|
| margin | 0.001449 | 0.9693 [0.9636, 0.9746] |
| raw.confidence | 0.001501 | 0.9678 [0.9617, 0.9731] |
| top-1(probs) | 0.001509 | 0.9675 [0.9615, 0.9729] |
| −entropy | 0.001801 | 0.9584 [0.9516, 0.9646] |

All four scores separate correct from incorrect well (AUROC ≈ 0.96–0.97; CIs overlap). Margin is nominally best on both metrics; the differences are within bootstrap noise.

**Deferral table (top-1 probability; full tables for all scores in `tables/deferral_*.csv`):**

| Threshold | Coverage | Retained accuracy | Deferred n | Error rate among deferred |
|---|---|---|---|---|
| ≥ 0.48 | 99.0% (7,399) | 97.42% | 74 | 51.4% |
| ≥ 0.65 | 95.0% (7,100) | 98.76% | 373 | 37.8% |
| ≥ 0.80 | 90.2% (6,740) | 99.67% | 733 | 28.2% |
| ≥ 0.96 | 77.0% (5,751) | 99.95% | 1,722 | 13.1% |
| = 1.00 | 53.4% (3,993) | **100.00%** | 3,480 | 6.6% |

**Headline sentences (computed):** abstaining on the 1% least-confident items raises accuracy from 96.94% to 97.42% (7,398 retained); on 2% → 97.91% (7,324); on 5% → 98.76% (7,099); on 10% → **99.70%** (6,726). (`tables/abstain.csv`.)

**Confidently-wrong rate: 0.** Zero items were incorrect with top-1 = 1.0 (0/229 errors, 0.0%); zero with raw.confidence ≥ 0.99 (0/229). Every one of the 229 errors came with stated doubt.

**What this means.** This is the practically important result. A System One model is sold as a decision primitive; a decision primitive that (a) is never wrong at maximum confidence and (b) reaches 99.7% accuracy by deferring one item in ten is deployable in a way a chat model parsing JSON is not. The caveat: this is one dataset, one run, and 3,993 perfect-at-1.0 items is a large but finite sample — the true error rate at confidence 1.0 is bounded above only by ~3/3,993 ≈ 0.075% (95% rule of three), not proven to be zero.

## 6. Position and label bias

| Letter | Gold n (%) | Predicted n (%) |
|---|---|---|
| A | 1,876 (25.10%) | 1,893 (25.33%) |
| B | 1,846 (24.70%) | 1,851 (24.77%) |
| C | 1,872 (25.05%) | 1,863 (24.93%) |
| D | 1,879 (25.14%) | 1,866 (24.97%) |

Chi-square goodness-of-fit of predicted marginals against gold marginals: χ²(3) = 0.301, p = 0.960 — no detectable position preference. Per-letter accuracy: A 97.44% (n=1,876), B 97.13% (1,846), C 96.58% (1,872), D 96.59% (1,879) — spread 0.86 pp, within overlapping Wilson CIs. The 4×4 confusion matrix (`figures/confusion_matrix.png`, `tables/confusion_matrix.csv`) is cleanly diagonal; off-diagonal mass is symmetric (e.g. gold-C→pred-A 26 and gold-D→pred-C 26 are the largest cells). Mean probability mass per position: A 0.256, B 0.248, C 0.246, D 0.250 — a 0.7-pp tilt toward A that does not rise to significance in the marginals test.

**What this means.** No evidence of position or label bias, so the headline accuracy is not inflated by exploiting the dataset's (near-uniform) answer key. Caution: option order was never shuffled, so this measures the *absence* of bias under the given ordering, not robustness to reordering (§10).

## 7. Error analysis

### 7.1 Difficulty correlates

Accuracy by quartile/category with Wilson CIs: `figures/feature_accuracy.png`, `tables/feature_bins.csv` (all n ≈ 1,800–1,900 per quartile).

- **Question length** (chars): Q1 98.7% → Q4 94.6%. Word count and `input_tokens` show the same gradient (input tokens Q1 98.8% → Q4 94.1%). Longer problems are harder.
- **Number of numeric literals in the text:** 0–2 literals 98.6% (n=2,364) → 4–15 literals **92.9%** (n=1,525). The strongest single correlate — more numbers to track, more errors.
- **Option spread (max/min ratio):** tightest quartile (ratio 1.1–9.3) **93.8%** vs widest quartile 98.7%. Plausible, close distractors are harder — direct evidence that *distractor quality*, not just problem difficulty, drives errors. Option CV shows the same pattern at both tails (Q1 95.9%, Q4 96.1%, middle ≈97.8%).
- **Gold rank among options:** gold is the **smallest** option → 98.5% (n=5,276); **largest** → 95.8% (n=946); **neither** → **91.2%** (n=1,251). Note the skew: 70.6% of items have the gold as the smallest option — the dataset's distractors are mostly inflated decoys, which a magnitude-sanity-check can eliminate. Part of the high headline accuracy is attributable to this dataset property.
- **Gold magnitude:** (0,10) 99.0% (n=1,360) and [10,100) 97.7% (n=3,603) vs [100,1000) **93.6%** (n=1,709); [1000,+) recovers to 96.9% (n=797). Negative (n=3) and zero (n=1) golds are too rare to interpret; all 4 items with a negative/zero option were correct.
- All golds are integers (7,473/7,473), so integer-vs-not cannot be sliced.

### 7.2 Near-miss structure

For the 229 errors, relative error |pred − gold| / |gold|: median 0.493, IQR [0.188, 0.812]; 12.2% (28/229) are within 10% of the gold value, 51.5% (118/229) within 50%, and 9.2% (21/229) are off by more than 2×. **In 71.2% of errors (163/229), the predicted option was the numerically nearest distractor to the gold value.** Errors look like arithmetic slips that land close to the right answer, not wrong approaches that land arbitrarily far away.

### 7.3 Surface slices (exploratory)

Regex slices on question text, sorted by accuracy (`tables/keyword_slices.csv`); none are underpowered (all n ≥ 30). **These are exploratory, nine correlated comparisons — no significance claims.**

| Slice | n | Accuracy [Wilson 95% CI] |
|---|---|---|
| fraction (`/` or "fraction") | 614 | **94.79%** [92.74, 96.28] |
| geometry | 68 | 95.59% [87.81, 98.49] |
| ratio/proportion | 47 | 95.74% [85.75, 98.83] |
| interest/finance | 1,007 | 95.83% [94.41, 96.90] |
| work-rate | 314 | 96.50% [93.84, 98.03] |
| probability/combinatorics | 116 | 96.55% [91.47, 98.65] |
| percent | 1,009 | 97.03% [95.79, 97.91] |
| speed/distance/time | 300 | 97.33% [94.83, 98.64] |
| explicit negation | 238 | 97.48% [94.61, 98.84] |

Even the worst slice is only ~2 pp below the headline; no category collapses.

### 7.4 Failure taxonomy

Hand-assigned by reading the 95-item qualitative dump (`errors_sample.md`: 40 highest-confidence incorrect, 40 seeded-random incorrect, 15 lowest-confidence correct). **Not exhaustive; shares are estimates from the 80 incorrect items read.**

| # | Category | Est. share of errors | Example ids |
|---|---|---|---|
| 1 | **Dropped constraint / wrong final aggregation** — correct intermediates, then a step is skipped or the wrong quantity is answered | ~25% | mathqa-01924 (ignored the 5 torn hats), mathqa-03125, mathqa-07062, mathqa-00449 |
| 2 | **Arithmetic near-miss** — predicted value within a few units of gold, picks the nearest distractor | ~20% | mathqa-03515 (19 vs 18), mathqa-01311 (65 vs 67), mathqa-01409 (5 vs 1), mathqa-00732 (135 vs 144) |
| 3 | **Chained-relation tracking** — "X more/fewer than", "twice as many as" across 3+ entities, one link inverted or lost | ~20% | mathqa-03925, mathqa-06858, mathqa-04900, mathqa-04134 |
| 4 | **Ambiguous or arguably mislabeled items** — "three times more than" read as 4× by the key; underspecified problems; questions whose literal answer is not among the options | ~12% | mathqa-06138, mathqa-05231, mathqa-01805, mathqa-04806 (asks "how much did Mark deposit" — stated as $88 in the text, key says 400) |
| 5 | **Temporal/reference-frame shifts** — "in N years", week↔month conversions | ~8% | mathqa-04472, mathqa-05541, mathqa-04770 |
| 6 | **Unit conversion / world knowledge** — ounces↔pounds, coins, animal legs | ~6% | mathqa-04421, mathqa-06801, mathqa-03344 |
| 7 | **Long multi-step money/percent problems** (residual) | ~9% | mathqa-07330, mathqa-03875, mathqa-01478 |

**What this means (7.1–7.4).** The error profile is exactly what a "fast intuition" model should produce: it fails the way a rushed careful person fails — lost constraints, off-by-a-little arithmetic, one inverted relation in a chain — and not by hallucinating structure or by position guessing. Two qualifiers temper the headline: the dataset's distractors are frequently eliminable by magnitude (70.6% of golds are the smallest option), so some of the 96.94% is answer-key savvy rather than arithmetic; and ~12% of the *errors* are on items whose phrasing is genuinely ambiguous or whose key is debatable, so the true model error rate is arguably below 3.06%.

## 8. Cost, latency, throughput

Primary latency figures exclude the 2 retried rows (n = 7,471).

| Metric | Value |
|---|---|
| Latency mean ± sd | 800.3 ± 80.9 ms |
| p50 / p90 / p95 / p99 | 794.0 / 888.8 / 919.6 / 1,016.8 ms |
| min / max | 382.9 / 1,962.6 ms |
| Total cost (sum of `usage.cost`) | **$0.129608** |
| Mean cost per item / per 1,000 items | $1.7344e-05 / **$0.0173** |
| Cost per correct answer | $1.7892e-05 |
| Input tokens | mean 412.9, range 364–563 |
| Output tokens | **exactly 45 on every row (1 distinct value)** |
| Wall-clock / throughput | 499 s / 14.98 items/s |

Figures: `figures/latency_hist.png`, `latency_ecdf.png`, `latency_vs_tokens.png`; table `tables/latency_cost.csv`.

- **Latency vs input length:** Pearson r = −0.015 (p = 0.19), Spearman ρ = −0.017 (p = 0.13), n = 7,471. Effectively constant — consistent with a fixed-cost single forward judgment, not token-by-token generation.
- **Latency by correctness:** correct mean 800.3 ms vs incorrect 798.4 ms (medians 793.9 / 795.2); Mann–Whitney U = 830,586, p = 0.966. The model does **not** take longer when wrong — confirming the System One prediction.
- **Cost reconciliation:** the sum of `usage.cost` ($0.129608) matches `experiment.md`'s $0.13 headline (rounding) and matches *exactly* the expected cost at the published $0.042/M input price ($0.129608); implied price from tokens = $0.042/M input. Billing is consistent to the last digit.
- **`output_tokens` = 45 always.** With no generated text, this is a fixed billing artifact of the typed response envelope, not content — and since output is priced at $0, it is cosmetic.
- **Throughput** hit 14.98 items/s against the harness's 15 req/s cap with 12 workers: the run was pacing-limited, not model-limited. At the API's stated 1,200 req/min ceiling the same run would take ~6–7 minutes single-client.

**What this means.** The economics are as advertised and then some: $0.017 per 1,000 judgments, flat ~800 ms latency independent of problem size, no "thinking longer when unsure." For the paper, the constant-latency result is direct behavioral evidence for the single-pass decision framing.

## 9. Robustness

- **Bootstrap accuracy CI** (2,000 resamples, seed 20260928): [96.55%, 97.32%] — agrees with the Wilson interval [96.52%, 97.30%].
- **Split-half stability** (by timestamp): first half 96.60% (n=3,736) vs second half 97.27% (n=3,737); two-proportion z = −1.680, p = 0.093 — no significant drift or throttling effect (the small improvement direction, if anything).
- **Leave-one-slice-out:** removing the worst slice (fraction, 94.79%, n=614) moves the headline to 97.13% — a **+0.19 pp** shift, under the 0.5 pp threshold. The headline is not slice-sensitive.
- **Ambiguity floor:** 0 duplicate-option rows and 0 malformed rows → the maximum achievable accuracy on this dataset is 100%; no ambiguity discount applies. (Separately, §7.4 estimates ~12% of *errors* sit on ambiguous items, but that is a keying-quality observation, not an argmax ambiguity.)

**What this means.** The 96.94% figure is stable to resampling, to run position, and to slice removal. The remaining uncertainty is statistical (±0.4 pp), not methodological.

## 10. Threats to validity

- **Contamination — not testable here, and likely.** NLP2025-math is public on Kaggle and predates Jev; whether it was in training data cannot be determined from `results.jsonl`. What the data does show: the dataset contains **zero exact-duplicate** question texts internally and only **2 items** (1 pair) with shingle-Jaccard ≥ 0.6 near-duplication, both answered correctly — so internal memorization shortcuts within the run are negligible. External contamination remains entirely open.
- **Single run, single model version, no seed variation.** Run-to-run variance is unestimated. The API exposes no sampling controls, so variance may be zero — but that is an assumption, not a measurement.
- **No baseline against a chat LLM on the same items.** Nothing here supports "Jev beats/trails GPT-x/Claude/Gemini on math." Any relative claim needs a same-items head-to-head.
- **Multiple choice ≠ free-form math.** Elimination is available and the dataset makes it easy: 70.6% of gold values are the smallest option and all golds are integers, while distractors include absurd magnitudes (e.g. 620, 536, 379 against 10). §7.1 (gold-is-smallest 98.5% vs neither 91.2%; wide option spread 98.7%) shows accuracy partly tracks distractor eliminability. Free-form accuracy would be lower by an unknown amount.
- **Prompt not varied.** One instruction string, one option ordering. Position bias is absent (§6) but robustness to shuffling and rewording is untested.
- **API artifacts vs model behavior.** `output_tokens` (constant 45) is a billing envelope, not content; `raw.confidence` is an API-provided scalar whose relationship to the distribution is undocumented — both are reported as API artifacts, not model internals. The 8 tie rows (§1) show the API's own `choice` tie-break is opaque.
- **Quantized probabilities** (2-decimal, 53% degenerate) bound the resolution of every calibration and selective-prediction claim (§3).

## 11. Open questions for the paper

1. **Is accuracy at confidence 1.0 really 100%?** Observed 3,993/3,993, 95% upper bound ≈ 0.075% (rule of three). Needs: a second, harder dataset (or a shuffled-distractor variant) run through the same harness to stress the perfect record.
2. **How much of the headline is elimination rather than solving?** Needs: a free-form variant (no options, `score`-type or exact-match answers) on the same items, or a distractor-hardened multiple-choice set where all options are near-misses.
3. **Would a chat LLM do better or worse per dollar on identical items?** Needs: same-items runs of 2–3 frontier chat models with the harness's recording discipline; cost and latency are directly comparable against §8.
4. **Is the run deterministic?** Needs: a repeat run of a 500-item subset; compare row-level outputs including the 8 tie rows.
5. **What is `raw.confidence`, mechanically?** It is always ≤ top-1 and more conservative. Needs: documentation or a probe set designed to make the two signals diverge (e.g. near-tie distributions).
6. **Position robustness.** Needs: one rerun with per-item shuffled option order (the harness supports it by permuting `criteria`); compare against §6's null result.
7. **Contamination check.** Needs: a post-cutoff synthetic math set in the same style, or membership-inference evidence.

## Appendix A: all figures

| File | Contents |
|---|---|
| `figures/prob_values_hist.png` | All 29,892 assigned probability values, log-count |
| `figures/reliability_top1_{eqcount,eqwidth,zoom}.png` | Reliability diagrams, top-1 score (N=7,473) |
| `figures/reliability_rawconf_{eqcount,eqwidth,zoom}.png` | Reliability diagrams, raw.confidence (N=7,473) |
| `figures/conf_hist_correct_incorrect.png` | Confidence histograms by correctness, both scores, log-count |
| `figures/risk_coverage.png` | Risk–coverage curves, four scores, log-error (N=7,473) |
| `figures/confusion_matrix.png` | Gold × predicted letter, row-normalised (N=7,473) |
| `figures/feature_accuracy.png` | Accuracy by item-feature bins with Wilson CIs |
| `figures/latency_hist.png` / `latency_ecdf.png` | Latency distribution, attempts==1 (n=7,471) |
| `figures/latency_vs_tokens.png` | Latency vs input tokens scatter (n=7,471) |

## Appendix B: reproduction

```
./venv/bin/pip install numpy pandas matplotlib scipy requests
./venv/bin/python jev.py          # the run of record (results.jsonl) — already executed
./venv/bin/python analyze.py      # regenerates everything in ~19 s, byte-identical
```

Versions: Python 3.12.3, numpy 2.5.3, pandas 3.0.6, matplotlib 3.11.2, scipy 1.18.1. Seeds: sample seed 42, bootstrap seed 20260928, 2,000 bootstrap resamples, NLL clipping ε = 1e-3. Analysis runtime: 18.9 s. Tables for every number in this report: `tables/*.csv`; per-item data: `derived.csv`; all scalars: `stats.json`; qualitative error dump: `errors_sample.md`.
