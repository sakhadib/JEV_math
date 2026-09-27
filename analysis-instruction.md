# Analysis Instructions — "Can Jev do math?"

**Audience:** the coding agent running analysis on `results.jsonl`.
**Goal:** produce `analysis.md`, a complete, self-contained analysis report that a paper author can read *instead of* the raw data. Every claim the paper will make must be traceable to a number, table, or figure in `analysis.md`.

---

## 0. Ground rules

1. **Never invent a number.** If something cannot be computed from `results.jsonl`, write `NOT COMPUTABLE` and say exactly why. A missing number is fine; a fabricated one destroys the paper.
2. **Every number carries its `n`.** "96.9%" is useless; "96.92% (7,243/7,473)" is usable.
3. **Report to 4 significant figures** internally; round only in prose, and say when you round.
4. **All analysis lives in one deterministic script**, `analyze.py`. Set every seed. Running it twice must produce byte-identical outputs. No notebook-only state.
5. **Write down what surprised you.** A section of anomalies is worth more than a section of confirmations. If a result contradicts the framing in `experiment.md`, say so plainly — do not smooth it over.
6. **Distinguish "the model did X" from "the API reported X".** Several quantities below are API artifacts, not model behavior. Keep them separate.
7. No plotting style requirements beyond legibility: axis labels, units, `n` in the caption.

---

## 1. Inputs and outputs

**Input:** `results.jsonl` — one JSON object per line, schema documented in `experiment.md`. Expect 7,473 lines.

**Produce, in the repo root:**

| File | Contents |
|---|---|
| `analyze.py` | the single deterministic analysis script |
| `analysis.md` | the report (structure specified in §12) |
| `figures/*.png` | all figures, 150 dpi, referenced from `analysis.md` |
| `tables/*.csv` | machine-readable version of every table in the report |
| `errors_sample.md` | qualitative dump (see §8.4) |
| `derived.parquet` (or `.csv`) | the per-item dataframe after Phase 0, so results are re-checkable |

---

## 2. Phase 0 — Load and validate (do this before anything else)

Compute and report all of:

- Line count; count of lines that fail to parse as JSON.
- Duplicate `id` count. If duplicates exist, report how you resolved them (keep first) and how many rows were affected.
- `status` distribution. Expected: all `ok`. Report any `error` rows verbatim and **exclude them from accuracy**, stating the exclusion explicitly.
- Distribution of `attempts` (how many calls needed a retry). Retries are a reliability finding.
- `model` field: confirm a **single** resolved model id across all rows. If more than one appears, break every downstream metric out by resolved model — that would be a silent mid-run model change and is a major caveat.
- Schema completeness: for each field in the documented schema, the count of rows where it is missing or null.
- **Gold-label sanity:** every `gold` ∈ {0,1,2,3}; every row has exactly 4 `choices`; `choices[gold]` is well-defined.
- **Duplicate-option check:** rows where two or more of the 4 options are numerically equal. On those rows argmax accuracy is ambiguous — count them, and report headline accuracy both including and excluding them.
- **Internal consistency:** verify that `predicted_index` equals argmax of `probs`, that `predicted_value == choices[predicted_index]`, that `correct == (predicted_index == gold)`, that `confidence == max(probs)`, that `margin == top1 − top2`, and that `entropy_nats` matches Shannon entropy of `probs` in nats. Report the count of rows failing each check. Do not silently repair — report, then use recomputed values downstream and say so.
- `ts` range: first and last timestamp, wall-clock span of the run.

State up front the final analysis population `N` used for the rest of the report.

---

## 3. Phase 1 — Headline performance

- Accuracy with a **Wilson 95% confidence interval**.
- Baselines for comparison, computed from this dataset, not assumed:
  - uniform random over 4 options (analytic, 25%);
  - always-predict-a-fixed-letter, for each of A/B/C/D (uses the actual gold letter marginals);
  - the best of those fixed-letter baselines.
- Exact counts: correct, incorrect, total.
- Report accuracy on the duplicate-option subset separately if non-empty.

---

## 4. Phase 2 — Are the probabilities real? (do this before any calibration)

This phase decides whether Phase 3 is meaningful. **Do not skip it.**

The example record in `experiment.md` shows `probs` of `{A:0, B:0, C:0, D:1}` with a separate `raw.answers.answer.confidence` of `0.99`. That is suspicious: the distribution appears quantized while a second, different confidence signal exists.

Determine empirically:

1. **Precision / quantization.** Collect all distinct probability values across all rows. How many distinct values are there? What is the smallest non-zero value? Are values consistent with 2-decimal rounding, or with full float precision? Produce a histogram of all probability values on a log-count y-axis.
2. **Degeneracy rate.** Fraction of rows where the top probability is exactly 1.0; where it is ≥0.99; where at least one option has probability exactly 0.0. Report the distribution of "number of options with non-zero probability" (1, 2, 3, or 4).
3. **Do they sum to 1?** Distribution of `sum(probs)`; count of rows deviating by more than 1e-6.
4. **The two confidence signals.** `top1(probs)` versus `raw.answers.answer.confidence` are different quantities. Extract both. Report: their joint distribution, how often they disagree, the distribution of `raw.confidence` on its own (distinct values, min, max), and the correlation between them. **Carry both forward through Phases 3 and 4 as separate candidate confidence scores** and report which one is better calibrated / more useful for selective prediction. This may be the most interesting finding in the whole study.
5. **Verdict.** State explicitly whether the returned distribution is informative enough to support calibration analysis, or whether it is effectively a hard label plus a coarse confidence flag. The paper's claims about "calibrated probabilities" depend entirely on this answer.

---

## 5. Phase 3 — Calibration

Run for **each** confidence score identified in §4.4 (top-1 probability, and `raw.confidence`), clearly labelled.

- **Reliability diagram.** Use equal-*count* bins (10 bins) as the primary, since the mass is concentrated near 1.0; also produce an equal-*width* version. Show bin counts on the plot. If most mass falls in one bin, say so and add a zoomed diagram over the top decile (e.g. 0.9–1.0 split finely).
- **Expected Calibration Error (ECE)** and **Maximum Calibration Error (MCE)**, both binning schemes, bin count stated.
- **Brier score** (multiclass / one-vs-all against the 4-option distribution) and its decomposition into reliability, resolution, and uncertainty if the distribution supports it.
- **Negative log-likelihood** of the gold option. Zero probabilities make this infinite — report the count of rows where the gold option received exactly 0.0 probability, and report NLL with a stated clipping ε (e.g. 1e-3), being explicit that the value depends on ε.
- **Overconfidence gap:** mean confidence − accuracy.
- **Confidence histograms** split by correct vs incorrect, overlaid.

If Phase 2 concludes the distribution is degenerate, still run these, but open the section with a one-paragraph statement of what the numbers can and cannot mean.

---

## 6. Phase 4 — Selective prediction (does it know when it doesn't know?)

This is the practically important question for a typed decision model, and it works even if the probabilities are coarse.

For each candidate score *s* ∈ {top-1 probability, `raw.confidence`, `margin`, `−entropy_nats`}:

- **Risk–coverage curve:** sweep the threshold, plot error rate among retained items against coverage. All four scores on one figure.
- **AURC** (area under risk–coverage) and **AUROC** for the binary task "is this prediction correct?" using *s* as the score. Report AUROC with a 95% bootstrap CI (2,000 resamples, seed fixed).
- **A deferral table** — the single most useful artifact for the paper:

| Threshold | Coverage | Retained accuracy | Deferred n | Error rate among deferred |
|---|---|---|---|---|

  Sweep thresholds at meaningful points given the observed value distribution (do not hardcode 0.5/0.6/0.7 if all mass sits above 0.99 — choose thresholds from the empirical quantiles).
- **Headline sentence to compute:** "abstaining on the *k*% least-confident items raises accuracy from X% to Y% on the remaining items," for k ∈ {1, 2, 5, 10}.
- **Confidently wrong rate:** fraction of all items that are incorrect *and* at maximum confidence (top-1 = 1.0, and separately `raw.confidence` ≥ 0.99). Report as count and as a share of all errors. If most errors are confidently wrong, that is a headline finding and should be stated bluntly.

---

## 7. Phase 5 — Position and label bias

A `choice` model over caller-ordered options may prefer positions.

- Marginal distribution of `gold` letter vs marginal distribution of `predicted_letter`, as counts and percentages, side by side.
- Chi-square goodness-of-fit of predicted marginals against gold marginals; report statistic, df, p.
- **Per-letter accuracy:** accuracy conditional on the gold answer being A / B / C / D. A large spread is position bias.
- **Confusion matrix** 4×4, gold letter × predicted letter, with row-normalised percentages. Figure + CSV.
- **Mean probability mass assigned to each position** across all items, regardless of correctness. If position A systematically receives more mass, report it.

---

## 8. Phase 6 — Error analysis

### 8.1 Where the errors are
For each of the following item features, bin into quartiles (or natural categories) and report accuracy per bin with `n` and Wilson CI, plus a figure:

- question character length and word count;
- `input_tokens`;
- number of distinct numeric literals in the question text;
- magnitude of the gold answer (log-scaled bins);
- **option spread**: ratio of max(choices) to min(choices), and the coefficient of variation of the 4 options — tests whether tightly-clustered distractors are harder;
- whether the gold answer is the smallest / largest / neither among the 4 options;
- whether the gold answer is an integer vs non-integer;
- whether any option is negative or zero.

### 8.2 Near-miss structure
For incorrect items, how close was the predicted value to the gold value? Report the distribution of relative error `|pred − gold| / |gold|`, and the fraction of errors where the predicted value was the numerically *nearest* distractor to the gold value. This distinguishes "arithmetic slip" from "wrong approach".

### 8.3 Surface-feature keyword slices
Using simple case-insensitive regex on the question text, report accuracy for items containing: fractions (`/` or "fraction"), percent (`%`/"percent"), ratio/proportion, speed/distance/time words, work-rate words ("together", "alone", "rate"), probability/combinatorics words, geometry words (area, perimeter, radius, triangle), interest/finance words, explicit negation ("not", "except", "least"). Report `n` and accuracy per slice, sorted by accuracy ascending. Flag slices with `n < 30` as underpowered. **These are exploratory slices — say so, and do not apply significance claims to them without a multiple-comparison note.**

### 8.4 Qualitative dump — `errors_sample.md`
Write a separate file containing:
- **all** incorrect items where top-1 probability is 1.0 (or the 40 with highest confidence, if that set exceeds 40), and
- a seeded random sample of 40 further incorrect items, and
- 15 correct items with the *lowest* confidence (lucky guesses / genuine uncertainty).

For each: `id`, full question text, the 4 options, gold value, predicted value, the full probability distribution, `raw.confidence`, and latency. Group them under headings. Add **no interpretation** — this file is raw material for the paper author to read.

Then, in `analysis.md`, propose a **failure taxonomy** (4–7 categories) derived from actually reading that dump, with the count or estimated share per category and 2–3 example ids each. Be explicit that the taxonomy is hand-assigned from a sample, not exhaustive.

---

## 9. Phase 7 — Cost, latency, throughput

- Latency: mean, sd, p50, p90, p95, p99, min, max. Histogram, and an ECDF.
- Latency vs `input_tokens`: scatter + Pearson and Spearman correlation. Is latency input-length dependent, or effectively constant? A constant answer supports the "single judgment, no generation" framing and is worth stating.
- Latency for correct vs incorrect items: means, and a Mann–Whitney U test. Does the model take longer when it is wrong? (Under the System One framing it should not — confirm or refute.)
- Latency by `attempts` (retried calls will be slower; exclude `attempts > 1` from the primary latency figures and say so).
- Cost: total, mean per item, **cost per correct answer**, cost per 1,000 items. Reconcile the sum of `usage.cost` against the $0.13 headline in `experiment.md`; if they differ, report both and flag the discrepancy.
- Tokens: distribution of `input_tokens` and `output_tokens`. **`output_tokens` is reported as 45 in the example despite there being no free text** — report the distribution of `output_tokens` and whether it is effectively constant. If it is, note that output tokens appear to be a fixed billing artifact of the typed response, not generated content. Verify against the stated $0.042/M input, output-free pricing: recompute expected cost from tokens and compare to reported cost; report the implied price per million input tokens.
- Wall-clock: total run duration from `ts`, achieved throughput (items/sec), and how that compares to the harness's 12 workers × ≤15 req/s cap.

---

## 10. Phase 8 — Robustness of the headline

- **Bootstrap** the accuracy (2,000 resamples, fixed seed) and report the CI; compare to the Wilson interval.
- **Split-half stability:** split by `ts` into first and second half of the run; report accuracy for each and whether they differ (two-proportion test). This detects drift or throttling effects mid-run.
- **Leave-one-slice-out sensitivity:** does removing the single worst-performing §8.3 slice move the headline by more than 0.5 points? If yes, the headline is slice-sensitive; say so.
- **Ambiguity floor:** combining duplicate-option rows and any malformed rows from Phase 0, state the maximum accuracy that was actually achievable on this dataset.

---

## 11. Phase 9 — Threats to validity

Write this section as bullet points, each one specific and falsifiable-or-not-testable-and-labelled-as-such:

- **Contamination.** The NLP2025-math dataset is public on Kaggle and predates the model. This cannot be tested from `results.jsonl` alone — say so explicitly. What you *can* do: report the count of exact-duplicate and near-duplicate question texts *within* the dataset (normalised whitespace/casing; also a cheap shingle-based near-duplicate pass), and report accuracy on duplicated vs unique items.
- **Single run, single model version, no seeds varied** — no estimate of run-to-run variance. Note that the API has no sampling, so variance may be zero, but that is an assumption, not a measurement, unless you find evidence for it.
- **No baseline comparison against a chat LLM on the same items.** The study cannot claim relative performance. State this.
- **Four-option multiple choice is not free-form math.** Accuracy here is not accuracy at solving; elimination strategies are available. Where §8.1/§8.2 gives evidence for or against elimination behaviour, cite it.
- **Prompt not varied** — one instruction string, one option ordering. Position bias (§7) is measured but not controlled for by shuffling.
- Any anomaly surfaced in Phase 0 or Phase 2 that limits interpretation.

---

## 12. Required structure of `analysis.md`

Write for a reader who has *not* seen the data. Prose paragraphs, not bullet fragments, in the interpretation sections. Every table also saved to `tables/`.

```
# Analysis: Jev-1.13 on NLP2025-math

## 0. Summary of findings
   5-10 bullets. The most surprising finding first, not the headline accuracy.
   Include one "if you read nothing else" paragraph.

## 1. Data integrity and analysis population      (Phase 0)
## 2. Headline performance and baselines          (Phase 1)
## 3. The nature of the returned probabilities    (Phase 2)  <- gating section
## 4. Calibration                                 (Phase 3)
## 5. Selective prediction and deferral           (Phase 4)
## 6. Position and label bias                     (Phase 5)
## 7. Error analysis                              (Phase 6)
   7.1 Difficulty correlates
   7.2 Near-miss structure
   7.3 Surface slices (exploratory)
   7.4 Failure taxonomy
## 8. Cost, latency, throughput                   (Phase 7)
## 9. Robustness                                  (Phase 8)
## 10. Threats to validity                        (Phase 9)
## 11. Open questions for the paper
   Things you could not resolve, and the specific extra experiment each would need.
## Appendix A: all figures
## Appendix B: exact reproduction commands, library versions, runtime
```

Each numbered section ends with a short **"What this means"** paragraph — the author's-eye interpretation, hedged appropriately.

---

## 13. What not to do

- Do not modify `results.jsonl`. It is the run of record.
- Do not drop rows silently for any reason. Every exclusion is counted and named.
- Do not report a p-value without the test name, the statistic, and `n`.
- Do not present exploratory slice findings (§8.3) with the same confidence as pre-specified analyses.
- Do not write marketing prose. "96.92% accuracy at $0.13" is a fact; "remarkable" is not.
- Do not round 96.92% to 97% in tables.
- Do not skip Phase 2 because Phase 3 looks more interesting. Phase 3's validity depends on it.
- If any phase turns out to be impossible or empty, keep the section heading and write one sentence saying why. Do not delete it — the paper author needs to know it was checked.
