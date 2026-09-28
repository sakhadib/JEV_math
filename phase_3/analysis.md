# Phase 3 analysis: HARP — genuinely hard competition math

The final experiment. **HARP** ([Yue et al. 2024](https://github.com/aadityasingh/HARP)): 4,110 five-option items of human-annotated US competition mathematics (AHSME/AJHSME/AMC 8/10/12), with level (1–4), subject, and contest metadata. Same four systems, same zero-shot direct-choice protocol (phase 3 prompt additionally appends "REASONING IS PROHIBITED"), same harness. All runs complete: 4,110/4,110 rows per system, zero API errors. Deterministic analysis: `analyze_phase3.py` → `stats.json`, `tables/`.

**Parsing policy.** Strict as-run parsing (letter-only) is the primary metric. Because verbose-but-decisive answers exist (llama ignored the format and ended responses with `\boxed{X}`), an **extended re-parse** of stored response text (boxed / "final answer is" patterns) is reported alongside. Empty responses (reasoning models burning the whole token budget before emitting anything) cannot be recovered by any parser — those stay failures.

## 0. Headline

| System | Accuracy (strict e2e) | Wilson 95% CI | Extended parse | Parse fails | Hidden reasoning tokens |
|---|---|---|---|---|---|
| **space-bunny-alpha** | **81.29%** (3,341/4,110) | [80.07, 82.45] | 81.29% | 490 (488 empty/`length`) | 0 reported, but truncated-empty behavior says otherwise |
| **gpt-oss-20b** | 70.29% (2,889/4,110) | [68.87, 71.68] | 70.29% | 867 (863 empty/`length`) | **93.5% of rows**, mean 274 |
| **jev-1.13** | 54.70% (2,250/4,110) | [53.17, 56.22] | n/a (typed) | 0 | n/a |
| **llama-3.1-8b-instruct** | 18.32% (753/4,110) | [17.15, 19.54] | 22.65% | 538 (verbose CoT) | 0 |

McNemar vs Jev (paired, n=4,110): space-bunny p = 8.4e-166, gpt-oss p = 2.1e-57 (both decisively above Jev); llama p = 1.4e-268 below. **266 items (6.5%) were missed by all four systems.**

On items the chat models actually answered, accuracy is higher still: space-bunny 92.3% (3,341/3,620), gpt-oss 89.1% (2,889/3,243) — their empty answers are protocol failures, not wrong answers.

## 1. What HARP proves

**Phases 1–2 were dataset problems; phase 3 is a capability measurement.** NLP2025-math and CompMath-MCQ let every strong system sit above 94%. HARP spreads them across 63 points (18%→81%). The ordering is exactly what the System One / System Two framing predicts:

1. **Single-pass intuition caps out.** Jev at 54.7% (random floor 20%) — real capability, far above chance, but nowhere near reasoners. Its accuracy degrades gently with difficulty: level 1 → 4 goes 58.2% → 57.6% → 51.8% → 46.2%. Weakest subject: number theory (39.7%); strongest: prealgebra (62.9%).
2. **Reasoning dominates hard math.** space-bunny-alpha — a stealth frontier model whose 488 empty/`length` truncations reveal hidden deliberation despite reporting zero reasoning tokens — is the standout at 81.3% strict / 92.3% of answered, with the shallowest level gradient (86.2% → 66.7%). gpt-oss-20b's hidden CoT (93.5% of rows) earns it 70.3% strict / 89.1% of answered, but its level gradient is the steepest of the strong models (84.0% → 43.5% at level 4).
3. **Small instruct models collapse without CoT** — llama at 18.3% strict is *below* the 20% random floor, with 46.8% of its answers on option A (χ² position bias p ≈ 0, third consecutive dataset).

## 2. Jev's confidence under real pressure — the finding of the phase

HARP is the first dataset where Jev cannot just be right. Its response is the most informative behavior we've observed:

- **Certainty collapsed appropriately.** Degenerate p=1.0 rows: **5.2%** (213/4,110), vs 62.1% on CompMath and 53.4% on MathQA. Jev "knows" this dataset is hard and commits fully only 1 time in 20.
- **When it is certain, it is still nearly always right:** 99.06% accuracy at p=1.0 (211/213; the 2 confidently-wrong items are the same failure count as all of phase 2).
- **Calibration stayed tight while confidence collapsed:** mean top-1 confidence 0.5817 vs accuracy 0.5470 — gap **+3.47 pp** (first dataset where the sign flips to overconfidence, and it is small). ECE 0.0347 (equal-count, 10 bins). Error confidence histogram peaks at 0.3–0.5: its 1,860 errors overwhelmingly arrive *labelled as guesses*.
- **But deferral buys much less on hard tasks:** abstaining on the 1/2/5/10% least-confident items lifts accuracy only to 55.05% / 55.44% / 56.48% / **58.04%** (vs 97.82% at 90% coverage in phase 2). When most items carry low confidence, thresholding cannot manufacture accuracy — the confidence is honest precisely because there is little to be confident about.

**What this means.** Across three datasets, Jev's probability output has now been right *about itself* at every difficulty level: certain-and-perfect on easy data (100%/3,993), certain-and-99.8% on medium data, uncertain-and-honest on hard data. That is the strongest evidence in this study that the calibration claim is real and not an artifact of easy benchmarks.

## 3. Efficiency

| System | Wall-clock (4,110 items) | Latency p50 | Total cost |
|---|---|---|---|
| jev-1.13 | **428 s** | **793 ms** (flat for the third dataset) | $0.0832 |
| llama-3.1-8b | 916 s | 923 ms | $0.0299 |
| space-bunny-alpha | 4,552 s | 2,103 ms | $0.00 |
| gpt-oss-20b | 3,482 s | 4,050 ms | $0.2102 |

## 4. Cross-phase grand summary

| Metric | Phase 1: MathQA-style | Phase 2: CompMath-MCQ | Phase 3: HARP |
|---|---|---|---|
| Jev accuracy | 96.94% | 94.89% | 54.70% |
| Jev p=1.0 share / acc at p=1.0 | 53.4% / 100.00% | 62.1% / 99.79% | 5.2% / 99.06% |
| Jev conf−acc gap | −2.4 pp | −0.8 pp | +3.5 pp |
| Jev acc at 90% coverage | 99.70% | 97.82% | 58.04% |
| Best baseline (strict) | gpt-oss 98.30% | gpt-oss 96.79% | **space-bunny 81.29%** |
| space-bunny vs Jev | tie (p=0.51) | tie (p=0.53) | **space-bunny wins (p=8e-166)** |
| llama | 40.09% | 60.12% | 18.32% (below floor) |
| Jev latency p50 | 794 ms | 793 ms | 793 ms |
| Jev total cost | $0.130 | $0.025 | $0.083 |

**The three-phase story:** when math is easy, everyone scores 95%+ and Jev's edge is operational (typed output, deferral, speed, price). When math is hard, the ranking flips to reasoning models by 26 points, and Jev's edge is *epistemic* — it is the only system that told us, item by item and correctly, that it was guessing. Both edges are real; the paper should claim each where it holds and not confuse them.

## 5. Threats specific to phase 3

- **HARP (2024) predates all four models' likely training cutoffs.** Contamination is plausible for every system and unresolvable from this data. It inflates everyone roughly together, but differential contamination (Jev's training data is unknown) cannot be ruled out.
- **The strict/extended parse asymmetry favors models that complied with formatting.** llama's extended-parse gain (+4.3 pp) is real decisions recovered from stored text; the empty-answer models (gpt-oss, space-bunny) got no such recovery — their "of answered" figures (89.1% / 92.3%) are the fair ceiling estimates.
- **space-bunny's reasoning is unobservable** (0 reported reasoning tokens, but 488 truncation-empties); claims about its category rest on behavior, not billing data.
- **Jev's protocol never varies** — one instruction string across 9,000 items; results may not transfer to differently-worded criteria.

## Reproduction

```bash
./venv/bin/python phase_3/run_jev.py          # Jev run — already executed
./venv/bin/python phase_3/run_baseline.py     # 3 chat baselines — already executed
./venv/bin/python phase_3/analyze_phase3.py   # regenerates stats.json + tables/, byte-identical
```
