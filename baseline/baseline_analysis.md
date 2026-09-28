# Baseline analysis: chat LLMs vs Jev on NLP2025-math

Companion to `../analysis.md`. Three chat models were run over the **same 7,473 items** via OpenRouter under a **zero-shot direct-choice protocol**: the question and lettered options, with the instruction *"Answer with only the letter of the correct option (A, B, C, or D). Do not explain, do not show any work."* — temperature 0, `reasoning: {effort: "low"}`, max_tokens 512. This is the closest chat-model analog to Jev's single typed judgment.

All runs complete: 7,473/7,473 rows per model, zero API errors, single resolved model id per file. Deterministic analysis: `analyze_baseline.py` (no randomness); machine-readable outputs in `baseline_stats.json` and `tables/`.

**Primary metric: end-to-end accuracy** — an unparseable or empty answer counts as wrong, because a system that must return a decision has failed to return one. Accuracy over parsed responses only is reported alongside.

## Headline comparison

| System | Accuracy (e2e) | Wilson 95% CI | Parsed-only acc | Parse fails | Latency p50 (mean) | Total cost | Cost/correct |
|---|---|---|---|---|---|---|---|
| **gpt-oss-20b** | **98.30%** (7,346/7,473) | [97.98, 98.57] | 99.57% | 95 (all empty) | 1,817 ms (3,208 ms) | $0.1404 | $1.91e-05 |
| **space-bunny-alpha** | 97.11% (7,257/7,473) | [96.70, 97.47] | 97.67% | 43 (all empty) | 1,427 ms (1,640 ms) | $0.00 (free) | $0.00 |
| **jev-1.13** | 96.92% (7,243/7,473) | [96.51, 97.29] | n/a (typed, always parseable) | 0 | 794 ms (800 ms) | $0.1296 | $1.79e-05 |
| **llama-3.1-8b-instruct** | 40.09% (2,996/7,473) | [38.99, 41.21] | 40.13% | 7 | 1,155 ms (1,209 ms) | $0.0220 | $7.35e-06 |

(Jev figures use the recorded fields, 7,243/7,473; the recomputed value 96.94% from `../analysis.md` differs by one tie-break item and changes nothing here.)

## Statistical significance (McNemar, paired on the same 7,473 items)

| Comparison | Jev right / baseline wrong | Jev wrong / baseline right | p (exact binomial) | Verdict |
|---|---|---|---|---|
| Jev vs **gpt-oss-20b** | 107 | 210 | 7.5e-09 | **gpt-oss significantly more accurate** |
| Jev vs **space-bunny-alpha** | 188 | 202 | 0.510 | **statistical tie** |
| Jev vs **llama-3.1-8b** | 4,300 | 53 | < 1e-300 | **Jev vastly more accurate** |

Only **4 items** were missed by all four systems (mathqa-04271, mathqa-04810, mathqa-05842, mathqa-06431) — natural candidates for the paper's "hardest items" appendix.

## The critical caveat: gpt-oss-20b cheated the protocol

The baseline protocol forbids reasoning. Two models complied; one did not:

- **gpt-oss-20b burned hidden reasoning tokens on 6,958 of 7,473 items (93.1%)** — mean 79.5 reasoning tokens per item — despite `reasoning: {effort: "low"}` and "do not explain" in the prompt. Its visible answer was a bare letter; its accuracy was earned with internal chain-of-thought. Its 95 parse failures are all *empty responses*, 93 of them with `finish_reason: "length"` — items where it spent the entire 512-token budget thinking and returned nothing at all. It also has by far the worst latency tail (mean 3.2 s, p95 8.7 s, max 405 s).
- **space-bunny-alpha** and **llama-3.1-8b** used **zero reasoning tokens on every row** — true single-pass answers, like Jev. (Space-bunny's 43 empty `length` responses suggest it occasionally spun up internally too, but none of it was billed or visible.)

So the honest apples-to-apples comparison — *single judgment, no reasoning* — is **Jev 96.92% vs space-bunny 97.11%: a statistical tie** (p = 0.51), and **vs llama-3.1-8b 40.09%**. gpt-oss-20b's 98.30% belongs to a different protocol category ("reasoning model with hidden CoT"), and even there it fails to return any answer on 1.27% of items — a failure mode Jev structurally cannot have.

## Position bias

- **llama-3.1-8b: severe.** Predicted marginals A 44.7% / B 33.0% / C 7.4% / D 14.7% against a ~25%-uniform key (χ²(3) p ≈ 0). Much of its 40% is letter-guessing behavior, not math.
- gpt-oss-20b: χ²(3) p = 0.984 — clean.
- space-bunny-alpha: χ²(3) p = 0.256 — clean.
- (Jev: p = 0.960, from `../analysis.md` — clean.)

## Efficiency

| System | Wall-clock for 7,473 items | Latency sd | Notes |
|---|---|---|---|
| jev-1.13 | **499 s** (14.98 items/s) | 81 ms | flat, input-length-independent |
| llama-3.1-8b | 859 s | 523 ms | fast but 40% accurate |
| gpt-oss-20b | 2,859 s | 9,633 ms | hidden reasoning tail |
| space-bunny-alpha | 3,881 s | 1,045 ms | free tier, presumably throttled |

## What this means for the paper

1. **On raw accuracy, Jev is competitive but not uniquely best.** It ties a free stealth model (space-bunny-alpha, p = 0.51) and loses to a 20B reasoning model that ignored the no-reasoning instruction (p = 7.5e-9). Claims of superior math ability are not supported; claims must rest elsewhere.
2. **Jev's defensible advantages are operational, not accuracy:** (a) it is the *only* system that returns a usable confidence — enabling deferral to 99.70% at 90% coverage and a 53.4%-coverage tier at a perfect 3,993/3,993 (`../analysis.md` §5); no chat baseline offers any confidence signal under this protocol; (b) zero parse failures by construction vs 43–95 empty answers; (c) the flattest latency (800 ± 81 ms) and the fastest full run; (d) mid-pack cost at $0.13, with exact billing reconciliation.
3. **The baseline also validates the difficulty of the task:** an 8B instruct model at 40% with collapsed letter marginals shows direct-choice MathQA is not trivially gameable, and the 4 universally-missed items give the paper concrete hard cases.
4. **Protocol honesty cuts both ways:** gpt-oss-20b demonstrates that "zero-shot direct" is unenforceable for reasoning models via prompt + `effort: low` alone — a methodological finding worth one paragraph in the paper.

## Reproduction

```bash
./venv/bin/python baseline/run_baseline.py        # the three runs (results_*.jsonl) — already executed
./venv/bin/python baseline/analyze_baseline.py    # regenerates baseline_stats.json + tables/, byte-identical
```

Per-item data: `results_meta-llama__llama-3.1-8b-instruct.jsonl`, `results_openai__gpt-oss-20b.jsonl`, `results_stealth__space-bunny-alpha.jsonl` (full response text, usage incl. reasoning-token counts, latency, cost per item).
