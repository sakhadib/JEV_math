# Experiment: Can Jev do math?

## What is Jev?

**Jev** is the first "System One" model from **TypeSafe AI**, accessed here through OpenRouter (model ID `typesafe/jev-1.13`). It is named after Kahneman's fast, intuitive "System 1" thinking — and it is **not a chat model**.

A conventional LLM generates text token-by-token; getting a structured answer out of it means parsing free text and hoping the format holds. Jev does the opposite:

1. You send it an unstructured **state** (any text) plus one or more **typed questions**.
2. It returns **typed decisions with calibrated probabilities** in a single call — never free text, never an out-of-schema value.

The three question types:

| Type | Returns | Example use |
|------|---------|-------------|
| `noul` | probability 0–1 | "Is this customer asking for a refund?" |
| `choice` | one option + full probability distribution over caller-defined options (up to 255) | classification, picking an answer |
| `score` | a number on a caller-defined scale | ratings, urgency |

Calls go to a dedicated endpoint, `POST https://openrouter.ai/api/v1/systemone` (not `/chat/completions`), with an OpenRouter API key. There is no chain-of-thought, no sampling, no system prompt — one state in, one typed distribution out, typically in well under a second, at roughly **$0.042 per million input tokens** (output free).

## The experiment

**Research question:** can a System One decision model solve math word problems — with no reasoning text, no working-out, just a single intuitive judgment — and is its confidence calibrated?

**Dataset:** [NLP2025-math](https://www.kaggle.com/competitions/nlp-2025-math) (Kaggle, academic use) — 7,473 math word problems, each with 4 numeric options and a gold label (`mathqa.jsonl`, not redistributed in this repo).

**Procedure** (implemented in `jev.py`): for each dataset item, one System One call —

- `state` = the question text, verbatim
- one `choice` question named `answer`:
  - instructions: *"Solve the math word problem given in the state. Which of the listed options is the correct numerical answer?"*
  - `criteria` = the four options keyed `A`–`D`
- prediction = argmax of the returned probability distribution; correct if it matches the gold index

The harness is multithreaded (12 workers, paced to ≤15 req/s), retries 429/5xx with exponential backoff, **fsyncs every result to `results.jsonl` the instant it completes**, and skips already-completed ids on restart (interrupt-safe, resumable).

**Run:** all 7,473 items through `jev-1.13` (resolved model `typesafe/jev-1.13-20260917`), 2026-09-27. Zero errors.

### Headline results

| Metric | Value |
|---|---|
| Accuracy | **96.92%** (7,243 / 7,473) |
| Errors (API) | 0 |
| Total cost | **$0.13** |
| Latency | p50 = 794 ms, p95 = 920 ms per item |

## Structure of `results.jsonl`

One JSON object per line, one per dataset item (fsync'd at completion; safe to resume / interrupt).

| Field | Meaning |
|---|---|
| `id`, `question`, `choices`, `gold` | joined from the dataset; `gold` = correct option index |
| `model_requested`, `model`, `provider`, `response_id`, `ts` | reproducibility: requested vs resolved model id, response id, UTC timestamp |
| `status` | `"ok"` or `"error"` |
| `attempts` | HTTP attempts used (retries included) |
| `latency_ms` | client-measured wall time for the call |
| `predicted_letter`, `predicted_index`, `predicted_value` | Jev's argmax answer, mapped back to the dataset |
| `correct` | `predicted_index == gold` |
| `probs` | full probability distribution over options `A`–`D` |
| `confidence` | top probability |
| `margin` | p(top1) − p(top2) |
| `entropy_nats` | Shannon entropy of the distribution (nats) |
| `usage` | `input_tokens`, `output_tokens`, `cost` (USD) |
| `raw` | the complete, unmodified API response — nothing is lost |
| `error` | error detail, present only when `status` = `"error"` |

## Example record

```json
{
  "id": "mathqa-00006",
  "question": "Leo and Ryan together have $48. Ryan owns 2/3 of the amount. Leo remembered that Ryan owed him $10 but he also owed Ryan $7. After the debts had been settled, how much money does Leo have?",
  "choices": [71, 505, 250, 19],
  "gold": 3,
  "model_requested": "jev-1.13",
  "status": "ok",
  "attempts": 1,
  "latency_ms": 785.4,
  "ts": "2026-09-27T22:43:47Z",
  "model": "typesafe/jev-1.13-20260917",
  "response_id": "gen-dec-1790549027-jLBjBF8S0QxnGghwFc97",
  "provider": "TypeSafe",
  "predicted_letter": "D",
  "predicted_index": 3,
  "predicted_value": 19,
  "correct": true,
  "probs": {"A": 0.0, "B": 0.0, "C": 0.0, "D": 1.0},
  "confidence": 1.0,
  "margin": 1.0,
  "entropy_nats": -0.0,
  "usage": {"input_tokens": 402, "output_tokens": 45, "cost": 1.6884e-05},
  "raw": {
    "model": "typesafe/jev-1.13-20260917",
    "answers": {
      "answer": {
        "type": "choice",
        "choice": "D",
        "probabilities": {"A": 0, "B": 0, "C": 0, "D": 1},
        "confidence": 0.99
      }
    },
    "usage": {"input_tokens": 402, "output_tokens": 45, "cost": 1.6884e-05},
    "id": "gen-dec-1790549027-jLBjBF8S0QxnGghwFc97",
    "provider": "TypeSafe"
  }
}
```

Here Jev placed 100% probability on option D (19) — the correct answer — in a single ~785 ms judgment call costing $0.000017.
