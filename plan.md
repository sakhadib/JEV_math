# Plan

## calling jev

**Jev is NOT a chat model.** It is TypeSafe AI's "System One" model: instead of generating text token-by-token, it takes your app's **state** (unstructured text) plus one or more **typed questions** and returns **typed decisions with calibrated probabilities** in a single call. It cannot produce free text, type errors, or out-of-schema values. Think classification/routing/scoring, not conversation.

- **Model ID (OpenRouter):** `typesafe/jev-1.13` (pinned); `jev-1.13` bare ID is auto-namespaced to the same; `~typesafe/jev-latest` / `jev-latest` tracks the newest release
- **Endpoint (dedicated, NOT chat/completions):** `POST https://openrouter.ai/api/v1/systemone`
- **Auth:** `Authorization: Bearer $OPENROUTER_API_KEY`
- **Context:** 64K tokens/request direct (OpenRouter catalog lists 32K); input is text only
- **Pricing:** ~$0.042 / 1M input tokens, output free; per-request cost in `usage.cost`
- **Latency:** ~70–500ms; limits ~1,200 req/min, 250K tok/s
- **Note:** it does not appear in OpenRouter's `/api/v1/models` list (that only shows the `typesafe/jev-router` wrapper, which routes chat traffic through Jev). The SDK's `models.list()` doesn't work against OpenRouter either.

### Question types

| Type | Returns | Use |
|------|---------|-----|
| `noul` | probability 0–1 | yes/no questions ("is this correct?") |
| `choice` | one option + full probability distribution over `criteria` you define (up to 255) | picking a category/answer |
| `score` | a number on a scale you define | ratings, urgency |

### Request / response shape

Request: `model`, `state` (string/JSON/array to evaluate), `questions` (map of named typed questions).
Response: `{ id, model, provider, answers: { <name>: { type, ... } }, usage: { input_tokens, output_tokens, cost } }`

```bash
curl https://openrouter.ai/api/v1/systemone \
  -H "Authorization: Bearer $OPENROUTER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "jev-1.13",
    "state": "I was charged twice for my subscription.",
    "questions": {
      "refund": {"type": "noul", "instructions": "Is the customer asking for money back?"}
    }
  }'
```

Python SDK (`typesafe_sdk`):

```python
import os
from typesafe_sdk import TypeSafeClient

client = TypeSafeClient(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api",   # SDK appends /v1/systemone
)

result = client.system_one(
    model="jev-1.13",
    state="I was charged twice for my subscription.",
    questions={
        "refund": {"type": "noul", "instructions": "Is the customer asking for money back?"},
        "department": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {"billing": "Charges and refunds", "technical": "Bugs and outages"},
        },
    },
)
print(result.answers["refund"])  # { type: "noul", noul: 0.98 }
```

### Relevance to MathQA

Jev fits this dataset naturally as a `choice` question: `state` = the word problem, `criteria` = the answer options, and the returned probability distribution gives both a predicted answer and a calibrated confidence per item — no text parsing needed.

## research program: can jev do math?

**Question:** can Jev (a System One decision model, no chain-of-thought, no text generation) solve MathQA-style word problems, and how well-calibrated is its confidence?

### Data

`mathqa.jsonl` — 7,473 items, now with stable ids (`mathqa-00000` … `mathqa-07472`). Fields: `id`, `Question`, `Choices` (4 numeric options), `label` (gold index). Original pre-id version kept in `mathqa.jsonl.bak`.

### Per-item call

One `/api/v1/systemone` call per item:

- `state` = the question text (verbatim)
- one `choice` question named `answer`:
  - `instructions`: e.g. "Solve the math word problem in the state. Which option is the correct answer?"
  - `criteria`: options keyed `"A"`–`"D"` (letters, not values — values can collide), description = the stringified value
- prediction = argmax over the returned distribution, mapped back to the option index

### What to record per item (`results.jsonl`, one line per completed item)

| Field | Why |
|---|---|
| `id`, `question`, `choices`, `gold` | join back to dataset, audit |
| `predicted_index`, `predicted_value`, `correct` | accuracy |
| `probs` (full A–D distribution) | calibration, error analysis |
| `confidence` (top prob), `margin` (p1−p2), `entropy` | calibration / selective prediction |
| `usage` (input/output tokens), `cost` | cost analysis |
| `latency_ms` (client-measured) | speed claims check |
| `model` (exact resolved id from response), `response_id` | reproducibility |
| `raw` (full response JSON) | nothing lost; derive more later |
| `status` (`ok` / `error`), `error`, `attempts`, `ts` | resumability, failure analysis |

### Engineering requirements

- **Multithreading:** `ThreadPoolExecutor` with N workers (configurable, default ~8–16; keep under the 1,200 req/min limit — add a simple rate limiter if needed).
- **Immediate persistence:** each completed result is appended to `results.jsonl` and **flushed to disk instantly** (single writer guarded by a lock, `flush()` + `os.fsync()` per line). Nothing lives only in memory.
- **Resume:** on startup, scan `results.jsonl`, collect ids with `status == "ok"` (and optionally permanent errors), skip those; rerun picks up where it stopped. Interrupted/duplicate lines handled gracefully (last-write-wins by id when analyzing).
- **Retries:** exponential backoff with jitter on 429/5xx; after max attempts record an `error` row so the run still completes and the item can be retried selectively later.
- **Progress:** periodic stderr summary (done / remaining / running accuracy / elapsed / ETA).
- **Config via env/CLI flags:** model id (`jev-1.13` default), workers, `--limit N` for smoke tests, input/output paths.

### Analysis (after the run)

1. **Accuracy** overall; by confidence decile (does Jev know when it knows?).
2. **Calibration:** reliability curve, ECE, Brier score, NLL from the probability distributions.
3. **Selective prediction:** accuracy vs coverage curve (abstain below confidence threshold).
4. **Error analysis:** entropy/margin of wrong answers vs right ones; common failure shapes.
5. **Cost/latency:** total $, tokens, p50/p95 latency vs TypeSafe's claims (~70–500ms, $0.042/Mtok → full set ≈ a few cents).
