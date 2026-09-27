# JEV_math — Can Jev do math?

A mini research project testing whether **Jev**, TypeSafe AI's "System One" model (accessed via OpenRouter), can solve math word problems — and how well-calibrated its confidence is.

Unlike a chat model, Jev never generates text: it takes an unstructured **state** plus **typed questions** and returns **typed decisions with calibrated probabilities** in a single call. That makes it a natural fit for multiple-choice math: the word problem is the state, the four options are a `choice` question, and Jev returns a full probability distribution over the options — a predicted answer *and* a confidence signal, with no parsing.

See [`plan.md`](plan.md) for API details and the full research program.

## Dataset

`mathqa.jsonl` — 7,473 math word problems, each with 4 numeric options and a gold label:

```json
{"id": "mathqa-00000", "Question": "...", "Choices": [620, 10, 536, 379], "label": 1}
```

Source: [NLP2025-math](https://www.kaggle.com/competitions/nlp-2025-math) (Kaggle), permitted for **academic use** by the author. The raw data is not redistributed in this repo (it is gitignored, along with results files, which embed dataset content) — obtain it from the Kaggle competition page and place your copy at `mathqa.jsonl` in the repo root.

### Citation

```bibtex
@misc{nlp-2025-math,
    author = {zzzzzyyyy},
    title = {NLP2025-math},
    year = {2025},
    howpublished = {\url{https://www.kaggle.com/competitions/nlp-2025-math}},
    note = {Kaggle}
}
```

## Setup

```bash
python3 -m venv venv
./venv/bin/pip install requests
echo 'OPENROUTER_API_KEY=sk-or-...' > .env   # get a key at https://openrouter.ai/settings/keys
```

## Running the experiment

```bash
# smoke test — first 20 pending items
./venv/bin/python jev.py --limit 20

# full run — all 7,473 items
./venv/bin/python jev.py
```

Useful flags: `--workers N` (default 12), `--max-rps F` (default 15, OpenRouter allows ~20/s), `--model` (default `jev-1.13`), `--input` / `--output` paths.

### Resumability

Every completed item is appended to `results.jsonl` and **fsync'd to disk immediately** — nothing lives only in memory. If the run is interrupted (Ctrl+C, crash, network loss), just rerun the same command: items already present in `results.jsonl` are skipped and the run picks up where it left off. Retries with exponential backoff handle 429/5xx; items that exhaust retries are recorded with `status: "error"` so the run still completes.

## What gets recorded (per item, one JSON line in `results.jsonl`)

- **Prediction:** `predicted_letter/index/value`, `correct`
- **Calibration signals:** `probs` (full A–D distribution), `confidence` (top prob), `margin` (p1−p2), `entropy_nats`
- **Cost & speed:** `usage` (tokens), `cost`, `latency_ms` (client-measured)
- **Reproducibility:** resolved `model`, `response_id`, `provider`, `ts`, and the complete `raw` API response
- **Resilience:** `status`, `attempts`, `error` (when failed)

## Analysis plan

1. Overall accuracy; accuracy by confidence decile
2. Calibration: reliability curve, ECE, Brier score, NLL
3. Selective prediction: accuracy vs coverage as the confidence threshold varies
4. Error analysis: entropy/margin of wrong vs right answers
5. Cost & latency vs TypeSafe's published claims
