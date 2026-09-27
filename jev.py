#!/usr/bin/env python3
"""
Can Jev do math? — run MathQA through TypeSafe's Jev (System One) model via OpenRouter.

Design:
  - One /api/v1/systemone call per dataset item: state = question text,
    a single `choice` question with criteria A-D mapped to the options.
  - Multithreaded (ThreadPoolExecutor), rate-limited under OpenRouter's caps.
  - Every completed result is appended to results.jsonl and fsync'd immediately.
  - Resume-safe: on startup, ids already present in results.jsonl are skipped.
"""

import argparse
import json
import math
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

ENDPOINT = "https://openrouter.ai/api/v1/systemone"
LETTERS = ["A", "B", "C", "D"]

INSTRUCTIONS = (
    "Solve the math word problem given in the state. "
    "Which of the listed options is the correct numerical answer?"
)

_write_lock = threading.Lock()
_rate_lock = threading.Lock()
_last_request_ts = 0.0
_progress_lock = threading.Lock()
_progress = {"done": 0, "correct": 0, "errors": 0, "skipped": 0}


def load_env(path=".env"):
    env = {}
    p = Path(path)
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def load_dataset(path):
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def load_completed_ids(path):
    """Ids already recorded (ok or error) — skipped on rerun. Tolerates a
    truncated final line from an interrupted write."""
    done = set()
    p = Path(path)
    if not p.exists():
        return done
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def append_result(path, record):
    """Append one JSON line and force it to disk immediately."""
    data = json.dumps(record, ensure_ascii=False)
    with _write_lock:
        with open(path, "a") as f:
            f.write(data + "\n")
            f.flush()
            os.fsync(f.fileno())


def rate_limit(min_interval):
    """Simple global pacing: at least min_interval seconds between request starts."""
    global _last_request_ts
    if min_interval <= 0:
        return
    with _rate_lock:
        now = time.monotonic()
        wait = _last_request_ts + min_interval - now
        if wait > 0:
            time.sleep(wait)
        _last_request_ts = time.monotonic()


def extract_answer(answer_obj):
    """Pull (chosen_letter, {letter: prob}) out of a Jev choice answer,
    tolerating shape differences across versions."""
    if not isinstance(answer_obj, dict):
        return None, {}
    probs = {}
    for key in ("probabilities", "distribution", "probs", "choices", "scores"):
        cand = answer_obj.get(key)
        if isinstance(cand, dict):
            probs = {str(k).upper(): float(v) for k, v in cand.items()
                     if isinstance(v, (int, float))}
            if probs:
                break
    chosen = None
    for key in ("choice", "selected", "answer", "value", "winner"):
        if isinstance(answer_obj.get(key), str):
            chosen = answer_obj[key].upper()
            break
    if chosen is None and probs:
        chosen = max(probs, key=probs.get)
    if not probs and chosen is not None:
        probs = {chosen: 1.0}
    return chosen, probs


def call_jev(item, api_key, model, min_interval, max_retries, timeout):
    """One item -> result record. Never raises; errors land in the record."""
    letters = LETTERS[: len(item["Choices"])]
    criteria = {L: str(v) for L, v in zip(letters, item["Choices"])}
    payload = {
        "model": model,
        "state": item["Question"],
        "questions": {
            "answer": {
                "type": "choice",
                "instructions": INSTRUCTIONS,
                "criteria": criteria,
            }
        },
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    attempts = 0
    t0 = time.monotonic()
    last_err = None
    raw = None
    status = "error"

    while attempts < max_retries:
        attempts += 1
        rate_limit(min_interval)
        try:
            resp = requests.post(ENDPOINT, json=payload, headers=headers, timeout=timeout)
            if resp.status_code == 200:
                raw = resp.json()
                status = "ok"
                break
            last_err = f"HTTP {resp.status_code}: {resp.text[:500]}"
            if resp.status_code in (429, 500, 502, 503, 504):
                time.sleep(min(2 ** attempts + random.random(), 60))
                continue
            break  # 4xx other than 429: not retryable
        except requests.RequestException as e:
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(min(2 ** attempts + random.random(), 60))

    latency_ms = round((time.monotonic() - t0) * 1000, 1)

    record = {
        "id": item["id"],
        "question": item["Question"],
        "choices": item["Choices"],
        "gold": item["label"],
        "model_requested": model,
        "status": status,
        "attempts": attempts,
        "latency_ms": latency_ms,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    if status == "ok":
        answers = raw.get("answers", {})
        chosen, probs = extract_answer(answers.get("answer", {}))
        predicted_index = letters.index(chosen) if chosen in letters else None
        p_sorted = sorted(probs.values(), reverse=True)
        top = p_sorted[0] if p_sorted else None
        margin = (p_sorted[0] - p_sorted[1]) if len(p_sorted) > 1 else None
        entropy = None
        if probs:
            entropy = -sum(p * math.log(p) for p in probs.values() if p > 0)
        record.update({
            "model": raw.get("model"),
            "response_id": raw.get("id"),
            "provider": raw.get("provider"),
            "predicted_letter": chosen,
            "predicted_index": predicted_index,
            "predicted_value": (
                item["Choices"][predicted_index] if predicted_index is not None else None
            ),
            "correct": (predicted_index == item["label"]) if predicted_index is not None else None,
            "probs": probs,
            "confidence": top,
            "margin": margin,
            "entropy_nats": entropy,
            "usage": raw.get("usage"),
            "raw": raw,
        })
    else:
        record["error"] = last_err

    return record


def main():
    ap = argparse.ArgumentParser(description="Run MathQA through Jev (System One) via OpenRouter.")
    ap.add_argument("--input", default="mathqa.jsonl")
    ap.add_argument("--output", default="results.jsonl")
    ap.add_argument("--model", default="jev-1.13")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=None, help="only run the first N pending items")
    ap.add_argument("--max-rps", type=float, default=15.0,
                    help="max requests per second (OpenRouter allows ~20/s)")
    ap.add_argument("--max-retries", type=int, default=6)
    ap.add_argument("--timeout", type=float, default=120.0)
    args = ap.parse_args()

    env = load_env()
    api_key = os.environ.get("OPENROUTER_API_KEY") or env.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("OPENROUTER_API_KEY not found in environment or .env")

    items = load_dataset(args.input)
    done = load_completed_ids(args.output)
    pending = [it for it in items if it["id"] not in done]
    if args.limit is not None:
        pending = pending[: args.limit]

    total = len(items)
    print(f"dataset: {total} items | already in {args.output}: {len(done)} | "
          f"to do this run: {len(pending)} | workers: {args.workers} | model: {args.model}",
          file=sys.stderr)
    if not pending:
        print("nothing to do — all items already have results.", file=sys.stderr)
        return

    min_interval = 1.0 / args.max_rps if args.max_rps > 0 else 0
    t_start = time.monotonic()
    target = len(pending)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(call_jev, it, api_key, args.model, min_interval,
                        args.max_retries, args.timeout): it
            for it in pending
        }
        for fut in as_completed(futures):
            record = fut.result()
            append_result(args.output, record)
            with _progress_lock:
                _progress["done"] += 1
                if record["status"] == "ok" and record.get("correct"):
                    _progress["correct"] += 1
                if record["status"] != "ok":
                    _progress["errors"] += 1
                d, c, e = _progress["done"], _progress["correct"], _progress["errors"]
            if d % 25 == 0 or d == target:
                elapsed = time.monotonic() - t_start
                rate = d / elapsed if elapsed > 0 else 0
                eta = (target - d) / rate if rate > 0 else float("inf")
                acc = c / max(d - e, 1) if d > e else 0
                print(f"[{d}/{target}] acc( this run ): {acc:.3f} | errors: {e} | "
                      f"{rate:.1f}/s | ETA {eta/60:.1f} min", file=sys.stderr)

    elapsed = time.monotonic() - t_start
    d, c, e = _progress["done"], _progress["correct"], _progress["errors"]
    acc = c / max(d - e, 1) if d > e else 0
    print(f"DONE. {d} items in {elapsed/60:.1f} min | run accuracy: {acc:.4f} | "
          f"errors: {e} | results -> {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
