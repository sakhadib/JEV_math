#!/usr/bin/env python3
"""
Phase 2: run CompMath-MCQ (3-option, higher-level math) through Jev via OpenRouter.

Same discipline as ../jev.py: one typed `choice` call per item, multithreaded,
every result fsync'd to results_jev.jsonl immediately, resume by id.
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
    "Solve the mathematics problem given in the state. "
    "Which of the listed options is the correct answer?"
)

_write_lock = threading.Lock()
_rate_lock = threading.Lock()
_last_request_ts = 0.0
_progress_lock = threading.Lock()


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
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def load_completed_ids(path):
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
    data = json.dumps(record, ensure_ascii=False)
    with _write_lock:
        with open(path, "a") as f:
            f.write(data + "\n")
            f.flush()
            os.fsync(f.fileno())


def rate_limit(min_interval):
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
    letters = LETTERS[: len(item["options"])]
    criteria = {L: str(v) for L, v in zip(letters, item["options"])}
    payload = {
        "model": model,
        "state": item["question"],
        "questions": {
            "answer": {
                "type": "choice",
                "instructions": INSTRUCTIONS,
                "criteria": criteria,
            }
        },
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

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
            break
        except requests.RequestException as e:
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(min(2 ** attempts + random.random(), 60))

    latency_ms = round((time.monotonic() - t0) * 1000, 1)
    record = {
        "id": item["id"], "question": item["question"], "options": item["options"],
        "gold": item["correct_label"], "model_requested": model, "status": status,
        "attempts": attempts, "latency_ms": latency_ms,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    if status == "ok":
        answers = raw.get("answers", {})
        chosen, probs = extract_answer(answers.get("answer", {}))
        predicted_index = letters.index(chosen) if chosen in letters else None
        p_sorted = sorted(probs.values(), reverse=True)
        top = p_sorted[0] if p_sorted else None
        margin = (p_sorted[0] - p_sorted[1]) if len(p_sorted) > 1 else None
        entropy = -sum(p * math.log(p) for p in probs.values() if p > 0) if probs else None
        record.update({
            "model": raw.get("model"),
            "response_id": raw.get("id"),
            "provider": raw.get("provider"),
            "predicted_letter": chosen,
            "predicted_index": predicted_index,
            "predicted_value": item["options"][predicted_index] if predicted_index is not None else None,
            "correct": (predicted_index == item["correct_label"]) if predicted_index is not None else None,
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
    ap = argparse.ArgumentParser(description="Run CompMath-MCQ through Jev (System One).")
    ap.add_argument("--input", default="phase_2/mcq_lm_eval_data.jsonl")
    ap.add_argument("--output", default="phase_2/results_jev.jsonl")
    ap.add_argument("--model", default="jev-1.13")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-rps", type=float, default=15.0)
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
    print(f"dataset: {len(items)} | already done: {len(done)} | this run: {len(pending)}",
          file=sys.stderr)
    if not pending:
        print("nothing to do.", file=sys.stderr)
        return

    min_interval = 1.0 / args.max_rps if args.max_rps > 0 else 0
    prog = {"done": 0, "correct": 0, "errors": 0}
    t_start = time.monotonic()
    target = len(pending)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(call_jev, it, api_key, args.model, min_interval,
                               args.max_retries, args.timeout): it for it in pending}
        for fut in as_completed(futures):
            record = fut.result()
            append_result(args.output, record)
            with _progress_lock:
                prog["done"] += 1
                if record["status"] == "ok" and record.get("correct"):
                    prog["correct"] += 1
                if record["status"] != "ok":
                    prog["errors"] += 1
                d, c, e = prog["done"], prog["correct"], prog["errors"]
            if d % 25 == 0 or d == target:
                elapsed = time.monotonic() - t_start
                rate = d / elapsed if elapsed else 0
                eta = (target - d) / rate if rate else float("inf")
                acc = c / max(d - e, 1)
                print(f"[{d}/{target}] acc: {acc:.3f} | errors: {e} | "
                      f"{rate:.1f}/s | ETA {eta/60:.1f} min", file=sys.stderr)

    d, c, e = prog["done"], prog["correct"], prog["errors"]
    print(f"DONE. {d} items in {(time.monotonic()-t_start)/60:.1f} min | "
          f"acc {c / max(d - e, 1):.4f} | errors {e} -> {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
