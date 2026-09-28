#!/usr/bin/env python3
"""
Phase 2: CompMath-MCQ chat-LLM baseline via OpenRouter.

Zero-shot direct-choice protocol (same as phase 1): question + lettered options,
"Answer with only the letter ... Do not explain", temperature 0,
reasoning effort low. Multithreaded, fsync per result, resume by id.
"""

import argparse
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
LETTERS = ["A", "B", "C", "D"]

MODELS = [
    "meta-llama/llama-3.1-8b-instruct",
    "openai/gpt-oss-20b",
    "stealth/space-bunny-alpha",
]

PROMPT_TEMPLATE = """{question}

Options:
{options}

Answer with only the letter of the correct option ({letter_list}). Do not explain, do not show any work."""

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


def parse_answer(text, letters):
    if not text:
        return None, "empty"
    letter_class = "".join(letters)
    hits = re.findall(rf"ANSWER:\s*\(?([{letter_class}])\)?", text, flags=re.IGNORECASE)
    if hits:
        return hits[-1].upper(), "answer_tag"
    hits = re.findall(rf"answer is[:\s]*\(?([{letter_class}])\)?", text, flags=re.IGNORECASE)
    if hits:
        return hits[-1].upper(), "answer_is"
    tail = text.strip().splitlines()[-1].strip()
    m = re.fullmatch(rf"\(?([{letter_class}])\)?[.!]?", tail)
    if m:
        return m.group(1), "lone_letter"
    return None, "unparsed"


def call_model(item, api_key, model, min_interval, max_retries, timeout, max_tokens, temperature):
    letters = sorted(item["choices"].keys())  # A-E
    options_text = "\n".join(f"{L}) {item['choices'][L]}" for L in letters)
    gold_index = letters.index(item["answer_choice"])
    payload = {
        "model": model,
        "messages": [{"role": "user",
                      "content": PROMPT_TEMPLATE.format(
                          question=item["problem"], options=options_text,
                          letter_list=", ".join(letters))}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "reasoning": {"effort": "low"},
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
        "id": item["id"], "question": item["problem"], "options": item["choices"],
        "gold": gold_index, "gold_letter": item["answer_choice"],
        "level": item.get("level"), "subject": item.get("subject"),
        "contest": item.get("contest"), "year": item.get("year"),
        "model_requested": model, "status": status,
        "attempts": attempts, "latency_ms": latency_ms,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    if status == "ok":
        choice0 = (raw.get("choices") or [{}])[0]
        msg = choice0.get("message") or {}
        text = msg.get("content") or ""
        letter, method = parse_answer(text, letters)
        predicted_index = letters.index(letter) if letter in letters else None
        record.update({
            "model": raw.get("model"),
            "response_id": raw.get("id"),
            "provider": raw.get("provider"),
            "finish_reason": choice0.get("finish_reason"),
            "response_text": text,
            "reasoning_text": msg.get("reasoning"),
            "parsed_letter": letter,
            "parse_method": method,
            "parse_ok": letter is not None,
            "predicted_index": predicted_index,
            "predicted_value": item["choices"][letters[predicted_index]] if predicted_index is not None else None,
            "correct": (predicted_index == gold_index) if predicted_index is not None else None,
            "usage": raw.get("usage"),
        })
    else:
        record["error"] = last_err
    return record


def run_model(model, items, args, api_key):
    slug = model.replace("/", "__")
    out_path = Path(args.outdir) / f"results_{slug}.jsonl"
    done = load_completed_ids(out_path)
    pending = [it for it in items if it["id"] not in done]
    if args.limit is not None:
        pending = pending[: args.limit]
    print(f"\n=== {model} ===", file=sys.stderr)
    print(f"already done: {len(done)} | this run: {len(pending)} -> {out_path}", file=sys.stderr)
    if not pending:
        print("nothing to do.", file=sys.stderr)
        return

    min_interval = 1.0 / args.max_rps if args.max_rps > 0 else 0
    prog = {"done": 0, "correct": 0, "errors": 0, "parse_fail": 0}
    t_start = time.monotonic()
    target = len(pending)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(call_model, it, api_key, model, min_interval,
                               args.max_retries, args.timeout, args.max_tokens,
                               args.temperature): it for it in pending}
        for fut in as_completed(futures):
            record = fut.result()
            append_result(out_path, record)
            with _progress_lock:
                prog["done"] += 1
                if record["status"] == "ok" and record.get("correct"):
                    prog["correct"] += 1
                if record["status"] != "ok":
                    prog["errors"] += 1
                if record["status"] == "ok" and not record.get("parse_ok"):
                    prog["parse_fail"] += 1
                d, c, e, pf = prog["done"], prog["correct"], prog["errors"], prog["parse_fail"]
            if d % 25 == 0 or d == target:
                elapsed = time.monotonic() - t_start
                rate = d / elapsed if elapsed else 0
                eta = (target - d) / rate if rate else float("inf")
                acc = c / max(d - e - pf, 1)
                print(f"[{model} {d}/{target}] acc: {acc:.3f} | errors: {e} | "
                      f"parse_fail: {pf} | {rate:.1f}/s | ETA {eta/60:.1f} min",
                      file=sys.stderr)

    d, c, e, pf = prog["done"], prog["correct"], prog["errors"], prog["parse_fail"]
    print(f"DONE {model}: {d} items in {(time.monotonic()-t_start)/60:.1f} min | "
          f"acc {c / max(d - e - pf, 1):.4f} | errors {e} | parse_fail {pf}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description="CompMath-MCQ chat-LLM baseline via OpenRouter.")
    ap.add_argument("--input", default="phase_3/HARP_mcq.jsonl")
    ap.add_argument("--outdir", default="phase_3")
    ap.add_argument("--model", default=None)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-rps", type=float, default=10.0)
    ap.add_argument("--max-retries", type=int, default=6)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    env = load_env()
    api_key = os.environ.get("OPENROUTER_API_KEY") or env.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("OPENROUTER_API_KEY not found in environment or .env")

    Path(args.outdir).mkdir(exist_ok=True)
    items = load_dataset(args.input)
    models = [args.model] if args.model else MODELS
    for model in models:
        run_model(model, items, args, api_key)
    print("\nall baseline runs complete.", file=sys.stderr)


if __name__ == "__main__":
    main()
