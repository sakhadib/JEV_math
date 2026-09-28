#!/usr/bin/env python3
"""
Deterministic analysis of the chat-LLM baseline runs vs Jev.

Inputs:  baseline/results_<model>.jsonl (three files), ../results.jsonl (Jev run).
Outputs: baseline/baseline_stats.json, baseline/tables/*.csv
No randomness; byte-identical reruns.
"""

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sstats

LETTERS = ["A", "B", "C", "D"]
BASE = Path(__file__).parent
FILES = {
    "llama-3.1-8b-instruct": BASE / "results_meta-llama__llama-3.1-8b-instruct.jsonl",
    "gpt-oss-20b": BASE / "results_openai__gpt-oss-20b.jsonl",
    "space-bunny-alpha": BASE / "results_stealth__space-bunny-alpha.jsonl",
}
JEV = BASE.parent / "results.jsonl"
(BASE / "tables").mkdir(exist_ok=True)


def wilson(k, n, z=1.959964):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (p, max(0.0, centre - half), min(1.0, centre + half))


def load(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    ok = [r for r in rows if r["status"] == "ok"]
    return rows, ok


def mcnemar(correct_a, correct_b):
    """Two-sided exact binomial McNemar on discordant pairs."""
    a, b = np.asarray(correct_a, dtype=bool), np.asarray(correct_b, dtype=bool)
    b01 = int((a & ~b).sum())   # a right, b wrong
    c10 = int((~a & b).sum())   # a wrong, b right
    n = b01 + c10
    if n == 0:
        return b01, c10, 1.0
    p = sstats.binomtest(min(b01, c10), n, 0.5).pvalue
    return b01, c10, float(p)


out = {}

# ---- Jev reference
jev_rows, jev_ok = load(JEV)
jev_by_id = {r["id"]: r for r in jev_ok}
jev_correct = {r["id"]: bool(r["correct"]) for r in jev_ok}
N = len(jev_ok)
out["jev"] = {
    "N": N, "correct": int(sum(jev_correct.values())),
    "accuracy": sum(jev_correct.values()) / N,
    "cost_total": float(sum((r.get("usage") or {}).get("cost") or 0 for r in jev_ok)),
    "latency_p50": float(np.percentile([r["latency_ms"] for r in jev_ok if r["attempts"] == 1], 50)),
    "latency_mean": float(np.mean([r["latency_ms"] for r in jev_ok if r["attempts"] == 1])),
}

comparison_rows = []
per_model = {}

for name, path in FILES.items():
    rows, ok = load(path)
    n = len(ok)
    by_id = {r["id"]: r for r in ok}

    parsed = [r for r in ok if r.get("parse_ok")]
    unparsed = [r for r in ok if not r.get("parse_ok")]
    # primary metric: end-to-end accuracy (unparsed counts as wrong)
    correct_e2e = sum(1 for r in ok if r.get("correct") is True)
    acc_e2e, lo_e2e, hi_e2e = wilson(correct_e2e, n)
    # secondary: accuracy over parsed only
    correct_parsed = sum(1 for r in parsed if r["correct"])
    acc_p, lo_p, hi_p = wilson(correct_parsed, len(parsed))

    finish = Counter(r.get("finish_reason") for r in ok)
    parse_methods = Counter(r.get("parse_method") for r in ok)
    att = Counter(r.get("attempts") for r in ok)
    models_seen = Counter(r.get("model") for r in ok)

    prim = [r for r in ok if r["attempts"] == 1]
    lat = np.array([r["latency_ms"] for r in prim])
    cost = np.array([(r.get("usage") or {}).get("cost") or 0 for r in ok])
    ptok = np.array([(r.get("usage") or {}).get("prompt_tokens") or 0 for r in ok])
    ctok = np.array([(r.get("usage") or {}).get("completion_tokens") or 0 for r in ok])
    rtok = np.array([((r.get("usage") or {}).get("completion_tokens_details") or {}).get("reasoning_tokens") or 0 for r in ok])

    ts = sorted(r["ts"] for r in rows)
    from datetime import datetime
    wall = (datetime.fromisoformat(ts[-1].replace("Z", "+00:00")) -
            datetime.fromisoformat(ts[0].replace("Z", "+00:00"))).total_seconds()

    # position bias
    gold_letters = [LETTERS[r["gold"]] for r in ok]
    pred_letters = [r["parsed_letter"] for r in ok if r.get("parsed_letter")]
    gold_marg = Counter(gold_letters)
    pred_marg = Counter(pred_letters)
    chi2 = sstats.chisquare([pred_marg.get(L, 0) for L in LETTERS],
                            f_exp=[gold_marg[L] / n * len(pred_letters) for L in LETTERS])

    # empty-content parse failures
    empty_resp = sum(1 for r in unparsed if not (r.get("response_text") or "").strip())

    # McNemar vs Jev (paired on ids present in both)
    common = [i for i in jev_correct if i in by_id]
    c_jev = [jev_correct[i] for i in common]
    c_base = [bool(by_id[i].get("correct")) for i in common]
    b01, c10, p_mcn = mcnemar(c_jev, c_base)

    # items this model got wrong that Jev got right, and vice versa
    base_wrong_jev_right = sum(1 for i in common if jev_correct[i] and not by_id[i].get("correct"))

    per_model[name] = {
        "n": n, "correct_e2e": correct_e2e, "acc_e2e": acc_e2e,
        "wilson_e2e": [lo_e2e, hi_e2e],
        "parse_fail": len(unparsed), "parse_fail_empty_response": empty_resp,
        "acc_parsed_only": acc_p, "wilson_parsed": [lo_p, hi_p], "n_parsed": len(parsed),
        "finish_reasons": dict(finish), "parse_methods": dict(parse_methods),
        "attempts": {str(k): v for k, v in sorted(att.items())},
        "resolved_models": dict(models_seen),
        "latency": {"mean": float(lat.mean()), "sd": float(lat.std()),
                    "p50": float(np.percentile(lat, 50)), "p95": float(np.percentile(lat, 95)),
                    "min": float(lat.min()), "max": float(lat.max()), "n": int(len(lat))},
        "cost_total": float(cost.sum()), "cost_mean": float(cost.mean()),
        "cost_per_correct": float(cost.sum() / max(correct_e2e, 1)),
        "prompt_tokens_mean": float(ptok.mean()), "completion_tokens_mean": float(ctok.mean()),
        "reasoning_tokens_mean": float(rtok.mean()),
        "reasoning_tokens_nonzero_rows": int((rtok > 0).sum()),
        "wall_clock_s": wall, "throughput_per_s": n / wall if wall else None,
        "gold_marginals": dict(gold_marg), "pred_marginals": dict(pred_marg),
        "chi2_position": {"stat": float(chi2.statistic), "df": 3, "p": float(chi2.pvalue)},
        "mcnemar_vs_jev": {"jev_right_base_wrong": b01, "jev_wrong_base_right": c10,
                           "p": p_mcn, "n_common": len(common)},
    }

# error overlap across all four systems
base_correct = {}
for name, path in FILES.items():
    _, ok = load(path)
    base_correct[name] = {r["id"]: bool(r.get("correct")) for r in ok}
all_ids = set(jev_correct)
for name in FILES:
    all_ids &= set(base_correct[name])
wrong_all = [i for i in sorted(all_ids)
             if not jev_correct[i] and all(not base_correct[name][i] for name in FILES)]
out["all_four_wrong"] = {"n": len(wrong_all), "ids": wrong_all}

out["models"] = per_model

# comparison table
rows = []
rows.append({"system": "jev-1.13", "accuracy": out["jev"]["accuracy"],
             "n": N, "cost_total": out["jev"]["cost_total"],
             "latency_p50_ms": out["jev"]["latency_p50"],
             "confidence_signal": "yes (per-option distribution)",
             "never_confidently_wrong": "yes (0/3993 at p=1.0)"})
for name, m in per_model.items():
    rows.append({"system": name, "accuracy": m["acc_e2e"], "n": m["n"],
                 "cost_total": m["cost_total"], "latency_p50_ms": m["latency"]["p50"],
                 "confidence_signal": "no (bare letter)",
                 "never_confidently_wrong": "n/a"})
pd.DataFrame(rows).to_csv(BASE / "tables" / "comparison.csv", index=False)

per_model_tbl = []
for name, m in per_model.items():
    per_model_tbl.append({
        "model": name, "n": m["n"], "correct": m["correct_e2e"],
        "acc_e2e": m["acc_e2e"], "wilson_lo": m["wilson_e2e"][0], "wilson_hi": m["wilson_e2e"][1],
        "parse_fail": m["parse_fail"], "acc_parsed_only": m["acc_parsed_only"],
        "cost_total": m["cost_total"], "cost_per_correct": m["cost_per_correct"],
        "latency_mean_ms": m["latency"]["mean"], "latency_p50_ms": m["latency"]["p50"],
        "latency_p95_ms": m["latency"]["p95"], "wall_s": m["wall_clock_s"],
        "prompt_tok_mean": m["prompt_tokens_mean"], "completion_tok_mean": m["completion_tokens_mean"],
        "reasoning_tok_mean": m["reasoning_tokens_mean"],
        "chi2_p_position": m["chi2_position"]["p"],
        "mcnemar_p_vs_jev": m["mcnemar_vs_jev"]["p"],
        "jev_right_base_wrong": m["mcnemar_vs_jev"]["jev_right_base_wrong"],
        "jev_wrong_base_right": m["mcnemar_vs_jev"]["jev_wrong_base_right"],
    })
pd.DataFrame(per_model_tbl).to_csv(BASE / "tables" / "per_model.csv", index=False)

with open(BASE / "baseline_stats.json", "w") as f:
    json.dump(out, f, indent=2)

print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk in
      ("acc_e2e", "wilson_e2e", "parse_fail", "acc_parsed_only", "cost_total",
       "wall_clock_s", "mcnemar_vs_jev", "reasoning_tokens_nonzero_rows")}
      for k, v in per_model.items()}, indent=2))
print("all_four_wrong:", out["all_four_wrong"]["n"])
