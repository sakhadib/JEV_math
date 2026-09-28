#!/usr/bin/env python3
"""
Phase 3 analysis: HARP (US competition math, 5 options, genuinely hard).
Deterministic; outputs phase_3/stats.json, phase_3/tables/*.csv.

Parsing policy: results files store the strict as-run parse (letter-only).
This script ADDITIONALLY re-parses stored response_text with extended patterns
(\\boxed{X}, "final answer is X") — reported as acc_extended_parse — because
verbose answers contain real decisions. Strict e2e remains the primary metric.
"""

import json
import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sstats

BASE = Path(__file__).parent
LETTERS = ["A", "B", "C", "D", "E"]
SYSTEMS = {
    "jev-1.13": BASE / "results_jev.jsonl",
    "llama-3.1-8b-instruct": BASE / "results_meta-llama__llama-3.1-8b-instruct.jsonl",
    "gpt-oss-20b": BASE / "results_openai__gpt-oss-20b.jsonl",
    "space-bunny-alpha": BASE / "results_stealth__space-bunny-alpha.jsonl",
}
(BASE / "tables").mkdir(exist_ok=True)


def wilson(k, n, z=1.959964):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, max(0.0, c - h), min(1.0, c + h))


def mcnemar(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    b01, c10 = int((a & ~b).sum()), int((~a & b).sum())
    n = b01 + c10
    return b01, c10, (sstats.binomtest(min(b01, c10), n, 0.5).pvalue if n else 1.0)


def load(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    return [r for r in rows if r["status"] == "ok"], rows


def parse_extended(text):
    if not text:
        return None
    hits = re.findall(r"\\boxed\{\s*\(?([A-E])\)?\s*\}", text)
    if hits:
        return hits[-1]
    hits = re.findall(r"final answer is[:\s]*\$?\\?(?:boxed\{)?\(?([A-E])\)?", text, flags=re.IGNORECASE)
    if hits:
        return hits[-1].upper()
    hits = re.findall(r"ANSWER:\s*\(?([A-E])\)?", text, flags=re.IGNORECASE)
    if hits:
        return hits[-1].upper()
    hits = re.findall(r"answer is[:\s]*\(?([A-E])\)?", text, flags=re.IGNORECASE)
    if hits:
        return hits[-1].upper()
    tail = text.strip().splitlines()[-1].strip()
    m = re.fullmatch(r"\(?([A-E])\)?[.!]?", tail)
    return m.group(1) if m else None


out = {"dataset": {"name": "HARP", "n": 4110, "options": 5}}
sys_correct_strict = {}
sys_correct_ext = {}
per = {}

for name, path in SYSTEMS.items():
    ok, allrows = load(path)
    n = len(ok)
    by_id = {r["id"]: r for r in ok}

    if name == "jev-1.13":
        for r in ok:
            probs = {L: float(r["probs"].get(L, 0)) for L in LETTERS}
            best = max(LETTERS, key=lambda L: (probs[L], -LETTERS.index(L)))
            r["_pred_strict"] = best
            r["_pred_ext"] = best
            r["_conf"] = max(probs.values())
            r["_rawconf"] = r["raw"]["answers"]["answer"].get("confidence") if r.get("raw") else None
            r["_probs"] = probs
    else:
        for r in ok:
            r["_pred_strict"] = r.get("parsed_letter")
            ext = r["_pred_strict"] or parse_extended(r.get("response_text") or "")
            r["_pred_ext"] = ext
            r["_conf"] = None

    for r in ok:
        r["_gold_letter"] = r["gold_letter"]
        r["_correct_strict"] = r["_pred_strict"] == r["_gold_letter"]
        r["_correct_ext"] = r["_pred_ext"] == r["_gold_letter"]

    cs = sum(r["_correct_strict"] for r in ok)
    ce = sum(r["_correct_ext"] for r in ok)
    acc_s, lo_s, hi_s = wilson(cs, n)
    acc_e, lo_e, hi_e = wilson(ce, n)
    sys_correct_strict[name] = {r["id"]: r["_correct_strict"] for r in ok}
    sys_correct_ext[name] = {r["id"]: r["_correct_ext"] for r in ok}

    lat = np.array([r["latency_ms"] for r in ok if r["attempts"] == 1])
    cost = np.array([(r.get("usage") or {}).get("cost") or 0 for r in ok])
    ts = sorted(r["ts"] for r in allrows)
    wall = (datetime.fromisoformat(ts[-1].replace("Z", "+00:00")) -
            datetime.fromisoformat(ts[0].replace("Z", "+00:00"))).total_seconds()

    pred_marg = Counter(r["_pred_strict"] for r in ok)
    gold_marg = Counter(r["_gold_letter"] for r in ok)
    n_pred = sum(pred_marg.get(L, 0) for L in LETTERS)
    chi2 = sstats.chisquare([pred_marg.get(L, 0) for L in LETTERS],
                            f_exp=[gold_marg[L] / n * n_pred for L in LETTERS])

    # slices: level and subject (extended parse)
    def slice_acc(key):
        out_s = {}
        for val in sorted({r.get(key) for r in ok}, key=str):
            sub = [r for r in ok if r.get(key) == val]
            k = sum(r["_correct_ext"] for r in sub)
            p, lo, hi = wilson(k, len(sub))
            out_s[str(val)] = {"n": len(sub), "correct": int(k), "acc": p,
                               "lo": lo, "hi": hi}
        return out_s

    rec = {
        "n": n, "correct_strict": int(cs), "acc_strict": acc_s, "wilson_strict": [lo_s, hi_s],
        "correct_extended": int(ce), "acc_extended": acc_e, "wilson_extended": [lo_e, hi_e],
        "cost_total": float(cost.sum()), "cost_per_correct_strict": float(cost.sum() / max(cs, 1)),
        "latency": {"mean": float(lat.mean()), "p50": float(np.percentile(lat, 50)),
                    "p95": float(np.percentile(lat, 95)), "sd": float(lat.std())},
        "wall_s": wall,
        "chi2_position_p": float(chi2.pvalue),
        "pred_marginals": {str(k): v for k, v in pred_marg.items()},
        "by_level": slice_acc("level"),
        "by_subject": slice_acc("subject"),
    }

    if name == "jev-1.13":
        conf = np.array([r["_conf"] for r in ok])
        rawc = np.array([r["_rawconf"] for r in ok])
        wrong = np.array([not r["_correct_strict"] for r in ok])
        deg = conf == 1.0
        qs = np.unique(np.quantile(conf, np.linspace(0, 1, 11)))
        idx = np.clip(np.digitize(conf, qs[1:-1], right=True), 0, len(qs) - 2)
        labs = (~wrong).astype(float)
        ece = float(sum(m.sum() / n * abs(labs[m].mean() - conf[m].mean())
                        for m in (idx == b for b in range(len(qs) - 1)) if m.sum()))
        rec["jev"] = {
            "degenerate_p1_frac": float(deg.mean()), "n_at_p1": int(deg.sum()),
            "acc_at_p1": float(labs[deg].mean()) if deg.any() else None,
            "errors_at_p1": int((wrong & deg).sum()), "errors": int(wrong.sum()),
            "mean_conf": float(conf.mean()), "mean_rawconf": float(rawc.mean()),
            "overconfidence_gap": float(conf.mean() - acc_s),
            "ece_eqcount10": ece,
            "confidently_wrong_rawconf_ge99": int((wrong & (rawc >= 0.99)).sum()),
            "abstain": {kp: (lambda k_: {
                "dropped": k_,
                "retained_acc": float(labs[np.argsort(conf, kind="mergesort")[k_:]].mean()),
                "retained_n": int(n - k_)})(int(round(n * kp / 100)))
                for kp in (1, 2, 5, 10)},
            "error_conf_hist": {str(k): v for k, v in
                                sorted(Counter(round(float(c), 1) for c in conf[wrong]).items())},
        }
    else:
        rt = np.array([((r.get("usage") or {}).get("completion_tokens_details") or {})
                       .get("reasoning_tokens") or 0 for r in ok])
        pf_strict = [r for r in ok if r["_pred_strict"] is None]
        rec["parse_fail_strict"] = len(pf_strict)
        rec["parse_fail_empty"] = sum(1 for r in pf_strict if not (r.get("response_text") or "").strip())
        rec["parse_fail_recovered_by_extended"] = sum(
            1 for r in pf_strict if r["_pred_ext"] is not None)
        rec["finish_reasons_of_failures"] = dict(Counter(r.get("finish_reason") for r in pf_strict))
        rec["reasoning_nonzero_rows"] = int((rt > 0).sum())
        rec["reasoning_mean"] = float(rt.mean())

    per[name] = rec

common = set(sys_correct_strict["jev-1.13"])
for name in SYSTEMS:
    common &= set(sys_correct_strict[name])
for name in SYSTEMS:
    if name == "jev-1.13":
        continue
    b, c, p = mcnemar([sys_correct_strict["jev-1.13"][i] for i in common],
                      [sys_correct_strict[name][i] for i in common])
    be, ce_, pe = mcnemar([sys_correct_strict["jev-1.13"][i] for i in common],
                          [sys_correct_ext[name][i] for i in common])
    per[name]["mcnemar_vs_jev_strict"] = {"jev_right": b, "base_right": c, "p": p}
    per[name]["mcnemar_vs_jev_extended"] = {"jev_right": be, "base_right": ce_, "p": pe}

wrong_all = [i for i in sorted(common) if not any(sys_correct_ext[s][i] for s in SYSTEMS)]
out["all_systems_wrong_extended"] = {"n": len(wrong_all), "ids": wrong_all}
out["systems"] = per

rows = []
for s in SYSTEMS:
    m = per[s]
    rows.append({"system": s, "n": m["n"], "correct_strict": m["correct_strict"],
                 "acc_strict": m["acc_strict"], "wilson_lo": m["wilson_strict"][0],
                 "wilson_hi": m["wilson_strict"][1], "acc_extended": m["acc_extended"],
                 "parse_fail_strict": m.get("parse_fail_strict", 0),
                 "cost_total": m["cost_total"], "latency_p50": m["latency"]["p50"],
                 "wall_s": m["wall_s"], "chi2_p_position": m["chi2_position_p"]})
pd.DataFrame(rows).to_csv(BASE / "tables" / "phase3_summary.csv", index=False)

with open(BASE / "stats.json", "w") as f:
    json.dump(out, f, indent=2, default=str)

for s in SYSTEMS:
    m = per[s]
    line = (f"{s:26s} strict={m['acc_strict']:.4f} ext={m['acc_extended']:.4f} "
            f"pf={m.get('parse_fail_strict', 0)}")
    if "jev" in m:
        j = m["jev"]
        line += (f" | deg {j['degenerate_p1_frac']*100:.1f}% acc@p1 {j['acc_at_p1']:.4f} "
                 f"cw {j['errors_at_p1']} ECE {j['ece_eqcount10']:.4f} gap {j['overconfidence_gap']:+.4f}")
    if "mcnemar_vs_jev_strict" in m:
        line += f" | mcn p={m['mcnemar_vs_jev_strict']['p']:.2g}"
    print(line)
print("wrong by all (extended):", out["all_systems_wrong_extended"]["n"])
