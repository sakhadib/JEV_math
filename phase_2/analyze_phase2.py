#!/usr/bin/env python3
"""
Phase 2 analysis: CompMath-MCQ — Jev + three chat baselines.
Deterministic; outputs phase_2/stats.json, phase_2/tables/*.csv.
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
LETTERS = ["A", "B", "C"]
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


def skel(q):
    return re.sub(r"\s+", " ", re.sub(r"\d+(?:\.\d+)?", "#", q))


data = [json.loads(l) for l in open(BASE / "mcq_lm_eval_data.jsonl")]
N_data = len(data)
skels = Counter(skel(r["question"]) for r in data)
multi_ids = {r["id"] for r in data if skels[skel(r["question"])] > 1}

out = {"dataset": {
    "n": N_data, "options_per_item": 3,
    "gold_marginals": dict(Counter(r["correct_label"] for r in data)),
    "unique_skeletons": len(skels),
    "skeletons_multi": sum(1 for c in skels.values() if c > 1),
    "items_sharing_skeleton": len(multi_ids),
    "items_sharing_skeleton_pct": len(multi_ids) / N_data * 100,
}}

sys_correct = {}
per = {}
for name, path in SYSTEMS.items():
    ok, allrows = load(path)
    n = len(ok)
    by_id = {r["id"]: r for r in ok}

    if name == "jev-1.13":
        for r in ok:
            probs = {L: float(r["probs"].get(L, 0)) for L in LETTERS}
            best = max(LETTERS, key=lambda L: (probs[L], -LETTERS.index(L)))
            r["_correct"] = LETTERS.index(best) == r["gold"]
            r["_conf"] = max(probs.values())
            r["_rawconf"] = (r["raw"]["answers"]["answer"].get("confidence")
                             if r.get("raw") else None)
            r["_probs"] = probs
    else:
        for r in ok:
            r["_correct"] = bool(r.get("correct"))
            r["_conf"] = None

    correct = sum(r["_correct"] for r in ok)
    acc, lo, hi = wilson(correct, n)
    sys_correct[name] = {r["id"]: r["_correct"] for r in ok}

    lat = np.array([r["latency_ms"] for r in ok if r["attempts"] == 1])
    cost = np.array([(r.get("usage") or {}).get("cost") or 0 for r in ok])
    ts = sorted(r["ts"] for r in allrows)
    wall = (datetime.fromisoformat(ts[-1].replace("Z", "+00:00")) -
            datetime.fromisoformat(ts[0].replace("Z", "+00:00"))).total_seconds()

    pred_letters = Counter(r.get("parsed_letter") or r.get("predicted_letter") for r in ok)
    gold_letters = Counter(LETTERS[r["gold"]] for r in ok)
    n_pred = sum(pred_letters.get(L, 0) for L in LETTERS)
    chi2 = sstats.chisquare([pred_letters.get(L, 0) for L in LETTERS],
                            f_exp=[gold_letters[L] / n * n_pred for L in LETTERS])

    rec = {
        "n": n, "correct": int(correct), "acc": acc, "wilson": [lo, hi],
        "cost_total": float(cost.sum()), "cost_per_correct": float(cost.sum() / max(correct, 1)),
        "latency": {"mean": float(lat.mean()), "p50": float(np.percentile(lat, 50)),
                    "p95": float(np.percentile(lat, 95)), "sd": float(lat.std())},
        "wall_s": wall,
        "acc_template_items": float(np.mean([by_id[i]["_correct"] for i in multi_ids if i in by_id])),
        "acc_unique_items": float(np.mean([by_id[i]["_correct"] for i in by_id if i not in multi_ids])),
        "chi2_position_p": float(chi2.pvalue),
        "pred_marginals": dict(pred_letters),
    }

    if name == "jev-1.13":
        conf = np.array([r["_conf"] for r in ok])
        rawc = np.array([r["_rawconf"] for r in ok])
        deg = conf == 1.0
        wrong = np.array([not r["_correct"] for r in ok])
        rec["jev"] = {
            "degenerate_p1_frac": float(deg.mean()),
            "acc_at_p1": float((~wrong[deg]).mean()) if deg.any() else None,
            "n_at_p1": int(deg.sum()), "errors_at_p1": int((wrong & (conf == 1.0)).sum()),
            "errors": int(wrong.sum()),
            "mean_conf": float(conf.mean()), "mean_rawconf": float(rawc.mean()),
            "overconfidence_gap": float(conf.mean() - acc),
            "conf_distinct": int(len(set(conf.tolist()))),
            # ECE, equal-count 10 bins
            **(lambda qs, idx, labs: {
                "ece_eqcount10": float(sum(
                    m.sum() / n * abs(labs[m].mean() - conf[m].mean())
                    for m in (idx == b for b in range(len(qs) - 1)) if m.sum())),
            })(np.unique(np.quantile(conf, np.linspace(0, 1, 11))),
               np.clip(np.digitize(conf, np.unique(np.quantile(conf, np.linspace(0, 1, 11)))[1:-1], right=True),
                       0, len(np.unique(np.quantile(conf, np.linspace(0, 1, 11)))) - 2),
               (~wrong).astype(float)),
            # abstention
            "abstain": {kp: (lambda k_: {
                "dropped": k_,
                "retained_acc": float((~wrong)[np.argsort(conf, kind="mergesort")[k_:]].mean()),
                "retained_n": int(n - k_)})(int(round(n * kp / 100)))
                for kp in (1, 2, 5, 10)},
            "confidently_wrong_rawconf_ge99": int((wrong & (rawc >= 0.99)).sum()),
        }
    else:
        rt = np.array([((r.get("usage") or {}).get("completion_tokens_details") or {})
                       .get("reasoning_tokens") or 0 for r in ok])
        rec["parse_fail"] = sum(1 for r in ok if not r.get("parse_ok"))
        rec["acc_parsed_only"] = float(np.mean([r["_correct"] for r in ok if r.get("parse_ok")]))
        rec["reasoning_nonzero_rows"] = int((rt > 0).sum())
        rec["reasoning_mean"] = float(rt.mean())

    per[name] = rec

# McNemar vs Jev
common = set(sys_correct["jev-1.13"])
for name in SYSTEMS:
    common &= set(sys_correct[name])
for name in SYSTEMS:
    if name == "jev-1.13":
        continue
    b01, c10, p = mcnemar([sys_correct["jev-1.13"][i] for i in common],
                          [sys_correct[name][i] for i in common])
    per[name]["mcnemar_vs_jev"] = {"jev_right_base_wrong": b01,
                                   "jev_wrong_base_right": c10, "p": p}

wrong_all = [i for i in sorted(common) if not any(sys_correct[s][i] for s in SYSTEMS)]
out["all_systems_wrong"] = {"n": len(wrong_all), "ids": wrong_all}
out["systems"] = per

pd.DataFrame([{"system": s, "n": per[s]["n"], "correct": per[s]["correct"],
               "acc": per[s]["acc"], "wilson_lo": per[s]["wilson"][0],
               "wilson_hi": per[s]["wilson"][1], "cost_total": per[s]["cost_total"],
               "latency_p50": per[s]["latency"]["p50"], "wall_s": per[s]["wall_s"],
               "acc_template": per[s]["acc_template_items"],
               "acc_unique": per[s]["acc_unique_items"],
               "chi2_p_position": per[s]["chi2_position_p"]}
              for s in SYSTEMS]).to_csv(BASE / "tables" / "phase2_summary.csv", index=False)

with open(BASE / "stats.json", "w") as f:
    json.dump(out, f, indent=2, default=str)

for s in SYSTEMS:
    m = per[s]
    extra = ""
    if "jev" in m:
        j = m["jev"]
        extra = (f" | deg {j['degenerate_p1_frac']*100:.1f}% acc@p1 {j['acc_at_p1']:.4f} "
                 f"cw {j['errors_at_p1']}/{j['errors_at_p1']+0} ECE {j['ece_eqcount10']:.4f}")
    if "mcnemar_vs_jev" in m:
        extra += f" | mcnemar p={m['mcnemar_vs_jev']['p']:.3g}"
    print(f"{s:26s} acc={m['acc']:.4f} [{m['wilson'][0]:.4f},{m['wilson'][1]:.4f}] "
          f"tmpl={m['acc_template_items']:.3f} uniq={m['acc_unique_items']:.3f}{extra}")
print("wrong by all four:", out["all_systems_wrong"])
