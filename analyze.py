#!/usr/bin/env python3
"""
Deterministic analysis of results.jsonl for "Can Jev do math?".

Produces: figures/*.png, tables/*.csv, errors_sample.md, derived.csv, stats.json.
All seeds fixed. Running twice produces byte-identical outputs.
Never modifies results.jsonl.
"""

import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sstats

SEED = 42
rng = np.random.RandomState(SEED)
BOOT_SEED = 20260928
N_BOOT = 2000
NLL_EPS = 1e-3

RESULTS = "results.jsonl"
LETTERS = ["A", "B", "C", "D"]
SCHEMA_FIELDS = ["id", "question", "choices", "gold", "model_requested", "status",
                 "attempts", "latency_ms", "ts", "model", "response_id", "provider",
                 "predicted_letter", "predicted_index", "predicted_value", "correct",
                 "probs", "confidence", "margin", "entropy_nats", "usage", "raw"]

Path("figures").mkdir(exist_ok=True)
Path("tables").mkdir(exist_ok=True)

STATS = {}          # scalars for the report
T0 = time.monotonic()


def savefig(fig, name):
    fig.savefig(f"figures/{name}", dpi=150, bbox_inches="tight")
    plt.close(fig)


def wilson(k, n, z=1.959964):
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (p, max(0.0, centre - half), min(1.0, centre + half))


def auroc(scores, labels):
    """Rank-based AUROC with tie handling. labels: 1 = correct."""
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    s_sorted = np.asarray(scores)[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    labels = np.asarray(labels)
    n1 = labels.sum()
    n0 = len(labels) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return (ranks[labels == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


# ---------------------------------------------------------------- Phase 0
rows_raw = []
parse_fail = 0
with open(RESULTS) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            rows_raw.append(json.loads(line))
        except json.JSONDecodeError:
            parse_fail += 1

line_count = len(rows_raw) + parse_fail
seen = set()
rows = []
dup_ids = 0
for r in rows_raw:
    if r.get("id") in seen:
        dup_ids += 1
        continue
    seen.add(r.get("id"))
    rows.append(r)

status_counts = Counter(r.get("status") for r in rows)
error_rows = [r for r in rows if r.get("status") != "ok"]
ok_rows = [r for r in rows if r.get("status") == "ok"]

attempts_counts = Counter(r.get("attempts") for r in rows)
model_ids = Counter(r.get("model") for r in ok_rows)

schema_missing = {}
for field in SCHEMA_FIELDS:
    schema_missing[field] = sum(1 for r in rows if field not in r or r[field] is None)

gold_bad = sum(1 for r in ok_rows if r.get("gold") not in (0, 1, 2, 3))
choices_bad = sum(1 for r in ok_rows if len(r.get("choices", [])) != 4)

dup_option_rows = []
for r in ok_rows:
    ch = r["choices"]
    if len(set(map(str, ch))) != len(ch):
        dup_option_rows.append(r)

# internal consistency: recompute from probs
consistency = Counter()
for r in ok_rows:
    probs = r.get("probs") or {}
    p = {L: float(probs.get(L, 0.0)) for L in LETTERS}
    # argmax, ties -> earliest letter
    bestL = max(LETTERS, key=lambda L: (p[L], -LETTERS.index(L)))
    recomp_index = LETTERS.index(bestL)
    if r.get("predicted_index") != recomp_index:
        consistency["predicted_index_mismatch"] += 1
    if r["choices"][recomp_index] != r.get("predicted_value"):
        consistency["predicted_value_mismatch"] += 1
    if (recomp_index == r["gold"]) != bool(r.get("correct")):
        consistency["correct_mismatch"] += 1
    conf = max(p.values())
    if abs(conf - float(r.get("confidence", -1))) > 1e-9:
        consistency["confidence_mismatch"] += 1
    svals = sorted(p.values(), reverse=True)
    margin = svals[0] - svals[1]
    if abs(margin - float(r.get("margin", -1))) > 1e-9:
        consistency["margin_mismatch"] += 1
    ent = -sum(v * math.log(v) for v in p.values() if v > 0)
    if abs(ent - float(r.get("entropy_nats", -1))) > 1e-6:
        consistency["entropy_mismatch"] += 1
    r["_probs"] = p
    r["_pred_index"] = recomp_index
    r["_correct"] = recomp_index == r["gold"]
    r["_conf"] = conf
    r["_margin"] = margin
    r["_entropy"] = ent
    # second confidence signal
    try:
        r["_rawconf"] = float(r["raw"]["answers"]["answer"]["confidence"])
    except (KeyError, TypeError, ValueError):
        r["_rawconf"] = float("nan")

ts_vals = sorted(r["ts"] for r in rows if r.get("ts"))
ts_first, ts_last = ts_vals[0], ts_vals[-1]

N = len(ok_rows)
STATS["phase0"] = {
    "line_count": line_count, "parse_failures": parse_fail,
    "duplicate_ids_removed": dup_ids, "status_counts": dict(status_counts),
    "n_error_rows_excluded": len(error_rows),
    "error_rows_verbatim": [{k: r.get(k) for k in ("id", "error", "attempts")} for r in error_rows],
    "attempts_distribution": {str(k): v for k, v in sorted(attempts_counts.items())},
    "resolved_model_ids": dict(model_ids),
    "schema_missing": schema_missing,
    "gold_out_of_range": gold_bad, "choices_not_4": choices_bad,
    "duplicate_option_rows": len(dup_option_rows),
    "duplicate_option_ids": [r["id"] for r in dup_option_rows],
    "consistency_failures": dict(consistency),
    "ts_first": ts_first, "ts_last": ts_last,
    "N_analysis": N,
}
pd.DataFrame([{"field": k, "missing_or_null": v} for k, v in schema_missing.items()]
             ).to_csv("tables/schema_completeness.csv", index=False)
pd.DataFrame([{"check": k, "failures": v} for k, v in consistency.items()]
             ).to_csv("tables/consistency_checks.csv", index=False)

# derived dataframe (per-item, re-checkable)
derived = pd.DataFrame([{
    "id": r["id"], "gold": r["gold"], "pred_index": r["_pred_index"],
    "correct": r["_correct"], "pA": r["_probs"]["A"], "pB": r["_probs"]["B"],
    "pC": r["_probs"]["C"], "pD": r["_probs"]["D"], "conf_top1": r["_conf"],
    "raw_confidence": r["_rawconf"], "margin": r["_margin"],
    "entropy_nats": r["_entropy"], "latency_ms": r["latency_ms"],
    "attempts": r["attempts"], "input_tokens": (r.get("usage") or {}).get("input_tokens"),
    "output_tokens": (r.get("usage") or {}).get("output_tokens"),
    "cost": (r.get("usage") or {}).get("cost"), "ts": r["ts"],
    "question": r["question"], "choices": json.dumps(r["choices"]),
    "pred_value": r["choices"][r["_pred_index"]], "gold_value": r["choices"][r["gold"]],
} for r in ok_rows])
derived.to_csv("derived.csv", index=False)

# ---------------------------------------------------------------- Phase 1
k = int(derived["correct"].sum())
acc, acc_lo, acc_hi = wilson(k, N)
gold_letters = [LETTERS[g] for g in derived["gold"]]
pred_letters = [LETTERS[i] for i in derived["pred_index"]]
gold_marg = Counter(gold_letters)
baselines = {"uniform_random (analytic)": 0.25}
for L in LETTERS:
    baselines[f"always_{L}"] = gold_marg[L] / N
best_fixed = max(v for k_, v in baselines.items() if k_.startswith("always_"))

dup_ids_set = {r["id"] for r in dup_option_rows}
dup_mask = derived["id"].isin(dup_ids_set)
acc_excl_dup = derived.loc[~dup_mask, "correct"].mean() if (~dup_mask).any() else float("nan")
acc_dup_only = derived.loc[dup_mask, "correct"].mean() if dup_mask.any() else float("nan")

STATS["phase1"] = {
    "correct": k, "incorrect": N - k, "N": N,
    "accuracy": acc, "wilson_lo": acc_lo, "wilson_hi": acc_hi,
    "baselines": baselines, "best_fixed_letter_baseline": best_fixed,
    "accuracy_excluding_duplicate_option_rows": acc_excl_dup,
    "accuracy_duplicate_option_rows_only": acc_dup_only,
    "n_excluding_dup": int((~dup_mask).sum()),
}
pd.DataFrame([{"metric": "accuracy", "value": acc, "lo": acc_lo, "hi": acc_hi, "n": N}]
             + [{"metric": m, "value": v, "lo": "", "hi": "", "n": N}
                for m, v in baselines.items()]
             ).to_csv("tables/headline.csv", index=False)

# ---------------------------------------------------------------- Phase 2
all_probs = np.concatenate([[r["_probs"][L] for L in LETTERS] for r in ok_rows])
distinct_vals = sorted(set(all_probs.tolist()))
nonzero = [v for v in distinct_vals if v > 0]
two_decimal = all(abs(v * 100 - round(v * 100)) < 1e-9 for v in distinct_vals)

top1 = derived["conf_top1"].to_numpy()
degen_exact1 = float((top1 == 1.0).mean())
degen_ge99 = float((top1 >= 0.99).mean())
has_zero = float((np.array([[r["_probs"][L] for L in LETTERS] for r in ok_rows]) == 0.0).any(axis=1).mean())
nonzero_options = Counter(int(sum(1 for L in LETTERS if r["_probs"][L] > 0)) for r in ok_rows)
sums = np.array([sum(r["_probs"][L] for L in LETTERS) for r in ok_rows])
sum_dev = int((np.abs(sums - 1.0) > 1e-6).sum())

rawconf = derived["raw_confidence"].to_numpy()
rawconf_nan = int(np.isnan(rawconf).sum())
rawconf_distinct = sorted(set(rawconf[~np.isnan(rawconf)].tolist()))
disagree = int((np.abs(top1 - rawconf) > 1e-9).sum())
valid = ~np.isnan(rawconf)
pearson_c = float(sstats.pearsonr(top1[valid], rawconf[valid]).statistic) if valid.any() else float("nan")
spearman_c = float(sstats.spearmanr(top1[valid], rawconf[valid]).statistic) if valid.any() else float("nan")

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(all_probs, bins=100, log=True)
ax.set_xlabel("probability value")
ax.set_ylabel("count (log scale)")
ax.set_title(f"All assigned probability values (n={len(all_probs):,} values from N={N} items)")
savefig(fig, "prob_values_hist.png")

STATS["phase2"] = {
    "n_distinct_prob_values": len(distinct_vals),
    "distinct_prob_values": [round(v, 6) for v in distinct_vals],
    "smallest_nonzero": min(nonzero) if nonzero else None,
    "consistent_with_2dp_rounding": two_decimal,
    "top1_exactly_1.0_frac": degen_exact1,
    "top1_ge_0.99_frac": degen_ge99,
    "rows_with_any_zero_option_frac": has_zero,
    "nonzero_option_count_dist": {str(k_): v for k_, v in sorted(nonzero_options.items())},
    "probs_sum_min": float(sums.min()), "probs_sum_max": float(sums.max()),
    "probs_sum_dev_gt_1e-6": sum_dev,
    "rawconf_missing": rawconf_nan,
    "rawconf_n_distinct": len(rawconf_distinct),
    "rawconf_distinct": [round(v, 6) for v in rawconf_distinct],
    "rawconf_min": float(np.nanmin(rawconf)), "rawconf_max": float(np.nanmax(rawconf)),
    "top1_vs_rawconf_disagree_n": disagree,
    "top1_vs_rawconf_pearson": pearson_c, "top1_vs_rawconf_spearman": spearman_c,
}

# ---------------------------------------------------------------- Phase 3
def calibration(score, labels, scheme, n_bins=10):
    """Equal-count or equal-width binned calibration. Returns bin df, ECE, MCE."""
    score = np.asarray(score, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if scheme == "eqcount":
        qs = np.quantile(score, np.linspace(0, 1, n_bins + 1))
        edges = np.unique(qs)
        if len(edges) < 3:  # too few distinct values -> degenerate binning
            edges = np.linspace(score.min(), score.max() + 1e-12, n_bins + 1)
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(score, edges[1:-1], right=True), 0, len(edges) - 2)
    recs, ece, mce = [], 0.0, 0.0
    for b in range(len(edges) - 1):
        m = idx == b
        nb = int(m.sum())
        if nb == 0:
            recs.append({"bin": b, "lo": edges[b], "hi": edges[b + 1], "n": 0,
                         "mean_conf": float("nan"), "acc": float("nan")})
            continue
        mc = float(score[m].mean())
        ac = float(labels[m].mean())
        recs.append({"bin": b, "lo": edges[b], "hi": edges[b + 1], "n": nb,
                     "mean_conf": mc, "acc": ac})
        ece += nb / len(score) * abs(ac - mc)
        mce = max(mce, abs(ac - mc))
    return pd.DataFrame(recs), float(ece), float(mce)


def reliability_plot(df, title, name):
    df = df[df["n"] > 0]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    ax.errorbar(df["mean_conf"], df["acc"],
                xerr=[np.clip(df["mean_conf"] - df["lo"], 0, None),
                      np.clip(df["hi"] - df["mean_conf"], 0, None)],
                fmt="o-", capsize=3, label="observed")
    for _, row in df.iterrows():
        ax.annotate(f'{int(row["n"])}', (row["mean_conf"], row["acc"]),
                    textcoords="offset points", xytext=(0, 7), fontsize=7, ha="center")
    ax.set_xlabel("mean confidence (bin)")
    ax.set_ylabel("accuracy (bin)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    savefig(fig, name)


labels_bin = derived["correct"].astype(int).to_numpy()
probs_mat = derived[["pA", "pB", "pC", "pD"]].to_numpy()
gold_idx = derived["gold"].to_numpy()

cal = {}
for name, score in [("top1", top1), ("rawconf", rawconf)]:
    cal[name] = {}
    for scheme in ("eqcount", "eqwidth"):
        df, ece, mce = calibration(score, labels_bin, scheme)
        df.to_csv(f"tables/calibration_{name}_{scheme}.csv", index=False)
        cal[name][scheme] = {"ece": ece, "mce": mce}
        reliability_plot(df, f"Reliability ({name}, {scheme}, 10 bins, N={N})",
                         f"reliability_{name}_{scheme}.png")
    # zoomed top-decile diagram: fine bins in [0.9, 1.0]
    mask_zoom = score >= 0.9
    if mask_zoom.sum() > 50:
        edges = np.linspace(0.9, 1.0000001, 11)
        idx = np.clip(np.digitize(score[mask_zoom], edges[1:-1], right=True), 0, 9)
        recs = []
        for b in range(10):
            m = idx == b
            nb = int(m.sum())
            recs.append({"bin": b, "lo": edges[b], "hi": edges[b + 1], "n": nb,
                         "mean_conf": float(score[mask_zoom][m].mean()) if nb else float("nan"),
                         "acc": float(labels_bin[mask_zoom][m].mean()) if nb else float("nan")})
        zdf = pd.DataFrame(recs)
        zdf.to_csv(f"tables/calibration_{name}_zoom.csv", index=False)
        reliability_plot(zdf, f"Reliability zoom 0.9–1.0 ({name}, N={N})",
                         f"reliability_{name}_zoom.png")

# multiclass Brier
onehot = np.zeros_like(probs_mat)
onehot[np.arange(N), gold_idx] = 1.0
brier_mc = float(np.mean(np.sum((probs_mat - onehot) ** 2, axis=1)))

# binary Brier decomposition on (top1 confidence, correct), eqcount bins
df_b, _, _ = calibration(top1, labels_bin, "eqcount")
df_b = df_b[df_b["n"] > 0]
overall_acc = labels_bin.mean()
rel = float((df_b["n"] * (df_b["acc"] - df_b["mean_conf"]) ** 2).sum() / N)
res = float((df_b["n"] * (df_b["acc"] - overall_acc) ** 2).sum() / N)
unc = float(overall_acc * (1 - overall_acc))
brier_bin = float(np.mean((top1 - labels_bin) ** 2))

gold_probs = probs_mat[np.arange(N), gold_idx]
n_gold_zero = int((gold_probs == 0.0).sum())
nll = float(-np.mean(np.log(np.clip(gold_probs, NLL_EPS, 1.0))))
nll_raw_inf = int((gold_probs == 0.0).sum())

overconf_gap_top1 = float(top1.mean() - overall_acc)
overconf_gap_raw = float(np.nanmean(rawconf) - overall_acc)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, (nm, sc) in zip(axes, [("top-1 probability", top1), ("raw.confidence", rawconf)]):
    bins = np.linspace(np.nanmin(sc), 1.0, 41)
    ax.hist(sc[labels_bin == 1], bins=bins, alpha=0.6, label=f"correct (n={int(labels_bin.sum())})")
    ax.hist(sc[labels_bin == 0], bins=bins, alpha=0.6, label=f"incorrect (n={int((1-labels_bin).sum())})")
    ax.set_xlabel(nm)
    ax.set_ylabel("count")
    ax.set_yscale("log")
    ax.legend()
fig.suptitle(f"Confidence by correctness (log count, N={N})")
savefig(fig, "conf_hist_correct_incorrect.png")

STATS["phase3"] = {
    "calibration": cal,
    "brier_multiclass": brier_mc,
    "brier_binary_top1": brier_bin,
    "brier_decomposition_eqcount": {"reliability": rel, "resolution": res, "uncertainty": unc,
                                    "rel_minus_res_plus_unc": rel - res + unc},
    "n_gold_prob_exactly_zero": n_gold_zero,
    "nll_eps_1e-3": nll,
    "overconfidence_gap_top1": overconf_gap_top1,
    "overconfidence_gap_rawconf": overconf_gap_raw,
    "mean_top1": float(top1.mean()), "mean_rawconf": float(np.nanmean(rawconf)),
    "zoom_bin_mass_note": "eqcount bins degenerate when >90% of mass is at 1.0",
}

# ---------------------------------------------------------------- Phase 4
scores_dict = {
    "top1_prob": top1,
    "raw_confidence": rawconf,
    "margin": derived["margin"].to_numpy(),
    "neg_entropy": -derived["entropy_nats"].to_numpy(),
}

fig, ax = plt.subplots(figsize=(7, 5))
sel = {}
boot = np.random.RandomState(BOOT_SEED)
for nm, s in scores_dict.items():
    s = np.asarray(s, dtype=float)
    order = np.argsort(-s, kind="mergesort")
    lab_sorted = labels_bin[order]
    cum_correct = np.cumsum(lab_sorted)
    coverage = np.arange(1, N + 1) / N
    risk = 1 - cum_correct / np.arange(1, N + 1)
    aurc = float(np.trapezoid(risk, coverage))
    auc = auroc(s, labels_bin)
    # bootstrap CI for AUROC
    boots = []
    for _ in range(N_BOOT):
        ii = boot.randint(0, N, N)
        if labels_bin[ii].sum() in (0, N):
            continue
        boots.append(auroc(s[ii], labels_bin[ii]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    sel[nm] = {"aurc": aurc, "auroc": float(auc), "auroc_lo": float(lo), "auroc_hi": float(hi)}
    ax.plot(coverage, risk, label=f"{nm} (AURC={aurc:.4f})")
ax.set_xlabel("coverage (fraction retained)")
ax.set_ylabel("error rate among retained")
ax.set_title(f"Risk–coverage, all scores (N={N})")
ax.legend()
ax.set_yscale("log")
savefig(fig, "risk_coverage.png")

# deferral tables: thresholds from empirical quantiles
def deferral_table(s, nm):
    qs = [0.0, 0.01, 0.02, 0.05, 0.10, 0.25, 0.5]
    cand = sorted(set(round(float(np.quantile(s, q)), 6) for q in qs))
    recs = []
    for t in cand:
        keep = s >= t
        nk = int(keep.sum())
        recs.append({"threshold": t, "coverage": nk / N,
                     "retained_accuracy": float(labels_bin[keep].mean()) if nk else float("nan"),
                     "retained_n": nk, "deferred_n": N - nk,
                     "deferred_error_rate": float(1 - labels_bin[~keep].mean()) if N - nk else float("nan")})
    df = pd.DataFrame(recs)
    df.to_csv(f"tables/deferral_{nm}.csv", index=False)
    return recs

deferral = {nm: deferral_table(np.asarray(s, float), nm) for nm, s in scores_dict.items()}

abstain = {}
for nm, s in scores_dict.items():
    s = np.asarray(s, float)
    order = np.argsort(s, kind="mergesort")  # ascending: least confident first
    abstain[nm] = {}
    for kp in (1, 2, 5, 10):
        kdrop = int(round(N * kp / 100))
        kept = labels_bin[order[kdrop:]]
        abstain[nm][kp] = {"dropped": kdrop, "retained_acc": float(kept.mean()),
                           "retained_n": int(len(kept))}

cw_top1 = int(((top1 == 1.0) & (labels_bin == 0)).sum())
cw_raw = int(((rawconf >= 0.99) & (labels_bin == 0)).sum())
n_err = int((labels_bin == 0).sum())

pd.DataFrame([{"score": nm, **sel[nm]} for nm in scores_dict]).to_csv("tables/selective.csv", index=False)
pd.DataFrame([{"score": nm, "k_pct": kp, **abstain[nm][kp]}
              for nm in scores_dict for kp in (1, 2, 5, 10)]).to_csv("tables/abstain.csv", index=False)

STATS["phase4"] = {
    "selective": sel, "deferral": deferral, "abstain": abstain,
    "confidently_wrong_top1_eq1": cw_top1, "confidently_wrong_rawconf_ge99": cw_raw,
    "n_errors": n_err,
    "confidently_wrong_top1_share_of_errors": cw_top1 / n_err if n_err else None,
    "confidently_wrong_raw_share_of_errors": cw_raw / n_err if n_err else None,
}

# ---------------------------------------------------------------- Phase 5
pred_marg = Counter(pred_letters)
marg_tbl = []
for L in LETTERS:
    marg_tbl.append({"letter": L, "gold_n": gold_marg[L], "gold_pct": gold_marg[L] / N * 100,
                     "pred_n": pred_marg[L], "pred_pct": pred_marg[L] / N * 100})
pd.DataFrame(marg_tbl).to_csv("tables/position_marginals.csv", index=False)

gold_counts = np.array([gold_marg[L] for L in LETTERS], dtype=float)
pred_counts = np.array([pred_marg[L] for L in LETTERS], dtype=float)
chi2 = sstats.chisquare(pred_counts, f_exp=gold_counts)
per_letter = []
for L in LETTERS:
    m = np.array(gold_letters) == L
    kk = int(labels_bin[m].sum())
    p, lo, hi = wilson(kk, int(m.sum()))
    per_letter.append({"gold_letter": L, "n": int(m.sum()), "correct": kk,
                       "acc": p, "lo": lo, "hi": hi})
pd.DataFrame(per_letter).to_csv("tables/per_letter_accuracy.csv", index=False)

conf_mat = np.zeros((4, 4), dtype=int)
for g, p_ in zip(derived["gold"], derived["pred_index"]):
    conf_mat[g, p_] += 1
cm_df = pd.DataFrame(conf_mat, index=[f"gold_{L}" for L in LETTERS],
                     columns=[f"pred_{L}" for L in LETTERS])
cm_df.to_csv("tables/confusion_matrix.csv")
fig, ax = plt.subplots(figsize=(5, 4))
row_norm = conf_mat / conf_mat.sum(axis=1, keepdims=True)
im = ax.imshow(row_norm, cmap="Blues", vmin=0, vmax=1)
for i in range(4):
    for j in range(4):
        ax.text(j, i, f"{conf_mat[i,j]}\n({row_norm[i,j]*100:.1f}%)",
                ha="center", va="center", fontsize=8)
ax.set_xticks(range(4), LETTERS)
ax.set_yticks(range(4), LETTERS)
ax.set_xlabel("predicted letter")
ax.set_ylabel("gold letter")
ax.set_title(f"Confusion matrix, row-normalised (N={N})")
fig.colorbar(im)
savefig(fig, "confusion_matrix.png")

mean_mass = {L: float(derived[f"p{L}"].mean()) for L in LETTERS}
STATS["phase5"] = {
    "gold_marginals": dict(gold_marg), "pred_marginals": dict(pred_marg),
    "chi2_stat": float(chi2.statistic), "chi2_df": 3, "chi2_p": float(chi2.pvalue),
    "per_letter": per_letter, "confusion": conf_mat.tolist(),
    "mean_prob_mass_per_position": mean_mass,
}

# ---------------------------------------------------------------- Phase 6
def num_literals(text):
    return len(re.findall(r"\d+(?:\.\d+)?", text))

feats = pd.DataFrame({
    "id": derived["id"],
    "char_len": [len(r["question"]) for r in ok_rows],
    "word_count": [len(r["question"].split()) for r in ok_rows],
    "input_tokens": derived["input_tokens"],
    "n_numeric_literals": [num_literals(r["question"]) for r in ok_rows],
    "gold_value": derived["gold_value"].astype(float),
    "pred_value": derived["pred_value"].astype(float),
    "choices": [r["choices"] for r in ok_rows],
    "question": [r["question"] for r in ok_rows],
    "correct": labels_bin,
})

def gold_mag_bin(v):
    if v < 0: return "negative"
    if v == 0: return "zero"
    if v < 10: return "(0,10)"
    if v < 100: return "[10,100)"
    if v < 1000: return "[100,1000)"
    return "[1000,+)"

feats["gold_mag"] = feats["gold_value"].map(gold_mag_bin)
def spread_ratio(ch):
    a = [abs(float(x)) for x in ch]
    mn = min(a)
    return max(a) / mn if mn > 0 else float("nan")
def cv(ch):
    v = np.asarray(ch, dtype=float)
    m = v.mean()
    return float(v.std() / abs(m)) if m != 0 else float("nan")
feats["option_spread"] = [spread_ratio(c) for c in feats["choices"]]
feats["option_cv"] = [cv(c) for c in feats["choices"]]
feats["gold_rank"] = [("smallest" if r["choices"][r["gold"]] == min(r["choices"])
                       else "largest" if r["choices"][r["gold"]] == max(r["choices"])
                       else "neither") for r in ok_rows]
feats["gold_is_int"] = [float(r["choices"][r["gold"]]).is_integer() for r in ok_rows]
feats["has_neg_or_zero_option"] = [any(float(x) <= 0 for x in r["choices"]) for r in ok_rows]

def bin_accuracy(df, col, bins, labels_):
    out = []
    cat = pd.cut(df[col], bins=bins, labels=labels_, include_lowest=True)
    for lab in labels_:
        m = (cat == lab).to_numpy()
        n_ = int(m.sum())
        kk = int(df.loc[m, "correct"].sum())
        p, lo, hi = wilson(kk, n_)
        out.append({"feature": col, "bin": str(lab), "n": n_, "correct": kk,
                    "acc": p, "lo": lo, "hi": hi})
    return out

feat_rows = []
q_specs = []
for col in ("char_len", "word_count", "input_tokens", "n_numeric_literals",
            "option_spread", "option_cv"):
    s_ = feats[col].astype(float)
    qs = np.unique(np.nanquantile(s_, [0, .25, .5, .75, 1]))
    labs = [f"Q{i+1} [{qs[i]:.3g},{qs[i+1]:.3g}]" for i in range(len(qs) - 1)]
    qs[0] -= 1e-9; qs[-1] += 1e-9
    feat_rows += bin_accuracy(feats, col, qs, labs)
    q_specs.append((col, labs))
for col, order in [("gold_mag", ["negative", "zero", "(0,10)", "[10,100)", "[100,1000)", "[1000,+)"]),
                   ("gold_rank", ["smallest", "largest", "neither"]),
                   ("gold_is_int", [True, False]),
                   ("has_neg_or_zero_option", [True, False])]:
    for val in order:
        m = (feats[col] == val).to_numpy()
        n_ = int(m.sum())
        kk = int(feats.loc[m, "correct"].sum())
        p, lo, hi = wilson(kk, n_)
        feat_rows.append({"feature": col, "bin": str(val), "n": n_, "correct": kk,
                          "acc": p, "lo": lo, "hi": hi})
feat_df = pd.DataFrame(feat_rows)
feat_df.to_csv("tables/feature_bins.csv", index=False)

fig, axes = plt.subplots(3, 4, figsize=(18, 10))
for ax, (feat_name, grp) in zip(axes.ravel(), feat_df.groupby("feature", sort=False)):
    grp = grp.reset_index(drop=True)
    xs = np.arange(len(grp))
    ax.errorbar(xs, grp["acc"],
                yerr=[np.clip(grp["acc"] - grp["lo"], 0, None),
                      np.clip(grp["hi"] - grp["acc"], 0, None)],
                fmt="o-", capsize=3)
    ax.axhline(overall_acc, color="k", ls="--", lw=0.8)
    ax.set_xticks(xs, [f'{b}\n(n={n})' for b, n in zip(grp["bin"], grp["n"])], fontsize=6)
    ax.set_title(feat_name, fontsize=9)
    ax.set_ylim(0.7, 1.02)
for ax in axes.ravel()[len(feat_df["feature"].unique()):]:
    ax.axis("off")
fig.suptitle(f"Accuracy by item feature (Wilson 95% CI, N={N})")
fig.tight_layout()
savefig(fig, "feature_accuracy.png")

# 8.2 near-miss
err_df = feats[~feats["correct"].astype(bool)].copy()
rel_err = np.abs(err_df["pred_value"] - err_df["gold_value"]) / np.maximum(np.abs(err_df["gold_value"]), 1e-12)
nearest_flags = []
for _, row in err_df.iterrows():
    gold = row["gold_value"]
    vals = [float(x) for x in row["choices"]]
    vals.remove(float(gold))  # distractors only
    pred_d = abs(row["pred_value"] - gold)
    nearest_distractor_d = min(abs(v - gold) for v in vals)
    nearest_flags.append(pred_d <= nearest_distractor_d + 1e-12)
nearest_flags = np.array(nearest_flags)

STATS["phase6_nearmiss"] = {
    "n_errors": int(len(err_df)),
    "rel_err_median": float(rel_err.median()),
    "rel_err_mean": float(rel_err.mean()),
    "rel_err_p25": float(rel_err.quantile(.25)), "rel_err_p75": float(rel_err.quantile(.75)),
    "rel_err_le_0.1_frac": float((rel_err <= 0.1).mean()),
    "rel_err_le_0.5_frac": float((rel_err <= 0.5).mean()),
    "rel_err_gt_2_frac": float((rel_err > 2).mean()),
    "pred_is_nearest_distractor_frac": float(nearest_flags.mean()),
    "pred_is_nearest_distractor_n": int(nearest_flags.sum()),
}

# 8.3 keyword slices
slices = {
    "fraction (/ or 'fraction')": r"\/|\bfraction",
    "percent (% or 'percent')": r"%|\bpercent",
    "ratio/proportion": r"\bratio|\bproportion",
    "speed/distance/time": r"\bspeed\b|\bdistance\b|\bmph\b|\bkm/h\b|\bhours?\b.*\bmiles?\b|\btravel",
    "work-rate": r"\btogether\b|\balone\b|\brate\b",
    "probability/combinatorics": r"\bprobabilit|\bcombinat|\bpermut|\bchance\b|\bdice\b|\bcards?\b",
    "geometry": r"\barea\b|\bperimeter\b|\bradius\b|\btriangle\b|\bcircle\b|\brectangle\b",
    "interest/finance": r"\binterest\b|\binvest|\bprofit\b|\bdiscount\b|\bprice\b|\bcost\b|\bsalary\b|\bdepreciat",
    "explicit negation": r"\bnot\b|\bexcept\b|\bleast\b",
}
slice_rows = []
for nm, pat in slices.items():
    m = feats["question"].str.contains(pat, case=False, regex=True).to_numpy()
    n_ = int(m.sum())
    kk = int(feats.loc[m, "correct"].sum())
    p, lo, hi = wilson(kk, n_)
    slice_rows.append({"slice": nm, "n": n_, "correct": kk, "acc": p, "lo": lo, "hi": hi,
                       "underpowered": n_ < 30})
slice_df = pd.DataFrame(slice_rows).sort_values("acc")
slice_df.to_csv("tables/keyword_slices.csv", index=False)
STATS["phase6_slices"] = slice_rows

# 8.4 qualitative dump
err_rows_full = [r for r in ok_rows if not r["_correct"]]
conf_wrong = [r for r in err_rows_full if r["_conf"] == 1.0]
conf_wrong_sorted = sorted(conf_wrong, key=lambda r: -r["_rawconf"] if not math.isnan(r["_rawconf"]) else 0)[:40]
rng_sample = np.random.RandomState(SEED)
rest = [r for r in err_rows_full if r["id"] not in {x["id"] for x in conf_wrong_sorted}]
idx = rng_sample.choice(len(rest), size=min(40, len(rest)), replace=False)
rand_wrong = [rest[i] for i in sorted(idx)]
low_conf_correct = sorted([r for r in ok_rows if r["_correct"]], key=lambda r: r["_conf"])[:15]

def dump_item(f, r):
    f.write(f"### {r['id']}\n\n")
    f.write(f"- **Question:** {r['question']}\n")
    f.write(f"- **Options:** {r['choices']}\n")
    f.write(f"- **Gold:** {r['choices'][r['gold']]} (index {r['gold']}) | "
            f"**Predicted:** {r['choices'][r['_pred_index']]} (index {r['_pred_index']})\n")
    f.write(f"- **probs:** {json.dumps(r['_probs'])} | **raw.confidence:** {r['_rawconf']} | "
            f"**latency_ms:** {r['latency_ms']}\n\n")

with open("errors_sample.md", "w") as f:
    f.write("# Error sample for qualitative review\n\n")
    f.write(f"Confidently-wrong items (top-1 prob = 1.0): {len(conf_wrong)} total; "
            f"showing up to 40 highest raw.confidence.\n\n")
    f.write("## Group 1: confidently wrong (top-1 = 1.0)\n\n")
    if not conf_wrong_sorted:
        f.write("*None — zero incorrect items had top-1 probability 1.0.*\n\n")
    for r in conf_wrong_sorted:
        dump_item(f, r)
    # supplementary: highest-confidence incorrect, since Group 1 is empty
    hi_conf_wrong = sorted(err_rows_full,
                           key=lambda r: (r["_conf"], -0.0 if math.isnan(r["_rawconf"]) else r["_rawconf"]),
                           reverse=True)[:40]
    f.write("## Group 1b (supplementary): 40 incorrect items with the highest confidence\n\n")
    for r in hi_conf_wrong:
        dump_item(f, r)
    f.write("## Group 2: seeded random sample of further incorrect items (seed=42)\n\n")
    for r in rand_wrong:
        dump_item(f, r)
    f.write("## Group 3: 15 correct items with the lowest confidence\n\n")
    for r in low_conf_correct:
        dump_item(f, r)

STATS["phase6_dump"] = {"n_confidently_wrong": len(conf_wrong),
                        "n_random_wrong_sampled": len(rand_wrong),
                        "n_hi_conf_wrong_shown": len(hi_conf_wrong)}

# ---------------------------------------------------------------- Phase 7
prim = derived[derived["attempts"] == 1]
lat = prim["latency_ms"].to_numpy()
lat_stats = {"mean": float(lat.mean()), "sd": float(lat.std()), "p50": float(np.percentile(lat, 50)),
             "p90": float(np.percentile(lat, 90)), "p95": float(np.percentile(lat, 95)),
             "p99": float(np.percentile(lat, 99)), "min": float(lat.min()), "max": float(lat.max()),
             "n": int(len(lat)), "excluded_attempts_gt1": int(N - len(lat))}

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(lat, bins=80)
ax.set_xlabel("latency (ms)")
ax.set_ylabel("count")
ax.set_title(f"Latency, attempts==1 only (n={len(lat)})")
savefig(fig, "latency_hist.png")

fig, ax = plt.subplots(figsize=(7, 4))
xs = np.sort(lat)
ax.plot(xs, np.arange(1, len(xs) + 1) / len(xs))
ax.set_xlabel("latency (ms)")
ax.set_ylabel("ECDF")
ax.set_title(f"Latency ECDF, attempts==1 (n={len(lat)})")
savefig(fig, "latency_ecdf.png")

tok = prim["input_tokens"].to_numpy(dtype=float)
pear = sstats.pearsonr(tok, lat)
spear = sstats.spearmanr(tok, lat)
fig, ax = plt.subplots(figsize=(7, 4))
ax.scatter(tok, lat, s=4, alpha=0.3)
ax.set_xlabel("input tokens")
ax.set_ylabel("latency (ms)")
ax.set_title(f"Latency vs input tokens (Pearson r={pear.statistic:.3f}, Spearman ρ={spear.statistic:.3f}, n={len(lat)})")
savefig(fig, "latency_vs_tokens.png")

lat_c = prim.loc[prim["correct"], "latency_ms"].to_numpy()
lat_w = prim.loc[~prim["correct"], "latency_ms"].to_numpy()
mw = sstats.mannwhitneyu(lat_c, lat_w, alternative="two-sided")

cost_total = float(derived["cost"].sum())
cost_mean = float(derived["cost"].mean())
cost_per_correct = cost_total / k
cost_per_1000 = cost_mean * 1000
in_tok = derived["input_tokens"].to_numpy(dtype=float)
out_tok = derived["output_tokens"].to_numpy(dtype=float)
implied_price = cost_total / in_tok.sum() * 1e6
expected_cost = in_tok.sum() * 0.042 / 1e6

from datetime import datetime
t_first = datetime.fromisoformat(ts_first.replace("Z", "+00:00"))
t_last = datetime.fromisoformat(ts_last.replace("Z", "+00:00"))
wall_s = (t_last - t_first).total_seconds()
throughput = N / wall_s if wall_s > 0 else float("nan")

pd.DataFrame([{"metric": m, "value": v} for m, v in {
    **{f"latency_{k_}": v for k_, v in lat_stats.items()},
    "pearson_r_latency_tokens": float(pear.statistic), "pearson_p": float(pear.pvalue),
    "spearman_rho_latency_tokens": float(spear.statistic), "spearman_p": float(spear.pvalue),
    "latency_mean_correct": float(lat_c.mean()), "latency_mean_incorrect": float(lat_w.mean()),
    "mannwhitney_U": float(mw.statistic), "mannwhitney_p": float(mw.pvalue),
    "cost_total": cost_total, "cost_mean_per_item": cost_mean,
    "cost_per_correct": cost_per_correct, "cost_per_1000_items": cost_per_1000,
    "expected_cost_at_0.042_per_M": expected_cost, "implied_price_per_M_input": implied_price,
    "wall_clock_s": wall_s, "throughput_items_per_s": throughput,
}.items()]).to_csv("tables/latency_cost.csv", index=False)

STATS["phase7"] = {
    "latency": lat_stats,
    "pearson_latency_tokens": {"r": float(pear.statistic), "p": float(pear.pvalue)},
    "spearman_latency_tokens": {"rho": float(spear.statistic), "p": float(spear.pvalue)},
    "latency_mean_correct": float(lat_c.mean()), "latency_mean_incorrect": float(lat_w.mean()),
    "latency_median_correct": float(np.median(lat_c)), "latency_median_incorrect": float(np.median(lat_w)),
    "mannwhitney": {"U": float(mw.statistic), "p": float(mw.pvalue)},
    "cost": {"total": cost_total, "mean_per_item": cost_mean, "per_correct": cost_per_correct,
             "per_1000": cost_per_1000, "expected_at_0.042_per_M": expected_cost,
             "implied_price_per_M_input": implied_price},
    "tokens": {"input_mean": float(in_tok.mean()), "input_min": float(in_tok.min()),
               "input_max": float(in_tok.max()),
               "output_distinct": sorted(set(out_tok.tolist()))[:20],
               "output_n_distinct": len(set(out_tok.tolist())),
               "output_mean": float(out_tok.mean())},
    "wall_clock_s": wall_s, "throughput_items_per_s": throughput,
}

# ---------------------------------------------------------------- Phase 8
boot2 = np.random.RandomState(BOOT_SEED)
baccs = []
for _ in range(N_BOOT):
    ii = boot2.randint(0, N, N)
    baccs.append(labels_bin[ii].mean())
blo, bhi = np.percentile(baccs, [2.5, 97.5])

order_ts = np.argsort(derived["ts"].to_numpy(), kind="mergesort")
half = N // 2
h1, h2 = labels_bin[order_ts[:half]], labels_bin[order_ts[half:]]
p1_, p2_ = h1.mean(), h2.mean()
pp = (h1.sum() + h2.sum()) / N
se = math.sqrt(pp * (1 - pp) * (1 / len(h1) + 1 / len(h2)))
z = (p1_ - p2_) / se if se > 0 else 0.0
p_two_prop = 2 * (1 - sstats.norm.cdf(abs(z)))

worst_slice = slice_df.iloc[0]
m_worst = feats["question"].str.contains(slices[worst_slice["slice"]], case=False, regex=True).to_numpy()
acc_wo_worst = float(labels_bin[~m_worst].mean())

max_achievable = (N - len(dup_option_rows)) / N

STATS["phase8"] = {
    "bootstrap_acc_lo": float(blo), "bootstrap_acc_hi": float(bhi), "n_boot": N_BOOT,
    "split_half": {"first_acc": float(p1_), "second_acc": float(p2_), "n1": int(len(h1)),
                   "n2": int(len(h2)), "z": float(z), "p_two_proportion": float(p_two_prop)},
    "worst_slice": worst_slice["slice"], "worst_slice_acc": float(worst_slice["acc"]),
    "acc_without_worst_slice": acc_wo_worst,
    "headline_shift_pp": (acc_wo_worst - overall_acc) * 100,
    "max_achievable_accuracy": max_achievable,
}

# ---------------------------------------------------------------- Phase 9 (contamination)
norm_q = [" ".join(r["question"].lower().split()) for r in ok_rows]
norm_counts = Counter(norm_q)
dup_text_items = sum(1 for q in norm_q if norm_counts[q] > 1)
exact_dup_groups = sum(1 for c in norm_counts.values() if c > 1)

def shingles(text, w=5):
    toks = text.split()
    return set(hash(" ".join(toks[i:i + w])) for i in range(max(1, len(toks) - w + 1)))

shingle_sets = [shingles(q) for q in norm_q]
inv = defaultdict(list)
for i, ss in enumerate(shingle_sets):
    for h in ss:
        inv[h].append(i)
cand = Counter()
for h, docs in inv.items():
    if len(docs) > 50:
        continue
    for a_i in range(len(docs)):
        for b_i in range(a_i + 1, len(docs)):
            cand[(docs[a_i], docs[b_i])] += 1
near_dup_items = set()
for (a_i, b_i), shared in cand.items():
    if shared < 3:
        continue
    union = len(shingle_sets[a_i] | shingle_sets[b_i])
    if union and shared / union >= 0.6:
        near_dup_items.add(a_i)
        near_dup_items.add(b_i)
dupmask = np.array([(norm_counts[q] > 1) or (i in near_dup_items)
                    for i, q in enumerate(norm_q)])
STATS["phase9_contamination"] = {
    "exact_duplicate_text_items": dup_text_items,
    "exact_duplicate_groups": exact_dup_groups,
    "near_duplicate_items_jaccard_ge_0.6": len(near_dup_items),
    "items_involved_in_any_duplicate": int(dupmask.sum()),
    "accuracy_on_duplicated_items": float(labels_bin[dupmask].mean()) if dupmask.any() else None,
    "accuracy_on_unique_items": float(labels_bin[~dupmask].mean()) if (~dupmask).any() else None,
    "n_duplicated": int(dupmask.sum()), "n_unique": int((~dupmask).sum()),
}

# ---------------------------------------------------------------- wrap-up
import platform
STATS["repro"] = {
    "python": platform.python_version(),
    "numpy": np.__version__, "pandas": pd.__version__,
    "matplotlib": matplotlib.__version__, "scipy": sstats.__version__ if hasattr(sstats, "__version__") else __import__("scipy").__version__,
    "seed": SEED, "boot_seed": BOOT_SEED, "n_boot": N_BOOT, "nll_eps": NLL_EPS,
    "runtime_s": round(time.monotonic() - T0, 1),
    "command": "./venv/bin/python analyze.py",
}

with open("stats.json", "w") as f:
    json.dump(STATS, f, indent=2, default=str)

print(f"analysis complete in {STATS['repro']['runtime_s']}s — N={N}, acc={acc:.4f}")
print("outputs: stats.json, derived.csv, figures/, tables/, errors_sample.md")
