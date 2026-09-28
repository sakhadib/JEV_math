#!/usr/bin/env python3
"""Build every figure for the paper as vector PDF.

Usage:  python3 make_figures.py [path/to/jev_math_repo]

Figures carry no embedded titles: the LaTeX captions do that work.
Palette is shared with the LaTeX table highlighting (see acl_latex.tex).
"""
import sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

REPO = sys.argv[1] if len(sys.argv) > 1 else "../sakhadib/jev_math"

# ----------------------------------------------------------------- palette
CYAN     = "#00B8D4"
ELECTRIC = "#22F0FF"
BABYPINK = "#FFAEDC"
MAGENTA  = "#FF2BAF"
YELLOW   = "#FFD84D"
INK      = "#1A1A2E"
GREY     = "#9AA0B4"

JEV = LinearSegmentedColormap.from_list(
    "jev", [CYAN, ELECTRIC, BABYPINK, MAGENTA, YELLOW])

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.linewidth": 0.6,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "figure.facecolor": "white",
})


# ------------------------------------------------------------------- data
def load_jev():
    """Per-item (top1, p2, correct) for each phase, plus option count."""
    out = {}

    d1 = pd.read_csv(f"{REPO}/derived.csv")
    P = d1[["pA", "pB", "pC", "pD"]].to_numpy()
    S = np.sort(P, axis=1)[:, ::-1]
    out["MathQA-style"] = dict(p1=S[:, 0], p2=S[:, 1],
                               correct=d1["correct"].astype(bool).to_numpy(), k=4)

    for tag, path, k in (("CompMath-MCQ", "phase_2/results_jev.jsonl", 3),
                         ("HARP", "phase_3/results_jev.jsonl", 5)):
        p1, p2, co = [], [], []
        for line in open(f"{REPO}/{path}"):
            r = json.loads(line)
            if r.get("status") != "ok":
                continue
            pr = r["probs"]
            v = sorted(pr.values(), reverse=True)
            p1.append(v[0]); p2.append(v[1] if len(v) > 1 else 0.0)
            # Recompute argmax with A-first tie-breaking, as the analysis
            # scripts do: exact ties are common once probabilities are
            # quantised to whole percents (71 rows on HARP).
            top = sorted(k for k, x in pr.items() if x == v[0])[0]
            gl = r.get("gold_letter") or "ABCDE"[r["gold"]]
            co.append(top == gl)
        out[tag] = dict(p1=np.array(p1), p2=np.array(p2),
                        correct=np.array(co), k=k)
    return out


DATA = load_jev()
ORDER = ["MathQA-style", "CompMath-MCQ", "HARP"]
for t in ORDER:
    d = DATA[t]
    deg = d["p1"] == 1.0
    print(f"{t:14} n={len(d['p1']):5d}  acc={d['correct'].mean():.4f}  "
          f"p=1.0 share={deg.mean():.4f}  acc@p1={d['correct'][deg].mean():.4f}")


# ================================================== 2. DISTRIBUTION COLLAPSE
def simplex3():
    """Stepped 3D histograms of (p1, p2) for the three benchmarks."""
    fig = plt.figure(figsize=(7.0, 1.85))
    for i, tag in enumerate(ORDER, 1):
        d = DATA[tag]
        p1, p2, k = d["p1"], d["p2"], d["k"]
        step = 0.05
        lo1 = np.floor((1.0 / k) / step) * step

        def edges(a, b):
            nb = int(round((b - a) / step))
            e = np.linspace(a, b, nb + 1); e[-1] += 1e-9
            return e

        xe, ye = edges(lo1, 1.0), edges(0.0, 0.5)
        H, xe, ye = np.histogram2d(p1, p2, bins=[xe, ye])
        assert int(H.sum()) == len(p1), f"{tag}: dropped items"
        Z = np.log10(1.0 + H)
        xc, yc = 0.5 * (xe[:-1] + xe[1:]), 0.5 * (ye[:-1] + ye[1:])
        imp = (yc[None, :] > xc[:, None] + step) | \
              (xc[:, None] + yc[None, :] > 1.0 + step)
        Z = np.where(imp, np.nan, Z)
        xs, ys = np.repeat(xe, 2)[1:-1], np.repeat(ye, 2)[1:-1]
        Zs = np.repeat(np.repeat(Z, 2, axis=0), 2, axis=1)
        X, Y = np.meshgrid(xs, ys, indexing="ij")

        norm = matplotlib.colors.PowerNorm(gamma=0.55, vmin=0, vmax=3.6)
        FLOOR = -1.0
        ax = fig.add_subplot(1, 3, i, projection="3d")
        ax.plot_surface(X, Y, Zs, cmap=JEV, norm=norm, rstride=1, cstride=1,
                        linewidth=0.15, edgecolors=(1, 1, 1, 0.3),
                        shade=True, alpha=1.0)
        ax.contourf(X, Y, np.ma.masked_invalid(Zs), zdir="z", offset=FLOOR,
                    levels=12, cmap=JEV, norm=norm, alpha=0.5)
        w = ~d["correct"]
        ax.scatter(p1[w], p2[w], np.full(w.sum(), FLOOR),
                   s=1.4, c=INK, alpha=0.35, depthshade=False, linewidths=0)
        ax.set_xlim(lo1, 1.0); ax.set_ylim(0.0, 0.5); ax.set_zlim(FLOOR, 3.7)
        ax.set_xlabel("$p_{(1)}$", labelpad=-9, fontsize=7)
        ax.set_ylabel("$p_{(2)}$", labelpad=-9, fontsize=7)
        ax.set_zticks([0, 1, 2, 3]); ax.set_yticks([0.1, 0.3])
        ax.set_xticks([0.4, 0.7, 1.0] if k == 4 else
                      ([0.4, 0.7, 1.0] if k == 3 else [0.2, 0.6, 1.0]))
        if i == 1:
            ax.set_zlabel("$\\log_{10}(1+n)$", labelpad=-9, fontsize=7)
        ax.set_box_aspect((1.35, 0.85, 0.72))
        ax.view_init(elev=16, azim=-42)
        ax.tick_params(pad=-3, labelsize=5.8)
        ax.grid(False)
        for a in (ax.xaxis, ax.yaxis, ax.zaxis):
            a.pane.fill = False
            a.pane.set_edgecolor((1, 1, 1, 0))
            a.line.set_color((0.55, 0.57, 0.65, 0.9))
            a.line.set_linewidth(0.5)
    fig.subplots_adjust(wspace=0.0, left=0.02, right=0.98, top=1.06, bottom=0.02)
    fig.savefig("figures/simplex3.pdf")
    plt.close(fig)


# ===================================================== 3. DIFFICULTY GRADIENT
def difficulty():
    s = json.load(open(f"{REPO}/phase_3/stats.json"))
    spec = [("space-bunny-alpha", BABYPINK, "o"),
            ("gpt-oss-20b", MAGENTA, "s"),
            ("jev-1.13", CYAN, "*"),
            ("llama-3.1-8b-instruct", YELLOW, "D")]
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    for name, col, mk in spec:
        bl = s["systems"][name]["by_level"]
        lv = [1, 2, 3, 4]
        acc = [bl[str(l)]["acc"] * 100 for l in lv]
        lo = [(bl[str(l)]["acc"] - bl[str(l)]["lo"]) * 100 for l in lv]
        hi = [(bl[str(l)]["hi"] - bl[str(l)]["acc"]) * 100 for l in lv]
        ax.errorbar(lv, acc, yerr=[lo, hi], fmt="-", lw=1.3, color=col,
                    capsize=2, elinewidth=0.8, ecolor=col)
        ax.scatter(lv, acc, s=46 if mk == "*" else 22, marker=mk, color=col,
                   zorder=4, edgecolor="white", linewidth=0.6,
                   label=name.replace("-instruct", ""))
    ax.axhline(20, ls=":", lw=0.9, color=GREY)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xlabel("HARP difficulty level")
    ax.set_ylabel("accuracy (%)")
    ax.set_ylim(12, 92)
    ax.grid(alpha=0.16, lw=0.4, color=GREY)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.savefig("figures/difficulty.pdf")
    plt.close(fig)


# ================================================ 4. DEFERRAL ACROSS PHASES
def coverage3():
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    cols = {"MathQA-style": MAGENTA, "CompMath-MCQ": BABYPINK, "HARP": CYAN}
    mks = {"MathQA-style": "o", "CompMath-MCQ": "s", "HARP": "D"}
    for tag in ORDER:
        d = DATA[tag]
        conf, corr = d["p1"], d["correct"]
        o = np.argsort(-conf, kind="stable")
        c = corr[o]
        cov = np.arange(1, len(c) + 1) / len(c)
        run = np.cumsum(c) / np.arange(1, len(c) + 1)
        keep = cov >= 0.30
        ax.plot(cov[keep] * 100, run[keep] * 100, lw=1.4, color=cols[tag])
        for q in (0.5, 0.75, 0.9, 1.0):
            j = int(q * len(c)) - 1
            ax.scatter(cov[j] * 100, run[j] * 100, s=16, marker=mks[tag],
                       color=cols[tag], zorder=4,
                       edgecolor="white", linewidth=0.5,
                       label=tag if q == 1.0 else None)
    ax.set_xlabel("coverage (% of items answered)")
    ax.set_ylabel("accuracy on retained (%)")
    ax.set_xlim(28, 103)
    ax.set_ylim(50, 102)
    ax.grid(alpha=0.16, lw=0.4, color=GREY)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.savefig("figures/coverage3.pdf")
    plt.close(fig)


# ============================================ 5. POSITION BIAS VS CONFIDENCE
def position():
    from collections import Counter
    rows = [json.loads(l) for l in open(f"{REPO}/phase_3/results_jev.jsonl")]
    ok = [r for r in rows if r.get("status") == "ok"]
    L = ["A", "B", "C", "D", "E"]
    bands = [(0.0, 0.4, "$p_{(1)}<0.4$"), (0.4, 0.7, "$0.4\\!-\\!0.7$"),
             (0.7, 1.01, "$p_{(1)}\\geq0.7$")]
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 1.95), sharey=True)
    for ax, (lo, hi, lab) in zip(axes, bands):
        sub = [r for r in ok if lo <= r["confidence"] < hi]
        n = len(sub)
        pm = Counter(r["predicted_letter"] for r in sub)
        gm = Counter(r["gold_letter"] for r in sub)
        x = np.arange(5)
        ax.bar(x - 0.2, [100 * gm[c] / n for c in L], width=0.4,
               color=GREY, alpha=0.55, label="gold key",
               edgecolor="white", linewidth=0.5)
        ax.bar(x + 0.2, [100 * pm[c] / n for c in L], width=0.4,
               color=MAGENTA, alpha=0.85, label="Jev predictions",
               edgecolor="white", linewidth=0.5)
        ax.set_xticks(x); ax.set_xticklabels(L, fontsize=7)
        ax.set_xlabel(f"{lab}  ($n={n:,}$)".replace(",", "{,}"), fontsize=7)
        ax.grid(alpha=0.16, lw=0.4, color=GREY, axis="y")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("share of items (%)")
    fig.savefig("figures/position.pdf")
    plt.close(fig)


# ===================================================== 6. ERROR CONFIDENCE
def confidence():
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 1.95), sharey=True)
    for ax, tag in zip(axes, ORDER):
        d = DATA[tag]
        bins = np.linspace(0, 1, 31)
        ax.hist(d["p1"][d["correct"]], bins=bins, color=CYAN, alpha=0.85,
                log=True, label="correct", edgecolor="white", linewidth=0.2)
        ax.hist(d["p1"][~d["correct"]], bins=bins, color=MAGENTA, alpha=0.85,
                log=True, label="incorrect", edgecolor="white", linewidth=0.2)
        ax.set_xlabel("$p_{(1)}$", fontsize=7)
        ax.set_xlim(0, 1.02)
        ax.grid(alpha=0.16, lw=0.4, color=GREY, axis="y")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("count (log)")
    fig.savefig("figures/confidence.pdf")
    plt.close(fig)


for fn in (simplex3, difficulty, coverage3, position, confidence):
    fn()
    print("ok:", fn.__name__)


# ================================================== 7. RELIABILITY, 3 PANELS
def reliability():
    """Equal-count reliability with a shared axis, one panel per benchmark."""
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.25), sharey=True)
    for ax, tag in zip(axes, ORDER):
        d = DATA[tag]
        c, y = d["p1"], d["correct"].astype(float)
        qs = np.unique(np.quantile(c, np.linspace(0, 1, 11)))
        idx = np.clip(np.digitize(c, qs[1:-1], right=True), 0, len(qs) - 2)
        mc, ac, ns = [], [], []
        for b in range(len(qs) - 1):
            m = idx == b
            if m.sum() == 0:
                continue
            mc.append(c[m].mean()); ac.append(y[m].mean()); ns.append(int(m.sum()))
        mc, ac = np.array(mc), np.array(ac)
        ece = np.sum(np.array(ns) / len(c) * np.abs(ac - mc))
        lo = 0.18
        ax.plot([lo, 1.02], [lo, 1.02], ls="--", lw=0.9, color=GREY,
                label="perfect", zorder=1)
        ax.plot(mc, ac, "-", lw=1.3, color=MAGENTA, zorder=3)
        ax.scatter(mc, ac, s=np.clip(np.array(ns) / 26, 8, 80), color=MAGENTA,
                   edgecolor="white", linewidth=0.5, zorder=4,
                   label="observed")
        ax.set_xlim(lo, 1.02); ax.set_ylim(lo, 1.02)
        ax.set_xlabel(f"mean $p_{{(1)}}$\n(ECE {ece:.3f})", fontsize=7)
        ax.grid(alpha=0.16, lw=0.4, color=GREY)
        ax.set_aspect("equal", adjustable="box")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("observed accuracy")
    fig.savefig("figures/reliability.pdf")
    plt.close(fig)


reliability()
print("ok: reliability")
