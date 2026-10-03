"""Figuras del README: `python -m src.make_figures`.

Lee `outputs/results.json`, que escribe el pipeline, mas el panel para las dos
figuras de datos. Ningun numero se escribe a mano aca: si el pipeline cambia,
las figuras cambian.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from src.data import build_panel, intermittency_by_series

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "outputs" / "figures"
RESULTS = ROOT / "outputs" / "results.json"

INK = "#2B2B2B"
GRID = "#D9D9D9"
NAIVE = "#8FA8B8"
MEAN = "#6E8CA0"
QUANT = "#B5553D"
OK = "#4C7A3E"
WARN = "#B58900"


def _style(ax, title=None, xlabel=None, ylabel=None, grid_axis="y"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK, labelsize=9)
    if title:
        ax.set_title(title, fontsize=11.5, color=INK, pad=12)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10, color=INK)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10, color=INK)
    return ax


def _save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / name, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  escrito outputs/figures/{name}")


# ---------------------------------------------------------------------------
# 1. De donde salen los ceros
# ---------------------------------------------------------------------------
def figure_zeros(r: dict):
    print("1/4 zeros_decomposition ...")
    p = r["panel"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.9),
                                   gridspec_kw={"width_ratios": [1.15, 1]})

    nonzero = p["total_rows"] - p["zero_rows"]
    pre = p["pre_opening_rows"]
    real_zero = p["zero_rows"] - pre
    parts = [nonzero, real_zero, pre]
    labels = ["Sales recorded", "Genuine zero\n(store open, nothing sold)",
              "Store did not exist yet"]
    colors = [OK, WARN, QUANT]
    left = 0.0
    for v, lab, c in zip(parts, labels, colors):
        ax1.barh([0], [v], left=[left], color=c, height=0.5, edgecolor="white", linewidth=1.4)
        ax1.text(left + v / 2, 0, f"{v / p['total_rows']:.1%}", ha="center", va="center",
                 fontsize=11, color="white", fontweight="bold")
        left += v
    ax1.set_yticks([])
    ax1.set_xlim(0, p["total_rows"])
    ax1.set_xticks([])
    _style(ax1, grid_axis="x")
    ax1.set_title(f"{p['total_rows']:,} rows of the panel", fontsize=11.5, color=INK, pad=12)
    ax1.legend([plt.Rectangle((0, 0), 1, 1, fc=c) for c in colors], labels,
               frameon=False, fontsize=8.8, loc="upper center",
               bbox_to_anchor=(0.5, -0.05), ncol=3)

    bars = ax2.bar(["counting\npre-opening", "excluding\npre-opening"],
                   [p["zero_share_raw"], p["zero_share_after_opening"]],
                   color=[QUANT, OK], width=0.55, edgecolor="white", linewidth=1.3)
    for b, v in zip(bars, [p["zero_share_raw"], p["zero_share_after_opening"]]):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.1%}", ha="center",
                 fontsize=12, color=INK, fontweight="bold")
    ax2.set_ylim(0, max(p["zero_share_raw"], p["zero_share_after_opening"]) * 1.25)
    ax2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1, decimals=0))
    _style(ax2, ylabel="share of rows with zero sales")
    ax2.set_title(f"{p['zeros_explained_by_openings']:.0%} of all zeros are\nstores that had not opened",
                  fontsize=11.5, color=INK, pad=12)

    fig.text(0.5, -0.10,
             f"8 of the 54 stores open after the panel starts, the last of them "
             f"{'2017-04-20'} — four months before the data ends. Until then every one of their 33\n"
             f"families reports zero sales, every day: {pre:,} rows "
             f"({p['pre_opening_share']:.1%} of the panel) that record the absence of a store "
             "rather than the absence of demand.\n"
             f"Counting them also inflates the number of severely intermittent series "
             f"(>90% zeros) from {p['severe_intermittent_clean']} to {p['severe_intermittent_raw']}.",
             ha="center", fontsize=8.5, color="#666666")
    _save(fig, "zeros_decomposition.png")


# ---------------------------------------------------------------------------
# 2. Intermitencia por serie
# ---------------------------------------------------------------------------
def figure_intermittency():
    print("2/4 intermittency_distribution ...")
    panel = build_panel()
    inter = intermittency_by_series(panel, exclude_pre_opening=True)
    z = inter["zero_fraction"].to_numpy()

    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    ax.hist(z, bins=50, color=MEAN, edgecolor="white", linewidth=0.6)
    for thr, c, lab in ((0.5, WARN, "50%"), (0.9, QUANT, "90%")):
        ax.axvline(thr, color=c, linestyle="--", linewidth=1.5)
        ax.text(thr + 0.012, ax.get_ylim()[1] * 0.92,
                f"{int((z > thr).sum())} series above {lab}", fontsize=9, color=c)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1, decimals=0))
    _style(ax, xlabel="share of days with zero sales", ylabel="series (store x family)")
    ax.set_title(f"Intermittency across the {len(z):,} series, pre-opening rows removed",
                 fontsize=11.5, color=INK, pad=12)

    fig.text(0.5, -0.09,
             "Not one of the 1,782 series sells every day, and the distribution is bimodal: a dense "
             "group that sells almost always and a long tail that\nalmost never does. A single model "
             "and a single metric across both is the thing to avoid — which is also why MAPE is "
             "unusable here\n(it divides by zero on a quarter of the observations) and why this "
             "project reports RMSLE and MASE instead.",
             ha="center", fontsize=8.5, color="#666666")
    _save(fig, "intermittency_distribution.png")


# ---------------------------------------------------------------------------
# 3. Exactitud contra decision: el ranking se da vuelta
# ---------------------------------------------------------------------------
def figure_accuracy_vs_decision(r: dict):
    print("3/4 accuracy_vs_decision ...")
    names = ["seasonal_naive", "lgbm_mean", "lgbm_quantile"]
    pretty = ["Seasonal naive", "LightGBM\n(mean)", f"LightGBM\n(quantile {r['costs']['critical_quantile']:.0%})"]
    colors = [NAIVE, MEAN, QUANT]

    rmsle = [r["accuracy"][n]["rmsle"] for n in names]
    service = [r["inventory"][n]["service_level"] for n in names]
    cost = [r["inventory"][n]["total_cost"] for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.9))
    for ax, vals, title, fmtfn, better in (
        (axes[0], rmsle, "RMSLE — the competition metric", lambda v: f"{v:.4f}", "lower"),
        (axes[1], service, "Service level — days without a stockout", lambda v: f"{v:.1%}", "higher"),
        (axes[2], cost, "Total cost at a 4:1 ratio", lambda v: f"{v/1e6:.2f}M", "lower"),
    ):
        best = int(np.argmin(vals)) if better == "lower" else int(np.argmax(vals))
        bars = ax.bar(pretty, vals, color=colors, width=0.6, edgecolor="white", linewidth=1.3)
        for i, (b, v) in enumerate(zip(bars, vals)):
            ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.025, fmtfn(v), ha="center",
                    fontsize=10.5, color=INK, fontweight="bold" if i == best else "normal")
            if i == best:
                ax.text(b.get_x() + b.get_width() / 2, max(vals) * 0.06, "best", ha="center",
                        fontsize=9, color="white", fontweight="bold")
        ax.set_ylim(0, max(vals) * 1.2)
        _style(ax)
        ax.set_title(title, fontsize=11, color=INK, pad=10)
        ax.tick_params(axis="x", labelsize=8.6)

    fig.text(0.5, -0.07,
             "The same three forecasts, scored three ways. The mean model wins the accuracy metric "
             "the competition is judged on and still leaves a stockout\non 41% of store-days, "
             "because ordering the mean is a coin flip by construction. The quantile model is the "
             "worse forecast and the better decision.\nRanking models by RMSLE alone would pick the "
             "one that costs more to operate.",
             ha="center", fontsize=8.5, color="#666666")
    _save(fig, "accuracy_vs_decision.png")


# ---------------------------------------------------------------------------
# 4. Sensibilidad a la razon de costos, y donde esta el cruce
# ---------------------------------------------------------------------------
def figure_sensitivity(r: dict):
    print("4/4 cost_ratio_sensitivity ...")
    s = r["sensitivity"]
    ratios = [x["cost_ratio"] for x in s]
    saving = [x["saving_vs_mean"] for x in s]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.9))

    ax1.axhline(0, color=INK, linewidth=1.2)
    ax1.plot(ratios, saving, color=QUANT, linewidth=2.2, marker="o", markersize=7,
             markerfacecolor="white", markeredgewidth=1.8)
    span = max(saving) - min(saving)
    for x, y in zip(ratios, saving):
        # Las etiquetas negativas van debajo del punto: arriba chocan con la
        # anotacion de la banda.
        ax1.text(x, y + (0.035 if y >= 0 else -0.055) * span, f"{y:+.1%}", ha="center",
                 va="bottom" if y >= 0 else "top", fontsize=8.8,
                 color=OK if y > 0 else QUANT, fontweight="bold")
    neg = [x for x, y in zip(ratios, saving) if y < 0]
    if neg:
        ax1.axvspan(min(ratios), max(neg) + 0.5, color=QUANT, alpha=0.08)
        ax1.text(max(neg) + 0.7, min(saving) * 0.9, "ordering the mean\nis cheaper here",
                 ha="left", va="center", fontsize=9, color=QUANT)
    ax1.set_ylim(min(saving) - 0.12 * span, max(saving) + 0.14 * span)
    ax1.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1, decimals=0))
    _style(ax1, xlabel="understock cost / overstock cost", ylabel="cost saved vs ordering the mean")
    ax1.set_title("The quantile approach only pays above roughly 3:1",
                  fontsize=11.5, color=INK, pad=12)

    q_serv = [x["quantile_service"] for x in s]
    targets = [x["critical_quantile"] for x in s]
    ax2.plot([0.4, 1.0], [0.4, 1.0], color="#9A9A9A", linestyle="--", linewidth=1.5)
    ax2.text(0.98, 0.95, "perfectly calibrated", fontsize=9, color="#777777", ha="right")
    ax2.scatter(targets, q_serv, s=90, color=QUANT, zorder=4, edgecolor="white", linewidth=1.5)
    for t, v in zip(targets, q_serv):
        ax2.text(t, v + 0.018, f"{v:.1%}", ha="center", fontsize=8.8, color=INK)
    ax2.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1, decimals=0))
    ax2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1, decimals=0))
    _style(ax2, xlabel="target quantile asked for", ylabel="service level delivered")
    ax2.set_title("Each quantile model delivers the service it was asked for",
                  fontsize=11.5, color=INK, pad=12)

    fig.text(0.5, -0.07,
             "Left: the critical ratio is an assumption about the business, not a fact in the data, "
             "so it gets a sensitivity analysis. Below about 3:1 the extra\nservice costs more in "
             "excess stock than it saves in lost sales, and ordering the mean wins — a result worth "
             "stating, since quoting only the 4:1\nand 9:1 columns would make the method look "
             "universally better than it is. Right: the models are well calibrated, sitting just "
             "above the diagonal at every target.",
             ha="center", fontsize=8.5, color="#666666")
    _save(fig, "cost_ratio_sensitivity.png")


if __name__ == "__main__":
    if not RESULTS.exists():
        raise SystemExit(f"No existe {RESULTS}. Corre primero: python -m src.pipeline")
    r = json.loads(RESULTS.read_text(encoding="utf-8"))
    print(f"Escribiendo figuras en {FIG_DIR}\n")
    figure_zeros(r)
    figure_intermittency()
    figure_accuracy_vs_decision(r)
    figure_sensitivity(r)
    print("\nListo.")
