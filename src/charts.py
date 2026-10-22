"""Visual outputs for the pipeline (the 'visual matrix' deliverable).

Produces three figures in ``outputs/``:

1. ``price_sentiment.png`` - asset price with daily X sentiment on a
   twin axis: how sentiment changed against price over time.
2. ``visual_matrix.png``   - heatmap matrix of out-of-sample RMSE
   (in return space) with **walk-forward folds as rows and forecast
   horizons n+0..n+5 as columns** - the visual matrix of model accuracy.
3. ``equity_drawdown.png`` - strategy vs buy-and-hold equity curves with
   the strategy drawdown underneath.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless - no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


def chart_price_sentiment(features: pd.DataFrame, out: Path | None = None) -> Path:
    """Dual-axis chart: BTC price (left) and daily sentiment (right)."""
    out = out or (config.OUTPUTS_DIR / "price_sentiment.png")
    df = features.copy()
    df["date"] = pd.to_datetime(df["date"])

    fig, ax_price = plt.subplots(figsize=(12, 5))
    ax_price.plot(df["date"], df["close"], color="#1f77b4", lw=1.4, label="BTC close")
    ax_price.set_ylabel("Price (USD)")
    ax_price.set_xlabel("Date")

    ax_sent = ax_price.twinx()
    ax_sent.bar(
        df["date"], df["sent_mean"], width=1.0,
        color=["#2ca02c" if v >= 0 else "#d62728" for v in df["sent_mean"]],
        alpha=0.45, label="Daily sentiment (VADER mean)",
    )
    ax_sent.plot(
        df["date"], df["sent_mom_7d"], color="#ff7f0e", lw=1.2,
        label="Sentiment 7d momentum",
    )
    ax_sent.axhline(0, color="grey", lw=0.8)
    ax_sent.set_ylabel("Sentiment")

    ax_price.set_title("Bitcoin price vs X (Twitter) sentiment")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def chart_visual_matrix(preds: pd.DataFrame, out: Path | None = None) -> Path:
    """Heatmap of OOS RMSE (returns) by fold (rows) x horizon (columns)."""
    out = out or (config.OUTPUTS_DIR / "visual_matrix.png")
    folds = sorted(preds["fold"].unique())
    matrix = np.zeros((len(folds), len(config.HORIZONS)))
    for i, fold in enumerate(folds):
        block = preds[preds["fold"] == fold]
        for j, h in enumerate(config.HORIZONS):
            err = block[f"pred_ret_n{h}"] - block[f"actual_ret_n{h}"]
            matrix[i, j] = float(np.sqrt(np.mean(err ** 2)))

    fig, ax = plt.subplots(figsize=(8, max(3.0, 0.55 * len(folds))))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(config.HORIZONS)),
                  [f"n+{h}" for h in config.HORIZONS])
    ax.set_yticks(range(len(folds)), [f"fold {f}" for f in folds])
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("Walk-forward fold")
    ax.set_title("Visual matrix: OOS return RMSE by fold x horizon")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center",
                    color="white", fontsize=8)
    fig.colorbar(im, ax=ax, label="RMSE (log return)")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def chart_equity_drawdown(preds: pd.DataFrame, out: Path | None = None) -> Path:
    """Strategy vs buy-and-hold equity curves + strategy drawdown."""
    out = out or (config.OUTPUTS_DIR / "equity_drawdown.png")
    df = preds.copy()
    df["date"] = pd.to_datetime(df["date"])
    strat_equity = (1.0 + df["strategy_ret"]).cumprod()
    bh_equity = (1.0 + df["buyhold_ret"]).cumprod()
    drawdown = strat_equity / strat_equity.cummax() - 1.0

    fig, (ax_eq, ax_dd) = plt.subplots(
        2, 1, figsize=(12, 6), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    ax_eq.plot(df["date"], strat_equity, label="Sentiment strategy (long/cash)")
    ax_eq.plot(df["date"], bh_equity, label="Buy and hold", alpha=0.8)
    ax_eq.set_ylabel("Equity (start = 1)")
    ax_eq.legend()
    ax_eq.set_title("Out-of-sample equity curve and drawdown")

    ax_dd.fill_between(df["date"], drawdown, 0, color="#d62728", alpha=0.5)
    ax_dd.set_ylabel("Drawdown")
    ax_dd.set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def make_all_charts(features: pd.DataFrame, preds: pd.DataFrame) -> list[Path]:
    return [
        chart_price_sentiment(features),
        chart_visual_matrix(preds),
        chart_equity_drawdown(preds),
    ]
