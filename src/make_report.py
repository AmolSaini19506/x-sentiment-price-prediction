"""Generate the Markdown deliverables from pipeline outputs.

* ``reports/evaluation_report.md`` - the evaluation report: methodology,
  data, features, walk-forward design, per-horizon accuracy, and the
  strategy backtest metrics (Sharpe, Sortino, Max Drawdown) against a
  buy-and-hold benchmark.
* ``reports/sentiment_outlook.md`` - data-driven bullet facts for the
  1-page sentiment outlook. The numbers and headlines here are factual
  extracts from the ingested data; the final prose summary is written
  from these bullets.

Run after ``src.run_pipeline``: ``python -m src.make_report``
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


def _pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def _load_metrics() -> dict:
    path = config.OUTPUTS_DIR / "metrics.json"
    if not path.exists():
        raise FileNotFoundError("outputs/metrics.json missing - run the pipeline first")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def build_evaluation_report(metrics: dict) -> str:
    lines: list[str] = []
    lines.append("# Evaluation Report - X Sentiment Price Forecasting (Bitcoin)")
    lines.append("")
    lines.append("## 1. Objective")
    lines.append(
        "Predict Bitcoin's (BTC-USD) price at horizons n+0, n+1, n+2, n+3, "
        "n+4 and n+5 days using **only** X (Twitter) sentiment features and "
        "price history, and evaluate the forecasts with rigorous "
        "out-of-sample walk-forward validation."
    )
    lines.append("")
    lines.append("## 2. Data")
    lines.append(
        f"- **Prices:** daily OHLCV for {config.TICKER} via yfinance, "
        f"last {config.LOOKBACK_DAYS} days."
    )
    lines.append(
        "- **X posts:** scraped with Apify (search terms "
        f"{', '.join(config.TWITTER_SEARCH_TERMS)}), timestamps aligned to "
        "UTC days; engagement (likes, reposts, views) and author follower "
        "counts retained."
    )
    lines.append(
        "- **News:** recent Bitcoin coverage via the Tavily API, used as "
        "qualitative context for sentiment moves."
    )
    lines.append(
        "- Prices and posts are joined in a SQLite database "
        "(`data/agent.db`, tables `prices` and `tweets`)."
    )
    lines.append("")
    lines.append("## 3. Features (sentiment + price only)")
    lines.append(
        "- **Sentiment (VADER):** daily mean compound, engagement-weighted "
        "mean, follower-weighted mean, post volume, bullish share "
        "(compound > 0.05), 3d/7d sentiment momentum, 1-day sentiment "
        "change, 7-day volume change."
    )
    lines.append(
        "- **Price:** lagged log returns (1/2/3/5/7d), rolling volatility "
        "(7d/14d), momentum (7d/14d), distance from the 14-day moving average."
    )
    lines.append("- **Calendar:** day of week. No other data sources are used.")
    lines.append("")
    lines.append("## 4. Validation design")
    lines.append(
        f"- Expanding-window walk-forward: initial training window "
        f"{config.WF_INITIAL_TRAIN_DAYS} days, test blocks of "
        f"{config.WF_TEST_STEP_DAYS} days, refit every block."
    )
    lines.append(
        "- One regressor per horizon, trained on forward log returns; "
        "price predictions are `close * exp(pred_return)`."
    )
    lines.append(
        "- Ridge vs RandomForest is selected **inside** each training "
        "window using a time-ordered 80/20 split, then refit on the full "
        "window. Test data is never used for model selection."
    )
    lines.append(f"- Out-of-sample period: {metrics['oos_start']} to "
                 f"{metrics['oos_end']} ({metrics['oos_days']} days).")
    lines.append("")
    lines.append("## 5. Forecast accuracy (out-of-sample)")
    lines.append("")
    lines.append("| Horizon | RMSE (price, USD) | MAE (price, USD) | Directional accuracy |")
    lines.append("|---|---|---|---|")
    for horizon, vals in metrics["per_horizon"].items():
        da = vals.get("directional_accuracy")
        da_txt = f"{da * 100:.1f}%" if da is not None else "n/a (persistence)"
        lines.append(
            f"| {horizon} | {vals['rmse_price']:.2f} | {vals['mae_price']:.2f} | {da_txt} |"
        )
    lines.append("")
    lines.append("Candidate comparison (OOS RMSE of returns, both models refit "
                 "per fold; selection itself used only in-window validation):")
    lines.append("")
    lines.append("| Horizon | Ridge | RandomForest |")
    lines.append("|---|---|---|")
    for horizon, vals in metrics["per_horizon"].items():
        cand = vals.get("model_rmse_by_candidate", {})
        ridge = cand.get("ridge")
        rf = cand.get("random_forest")
        fmt = lambda v: f"{v:.4f}" if v is not None else "n/a"
        lines.append(f"| {horizon} | {fmt(ridge)} | {fmt(rf)} |")
    lines.append("")
    lines.append("## 6. Strategy backtest (out-of-sample)")
    lines.append(
        "Rule: hold BTC when the predicted next-day return is positive, "
        "hold cash otherwise. No leverage, no shorting, no transaction "
        "costs (see limitations)."
    )
    lines.append("")
    lines.append("| Metric | Sentiment strategy | Buy and hold |")
    lines.append("|---|---|---|")
    s, b = metrics["strategy"], metrics["buy_and_hold"]
    rows = [
        ("Total return", _pct(s["total_return"]), _pct(b["total_return"])),
        ("Sharpe ratio (ann.)", f"{s['sharpe']:.2f}", f"{b['sharpe']:.2f}"),
        ("Sortino ratio (ann.)", f"{s['sortino']:.2f}", f"{b['sortino']:.2f}"),
        ("Max drawdown", _pct(s["max_drawdown"]), _pct(b["max_drawdown"])),
        ("Volatility (ann.)", _pct(s["volatility_ann"]), _pct(b["volatility_ann"])),
        ("Days", str(s["n_days"]), str(b["n_days"])),
    ]
    for name, sv, bv in rows:
        lines.append(f"| {name} | {sv} | {bv} |")
    lines.append("")
    lines.append("Charts: `outputs/price_sentiment.png` (price vs sentiment), "
                 "`outputs/visual_matrix.png` (RMSE matrix by fold x horizon), "
                 "`outputs/equity_drawdown.png` (equity curve and drawdown).")
    lines.append("")
    lines.append("## 7. Limitations")
    lines.append(
        "- VADER is a general-purpose lexicon model; crypto slang and "
        "sarcasm are only partially captured. Sentiment is a proxy, not "
        "ground truth."
    )
    lines.append(
        "- Scraped X data is a sample, not the full firehose; platform "
        "sampling and deleted posts bias the sentiment series."
    )
    lines.append(
        "- Daily bars hide intraday timing: a post late in the UTC day is "
        "treated the same as one early in the day."
    )
    lines.append(
        "- The backtest ignores fees, slippage and funding; live results "
        "would be worse."
    )
    lines.append(
        "- 180 days is a short sample spanning limited market regimes; "
        "metrics should be read as indicative, not as proof of edge."
    )
    lines.append("")
    return "\n".join(lines)


def build_sentiment_outlook() -> str:
    """Assemble factual bullets for the 1-page sentiment outlook.

    The prose narrative is finalised separately; this file is the
    data pack behind it - recent sentiment aggregates and the top
    recent headlines from the ingested news.
    """
    lines = ["# Sentiment Outlook - data pack (Bitcoin)", ""]
    feats_path = config.DAILY_FEATURES_CSV
    if feats_path.exists():
        df = pd.read_csv(feats_path)
        recent = df.tail(14)
        lines.append("## Last 14 days (daily aggregates)")
        lines.append("")
        lines.append("| Date | Close (USD) | Sentiment mean | Bullish share | Posts |")
        lines.append("|---|---|---|---|---|")
        for _, r in recent.iterrows():
            lines.append(
                f"| {r['date']} | {r['close']:.0f} | {r['sent_mean']:.3f} | "
                f"{r['bullish_share'] * 100:.0f}% | {int(r['post_volume'])} |"
            )
        lines.append("")
        last7 = df.tail(7)
        prev7 = df.tail(14).head(7)
        lines.append("## Week-on-week")
        lines.append("")
        lines.append(
            f"- Mean sentiment, last 7d: {last7['sent_mean'].mean():.3f} "
            f"(prior 7d: {prev7['sent_mean'].mean():.3f})"
        )
        lines.append(
            f"- Mean post volume, last 7d: {last7['post_volume'].mean():.0f}/day "
            f"(prior 7d: {prev7['post_volume'].mean():.0f}/day)"
        )
        lines.append(
            f"- Price change, last 7d: "
            f"{(df['close'].iloc[-1] / df['close'].iloc[-8] - 1) * 100:.2f}%"
        )
        lines.append("")
    news_path = config.NEWS_JSON
    if news_path.exists():
        with open(news_path, encoding="utf-8") as fh:
            news = json.load(fh)
        lines.append("## Recent headlines (Tavily)")
        lines.append("")
        for item in news[:10]:
            title = item.get("title", "").strip()
            url = item.get("url", "")
            if title:
                lines.append(f"- {title} ({url})")
        lines.append("")
    lines.append("_Prose summary to be written from the facts above._")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    metrics = _load_metrics()
    report = build_evaluation_report(metrics)
    (config.REPORTS_DIR / "evaluation_report.md").write_text(report, encoding="utf-8")
    outlook = build_sentiment_outlook()
    (config.REPORTS_DIR / "sentiment_outlook.md").write_text(outlook, encoding="utf-8")
    print("[report] wrote reports/evaluation_report.md and reports/sentiment_outlook.md")


if __name__ == "__main__":
    main()
