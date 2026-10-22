"""Walk-forward model training, evaluation and strategy backtest.

Validation design (the important part):

* **Expanding-window walk-forward.** The first ``WF_INITIAL_TRAIN_DAYS``
  usable rows form the first training window; the model is then tested
  on the next ``WF_TEST_STEP_DAYS`` rows, the window expands, and so on.
  No future information ever enters a training window.
* **One model per horizon.** For each target horizon n+0..n+5 a separate
  regressor predicts the forward log return to that horizon; predicted
  price levels are recovered as ``close_t * exp(pred_return)``.
  Horizon n+0 is the current price by construction (persistence).
* **Honest model selection.** Ridge vs RandomForest is chosen *inside*
  each training window: both are fit on the first 80% of the window and
  compared on the last 20% (a time-ordered validation slice). The winner
  is refit on the full window. Test folds are never used for selection.
* **Strategy backtest.** A long/cash rule on the predicted next-day
  return (long when predicted n+1 return > 0, else cash) is evaluated on
  the out-of-sample predictions only. Sharpe, Sortino and Max Drawdown
  are reported next to a buy-and-hold benchmark over the same days.

All annualisation uses 365 days - Bitcoin trades every day.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.features import FEATURE_COLUMNS

TRADING_DAYS_PER_YEAR = 365


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------
def performance_metrics(simple_returns: pd.Series) -> dict[str, float]:
    """Sharpe, Sortino, Max Drawdown and friends for a daily return series."""
    r = pd.Series(simple_returns, dtype=float).dropna()
    if len(r) < 2:
        return {
            "total_return": 0.0, "sharpe": 0.0, "sortino": 0.0,
            "max_drawdown": 0.0, "volatility_ann": 0.0, "n_days": int(len(r)),
        }
    equity = (1.0 + r).cumprod()
    total_return = float(equity.iloc[-1] - 1.0)

    std = r.std(ddof=1)
    sharpe = float(r.mean() / std * np.sqrt(TRADING_DAYS_PER_YEAR)) if std > 0 else 0.0

    downside = r[r < 0]
    dstd = downside.std(ddof=1) if len(downside) > 1 else 0.0
    sortino = (
        float(r.mean() / dstd * np.sqrt(TRADING_DAYS_PER_YEAR)) if dstd > 0 else 0.0
    )

    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    max_dd = float(drawdown.min())

    return {
        "total_return": total_return,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "volatility_ann": float(std * np.sqrt(TRADING_DAYS_PER_YEAR)),
        "n_days": int(len(r)),
    }


# ---------------------------------------------------------------------------
# Candidate models
# ---------------------------------------------------------------------------
def _candidates() -> dict[str, object]:
    return {
        # Linear model on scaled features - strong, hard-to-overfit baseline.
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        # Non-linear ensemble - can capture sentiment/price interactions.
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=5,
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def _fit_and_select(X_train, y_train):
    """Pick Ridge vs RF using a time-ordered validation slice of the
    training window only, then refit the winner on the full window.

    Returns (fitted_model, chosen_name, validation_rmse_by_model).
    """
    split = max(int(len(X_train) * 0.8), 10)
    X_fit, X_val = X_train.iloc[:split], X_train.iloc[split:]
    y_fit, y_val = y_train.iloc[:split], y_train.iloc[split:]

    val_rmse: dict[str, float] = {}
    models = _candidates()
    for name, model in models.items():
        model.fit(X_fit, y_fit)
        pred = model.predict(X_val)
        val_rmse[name] = float(np.sqrt(np.mean((pred - y_val) ** 2)))

    chosen = min(val_rmse, key=val_rmse.get)
    winner = _candidates()[chosen]
    winner.fit(X_train, y_train)
    return winner, chosen, val_rmse


# ---------------------------------------------------------------------------
# Walk-forward evaluation
# ---------------------------------------------------------------------------
def walk_forward(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Run the walk-forward evaluation.

    Returns (predictions_df, metrics_dict). ``predictions_df`` has one row
    per out-of-sample test day with predicted/actual prices per horizon,
    the fold id, the chosen model per horizon, and strategy columns.
    """
    frame = frame.reset_index(drop=True)
    n = len(frame)
    step = config.WF_TEST_STEP_DAYS
    train_end = config.WF_INITIAL_TRAIN_DAYS

    pred_rows: list[dict] = []
    # rmse accumulators: horizon -> model name -> list of squared errors
    fold_rmse: dict[int, dict[str, list[float]]] = {
        h: {"ridge": [], "random_forest": []} for h in config.HORIZONS if h > 0
    }

    fold_id = 0
    while train_end < n:
        test_end = min(train_end + step, n)
        train = frame.iloc[:train_end]
        test = frame.iloc[train_end:test_end]
        X_test = test[FEATURE_COLUMNS]

        # Train one selected model per horizon on this window.
        models_by_horizon: dict[int, tuple[object, str]] = {}
        for h in config.HORIZONS:
            if h == 0:
                continue  # persistence - handled below, nothing to train
            model, chosen, _val_rmse = _fit_and_select(
                train[FEATURE_COLUMNS], train[f"target_ret_n{h}"]
            )
            models_by_horizon[h] = (model, chosen)

            # Score BOTH candidates on this test fold for the comparison
            # table. Selection above did not use these numbers - they are
            # reporting only, so no information leaks into the choice.
            y_test = test[f"target_ret_n{h}"]
            for name, candidate in _candidates().items():
                candidate.fit(train[FEATURE_COLUMNS], train[f"target_ret_n{h}"])
                err = candidate.predict(X_test) - y_test
                fold_rmse[h][name].extend((err ** 2).tolist())

        # Predict the whole test block at once, then unpack per row.
        pred_ret = {
            h: model.predict(X_test)
            for h, (model, _chosen) in models_by_horizon.items()
        }
        for i, (_, row) in enumerate(test.iterrows()):
            out: dict = {
                "date": row["date"],
                "fold": fold_id,
                "close": float(row["close"]),
            }
            for h in config.HORIZONS:
                out[f"actual_price_n{h}"] = float(row[f"target_price_n{h}"])
                if h == 0:
                    out[f"pred_price_n{h}"] = float(row["close"])
                    out[f"pred_ret_n{h}"] = 0.0
                    out[f"model_n{h}"] = "persistence"
                else:
                    pr = float(pred_ret[h][i])
                    out[f"pred_ret_n{h}"] = pr
                    out[f"pred_price_n{h}"] = float(row["close"] * np.exp(pr))
                    out[f"model_n{h}"] = models_by_horizon[h][1]
                out[f"actual_ret_n{h}"] = float(row[f"target_ret_n{h}"])
            pred_rows.append(out)

        fold_id += 1
        train_end = test_end

    preds = pd.DataFrame(pred_rows)
    metrics = _summarise(preds, fold_rmse)
    return preds, metrics


def _summarise(preds: pd.DataFrame, fold_rmse: dict) -> dict:
    """Aggregate per-horizon accuracy + strategy/benchmark performance."""
    per_horizon: dict[str, dict] = {}
    for h in config.HORIZONS:
        err = preds[f"pred_price_n{h}"] - preds[f"actual_price_n{h}"]
        entry = {
            "rmse_price": float(np.sqrt(np.mean(err ** 2))),
            "mae_price": float(np.mean(np.abs(err))),
            "model_rmse_by_candidate": {
                name: float(np.sqrt(np.mean(vals))) if vals else None
                for name, vals in fold_rmse.get(h, {}).items()
            },
        }
        if h > 0:
            sign_hit = np.sign(preds[f"pred_ret_n{h}"]) == np.sign(
                preds[f"actual_ret_n{h}"]
            )
            entry["directional_accuracy"] = float(sign_hit.mean())
        per_horizon[f"n+{h}"] = entry

    # --- strategy: long when predicted next-day return > 0, else cash --
    actual_simple = np.expm1(preds["actual_ret_n1"])  # log -> simple return
    position = (preds["pred_ret_n1"] > 0).astype(float)
    strategy_returns = position * actual_simple
    preds["position"] = position
    preds["strategy_ret"] = strategy_returns
    preds["buyhold_ret"] = actual_simple

    return {
        "per_horizon": per_horizon,
        "strategy": performance_metrics(strategy_returns),
        "buy_and_hold": performance_metrics(actual_simple),
        "oos_days": int(len(preds)),
        "oos_start": str(preds["date"].iloc[0]) if len(preds) else None,
        "oos_end": str(preds["date"].iloc[-1]) if len(preds) else None,
    }
