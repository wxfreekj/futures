"""Educational ES/NQ trade-event meta-label starter.

Runs CPU logistic regression and gradient boosting on REAL labeled trade-event CSVs.
Strictly chronological train / validation / untouched test. Purges rows whose
labels end after the next segment begins. Does NOT place orders or imply edge.

Usage:
  python trade_event_baseline.py --csv trade_events.csv --instrument ES

CSV required columns:
  event_time, label_end_time, instrument, direction,
  atr7, atr21, adx21, box_width_atr, prebreak_volume_ratio,
  trend_alignment, hour_ct, baseline_net_pnl

All timestamp columns must be ISO 8601 with offsets or UTC 'Z'. All features
must be known as of event_time. Each baseline_net_pnl must include simulated
commissions, spread, slippage and the fixed strategy's actual exit policy.
The provided features are a proposed schema, not demonstrated predictive signals.

This is a starter example, NOT a production-grade complete walk-forward tester:
it uses one train/validation/test split; add more folds, cost sensitivity, and
uncertainty analysis before treating any result as evidence of robustness.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["atr7", "atr21", "adx21", "box_width_atr",
            "prebreak_volume_ratio", "trend_alignment", "hour_ct"]
REQUIRED = ["event_time", "label_end_time", "instrument", "direction",
            "baseline_net_pnl"] + FEATURES


def score_stats(df: pd.DataFrame, take: np.ndarray, name: str) -> dict:
    pnl = df.loc[take, "baseline_net_pnl"].to_numpy(dtype=float)
    n = int(pnl.size)
    cumulative = np.cumsum(pnl)
    drawdown = (np.maximum.accumulate(np.r_[0, cumulative])[1:] - cumulative)
    return {
        "strategy": name,
        "eligible_events": len(df),
        "trades_taken": n,
        "acceptance_rate": n / len(df) if len(df) else 0.0,
        "net_total_usd": float(pnl.sum()) if n else 0.0,
        "net_per_taken_usd": float(pnl.mean()) if n else float("nan"),
        "net_per_eligible_usd": float(pnl.sum() / len(df)) if len(df) else 0.0,
        "win_rate": float(np.mean(pnl > 0)) if n else float("nan"),
        "max_trade_to_trade_drawdown_usd": float(drawdown.max()) if n else 0.0,
    }


def pipeline(kind: str) -> Pipeline:
    if kind == "logistic":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ])
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", HistGradientBoostingClassifier(max_iter=120, max_leaf_nodes=7,
                                                   learning_rate=0.05, l2_regularization=2,
                                                   random_state=42)),
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--instrument", choices=["ES", "NQ"], required=True)
    parser.add_argument("--out", default="model_comparison.csv", type=Path)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    missing = sorted(set(REQUIRED).difference(df.columns))
    if missing:
        parser.error(f"Missing columns: {missing}")
    df = df[df.instrument.astype(str).str.upper() == args.instrument].copy()
    if len(df) < 150:
        parser.error(f"Only {len(df)} {args.instrument} events; require at least 150 for this starter split.")
    for col in ["event_time", "label_end_time"]:
        df[col] = pd.to_datetime(df[col], utc=True, errors="raise")
    df = df.sort_values("event_time").reset_index(drop=True)
    if df.event_time.duplicated().any():
        parser.error("Duplicate event_time detected. Add an event_id/instrument-time key and resolve duplicates.")
    if not (df.label_end_time >= df.event_time).all():
        parser.error("label_end_time cannot precede event_time.")
    for col in FEATURES + ["baseline_net_pnl"]:
        df[col] = pd.to_numeric(df[col], errors="raise")
    if df[FEATURES + ["baseline_net_pnl"]].isna().any().any():
        parser.error("Missing/NaN numeric values detected; inspect and resolve before research.")
    if not np.isfinite(df[FEATURES + ["baseline_net_pnl"]].to_numpy(float)).all():
        parser.error("Non-finite values detected.")
    df["positive_net_trade"] = (df.baseline_net_pnl > 0).astype(int)
    n=len(df); idx1=int(n*.6);idx2=int(n*.8)
    train=df.iloc[:idx1].copy();valid=df.iloc[idx1:idx2].copy();test=df.iloc[idx2:].copy()
    # Strict overlap exclusion: no training outcome extends into later segment.
    train=train[train.label_end_time < valid.event_time.min()].copy()
    valid=valid[valid.label_end_time < test.event_time.min()].copy()
    if min(map(len,[train,valid,test])) < 20:
        parser.error("After purging labels, a split has fewer than 20 events.")
    if train.positive_net_trade.nunique()<2:
        parser.error("Training labels contain only one class; cannot fit binary classifier.")
    print(f"Instrument: {args.instrument} | total events {n} | chronological split after purge:")
    for label,d in [("train",train),("validation",valid),("untouched test",test)]:
        print(f"  {label}: {len(d)} | {d.event_time.min()} through {d.event_time.max()}")
    # Fit only on train. Use validation to pick threshold. Evaluate test once.
    rows=[score_stats(test, np.ones(len(test),dtype=bool),"baseline: take all")]
    rules=(test.trend_alignment>=1)&(test.adx21>=25)
    rows.append(score_stats(test, rules.to_numpy(),"static: aligned and ADX>=25"))
    output=test[["event_time","label_end_time","instrument","direction","baseline_net_pnl"]].copy()
    for kind in ("logistic","boosted-tree"):
        model=pipeline(kind)
        model.fit(train[FEATURES],train.positive_net_trade)
        valid_scores=model.predict_proba(valid[FEATURES])[:,1]
        # Example only: maximizing validation net PnL can still overfit. Keep grid small.
        thresholds=[0.45,0.50,0.55,0.60,0.65,0.70]
        val_pnl=valid.baseline_net_pnl.to_numpy()
        evaluations=[(float(val_pnl[valid_scores >= t].sum()),t) for t in thresholds]
        _,cutoff=max(evaluations, key=lambda x:(x[0],-x[1]))
        scores=model.predict_proba(test[FEATURES])[:,1]
        take=scores>=cutoff
        output[f"{kind}_prob"] = scores
        output[f"{kind}_accept"] = take
        rows.append(score_stats(test,take,f"{kind}: cut={cutoff:.2f}"))
        print(f"  {kind} validation-chosen threshold: {cutoff:.2f}")
    report=pd.DataFrame(rows)
    print("\nHoldout net trading comparison (NOT a forward/live result):")
    print(report.to_string(index=False,float_format=lambda x:f"{x:.3f}"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.out,index=False)
    output.to_csv(args.out.with_name("test_event_predictions.csv"),index=False)
    print("\nReports written:",args.out, "and test_event_predictions.csv")
    print("NOTE: Add rolling purged walk-forward splits, model calibration, cost-stress tests,")
    print("uncertainty bounds, and simulated execution before any deployment consideration.")

if __name__ == "__main__":
    main()