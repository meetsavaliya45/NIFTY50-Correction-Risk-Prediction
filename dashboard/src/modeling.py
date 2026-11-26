from __future__ import annotations

from pathlib import Path
import warnings
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    average_precision_score, roc_auc_score, confusion_matrix,
    precision_recall_curve
)

TARGET = "correction_within_20d"
HORIZON = 20
RANDOM_STATE = 42

# Features intentionally exclude NIFTY P/E and USD/INR reference rate because
# the selected public quote feed does not reliably supply those fields.
FEATURE_COLUMNS = [
    "nifty50_open", "nifty50_high", "nifty50_low", "nifty50_close",
    "india_vix_close", "usd_inr_close",
    "nifty_return_1d_recalc", "nifty_return_5d_recalc",
    "nifty_return_20d_recalc", "nifty_return_60d_recalc",
    "nifty_volatility_20d_recalc", "vix_return_1d_recalc",
    "vix_return_5d_recalc", "usd_inr_return_1d_recalc",
    "usd_inr_return_5d_recalc",
]

def make_features(raw: pd.DataFrame) -> pd.DataFrame:
    d = raw.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    for c in ["nifty50_open", "nifty50_high", "nifty50_low", "nifty50_close",
              "india_vix_close", "usd_inr_close"]:
        if c not in d:
            d[c] = np.nan
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["nifty_return_1d_recalc"] = d["nifty50_close"].pct_change(fill_method=None)
    d["nifty_return_5d_recalc"] = d["nifty50_close"].pct_change(5, fill_method=None)
    d["nifty_return_20d_recalc"] = d["nifty50_close"].pct_change(20, fill_method=None)
    d["nifty_return_60d_recalc"] = d["nifty50_close"].pct_change(60, fill_method=None)
    d["nifty_volatility_20d_recalc"] = d["nifty_return_1d_recalc"].rolling(20, min_periods=20).std()
    d["vix_return_1d_recalc"] = d["india_vix_close"].pct_change(fill_method=None)
    d["vix_return_5d_recalc"] = d["india_vix_close"].pct_change(5, fill_method=None)
    d["usd_inr_return_1d_recalc"] = d["usd_inr_close"].pct_change(fill_method=None)
    d["usd_inr_return_5d_recalc"] = d["usd_inr_close"].pct_change(5, fill_method=None)
    return d

def split_chronologically(d: pd.DataFrame):
    n = len(d)
    a, b = int(n * .70), int(n * .85)
    train = d.iloc[:max(0, a-HORIZON)].copy()
    val = d.iloc[a:max(a, b-HORIZON)].copy()
    test = d.iloc[b:].copy()
    if min(len(train), len(val), len(test)) == 0:
        raise ValueError("Not enough labeled observations for time-ordered train/validation/test splits.")
    return train, val, test

def best_f1_threshold(y, p):
    precision, recall, thresholds = precision_recall_curve(y, p)
    if len(thresholds) == 0:
        return .5
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.nanargmax(f1))])

def load_and_train(path: Path):
    raw = pd.read_csv(path)
    if TARGET not in raw:
        raise ValueError(f"Dataset must contain {TARGET}.")
    d = make_features(raw)
    d[TARGET] = pd.to_numeric(d[TARGET], errors="coerce")
    d = d[d[TARGET].isin([0, 1])].copy()
    d[TARGET] = d[TARGET].astype(int)

    train, val, test = split_chronologically(d)
    feature_cols = [c for c in FEATURE_COLUMNS if c in d.columns]
    medians = train[feature_cols].median()

    lr = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE)),
    ])
    rf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, max_features="sqrt",
            class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1
        )),
    ])
    models = {"Logistic Regression": lr, "Random Forest": rf}
    thresholds, rows, cms, pred_rows = {}, [], {}, []
    Xtr, ytr = train[feature_cols], train[TARGET]
    Xv, yv = val[feature_cols], val[TARGET]
    Xt, yt = test[feature_cols], test[TARGET]

    for name, model in models.items():
        model.fit(Xtr, ytr)
        vprob = model.predict_proba(Xv)[:, 1]
        th = best_f1_threshold(yv, vprob)
        thresholds[name] = th
        prob = model.predict_proba(Xt)[:, 1]
        pred = (prob >= th).astype(int)
        cm = confusion_matrix(yt, pred, labels=[0, 1])
        cms[name] = cm
        rows.append({
            "model": name, "threshold": th, "test_rows": len(yt),
            "precision": precision_score(yt, pred, zero_division=0),
            "recall": recall_score(yt, pred, zero_division=0),
            "f1": f1_score(yt, pred, zero_division=0),
            "accuracy": accuracy_score(yt, pred),
            "pr_auc": average_precision_score(yt, prob),
            "roc_auc": roc_auc_score(yt, prob) if yt.nunique() > 1 else np.nan,
            "false_alarms": int(cm[0, 1]), "missed_positive_labels": int(cm[1, 0]),
        })
        pred_rows.append(pd.DataFrame({
            "date": test["date"].values, "model": name, "actual_label": yt.values,
            "predicted_label": pred, "score": prob, "threshold": th,
        }))

    # VIX threshold benchmark: tune on validation only.
    if "india_vix_close" in feature_cols:
        candidates = np.unique(np.nanquantile(val["india_vix_close"].dropna(), np.linspace(.05, .95, 91)))
        fallback = float(medians.get("india_vix_close", train["india_vix_close"].median()))
        best = None
        for th in candidates:
            vp = (val["india_vix_close"].fillna(fallback).to_numpy() >= th).astype(int)
            score = f1_score(yv, vp, zero_division=0)
            if best is None or score > best[0]:
                best = (score, float(th))
        vix_th = best[1]
        thresholds["India VIX threshold rule"] = vix_th
        vp = (test["india_vix_close"].fillna(fallback).to_numpy() >= vix_th).astype(int)
        cm = confusion_matrix(yt, vp, labels=[0, 1])
        cms["India VIX threshold rule"] = cm
        rows.append({
            "model": "India VIX threshold rule", "threshold": vix_th, "test_rows": len(yt),
            "precision": precision_score(yt, vp, zero_division=0),
            "recall": recall_score(yt, vp, zero_division=0),
            "f1": f1_score(yt, vp, zero_division=0),
            "accuracy": accuracy_score(yt, vp),
            "pr_auc": average_precision_score(yt, vp.astype(float)),
            "roc_auc": roc_auc_score(yt, vp) if yt.nunique() > 1 else np.nan,
            "false_alarms": int(cm[0, 1]), "missed_positive_labels": int(cm[1, 0]),
        })
        pred_rows.append(pd.DataFrame({
            "date": test["date"].values, "model": "India VIX threshold rule",
            "actual_label": yt.values, "predicted_label": vp, "score": vp.astype(float),
            "threshold": vix_th,
        }))

    rf_model = models["Random Forest"].named_steps["model"]
    importance = pd.DataFrame({"feature": feature_cols, "importance": rf_model.feature_importances_}).sort_values(
        "importance", ascending=False
    )
    return {
        "data": d, "models": models, "thresholds": thresholds,
        "metrics": pd.DataFrame(rows), "confusion_matrices": cms,
        "test_predictions": pd.concat(pred_rows, ignore_index=True),
        "feature_importance": importance, "training_medians": medians,
        "train": train, "validation": val, "test": test,
    }

def _download(ticker: str, period="2y"):
    import yfinance as yf
    frame = yf.download(ticker, period=period, interval="1d", auto_adjust=False, progress=False, threads=False)
    if frame is None or frame.empty:
        return pd.DataFrame()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    frame = frame.reset_index()
    date_col = "Date" if "Date" in frame.columns else frame.columns[0]
    frame = frame.rename(columns={date_col: "date"})
    frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None)
    return frame

def fetch_market_data():
    try:
        nifty = _download("^NSEI")
        vix = _download("^INDIAVIX")
        fx = _download("USDINR=X")
        if nifty.empty:
            return None
        out = nifty[["date", "Open", "High", "Low", "Close"]].rename(columns={
            "Open": "nifty50_open", "High": "nifty50_high", "Low": "nifty50_low", "Close": "nifty50_close"
        })
        if not vix.empty and "Close" in vix:
            out = out.merge(vix[["date", "Close"]].rename(columns={"Close": "india_vix_close"}), on="date", how="left")
        else:
            out["india_vix_close"] = np.nan
        if not fx.empty and "Close" in fx:
            out = out.merge(fx[["date", "Close"]].rename(columns={"Close": "usd_inr_close"}), on="date", how="left")
        else:
            out["usd_inr_close"] = np.nan
        return make_features(out).dropna(subset=["nifty50_close"]).tail(100)
    except Exception as exc:
        warnings.warn(f"Market-data retrieval failed: {exc}")
        return None

def build_live_feature_row(market: pd.DataFrame, training_medians: pd.Series):
    d = make_features(market)
    if d.empty:
        raise ValueError("No usable market rows were returned.")
    row = d.tail(1)[FEATURE_COLUMNS].copy()
    missing = [c for c in FEATURE_COLUMNS if c not in row or row[c].isna().iloc[0]]
    for c in FEATURE_COLUMNS:
        if c not in row:
            row[c] = np.nan
        if row[c].isna().iloc[0]:
            row.loc[row.index[0], c] = training_medians.get(c, np.nan)
    return row[FEATURE_COLUMNS], missing
