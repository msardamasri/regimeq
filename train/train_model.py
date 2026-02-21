"""
train_model.py
--------------
Trains XGBRegressor to predict the continuous regime score (0-100).
Thresholds are applied at inference time — not during training.

Exports:
    model.pkl          – XGBRegressor
    scaler.pkl         – StandardScaler
    feature_cols.pkl   – ordered feature list
    regime_history.csv – date-indexed scores + labels for chart overlay
"""

import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).parent))
from feature_engineering import build_features, FEATURE_COLS, BEAR_THRESHOLD, BULL_THRESHOLD

ARTIFACTS = Path(__file__).parent.parent / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

print("=== PulseCheck: Regression Model Training ===\n")
print("Building feature matrix...")
df = build_features()

X = df[FEATURE_COLS].values
y = df["regime_score"].values

print(f"  Dataset : {len(df)} rows | Features: {len(FEATURE_COLS)}")
print(f"  Score   : mean={y.mean():.1f}  std={y.std():.1f}  min={y.min():.1f}  max={y.max():.1f}")
dist = df["regime_label"].value_counts()
print(f"  Labels  : Bull={dist.get('Bull',0)}  Trans={dist.get('Transitional',0)}  Bear={dist.get('Bear',0)}")


# ── Time-series cross-validation ─────────────────────────────────────────────
print("\nRunning time-series CV (5 folds)...")
tscv = TimeSeriesSplit(n_splits=5)
mae_scores, r2_scores = [], []

for fold, (tr_idx, val_idx) in enumerate(tscv.split(X), 1):
    X_tr, X_val = X[tr_idx], X[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]

    sc  = StandardScaler().fit(X_tr)
    clf = XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbosity=0,
    )
    clf.fit(sc.transform(X_tr), y_tr,
            eval_set=[(sc.transform(X_val), y_val)],
            verbose=False)

    preds = clf.predict(sc.transform(X_val)).clip(0, 100)
    mae   = mean_absolute_error(y_val, preds)
    r2    = r2_score(y_val, preds)
    mae_scores.append(mae)
    r2_scores.append(r2)
    print(f"  Fold {fold}: MAE={mae:.2f} pts  R²={r2:.3f}")

print(f"\n  Mean CV  →  MAE={np.mean(mae_scores):.2f} ± {np.std(mae_scores):.2f}  |  R²={np.mean(r2_scores):.3f}")


# ── Final model on full dataset ───────────────────────────────────────────────
print("\nTraining final model on full dataset...")
scaler_final = StandardScaler().fit(X)
model_final  = XGBRegressor(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    verbosity=0,
)
model_final.fit(scaler_final.transform(X), y)

preds_train = model_final.predict(scaler_final.transform(X)).clip(0, 100)
print(f"  In-sample MAE: {mean_absolute_error(y, preds_train):.2f} pts")
print(f"  In-sample R² : {r2_score(y, preds_train):.4f}")


# ── Generate regime history for chart overlay ─────────────────────────────────
print("\nGenerating regime history...")
scores = preds_train
labels = np.where(scores > BULL_THRESHOLD, "Bull",
         np.where(scores < BEAR_THRESHOLD, "Bear", "Transitional"))

regime_history = pd.DataFrame({
    "regime_score" : scores,
    "regime_label" : labels,
    "spy_close"    : df["spy_close"].values,
    "vix_level"    : df["vix_level"].values,
}, index=df.index)
regime_history.index.name = "date"


# ── Export artifacts ──────────────────────────────────────────────────────────
print("\nExporting artifacts...")
joblib.dump(model_final,   ARTIFACTS / "model.pkl")
joblib.dump(scaler_final,  ARTIFACTS / "scaler.pkl")
joblib.dump(FEATURE_COLS,  ARTIFACTS / "feature_cols.pkl")
regime_history.to_csv(ARTIFACTS / "regime_history.csv")

print(f"""
✓ Artifacts saved to {ARTIFACTS}/
    model.pkl           XGBRegressor — predicts continuous score 0-100
    scaler.pkl
    feature_cols.pkl
    regime_history.csv  ({len(regime_history)} rows)

Thresholds at inference: Bear < {BEAR_THRESHOLD} | Transitional {BEAR_THRESHOLD}–{BULL_THRESHOLD} | Bull > {BULL_THRESHOLD}
Done.
""")
