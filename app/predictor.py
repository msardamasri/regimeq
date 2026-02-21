"""
predictor.py — wraps regression model inference.
Returns a continuous score 0-100; regime label is derived from thresholds.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd

BEAR_THRESHOLD = 35
BULL_THRESHOLD = 65

REGIME_META = {
    "Bull": {
        "emoji" : "🟢",
        "color" : "#22c55e",
        "bg"    : "#052e16",
        "desc"  : "Markets trending up with low volatility. Conditions favour risk assets.",
        "advice": "Consider maintaining or increasing equity exposure.",
    },
    "Transitional": {
        "emoji" : "🟡",
        "color" : "#eab308",
        "bg"    : "#422006",
        "desc"  : "Mixed signals. Momentum and volatility are not clearly aligned.",
        "advice": "Consider balanced positioning and reduced concentration risk.",
    },
    "Bear": {
        "emoji" : "🔴",
        "color" : "#ef4444",
        "bg"    : "#3f0f0f",
        "desc"  : "Elevated volatility and negative momentum. Risk-off environment.",
        "advice": "Consider defensive positioning or hedging strategies.",
    },
}


@dataclass
class Prediction:
    score: float          # continuous 0-100
    regime: str           # "Bear" | "Transitional" | "Bull"
    feature_vector: pd.Series


def _score_to_regime(score: float) -> str:
    if score >= BULL_THRESHOLD:
        return "Bull"
    if score <= BEAR_THRESHOLD:
        return "Bear"
    return "Transitional"


def predict(model, scaler, feature_cols: list, feature_row: pd.Series) -> Prediction:
    X = feature_row[feature_cols].values.reshape(1, -1)
    X_scaled = scaler.transform(X)
    raw = float(model.predict(X_scaled)[0])
    score = float(np.clip(raw, 0, 100))
    return Prediction(
        score=score,
        regime=_score_to_regime(score),
        feature_vector=feature_row[feature_cols],
    )


def simulate_prediction(model, scaler, feature_cols: list,
                         base_row: pd.Series, overrides: dict) -> Prediction:
    row = base_row[feature_cols].copy()
    for k, v in overrides.items():
        if k in row.index:
            row[k] = v
    return predict(model, scaler, feature_cols, row)
