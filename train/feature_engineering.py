"""
feature_engineering.py
-----------------------
Builds features and the continuous regime score (0-100).

Score interpretation:
  0   = extreme bear (max fear, max drawdown)
  50  = transitional / neutral
  100 = strong bull (low vol, strong momentum)

Thresholds applied AFTER prediction:
  score < 35  → Bear
  35–65       → Transitional
  score > 65  → Bull
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "artifacts"

FEATURE_COLS = [
    "ret_5d", "ret_21d", "vol_21d",
    "vix_level", "vix_change_5d",
    "drawdown", "rsi_14", "price_vs_200ma",
]

BEAR_THRESHOLD = 35
BULL_THRESHOLD = 65


def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _build_regime_score(df: pd.DataFrame) -> pd.Series:
    """
    Continuous market health score [0, 100] built from normalised signals.
    Each component is clamped to [-1, 1] then weighted into a 0-centred sum,
    finally shifted to [0, 100].

    Weights chosen so no single feature can swing the score by more than ±20pts.
    """
    s = pd.Series(0.0, index=df.index)

    # 21-day return: ±15% maps to ±1  →  ±22 pts
    s += np.clip(df["ret_21d"] / 0.15, -1, 1) * 22

    # VIX: 10 → +20, 40 → -20  (centred at VIX=20)
    s += np.clip((20 - df["vix_level"]) / 15, -1, 1) * 20

    # Realised vol: annualised 8% → +15, 45% → -15  (centred at 20%)
    s += np.clip((0.20 - df["vol_21d"]) / 0.20, -1, 1) * 15

    # Price vs 200-MA: ±10% maps to ±1  →  ±13 pts
    s += np.clip(df["price_vs_200ma"] / 0.10, -1, 1) * 13

    # RSI: centred at 50, ±30 maps to ±1  →  ±10 pts
    s += np.clip((df["rsi_14"] - 50) / 30, -1, 1) * 10

    # 5-day VIX change: ±30% maps to ±1  →  ±8 pts  (rising VIX is bearish)
    s += np.clip(-df["vix_change_5d"] / 0.30, -1, 1) * 8

    # 5-day return: ±5% maps to ±1  →  ±7 pts
    s += np.clip(df["ret_5d"] / 0.05, -1, 1) * 7

    # Drawdown: 0 → +5, -30% → -5  (minor contribution)
    s += np.clip(df["drawdown"] / -0.30, -1, 0) * -5

    # Shift from [-100, +100] → [0, 100] and clip
    score = (s + 100) / 2
    return score.clip(0, 100)


def build_features(spy_path=None, vix_path=None) -> pd.DataFrame:
    spy_path = spy_path or DATA_DIR / "spy_raw.csv"
    vix_path = vix_path or DATA_DIR / "vix_raw.csv"

    spy = pd.read_csv(spy_path, index_col="date", parse_dates=True)
    vix = pd.read_csv(vix_path, index_col="date", parse_dates=True)

    spy_close = spy["Close"].rename("spy_close")
    vix_close = vix["Close"].rename("vix_close")
    df = pd.concat([spy_close, vix_close], axis=1).dropna()

    df["ret_1d"]         = df["spy_close"].pct_change()
    df["ret_5d"]         = df["spy_close"].pct_change(5)
    df["ret_21d"]        = df["spy_close"].pct_change(21)
    df["vol_21d"]        = df["ret_1d"].rolling(21).std() * np.sqrt(252)
    df["vix_level"]      = df["vix_close"]
    df["vix_change_5d"]  = df["vix_close"].pct_change(5)
    rolling_max          = df["spy_close"].rolling(252, min_periods=1).max()
    df["drawdown"]       = (df["spy_close"] - rolling_max) / rolling_max
    df["rsi_14"]         = _compute_rsi(df["spy_close"], 14)
    ma200                = df["spy_close"].rolling(200).mean()
    df["price_vs_200ma"] = df["spy_close"] / ma200 - 1

    # Continuous target
    df["regime_score"] = _build_regime_score(df)

    # Discrete label derived from score (used only for chart overlay / history)
    df["regime_label"] = pd.cut(
        df["regime_score"],
        bins=[-np.inf, BEAR_THRESHOLD, BULL_THRESHOLD, np.inf],
        labels=["Bear", "Transitional", "Bull"],
    )

    df.dropna(inplace=True)
    return df


if __name__ == "__main__":
    df = build_features()
    print("=== Feature Matrix Diagnostics ===\n")
    print(f"Shape : {df.shape}")
    print(f"Range : {df.index[0].date()} → {df.index[-1].date()}")
    print(f"\nRegime score stats:")
    print(df["regime_score"].describe().round(2).to_string())
    print(f"\nRegime distribution (from thresholds):")
    print(df["regime_label"].value_counts().to_string())
    print(f"\nMissing values: {df[FEATURE_COLS].isna().sum().sum()}")
