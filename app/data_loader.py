"""
data_loader.py — loads artifacts + fetches/caches market data.
Self-contained: does NOT import from train/. No path hacks needed.
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

APP_DIR   = Path(__file__).resolve().parent
ROOT_DIR  = APP_DIR.parent
ARTIFACTS = ROOT_DIR / "artifacts"

# ── Auto-generate artifacts on first run (Streamlit Cloud) ────────────────────
import sys as _sys
_sys.path.insert(0, str(ROOT_DIR))
try:
    from startup import ensure_artifacts
    ensure_artifacts()
except Exception:
    pass  # artifacts present or will fail gracefully in load_artifacts()


# ── RSI (duplicated here so data_loader has zero external deps) ───────────────
def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ── Artifact loaders ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_artifacts():
    model        = joblib.load(ARTIFACTS / "model.pkl")
    scaler       = joblib.load(ARTIFACTS / "scaler.pkl")
    feature_cols = joblib.load(ARTIFACTS / "feature_cols.pkl")
    return model, scaler, feature_cols


@st.cache_data(show_spinner=False)
def load_regime_history() -> pd.DataFrame:
    return pd.read_csv(
        ARTIFACTS / "regime_history.csv", index_col="date", parse_dates=True
    )


# ── Live market data (re-fetches at most once per hour) ───────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def load_live_data(lookback_days: int = 450) -> pd.DataFrame:
    """Try yfinance first; fall back to cached CSVs if network unavailable."""
    try:
        import yfinance as yf
        spy = yf.download("SPY",  period=f"{lookback_days}d",
                          auto_adjust=True, progress=False)
        vix = yf.download("^VIX", period=f"{lookback_days}d",
                          auto_adjust=True, progress=False)
        for d in (spy, vix):
            if isinstance(d.columns, pd.MultiIndex):
                d.columns = d.columns.get_level_values(0)
            d.index.name = "date"
        df = pd.concat(
            [spy["Close"].rename("spy_close"),
             vix["Close"].rename("vix_close")], axis=1
        ).dropna()
        df.index = pd.to_datetime(df.index)
        if len(df) < 50:
            raise ValueError("Too few rows")
        return df
    except Exception:
        spy = pd.read_csv(ARTIFACTS / "spy_raw.csv", index_col="date", parse_dates=True)
        vix = pd.read_csv(ARTIFACTS / "vix_raw.csv", index_col="date", parse_dates=True)
        return pd.concat(
            [spy["Close"].rename("spy_close").iloc[-lookback_days:],
             vix["Close"].rename("vix_close").iloc[-lookback_days:]], axis=1
        ).dropna()


# ── Feature engineering (self-contained, mirrors train/feature_engineering.py) ─
def build_live_features(live_df: pd.DataFrame) -> pd.DataFrame:
    df = live_df.copy()
    df["ret_1d"]         = df["spy_close"].pct_change()
    df["ret_5d"]         = df["spy_close"].pct_change(5)
    df["ret_21d"]        = df["spy_close"].pct_change(21)
    df["vol_21d"]        = df["ret_1d"].rolling(21).std() * np.sqrt(252)
    df["vix_level"]      = df["vix_close"]
    df["vix_change_5d"]  = df["vix_close"].pct_change(5)
    rolling_max          = df["spy_close"].rolling(252, min_periods=21).max()
    df["drawdown"]       = (df["spy_close"] - rolling_max) / rolling_max
    df["rsi_14"]         = _compute_rsi(df["spy_close"], 14)
    ma200                = df["spy_close"].rolling(200, min_periods=50).mean()
    df["price_vs_200ma"] = df["spy_close"] / ma200 - 1
    df.dropna(inplace=True)
    return df