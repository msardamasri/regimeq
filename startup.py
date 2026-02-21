"""
startup.py — run once on Streamlit Cloud if artifacts are missing.
Streamlit Cloud calls this via the entrypoint trick, OR it is imported
by data_loader.py when artifacts are not found.

On your local machine you don't need this — just run train/train_model.py.
"""

import sys
from pathlib import Path

ROOT      = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
TRAIN     = ROOT / "train"
sys.path.insert(0, str(TRAIN))


def ensure_artifacts():
    """Re-run full training pipeline if any artifact is missing."""
    needed = ["model.pkl", "scaler.pkl", "feature_cols.pkl", "regime_history.csv"]
    if all((ARTIFACTS / f).exists() for f in needed):
        return  # nothing to do

    print("[startup] Artifacts missing — running training pipeline...")
    ARTIFACTS.mkdir(exist_ok=True)

    # 1. Download or generate data
    spy_path = ARTIFACTS / "spy_raw.csv"
    vix_path = ARTIFACTS / "vix_raw.csv"
    if not spy_path.exists() or not vix_path.exists():
        try:
            import yfinance as yf
            import pandas as pd
            print("[startup] Downloading SPY + VIX from Yahoo Finance...")
            spy = yf.download("SPY",  start="2000-01-01", auto_adjust=True, progress=False)
            vix = yf.download("^VIX", start="2000-01-01", auto_adjust=True, progress=False)
            if isinstance(spy.columns, pd.MultiIndex):
                spy.columns = spy.columns.get_level_values(0)
            if isinstance(vix.columns, pd.MultiIndex):
                vix.columns = vix.columns.get_level_values(0)
            spy.index.name = "date"
            vix.index.name = "date"
            spy.to_csv(spy_path)
            vix.to_csv(vix_path)
            print(f"[startup] Downloaded {len(spy)} rows of SPY, {len(vix)} rows of VIX")
        except Exception as e:
            print(f"[startup] yfinance failed ({e}), using synthetic data...")
            import generate_synthetic_data  # noqa: F401

    # 2. Train model
    import train_model  # noqa: F401
    print("[startup] Training complete.")


if __name__ == "__main__":
    ensure_artifacts()
