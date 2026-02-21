"""
download_data.py
----------------
Downloads SPY and VIX daily data from Yahoo Finance and saves to CSV.
Run once offline before training.

Usage:
    python train/download_data.py
"""

import yfinance as yf
import pandas as pd
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────
START_DATE = "2000-01-01"   # far back enough to include dot-com, GFC, COVID
END_DATE   = None           # None = today
DATA_DIR   = Path(__file__).parent.parent / "artifacts"
DATA_DIR.mkdir(exist_ok=True)

SPY_PATH = DATA_DIR / "spy_raw.csv"
VIX_PATH = DATA_DIR / "vix_raw.csv"


def download_ticker(ticker: str, start: str, end=None) -> pd.DataFrame:
    print(f"  Downloading {ticker}...")
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    # yfinance sometimes returns MultiIndex columns — flatten
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    return df


def main():
    print("=== PulseCheck: Data Download ===\n")

    spy = download_ticker("SPY", START_DATE, END_DATE)
    vix = download_ticker("^VIX", START_DATE, END_DATE)

    spy.to_csv(SPY_PATH)
    vix.to_csv(VIX_PATH)

    print(f"\n✓ SPY saved  → {SPY_PATH}  ({len(spy)} rows, {spy.index[0].date()} to {spy.index[-1].date()})")
    print(f"✓ VIX saved  → {VIX_PATH}  ({len(vix)} rows, {vix.index[0].date()} to {vix.index[-1].date()})")
    print("\nAll done. Run feature_engineering.py next.")


if __name__ == "__main__":
    main()
