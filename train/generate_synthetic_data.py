"""
generate_synthetic_data.py
--------------------------
Generates realistic synthetic SPY + VIX data for pipeline testing
when Yahoo Finance is unavailable (e.g., restricted network).

On YOUR machine, run download_data.py instead — this is only for validation.
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "artifacts"
DATA_DIR.mkdir(exist_ok=True)

np.random.seed(42)
n_days = 6000  # ~24 years of trading days

dates = pd.bdate_range(start="2000-01-03", periods=n_days)

# ── Simulate SPY with regime switches ────────────────────────────────────────
# Regime schedule: alternating bull/bear/transition periods
price = 100.0
prices = [price]
vix_vals = [15.0]

regime_schedule = []
t = 0
while t < n_days:
    r = np.random.choice(["bull", "bear", "trans"], p=[0.55, 0.20, 0.25])
    length = int(np.random.uniform(60, 400))
    regime_schedule.extend([r] * length)
    t += length
regime_schedule = regime_schedule[:n_days]

for i in range(1, n_days):
    reg = regime_schedule[i]
    if reg == "bull":
        mu, sigma, vix_target = 0.0004, 0.008, 14.0
    elif reg == "bear":
        mu, sigma, vix_target = -0.0005, 0.018, 32.0
    else:
        mu, sigma, vix_target = 0.0001, 0.012, 22.0

    ret = np.random.normal(mu, sigma)
    price = prices[-1] * (1 + ret)
    prices.append(max(price, 10))

    vix = vix_vals[-1] * 0.95 + vix_target * 0.05 + np.random.normal(0, 1.5)
    vix_vals.append(max(vix, 8))

# ── Build SPY DataFrame ───────────────────────────────────────────────────────
spy_close = np.array(prices)
spy_df = pd.DataFrame({
    "date"  : dates,
    "Open"  : spy_close * (1 + np.random.normal(0, 0.003, n_days)),
    "High"  : spy_close * (1 + np.abs(np.random.normal(0, 0.006, n_days))),
    "Low"   : spy_close * (1 - np.abs(np.random.normal(0, 0.006, n_days))),
    "Close" : spy_close,
    "Volume": np.random.randint(50_000_000, 200_000_000, n_days),
}).set_index("date")

# ── Build VIX DataFrame ───────────────────────────────────────────────────────
vix_arr = np.array(vix_vals)
vix_df = pd.DataFrame({
    "date"  : dates,
    "Open"  : vix_arr,
    "High"  : vix_arr * (1 + np.abs(np.random.normal(0, 0.04, n_days))),
    "Low"   : vix_arr * (1 - np.abs(np.random.normal(0, 0.04, n_days))),
    "Close" : vix_arr,
    "Volume": np.zeros(n_days, dtype=int),
}).set_index("date")

spy_df.to_csv(DATA_DIR / "spy_raw.csv")
vix_df.to_csv(DATA_DIR / "vix_raw.csv")

print(f"✓ Synthetic SPY saved ({len(spy_df)} rows)")
print(f"✓ Synthetic VIX saved ({len(vix_df)} rows)")
print("  NOTE: Replace with real data via download_data.py on your machine.")
