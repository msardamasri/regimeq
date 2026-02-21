# RegimeIQ — Market Regime Intelligence

A Streamlit prototype that classifies the S&P 500 market environment
into Bull / Transitional / Bear using an XGBoost regression model trained
on SPY and VIX signals.

## Features
- Continuous regime score (0–100) with live market data
- Regime overlay chart with historical regime bands
- Scenario simulator with what-if sliders
- Strategy backtest vs buy-and-hold with equity curve

## Local setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download real market data (needs internet)
python train/download_data.py

# 3. Train model
python train/train_model.py

# 4. Run app
streamlit run app/app.py
```
