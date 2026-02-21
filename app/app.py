"""
app.py — RegimeIQ | Market Regime Intelligence
Run with: streamlit run app/app.py
"""

import sys
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

APP_DIR  = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
for p in (str(APP_DIR), str(ROOT_DIR / "train")):
    if p not in sys.path:
        sys.path.insert(0, p)

from data_loader import load_artifacts, load_regime_history, load_live_data, build_live_features
from predictor   import predict, simulate_prediction, REGIME_META, BEAR_THRESHOLD, BULL_THRESHOLD
from backtest    import run_backtest
from charts      import (score_bar_chart, regime_overlay_chart, score_history_chart,
                          feature_importance_chart, regime_donut, simulate_gauge,
                          backtest_chart)

st.set_page_config(
    page_title="RegimeIQ · Market Regime",
    page_icon="📡", layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { background-color: #0f172a; }
    section[data-testid="stSidebar"] { background-color: #0f172a; border-right: 1px solid #1e293b; }
    [data-testid="stMetric"] {
        background-color: #1e293b; border: 1px solid #334155;
        border-radius: 10px; padding: 14px 18px;
    }
    [data-testid="stMetricLabel"] { font-size: 12px !important; color: #94a3b8 !important; }
    [data-testid="stMetricValue"] { font-size: 24px !important; color: #e2e8f0 !important; }
    .stTabs [data-baseweb="tab"] { color: #94a3b8; background: transparent; }
    .stTabs [aria-selected="true"] { color: #38bdf8 !important; border-bottom: 2px solid #38bdf8; }
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)


# ═══ SESSION STATE ════════════════════════════════════════════════════════════
def _init_state(key, default):
    if key not in st.session_state:
        st.session_state[key] = default


# ═══ SIDEBAR ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## RegimeIQ")
    st.markdown("*Market Regime Intelligence*")
    st.divider()
    lookback = st.slider("Chart lookback (trading days)", 60, 1000, 400, 20)
    donut_lookback = st.selectbox(
        "Regime distribution window",
        [63, 126, 252, 504],
        format_func=lambda x: {63:"3 months",126:"6 months",252:"1 year",504:"2 years"}[x],
        index=2,
    )
    st.divider()
    show_vix_overlay   = st.toggle("Show VIX overlay",   value=True)
    show_score_history = st.toggle("Show score history", value=True)
    st.divider()
    st.markdown("**Model**")
    st.caption(
        f"XGBRegressor → continuous score 0–100.\n\n"
        f"Bear < {BEAR_THRESHOLD} · Bull > {BULL_THRESHOLD}.\n\n"
        "Not financial advice."
    )
    st.divider()
    if st.button("Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()


# ═══ LOAD DATA ════════════════════════════════════════════════════════════════
with st.spinner("Loading model…"):
    model, scaler, feature_cols = load_artifacts()
with st.spinner("Fetching market data…"):
    live_raw = load_live_data(lookback_days=max(lookback + 220, 450))
    history  = load_regime_history()

live_features = build_live_features(live_raw)
if live_features.empty:
    st.error("Not enough data. Try refreshing.")
    st.stop()

latest_row  = live_features.iloc[-1]
pred        = predict(model, scaler, feature_cols, latest_row)
meta        = REGIME_META[pred.regime]
latest_date = live_features.index[-1]
data_source = (
    "Live (yfinance)"
    if live_raw.index[-1] >= pd.Timestamp.today() - pd.Timedelta(days=7)
    else "Cached data"
)

vix_now = float(latest_row["vix_level"])
vix_chg = float(latest_row["vix_change_5d"]) * 100
ret21   = float(latest_row["ret_21d"]) * 100
ret5    = float(latest_row["ret_5d"]) * 100
vol     = float(latest_row["vol_21d"]) * 100
rsi     = float(latest_row["rsi_14"])
dd      = float(latest_row["drawdown"]) * 100
vs_ma   = float(latest_row["price_vs_200ma"]) * 100

_init_state("sim_vix",   vix_now)
_init_state("sim_ret21", ret21)
_init_state("sim_rsi",   rsi)
_init_state("sim_vol",   vol)
_init_state("sim_dd",    dd)
_init_state("sim_vs_ma", vs_ma)


# ═══ HEADER ═══════════════════════════════════════════════════════════════════
hc, dc = st.columns([4, 1])
with hc:
    st.markdown("# RegimeIQ")
with dc:
    st.caption(f"**As of:** {latest_date.strftime('%b %d, %Y')}")
    st.caption(f"**Source:** {data_source}")


# ═══ SCORE BAR ════════════════════════════════════════════════════════════════
badge_col, score_col = st.columns([1, 3])

with badge_col:
    st.markdown(
        f'<div style="background:{meta["bg"]};border:1px solid {meta["color"]};'
        f'border-radius:10px;padding:14px 20px;text-align:center;height:130px;'
        f'display:flex;flex-direction:column;justify-content:center;gap:6px">'
        f'<div style="font-size:36px">{meta["emoji"]}</div>'
        f'<div style="font-size:20px;font-weight:800;color:{meta["color"]}">'
        f'{pred.regime.upper()}</div>'
        f'<div style="font-size:22px;font-weight:700;color:{meta["color"]}">'
        f'{pred.score:.1f}<span style="font-size:13px;color:#94a3b8"> / 100</span></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

with score_col:
    st.plotly_chart(
        score_bar_chart(pred.score, pred.regime),
        use_container_width=True,
        config={"displayModeBar": False},
    )
    st.markdown(
        f'<p style="font-size:13px;color:#94a3b8;margin-top:-10px;padding-left:4px">'
        f'💡 {meta["advice"]}</p>',
        unsafe_allow_html=True,
    )


# ═══ METRICS ══════════════════════════════════════════════════════════════════
st.markdown("<div style='margin-top:4px'></div>", unsafe_allow_html=True)
m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    st.metric("VIX (Fear Index)", f"{vix_now:.1f}", f"{vix_chg:+.1f}% (5d)", delta_color="inverse")
with m2:
    st.metric("21-day Return", f"{ret21:+.2f}%", f"5d: {ret5:+.2f}%")
with m3:
    st.metric("Realized Vol (ann.)", f"{vol:.1f}%")
with m4:
    st.metric("RSI (14)", f"{rsi:.1f}",
              "Overbought" if rsi > 70 else ("Oversold" if rsi < 30 else "Neutral"),
              delta_color="off")
with m5:
    st.metric("vs 200-day MA", f"{vs_ma:+.1f}%",
              "Above MA" if vs_ma > 0 else "Below MA",
              delta_color="normal" if vs_ma > 0 else "inverse")

st.divider()


# ═══ MAIN CHART ═══════════════════════════════════════════════════════════════
fig_overlay = regime_overlay_chart(history, lookback_days=lookback)
if not show_vix_overlay:
    fig_overlay.data = (fig_overlay.data[0],) + fig_overlay.data[2:]
st.plotly_chart(fig_overlay, use_container_width=True, config={"displayModeBar": False})


# ═══ TABS ═════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "Signal Breakdown",
    "Regime History",
    "Scenario Simulator",
    "Strategy Backtest",
])


# ── TAB 1: Signal Breakdown ───────────────────────────────────────────────────
with tab1:
    imp_col, sig_col = st.columns([1, 1])
    with imp_col:
        st.plotly_chart(feature_importance_chart(model, feature_cols),
                        use_container_width=True, config={"displayModeBar": False})
    with sig_col:
        st.markdown("#### Current Signal Readings")
        sig_df = pd.DataFrame({
            "Signal"   : ["5d Return","21d Return","Realized Vol","VIX Level",
                          "VIX 5d Chg","Drawdown","RSI","vs 200MA"],
            "Value"    : [f"{ret5:+.2f}%", f"{ret21:+.2f}%", f"{vol:.1f}%",
                          f"{vix_now:.1f}", f"{vix_chg:+.1f}%", f"{dd:.1f}%",
                          f"{rsi:.1f}", f"{vs_ma:+.1f}%"],
            "Bullish?" : ["✅" if ret5  > 0  else "❌",
                          "✅" if ret21 > 0  else "❌",
                          "✅" if vol < 15   else "❌",
                          "✅" if vix_now<18 else "❌",
                          "✅" if vix_chg<0  else "❌",
                          "✅" if dd > -5    else "❌",
                          "✅" if 45<rsi<70  else "❌",
                          "✅" if vs_ma > 0  else "❌"],
        })
        st.dataframe(sig_df, use_container_width=True, hide_index=True)

    if show_score_history:
        st.plotly_chart(score_history_chart(history, lookback_days=donut_lookback),
                        use_container_width=True, config={"displayModeBar": False})

    with st.expander("How does the model decide?", expanded=False):
        st.markdown(f"""
**Regression, not classification.** The model outputs a continuous score from 0 to 100:

| Zone | Score | Meaning |
|---|---|---|
| Bear | 0 – {BEAR_THRESHOLD} | High fear, negative momentum |
| Transitional | {BEAR_THRESHOLD} – {BULL_THRESHOLD} | Mixed, no clear direction |
| Bull | {BULL_THRESHOLD} – 100 | Low volatility, positive trend |

Training target was a **weighted composite** of 8 signals, capped so no single
feature moves the score more than ±22 pts. XGBRegressor (300 trees, depth 4) learns
the smooth mapping. Regime label applied **after** prediction via thresholds.
        """)
        st.json({f: float(round(latest_row[f], 6)) for f in feature_cols})


# ── TAB 2: Regime History ─────────────────────────────────────────────────────
with tab2:
    hist_col, donut_col = st.columns([2, 1])
    with hist_col:
        st.markdown("#### Last 30 Trading Days")
        cols_want = ["regime_label", "regime_score", "spy_close", "vix_level"]
        cols_have = [c for c in cols_want if c in history.columns]
        recent = history.tail(30)[cols_have].copy()
        recent.index = recent.index.strftime("%Y-%m-%d")
        recent.rename(columns={"regime_label":"Regime","regime_score":"Score",
                                "spy_close":"SPY Close","vix_level":"VIX"}, inplace=True)
        col_cfg = {
            "Regime"   : st.column_config.TextColumn("Regime", width="medium"),
            "SPY Close": st.column_config.NumberColumn("SPY Close", format="$%.2f"),
            "VIX"      : st.column_config.NumberColumn("VIX", format="%.1f"),
        }
        if "Score" in recent.columns:
            recent["Score"] = recent["Score"].round(1)
            col_cfg["Score"] = st.column_config.ProgressColumn(
                "Score (0–100)", min_value=0, max_value=100, format="%.1f"
            )
        st.data_editor(recent.iloc[::-1], use_container_width=True,
                       disabled=True, column_config=col_cfg)

    with donut_col:
        st.plotly_chart(regime_donut(history, lookback_days=donut_lookback),
                        use_container_width=True, config={"displayModeBar": False})
        streak_regime = history["regime_label"].iloc[-1]
        streak_count  = 1
        for lbl in reversed(history["regime_label"].values[:-1]):
            if lbl == streak_regime: streak_count += 1
            else: break
        sc = REGIME_META[streak_regime]["color"]
        sb = REGIME_META[streak_regime]["bg"]
        st.markdown(
            f'<div style="border:1px solid {sc};border-radius:10px;padding:14px;'
            f'text-align:center;background:{sb};margin-top:12px">'
            f'<div style="font-size:13px;color:#94a3b8">Current streak</div>'
            f'<div style="font-size:32px;font-weight:800;color:{sc}">{streak_count}</div>'
            f'<div style="font-size:13px;color:#e2e8f0">consecutive {streak_regime} days</div>'
            f'</div>', unsafe_allow_html=True,
        )


# ── TAB 3: Scenario Simulator ─────────────────────────────────────────────────
with tab3:
    st.markdown("#### Scenario Simulator")
    st.caption("Select a preset or adjust individual signals. Sliders update automatically.")

    pr1, pr2, pr3, pr4 = st.columns(4)
    PRESETS = {
        "bull" : {"sim_vix":12.,     "sim_ret21":8.,   "sim_rsi":65., "sim_vol":10., "sim_dd":-2.,  "sim_vs_ma":10.},
        "crash": {"sim_vix":45.,     "sim_ret21":-15., "sim_rsi":22., "sim_vol":55., "sim_dd":-30., "sim_vs_ma":-20.},
        "choppy":{"sim_vix":22.,     "sim_ret21":0.,   "sim_rsi":50., "sim_vol":18., "sim_dd":-8.,  "sim_vs_ma":1.},
        "live" : {"sim_vix":vix_now, "sim_ret21":ret21,"sim_rsi":rsi, "sim_vol":vol, "sim_dd":dd,   "sim_vs_ma":vs_ma},
    }
    def _apply_preset(name):
        for k, v in PRESETS[name].items():
            st.session_state[k] = v

    if pr1.button("Strong Bull",   use_container_width=True): _apply_preset("bull")
    if pr2.button("Market Crash",  use_container_width=True): _apply_preset("crash")
    if pr3.button("Choppy Market", use_container_width=True): _apply_preset("choppy")
    if pr4.button("Reset to Live", use_container_width=True): _apply_preset("live")

    preset_labels = {
        "bull" : ("Strong Bull",    "VIX 12 · +8% return · +10% above 200MA"),
        "crash": ("Market Crash",   "VIX 45 · -15% return · -30% drawdown"),
        "choppy":("Choppy Market",  "VIX 22 · flat returns · -8% drawdown"),
        "live" : ("Live values",    "All signals reflect current market"),
    }
    active_preset = None
    for pname, pvals in PRESETS.items():
        if all(abs(st.session_state.get(k, 0) - v) < 0.01 for k, v in pvals.items()):
            active_preset = pname
            break
    if active_preset:
        plabel, pdesc = preset_labels[active_preset]
        color = {"bull":"#22c55e","crash":"#ef4444","choppy":"#eab308","live":"#38bdf8"}[active_preset]
        st.markdown(
            f'<div style="background:#1e293b;border-left:3px solid {color};'
            f'border-radius:6px;padding:10px 16px;margin:8px 0 16px 0">'
            f'<b style="color:{color}">{plabel}</b>'
            f'<span style="color:#94a3b8;font-size:13px"> — {pdesc}</span></div>',
            unsafe_allow_html=True,
        )

    sim_c1, sim_c2 = st.columns(2)
    with sim_c1:
        st.slider("VIX Level",                  8.0,  60.0, key="sim_vix",   step=0.5)
        st.slider("21-day Return (%)",          -30.0, 30.0, key="sim_ret21", step=0.5)
        st.slider("RSI (14)",                    0.0, 100.0, key="sim_rsi",   step=1.0)
    with sim_c2:
        st.slider("Realized Vol — ann. (%)",     3.0,  80.0, key="sim_vol",   step=0.5)
        st.slider("Drawdown from 252d high (%)", -50.0, 0.0, key="sim_dd",    step=0.5)
        st.slider("Price vs 200-day MA (%)",    -30.0, 40.0, key="sim_vs_ma", step=0.5)

    overrides = {
        "vix_level"     : st.session_state["sim_vix"],
        "ret_21d"       : st.session_state["sim_ret21"] / 100,
        "rsi_14"        : st.session_state["sim_rsi"],
        "vol_21d"       : st.session_state["sim_vol"] / 100,
        "drawdown"      : st.session_state["sim_dd"]   / 100,
        "price_vs_200ma": st.session_state["sim_vs_ma"] / 100,
    }
    sim_pred = simulate_prediction(model, scaler, feature_cols, latest_row, overrides)
    sim_meta = REGIME_META[sim_pred.regime]

    st.markdown("---")
    gauge_col, bar_col2, explain_col = st.columns([1, 1, 1])
    with gauge_col:
        st.plotly_chart(simulate_gauge(sim_pred.score, sim_pred.regime),
                        use_container_width=True, config={"displayModeBar": False})
    with bar_col2:
        st.plotly_chart(score_bar_chart(sim_pred.score, sim_pred.regime),
                        use_container_width=True, config={"displayModeBar": False})
    with explain_col:
        st.markdown(f"#### {sim_meta['emoji']} {sim_pred.regime}")
        st.markdown(f"**Score: `{sim_pred.score:.1f} / 100`**")
        delta = sim_pred.score - pred.score
        sign  = "+" if delta >= 0 else ""
        dcol  = "#22c55e" if delta >= 0 else "#ef4444"
        st.markdown(
            f'<span style="color:{dcol};font-size:16px;font-weight:600">'
            f'{sign}{delta:.1f} pts vs live</span>', unsafe_allow_html=True,
        )
        st.markdown(f"*{sim_meta['desc']}*")
        with st.expander("Feature vector sent to model"):
            st.json({f: round(overrides.get(f, float(latest_row[f])), 6) for f in feature_cols})


# ── TAB 4: Strategy Backtest ──────────────────────────────────────────────────
with tab4:
    st.markdown("#### Strategy Backtest")
    st.caption(
        f"Buy SPY when score crosses **above {BULL_THRESHOLD}** (Bull). "
        f"Exit to cash when score drops **below {BEAR_THRESHOLD}** (Bear). "
        "Compared against buy-and-hold SPY over the same period."
    )

    # Date range selector
    min_date = history.index.min().date()
    max_date = history.index.max().date()
    d1, d2, _ = st.columns([1, 1, 2])
    with d1:
        bt_start = st.date_input("Start date", value=min_date,
                                  min_value=min_date, max_value=max_date)
    with d2:
        bt_end   = st.date_input("End date",   value=max_date,
                                  min_value=min_date, max_value=max_date)

    if bt_start >= bt_end:
        st.warning("End date must be after start date.")
        st.stop()

    result = run_backtest(history,
                          start_date=str(bt_start),
                          end_date=str(bt_end))

    # ── KPI cards ──────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5, k6 = st.columns(6)

    def _delta_color(val, inverse=False):
        if inverse:
            return "inverse"
        return "normal"

    better_ret = result.total_return - result.bh_return
    better_ann = result.annual_return - result.bh_annual_return
    better_sr  = result.sharpe        - result.bh_sharpe
    better_dd  = result.max_drawdown  - result.bh_max_drawdown  # less negative = better

    with k1:
        st.metric("Strategy Return",    f"{result.total_return:+.1f}%",
                  f"{better_ret:+.1f}% vs B&H",
                  delta_color="normal" if better_ret >= 0 else "inverse")
    with k2:
        st.metric("CAGR",               f"{result.annual_return:+.1f}%",
                  f"{better_ann:+.1f}% vs B&H",
                  delta_color="normal" if better_ann >= 0 else "inverse")
    with k3:
        st.metric("Sharpe Ratio",       f"{result.sharpe:.2f}",
                  f"{better_sr:+.2f} vs B&H",
                  delta_color="normal" if better_sr >= 0 else "inverse")
    with k4:
        st.metric("Max Drawdown",       f"{result.max_drawdown:.1f}%",
                  f"{better_dd:+.1f}% vs B&H",
                  delta_color="normal" if better_dd >= 0 else "inverse")
    with k5:
        st.metric("Time in Market",     f"{result.pct_invested:.0f}%",
                  help="% of trading days invested (less = more selective)")
    with k6:
        st.metric("Total Trades",       str(result.n_trades),
                  f"{result.years:.1f} year period")

    # ── Equity curve ───────────────────────────────────────────────────────
    st.plotly_chart(backtest_chart(result),
                    use_container_width=True, config={"displayModeBar": False})

    # ── Interpretation callout ─────────────────────────────────────────────
    outperforms = result.total_return > result.bh_return
    callout_col  = "#22c55e" if outperforms else "#eab308"
    callout_icon = "✅" if outperforms else "⚠️"
    callout_text = (
        f"Over this period the RegimeIQ strategy returned <b>{result.total_return:+.1f}%</b> "
        f"vs buy-and-hold <b>{result.bh_return:+.1f}%</b>, "
        f"spending only <b>{result.pct_invested:.0f}%</b> of days in the market."
        if outperforms else
        f"Over this period buy-and-hold outperformed by "
        f"<b>{abs(better_ret):.1f}%</b>. The strategy's value is in "
        f"risk reduction: max drawdown of <b>{result.max_drawdown:.1f}%</b> "
        f"vs <b>{result.bh_max_drawdown:.1f}%</b> for buy-and-hold."
    )
    st.markdown(
        f'<div style="background:#1e293b;border-left:3px solid {callout_col};'
        f'border-radius:6px;padding:12px 18px;margin-top:8px">'
        f'{callout_icon} {callout_text}</div>',
        unsafe_allow_html=True,
    )

    with st.expander("Methodology & limitations", expanded=False):
        st.markdown(f"""
**Entry:** Buy SPY when `regime_score` crosses above **{BULL_THRESHOLD}** (Bull threshold).

**Exit:** Sell to cash (0% return) when `regime_score` drops below **{BEAR_THRESHOLD}** (Bear threshold).

**Execution:** Signal uses previous day's score → trade at today's open (no lookahead bias).

**Assumptions:** No transaction costs, no slippage, fractional shares allowed, no taxes.

**Limitations:** This is a backtest on historical (or synthetic) data.
Past performance does not indicate future results. The model was trained on this same
data, so in-sample results are optimistic — treat as a proof of concept, not a trading strategy.
        """)


# ═══ FOOTER ═══════════════════════════════════════════════════════════════════
st.divider()
st.caption(
    "RegimeIQ is a university prototype — not financial advice. "
    "Data: Yahoo Finance (SPY, ^VIX) · Model: XGBoost Regressor · Built with Streamlit."
)