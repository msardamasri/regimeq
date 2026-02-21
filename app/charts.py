"""
charts.py — all Plotly figure builders for PulseCheck.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

REGIME_COLORS = {
    "Bull"        : "rgba(34,197,94,0.18)",
    "Transitional": "rgba(234,179,8,0.18)",
    "Bear"        : "rgba(239,68,68,0.18)",
}
REGIME_LINE = {
    "Bull"        : "#22c55e",
    "Transitional": "#eab308",
    "Bear"        : "#ef4444",
}
DARK_BG   = "#0f172a"
DARK_GRID = "#1e293b"
TEXT_COL  = "#e2e8f0"
ACCENT    = "#38bdf8"

BEAR_THRESHOLD = 35
BULL_THRESHOLD = 65


def _base_layout(title=""):
    return dict(
        title=dict(text=title, font=dict(size=14, color=TEXT_COL)),
        paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
        font=dict(color=TEXT_COL, family="Inter, sans-serif"),
        xaxis=dict(gridcolor=DARK_GRID, linecolor=DARK_GRID, showgrid=True),
        yaxis=dict(gridcolor=DARK_GRID, linecolor=DARK_GRID, showgrid=True),
        margin=dict(l=20, r=20, t=45, b=20),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=12)),
    )


# ── 1. THE SCORE BAR — main visual identity ───────────────────────────────────
def score_bar_chart(score: float, regime: str) -> go.Figure:
    """
    Horizontal gradient bar 0-100 showing the regime score as a pointer.
    Zones: Bear(0-35) Transitional(35-65) Bull(65-100).
    """
    fig = go.Figure()

    zone_defs = [
        (0,               BEAR_THRESHOLD,  "#7f1d1d", "#ef4444", "Bear"),
        (BEAR_THRESHOLD,  BULL_THRESHOLD,  "#422006", "#eab308", "Transitional"),
        (BULL_THRESHOLD,  100,             "#052e16", "#22c55e", "Bull"),
    ]

    # Draw zone bars
    for x0, x1, bg, line_col, label in zone_defs:
        fig.add_shape(type="rect",
                      x0=x0, x1=x1, y0=0.1, y1=0.9,
                      fillcolor=bg,
                      line=dict(color=line_col, width=1),
                      layer="below")
        fig.add_annotation(
            x=(x0 + x1) / 2, y=0.5,
            text=f"<b>{label}</b>",
            showarrow=False,
            font=dict(color=line_col, size=13),
            xanchor="center", yanchor="middle",
        )

    # Threshold lines
    for t in (BEAR_THRESHOLD, BULL_THRESHOLD):
        fig.add_shape(type="line",
                      x0=t, x1=t, y0=0.05, y1=0.95,
                      line=dict(color="#475569", width=1.5, dash="dot"))

    # Score pointer (triangle marker on a scatter)
    pointer_color = REGIME_LINE[regime]
    fig.add_trace(go.Scatter(
        x=[score], y=[0.5],
        mode="markers+text",
        marker=dict(
            symbol="triangle-down",
            size=22,
            color=pointer_color,
            line=dict(color="white", width=1.5),
        ),
        text=[f"<b>{score:.1f}</b>"],
        textposition="top center",
        textfont=dict(size=15, color=pointer_color),
        hovertemplate=f"Regime Score: <b>{score:.1f}</b><extra></extra>",
        showlegend=False,
    ))

    # Glowing line under pointer
    fig.add_shape(type="line",
                  x0=score, x1=score, y0=0.1, y1=0.9,
                  line=dict(color=pointer_color, width=3))

    fig.update_layout(
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_BG,
        height=130,
        margin=dict(l=10, r=10, t=35, b=10),
        title=dict(text="Market Regime Score", font=dict(size=13, color=TEXT_COL)),
        xaxis=dict(range=[0, 100], showgrid=False, showticklabels=True,
                   tickvals=[0, 25, 35, 50, 65, 75, 100],
                   ticktext=["0", "25", "<b>35</b>", "50", "<b>65</b>", "75", "100"],
                   tickfont=dict(color="#94a3b8", size=11),
                   linecolor=DARK_GRID),
        yaxis=dict(range=[0, 1], showgrid=False, showticklabels=False,
                   linecolor=DARK_GRID),
        showlegend=False,
    )
    return fig


# ── 2. Regime overlay chart ───────────────────────────────────────────────────
def regime_overlay_chart(history: pd.DataFrame, lookback_days: int = 500) -> go.Figure:
    h = history.tail(lookback_days).copy()
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.72, 0.28], vertical_spacing=0.04,
        subplot_titles=("SPY Price — Market Regime Overlay", "VIX (Fear Index)"),
    )

    prev_regime, band_start = None, None

    def _add_band(start, end, regime):
        for row in (1, 2):
            fig.add_vrect(x0=str(start), x1=str(end),
                          fillcolor=REGIME_COLORS[regime], line_width=0, row=row, col=1)

    for date, row in h.iterrows():
        r = row["regime_label"]
        if r != prev_regime:
            if prev_regime is not None:
                _add_band(band_start, date, prev_regime)
            band_start, prev_regime = date, r
    if prev_regime:
        _add_band(band_start, h.index[-1], prev_regime)

    fig.add_trace(go.Scatter(
        x=h.index, y=h["spy_close"], mode="lines", name="SPY",
        line=dict(color=ACCENT, width=2),
        hovertemplate="<b>%{x|%b %d %Y}</b><br>SPY: $%{y:.2f}<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=h.index, y=h["vix_level"], mode="lines", name="VIX",
        fill="tozeroy", line=dict(color="#f97316", width=1.5),
        fillcolor="rgba(249,115,22,0.15)",
        hovertemplate="<b>%{x|%b %d %Y}</b><br>VIX: %{y:.1f}<extra></extra>",
    ), row=2, col=1)

    for level, label, col in [(20, "VIX 20", "#eab308"), (30, "VIX 30", "#ef4444")]:
        fig.add_hline(y=level, line_dash="dash", line_color=col, line_width=1,
                      row=2, col=1, annotation_text=label,
                      annotation_font_color=col, annotation_position="right")

    for regime, color in REGIME_LINE.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=10, color=color, symbol="square"),
            name=regime, showlegend=True,
        ), row=1, col=1)

    layout = _base_layout()
    layout.update(
        height=520, hovermode="x unified", showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    layout["yaxis"]  = dict(gridcolor=DARK_GRID, linecolor=DARK_GRID,
                            showgrid=True, title="Price (USD)", color=TEXT_COL)
    layout["yaxis2"] = dict(gridcolor=DARK_GRID, linecolor=DARK_GRID,
                            showgrid=True, title="VIX", color=TEXT_COL)
    fig.update_layout(**layout)
    for ann in fig.layout.annotations:
        ann.font.color = TEXT_COL
        ann.font.size  = 12
    return fig


# ── 3. Score history line ─────────────────────────────────────────────────────
def score_history_chart(history: pd.DataFrame, lookback_days: int = 252) -> go.Figure:
    """Continuous regime score over time with threshold bands."""
    h = history.tail(lookback_days).copy()
    if "regime_score" not in h.columns:
        return go.Figure()

    fig = go.Figure()

    # Zone fills
    fig.add_hrect(y0=0,               y1=BEAR_THRESHOLD,  fillcolor="rgba(239,68,68,0.07)",  line_width=0)
    fig.add_hrect(y0=BEAR_THRESHOLD,  y1=BULL_THRESHOLD,  fillcolor="rgba(234,179,8,0.07)",  line_width=0)
    fig.add_hrect(y0=BULL_THRESHOLD,  y1=100,             fillcolor="rgba(34,197,94,0.07)",  line_width=0)

    # Threshold lines
    for t, col, label in [
        (BEAR_THRESHOLD, "#ef4444", f"Bear threshold ({BEAR_THRESHOLD})"),
        (BULL_THRESHOLD, "#22c55e", f"Bull threshold ({BULL_THRESHOLD})"),
    ]:
        fig.add_hline(y=t, line_dash="dash", line_color=col, line_width=1.2,
                      annotation_text=label, annotation_font_color=col,
                      annotation_position="right")

    # Score line — colour shifts with value
    fig.add_trace(go.Scatter(
        x=h.index, y=h["regime_score"].round(1),
        mode="lines", name="Regime Score",
        line=dict(color=ACCENT, width=2),
        fill="tozeroy",
        fillcolor="rgba(56,189,248,0.05)",
        hovertemplate="<b>%{x|%b %d %Y}</b><br>Score: %{y:.1f}<extra></extra>",
    ))

    layout = _base_layout("Regime Score History")
    layout.update(
        height=220,
        yaxis=dict(range=[0, 100], title="Score", gridcolor=DARK_GRID,
                   color=TEXT_COL, tickvals=[0, 35, 50, 65, 100]),
        hovermode="x unified",
        margin=dict(l=20, r=20, t=45, b=20),
    )
    fig.update_layout(**layout)
    return fig


# ── 4. Feature importance chart ───────────────────────────────────────────────
def feature_importance_chart(model, feature_cols: list) -> go.Figure:
    importances = model.get_booster().get_score(importance_type="gain")
    imp_vals = np.array([importances.get(f"f{i}", 0) for i in range(len(feature_cols))])
    imp_series = pd.Series(imp_vals, index=feature_cols).sort_values()
    pretty = {
        "ret_5d":"5-day Return","ret_21d":"21-day Return","vol_21d":"Realized Vol",
        "vix_level":"VIX Level","vix_change_5d":"VIX 5d Change","drawdown":"Max Drawdown",
        "rsi_14":"RSI (14)","price_vs_200ma":"Price vs 200MA",
    }
    labels = [pretty.get(f, f) for f in imp_series.index]
    norm   = (imp_series.values / imp_series.values.sum() * 100
              if imp_series.values.sum() > 0 else imp_series.values)
    fig = go.Figure(go.Bar(
        x=norm, y=labels, orientation="h",
        marker=dict(color=norm, colorscale=[[0,"#1e3a5f"],[0.5,ACCENT],[1,"#7dd3fc"]],
                    line=dict(width=0)),
        text=[f"{v:.1f}%" for v in norm],
        textposition="outside", textfont=dict(size=11, color=TEXT_COL),
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
    ))
    layout = _base_layout("Feature Importance (Gain)")
    layout.update(
        height=320,
        xaxis=dict(title="% Importance", showgrid=True, gridcolor=DARK_GRID, color=TEXT_COL),
        yaxis=dict(showgrid=False, color=TEXT_COL),
        margin=dict(l=20, r=60, t=45, b=20),
    )
    fig.update_layout(**layout)
    return fig


# ── 5. Regime donut ───────────────────────────────────────────────────────────
def regime_donut(history: pd.DataFrame, lookback_days: int = 252) -> go.Figure:
    counts = history.tail(lookback_days)["regime_label"].value_counts()
    labels = counts.index.tolist()
    fig = go.Figure(go.Pie(
        labels=labels, values=counts.values.tolist(), hole=0.62,
        marker=dict(colors=[REGIME_LINE.get(l, ACCENT) for l in labels],
                    line=dict(color=DARK_BG, width=3)),
        textfont=dict(size=13, color="white"),
        hovertemplate="%{label}: %{value} days (%{percent})<extra></extra>",
    ))
    layout = _base_layout(f"Regime Distribution — Last {lookback_days}d")
    layout["legend"] = dict(orientation="v", x=1.0, y=0.5)
    layout["margin"] = dict(l=20, r=20, t=45, b=20)
    fig.update_layout(
        **layout, height=280, showlegend=True,
        annotations=[dict(text=f"{lookback_days}d<br>lookback", x=0.5, y=0.5,
                          showarrow=False, font=dict(size=12, color=TEXT_COL))],
    )
    return fig


# ── 6. Simulate gauge ─────────────────────────────────────────────────────────
def simulate_gauge(score: float, regime: str) -> go.Figure:
    color = REGIME_LINE[regime]
    fig = go.Figure(go.Indicator(
        # gauge only — number added manually as annotation for precise positioning
        mode="gauge",
        value=round(score, 1),
        domain=dict(x=[0, 1], y=[0, 1]),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor=TEXT_COL,
                      tickfont=dict(color=TEXT_COL, size=10),
                      tickvals=[0, 35, 50, 65, 100]),
            bar=dict(color=color, thickness=0.25),
            bgcolor=DARK_GRID, borderwidth=0,
            steps=[
                dict(range=[0,  BEAR_THRESHOLD],            color="rgba(239,68,68,0.15)"),
                dict(range=[BEAR_THRESHOLD, BULL_THRESHOLD], color="rgba(234,179,8,0.15)"),
                dict(range=[BULL_THRESHOLD, 100],            color="rgba(34,197,94,0.15)"),
            ],
            threshold=dict(line=dict(color=color, width=3),
                           thickness=0.8, value=score),
        ),
    ))
    fig.update_layout(
        paper_bgcolor=DARK_BG,
        font=dict(color=TEXT_COL),
        height=240,
        margin=dict(l=30, r=30, t=40, b=10),
        # Score number centred inside the arc opening
        annotations=[
            dict(
                text=f'<b>{score:.1f}</b>',
                x=0.5, y=0.18,
                xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=42, color=color),
                xanchor="center", yanchor="middle",
            ),
            dict(
                text=regime.upper(),
                x=0.5, y=0.42,
                xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=13, color=TEXT_COL),
                xanchor="center", yanchor="middle",
            ),
        ],
    )
    return fig


# ── 7. Backtest equity curve ───────────────────────────────────────────────────
def backtest_chart(result) -> go.Figure:
    """
    Dual equity curve: RegimeIQ strategy vs Buy & Hold.
    Shaded area when strategy is in cash (out of market).
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.04,
        subplot_titles=("Portfolio Value (starting $100)", "Market Exposure"),
    )

    eq  = result.equity
    bh  = result.bh_equity
    pos = result.position

    # ── Cash zones (shaded) ───────────────────────────────────────────────
    in_cash = False
    cash_start = None
    for date, p in pos.items():
        if p == 0 and not in_cash:
            in_cash    = True
            cash_start = date
        elif p == 1 and in_cash:
            fig.add_vrect(
                x0=str(cash_start), x1=str(date),
                fillcolor="rgba(148,163,184,0.07)", line_width=0,
                row=1, col=1,
            )
            in_cash = False
    if in_cash:
        fig.add_vrect(
            x0=str(cash_start), x1=str(pos.index[-1]),
            fillcolor="rgba(148,163,184,0.07)", line_width=0,
            row=1, col=1,
        )

    # ── Equity lines ──────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=eq.index, y=eq.round(2),
        mode="lines", name="RegimeIQ Strategy",
        line=dict(color=ACCENT, width=2.5),
        hovertemplate="<b>%{x|%b %Y}</b><br>Strategy: $%{y:.2f}<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=bh.index, y=bh.round(2),
        mode="lines", name="Buy & Hold SPY",
        line=dict(color="#94a3b8", width=1.5, dash="dot"),
        hovertemplate="<b>%{x|%b %Y}</b><br>Buy & Hold: $%{y:.2f}<extra></extra>",
    ), row=1, col=1)

    # ── Position bar (0/1) ────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=pos.index, y=pos,
        mode="lines", name="In Market",
        line=dict(color=ACCENT, width=0),
        fill="tozeroy",
        fillcolor="rgba(56,189,248,0.2)",
        hovertemplate="<b>%{x|%b %Y}</b><br>Position: %{y}<extra></extra>",
        showlegend=False,
    ), row=2, col=1)

    layout = _base_layout()
    layout.update(
        height=480,
        hovermode="x unified",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    layout["yaxis"]  = dict(gridcolor=DARK_GRID, linecolor=DARK_GRID,
                            showgrid=True, title="Portfolio ($)", color=TEXT_COL,
                            tickprefix="$")
    layout["yaxis2"] = dict(gridcolor=DARK_GRID, linecolor=DARK_GRID,
                            showgrid=False, title="Exposure",
                            color=TEXT_COL, tickvals=[0, 1],
                            ticktext=["Cash", "Invested"])
    fig.update_layout(**layout)
    for ann in fig.layout.annotations:
        ann.font.color = TEXT_COL
        ann.font.size  = 12
    return fig