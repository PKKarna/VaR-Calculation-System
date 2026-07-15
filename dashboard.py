"""
dashboard.py
-------------
Sub-System 3: Visualization Dashboard (Streamlit + Plotly).

Run with:
    streamlit run dashboard.py

Responsibilities (BRD Section 6.5):
  FR-5.1  Streamlit as primary UI
  FR-5.2  Portfolio upload / source panel (drives Sub-System 1)
  FR-5.3  Three method-specific panels: P&L series vs that method's 1-day VaR
  FR-5.4  Backtesting exceptions flagged visually on each panel
  FR-5.5  Consolidated 1-day / 10-day VaR comparison table
  FR-5.6  Backtesting results (Kupiec, traffic light) per methodology
  FR-5.7  Plotly charts throughout (zoom / hover / legend toggling)

This layer contains NO calculation logic (NFR-5): it only orchestrates the
other three sub-systems and renders their outputs.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import (
    PortfolioInputFactory, PortfolioValidationError,
    PortfolioPnLBuilder, YahooFinanceProvider, SyntheticProvider, MarketDataError,
    VaREngine, VaRBacktester, VaRConfig, TrafficLightZone,
)

st.set_page_config(page_title="VaR Model Comparison System", layout="wide")

METHOD_COLORS = {
    "Historical": "#1f77b4",
    "Parametric": "#2ca02c",
    "Monte Carlo": "#d62728",
}
ZONE_COLORS = {
    TrafficLightZone.GREEN.value: "#2ca02c",
    TrafficLightZone.YELLOW.value: "#e6b800",
    TrafficLightZone.RED.value: "#d62728",
}


# ----------------------------------------------------------------------------
# Sidebar — configuration + portfolio sourcing (FR-5.2 / Sub-System 1)
# ----------------------------------------------------------------------------
st.sidebar.title("Configuration")

confidence = st.sidebar.selectbox("Confidence level", [0.99, 0.975, 0.95], index=0,
                                  format_func=lambda c: f"{c:.1%}")
lookback = st.sidebar.slider("VaR lookback (business days)", 100, 500, 250, step=50)
mc_sims = st.sidebar.select_slider("Monte Carlo simulations",
                                   options=[5_000, 10_000, 20_000, 50_000], value=20_000)
data_source = st.sidebar.radio(
    "Market data source",
    ["Yahoo Finance (live, needs internet)", "Synthetic (offline demo)"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.title("Trade Portfolio Input")
input_mode = st.sidebar.radio("Source", ["Upload CSV", "Upload Excel", "REST API"])

portfolio = None
try:
    if input_mode == "Upload CSV":
        f = st.sidebar.file_uploader("Portfolio CSV", type=["csv"])
        if f is not None:
            portfolio = PortfolioInputFactory.get_source("csv").load(f.getvalue(), name=f.name)
    elif input_mode == "Upload Excel":
        f = st.sidebar.file_uploader("Portfolio Excel", type=["xlsx"])
        if f is not None:
            portfolio = PortfolioInputFactory.get_source("excel").load(f.getvalue(), name=f.name)
    else:
        url = st.sidebar.text_input("Portfolio API URL",
                                    placeholder="https://your-oms/api/portfolio")
        if url and st.sidebar.button("Fetch portfolio"):
            portfolio = PortfolioInputFactory.get_source("api").load(url)
except PortfolioValidationError as e:
    st.sidebar.error(f"Portfolio validation failed: {e}")
except Exception as e:  # noqa: BLE001 — surface any sourcing error to the user
    st.sidebar.error(f"Could not load portfolio: {e}")

st.sidebar.caption(
    "Required columns: ticker, quantity, asset_class. "
    "Optional: trade_price, currency."
)

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.title("VaR Model Comparison System")
st.caption(
    "Historical vs Parametric vs Monte Carlo Value-at-Risk — computed side-by-side "
    "on the same portfolio and market data, then backtested against realized P&L. "
    "A decision-support comparison, not a single 'best' number."
)

if portfolio is None:
    st.info(
        "Load a trade portfolio from the sidebar to begin. "
        "A sample CSV (`sample_data/sample_portfolio.csv`) is included in the project."
    )
    st.stop()

st.subheader(f"Portfolio: {portfolio.name}")
st.dataframe(portfolio.to_dataframe(), use_container_width=True, hide_index=True)

# ----------------------------------------------------------------------------
# Pipeline: market data -> VaR engine -> backtesting  (cached per inputs)
# ----------------------------------------------------------------------------
config = VaRConfig(confidence=confidence, lookback_days=lookback, mc_simulations=mc_sims)


@st.cache_data(show_spinner="Sourcing market data and running all three VaR engines...")
def run_pipeline(portfolio_df_json: str, portfolio_name: str, conf: float,
                 lb: int, sims: int, source_label: str):
    """Cached end-to-end run keyed on portfolio + parameters (FR-2.5)."""
    from src import Portfolio, Position  # local import inside cached fn

    df = pd.read_json(portfolio_df_json)
    positions = [
        Position(ticker=r.ticker, quantity=float(r.quantity),
                 asset_class=r.asset_class,
                 trade_price=None if pd.isnull(r.trade_price) else float(r.trade_price),
                 currency=r.currency)
        for r in df.itertuples(index=False)
    ]
    pf = Portfolio(name=portfolio_name, positions=positions)

    cfg = VaRConfig(confidence=conf, lookback_days=lb, mc_simulations=sims)
    provider = SyntheticProvider() if source_label.startswith("Synthetic") else YahooFinanceProvider()
    builder = PortfolioPnLBuilder(provider, cfg)
    md = builder.build(pf)

    engine = VaREngine(cfg)
    results = engine.run_all(md)
    summary = engine.summary_frame(results)
    rolling = engine.rolling_all(md, window=min(lb, 250))

    backtester = VaRBacktester(cfg)
    bt_results = backtester.backtest_all(md.pnl, rolling)
    bt_summary = VaRBacktester.summary_frame(bt_results)

    breach_frames = {
        m: r.breach_series.rename("breach").to_frame() for m, r in bt_results.items()
    }
    zones = {m: r.traffic_light.value for m, r in bt_results.items()}

    return (md.pnl, md.gross_notional, summary, rolling, bt_summary, breach_frames, zones)


try:
    pnl, gross_notional, summary, rolling, bt_summary, breach_frames, zones = run_pipeline(
        portfolio.to_dataframe().to_json(), portfolio.name,
        confidence, lookback, mc_sims, data_source,
    )
except MarketDataError as e:
    st.error(
        f"Market data sourcing failed: {e}\n\n"
        "If you are offline or the public pricing API is rate-limiting, "
        "switch the sidebar to **Synthetic (offline demo)** mode."
    )
    st.stop()

# ----------------------------------------------------------------------------
# Consolidated comparison table (FR-5.5)
# ----------------------------------------------------------------------------
st.markdown("## VaR Summary — all three methodologies")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Gross notional", f"${gross_notional:,.0f}")
for col, (_, row) in zip((c2, c3, c4), summary.iterrows()):
    col.metric(
        f"{row['method']} 1d VaR ({confidence:.0%})",
        f"${row['var_1day']:,.0f}",
        delta=f"10d: ${row['var_10day']:,.0f}",
        delta_color="off",
    )

display_summary = summary.copy()
display_summary["confidence"] = display_summary["confidence"].map(lambda c: f"{c:.1%}")
display_summary["var_1day"] = display_summary["var_1day"].map(lambda v: f"${v:,.0f}")
display_summary["var_10day"] = display_summary["var_10day"].map(lambda v: f"${v:,.0f}")
display_summary.columns = ["Method", "Confidence", "1-day VaR", "10-day VaR", "As of"]
st.dataframe(display_summary, use_container_width=True, hide_index=True)
st.caption(
    "10-day VaR is derived via the square-root-of-time rule uniformly across methods, "
    "so cross-method differences reflect the methodologies themselves, not scaling choices."
)

# ----------------------------------------------------------------------------
# Three method panels: P&L vs rolling VaR with breach markers (FR-5.3 / FR-5.4)
# ----------------------------------------------------------------------------
st.markdown("## Method panels — P&L vs 1-day VaR, breaches flagged")

tabs = st.tabs(list(METHOD_COLORS.keys()))
for tab, method in zip(tabs, METHOD_COLORS.keys()):
    with tab:
        color = METHOD_COLORS[method]
        var_series = rolling[method].dropna()
        pnl_aligned = pnl.reindex(var_series.index.union(pnl.index)).dropna()
        pnl_view = pnl_aligned.loc[pnl_aligned.index >= var_series.index.min()]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=pnl_view.index, y=pnl_view.values, name="Daily P&L",
            marker_color=["#9ecae1" if v >= 0 else "#fdae6b" for v in pnl_view.values],
            opacity=0.85,
        ))
        fig.add_trace(go.Scatter(
            x=var_series.index, y=-var_series.values, mode="lines",
            name=f"-1d VaR ({method})", line=dict(color=color, width=2),
        ))

        # breach markers (FR-5.4): breach on day t+1 vs VaR estimated at t
        breaches = breach_frames[method]["breach"]
        breach_est_dates = breaches[breaches].index
        if len(breach_est_dates) > 0:
            next_day = {d: pnl.index[pnl.index.get_loc(d) + 1]
                        for d in breach_est_dates
                        if pnl.index.get_loc(d) + 1 < len(pnl.index)}
            bx = list(next_day.values())
            by = pnl.loc[bx].values
            fig.add_trace(go.Scatter(
                x=bx, y=by, mode="markers", name="VaR exception",
                marker=dict(symbol="x", size=10, color="black"),
            ))

        fig.update_layout(
            height=430, hovermode="x unified",
            legend=dict(orientation="h", y=1.08),
            yaxis_title="P&L ($)", margin=dict(t=30, b=10),
            barmode="overlay",
        )
        st.plotly_chart(fig, use_container_width=True)

        zone = zones[method]
        st.markdown(
            f"Backtest zone: "
            f"<span style='color:{ZONE_COLORS[zone]}; font-weight:700'>{zone}</span>",
            unsafe_allow_html=True,
        )

# ----------------------------------------------------------------------------
# Cross-method overlay + divergence view
# ----------------------------------------------------------------------------
st.markdown("## Where the methodologies diverge")
fig_all = go.Figure()
for method, color in METHOD_COLORS.items():
    s = rolling[method].dropna()
    fig_all.add_trace(go.Scatter(
        x=s.index, y=s.values, mode="lines", name=method,
        line=dict(color=color, width=2),
    ))
fig_all.update_layout(
    height=380, hovermode="x unified", yaxis_title="1-day VaR ($)",
    legend=dict(orientation="h", y=1.1), margin=dict(t=30, b=10),
)
st.plotly_chart(fig_all, use_container_width=True)
st.caption(
    "Historical VaR steps when large losses enter/exit the lookback window; "
    "Parametric is smooth but normality-bound; Monte Carlo tracks Parametric "
    "(same normal driver) plus simulation noise. Gaps between the lines are "
    "exactly the model-choice risk this tool is built to surface."
)

# ----------------------------------------------------------------------------
# Backtesting summary (FR-5.6 / Sub-System 4)
# ----------------------------------------------------------------------------
st.markdown("## Backtesting — Kupiec POF & Basel traffic light")


def _zone_style(val):
    color = ZONE_COLORS.get(val)
    return f"background-color: {color}; color: white; font-weight: 700" if color else ""


st.dataframe(
    bt_summary.style.applymap(_zone_style, subset=["Traffic light"]),
    use_container_width=True, hide_index=True,
)
st.caption(
    "Exceptions: days where realized loss exceeded the prior EOD VaR estimate. "
    "Kupiec POF tests whether the exception count is statistically consistent "
    "with the chosen confidence level (reject = model mis-calibrated). "
    "Traffic light applies the Basel 250-day zones: Green 0–4, Yellow 5–9, Red 10+."
)
