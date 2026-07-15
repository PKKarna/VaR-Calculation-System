# VaR Model Comparison System

Implements and backtests **Historical**, **Parametric (Variance-Covariance)**, and
**Monte Carlo** Value-at-Risk side-by-side on public market data, highlighting where
each approach diverges — answering the standing market-risk question:
*"which VaR methodology should we trust, and when?"*

Framed as a **decision-support comparison tool**, not a single "best" model: the
dashboard always shows all three methodologies together, with backtesting evidence
(Kupiec POF test + Basel traffic-light zones) so the user judges each model's live
performance instead of trusting a black box.

## Architecture — four sub-systems, one contract

```
                    ┌─────────────────────────────────────────────┐
                    │  Sub-System 3: Visualization (dashboard.py)  │
                    │  Streamlit + Plotly · no calculation logic   │
                    └───────▲──────────────▲──────────────▲───────┘
                            │              │              │
   ┌────────────────────┐   │   ┌──────────┴──────────┐   │   ┌───────────────────┐
   │ Sub-System 1        │  │   │ Sub-System 2         │   │   │ Sub-System 4       │
   │ Trade Portfolio     ├──┴──▶│ VaR Calculation      ├───┴──▶│ VaR Backtesting    │
   │ Input               │      │ Engine               │       │                    │
   │ portfolio_input.py  │      │ market_data.py       │       │ backtesting.py     │
   │ CSV / Excel / API   │      │ var_engine.py        │       │ Kupiec POF,        │
   │ → Portfolio object  │      │ 510-day P&L series → │       │ traffic light,     │
   │   (Prototype clone) │      │ Hist/Param/MC VaR    │       │ exception flags    │
   └────────────────────┘      └──────────────────────┘       └───────────────────┘
```

Data contracts connecting them: a normalized `Portfolio` object and a
`PortfolioMarketData` bundle (prices, returns, dollar exposures, daily P&L).

### Design patterns used

| Pattern | Where | Why |
|---|---|---|
| **Strategy** | `PortfolioSource` loaders; `VaRCalculator` (Historical / Parametric / MonteCarlo); `MarketDataProvider` | Interchangeable implementations behind one interface — add a 4th VaR method or a new data vendor without touching callers |
| **Prototype** | `Portfolio.clone()`, `VaRConfig.clone()` | Spin off what-if portfolios / config variants from a loaded baseline without re-parsing sources |
| **Factory** | `PortfolioInputFactory` | Resolve the right loader from a source-type string ("csv" / "excel" / "api") |
| **Facade** | `VaREngine` | One call runs all three methodologies over the same data and returns comparable results |

## Methodology (BRD defaults: 99% confidence, 1-day & 10-day)

- **Historical Simulation** — empirical 1% quantile of the trailing 500-day
  portfolio P&L distribution. No distributional assumption; steps when large
  losses enter/exit the window.
- **Parametric (Variance-Covariance)** — `VaR = z₀.₉₉·σ − μ` on the P&L window,
  with a full `√(wᵀCw)` covariance form exposed for SME validation. Smooth, fast,
  normality-bound.
- **Monte Carlo** — 20,000 correlated multivariate-normal one-day scenarios fitted
  to the trailing return window, portfolio revalued per scenario, empirical 1%
  quantile taken. Converges to Parametric under a normal driver, so MC-vs-Parametric
  gaps isolate simulation noise while Historical-vs-others isolates the normality
  assumption. The scenario generator is swappable (Student-t, bootstrap, ...).
- **10-day VaR** — square-root-of-time scaling applied uniformly across all three
  methods so cross-method comparisons are apples-to-apples (documented per
  FR-3.2/3.4/3.6).
- **Backtesting** — VaR estimated at EOD *t* is compared with realized P&L on
  *t+1*; exceptions counted per method; Kupiec (1995) POF likelihood-ratio test;
  Basel traffic-light zones over the trailing 250 days (Green 0–4, Yellow 5–9,
  Red 10+).

## Quick start

```bash
pip install -r requirements.txt

# 1. offline smoke test of the whole pipeline (synthetic data, no internet)
python run_demo.py

# 2. live data via Yahoo Finance
python run_demo.py --live

# 3. unit tests
python test_var_system.py            # or: python -m pytest test_var_system.py -v

# 4. dashboard
streamlit run dashboard.py
```

In the dashboard sidebar: pick confidence / lookback / MC simulation count, choose
**Yahoo Finance (live)** or **Synthetic (offline demo)** market data, and load a
portfolio via CSV, Excel, or a REST API URL. A sample portfolio is provided at
`sample_data/sample_portfolio.csv`.

### Portfolio file format

Required columns: `ticker`, `quantity`, `asset_class`.
Optional: `trade_price`, `currency`.

```csv
ticker,quantity,asset_class,trade_price,currency
AAPL,150,Equity,182.50,USD
EURUSD=X,50000,FX,1.0850,USD
GBPUSD=X,-30000,FX,1.2700,USD
```

Tickers must be resolvable by the pricing API (Yahoo symbols, e.g. `EURUSD=X` for FX).
Negative quantities are shorts. Duplicate tickers are aggregated.

## Project layout

```
app/
├── dashboard.py               # Sub-System 3: Streamlit + Plotly UI
├── run_demo.py                # CLI end-to-end demo (offline or live)
├── test_var_system.py         # 14 unit tests, offline-runnable
├── requirements.txt
├── sample_data/
│   └── sample_portfolio.csv
└── src/
    ├── utils.py               # VaRConfig (Prototype), enums, VaRResult
    ├── portfolio_input.py     # Sub-System 1
    ├── market_data.py         # Sub-System 2a: pricing + 510-day P&L
    ├── var_engine.py          # Sub-System 2b: 3 VaR strategies + facade
    └── backtesting.py         # Sub-System 4
```

## Known limitations (Phase 1, per BRD scope)

- Linear instruments only (equity/FX delta); no options Greeks / full revaluation.
- Single base currency; `EURUSD=X`-style quotes treated as return series on notional.
- √t scaling for 10-day VaR (no overlapping-window or multi-day path simulation yet —
  both are natural Phase 2 extensions behind the same `VaRCalculator` interface).
- Public API rate limits apply in live mode; the app degrades to a clear error and
  offers the synthetic mode.
