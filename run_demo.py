"""
run_demo.py
------------
End-to-end command-line demo of the four sub-systems, without the Streamlit UI.
Useful for smoke-testing the pipeline and for CI.

Usage:
    python run_demo.py                 # synthetic offline data
    python run_demo.py --live          # Yahoo Finance live data (needs internet)
"""
from __future__ import annotations

import argparse

from src import (
    PortfolioInputFactory,
    PortfolioPnLBuilder, SyntheticProvider, YahooFinanceProvider,
    VaREngine, VaRBacktester, VaRConfig,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="VaR Model Comparison System demo")
    parser.add_argument("--live", action="store_true",
                        help="use Yahoo Finance instead of synthetic data")
    parser.add_argument("--portfolio", default="sample_data/sample_portfolio.csv")
    args = parser.parse_args()

    # ---- Sub-System 1: portfolio input -------------------------------------
    loader = PortfolioInputFactory.get_source("csv")
    portfolio = loader.load(args.portfolio, name="Demo Portfolio")
    print(f"\nLoaded portfolio '{portfolio.name}' with {len(portfolio)} positions:")
    print(portfolio.to_dataframe().to_string(index=False))

    # Prototype pattern demo (FR-1.6): what-if clone with doubled first position
    what_if = portfolio.clone(new_name="What-if: double first position")
    what_if.positions[0].quantity *= 2

    # ---- Sub-System 2: market data + VaR engine ----------------------------
    config = VaRConfig()  # 99%, 1d & 10d, 510-day history — BRD defaults
    provider = YahooFinanceProvider() if args.live else SyntheticProvider()
    builder = PortfolioPnLBuilder(provider, config)

    md = builder.build(portfolio)
    print(f"\nP&L series: {len(md.pnl)} business days "
          f"({md.pnl.index.min().date()} -> {md.pnl.index.max().date()})")
    print(f"Gross notional: ${md.gross_notional:,.0f}")

    engine = VaREngine(config)
    results = engine.run_all(md)
    print("\n=== VaR Summary (99%, all methodologies) ===")
    print(engine.summary_frame(results).to_string(index=False))

    # ---- Sub-System 4: backtesting ------------------------------------------
    rolling = engine.rolling_all(md, window=250)
    backtester = VaRBacktester(config)
    bt = backtester.backtest_all(md.pnl, rolling)
    print("\n=== Backtesting (Kupiec POF + Basel traffic light) ===")
    print(VaRBacktester.summary_frame(bt).to_string(index=False))

    # ---- Prototype what-if run ----------------------------------------------
    md_wi = builder.build(what_if)
    results_wi = engine.run_all(md_wi)
    print(f"\n=== What-if clone: '{what_if.name}' ===")
    print(engine.summary_frame(results_wi).to_string(index=False))

    print("\nDone. Launch the dashboard with:  streamlit run dashboard.py")


if __name__ == "__main__":
    main()
