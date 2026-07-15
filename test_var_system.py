"""
test_var_system.py
-------------------
Unit tests for the core sub-systems, runnable offline via the
SyntheticProvider (NFR-5: calculation logic testable without the UI).

Run:  python -m pytest test_var_system.py -v
  or: python test_var_system.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src import (
    PortfolioInputFactory, PortfolioValidationError,
    PortfolioPnLBuilder, SyntheticProvider,
    VaREngine, HistoricalVaR, ParametricVaR, MonteCarloVaR,
    VaRBacktester, VaRConfig, TrafficLightZone,
    kupiec_pof_test, basel_traffic_light,
)


def _demo_market_data(config=None):
    loader = PortfolioInputFactory.get_source("csv")
    pf = loader.load("sample_data/sample_portfolio.csv", name="Test")
    builder = PortfolioPnLBuilder(SyntheticProvider(), config or VaRConfig())
    return builder.build(pf)


# ---------------- Sub-System 1 ----------------

def test_csv_load_and_validation():
    pf = PortfolioInputFactory.get_source("csv").load("sample_data/sample_portfolio.csv")
    assert len(pf) == 6
    assert "AAPL" in pf.tickers


def test_missing_column_rejected(tmp_path=None):
    import io
    bad = b"symbol,qty\nAAPL,10\n"
    try:
        PortfolioInputFactory.get_source("csv").load(bad)
        raise AssertionError("expected PortfolioValidationError")
    except PortfolioValidationError:
        pass


def test_prototype_clone_is_independent():
    pf = PortfolioInputFactory.get_source("csv").load("sample_data/sample_portfolio.csv")
    clone = pf.clone(new_name="clone")
    clone.positions[0].quantity *= 2
    assert clone.positions[0].quantity != pf.positions[0].quantity
    assert clone.name == "clone" and pf.name != "clone"


# ---------------- Sub-System 2 ----------------

def test_pnl_series_length():
    cfg = VaRConfig()
    md = _demo_market_data(cfg)
    # 510 price days -> 509 return/P&L days (pct_change drops the first row)
    assert len(md.pnl) >= cfg.lookback_days
    assert md.pnl.notna().all()


def test_parametric_matches_closed_form():
    cfg = VaRConfig()
    md = _demo_market_data(cfg)
    pnl = md.pnl.tail(cfg.lookback_days)
    expected = stats.norm.ppf(0.99) * pnl.std(ddof=1) - pnl.mean()
    got = ParametricVaR(cfg).calculate(md).var_1day
    assert abs(got - max(expected, 0)) < 1e-9


def test_historical_var_is_empirical_quantile():
    cfg = VaRConfig()
    md = _demo_market_data(cfg)
    pnl = md.pnl.tail(cfg.lookback_days)
    expected = -np.quantile(pnl.values, 0.01)
    got = HistoricalVaR(cfg).calculate(md).var_1day
    assert abs(got - max(expected, 0)) < 1e-9


def test_monte_carlo_close_to_parametric():
    # With a normal driver and many sims, MC should approximate Parametric.
    cfg = VaRConfig(mc_simulations=100_000, mc_random_seed=1)
    md = _demo_market_data(cfg)
    mc = MonteCarloVaR(cfg).calculate(md).var_1day
    pm = ParametricVaR(cfg).calculate(md).var_1day
    assert abs(mc - pm) / pm < 0.10   # within 10%


def test_ten_day_scaling():
    cfg = VaRConfig()
    md = _demo_market_data(cfg)
    for res in VaREngine(cfg).run_all(md).values():
        assert abs(res.var_10day - res.var_1day * np.sqrt(10)) < 1e-9


def test_var_positive_and_ordered_reasonably():
    md = _demo_market_data()
    results = VaREngine().run_all(md)
    for r in results.values():
        assert r.var_1day > 0
        assert r.var_10day > r.var_1day


# ---------------- Sub-System 4 ----------------

def test_kupiec_exact_expected_rate_not_rejected():
    lr, p = kupiec_pof_test(exceptions=3, observations=250, confidence=0.99)
    assert p > 0.05  # ~2.5 expected; 3 observed is consistent


def test_kupiec_gross_violation_rejected():
    lr, p = kupiec_pof_test(exceptions=25, observations=250, confidence=0.99)
    assert p < 0.01


def test_traffic_light_zones():
    assert basel_traffic_light(0) == TrafficLightZone.GREEN
    assert basel_traffic_light(4) == TrafficLightZone.GREEN
    assert basel_traffic_light(5) == TrafficLightZone.YELLOW
    assert basel_traffic_light(9) == TrafficLightZone.YELLOW
    assert basel_traffic_light(10) == TrafficLightZone.RED


def test_backtest_alignment_prior_day_var_vs_next_day_pnl():
    """A hand-built series where we know exactly which day breaches."""
    dates = pd.bdate_range("2025-01-01", periods=6)
    pnl = pd.Series([10, -5, -120, 8, -3, 4], index=dates, dtype=float)
    # constant VaR of 100 estimated at EOD of each of the first 5 days
    var = pd.Series(100.0, index=dates[:5])
    bt = VaRBacktester(VaRConfig())
    res = bt.backtest_series(pnl, var, "toy")
    # only the -120 on day 3 breaches the VaR estimated at EOD day 2
    assert res.exceptions == 1
    assert res.breach_dates == [dates[1]]  # indexed by estimation date


def test_full_pipeline_backtest_runs():
    cfg = VaRConfig()
    md = _demo_market_data(cfg)
    engine = VaREngine(cfg)
    rolling = engine.rolling_all(md, window=250)
    bt = VaRBacktester(cfg).backtest_all(md.pnl, rolling)
    summary = VaRBacktester.summary_frame(bt)
    assert len(summary) == 3
    assert set(summary["Method"]) == {"Historical", "Parametric", "Monte Carlo"}


if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} tests passed")
    sys.exit(1 if failed else 0)
