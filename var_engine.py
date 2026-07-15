"""
var_engine.py
--------------
Sub-System 2 (calculation half): VaR Calculation Engine.

Responsibilities (BRD Section 6.3):
  FR-3.1 / FR-3.2  Historical Simulation VaR, 99% 1-day and 10-day
  FR-3.3 / FR-3.4  Parametric (Variance-Covariance) VaR, 99% 1-day and 10-day
  FR-3.5 / FR-3.6  Monte Carlo VaR, 99% 1-day and 10-day
  FR-3.7           Common calculation interface across all three methods
  FR-3.8           Confidence / horizon / lookback configurable via VaRConfig

Design patterns:
  - Strategy: `VaRCalculator` is the abstract interface; Historical,
    Parametric and MonteCarlo are interchangeable concrete strategies.
  - The engine also produces a *rolling* daily VaR series per method, which
    is what Sub-System 4 (backtesting) consumes.

Conventions:
  - VaR is reported as a POSITIVE dollar amount (loss magnitude).
  - 10-day VaR uses the square-root-of-time rule uniformly across methods
    (documented per FR-3.2 / FR-3.4 / FR-3.6) so the cross-method
    comparison is apples-to-apples.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats

from .market_data import PortfolioMarketData
from .utils import VaRConfig, VaRMethod, VaRResult, logger, sqrt_time_scale


class VaRCalculator(ABC):
    """
    Strategy interface (FR-3.7): every methodology exposes the same
    two entry points so results are directly comparable.
    """

    method: VaRMethod

    def __init__(self, config: Optional[VaRConfig] = None):
        self.config = config or VaRConfig()

    # ---------------- point-in-time VaR ----------------
    @abstractmethod
    def _var_1day_from_pnl(self, pnl_window: pd.Series, market_data: PortfolioMarketData) -> float:
        """Return 1-day VaR (positive $) computed on the given P&L window."""
        raise NotImplementedError

    def calculate(self, market_data: PortfolioMarketData) -> VaRResult:
        """Current (as-of latest date) 1-day and 10-day VaR."""
        pnl = market_data.pnl.tail(self.config.lookback_days)
        var_1d = self._var_1day_from_pnl(pnl, market_data)
        var_10d = sqrt_time_scale(var_1d, self.config.horizon_days_long)
        result = VaRResult(
            method=self.method,
            confidence=self.config.confidence,
            var_1day=var_1d,
            var_10day=var_10d,
            as_of_date=pnl.index[-1],
        )
        logger.info("%s VaR @%.0f%%: 1d=%.2f, 10d=%.2f",
                    self.method.value, self.config.confidence * 100, var_1d, var_10d)
        return result

    # ---------------- rolling VaR series (for backtesting) ----------------
    def rolling_var_series(self, market_data: PortfolioMarketData,
                           window: Optional[int] = None) -> pd.Series:
        """
        For each business day t, compute the 1-day VaR using only data up to
        and including t (a trailing window). The value at date t is the VaR
        estimate made at EOD t, to be compared against P&L on t+1 by the
        backtesting sub-system.
        """
        window = window or min(self.config.lookback_days, 250)
        pnl = market_data.pnl
        out = {}
        for i in range(window, len(pnl)):
            pnl_window = pnl.iloc[i - window:i]
            out[pnl.index[i - 1]] = self._var_1day_from_pnl(pnl_window, market_data)
        series = pd.Series(out, name=f"var_{self.method.value}")
        series.index.name = "date"
        return series


class HistoricalVaR(VaRCalculator):
    """
    FR-3.1 / FR-3.2 — Historical Simulation.

    Empirical quantile of the realized P&L distribution: no distributional
    assumption; captures fat tails present in the sample but is blind to
    losses larger than anything in the lookback window.
    """
    method = VaRMethod.HISTORICAL

    def _var_1day_from_pnl(self, pnl_window: pd.Series, market_data: PortfolioMarketData) -> float:
        q = np.quantile(pnl_window.values, 1 - self.config.confidence)
        return float(max(-q, 0.0))


class ParametricVaR(VaRCalculator):
    """
    FR-3.3 / FR-3.4 — Parametric (Variance-Covariance / analytical).

    Assumes portfolio P&L is normally distributed. VaR is derived from the
    portfolio-level standard deviation of the P&L window and the normal
    z-score at the chosen confidence level:

        VaR = z * sigma - mu     (floored at 0)

    Fast and smooth, but understates tail risk when returns are fat-tailed.
    """
    method = VaRMethod.PARAMETRIC

    def _var_1day_from_pnl(self, pnl_window: pd.Series, market_data: PortfolioMarketData) -> float:
        mu = float(pnl_window.mean())
        sigma = float(pnl_window.std(ddof=1))
        z = stats.norm.ppf(self.config.confidence)
        return float(max(z * sigma - mu, 0.0))

    def covariance_var(self, market_data: PortfolioMarketData) -> float:
        """
        Full variance-covariance form (equivalent view): VaR from the
        position-weighted covariance matrix of asset returns,
        sigma_p = sqrt(w' C w) with w = dollar exposures. Provided for
        transparency / SME validation alongside the P&L-window form.
        """
        w = market_data.dollar_exposure.values
        C = market_data.covariance_matrix().values
        sigma_p = float(np.sqrt(w @ C @ w))
        z = stats.norm.ppf(self.config.confidence)
        return z * sigma_p


class MonteCarloVaR(VaRCalculator):
    """
    FR-3.5 / FR-3.6 — Monte Carlo simulation.

    Simulates N correlated one-day asset-return scenarios from a multivariate
    normal fitted to the trailing return window (mean vector + covariance
    matrix), revalues the portfolio under each scenario, and takes the
    empirical quantile of the simulated P&L distribution.

    With a normal driver it converges to Parametric VaR as N grows — the
    divergence between MC and Parametric on the dashboard therefore isolates
    pure simulation noise, while Historical vs the other two isolates the
    effect of the normality assumption. The scenario generator can be swapped
    (e.g. Student-t, bootstrapped) without touching the interface.
    """
    method = VaRMethod.MONTE_CARLO

    def _var_1day_from_pnl(self, pnl_window: pd.Series, market_data: PortfolioMarketData) -> float:
        cfg = self.config
        rng = np.random.default_rng(cfg.mc_random_seed)

        # align asset returns to the same window as the P&L series
        returns = market_data.returns.loc[
            market_data.returns.index.intersection(pnl_window.index)
        ][market_data.portfolio.tickers]

        mu = returns.mean().values
        cov = returns.cov().values
        w = market_data.dollar_exposure[market_data.portfolio.tickers].values

        # guard: single-asset portfolios or degenerate covariance
        cov = np.atleast_2d(cov)
        try:
            sims = rng.multivariate_normal(mu, cov, size=cfg.mc_simulations,
                                           method="cholesky")
        except np.linalg.LinAlgError:
            # fall back to eigenvalue decomposition for near-singular cov
            sims = rng.multivariate_normal(mu, cov, size=cfg.mc_simulations,
                                           method="eigh")

        simulated_pnl = sims @ w
        q = np.quantile(simulated_pnl, 1 - cfg.confidence)
        return float(max(-q, 0.0))


class VaREngine:
    """
    Facade that runs all three strategies over the same market data and
    returns comparable results + rolling series — the single object the
    dashboard (Sub-System 3) and backtester (Sub-System 4) consume.
    """

    def __init__(self, config: Optional[VaRConfig] = None):
        self.config = config or VaRConfig()
        self.calculators = [
            HistoricalVaR(self.config),
            ParametricVaR(self.config),
            MonteCarloVaR(self.config),
        ]

    def run_all(self, market_data: PortfolioMarketData) -> Dict[VaRMethod, VaRResult]:
        return {c.method: c.calculate(market_data) for c in self.calculators}

    def rolling_all(self, market_data: PortfolioMarketData,
                    window: Optional[int] = None) -> pd.DataFrame:
        frames = {
            c.method.value: c.rolling_var_series(market_data, window)
            for c in self.calculators
        }
        return pd.DataFrame(frames)

    def summary_frame(self, results: Dict[VaRMethod, VaRResult]) -> pd.DataFrame:
        rows = [r.to_dict() for r in results.values()]
        df = pd.DataFrame(rows)
        return df[["method", "confidence", "var_1day", "var_10day", "as_of_date"]]
