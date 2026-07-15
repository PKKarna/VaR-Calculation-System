"""
market_data.py
---------------
Sub-System 2 (data half): Market Data Sourcing & P&L Construction.

Responsibilities (BRD Section 6.2):
  FR-2.1  Retrieve daily historical prices per ticker from a public API
  FR-2.2  Retrieve >= 510 business days of history
  FR-2.3  Build daily portfolio P&L from per-ticker returns + position sizes
  FR-2.4  Handle missing price data consistently
  FR-2.5  Session-level caching to avoid redundant API calls

Design notes:
  - `MarketDataProvider` is an abstract Strategy so the pricing source can be
    swapped (Yahoo Finance today, a different vendor tomorrow) without
    touching the VaR engine.
  - `YahooFinanceProvider` uses `yfinance`, a free wrapper around Yahoo
    Finance's public (unauthenticated) pricing endpoints.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .portfolio_input import Portfolio
from .utils import VaRConfig, logger


class MarketDataError(Exception):
    pass


class MarketDataProvider(ABC):
    """Strategy interface for any public pricing data source (FR-2.1)."""

    @abstractmethod
    def get_price_history(self, tickers: List[str], business_days: int) -> pd.DataFrame:
        """Return a DataFrame of adjusted close prices, index=date, columns=tickers."""
        raise NotImplementedError


class YahooFinanceProvider(MarketDataProvider):
    """
    Public market-data provider backed by Yahoo Finance via `yfinance`.

    Requires internet access at runtime and the `yfinance` package
    (`pip install yfinance`). No API key is needed.
    """

    def __init__(self, request_buffer_days: int = 40):
        # extra calendar days requested beyond the business-day target, to
        # absorb weekends/holidays so we still end up with enough rows
        self.request_buffer_days = request_buffer_days

    def get_price_history(self, tickers: List[str], business_days: int) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError as e:
            raise MarketDataError(
                "yfinance is not installed. Run `pip install yfinance`."
            ) from e

        calendar_days = int(business_days * 7 / 5) + self.request_buffer_days
        start = (date.today() - timedelta(days=calendar_days)).isoformat()
        end = date.today().isoformat()

        logger.info("Fetching %d tickers from Yahoo Finance (%s -> %s)...",
                    len(tickers), start, end)
        raw = yf.download(
            tickers, start=start, end=end, auto_adjust=True,
            progress=False, group_by="ticker",
        )
        if raw is None or raw.empty:
            raise MarketDataError(f"No price data returned for tickers: {tickers}")

        if len(tickers) == 1:
            prices = raw[["Close"]].rename(columns={"Close": tickers[0]})
        else:
            prices = pd.DataFrame({t: raw[t]["Close"] for t in tickers if t in raw.columns.get_level_values(0)})

        prices = prices.sort_index()
        return prices.tail(business_days + self.request_buffer_days)


class SyntheticProvider(MarketDataProvider):
    """
    Deterministic synthetic price generator.

    Used for offline development, unit testing, and demoing the system when
    no internet connection / market-data API access is available -- keeps
    Sub-Systems 2-4 fully testable in isolation from the live API (NFR-5).
    """

    def __init__(self, annual_vol: float = 0.22, annual_drift: float = 0.06, seed: int = 7):
        self.annual_vol = annual_vol
        self.annual_drift = annual_drift
        self.seed = seed

    def get_price_history(self, tickers: List[str], business_days: int) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed)
        n = business_days + 10
        dates = pd.bdate_range(end=date.today(), periods=n)
        dt = 1 / 252
        data = {}
        for i, t in enumerate(tickers):
            local_rng = np.random.default_rng(self.seed + i)
            drift = self.annual_drift + local_rng.uniform(-0.03, 0.03)
            vol = self.annual_vol + local_rng.uniform(-0.05, 0.10)
            shocks = local_rng.standard_normal(n)
            log_returns = (drift - 0.5 * vol ** 2) * dt + vol * np.sqrt(dt) * shocks
            price_path = 100 * np.exp(np.cumsum(log_returns))
            data[t] = price_path
        return pd.DataFrame(data, index=dates)


class PortfolioPnLBuilder:
    """
    Builds the 510-business-day portfolio P&L time series (FR-2.2, FR-2.3)
    from a Portfolio and a MarketDataProvider, with simple, documented
    missing-data handling (FR-2.4).
    """

    def __init__(self, provider: MarketDataProvider, config: Optional[VaRConfig] = None):
        self.provider = provider
        self.config = config or VaRConfig()
        self._cache: Dict[tuple, pd.DataFrame] = {}   # FR-2.5 session cache

    def _fetch_prices(self, tickers: List[str]) -> pd.DataFrame:
        cache_key = (tuple(sorted(tickers)), self.config.total_history_days)
        if cache_key in self._cache:
            logger.info("Using cached market data for %s.", tickers)
            return self._cache[cache_key]

        prices = self.provider.get_price_history(tickers, self.config.total_history_days)

        # FR-2.4: missing data handling -- forward-fill gaps (documented),
        # then drop any leading rows still null, then trim to the requested
        # window. Forward-fill is used (rather than exclusion) so all
        # tickers stay aligned on a common date index for portfolio P&L.
        prices = prices.ffill().dropna(how="any")
        prices = prices.tail(self.config.total_history_days)

        if len(prices) < self.config.lookback_days:
            logger.warning(
                "Only %d business days of aligned price history available "
                "(requested %d). Results may be less statistically robust.",
                len(prices), self.config.total_history_days,
            )
        self._cache[cache_key] = prices
        return prices

    def build(self, portfolio: Portfolio) -> "PortfolioMarketData":
        """
        Returns a PortfolioMarketData bundle: prices, per-ticker returns,
        and the daily portfolio P&L series, ready for the VaR engine.
        """
        tickers = portfolio.tickers
        if not tickers:
            raise MarketDataError("Portfolio has no positions; cannot source market data.")

        prices = self._fetch_prices(tickers)
        returns = prices.pct_change().dropna(how="any")

        weights = portfolio.weights_by_ticker()
        last_prices = prices.iloc[-1]
        # dollar exposure per ticker at current market price (mark-to-market)
        dollar_exposure = pd.Series({t: weights[t] * last_prices[t] for t in tickers})

        # FR-2.3: daily portfolio P&L = sum over tickers of (exposure * return)
        pnl = (returns[tickers] * dollar_exposure[tickers]).sum(axis=1)
        pnl.name = "portfolio_pnl"

        return PortfolioMarketData(
            portfolio=portfolio, prices=prices, returns=returns,
            dollar_exposure=dollar_exposure, pnl=pnl,
        )


class PortfolioMarketData:
    """Simple data bundle passed from Sub-System 2 (data) into the VaR engine."""

    def __init__(self, portfolio: Portfolio, prices: pd.DataFrame, returns: pd.DataFrame,
                 dollar_exposure: pd.Series, pnl: pd.Series):
        self.portfolio = portfolio
        self.prices = prices
        self.returns = returns
        self.dollar_exposure = dollar_exposure
        self.pnl = pnl

    @property
    def gross_notional(self) -> float:
        return float(self.dollar_exposure.abs().sum())

    def covariance_matrix(self) -> pd.DataFrame:
        return self.returns[self.portfolio.tickers].cov()
