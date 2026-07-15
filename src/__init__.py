"""
VaR Model Comparison System — package exports.

Four sub-systems wired through common contracts:
  1. Trade Portfolio Input  -> portfolio_input.py
  2. VaR Calculation Engine -> market_data.py + var_engine.py
  3. Visualization Dashboard-> ../dashboard.py (Streamlit entry point)
  4. VaR Backtesting        -> backtesting.py
"""
from .utils import VaRConfig, VaRMethod, VaRResult, TrafficLightZone, AssetClass
from .portfolio_input import (
    Portfolio, Position, PortfolioInputFactory,
    CSVPortfolioSource, ExcelPortfolioSource, APIPortfolioSource,
    PortfolioValidationError,
)
from .market_data import (
    MarketDataProvider, YahooFinanceProvider, SyntheticProvider,
    PortfolioPnLBuilder, PortfolioMarketData, MarketDataError,
)
from .var_engine import (
    VaRCalculator, HistoricalVaR, ParametricVaR, MonteCarloVaR, VaREngine,
)
from .backtesting import VaRBacktester, BacktestResult, kupiec_pof_test, basel_traffic_light

__all__ = [
    "VaRConfig", "VaRMethod", "VaRResult", "TrafficLightZone", "AssetClass",
    "Portfolio", "Position", "PortfolioInputFactory",
    "CSVPortfolioSource", "ExcelPortfolioSource", "APIPortfolioSource",
    "PortfolioValidationError",
    "MarketDataProvider", "YahooFinanceProvider", "SyntheticProvider",
    "PortfolioPnLBuilder", "PortfolioMarketData", "MarketDataError",
    "VaRCalculator", "HistoricalVaR", "ParametricVaR", "MonteCarloVaR", "VaREngine",
    "VaRBacktester", "BacktestResult", "kupiec_pof_test", "basel_traffic_light",
]
