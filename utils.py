"""
utils.py
--------
Shared configuration objects, enums, and small helpers used across every
sub-system (portfolio input, market data, VaR engine, backtesting, dashboard).

Keeping these in one place is what lets the four sub-systems talk to each
other through a single, consistent contract instead of ad-hoc dicts.
"""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger("var_system")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)


class AssetClass(str, Enum):
    EQUITY = "Equity"
    FX = "FX"
    INDEX = "Index"
    OTHER = "Other"


class VaRMethod(str, Enum):
    HISTORICAL = "Historical"
    PARAMETRIC = "Parametric"
    MONTE_CARLO = "Monte Carlo"


class TrafficLightZone(str, Enum):
    GREEN = "Green"
    YELLOW = "Yellow"
    RED = "Red"


@dataclass
class VaRConfig:
    """
    Central, reusable configuration for a VaR run.

    Implements a light-weight Prototype pattern via `clone()`: callers can
    take a baseline config and cheaply spawn variants (e.g. a 95% confidence
    "what-if" run) without re-specifying every field.
    """
    confidence: float = 0.99
    horizon_days_short: int = 1
    horizon_days_long: int = 10
    lookback_days: int = 500          # trading-day lookback used for VaR calc
    total_history_days: int = 510     # total business days sourced (FR-2.2)
    backtest_window_days: int = 250   # Basel traffic-light window
    mc_simulations: int = 20_000
    mc_random_seed: Optional[int] = 42
    risk_free_rate: float = 0.0

    def clone(self, **overrides) -> "VaRConfig":
        """Prototype-style clone: deep-copy self, then apply overrides."""
        new_cfg = copy.deepcopy(self)
        for key, value in overrides.items():
            if not hasattr(new_cfg, key):
                raise AttributeError(f"VaRConfig has no field '{key}'")
            setattr(new_cfg, key, value)
        return new_cfg


@dataclass
class VaRResult:
    """Normalized output of any VaRCalculator implementation."""
    method: VaRMethod
    confidence: float
    var_1day: float
    var_10day: float
    as_of_date: object = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "method": self.method.value,
            "confidence": self.confidence,
            "var_1day": self.var_1day,
            "var_10day": self.var_10day,
            "as_of_date": self.as_of_date,
            **self.extra,
        }


def sqrt_time_scale(value_1day: float, horizon_days: int) -> float:
    """Square-root-of-time scaling rule (FR-3.4, FR-3.6)."""
    return value_1day * (horizon_days ** 0.5)
