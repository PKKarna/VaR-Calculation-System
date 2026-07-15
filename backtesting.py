"""
backtesting.py
---------------
Sub-System 4: VaR Backtesting.

Responsibilities (BRD Section 6.4):
  FR-4.1  Compare each prior-EOD 1-day VaR estimate against next-day realized P&L
  FR-4.2  Count / flag exceptions per methodology
  FR-4.3  Kupiec Proportion-of-Failures (POF) likelihood-ratio test
  FR-4.4  Basel-style traffic-light zone (Green / Yellow / Red) over 250 days
  FR-4.5  Comparative summary across all three methodologies

Backtest convention (FR-4.1):
  The rolling VaR series produced by the engine is indexed so that the value
  dated t is the VaR estimated at EOD t. It is compared against realized P&L
  on t+1. An "exception" is a day where realized loss exceeds that estimate:
      pnl[t+1] < -VaR[t]
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats

from .utils import TrafficLightZone, VaRConfig, VaRMethod, logger


@dataclass
class BacktestResult:
    method: str
    observations: int
    exceptions: int
    expected_exceptions: float
    exception_rate: float
    kupiec_lr: float
    kupiec_p_value: float
    kupiec_reject_95: bool
    traffic_light: TrafficLightZone
    exceptions_250d: int
    breach_dates: list = field(default_factory=list)
    breach_series: Optional[pd.Series] = None   # bool per date, for charting

    def to_row(self) -> dict:
        return {
            "Method": self.method,
            "Observations": self.observations,
            "Exceptions": self.exceptions,
            "Expected": round(self.expected_exceptions, 1),
            "Exception rate": f"{self.exception_rate:.2%}",
            "Kupiec LR": round(self.kupiec_lr, 3),
            "Kupiec p-value": round(self.kupiec_p_value, 4),
            "Reject @95%?": "Yes" if self.kupiec_reject_95 else "No",
            "Exceptions (250d)": self.exceptions_250d,
            "Traffic light": self.traffic_light.value,
        }


def kupiec_pof_test(exceptions: int, observations: int, confidence: float) -> tuple:
    """
    Kupiec (1995) Proportion-of-Failures likelihood-ratio test (FR-4.3).

    H0: the observed exception rate equals the model's expected rate p = 1 - confidence.
    LR_pof ~ chi-squared with 1 degree of freedom under H0.

    Returns (LR statistic, p-value).
    """
    p = 1.0 - confidence
    n, x = observations, exceptions
    if n == 0:
        return np.nan, np.nan

    phat = x / n
    # log-likelihood under H0 and under the observed rate; handle x=0 / x=n edges
    def _loglik(prob):
        prob = min(max(prob, 1e-12), 1 - 1e-12)
        return (n - x) * np.log(1 - prob) + x * np.log(prob)

    lr = -2.0 * (_loglik(p) - _loglik(phat))
    lr = max(lr, 0.0)
    p_value = 1.0 - stats.chi2.cdf(lr, df=1)
    return float(lr), float(p_value)


def basel_traffic_light(exceptions_250d: int) -> TrafficLightZone:
    """
    Basel Committee traffic-light zones for a 99% 1-day VaR model
    backtested over 250 business days (FR-4.4):
        Green : 0-4 exceptions
        Yellow: 5-9 exceptions
        Red   : 10+ exceptions
    """
    if exceptions_250d <= 4:
        return TrafficLightZone.GREEN
    if exceptions_250d <= 9:
        return TrafficLightZone.YELLOW
    return TrafficLightZone.RED


class VaRBacktester:
    """
    Runs the full backtest for one or more rolling VaR series against a
    realized P&L series, producing per-method BacktestResult objects and a
    comparative summary DataFrame (FR-4.5).
    """

    def __init__(self, config: Optional[VaRConfig] = None):
        self.config = config or VaRConfig()

    def backtest_series(self, pnl: pd.Series, rolling_var: pd.Series,
                        method_name: str) -> BacktestResult:
        # VaR at date t is compared with P&L at the NEXT business day (FR-4.1)
        var_shifted = rolling_var.copy()
        pnl_next = pnl.shift(-1).reindex(var_shifted.index).dropna()
        var_aligned = var_shifted.reindex(pnl_next.index)

        breaches = pnl_next < -var_aligned
        n = int(len(breaches))
        x = int(breaches.sum())
        p_expected = 1.0 - self.config.confidence

        lr, p_value = kupiec_pof_test(x, n, self.config.confidence)

        window = self.config.backtest_window_days
        breaches_250 = breaches.tail(window)
        x_250 = int(breaches_250.sum())
        zone = basel_traffic_light(x_250)

        result = BacktestResult(
            method=method_name,
            observations=n,
            exceptions=x,
            expected_exceptions=n * p_expected,
            exception_rate=(x / n) if n else np.nan,
            kupiec_lr=lr,
            kupiec_p_value=p_value,
            kupiec_reject_95=bool(p_value < 0.05) if not np.isnan(p_value) else False,
            traffic_light=zone,
            exceptions_250d=x_250,
            breach_dates=list(breaches[breaches].index),
            breach_series=breaches,
        )
        logger.info("Backtest %s: %d/%d exceptions (expected %.1f), "
                    "Kupiec p=%.4f, zone=%s",
                    method_name, x, n, n * p_expected, p_value, zone.value)
        return result

    def backtest_all(self, pnl: pd.Series,
                     rolling_var_frame: pd.DataFrame) -> Dict[str, BacktestResult]:
        return {
            col: self.backtest_series(pnl, rolling_var_frame[col].dropna(), col)
            for col in rolling_var_frame.columns
        }

    @staticmethod
    def summary_frame(results: Dict[str, BacktestResult]) -> pd.DataFrame:
        return pd.DataFrame([r.to_row() for r in results.values()])
