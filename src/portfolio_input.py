"""
portfolio_input.py
-------------------
Sub-System 1: Trade Portfolio Input.

Responsibilities (BRD Section 6.1):
  FR-1.1  CSV upload
  FR-1.2  Excel upload
  FR-1.3  API sourcing
  FR-1.4  Validation
  FR-1.5  Normalization into one internal Portfolio representation
  FR-1.6  Prototype-style re-use of a loaded portfolio (what-if scenarios)

Design pattern:
  - Factory: `PortfolioInputFactory` returns the right loader for a source type.
  - Strategy: every loader implements the same `PortfolioSource.load()` interface.
  - Prototype: `Portfolio.clone()` deep-copies an existing portfolio so a user
    can spin off a "what-if" variant without re-parsing the original file.
"""
from __future__ import annotations

import copy
import io
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Union

import pandas as pd

from .utils import AssetClass, logger

REQUIRED_COLUMNS = ["ticker", "quantity", "asset_class"]
OPTIONAL_COLUMNS = ["trade_price", "currency"]


@dataclass
class Position:
    ticker: str
    quantity: float
    asset_class: str = AssetClass.EQUITY.value
    trade_price: Optional[float] = None
    currency: str = "USD"

    def notional(self, market_price: float) -> float:
        return self.quantity * market_price


@dataclass
class Portfolio:
    """
    Normalized, internal representation of a trade portfolio (FR-1.5).
    This is the single object every downstream sub-system consumes.
    """
    name: str
    positions: List[Position] = field(default_factory=list)
    base_currency: str = "USD"

    @property
    def tickers(self) -> List[str]:
        return [p.ticker for p in self.positions]

    def weights_by_ticker(self) -> dict:
        return {p.ticker: p.quantity for p in self.positions}

    def clone(self, new_name: Optional[str] = None) -> "Portfolio":
        """
        Prototype pattern (FR-1.6): produce an independent deep copy of this
        portfolio so callers can build a "what-if" scenario (e.g. resize a
        position) without touching the original or re-reading the source file.
        """
        cloned = copy.deepcopy(self)
        if new_name:
            cloned.name = new_name
        return cloned

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            [{
                "ticker": p.ticker, "quantity": p.quantity,
                "asset_class": p.asset_class, "trade_price": p.trade_price,
                "currency": p.currency,
            } for p in self.positions]
        )

    def __len__(self) -> int:
        return len(self.positions)


class PortfolioValidationError(Exception):
    pass


class PortfolioSource(ABC):
    """Strategy interface every portfolio loader implements (FR-1.1..1.3)."""

    @abstractmethod
    def load(self, *args, **kwargs) -> Portfolio:
        raise NotImplementedError

    @staticmethod
    def _validate(df: pd.DataFrame) -> pd.DataFrame:
        """Shared validation logic (FR-1.4)."""
        df = df.copy()
        df.columns = [c.strip().lower() for c in df.columns]

        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise PortfolioValidationError(
                f"Missing required column(s): {missing}. "
                f"Required columns are: {REQUIRED_COLUMNS}"
            )

        for col in OPTIONAL_COLUMNS:
            if col not in df.columns:
                df[col] = None

        df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
        if df["ticker"].isnull().any() or (df["ticker"] == "").any():
            raise PortfolioValidationError("One or more rows have a blank ticker.")

        df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
        if df["quantity"].isnull().any():
            bad = df[df["quantity"].isnull()]["ticker"].tolist()
            raise PortfolioValidationError(f"Non-numeric quantity for ticker(s): {bad}")

        known = {a.value for a in AssetClass}
        df["asset_class"] = df["asset_class"].fillna(AssetClass.OTHER.value).astype(str)
        unknown = set(df["asset_class"]) - known
        if unknown:
            logger.warning("Unrecognized asset_class values %s -> mapped to 'Other'", unknown)
            df.loc[df["asset_class"].isin(unknown), "asset_class"] = AssetClass.OTHER.value

        if df["trade_price"].notnull().any():
            df["trade_price"] = pd.to_numeric(df["trade_price"], errors="coerce")

        df["currency"] = df["currency"].fillna("USD")

        if df["ticker"].duplicated().any():
            dupes = df[df["ticker"].duplicated()]["ticker"].tolist()
            logger.warning("Duplicate tickers %s will be aggregated by summing quantity.", dupes)
            df = df.groupby(
                ["ticker", "asset_class", "currency"], as_index=False
            ).agg({"quantity": "sum", "trade_price": "mean"})

        return df

    @staticmethod
    def _dataframe_to_portfolio(df: pd.DataFrame, name: str) -> Portfolio:
        positions = [
            Position(
                ticker=row.ticker, quantity=float(row.quantity),
                asset_class=row.asset_class,
                trade_price=None if pd.isnull(row.trade_price) else float(row.trade_price),
                currency=row.currency,
            )
            for row in df.itertuples(index=False)
        ]
        return Portfolio(name=name, positions=positions)


class CSVPortfolioSource(PortfolioSource):
    """FR-1.1: load a trade portfolio from a CSV file (path, bytes, or buffer)."""

    def load(self, file: Union[str, bytes, io.IOBase], name: str = "CSV Portfolio") -> Portfolio:
        if isinstance(file, (bytes, bytearray)):
            file = io.BytesIO(file)
        df = pd.read_csv(file)
        df = self._validate(df)
        portfolio = self._dataframe_to_portfolio(df, name)
        logger.info("Loaded %d positions from CSV into portfolio '%s'.", len(portfolio), name)
        return portfolio


class ExcelPortfolioSource(PortfolioSource):
    """FR-1.2: load a trade portfolio from an Excel (.xlsx) file."""

    def load(self, file: Union[str, bytes, io.IOBase], sheet_name=0,
              name: str = "Excel Portfolio") -> Portfolio:
        if isinstance(file, (bytes, bytearray)):
            file = io.BytesIO(file)
        df = pd.read_excel(file, sheet_name=sheet_name, engine="openpyxl")
        df = self._validate(df)
        portfolio = self._dataframe_to_portfolio(df, name)
        logger.info("Loaded %d positions from Excel into portfolio '%s'.", len(portfolio), name)
        return portfolio


class APIPortfolioSource(PortfolioSource):
    """
    FR-1.3: source a trade portfolio from a REST API.

    The endpoint is expected to return JSON: a list of records with the same
    fields as the CSV/Excel loaders. Any internal position-keeping system's
    API can be plugged in here as long as it returns that shape (or a caller
    supplies a `response_parser` to reshape a different payload).
    """

    def __init__(self, session=None):
        import requests  # local import: only needed if API sourcing is used
        self._requests = requests
        self._session = session or requests.Session()

    def load(self, url: str, name: str = "API Portfolio", headers: Optional[dict] = None,
              params: Optional[dict] = None, response_parser=None, timeout: int = 15) -> Portfolio:
        resp = self._session.get(url, headers=headers, params=params, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()
        records = response_parser(payload) if response_parser else payload
        df = pd.DataFrame(records)
        df = self._validate(df)
        portfolio = self._dataframe_to_portfolio(df, name)
        logger.info("Loaded %d positions from API into portfolio '%s'.", len(portfolio), name)
        return portfolio


class PortfolioInputFactory:
    """Factory (FR-1.5): returns the correct PortfolioSource for a source type."""

    _registry = {
        "csv": CSVPortfolioSource,
        "excel": ExcelPortfolioSource,
        "xlsx": ExcelPortfolioSource,
        "api": APIPortfolioSource,
    }

    @classmethod
    def get_source(cls, source_type: str) -> PortfolioSource:
        source_type = source_type.lower().strip()
        if source_type not in cls._registry:
            raise ValueError(
                f"Unknown portfolio source type '{source_type}'. "
                f"Valid options: {list(cls._registry.keys())}"
            )
        return cls._registry[source_type]()
