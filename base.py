"""Base Market Data Connector — abstract interface for market data sources."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

logger = logging.getLogger("ownex.trading.connectors.base")


class MarketDataType(StrEnum):
    """Types of market data."""

    OHLCV = "ohlcv"  # Open, High, Low, Close, Volume
    TICK = "tick"  # Individual trade ticks
    ORDERBOOK = "orderbook"  # Order book snapshot
    TRADES = "trades"  # Recent trades


@dataclass
class PriceData:
    """Price data point."""

    symbol: str
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    source: str
    metadata: dict[str, Any] | None = None


class MarketDataConnector(ABC):
    """Abstract base class for market data connectors."""

    source_name: str = "base"

    def __init__(self) -> None:
        self._initialized = False

    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the connector. Returns True if successful."""
        pass

    @abstractmethod
    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str = "1h",
        limit: int = 100,
    ) -> list[PriceData]:
        """Fetch OHLCV data for a symbol."""
        pass

    @abstractmethod
    def fetch_current_price(self, symbol: str) -> float | None:
        """Fetch current price for a symbol."""
        pass

    @abstractmethod
    def get_supported_symbols(self) -> list[str]:
        """Get list of supported symbols."""
        pass

    def is_healthy(self) -> bool:
        """Check if connector is healthy."""
        return self._initialized
