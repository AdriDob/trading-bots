"""Binance Connector — fetch crypto market data from Binance.

This connector uses Binance public API (no authentication required for public data).
Used for research and paper trading only.
"""

from __future__ import annotations

import asyncio
import logging

import requests

from cores.trading.connectors.base import MarketDataConnector, PriceData

logger = logging.getLogger("ownex.trading.connectors.binance")

# Retry configuration for external API calls
MAX_RETRIES = 3
BASE_RETRY_DELAY = 1.0  # seconds — exponential backoff: 1s, 2s, 4s
MAX_RETRY_DELAY = 10.0  # seconds


# Sleep helper for retry backoff (async)
_sleep = asyncio.sleep


class BinanceConnector(MarketDataConnector):
    """Binance market data connector."""

    source_name = "binance"

    def __init__(self) -> None:
        super().__init__()
        self._base_url = "https://api.binance.com"

    def initialize(self) -> bool:
        """Initialize the connector."""
        try:
            # Test connectivity
            import requests

            response = requests.get(f"{self._base_url}/api/v3/ping", timeout=5)
            self._initialized = response.status_code == 200
            logger.info("Binance connector initialized: %s", self._initialized)
            return self._initialized
        except Exception as e:
            logger.error("Failed to initialize Binance connector: %s", e)
            return False

    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str = "1h",
        limit: int = 100,
    ) -> list[PriceData]:
        """Fetch OHLCV data from Binance with retry and backoff."""
        if not self._initialized:
            logger.warning("Binance connector not initialized")
            return []

        def _request_ohlcv() -> list[PriceData]:

            params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
            response = requests.get(f"{self._base_url}/api/v3/klines", params=params, timeout=10)

            if response.status_code != 200:
                logger.error("Binance API error: %s", response.status_code)
                raise Exception(f"HTTP {response.status_code}")

            data = response.json()

            price_data = []
            for candle in data:
                price_data.append(
                    PriceData(
                        symbol=symbol,
                        timestamp=str(candle[0]),
                        open=float(candle[1]),
                        high=float(candle[2]),
                        low=float(candle[3]),
                        close=float(candle[4]),
                        volume=float(candle[5]),
                        source=self.source_name,
                    )
                )

            logger.info("Fetched %d OHLCV candles for %s", len(price_data), symbol)
            return price_data

        # Retry with exponential backoff
        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                return _request_ohlcv()
            except Exception as e:
                last_error = e
                delay = min(BASE_RETRY_DELAY * (2**attempt), MAX_RETRY_DELAY)
                logger.warning(
                    "Binance fetch_ohlcv attempt %d/%d failed: %s — retrying in %.1fs",
                    attempt + 1,
                    MAX_RETRIES,
                    e,
                    delay,
                )
                _sleep(delay)  # blocking sleep in sync function

        logger.error("Binance fetch_ohlcv failed after %d attempts: %s", MAX_RETRIES, last_error)
        return []

    def fetch_current_price(self, symbol: str) -> float | None:
        """Fetch current price from Binance with retry and backoff."""
        if not self._initialized:
            return None

        def _request_current_price() -> float | None:

            params = {"symbol": symbol.upper()}
            response = requests.get(f"{self._base_url}/api/v3/ticker/price", params=params, timeout=5)

            if response.status_code != 200:
                raise Exception(f"HTTP {response.status_code}")

            data = response.json()
            return float(data["price"])

        # Retry with exponential backoff
        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                return _request_current_price()
            except Exception as e:
                last_error = e
                delay = min(BASE_RETRY_DELAY * (2**attempt), MAX_RETRY_DELAY)
                logger.warning(
                    "Binance fetch_current_price attempt %d/%d failed: %s — retrying in %.1fs",
                    attempt + 1,
                    MAX_RETRIES,
                    e,
                    delay,
                )
                _sleep(delay)

        logger.error("Binance fetch_current_price failed after %d attempts: %s", MAX_RETRIES, last_error)
        return None

    def get_supported_symbols(self) -> list[str]:
        """Get list of supported trading pairs."""
        # This is a placeholder - real implementation would fetch from Binance
        return ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT"]
