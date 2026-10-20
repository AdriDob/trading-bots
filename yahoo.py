"""Yahoo Finance Connector — fetch stock market data from Yahoo Finance.

This connector uses yfinance library (public data only).
Used for research and paper trading only.
"""

from __future__ import annotations

import logging

from cores.trading.connectors.base import MarketDataConnector, PriceData

logger = logging.getLogger("ownex.trading.connectors.yahoo")


class YahooFinanceConnector(MarketDataConnector):
    """Yahoo Finance market data connector."""

    source_name = "yahoo"

    def __init__(self) -> None:
        super().__init__()

    def initialize(self) -> bool:
        """Initialize the connector."""
        try:
            import yfinance as yf

            # Test connectivity with a popular stock
            ticker = yf.Ticker("AAPL")
            info = ticker.info
            self._initialized = bool(info)
            logger.info("Yahoo Finance connector initialized: %s", self._initialized)
            return self._initialized
        except ImportError:
            logger.warning("yfinance not installed, Yahoo Finance connector unavailable")
            return False
        except Exception as e:
            logger.error("Failed to initialize Yahoo Finance connector: %s", e)
            return False

    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str = "1h",
        limit: int = 100,
    ) -> list[PriceData]:
        """Fetch OHLCV data from Yahoo Finance."""
        if not self._initialized:
            logger.warning("Yahoo Finance connector not initialized")
            return []

        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            # Map interval strings to yfinance format
            interval_map = {"1m": "1m", "5m": "5m", "15m": "15m", "1h": "1h", "1d": "1d", "1w": "1w"}
            yf_interval = interval_map.get(interval, "1h")

            hist = ticker.history(period=yf_interval, interval=yf_interval)

            price_data = []
            for timestamp, row in hist.iterrows():
                price_data.append(
                    PriceData(
                        symbol=symbol,
                        timestamp=str(timestamp),
                        open=float(row["Open"]),
                        high=float(row["High"]),
                        low=float(row["Low"]),
                        close=float(row["Close"]),
                        volume=float(row["Volume"]),
                        source=self.source_name,
                    )
                )

            logger.info("Fetched %d OHLCV candles for %s from Yahoo Finance", len(price_data), symbol)
            return price_data

        except Exception as e:
            logger.error("Failed to fetch OHLCV from Yahoo Finance: %s", e)
            return []

    def fetch_current_price(self, symbol: str) -> float | None:
        """Fetch current price from Yahoo Finance."""
        if not self._initialized:
            return None

        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            info = ticker.info
            return info.get("currentPrice") or info.get("regularMarketPrice")

        except Exception as e:
            logger.error("Failed to fetch current price from Yahoo Finance: %s", e)
            return None

    def get_supported_symbols(self) -> list[str]:
        """Get list of supported symbols."""
        # This is a placeholder - real implementation would fetch from Yahoo Finance
        return ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "NVDA", "META"]
