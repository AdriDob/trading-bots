"""CoinGecko Connector — fetch crypto market data from CoinGecko.

This connector uses CoinGecko public API (no authentication required).
Used for research and paper trading only.
"""

from __future__ import annotations

import logging

from cores.trading.connectors.base import MarketDataConnector, PriceData

logger = logging.getLogger("ownex.trading.connectors.coingecko")


class CoinGeckoConnector(MarketDataConnector):
    """CoinGecko market data connector."""

    source_name = "coingecko"

    def __init__(self) -> None:
        super().__init__()
        self._base_url = "https://api.coingecko.com/api/v3"

    def initialize(self) -> bool:
        """Initialize the connector."""
        try:
            import requests

            response = requests.get(f"{self._base_url}/ping", timeout=5)
            self._initialized = response.status_code == 200
            logger.info("CoinGecko connector initialized: %s", self._initialized)
            return self._initialized
        except Exception as e:
            logger.error("Failed to initialize CoinGecko connector: %s", e)
            return False

    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str = "1h",
        limit: int = 100,
    ) -> list[PriceData]:
        """Fetch OHLCV data from CoinGecko."""
        if not self._initialized:
            logger.warning("CoinGecko connector not initialized")
            return []

        try:
            import requests

            # CoinGecko uses coin IDs, not symbols
            # Map common symbols to CoinGecko IDs
            symbol_map = {
                "BTC": "bitcoin",
                "ETH": "ethereum",
                "SOL": "solana",
                "BNB": "binancecoin",
                "ADA": "cardano",
            }

            coin_id = symbol_map.get(symbol.upper(), symbol.lower())

            # Map interval to CoinGecko format
            interval_map = {"1d": "1", "1h": "60"}  # Limited options in free API
            days = interval_map.get(interval, "1")

            response = requests.get(
                f"{self._base_url}/coins/{coin_id}/market_chart",
                params={"vs_currency": "usd", "days": days},
                timeout=10,
            )

            if response.status_code != 200:
                logger.error("CoinGecko API error: %s", response.status_code)
                return []

            data = response.json()
            prices = data.get("prices", [])

            price_data = []
            for price_point in prices[:limit]:
                price_data.append(
                    PriceData(
                        symbol=symbol,
                        timestamp=str(price_point[0]),
                        open=price_point[1],
                        high=price_point[1],
                        low=price_point[1],
                        close=price_point[1],
                        volume=0.0,  # CoinGecko doesn't provide volume in this endpoint
                        source=self.source_name,
                    )
                )

            logger.info("Fetched %d price points for %s from CoinGecko", len(price_data), symbol)
            return price_data

        except Exception as e:
            logger.error("Failed to fetch OHLCV from CoinGecko: %s", e)
            return []

    def fetch_current_price(self, symbol: str) -> float | None:
        """Fetch current price from CoinGecko."""
        if not self._initialized:
            return None

        try:
            import requests

            symbol_map = {
                "BTC": "bitcoin",
                "ETH": "ethereum",
                "SOL": "solana",
                "BNB": "binancecoin",
                "ADA": "cardano",
            }

            coin_id = symbol_map.get(symbol.upper(), symbol.lower())

            response = requests.get(
                f"{self._base_url}/coins/{coin_id}",
                params={
                    "localization": "false",
                    "tickers": "false",
                    "market_data": "true",
                    "community_data": "false",
                    "developer_data": "false",
                    "sparkline": "false",
                },
                timeout=5,
            )

            if response.status_code != 200:
                return None

            data = response.json()
            return data.get("market_data", {}).get("current_price", {}).get("usd")

        except Exception as e:
            logger.error("Failed to fetch current price from CoinGecko: %s", e)
            return None

    def get_supported_symbols(self) -> list[str]:
        """Get list of supported symbols."""
        return ["BTC", "ETH", "SOL", "BNB", "ADA"]
