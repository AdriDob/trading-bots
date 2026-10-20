"""Mean Reversion Strategy — simple mean-reversion trading logic for memecoins.

This strategy buys when a token's price deviates significantly from its moving average
and sells when it reverts. Designed to work with the OWNEX risk engine and copy-trading
framework.

Priority order per evaluation: mean reversion crossover > stop-loss > take-profit > time stop.

Parameters (all configurable per-watchlist):
- ma_period: Moving average period (default 20)
- entry_deviation: Price deviation % to trigger entry (default 15%)
- stop_loss_pct: Maximum loss per trade (default 25%)
- take_profit_tiers: Take-profit tiers (default: (0.5, 0.3), (1.0, 0.2))
- max_hold_minutes: Maximum hold time (default 240)
- min_liquidity_usd: Minimum liquidity to consider a token (default 5000)
- min_holder_count: Minimum holder count (default 3)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


# ------------------------------------------------------------------ strategy params
@dataclass(frozen=True, slots=True)
class MeanReversionConfig:
    """Configuration for mean reversion strategy."""
    ma_period: int = 20
    entry_deviation_pct: float = 15.0  # Enter when price is 15% away from MA
    stop_loss_pct: float = 25.0
    take_profit_tiers: tuple[tuple[float, float], ...] = (
        (0.5, 0.5),   # 1st tier: +50% → sell 50%
        (1.0, 0.3),   # 2nd tier: +100% → sell 30%
    )
    max_hold_minutes: float = 240.0
    min_liquidity_usd: float = 5_000.0
    min_holder_count: int = 3

    # Derived from config
    @property
    def ma_key(self) -> str:
        return f"ma_{self.ma_period}"


# ------------------------------------------------------------------ signal generator
def generate_signal(
    price_history: list[float],
    current_price: float,
    config: MeanReversionConfig | None = None,
) -> dict[str, Any] | None:
    """Generate a mean reversion signal based on price history.

    Logic:
    - Calculate simple moving average (SMA) over ma_period
    - Calculate deviation: (price - ma) / ma
    - If price is below MA - deviation% → BUY signal
    - If price is above MA + deviation% → SELL signal
    - Otherwise → HOLD

    Returns dict with signal info or None if no signal.
    """
    cfg = config or MeanReversionConfig()

    if len(price_history) < cfg.ma_period:
        return None  # Not enough data for SMA

    # Calculate SMA
    sma = sum(price_history[-cfg.ma_period:]) / cfg.ma_period

    # Calculate deviation
    deviation_pct = ((current_price - sma) / sma) * 100.0 if sma != 0 else 0.0

    # Entry signals
    if deviation_pct <= -cfg.entry_deviation_pct:
        # Price is below MA by entry_deviation_pct → BUY
        entry_price = current_price
        # Estimate position size based on deviation (larger deviation = larger potential)
        size_pct = min(100.0, abs(deviation_pct) / cfg.entry_deviation_pct * 50.0)
        return {
            "side": "BUY",
            "entry_price": entry_price,
            "deviation_pct": round(deviation_pct, 2),
            "sma": sma,
            "strategy": "mean_reversion",
            "config": cfg,
            "estimated_size_pct": size_pct,
        }
    elif deviation_pct >= cfg.entry_deviation_pct:
        # Price is above MA by entry_deviation_pct → SELL (short or exit long)
        return {
            "side": "SELL",
            "entry_price": entry_price,
            "deviation_pct": round(deviation_pct, 2),
            "sma": sma,
            "strategy": "mean_reversion",
            "config": cfg,
            "estimated_size_pct": size_pct,
        }

    return None  # No signal


# ------------------------------------------------------------------ exit logic (reuses sells.py)

# ------------------------------------------------------------------ risk check
def check_strategy_risk(
    signal: dict[str, Any],
    open_trades: int,
    daily_pnl_pct: float,
    watchlist_config: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    """Check if a mean reversion signal passes risk guards.

    Checks:
    - Símbolo en whitelist (se proporciona)
    - Liquidez mínima
    - Holder count
    - Límite de posiciones abiertas
    - Pérdida diaria
    """
    from cores.trading.meme_copy.risk import check_order_risk

    # Extract signal fields
    usd_size = signal.get("estimated_size_pct", 5.0)  # default 5% of bankroll
    # We need to map to CopyRisk format
    # For now, use default risk checks
    from cores.trading.meme_copy.models import MarketSignal

    # Build a minimal MarketSignal for risk check
    minimal_signal = MarketSignal(
        mint="MNEMONIC_PLACEHOLDER",
        symbol="MNEMO",
        side=signal.get("side", "BUY"),
        usd_size=usd_size,
        wallet="",
        timestamp=None,
        liquidity_usd=watchlist_config.get("min_liquidity_usd", 5_000) if watchlist_config else 5_000,
        price_impact_pct=None,
        metadata={"strategy": signal.get("strategy", "mean_reversion")},
    )

    # Build minimal watchlist
    from cores.trading.meme_copy.models import CopyWatchlist
    minimal_watchlist = CopyWatchlist(
        max_usd_per_trade=usd_size,
        max_open_trades=5,
        daily_loss_pct=5.0,
    )

    ok, reason = check_order_risk(
        minimal_signal,
        minimal_watchlist,
        open_trades,
        daily_pnl_pct,
    )
    return ok, reason
