"""Generic Launchpad Adapter — unified interface for Solana token launchpads.

Supports multiple launchpad platforms:
- pump.fun (bonding curve, graduated to Raydium)
- moonshot (moonshot.money)
- bonk (bonk.fun)
- raydium (direct Raydium launchpad)
- meteora (Meteora DLMM)

Each platform has its own API/feed, but this adapter normalizes them
to a common schema for the trading pipeline.

Design:
- Pure functions for filtering (no network in verdicts)
- Fetch never raises (returns [] on failure)
- Read-only by default
- Live execution stays behind TradingConfig REAL + Human Gate + kill-switch
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("ownex.trading.launchpad")

# ------------------------------------------------------------------ platform registry
LAUNCHPAD_PLATFORMS = {
    "pumpfun": {
        "name": "Pump.fun",
        "coins_url": "https://frontend-api-v3.pump.fun/coins?offset=0&limit={limit}&sort=created_timestamp&order=DESC&includeNsfw=false",
        "graduation_mc": 69_000.0,  # Raydium migration
        "url_pattern": "https://pump.fun/coin/{mint}",
        "api_style": "pumpfun",
    },
    "moonshot": {
        "name": "Moonshot",
        "coins_url": "https://api.moonshot.money/v1/coins?limit={limit}&sort=created_at&order=desc",
        "graduation_mc": 50_000.0,  # approximate
        "url_pattern": "https://moonshot.money/coin/{mint}",
        "api_style": "moonshot",
    },
    "bonk": {
        "name": "Bonk.fun",
        "coins_url": "https://api.bonk.fun/v1/tokens?limit={limit}&sort=recent",
        "graduation_mc": 40_000.0,  # approximate
        "url_pattern": "https://bonk.fun/token/{mint}",
        "api_style": "bonk",
    },
}


@dataclass(frozen=True, slots=True)
class LaunchpadConfig:
    """Configuration for launchpad scanning."""

    platform: str = "pumpfun"  # default to pump.fun
    max_age_minutes: float = 30.0
    min_mcap_usd: float = 3_000.0
    max_mcap_usd: float = 0.0  # 0 = use platform default
    block_nsfw: bool = True
    max_launches: int = 25


def normalize_launchpad_token(
    raw: dict[str, Any], platform: str, sol_price_usd: float | None = None
) -> dict[str, Any] | None:
    """Normalize a raw launchpad token to a common schema.

    Returns None if the token cannot be normalized.
    """
    try:
        platform_config = LAUNCHPAD_PLATFORMS.get(platform, {})
        url_pattern = platform_config.get("url_pattern", "")

        # Common fields across platforms
        mint = raw.get("mint") or raw.get("address") or raw.get("id")
        if not mint:
            return None

        symbol = raw.get("symbol") or raw.get("name") or raw.get("ticker") or "?"

        # Market cap / liquidity
        mcap = _num(raw.get("market_cap") or raw.get("mcap") or raw.get("usd_market_cap"))
        liquidity = _num(raw.get("liquidity") or raw.get("liquidity_usd") or raw.get("curve_sol"))

        # Price
        price = _num(raw.get("price") or raw.get("price_usd"))

        # Created timestamp
        created = raw.get("created_timestamp") or raw.get("created_at") or raw.get("launch_time")
        if created:
            try:
                created_ts = float(created)
                if created_ts > 1e10:  # ms
                    created_ts = created_ts / 1000.0
            except (TypeError, ValueError):
                created_ts = 0.0
        else:
            created_ts = 0.0

        age_min = (time.time() - created_ts) / 60.0 if created_ts > 0 else 0.0

        # NSFW / banned
        nsfw = bool(raw.get("nsfw") or raw.get("banned") or raw.get("is_nsfw"))

        # Graduated (completed bonding curve)
        complete = bool(raw.get("complete") or raw.get("graduated") or raw.get("migrated"))

        url = url_pattern.format(mint=mint) if url_pattern else ""

        return {
            "mint": mint,
            "symbol": symbol,
            "name": raw.get("name", symbol),
            "platform": platform,
            "platform_name": platform_config.get("name", platform),
            "mcap_usd": mcap or 0.0,
            "liquidity_usd": liquidity or 0.0,
            "price_usd": price or 0.0,
            "created_timestamp": created_ts,
            "age_minutes": round(age_min, 1),
            "nsfw": nsfw,
            "complete": complete,
            "url": url,
            "source": platform,
        }
    except Exception as exc:
        logger.warning("Failed to normalize launchpad token from %s: %s", platform, exc)
        return None


def _num(value: Any) -> float | None:
    try:
        if value is None or isinstance(value, bool):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


async def scan_launchpad(
    platform: str = "pumpfun",
    config: LaunchpadConfig | None = None,
    sol_price_usd: float | None = None,
) -> list[dict[str, Any]]:
    """Scan a launchpad platform for fresh launches.

    Returns normalized token dicts. Never raises.
    """
    import httpx

    cfg = config or LaunchpadConfig(platform=platform)
    platform_config = LAUNCHPAD_PLATFORMS.get(platform)
    if not platform_config:
        logger.warning("Unknown launchpad platform: %s", platform)
        return []

    coins_url = platform_config["coins_url"].format(limit=cfg.max_launches)
    graduation_mc = platform_config["graduation_mc"]

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(coins_url)
            if resp.status_code != 200:
                logger.warning("%s feed HTTP %s", platform, resp.status_code)
                return []
            data = resp.json()
            # Different platforms return different structures
            raw_list = data if isinstance(data, list) else data.get("coins", data.get("tokens", data.get("data", [])))
            if not isinstance(raw_list, list):
                return []
    except Exception as exc:
        logger.warning("%s feed failed: %s", platform, exc)
        return []

    out = []
    for raw in raw_list[: cfg.max_launches]:
        if not isinstance(raw, dict):
            continue
        tok = normalize_launchpad_token(raw, platform, sol_price_usd)
        if tok is None:
            continue
        if tok["nsfw"] and cfg.block_nsfw:
            continue
        if tok["complete"]:
            continue
        if tok["age_minutes"] > cfg.max_age_minutes:
            continue
        if tok["mcap_usd"] < cfg.min_mcap_usd:
            continue
        max_mc = cfg.max_mcap_usd or graduation_mc
        if tok["mcap_usd"] > max_mc:
            continue
        out.append(tok)

    return out


async def scan_all_launchpads(
    platforms: list[str] | None = None,
    config: LaunchpadConfig | None = None,
    sol_price_usd: float | None = None,
) -> list[dict[str, Any]]:
    """Scan multiple launchpads concurrently.

    Returns deduplicated tokens (by mint), newest first.
    """
    import asyncio

    platforms = platforms or ["pumpfun"]
    cfg = config or LaunchpadConfig()

    tasks = [scan_launchpad(p, cfg, sol_price_usd) for p in platforms]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    seen_mints: set[str] = set()
    out = []
    for result in results:
        if isinstance(result, Exception):
            continue
        for tok in result:
            if tok["mint"] in seen_mints:
                continue
            seen_mints.add(tok["mint"])
            out.append(tok)

    # Sort by age (newest first)
    out.sort(key=lambda t: t.get("age_minutes", 999))
    return out


# ------------------------------------------------------------------ convenience
def get_platform_names() -> list[str]:
    """Return list of supported launchpad platform names."""
    return list(LAUNCHPAD_PLATFORMS.keys())


def get_platform_config(platform: str) -> dict[str, Any] | None:
    """Return config for a specific platform."""
    return LAUNCHPAD_PLATFORMS.get(platform)
