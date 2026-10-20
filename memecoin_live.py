"""Memecoin live loop — semiautonomous real execution with a human arming gate.

Parity with :mod:`cores.trading.paper.memecoin_paper`:

    scan (DexScreener) → hard filters + RugCheck → entry → track → staged exits

The same code paths are reused (``MemecoinPaperLoop.scan`` and
``strategies.memecoin_exits``). The differences are all fail-closed:

- ``TRADING_MODE=real`` is mandatory.
- A Solana wallet must be configured (``SOLANA_PRIVATE_KEY`` /
  ``SOLANA_KEYPAIR_PATH`` / IdentityVault).
- Arming requires a single-use HMAC approval token, a purpose, a session TTL
  and an explicit budget.
- Only *new entries* are gated by the arming session. Open positions keep
  being tracked and exited after expiry/disarm (risk reduction is never
  blocked by a human gate).
- Budget, max open positions, daily realized-loss, max daily entries,
  slippage and Jupiter price impact are enforced in code.
- A daily-loss breach trips the persisted STRATEGY kill switch and disarms.
  Re-arming requires a human unlock token.
- Mutating cycles are protected by a cross-process lock.

The paper loop remains the default. This module is inert until armed.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from cores.automation.safety import consume_approval
from cores.trading.config import TradingConfig
from cores.trading.paper.memecoin_paper import MemecoinPaperLoop, PaperConfig, dex_price_usd, paper_metrics
from cores.trading.risk import KillSwitchLevel, KillSwitchManager
from cores.trading.store import TradingStore
from cores.trading.strategies.memecoin_exits import ExitConfig, evaluate_exit
from cores.utils.oscompat import LockHeldError, acquire_lock, release_lock

logger = logging.getLogger("ownex.trading.memecoin_live")

STRATEGY_ID = "memecoin_live_v1"
STATE_KEY = "memecoin_live_state"
POSITIONS_KEY = "memecoin_live_positions"
LEDGER_KEY = "memecoin_live_trades"
PENDING_ENTRY_KEY = "memecoin_live_pending_entry"
LOCK_NAME = "memecoin_live.lock"

APPROVAL_ACTION = "memecoin_live_arm"
APPROVAL_UNLOCK_ACTION = "memecoin_live_unlock"
APPROVAL_TARGET = "memecoin_live"
WORKSPACE = "trading"

# Hard ceilings. User-provided limits can be lower, never higher.
MAX_POSITION_SOL = 0.5
MAX_BUDGET_SOL = 5.0
MAX_OPEN_POSITIONS = 3
MAX_DAILY_LOSS_SOL = 1.0
MAX_DAILY_ENTRIES = 12
MIN_HOLD_MINUTES = 5.0
MAX_HOLD_MINUTES = 240.0
MIN_SLIPPAGE_BPS = 50
MAX_SLIPPAGE_BPS = 500
MIN_LIQUIDITY_USD = 10_000.0
MAX_PRICE_IMPACT_PCT = 5.0
MIN_SESSION_HOURS = 0.25
MAX_SESSION_HOURS = 24.0
MIN_SCAN_INTERVAL_SECONDS = 60.0
MAX_SCAN_INTERVAL_SECONDS = 3600.0

_UNSET = object()


class LiveArmError(Exception):
    """Arming/config refusal with an HTTP status hint for the router."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True, slots=True)
class LiveLimits:
    """Risk envelope persisted with the arming session."""

    position_sol: float = 0.1
    max_open_positions: int = 2
    daily_loss_limit_sol: float = 0.25
    max_hold_minutes: float = 120.0
    max_daily_entries: int = 6
    slippage_bps: int = 300
    min_liquidity_usd: float = MIN_LIQUIDITY_USD
    max_price_impact_pct: float = MAX_PRICE_IMPACT_PCT
    session_hours: float = 8.0
    scan_interval_seconds: float = 600.0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> LiveLimits:
        raw = {k: v for k, v in (data or {}).items() if k in cls.__dataclass_fields__ and v is not None}
        try:
            return cls(**raw)
        except (TypeError, ValueError):
            logger.warning("invalid live limits in store — falling back to defaults")
            return cls()


def _as_float(value: Any) -> float | None:
    try:
        if value is None or isinstance(value, bool):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    try:
        if value is None or isinstance(value, bool):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _today(now: float) -> str:
    return time.strftime("%Y-%m-%d", time.gmtime(now))


def _impact_pct(value: Any) -> float | None:
    """Normalize Jupiter's ``priceImpactPct`` to percent.

    Jupiter returns a fraction (measured 2026-10: 0.000104 ≈ 0.0104%). Older
    callers/tests use percent directly; values > 1 are treated as percent.
    """
    raw = _as_float(value)
    if raw is None:
        return None
    return raw if raw > 1.0 else raw * 100.0


def _iso(ts: float | None) -> str | None:
    if not ts:
        return None
    try:
        return datetime.fromtimestamp(float(ts), UTC).isoformat()
    except (OSError, OverflowError, ValueError):
        return None


class MemecoinLiveLoop:
    """Live counterpart of the memecoin paper loop. Arm before entries."""

    def __init__(
        self,
        adapter: Any | None,
        store: TradingStore | None = None,
        kill: KillSwitchManager | None = None,
        config: LiveLimits | None = None,
        trading_config: TradingConfig | None = None,
        price_fn: Callable[[str], Awaitable[float | None]] | None = None,
        now_fn: Callable[[], float] | None = None,
        sol_price_fn: Callable[[], float | None] | None = None,
        confirm_timeout_seconds: float = 30.0,
    ) -> None:
        self._adapter = adapter
        self._store = store or TradingStore()
        self._kill = kill or KillSwitchManager(store=self._store)
        self._config = config or LiveLimits()
        self._trading_config = trading_config or TradingConfig.from_env()
        self._price_fn = price_fn or dex_price_usd
        self._now = now_fn or time.time
        self._sol_price_fn = sol_price_fn
        self._confirm_timeout_seconds = max(0.0, float(confirm_timeout_seconds))

    # ── persistence ────────────────────────────────────────────────────

    def _state(self) -> dict[str, Any]:
        raw = self._store.get(STATE_KEY)
        return dict(raw) if isinstance(raw, dict) else {}

    def _save_state(self, state: dict[str, Any]) -> None:
        state["updated_at"] = self._now()
        self._store.set(STATE_KEY, state)

    def _positions(self) -> list[dict[str, Any]]:
        raw = self._store.get(POSITIONS_KEY)
        if not isinstance(raw, list):
            return []
        return [dict(p) for p in raw if isinstance(p, dict) and p.get("mint")]

    def _save_positions(self, positions: list[dict[str, Any]]) -> None:
        self._store.set(POSITIONS_KEY, positions)

    def _ledger(self) -> list[dict[str, Any]]:
        raw = self._store.get(LEDGER_KEY)
        if not isinstance(raw, list):
            return []
        return [dict(t) for t in raw if isinstance(t, dict)]

    def _append_trade(self, trade: dict[str, Any]) -> None:
        ledger = self._ledger()
        ledger.append(trade)
        self._store.set(LEDGER_KEY, ledger)

    # ── status / arming ───────────────────────────────────────────────

    def _state_armed(self, state: dict[str, Any], now: float) -> bool:
        if not state.get("armed"):
            return False
        try:
            return float(state.get("armed_until") or 0) > now
        except (TypeError, ValueError):
            return False

    def is_armed(self, now: float | None = None) -> bool:
        return self._state_armed(self._state(), now if now is not None else self._now())

    def _today_realized(self, ledger: list[dict[str, Any]], now: float) -> float:
        day = _today(now)
        total = 0.0
        for trade in ledger:
            closed = _as_float(trade.get("closed_at"))
            if closed is None or _today(closed) != day:
                continue
            total += _as_float(trade.get("pnl_sol")) or 0.0
        return round(total, 6)

    def _metrics(self, positions: list[dict[str, Any]], ledger: list[dict[str, Any]]) -> dict[str, Any]:
        pnls = [SimpleNamespace(pnl_sol=_as_float(t.get("pnl_sol")) or 0.0) for t in ledger]
        summary = paper_metrics(pnls)  # type: ignore[arg-type]
        summary["open_positions"] = len(positions)
        summary["kill_active"] = self._kill.is_active(KillSwitchLevel.STRATEGY, STRATEGY_ID)
        summary["persisted"] = True
        return summary

    def status(self, now: float | None = None) -> dict[str, Any]:
        moment = now if now is not None else self._now()
        state = self._state()
        positions = self._positions()
        ledger = self._ledger()
        armed = self._state_armed(state, moment)
        try:
            until = float(state.get("armed_until") or 0)
        except (TypeError, ValueError):
            until = 0.0
        try:
            spent = float(state.get("spent_sol") or 0.0)
        except (TypeError, ValueError):
            spent = 0.0
        try:
            budget = float(state.get("budget_sol") or 0.0)
        except (TypeError, ValueError):
            budget = 0.0
        return {
            "strategy_id": STRATEGY_ID,
            "mode": self._trading_config.mode.value,
            "is_live": self._trading_config.is_live,
            "adapter_ready": bool(self._adapter is not None and not getattr(self._adapter, "is_dry_run", True)),
            "armed": armed,
            "armed_at": _iso(_as_float(state.get("armed_at"))),
            "armed_until": _iso(until),
            "expires_in_seconds": max(0, int(until - moment)) if armed else 0,
            "armed_purpose": state.get("purpose") or "",
            "limits": dataclasses.asdict(self._config),
            "budget_sol": round(budget, 6),
            "spent_sol": round(spent, 6),
            "budget_remaining_sol": round(max(0.0, budget - spent), 6),
            "positions": positions,
            "open_positions": len(positions),
            "metrics": self._metrics(positions, ledger),
            "today_realized_sol": self._today_realized(ledger, moment),
            "kill_switch": self._kill.get_status(),
            "last_cycle_at": _iso(_as_float(state.get("last_cycle_at"))),
            "last_scan_at": _iso(_as_float(state.get("last_scan_at"))),
            "last_error": state.get("last_error") or "",
            "disarmed_reason": state.get("disarmed_reason") or "",
            "disarmed_at": _iso(_as_float(state.get("disarmed_at"))),
        }

    def _validate_limits(self, limits: LiveLimits, budget_sol: float, sol_price_usd: float) -> None:
        if not (0 < limits.position_sol <= MAX_POSITION_SOL):
            raise LiveArmError(f"position_sol must be in (0, {MAX_POSITION_SOL}]", 400)
        if not (1 <= limits.max_open_positions <= MAX_OPEN_POSITIONS):
            raise LiveArmError(f"max_open_positions must be in [1, {MAX_OPEN_POSITIONS}]", 400)
        if not (0 < limits.daily_loss_limit_sol <= MAX_DAILY_LOSS_SOL):
            raise LiveArmError(f"daily_loss_limit_sol must be in (0, {MAX_DAILY_LOSS_SOL}]", 400)
        if not (MIN_HOLD_MINUTES <= limits.max_hold_minutes <= MAX_HOLD_MINUTES):
            raise LiveArmError(f"max_hold_minutes must be in [{MIN_HOLD_MINUTES}, {MAX_HOLD_MINUTES}]", 400)
        if not (1 <= limits.max_daily_entries <= MAX_DAILY_ENTRIES):
            raise LiveArmError(f"max_daily_entries must be in [1, {MAX_DAILY_ENTRIES}]", 400)
        if not (MIN_SLIPPAGE_BPS <= limits.slippage_bps <= MAX_SLIPPAGE_BPS):
            raise LiveArmError(f"slippage_bps must be in [{MIN_SLIPPAGE_BPS}, {MAX_SLIPPAGE_BPS}]", 400)
        if limits.min_liquidity_usd < MIN_LIQUIDITY_USD:
            raise LiveArmError(f"min_liquidity_usd must be >= {MIN_LIQUIDITY_USD}", 400)
        if not (0 < limits.max_price_impact_pct <= MAX_PRICE_IMPACT_PCT):
            raise LiveArmError(f"max_price_impact_pct must be in (0, {MAX_PRICE_IMPACT_PCT}]", 400)
        if not (MIN_SESSION_HOURS <= limits.session_hours <= MAX_SESSION_HOURS):
            raise LiveArmError(f"session_hours must be in [{MIN_SESSION_HOURS}, {MAX_SESSION_HOURS}]", 400)
        if not (MIN_SCAN_INTERVAL_SECONDS <= limits.scan_interval_seconds <= MAX_SCAN_INTERVAL_SECONDS):
            raise LiveArmError(
                f"scan_interval_seconds must be in [{MIN_SCAN_INTERVAL_SECONDS}, {MAX_SCAN_INTERVAL_SECONDS}]", 400
            )
        if not (limits.position_sol <= budget_sol <= MAX_BUDGET_SOL):
            raise LiveArmError(f"budget_sol must be in [position_sol, {MAX_BUDGET_SOL}]", 400)
        if sol_price_usd <= 0:
            raise LiveArmError("SOL price unavailable — refusing to size live risk", 503)
        if limits.position_sol * sol_price_usd > self._trading_config.max_position_size_usd:
            raise LiveArmError(
                "position exceeds TradingConfig.max_position_size_usd "
                f"(${self._trading_config.max_position_size_usd:.2f})",
                422,
            )

    def arm(
        self,
        *,
        purpose: str,
        approval_token: str,
        limits: LiveLimits,
        budget_sol: float,
        sol_price_usd: float | None = None,
        now: float | None = None,
    ) -> dict[str, Any]:
        """Enable NEW entries for a bounded session. Human token is single-use."""
        moment = now if now is not None else self._now()
        if not self._trading_config.is_live:
            raise LiveArmError("TRADING_MODE=real required for live trading", 412)
        if self._adapter is None or getattr(self._adapter, "is_dry_run", True):
            raise LiveArmError("Solana wallet not configured — live refuses to arm", 412)
        if self._kill.is_active(KillSwitchLevel.GLOBAL):
            raise LiveArmError("global kill switch active", 423)
        if self._kill.is_active(KillSwitchLevel.STRATEGY, STRATEGY_ID):
            raise LiveArmError("memecoin live strategy kill switch active — unlock with human token", 423)

        price = sol_price_usd
        if price is None:
            price = (self._sol_price_fn or _default_sol_price)()
        if price is None:
            raise LiveArmError("SOL price unavailable — refusing to size live risk", 503)
        self._validate_limits(limits, budget_sol, float(price))

        purpose = (purpose or "").strip()
        if len(purpose) < 3:
            raise LiveArmError("purpose is required (min 3 chars)", 400)

        state = self._state()
        positions = self._positions()
        try:
            spent = float(state.get("spent_sol") or 0.0)
        except (TypeError, ValueError):
            spent = 0.0
        # A flat strategy starts a fresh budget; with open exposure the spent
        # amount is carried over so re-arming cannot reset the risk envelope.
        if not positions:
            spent = 0.0
        elif budget_sol < spent + limits.position_sol:
            raise LiveArmError(f"budget_sol must cover already-spent {spent:.4f} SOL plus one position", 422)

        if not consume_approval(approval_token, APPROVAL_ACTION, APPROVAL_TARGET, WORKSPACE):
            raise LiveArmError("invalid, expired or already-used approval token", 403)

        state.update(
            {
                "armed": True,
                "armed_at": moment,
                "armed_until": moment + limits.session_hours * 3600.0,
                "purpose": purpose,
                "budget_sol": float(budget_sol),
                "spent_sol": spent,
                "position_sol": limits.position_sol,
                "max_open_positions": limits.max_open_positions,
                "daily_loss_limit_sol": limits.daily_loss_limit_sol,
                "max_hold_minutes": limits.max_hold_minutes,
                "max_daily_entries": limits.max_daily_entries,
                "slippage_bps": limits.slippage_bps,
                "min_liquidity_usd": limits.min_liquidity_usd,
                "max_price_impact_pct": limits.max_price_impact_pct,
                "session_hours": limits.session_hours,
                "scan_interval_seconds": limits.scan_interval_seconds,
                "disarmed_reason": "",
                "disarmed_at": 0.0,
                "last_error": "",
            }
        )
        self._save_state(state)
        self._audit("live_armed", f"armed for {purpose}", {"budget_sol": budget_sol}, 0.99, 0.8)
        self._notify("trading:live_armed", message=f"memecoin live armado: {purpose} (budget {budget_sol} SOL)")
        return self.status(now=moment)

    def disarm(self, reason: str = "manual", now: float | None = None) -> dict[str, Any]:
        """Stop NEW entries. Open positions remain tracked and can still exit."""
        moment = now if now is not None else self._now()
        state = self._state()
        state["armed"] = False
        state["disarmed_reason"] = (reason or "manual").strip()
        state["disarmed_at"] = moment
        self._save_state(state)
        self._audit("live_disarmed", state["disarmed_reason"], {}, 0.99, 0.1)
        self._notify("trading:live_disarmed", message=f"memecoin live desarmado: {state['disarmed_reason']}")
        return self.status(now=moment)

    # ── cycle ─────────────────────────────────────────────────────────

    def _expire_if_needed(self, now: float) -> None:
        state = self._state()
        if state.get("armed") and not self._state_armed(state, now):
            state["armed"] = False
            state["disarmed_reason"] = "session expired"
            state["disarmed_at"] = now
            self._save_state(state)
            self._audit("live_expired", "arming session expired", {}, 0.99, 0.1)
            self._notify("trading:live_expired", message="memecoin live: sesión expirada; no hay entradas nuevas")

    async def run_once(self, now: float | None = None) -> dict[str, Any]:
        moment = now if now is not None else self._now()
        self._expire_if_needed(moment)
        await self._recover_pending_entry(moment)
        closed = await self.track(now=moment)

        state = self._state()
        if not self._state_armed(state, moment):
            return self._summary("disarmed", [], closed, moment)
        if self._kill.is_active(KillSwitchLevel.GLOBAL) or self._kill.is_active(KillSwitchLevel.STRATEGY, STRATEGY_ID):
            return self._summary("kill_switch", [], closed, moment)
        if self._daily_loss_breached(moment):
            self._trip_kill(f"daily loss limit {self._config.daily_loss_limit_sol} SOL breached")
            return self._summary("daily_loss", [], closed, moment)
        if self._adapter is None or getattr(self._adapter, "is_dry_run", True):
            return self._summary("adapter_unavailable", [], closed, moment)

        last_scan = _as_float(state.get("last_scan_at")) or 0.0
        if moment - last_scan < self._config.scan_interval_seconds:
            return self._summary("scan_cooldown", [], closed, moment)

        state["last_scan_at"] = moment
        state["last_cycle_at"] = moment
        state["last_error"] = ""
        self._save_state(state)

        entered = await self._enter_candidates(moment)
        return self._summary("ok", entered, closed, moment)

    def _summary(self, status: str, entered: list[str], closed: list[dict[str, Any]], now: float) -> dict[str, Any]:
        state = self._state()
        state["last_cycle_at"] = now
        self._save_state(state)
        positions = self._positions()
        return {
            "success": status in ("ok", "scan_cooldown"),
            "status": status,
            "entered": entered,
            "closed": closed,
            "armed": self._state_armed(state, now),
            "open_positions": len(positions),
            "metrics": self._metrics(positions, self._ledger()),
            "today_realized_sol": self._today_realized(self._ledger(), now),
            "last_error": state.get("last_error") or "",
        }

    # ── entries ───────────────────────────────────────────────────────

    async def _enter_candidates(self, now: float) -> list[str]:
        positions = self._positions()
        if len(positions) >= self._config.max_open_positions:
            return []
        if self._adapter is None:
            return []

        state = self._state()
        day = _today(now)
        if state.get("entries_day") != day:
            state["entries_day"] = day
            state["entries_today"] = 0
            self._save_state(state)
        try:
            entries_today = int(state.get("entries_today") or 0)
        except (TypeError, ValueError):
            entries_today = 0
        try:
            spent = float(state.get("spent_sol") or 0.0)
        except (TypeError, ValueError):
            spent = 0.0
        try:
            budget = float(state.get("budget_sol") or 0.0)
        except (TypeError, ValueError):
            budget = 0.0

        slots = min(
            self._config.max_open_positions - len(positions),
            self._config.max_daily_entries - entries_today,
        )
        affordable = int(max(0.0, budget - spent) // self._config.position_sol)
        slots = min(slots, affordable)
        if slots <= 0:
            return []

        paper = MemecoinPaperLoop(
            adapter=self._adapter,
            store=None,
            kill=self._kill,
            config=PaperConfig(
                position_sol=self._config.position_sol,
                max_open_positions=self._config.max_open_positions,
                daily_loss_limit_sol=self._config.daily_loss_limit_sol,
                max_hold_minutes=self._config.max_hold_minutes,
                min_liquidity_usd=self._config.min_liquidity_usd,
                strategy_id=STRATEGY_ID,
                use_pumpfun_launches=False,
            ),
        )
        try:
            candidates = await paper.scan()
        except Exception as exc:
            logger.warning("live scan failed: %s", exc)
            state = self._state()
            state["last_error"] = f"scan failed: {exc}"
            self._save_state(state)
            return []

        entered: list[str] = []
        for candidate in candidates:
            if len(entered) >= slots:
                break
            mint = str(candidate.get("mint") or "")
            if not mint or any(p.get("mint") == mint for p in self._positions()):
                continue
            position = await self._enter(candidate, now)
            if position is not None:
                entered.append(mint)
        return entered

    async def _enter(self, candidate: dict[str, Any], now: float) -> dict[str, Any] | None:
        mint = str(candidate.get("mint") or "")
        price = _as_float(candidate.get("price_usd"))
        if not mint or price is None or price <= 0:
            return None
        liquidity = _as_float(candidate.get("liquidity_usd"))
        if liquidity is not None and liquidity < self._config.min_liquidity_usd:
            return None

        try:
            quote = await self._adapter.quote_buy(mint, self._config.position_sol, self._config.slippage_bps)
        except Exception as exc:
            self._set_error(f"quote failed for {mint}: {exc}")
            return None
        if not quote:
            self._set_error(f"no Jupiter route for {mint}")
            return None
        impact = _impact_pct(quote.get("price_impact_pct"))
        if impact is None:
            self._set_error(f"price impact unknown for {mint} (fail closed)")
            return None
        if impact > self._config.max_price_impact_pct:
            self._set_error(f"price impact {impact:.2f}% > {self._config.max_price_impact_pct}% for {mint}")
            return None

        state = self._state()
        try:
            spent_before = float(state.get("spent_sol") or 0.0)
            entries_today = int(state.get("entries_today") or 0)
        except (TypeError, ValueError):
            spent_before, entries_today = 0.0, 0
        budget = _as_float(state.get("budget_sol")) or 0.0
        if spent_before + self._config.position_sol > budget:
            return None

        self._store.set(
            PENDING_ENTRY_KEY,
            {
                "mint": mint,
                "symbol": str(candidate.get("symbol") or "?"),
                "amount_sol": self._config.position_sol,
                "price_usd": price,
                "submitted_at": now,
            },
        )

        sol_before = await self._sol_balance()
        try:
            result = await self._adapter.buy(mint, self._config.position_sol, self._config.slippage_bps)
        except Exception as exc:
            self._store.set(PENDING_ENTRY_KEY, {})
            self._set_error(f"buy failed for {mint}: {exc}")
            return None
        if not isinstance(result, dict) or result.get("status") != "executed":
            self._store.set(PENDING_ENTRY_KEY, {})
            self._set_error(f"buy not executed for {mint}: {result}")
            return None

        token_balance = await self._await_token_balance(mint)
        sol_after = await self._sol_balance()
        spent = self._config.position_sol
        if sol_before is not None and sol_after is not None and sol_before > sol_after:
            spent = (sol_before - sol_after) / 1_000_000_000
        spent = max(0.0, min(spent, self._config.position_sol * 1.1))

        position = {
            "mint": mint,
            "symbol": str(candidate.get("symbol") or "?"),
            "entry_sol": round(spent, 9),
            "entry_price_usd": price,
            "peak_price_usd": price,
            "last_price_usd": price,
            "remaining_pct": 100.0,
            "token_raw": int(token_balance),
            "entered_at": now,
            "txid_buy": str(result.get("txid") or ""),
            "unverified": token_balance <= 0,
            "last_error": "" if token_balance > 0 else "buy tx sent but token balance not observed",
        }

        positions = self._positions()
        positions.append(position)
        self._save_positions(positions)

        state = self._state()
        state["spent_sol"] = round(spent_before + spent, 9)
        state["entries_day"] = _today(now)
        state["entries_today"] = entries_today + 1
        state["last_cycle_at"] = now
        state["last_error"] = position["last_error"]
        self._save_state(state)
        self._store.set(PENDING_ENTRY_KEY, {})

        self._audit(
            "live_entry",
            f"bought {mint} with {spent:.6f} SOL",
            {"mint": mint, "spent_sol": spent, "txid": position["txid_buy"]},
            0.9,
            0.7,
        )
        self._notify("trading:live_entry", message=f"LIVE entry {position['symbol']} ({spent:.4f} SOL)")
        return position

    async def _recover_pending_entry(self, now: float) -> None:
        """Recover a buy that was submitted but not persisted (crash safety)."""
        pending = self._store.get(PENDING_ENTRY_KEY)
        if not isinstance(pending, dict) or not pending.get("mint"):
            return
        if self._adapter is None or getattr(self._adapter, "is_dry_run", True):
            return
        mint = str(pending.get("mint") or "")
        balance = await self._adapter.get_token_balance(mint)
        if balance > 0:
            entry_sol = _as_float(pending.get("amount_sol")) or self._config.position_sol
            price = _as_float(pending.get("price_usd")) or 0.0
            position = {
                "mint": mint,
                "symbol": str(pending.get("symbol") or "?"),
                "entry_sol": round(entry_sol, 9),
                "entry_price_usd": price,
                "peak_price_usd": price,
                "last_price_usd": price,
                "remaining_pct": 100.0,
                "token_raw": int(balance),
                "entered_at": _as_float(pending.get("submitted_at")) or now,
                "txid_buy": "",
                "unverified": False,
                "recovered": True,
                "last_error": "",
            }
            positions = self._positions()
            if not any(p.get("mint") == mint for p in positions):
                positions.append(position)
                self._save_positions(positions)
            self._audit("live_entry_recovered", f"recovered pending entry {mint}", {"mint": mint}, 0.8, 0.8)
        else:
            logger.warning("pending live entry for %s produced no tokens; clearing", mint)
        self._store.set(PENDING_ENTRY_KEY, {})

    # ── exits ─────────────────────────────────────────────────────────

    async def track(self, now: float | None = None) -> list[dict[str, Any]]:
        """Price open positions and execute staged exits. Runs even disarmed."""
        moment = now if now is not None else self._now()
        positions = self._positions()
        if not positions:
            return []
        if self._adapter is None or getattr(self._adapter, "is_dry_run", True):
            return []

        closed: list[dict[str, Any]] = []
        for position in list(positions):
            mint = str(position.get("mint") or "")
            entry_price = _as_float(position.get("entry_price_usd")) or 0.0
            peak = _as_float(position.get("peak_price_usd")) or entry_price
            entered_at = _as_float(position.get("entered_at")) or moment
            remaining = _as_float(position.get("remaining_pct")) or 0.0
            if not mint or entry_price <= 0 or remaining <= 0:
                continue
            try:
                price = await self._price_fn(mint)
            except Exception as exc:
                logger.warning("live price feed failed for %s: %s", mint, exc)
                continue
            current = _as_float(price)
            if current is None or current <= 0:
                continue

            position["last_price_usd"] = current
            age_min = max(0.0, (moment - entered_at) / 60.0)
            decision = evaluate_exit(
                entry_price,
                peak,
                current,
                age_min,
                ExitConfig(max_hold_minutes=self._config.max_hold_minutes),
            )
            position["peak_price_usd"] = decision.new_high
            if decision.action == "hold" or decision.sell_pct <= 0:
                continue

            sol_before = await self._sol_balance()
            try:
                result = await self._adapter.sell(mint, decision.sell_pct)
            except Exception as exc:
                position["last_error"] = f"sell failed: {exc}"
                self._set_error(position["last_error"])
                self._save_positions(positions)
                continue
            if not isinstance(result, dict) or result.get("status") != "executed":
                position["last_error"] = f"sell not executed: {result}"
                self._set_error(position["last_error"])
                self._save_positions(positions)
                continue

            sol_after = await self._sol_balance()
            received = 0.0
            if sol_before is not None and sol_after is not None and sol_after > sol_before:
                received = (sol_after - sol_before) / 1_000_000_000
            if received <= 0:
                quoted = _as_float(result.get("out_amount"))
                if quoted is not None and quoted > 0:
                    received = quoted / 1_000_000_000

            fraction_sold = (decision.sell_pct / 100.0) * (remaining / 100.0)
            entry_sol = _as_float(position.get("entry_sol")) or 0.0
            cost_basis = entry_sol * fraction_sold
            pnl = received - cost_basis
            trade = {
                "mint": mint,
                "symbol": str(position.get("symbol") or "?"),
                "entry_sol": round(cost_basis, 9),
                "exit_sol": round(received, 9),
                "pnl_sol": round(pnl, 9),
                "exit_reason": f"{decision.action}: {decision.reason}",
                "opened_at": entered_at,
                "closed_at": moment,
                "txid_sell": str(result.get("txid") or ""),
                "entry_price_usd": entry_price,
                "exit_price_usd": current,
                "remaining_after_pct": round(max(0.0, remaining - decision.sell_pct * remaining / 100.0), 6),
            }
            self._append_trade(trade)
            closed.append(trade)

            position["remaining_pct"] = trade["remaining_after_pct"]
            position["realized_sol"] = round((_as_float(position.get("realized_sol")) or 0.0) + pnl, 9)
            position["last_txid"] = trade["txid_sell"]
            position["last_exit_at"] = moment
            position["last_error"] = ""
            if position["remaining_pct"] <= 0.01:
                positions = [p for p in positions if p.get("mint") != mint]
            self._save_positions(positions)

            self._audit(
                "live_exit",
                f"sold {decision.sell_pct:.0f}% of {mint}: {decision.reason}",
                {"mint": mint, "pnl_sol": pnl, "txid": trade["txid_sell"], "reason": decision.reason},
                0.9,
                0.6,
            )
            self._notify(
                "trading:live_exit",
                message=f"LIVE exit {trade['symbol']} {decision.sell_pct:.0f}% pnl {pnl:+.4f} SOL",
            )
            self._report_to_copy(pnl)

        self._save_positions(positions)
        self._check_daily_loss(moment)
        return closed

    # ── risk ──────────────────────────────────────────────────────────

    def _daily_loss_breached(self, now: float) -> bool:
        limit = self._config.daily_loss_limit_sol
        if limit <= 0:
            return False
        return self._today_realized(self._ledger(), now) <= -abs(limit)

    def _check_daily_loss(self, now: float) -> None:
        if not self._daily_loss_breached(now):
            return
        reason = f"daily loss limit {self._config.daily_loss_limit_sol} SOL breached"
        if self._kill.is_active(KillSwitchLevel.STRATEGY, STRATEGY_ID):
            return
        self._trip_kill(reason)

    def _trip_kill(self, reason: str) -> None:
        self._kill.activate(
            KillSwitchLevel.STRATEGY,
            reason,
            trigger="memecoin_live",
            affected=[STRATEGY_ID],
            triggered_by="system",
        )
        state = self._state()
        state["armed"] = False
        state["disarmed_reason"] = reason
        state["disarmed_at"] = self._now()
        self._save_state(state)
        self._audit("live_kill_switch", reason, {}, 0.99, 1.0)
        self._notify("trading:live_kill_switch", message=f"LIVE KILL SWITCH: {reason}")

    # ── helpers ───────────────────────────────────────────────────────

    async def _sol_balance(self) -> int | None:
        if self._adapter is None or not hasattr(self._adapter, "get_sol_balance"):
            return None
        try:
            return int(await self._adapter.get_sol_balance())
        except Exception:
            return None

    async def _await_token_balance(self, mint: str) -> int:
        deadline = time.monotonic() + self._confirm_timeout_seconds
        while True:
            try:
                balance = int(await self._adapter.get_token_balance(mint))
            except Exception:
                balance = 0
            if balance > 0:
                return balance
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return 0
            await asyncio.sleep(min(1.0, max(0.05, remaining)))

    def _set_error(self, message: str) -> None:
        logger.warning("memecoin live: %s", message)
        state = self._state()
        state["last_error"] = message[:500]
        self._save_state(state)

    def _report_to_copy(self, pnl_sol: float) -> None:
        if pnl_sol == 0:
            return
        try:
            from cores.trading.copy_trading import CopyTradingEngine

            CopyTradingEngine().record_daily_pnl("memecoin_live", pnl_sol)
        except Exception as exc:
            logger.warning("could not report live PnL to copy engine: %s", exc)

    def _audit(
        self,
        action: str,
        reason: str,
        data: dict[str, Any],
        confidence: float = 0.0,
        risk_score: float = 0.0,
    ) -> None:
        try:
            from cores.decision_journal.journal import log_decision

            log_decision(
                app_id="trading",
                agent_id="memecoin_live",
                action=action,
                reason=reason,
                data_snapshot=data,
                confidence=confidence,
                risk_score=risk_score,
            )
        except Exception:
            logger.debug("decision journal unavailable for %s", action)

    def _notify(self, event_type: str, **payload: Any) -> None:
        try:
            from cores.notifications.emit import emit

            emit(event_type, **payload)
        except Exception:
            logger.debug("notification %s skipped", event_type)


def _default_sol_price() -> float | None:
    from cores.trading.pumpfun_launch import get_sol_price_usd

    return get_sol_price_usd()


def build_live_adapter(store: TradingStore | None = None) -> Any | None:
    """Build a live MemecoinAdapter only when a Solana wallet is available."""
    from cores.investment.adapters.memecoin_adapter import MemecoinAdapter
    from cores.trading.dex.solana_wallet import SolanaWallet

    wallet = SolanaWallet()
    if not wallet.is_loaded:
        try:
            from cores.identity_vault import IdentityVault

            wallet.load_from_vault(IdentityVault())
        except Exception as exc:
            logger.debug("IdentityVault wallet load skipped: %s", exc)
    if not wallet.is_loaded:
        return None

    trading = TradingConfig.from_env()
    state = (store or TradingStore()).get(STATE_KEY) or {}
    return MemecoinAdapter(
        {
            "allow_live": True,
            "private_key": wallet.export_private_key_b58(),
            "rpc_url": trading.rpc_url,
            "max_slippage_bps": int(state.get("slippage_bps") or 300),
            "min_liquidity_usd": float(state.get("min_liquidity_usd") or MIN_LIQUIDITY_USD),
        }
    )


def build_live_loop(
    store: TradingStore | None = None,
    adapter: Any = _UNSET,
    kill: KillSwitchManager | None = None,
) -> MemecoinLiveLoop:
    """Production factory. ``adapter=None`` means status/disarm without a wallet."""
    s = store or TradingStore()
    resolved = build_live_adapter(s) if adapter is _UNSET else adapter
    config = LiveLimits.from_dict(s.get(STATE_KEY) or {})
    return MemecoinLiveLoop(
        adapter=resolved,
        store=s,
        kill=kill or KillSwitchManager(store=s),
        config=config,
    )


def live_status_payload(store: TradingStore | None = None, now: float | None = None) -> dict[str, Any]:
    """Read-only status for the API/UI. Never builds a wallet or touches the network."""
    from cores.trading.dex.solana_wallet import SolanaWallet

    s = store or TradingStore()
    try:
        trading = TradingConfig.from_env()
    except ValueError as exc:
        return {"strategy_id": STRATEGY_ID, "mode": "invalid", "is_live": False, "config_error": str(exc)}
    config = LiveLimits.from_dict(s.get(STATE_KEY) or {})
    loop = MemecoinLiveLoop(
        adapter=None,
        store=s,
        kill=KillSwitchManager(store=s),
        config=config,
        trading_config=trading,
    )
    status = loop.status(now=now)
    wallet_configured = SolanaWallet().is_loaded
    status["wallet_configured"] = wallet_configured
    status["adapter_ready"] = bool(wallet_configured and trading.is_live)
    status["needs_attention"] = bool(status["open_positions"] and not status["adapter_ready"])
    return status


def unlock_kill_switch(level: str, approval_token: str, store: TradingStore | None = None) -> dict[str, Any]:
    """Human-only unlock. Requires a single-use token scoped to the level."""
    normalized = (level or "").strip().lower()
    if normalized not in ("strategy", "global"):
        raise LiveArmError("level must be 'strategy' or 'global'", 400)
    if not consume_approval(approval_token, APPROVAL_UNLOCK_ACTION, normalized, WORKSPACE):
        raise LiveArmError("invalid, expired or already-used unlock token", 403)
    s = store or TradingStore()
    kill = KillSwitchManager(store=s)
    if normalized == "global":
        ok = kill.deactivate(KillSwitchLevel.GLOBAL, triggered_by="human")
    else:
        ok = kill.deactivate(KillSwitchLevel.STRATEGY, affected=[STRATEGY_ID], triggered_by="human")
    return {"success": ok, "level": normalized, "strategy_id": STRATEGY_ID, "status": kill.get_status()}


async def run_live_cycle_once(
    store: TradingStore | None = None,
    adapter: Any = _UNSET,
    kill: KillSwitchManager | None = None,
    config: LiveLimits | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    """One locked cycle. Exits always run; entries only while armed."""
    s = store or TradingStore()
    lock_path = Path(s.data_dir) / LOCK_NAME
    try:
        acquire_lock(lock_path)
    except LockHeldError as exc:
        return {"success": False, "status": "locked", "error": str(exc), "entered": [], "closed": []}
    try:
        resolved = build_live_adapter(s) if adapter is _UNSET else adapter
        loop = MemecoinLiveLoop(
            adapter=resolved,
            store=s,
            kill=kill or KillSwitchManager(store=s),
            config=config or LiveLimits.from_dict(s.get(STATE_KEY) or {}),
        )
        return await loop.run_once(now=now)
    finally:
        release_lock(lock_path)


def run_live_cycle_sync(
    store: TradingStore | None = None,
    adapter: Any = _UNSET,
    kill: KillSwitchManager | None = None,
    config: LiveLimits | None = None,
) -> dict[str, Any]:
    """Scheduler entrypoint. Never raises; returns a JSON-serializable summary."""
    import threading

    try:
        return asyncio.run(run_live_cycle_once(store=store, adapter=adapter, kill=kill, config=config))
    except RuntimeError:
        result: dict[str, Any] = {}

        def _run() -> None:
            try:
                result.update(asyncio.run(run_live_cycle_once(store=store, adapter=adapter, kill=kill, config=config)))
            except Exception as exc:  # pragma: no cover - defensive scheduler guard
                logger.error("live cycle failed: %s", exc)
                result.update({"success": False, "status": "error", "error": str(exc)})

        worker = threading.Thread(target=_run, name="memecoin-live", daemon=True)
        worker.start()
        worker.join(timeout=600)
        if worker.is_alive():
            return {"success": False, "status": "timeout", "entered": [], "closed": []}
        return result
    except Exception as exc:
        logger.error("live cycle failed: %s", exc)
        return {"success": False, "status": "error", "error": str(exc), "entered": [], "closed": []}
