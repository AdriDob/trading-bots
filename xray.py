"""Memecoin X-Ray: score 0-100 + veredicto para un mint de Solana.

Extiende (sin reemplazar) `memecoin_filters.evaluate_pair` y los fetchers de
`memecoin_adapter`: agrega lo que esos no tienen — score agregado con tiers,
detección de bundles (clustering por financiador vía RPC), retención del
dev, comparación de clones y un hook de sentimiento (X/Telegram) que queda
neutral-sin-datos hasta tener keys/canales (fail-open documentado, no inventa).

Fuentes (todas keyless): DexScreener, RugCheck, Solana RPC público.
Cero dependencias nuevas (httpx ya es del stack).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("ownex.trading.xray")

DEXSCREENER_TOKEN_URL = "https://api.dexscreener.com/latest/dex/tokens/{mint}"
DEXSCREENER_SEARCH_URL = "https://api.dexscreener.com/latest/dex/search?q={query}"
RUGCHECK_REPORT_URL = "https://api.rugcheck.xyz/v1/tokens/{mint}/report"
RPC_URLS: tuple[str, ...] = (
    "https://api.mainnet-beta.solana.com",
    "https://solana-rpc.publicnode.com",
)
HTTP_TIMEOUT_S = 15.0
MAX_TOP_WALLETS = 10
BUNDLE_MIN_LINKED = 5
BUNDLE_MIN_SUPPLY_PCT = 10.0
DEV_RETENTION_PCT = 10.0
TOO_EARLY_MINUTES = 30.0
CORE_SYMBOLS = frozenset({"SOL", "USDC", "USDT", "JUP", "JITOSOL", "MSOL", "WSOL"})

VERDICT_CLEAN = "CLEAN"
VERDICT_OK = "OK"
VERDICT_RISKY = "RISKY"
VERDICT_DANGER = "DANGER"
VERDICT_TOO_EARLY = "TOO_EARLY"


@dataclass(frozen=True, slots=True)
class XRayCheck:
    """Resultado de un check individual (puntos sobre su máximo)."""

    name: str
    points: float
    max_points: float
    detail: str = ""
    skipped: bool = False


@dataclass(frozen=True, slots=True)
class XRayVerdict:
    """Veredicto agregado de un scan."""

    mint: str
    symbol: str
    score: float
    verdict: str
    reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    checks: tuple[XRayCheck, ...] = ()
    scanned_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "mint": self.mint,
            "symbol": self.symbol,
            "score": self.score,
            "verdict": self.verdict,
            "reasons": list(self.reasons),
            "warnings": list(self.warnings),
            "checks": [
                {
                    "name": c.name,
                    "points": c.points,
                    "max_points": c.max_points,
                    "detail": c.detail,
                    "skipped": c.skipped,
                }
                for c in self.checks
            ],
            "scanned_at": self.scanned_at,
        }


def _num(value: Any) -> float | None:
    try:
        if value is None or isinstance(value, bool):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def tier_for_score(score: float, age_minutes: float | None) -> str:
    """Tier del veredicto. Pares <30m son TOO_EARLY (no hay historia)."""
    if age_minutes is not None and age_minutes < TOO_EARLY_MINUTES:
        return VERDICT_TOO_EARLY
    if score >= 80:
        return VERDICT_CLEAN
    if score >= 60:
        return VERDICT_OK
    if score >= 40:
        return VERDICT_RISKY
    return VERDICT_DANGER


def group_by_funder(wallets: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Agrupa wallets por financiador común (detección de bundles).

    Cada wallet aporta `address` + `funder` (quien le mandó el primer SOL o
    pagó su primera tx). Wallets sin funder van cada una a su propio grupo.
    Pura — testeable sin red.
    """
    groups: dict[str, list[str]] = {}
    for w in wallets:
        addr = str(w.get("address", ""))
        if not addr:
            continue
        funder = str(w.get("funder") or f"self:{addr}")
        groups.setdefault(funder, []).append(addr)
    return groups


def score_bundles(
    wallets: list[dict[str, Any]],
    total_supply_pct: float | None,
    max_points: float = 12.0,
) -> XRayCheck:
    """Penaliza clusters de wallets con el mismo financiador."""
    if not wallets:
        return XRayCheck("bundles", max_points, max_points, "sin datos de holders (neutral)", skipped=False)
    groups = group_by_funder(wallets)
    biggest = max((len(v) for v in groups.values()), default=1)
    pct = total_supply_pct or 0.0
    if biggest >= BUNDLE_MIN_LINKED or pct >= BUNDLE_MIN_SUPPLY_PCT:
        return XRayCheck(
            "bundles",
            0.0,
            max_points,
            f"bundled: {biggest} wallets vinculadas ({pct:.1f}% supply en cluster)",
        )
    if biggest >= 3:
        half = round(max_points / 2, 1)
        return XRayCheck("bundles", half, max_points, f"{biggest} wallets comparten financiador (sospechoso)")
    return XRayCheck("bundles", max_points, max_points, f"{len(wallets)} wallets, sin cluster dominante")


def compare_clones(
    symbol: str,
    own_fdv: float | None,
    own_listed_ms: float | None,
    candidates: list[dict[str, Any]],
    max_points: float = 6.0,
) -> XRayCheck:
    """¿Este ticker es el original o un clon? Pura — sin red.

    Un candidato solo cuenta como 'original' si listó no después que el par
    escaneado (clones frescos con FDV inflado no invalidan al real).
    """
    if not symbol or symbol.upper() in CORE_SYMBOLS:
        if not symbol:
            return XRayCheck("clones", max_points, max_points, "sin ticker (neutral)")
        return XRayCheck("clones", max_points, max_points, "asset core (checks de ticker omitidos)")
    same = [c for c in candidates if str(c.get("symbol", "")).upper() == symbol.upper()]
    if not same:
        return XRayCheck("clones", max_points, max_points, "sin clones con el mismo ticker")
    originals = 0
    for c in same:
        c_fdv = _num(c.get("fdv"))
        c_listed = _num(c.get("listed_ms"))
        if c_fdv is None or own_fdv is None:
            continue
        if c_fdv <= own_fdv:
            continue
        if c_listed is not None and own_listed_ms is not None and c_listed > own_listed_ms:
            continue  # clon más joven con FDV inflado: no cuenta
        originals += 1
    if originals > 0:
        return XRayCheck("clones", 0.0, max_points, f"posible clon: {originals} token(s) mayor(es) y anterior(es)")
    return XRayCheck("clones", max_points, max_points, f"{len(same)} mismo-ticker, este parece el original")


def score_market(pair: dict[str, Any] | None, max_points: float = 20.0) -> XRayCheck:
    """Liquidez, volumen, edad y balance buys/sells desde DexScreener."""
    from cores.trading.strategies.memecoin_filters import normalize_pair

    if not pair:
        return XRayCheck("market", 0.0, max_points, "sin datos de mercado (fail closed)")
    p = normalize_pair(pair)
    liq = _num(p.get("liquidity_usd")) or 0.0
    vol = _num(p.get("volume_h24")) or 0.0
    pts = 0.0
    notes: list[str] = []
    if liq >= 100_000:
        pts += 8.0
    elif liq >= 10_000:
        pts += 4.0
        notes.append(f"liq baja ${liq:,.0f}")
    else:
        notes.append(f"liq crítica ${liq:,.0f}")
    if vol >= 100_000:
        pts += 7.0
    elif vol >= 10_000:
        pts += 3.0
        notes.append(f"vol bajo ${vol:,.0f}")
    else:
        notes.append(f"vol crítico ${vol:,.0f}")
    txns = p.get("txns") or {}
    buys = _num((txns.get("h24") or {}).get("buys")) or 0.0
    sells = _num((txns.get("h24") or {}).get("sells")) or 0.0
    total = buys + sells
    if total > 0 and sells > 0 and buys / sells >= 1.0:
        pts += 5.0
    elif total > 0:
        pts += 2.0
        notes.append(f"presión vendedora {buys:.0f}/{sells:.0f}")
    else:
        notes.append("sin txns h24")
    detail = "; ".join(notes) if notes else f"liq ${liq:,.0f} vol ${vol:,.0f}"
    return XRayCheck("market", round(pts, 1), max_points, detail)


def score_rug(rug: dict[str, Any] | None, max_points: float = 20.0) -> XRayCheck:
    """Reporte RugCheck: dangers restan todo, warns la mitad."""
    if not rug or rug.get("error"):
        return XRayCheck("rug", 0.0, max_points, "sin reporte RugCheck (fail closed)")
    dangers = 0
    warns = 0
    for risk in rug.get("risks") or []:
        if not isinstance(risk, dict):
            continue
        level = str(risk.get("level", "")).lower()
        if level == "danger":
            dangers += 1
        elif level in ("warn", "warning"):
            warns += 1
    if dangers > 0:
        return XRayCheck("rug", 0.0, max_points, f"{dangers} danger(s) en RugCheck")
    if warns > 0:
        half = round(max_points / 2, 1)
        return XRayCheck("rug", half, max_points, f"{warns} warn(s) en RugCheck")
    return XRayCheck("rug", max_points, max_points, "sin riesgos en RugCheck")


def score_holders(rug: dict[str, Any] | None, max_points: float = 12.0) -> XRayCheck:
    """Concentración top-10 + conteo de holders (vía RugCheck)."""
    if not rug or rug.get("error"):
        return XRayCheck("holders", 0.0, max_points, "sin datos de holders (fail closed)")
    holders = _num(rug.get("holder_count"))
    top10 = _num(rug.get("top_10_pct"))
    if holders is None or top10 is None or holders <= 0:
        return XRayCheck("holders", 0.0, max_points, "holders/top10 desconocidos (fail closed)")
    if top10 > 30.0:
        return XRayCheck("holders", 0.0, max_points, f"top10 {top10:.1f}% (concentrado)")
    pts = max_points
    detail = f"top10 {top10:.1f}% holders {holders:.0f}"
    if holders < 50:
        pts = round(max_points / 2, 1)
        detail += " (pocos holders)"
    return XRayCheck("holders", pts, max_points, detail)


def score_authorities(rug: dict[str, Any] | None, max_points: float = 12.0) -> XRayCheck:
    """Mint/freeze authority deshabilitadas (vía RugCheck flags)."""
    if not rug or rug.get("error"):
        return XRayCheck("authorities", 0.0, max_points, "sin flags de authority (fail closed)")
    problems: list[str] = []
    if rug.get("mint_disabled") is not True:
        problems.append("mint activa")
    if rug.get("freeze_disabled") is not True:
        problems.append("freeze activa")
    if problems:
        return XRayCheck("authorities", 0.0, max_points, ", ".join(problems))
    return XRayCheck("authorities", max_points, max_points, "mint+freeze deshabilitadas")


def score_dev(top_funder_pct: float | None, max_points: float = 8.0) -> XRayCheck:
    """Retención del dev/principal financiador: >10% es bandera roja."""
    if top_funder_pct is None:
        return XRayCheck("dev", max_points, max_points, "sin datos del dev (neutral)")
    if top_funder_pct > DEV_RETENTION_PCT:
        return XRayCheck("dev", 0.0, max_points, f"dev retiene {top_funder_pct:.1f}%")
    return XRayCheck("dev", max_points, max_points, f"dev retiene {top_funder_pct:.1f}%")


def score_age(age_minutes: float | None, max_points: float = 5.0) -> XRayCheck:
    """Edad del par: <30m es TOO_EARLY (se resuelve en el tier)."""
    if age_minutes is None:
        return XRayCheck("age", 0.0, max_points, "edad desconocida")
    if age_minutes < TOO_EARLY_MINUTES:
        return XRayCheck("age", 0.0, max_points, f"par de {age_minutes:.0f}m (<30m)")
    if age_minutes < 24 * 60:
        return XRayCheck("age", round(max_points / 2, 1), max_points, f"par de {age_minutes:.0f}m (<24h)")
    return XRayCheck("age", max_points, max_points, f"par de {age_minutes / 60:.0f}h")


def score_depth(pair: dict[str, Any] | None, max_points: float = 5.0) -> XRayCheck:
    """Profundidad: volumen vs liquidez (churn saludable)."""
    if not pair:
        return XRayCheck("depth", 0.0, max_points, "sin datos (fail closed)")
    try:
        from cores.trading.strategies.memecoin_filters import normalize_pair

        p = normalize_pair(pair)
    except Exception:
        return XRayCheck("depth", 0.0, max_points, "par inválido")
    liq = _num(p.get("liquidity_usd")) or 0.0
    vol = _num(p.get("volume_h24")) or 0.0
    if liq <= 0:
        return XRayCheck("depth", 0.0, max_points, "sin liquidez medible")
    ratio = vol / liq
    if ratio >= 1.0:
        return XRayCheck("depth", max_points, max_points, f"vol/liq {ratio:.1f}x")
    if ratio >= 0.2:
        return XRayCheck("depth", round(max_points / 2, 1), max_points, f"vol/liq {ratio:.1f}x (fino)")
    return XRayCheck("depth", 0.0, max_points, f"vol/liq {ratio:.1f}x (ilíquido)")


def score_sentiment(max_points: float = 0.0) -> XRayCheck:
    """Hook de sentimiento X/Telegram: neutral hasta tener keys/canales.

    No inventa señal: sin XAI_API_KEY ni canales trackeados, skipped con
    peso 0. El wiring futuro no cambia el score histórico.
    """
    return XRayCheck("sentiment", 0.0, max_points, "sin fuentes de sentimiento (neutral)", skipped=True)


def aggregate(checks: list[XRayCheck], mint: str, symbol: str, age_minutes: float | None) -> XRayVerdict:
    """Suma puntos → score 0-100 → tier. Pura."""
    earned = sum(c.points for c in checks)
    total = sum(c.max_points for c in checks)
    score = round((earned / total * 100.0) if total > 0 else 0.0, 1)
    verdict = tier_for_score(score, age_minutes)
    reasons = tuple(f"{c.name}: {c.detail}" for c in checks if c.points <= 0 and not c.skipped)
    warnings = tuple(f"{c.name}: {c.detail}" for c in checks if 0 < c.points < c.max_points)
    return XRayVerdict(
        mint=mint,
        symbol=symbol,
        score=score,
        verdict=verdict,
        reasons=reasons,
        warnings=warnings,
        checks=tuple(checks),
        scanned_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )


async def _http_get_json(url: str) -> dict[str, Any] | None:
    import httpx

    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_S) as client:
            resp = await client.get(url, headers={"User-Agent": "ownex-xray/1.0"})
            if resp.status_code != 200:
                return None
            data = resp.json()
            return data if isinstance(data, dict) else None
    except Exception as exc:
        logger.warning("xray GET %s failed: %s", url, exc)
        return None


async def _rpc(method: str, params: list[Any]) -> Any | None:
    import httpx

    payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    for base in RPC_URLS:
        try:
            async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_S) as client:
                resp = await client.post(base, json=payload)
                if resp.status_code != 200:
                    continue
                data = resp.json()
                if isinstance(data, dict) and "result" in data:
                    return data["result"]
        except Exception as exc:
            logger.warning("xray RPC %s @ %s failed: %s", method, base, exc)
    return None


async def fetch_pair(mint: str) -> dict[str, Any] | None:
    """Mejor par de DexScreener para el mint (por liquidez)."""
    data = await _http_get_json(DEXSCREENER_TOKEN_URL.format(mint=mint))
    if not data:
        return None
    pairs = data.get("pairs") or []
    if not pairs:
        return None
    best: dict[str, Any] | None = None
    best_liq = -1.0
    for p in pairs:
        if not isinstance(p, dict):
            continue
        liq = _num((p.get("liquidity") or {}).get("usd")) or 0.0
        if liq > best_liq:
            best_liq = liq
            best = p
    return best


async def fetch_rug(mint: str) -> dict[str, Any] | None:
    """Reporte RugCheck normalizado al shape que lee memecoin_filters."""
    data = await _http_get_json(RUGCHECK_REPORT_URL.format(mint=mint))
    if not data:
        return None
    return {
        "holder_count": data.get("holderCount", 0),
        "top_10_pct": data.get("top10HolderPercent", 0),
        "liquidity_locked": data.get("liquidityLocked", False),
        "mint_disabled": data.get("mintDisabled", False),
        "freeze_disabled": data.get("freezeDisabled", False),
        "score": data.get("score", 0),
        "risks": data.get("risks", []),
    }


async def fetch_top_wallets(mint: str, limit: int = MAX_TOP_WALLETS) -> list[dict[str, Any]]:
    """Top holders vía RPC + financiador de su primera tx (bounded, depth 1)."""
    largest = await _rpc("getTokenLargestAccounts", [mint, {"commitment": "confirmed"}])
    if not largest or not isinstance(largest, dict):
        return []
    accounts = (largest.get("value") or [])[:limit]
    out: list[dict[str, Any]] = []
    for acc in accounts:
        addr = acc.get("address", "")
        if not addr:
            continue
        out.append({"address": addr, "amount": acc.get("amount", "0"), "funder": await _first_funder(addr)})
    return out


async def _first_funder(address: str) -> str | None:
    """Financiador: firmante de la primera tx de la wallet (1 sig, bounded)."""
    sigs = await _rpc("getSignaturesForAddress", [address, {"limit": 1}])
    if not sigs:
        return None
    sig = sigs[-1].get("signature", "") if isinstance(sigs, list) else ""
    if not sig:
        return None
    tx = await _rpc("getTransaction", [sig, {"maxSupportedTransactionVersion": 0}])
    if not isinstance(tx, dict):
        return None
    try:
        signers = tx["transaction"]["message"]["accountKeys"]
        first = signers[0] if signers else None
        if isinstance(first, dict):
            return str(first.get("pubkey", ""))
        return str(first) if first else None
    except (KeyError, TypeError, IndexError):
        return None


async def fetch_clones(symbol: str) -> list[dict[str, Any]]:
    """Candidatos con el mismo ticker vía DexScreener search."""
    import urllib.parse

    if not symbol:
        return []
    data = await _http_get_json(DEXSCREENER_SEARCH_URL.format(query=urllib.parse.quote(symbol)))
    if not data:
        return []
    out: list[dict[str, Any]] = []
    for p in data.get("pairs") or []:
        if not isinstance(p, dict):
            continue
        base = p.get("baseToken") or {}
        fdv = _num(p.get("fdv"))
        out.append({"symbol": base.get("symbol", ""), "fdv": fdv, "listed_ms": _num(p.get("pairCreatedAt"))})
    return out


async def score_token(mint: str) -> XRayVerdict:
    """Orquesta los 10 checks y devuelve el veredicto. Nunca levanta."""
    import asyncio

    mint = (mint or "").strip()
    if not mint:
        return aggregate(
            [XRayCheck("input", 0.0, 100.0, "mint vacío")],
            mint,
            "",
            None,
        )
    try:
        pair, rug, clones_data = await asyncio.gather(
            fetch_pair(mint),
            fetch_rug(mint),
            _clones_for_pair(mint),
        )
    except Exception as exc:
        logger.warning("xray fetch failed for %s: %s", mint, exc)
        pair, rug, clones_data = None, None, ([], None, None)

    symbol = ""
    age_minutes: float | None = None
    if pair:
        try:
            from cores.trading.strategies.memecoin_filters import normalize_pair

            p = normalize_pair(pair)
            symbol = str(p.get("symbol", ""))
            created = _num(p.get("pair_created_at"))
            if created is not None:
                age_minutes = (time.time() * 1000.0 - created) / 60_000.0
        except Exception:
            pass

    wallets: list[dict[str, Any]] = []
    if rug and not rug.get("error"):
        try:
            wallets = await fetch_top_wallets(mint)
        except Exception as exc:
            logger.warning("xray wallets failed for %s: %s", mint, exc)

    clone_list, own_fdv, own_listed = clones_data
    top_pct: float | None = None
    if wallets:
        try:
            total = sum(int(float(w.get("amount", 0) or 0)) for w in wallets)
            if total > 0:
                groups = group_by_funder(wallets)
                biggest_amt = 0
                for addrs in groups.values():
                    amt = sum(int(float(next(w["amount"] for w in wallets if w["address"] == a) or 0)) for a in addrs)
                    biggest_amt = max(biggest_amt, amt)
                top_pct = biggest_amt / total * 100.0
        except (ValueError, TypeError, StopIteration):
            top_pct = None

    checks = [
        score_market(pair),
        score_rug(rug),
        score_holders(rug),
        score_authorities(rug),
        score_bundles(wallets, top_pct),
        score_dev(top_pct),
        compare_clones(symbol, own_fdv, own_listed, clone_list),
        score_depth(pair),
        score_age(age_minutes),
        score_sentiment(),
    ]
    return aggregate(checks, mint, symbol, age_minutes)


async def _clones_for_pair(mint: str) -> tuple[list[dict[str, Any]], float | None, float | None]:
    """Clones + FDV/edad propios para comparar. Todo fail-soft."""
    try:
        pair = await fetch_pair(mint)
        if not pair:
            return [], None, None
        from cores.trading.strategies.memecoin_filters import normalize_pair

        p = normalize_pair(pair)
        symbol = str(p.get("symbol", ""))
        own_fdv = _num(pair.get("fdv"))
        own_listed = _num(p.get("pair_created_at"))
        clones = await fetch_clones(symbol) if symbol else []
        return clones, own_fdv, own_listed
    except Exception as exc:
        logger.warning("xray clones failed for %s: %s", mint, exc)
        return [], None, None
