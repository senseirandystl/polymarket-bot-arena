"""FastAPI dashboard backend for the Bot Arena."""

import json
import logging
import secrets
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, Request, Depends, HTTPException, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
import config
import credentials_store
import db
import learning
from arena.market_utils import is_5min_market

security = HTTPBasic()

DASHBOARD_USER = "admin"
DASHBOARD_PASS = "Thor"

logger = logging.getLogger(__name__)


def verify_auth(credentials: HTTPBasicCredentials = Depends(security)):
    correct_user = secrets.compare_digest(credentials.username, DASHBOARD_USER)
    correct_pass = secrets.compare_digest(credentials.password, DASHBOARD_PASS)
    if not (correct_user and correct_pass):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


app = FastAPI(title="Polymarket Bot Arena Dashboard", dependencies=[Depends(verify_auth)])

# Balance cache: key -> {"balance": float, "fetched_at": float}
_balance_cache = {}
BALANCE_CACHE_TTL = 60  # seconds


def _fetch_slot_balance(api_key):
    """Fetch balance for a Simmer account."""
    import requests
    try:
        headers = {"Authorization": f"Bearer {api_key}"}
        resp = requests.get(
            f"{config.SIMMER_BASE_URL}/api/sdk/agents/me",
            headers=headers, timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            return data.get("balance")
    except Exception:
        pass
    return None


def get_bot_balance(slot_name, bot_keys, trading_mode="paper"):
    """Get cached or fresh balance for a bot slot. Live bots show Polymarket USDC balance."""
    cache_key = "polymarket_live" if trading_mode == "live" else slot_name
    now = time.time()
    cached = _balance_cache.get(cache_key)
    if cached and (now - cached["fetched_at"]) < BALANCE_CACHE_TTL:
        return cached["balance"], trading_mode == "live"

    if trading_mode == "live":
        # Read Polymarket L2 credentials from the encrypted store.
        api_key = credentials_store.get_credential("polymarket_api_key")
        api_secret = credentials_store.get_credential("polymarket_api_secret")
        api_passphrase = credentials_store.get_credential("polymarket_api_passphrase")
        signer_address = credentials_store.get_credential("polymarket_signer_address")
        if not (api_key and api_secret and api_passphrase and signer_address):
            balance = None
        else:
            try:
                import hmac as _hmac
                import hashlib as _hashlib
                import base64 as _base64
                import requests as _req
                # Build HMAC signature for Level 2 auth
                # signature_type=1 = POLY_PROXY (queries funder/proxy wallet balance)
                ts = str(int(time.time()))
                msg = ts + "GET" + "/balance-allowance"
                secret_bytes = _base64.urlsafe_b64decode(api_secret)
                sig = _base64.urlsafe_b64encode(
                    _hmac.new(secret_bytes, msg.encode(), _hashlib.sha256).digest()
                ).decode()
                headers = {
                    "POLY_ADDRESS": signer_address,
                    "POLY_SIGNATURE": sig,
                    "POLY_TIMESTAMP": ts,
                    "POLY_API_KEY": api_key,
                    "POLY_PASSPHRASE": api_passphrase,
                }
                resp = _req.get(
                    "https://clob.polymarket.com/balance-allowance"
                    "?asset_type=COLLATERAL&signature_type=1",
                    headers=headers, timeout=10,
                )
                data = resp.json()
                raw = data.get("balance", "0") if isinstance(data, dict) else "0"
                balance = int(raw) / 1e6
            except Exception:
                balance = None
        _balance_cache[cache_key] = {"balance": balance, "fetched_at": now}
        return balance, True

    api_key = bot_keys.get(slot_name)
    if not api_key:
        return None, False
    balance = _fetch_slot_balance(api_key)
    _balance_cache[cache_key] = {"balance": balance, "fetched_at": now}
    return balance, False


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).parent / "index.html"
    return html_path.read_text()


@app.get("/api/status")
async def get_status():
    warnings: list = []
    if not config.is_credential_configured("simmer_api_key"):
        warnings.append({
            "level": "error",
            "category": "credentials",
            "message": (
                "No Simmer API key configured. The arena is running but bots cannot "
                "trade. Open the Settings tab to enter your Simmer API key."
            ),
        })
    elif not config.is_credential_configured("simmer_bot_keys"):
        # Single-account mode is fine \u2014 only flag if zero per-bot keys AND no
        # default key. Actually we already checked the default key above, so
        # this branch is informational only (multi-account mode is optional).
        pass
    if config.get_current_mode() == "live":
        pm_missing = [
            name for name in (
                "polymarket_api_key", "polymarket_api_secret",
                "polymarket_api_passphrase", "polymarket_signer_address",
            )
            if not config.is_credential_configured(name)
        ]
        if pm_missing:
            warnings.append({
                "level": "error",
                "category": "credentials",
                "message": (
                    f"Live trading mode but missing Polymarket credentials: "
                    f"{', '.join(pm_missing)}. Switch to paper mode or add them in Settings."
                ),
            })

    return {
        "mode": config.get_current_mode(),
        "venue": config.get_venue(),
        "max_position": config.get_max_position(),
        "max_daily_loss_per_bot": config.get_max_daily_loss_per_bot(),
        "max_daily_loss_total": config.get_max_daily_loss_total(),
        "warnings": warnings,
    }


@app.post("/api/mode")
async def set_mode(request: Request):
    body = await request.json()
    mode = body.get("mode")
    if mode not in ("paper", "live"):
        return JSONResponse({"error": "Mode must be 'paper' or 'live'"}, 400)
    config.set_trading_mode(mode)
    return {"mode": config.get_current_mode()}


@app.post("/api/bots/{bot_name}/mode")
async def set_bot_mode(bot_name: str, request: Request):
    body = await request.json()
    mode = body.get("mode")
    if mode not in ("paper", "live"):
        return JSONResponse({"error": "Mode must be 'paper' or 'live'"}, 400)
    db.set_bot_mode(bot_name, mode)
    return {"bot_name": bot_name, "trading_mode": mode}


def _to_et(dt_utc):
    """Convert UTC datetime to ET, handling EST/EDT without requiring tzdata."""
    try:
        from zoneinfo import ZoneInfo
        return dt_utc.astimezone(ZoneInfo("America/New_York"))
    except Exception:
        from datetime import timedelta, timezone, datetime
        year = dt_utc.year
        mar1 = datetime(year, 3, 1, tzinfo=timezone.utc)
        dst_start = mar1 + timedelta(days=(6 - mar1.weekday()) % 7) + timedelta(weeks=1, hours=7)
        nov1 = datetime(year, 11, 1, tzinfo=timezone.utc)
        dst_end = nov1 + timedelta(days=(6 - nov1.weekday()) % 7) + timedelta(hours=6)
        return dt_utc + timedelta(hours=-4 if dst_start <= dt_utc < dst_end else -5)


@app.get("/api/markets")
async def get_markets():
    """Get active BTC fast markets as {current, upcoming_count, upcoming}.

    Queries two sources: SDK (upcoming tagged markets) + public API (live markets).
    The SDK endpoint drops live markets from its results once they enter their window.

    Failure modes (network timeout, JSON decode, malformed market entries) are
    caught per-source so a single Simmer-side hiccup can never blank the BTC
    5-Min Markets card. Worst case (both sources unreachable) returns HTTP 200
    with empty-shape payload + ``warnings`` field; the frontend degrades to a
    "Simmer unavailable" hint rather than a 500 stack trace in the browser.
    """
    import requests as req
    from datetime import datetime, timezone

    api_key = credentials_store.get_credential("simmer_api_key")
    if not api_key:
        return JSONResponse({"current": None, "upcoming_count": 0, "upcoming": [],
                             "error": "No Simmer API key configured"})
    headers = {"Authorization": f"Bearer {api_key}"}

    markets_list: list = []
    warnings: list = []

    # Source 1: SDK upcoming markets.
    # Per-source try/except: if this side fails, source 2 can still populate
    # the card. Narrow ``requests.RequestException`` instead of bare
    # ``Exception`` so genuine programming bugs (AttributeError, etc.) still
    # escalate instead of being silently suppressed.
    try:
        r1 = req.get(
            f"{config.SIMMER_BASE_URL}/api/sdk/markets",
            headers=headers,
            params={"limit": 50, "tags": "fast-5m"},
            timeout=10,
        )
        if r1.status_code == 200:
            d = r1.json()
            # Three-shape dispatch. Simmer has shipped ``null`` bodies during
            # outages; we still return 200 with an empty card (the user's
            # accept criterion), but we now log "shape not recognized" so the
            # empty card is distinguishable from a real "no markets" case in
            # the launchd dashboard log.
            if isinstance(d, dict):
                for m in d.get("markets") or []:
                    # Defensive: any non-dict entry (str/None/number) is skipped.
                    if isinstance(m, dict) and m.get("id"):
                        markets_list.append(m)
            elif isinstance(d, list):
                for m in d:
                    if isinstance(m, dict) and m.get("id"):
                        markets_list.append(m)
            else:
                warnings.append("sdk_markets: response shape not recognized")
        else:
            warnings.append(f"sdk_markets: HTTP {r1.status_code}")
    except req.RequestException as e:
        warnings.append(f"sdk_markets: {type(e).__name__}")
        logger.warning("markets: SDK source1 unreachable: %s", e)
    except (ValueError, TypeError) as e:
        # JSONDecodeError is a subclass of ValueError; covers non-JSON bodies.
        warnings.append(f"sdk_markets: JSON decode error: {e}")
        logger.warning("markets: SDK source1 bad JSON: %s", e)

    # Source 2: public endpoint for currently-live markets.
    try:
        r2 = req.get(
            f"{config.SIMMER_BASE_URL}/api/markets",
            headers=headers,
            params={"limit": 20},
            timeout=10,
        )
        if r2.status_code == 200:
            d = r2.json()
            seen_ids = {m.get("id") for m in markets_list if isinstance(m, dict)}
            # Same three-shape dispatch as Source 1 -- a parsed-null body
            # (no exception) is logged so operators can tell empty-card from
            # truly-no-markets.
            if isinstance(d, dict):
                for m in d.get("markets") or []:
                    mid = m.get("id") if isinstance(m, dict) else None
                    if mid and mid not in seen_ids:
                        markets_list.append(m)
                        seen_ids.add(mid)
            elif isinstance(d, list):
                for m in d:
                    mid = m.get("id") if isinstance(m, dict) else None
                    if mid and mid not in seen_ids:
                        markets_list.append(m)
                        seen_ids.add(mid)
            else:
                warnings.append("public_markets: response shape not recognized")
        else:
            warnings.append(f"public_markets: HTTP {r2.status_code}")
    except req.RequestException as e:
        warnings.append(f"public_markets: {type(e).__name__}")
        logger.warning("markets: SDK source2 unreachable: %s", e)
    except (ValueError, TypeError) as e:
        warnings.append(f"public_markets: JSON decode error: {e}")
        logger.warning("markets: SDK source2 bad JSON: %s", e)

    now_utc = datetime.now(timezone.utc)
    btc_markets: list = []
    for m in markets_list:
        try:
            q = (m.get("question") or "").lower()
            tags = m.get("tags") or []
            is_btc_updown = (
                ("bitcoin" in q or "btc" in q)
                and ("up or down" in q or "up/down" in q)
            ) or ("fast-5m" in tags and ("bitcoin" in q or "btc" in q))
            if not is_btc_updown:
                continue

            # Only ever surface 5-minute windows -- a 15-min BTC up/down
            # market must never appear on the card (see the July-2026
            # next-day 8:15-8:30 regression).
            if not is_5min_market(m.get("question", "") or ""):
                continue

            resolves_at_str = m.get("resolves_at")
            time_remaining = None
            if resolves_at_str:
                try:
                    rs = resolves_at_str.replace("Z", "+00:00").replace(" ", "T")
                    resolves_at = datetime.fromisoformat(rs)
                    if resolves_at.tzinfo is None:
                        resolves_at = resolves_at.replace(tzinfo=timezone.utc)
                    time_remaining = (resolves_at - now_utc).total_seconds()
                except (ValueError, TypeError, AttributeError):
                    # Malformed timestamp -- leave time_remaining=None so the
                    # market still shows up in the upcoming list rather than
                    # being silently dropped, and the soonest-fallback filter
                    # is robust to None entries (it coalesces to 999999).
                    time_remaining = None

            if time_remaining is not None and time_remaining < 0:
                continue

            btc_markets.append({
                "id": m.get("id"),
                "question": m.get("question"),
                "current_price": m.get("current_price"),
                "resolves_at": m.get("resolves_at"),
                "time_remaining_seconds": time_remaining,
                # Current iff the REAL resolves_at timestamp puts us inside
                # its 5-min window (0 < remaining <= 300). Never keyed off ET
                # time-of-day, so a future-dated window whose clock time
                # straddles "now" can't masquerade as current.
                "is_current_window": (
                    time_remaining is not None and 0 < time_remaining <= 300
                ),
                "url": m.get("url"),
            })
        except (TypeError, AttributeError, KeyError) as e:
            # Malformed market dict from Simmer -- skip the one entry, don't
            # blank the card. Logged at debug so it's traceable without
            # spamming normal-operation logs.
            logger.debug("markets: skipping malformed market id=%s: %s",
                         m.get("id") if isinstance(m, dict) else None, e)
            continue

    btc_markets.sort(key=lambda x: x.get("time_remaining_seconds") or 999999)

    # Priority 1: market whose question window contains now.
    current = next((m for m in btc_markets if m["is_current_window"]), None)
    # Priority 2: soonest market closing within 20 min.
    if not current:
        soon = [m for m in btc_markets
                if (m.get("time_remaining_seconds") or 999999) <= 1200]
        current = soon[0] if soon else None

    upcoming = [m for m in btc_markets if m is not current]

    # Attach the trades bots have actually placed on the current + next
    # market so the Overview cards can show who traded each window. Only the
    # two visible markets are queried (cheap point lookups by market_id).
    def _attach_trades(market):
        if not market or not market.get("id"):
            return
        with db.get_conn() as conn:
            rows = conn.execute(
                "SELECT bot_name, side, amount, shares_bought, outcome, pnl, created_at "
                "FROM trades WHERE market_id=? ORDER BY created_at ASC",
                (market["id"],),
            ).fetchall()
        market["trades"] = [dict(r) for r in rows]

    _attach_trades(current)
    if upcoming:
        _attach_trades(upcoming[0])

    payload = {
        "current": current,
        "next": upcoming[0] if upcoming else None,
        "upcoming_count": len(upcoming),
        "upcoming": upcoming,
    }
    if warnings:
        payload["warnings"] = warnings
    return JSONResponse(payload)


@app.get("/api/maker-status")
async def get_maker_status():
    """Return the latest snapshot the arena's secondary-bot tick published.

    Powers the Maker Section card on the Overview tab.  Always returns
    200 -- if the arena hasn't published yet (no API key, fresh startup,
    or the on_cycle_complete hook fired before a discovery scan finished),
    mode is set to IDLE so the card renders gracefully.
    """
    from datetime import datetime, timezone
    raw = db.get_arena_state("maker_state")
    if not raw:
        # Never been written -- field is just absent from arena_state. Return
        # an IDLE-shaped payload so the frontend doesn't have to special-case
        # the empty-key path.
        return JSONResponse({
            "mode": "IDLE",
            "market_id": None,
            "market_question": None,
            "time_remaining_seconds": None,
            "resolves_at": None,
            "target_count": 0,
            "updated_at": None,
            "staleness_seconds": None,
        })
    try:
        snap = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return JSONResponse({"error": "Corrupt maker_state JSON"}, status_code=500)

    # Compute staleness (seconds since the arena last published a snapshot).
    # The frontend uses this to dim the card when the arena process is
    # clearly dead or stuck.
    staleness = None
    updated = snap.get("updated_at")
    if updated:
        try:
            ts = updated
            # The arena writes ISO with a trailing 'Z'; Python's
            # fromisoformat() needs an explicit +HH:MM offset.
            if ts.endswith("Z"):
                ts = ts[:-1] + "+00:00"
            ts_dt = datetime.fromisoformat(ts)
            if ts_dt.tzinfo is None:
                ts_dt = ts_dt.replace(tzinfo=timezone.utc)
            # Clamp to >= 0 so the dashboard's isStale check correctly reports
            # "fresh" instead of going negative when the dashboard process's
            # wall clock is BEHIND the arena's (clock skew across the two
            # processes).  Without this, _isMakerStale returns false for
            # skew-caused negative staleness and the card shows LIVE / PRE-WINDOW
            # against a snapshot the arena already considers ancient.
            # Clamp staleness on both ends so clock skew (forward or backward)
            # doesn't mislead the operator:
            #   * max(0.0, ...)                        defense against the
            #                                          dashboard process's
            #                                          clock being BEHIND
            #                                          the arena's (negative
            #                                          raw delta).
            #   * min(STALENESS_DISPLAY_MAX_SEC, ...) defense against the
            #                                          dashboard clock being
            #                                          AHEAD (inflated
            #                                          staleness).  Cap is
            #                                          5min -- still triggers
            #                                          the STALE display
            #                                          (>120s), but caps the
            #                                          values shown to the
            #                                          operator at a sane
            #                                          range.
            _raw_staleness = (datetime.now(timezone.utc) - ts_dt).total_seconds()
            staleness = max(0.0, min(config.STALENESS_DISPLAY_MAX_SEC, _raw_staleness))
        except (ValueError, TypeError, AttributeError):
            # Narrowed from ``Exception``: genuinely unexpected exceptions
            # would otherwise silently mark every dashboard card as
            # permanently stale, hiding the real bug.  Genuine parse errors
            # (None / bogus ``updated_at`` values) cleanly map to None.
            staleness = None
    # Raw signed clock delta between the dashboard's wall clock and the
    # arena's ``updated_at``.  Operator-visible diagnostic: pre-clamp
    # value so they see ACTUAL skew (e.g. +5s if dashboard is ahead of
    # arena, -2s if behind) instead of a sanitised display number.
    # A drift distinctly different from ``staleness_seconds`` (which is
    # clamped) is the operator's signal that clock skew is occurring.
    snap["clock_drift_seconds"] = _raw_staleness
    snap["staleness_seconds"] = staleness
    snap.setdefault("mode", "IDLE")
    snap.setdefault("target_count", 0)
    return JSONResponse(snap)


@app.get("/api/overview")
async def get_overview():
    stats = db.get_dashboard_stats()
    active_bots = db.get_active_bots()
    return JSONResponse({
        "stats": stats,
        "active_bots": active_bots,
        "mode": config.get_current_mode(),
    })


@app.get("/api/bots")
async def get_bots():
    active = db.get_active_bots()

    # Load bot keys for balance fetching
    bot_keys = {}
    try:
        with open(config.SIMMER_BOT_KEYS_PATH) as f:
            bot_keys = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    result = []
    for i, bot_cfg in enumerate(active):
        # Parse params JSON string if needed
        cfg = dict(bot_cfg)
        if isinstance(cfg.get("params"), str):
            try:
                cfg["params"] = json.loads(cfg["params"])
            except (json.JSONDecodeError, TypeError):
                pass
        trading_mode = db.get_bot_mode(cfg["bot_name"])
        is_live = trading_mode == "live"

        # Live bots: show all-time live-only stats; paper bots: show 12h/24h paper stats
        if is_live:
            perf_12h = db.get_bot_performance(cfg["bot_name"], hours=None, mode="live")
            perf_24h = perf_12h  # same — all live trades
        else:
            perf_12h = db.get_bot_performance(cfg["bot_name"], hours=12)
            perf_24h = db.get_bot_performance(cfg["bot_name"], hours=24)

        trades = db.get_bot_trades(cfg["bot_name"], limit=10)
        # Count pending (unresolved) trades so dashboard shows activity
        with db.get_conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as c FROM trades WHERE bot_name=? AND outcome IS NULL",
                (cfg["bot_name"],)
            ).fetchone()
            pending_count = dict(row)["c"]

        # Balance: Polymarket USDC for live bots, Simmer SIM for paper bots
        slot_name = f"slot_{i}"
        balance, balance_is_live = get_bot_balance(slot_name, bot_keys, trading_mode)

        # For live bots, include the trading key address so dashboard can show where to deposit
        trading_key_address = None
        if trading_mode == "live":
            trading_key_address = credentials_store.get_credential("polymarket_signer_address")

        result.append({
            "config": cfg,
            "performance_12h": perf_12h,
            "performance_24h": perf_24h,
            "recent_trades": trades,
            "pending_trades": pending_count,
            "trading_mode": trading_mode,
            "balance": balance,
            "balance_is_live": balance_is_live,
            "trading_key_address": trading_key_address,
        })
    return JSONResponse(result)


@app.get("/api/evolution")
async def get_evolution():
    history = db.get_evolution_history(limit=20)
    for h in history:
        for key in ("survivors", "replaced", "new_bots", "rankings"):
            if isinstance(h.get(key), str):
                h[key] = json.loads(h[key])
    return JSONResponse(history)


@app.get("/api/trades")
async def get_trades(bot: str = None, limit: int = 50):
    if bot:
        return JSONResponse(db.get_bot_trades(bot, limit=limit))
    with db.get_conn() as conn:
        # Sort PENDING trades first (newest activity), then resolved by
        # recency. Previously resolved sorted first and pending last, so with
        # hundreds of resolved trades the handful of pending rows were pushed
        # past the LIMIT and never appeared in Recent Trades — even though the
        # Active Bots cards counted them. Surfacing pending at the top keeps
        # the two views reconciled. COALESCE(resolved_at, created_at) orders
        # pending by placement time and resolved by settlement time.
        # Show every trade, including 1h-stale-expired (outcome='expired',
        # pnl=0) rows; the "phantom pnl=0" filter no longer applies.
        rows = conn.execute(
            """SELECT * FROM trades
               ORDER BY
                   CASE WHEN outcome IS NULL THEN 0 ELSE 1 END,
                   COALESCE(resolved_at, created_at) DESC
               LIMIT ?""", (limit,)
        ).fetchall()
        return JSONResponse([dict(r) for r in rows])


@app.get("/api/copytrading")
async def get_copytrading():
    wallets = db.list_copy_wallets()
    result = []
    for w in wallets:
        bot_name = f"copy-{w['label']}"
        with db.get_conn() as conn:
            perf = conn.execute(
                """SELECT COUNT(*) as total,
                          SUM(CASE WHEN outcome='win' THEN 1 ELSE 0 END) as wins,
                          SUM(CASE WHEN outcome='loss' THEN 1 ELSE 0 END) as losses,
                          ROUND(SUM(pnl), 2) as pnl
                   FROM trades WHERE bot_name=?""",
                (bot_name,),
            ).fetchone()
            recent = conn.execute(
                """SELECT side, amount, market_question, outcome, pnl, created_at
                   FROM trades WHERE bot_name=? ORDER BY created_at DESC LIMIT 5""",
                (bot_name,),
            ).fetchall()
        p = dict(perf)
        total = (p.get("wins") or 0) + (p.get("losses") or 0)
        result.append({
            "wallet": w["address"],
            "label": w["label"],
            "mode": w.get("trading_mode", "paper"),
            "total_trades": p.get("total") or 0,
            "resolved_trades": total,
            "win_rate": (p.get("wins") or 0) / total if total > 0 else None,
            "pnl": p.get("pnl") or 0,
            "recent_trades": [dict(r) for r in recent],
        })
    return JSONResponse(result)


@app.get("/api/earnings")
async def get_earnings():
    with db.get_conn() as conn:
        # Bucket by ET calendar date (created_at is stored UTC). Grouping in
        # Python keeps the day boundary DST-correct and consistent with the
        # ET-anchored "Today" stats on the Overview tab.
        resolved = conn.execute(
            "SELECT created_at, pnl FROM trades WHERE outcome IN ('win', 'loss')"
        ).fetchall()
        buckets: dict = {}
        for r in resolved:
            day = db.utc_to_et_date(r["created_at"])
            b = buckets.setdefault(day, {"pnl": 0.0, "trades": 0, "wins": 0})
            pnl = r["pnl"] or 0
            b["pnl"] += pnl
            b["trades"] += 1
            if pnl > 0:
                b["wins"] += 1
        daily = [
            {"day": day, "pnl": round(v["pnl"], 2), "trades": v["trades"], "wins": v["wins"]}
            for day, v in sorted(buckets.items(), reverse=True)[:30]
        ]

        best = conn.execute(
            "SELECT * FROM trades WHERE pnl IS NOT NULL ORDER BY pnl DESC LIMIT 5"
        ).fetchall()

        worst = conn.execute(
            "SELECT * FROM trades WHERE pnl IS NOT NULL ORDER BY pnl ASC LIMIT 5"
        ).fetchall()

        return JSONResponse({
            "daily": [dict(r) for r in daily],
            "best_trades": [dict(r) for r in best],
            "worst_trades": [dict(r) for r in worst],
        })


@app.get("/api/learning")
async def get_learning():
    active = db.get_active_bots()
    result = {}
    for bot_cfg in active:
        name = bot_cfg["bot_name"]
        result[name] = learning.get_bot_learning_summary(name)
    return JSONResponse(result)


# ---------------------------------------------------------------------------
# Credentials store: Settings tab + warning banner wiring
# ---------------------------------------------------------------------------


@app.get("/api/credentials/status")
async def credentials_status_endpoint():
    """Return the list of credential fields and which ones are currently set.

    Powers both the Settings tab form and the dashboard warning banner.
    """
    return JSONResponse({
        "credentials": credentials_store.credentials_status(),
        "store_file": str(credentials_store.CREDENTIALS_FILE),
        "key_file": str(credentials_store.CREDENTIALS_KEY_FILE),
    })


@app.post("/api/credentials/save")
async def credentials_save(request: Request):
    """Save a subset of credential fields. Empty strings clear the field.

    Keys outside the allowed set are silently dropped.
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Body must be JSON"}, 400)
    if not isinstance(body, dict):
        return JSONResponse({"error": "Body must be a JSON object"}, 400)

    valid_keys = {s["key"] for s in credentials_store.credentials_status()}
    # Multi-account Simmer keys come in as a JSON string keyed by slot.
    updates = {}
    for k, v in body.items():
        if k in valid_keys:
            updates[k] = v
    if not updates:
        return JSONResponse({"error": "No recognized credential fields in request"}, 400)

    try:
        config.set_credentials(updates)
    except Exception as e:
        return JSONResponse({"error": f"Save failed: {e}"}, 500)

    # Invalidate the balance cache so live bots pick up new Polymarket credentials
    # on the next refresh instead of waiting out the TTL.
    _balance_cache.clear()

    return JSONResponse({
        "saved": list(updates.keys()),
        "credentials": credentials_store.credentials_status(),
    })


@app.post("/api/credentials/test")
async def credentials_test(request: Request):
    """Test connectivity with currently-configured credentials.

    Body: {"which": "simmer" | "polymarket" | "all"} (default "all").
    Returns key-by-key results without persisting anything.
    """
    try:
        body = await request.json()
    except Exception:
        body = {}
    which = (body or {}).get("which", "all")
    results = {}

    if which in ("simmer", "all"):
        api_key = credentials_store.get_credential("simmer_api_key")
        if not api_key:
            results["simmer"] = {"ok": False, "error": "Simmer API key not configured"}
        else:
            try:
                import requests as _req
                resp = _req.get(
                    f"{config.SIMMER_BASE_URL}/api/sdk/agents/me",
                    headers={"Authorization": f"Bearer {api_key}"},
                    timeout=10,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    results["simmer"] = {
                        "ok": True,
                        "agent_name": data.get("name"),
                        "agent_id": data.get("agent_id"),
                        "balance": data.get("balance"),
                    }
                else:
                    results["simmer"] = {
                        "ok": False,
                        "error": f"HTTP {resp.status_code}: {resp.text[:200]}",
                    }
            except Exception as e:
                results["simmer"] = {"ok": False, "error": str(e)}

    if which in ("polymarket", "all"):
        pm_creds = {
            name: credentials_store.get_credential(name)
            for name in (
                "polymarket_api_key", "polymarket_api_secret",
                "polymarket_api_passphrase", "polymarket_signer_address",
            )
        }
        missing = [k for k, v in pm_creds.items() if not v]
        if missing:
            results["polymarket"] = {
                "ok": False,
                "error": f"Missing fields: {missing}",
            }
        else:
            try:
                import hmac as _hmac
                import hashlib as _hashlib
                import base64 as _base64
                import requests as _req
                ts = str(int(time.time()))
                msg = ts + "GET" + "/balance-allowance"
                secret_bytes = _base64.urlsafe_b64decode(pm_creds["polymarket_api_secret"])
                sig = _base64.urlsafe_b64encode(
                    _hmac.new(secret_bytes, msg.encode(), _hashlib.sha256).digest()
                ).decode()
                headers = {
                    "POLY_ADDRESS": pm_creds["polymarket_signer_address"],
                    "POLY_SIGNATURE": sig,
                    "POLY_TIMESTAMP": ts,
                    "POLY_API_KEY": pm_creds["polymarket_api_key"],
                    "POLY_PASSPHRASE": pm_creds["polymarket_api_passphrase"],
                }
                resp = _req.get(
                    "https://clob.polymarket.com/balance-allowance"
                    "?asset_type=COLLATERAL&signature_type=1",
                    headers=headers, timeout=10,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    raw = data.get("balance", "0") if isinstance(data, dict) else "0"
                    balance = int(raw) / 1e6
                    results["polymarket"] = {
                        "ok": True,
                        "balance_usdc": balance,
                        "signer_address": pm_creds["polymarket_signer_address"],
                    }
                else:
                    results["polymarket"] = {
                        "ok": False,
                        "error": f"HTTP {resp.status_code}: {resp.text[:200]}",
                    }
            except Exception as e:
                results["polymarket"] = {"ok": False, "error": str(e)}

    return JSONResponse(results)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.DASHBOARD_HOST, port=config.DASHBOARD_PORT)
