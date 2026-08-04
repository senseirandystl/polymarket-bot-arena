"""FastAPI dashboard backend for the Bot Arena."""

import json
import logging
import os
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

logger = logging.getLogger(__name__)

# Dashboard Basic-auth credentials — read from the environment so the secret is
# not hardcoded. Defaults preserve the historical local-dev values so nothing
# breaks on a fresh clone (the dashboard binds to localhost). Set DASHBOARD_USER
# / DASHBOARD_PASS in the environment (or the launchd plist) to override; the
# bin/arena probe reads the SAME env vars so the two stay in sync.
_DEFAULT_USER = "admin"
_DEFAULT_PASS = "Thor"
DASHBOARD_USER = os.environ.get("DASHBOARD_USER", _DEFAULT_USER)
DASHBOARD_PASS = os.environ.get("DASHBOARD_PASS", _DEFAULT_PASS)
if DASHBOARD_PASS == _DEFAULT_PASS:
    logger.warning(
        "Dashboard is using the DEFAULT password — set DASHBOARD_PASS in the "
        "environment before exposing the dashboard beyond localhost."
    )


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


@app.middleware("http")
async def _healthz(request: Request, call_next):
    """Unauthenticated liveness probe at ``/healthz`` for watchdogs/monitors.

    Runs as middleware so it bypasses the app-wide Basic-auth dependency (an
    external uptime check should not need credentials). Cheap: no DB, just the
    age of ``arena.log`` so a monitor can tell a HUNG arena (log not advancing)
    from a healthy one. ``stale`` flips true past ARENA_LOG_STALE_SEC; a watchdog
    (arena_watchdog.sh) can restart on it. Every other path falls through to the
    normal authenticated routes untouched.
    """
    if request.url.path == "/healthz":
        stale_after = int(os.environ.get(
            "ARENA_LOG_STALE_SEC",
            str(getattr(config, "ARENA_LOG_STALE_SEC", 300)),
        ))
        log_path = config.LOG_DIR / "arena.log"
        age = None
        try:
            age = time.time() - log_path.stat().st_mtime
        except OSError:
            pass
        # Lightweight kill-switch hint (no heavy imports if possible)
        killed = False
        try:
            ks = db.get_arena_state("kill_switch")
            killed = ks in ("1", "true", "on")
            if not killed:
                from pathlib import Path as _P
                kf = _P(getattr(config, "RISK_KILL_SWITCH_FILE", "") or "")
                killed = kf.is_file() and kf.read_text(errors="ignore").strip().lower() not in (
                    "0", "false", "off", "no", "clear", "disarm",
                )
        except Exception:
            pass
        stale = age is not None and age > stale_after
        return JSONResponse({
            "status": "degraded" if (stale or killed) else "ok",
            "ts": time.time(),
            "arena_log_age_sec": round(age, 1) if age is not None else None,
            "arena_log_stale": stale,
            "kill_switch": killed,
        })
    return await call_next(request)


# Balance cache: key -> {"balance": float, "fetched_at": float}
_balance_cache = {}
BALANCE_CACHE_TTL = 60  # seconds


def get_bot_balance(trading_mode="paper"):
    """Balance for a bot. Paper bots share the virtual bankroll; live bots show
    real Polymarket USDC. Returns ``(balance, is_live)``."""
    # Paper: all bots draw from ONE shared virtual USDC bankroll (set in the
    # dashboard Settings tab). Show the currently-available pool cash.
    if trading_mode != "live":
        return db.get_paper_available(), False

    cache_key = "polymarket_live"
    now = time.time()
    cached = _balance_cache.get(cache_key)
    if cached and (now - cached["fetched_at"]) < BALANCE_CACHE_TTL:
        return cached["balance"], True

    # Live: query the real Polymarket wallet USDC (paper already returned above).
    if True:  # noqa: SIM103 - kept for a clear indent level; paper returned early
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


@app.get("/", response_class=HTMLResponse)
def index():
    html_path = Path(__file__).parent / "index.html"
    return html_path.read_text()


@app.get("/api/ops")
def get_ops(_auth: str = Depends(verify_auth)):
    """Command-center snapshot: regime, risk, allocation, signals, health."""
    from arena.ops_snapshot import ops_snapshot
    try:
        return JSONResponse(ops_snapshot())
    except Exception as e:
        logger.exception("ops snapshot failed")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/health")
def get_health(_auth: str = Depends(verify_auth)):
    """Deep health checks + restart recommendations (authenticated)."""
    from arena.health import run_health_checks
    try:
        return JSONResponse(run_health_checks())
    except Exception as e:
        logger.exception("health check failed")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/alerts")
def get_alerts(_auth: str = Depends(verify_auth)):
    """Alert channel config, credential presence, and recent alert log."""
    from arena import alerts
    try:
        return JSONResponse({
            "config": alerts.load_config(),
            "channels": alerts.channel_status(),
            "log": alerts.get_alert_log(40),
            "event_types": [e for e in alerts.EVENT_TYPES if e != "test"],
            "event_labels": dict(getattr(alerts, "EVENT_LABELS", {})),
        })
    except Exception as e:
        logger.exception("alerts get failed")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/alerts")
async def update_alerts(request: Request, _auth: str = Depends(verify_auth)):
    """Update alert config and/or send a test message.

    Body: {enabled?, channels?, events?, min_level?, debounce_sec?, test?: bool|channel}
    """
    from arena import alerts
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        return JSONResponse({"error": "JSON object required"}, status_code=400)
    try:
        cfg = alerts.load_config()
        patch = {k: body[k] for k in (
            "enabled", "channels", "events", "min_level", "debounce_sec"
        ) if k in body}
        if patch:
            cfg = alerts.save_config({**cfg, **patch})
        test_result = None
        if body.get("test"):
            ch = body["test"] if isinstance(body["test"], str) else None
            test_result = alerts.send_test(ch)
        return JSONResponse({
            "success": True,
            "config": cfg,
            "channels": alerts.channel_status(),
            "test": test_result,
            "log": alerts.get_alert_log(20),
        })
    except Exception as e:
        logger.exception("alerts update failed")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/status")
def get_status():
    warnings: list = []
    # Paper mode needs no credentials \u2014 it simulates against public Polymarket
    # order books. Only live mode requires Polymarket CLOB credentials.
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
def get_markets():
    """Current + upcoming BTC 5-min markets, from Polymarket (Gamma + CLOB).

    The dashboard runs as its own process, so it can't read the arena's
    in-memory discovery snapshot — it does its own Polymarket discovery here,
    using the same helpers (``select_current_market`` keyed off the real
    ``resolves_at`` timestamp). No credentials needed; market data is public.
    """
    from datetime import datetime, timezone
    import polymarket_markets
    from arena.market_utils import (
        compute_time_remaining_seconds, is_5min_market, select_current_market,
    )

    now_utc = datetime.now(timezone.utc)
    btc_markets = []
    for m in polymarket_markets.discover_markets():
        if not is_5min_market(m.get("question", "") or ""):
            continue
        tr = compute_time_remaining_seconds(m, now_utc)
        if tr is not None and tr < 0:
            continue
        m["time_remaining_seconds"] = tr
        btc_markets.append(m)
    btc_markets.sort(key=lambda x: x.get("time_remaining_seconds") or 999999)

    # Current = the market whose real window contains now (0 < remaining <= 300).
    current = select_current_market(btc_markets, now_utc)
    if current is None:
        soon = [m for m in btc_markets
                if 0 < (m.get("time_remaining_seconds") or 999999) <= 1200]
        current = soon[0] if soon else None
    upcoming = [m for m in btc_markets if m is not current]

    # Fresh CLOB prices for the visible markets in ONE batch call (POST
    # /midpoints) — atomic snapshot, one round trip instead of a /midpoint GET
    # per market. Prices current + the next card; both UP and DOWN are set.
    polymarket_markets.price_markets([current, upcoming[0] if upcoming else None])

    def _shape(m, *, with_strike: bool = False):
        if not m:
            return None
        tr = m.get("time_remaining_seconds")
        yes = m.get("current_price")
        no = m.get("no_price")
        if no is None and yes is not None:
            no = round(1.0 - yes, 4)
        event_start = m.get("event_start_time") or m.get("eventStartTime")
        shaped = {
            "id": m.get("id"),
            "question": m.get("question"),
            "current_price": yes,                     # UP token mid (0-1)
            "no_price": no,                           # DOWN token mid (0-1)
            "resolves_at": m.get("resolves_at"),
            "event_start_time": event_start,
            "time_remaining_seconds": tr,
            "is_current_window": tr is not None and 0 < tr <= 300,
            "url": None,
            "strike": None,                           # price-to-beat (PM openPrice)
        }
        if with_strike and event_start and shaped["id"]:
            try:
                from signals.strike import get_strike_registry
                shaped["strike"] = get_strike_registry().get_strike(
                    shaped["id"], event_start,
                )
            except Exception:
                shaped["strike"] = None
        with db.get_conn() as conn:
            rows = conn.execute(
                "SELECT bot_name, side, amount, shares_bought, entry_price, "
                "outcome, pnl, created_at "
                "FROM trades WHERE market_id=? ORDER BY created_at ASC",
                (shaped["id"],),
            ).fetchall()
        shaped["trades"] = [dict(r) for r in rows]
        return shaped

    # Live BTC from arena price_feed_status (shared SQLite); fallback None.
    btc_price = None
    btc_stale = True
    try:
        import json as _json
        raw = db.get_arena_state("price_feed_status")
        pf = _json.loads(raw) if raw else {}
        btc = ((pf or {}).get("symbols") or {}).get("btc") or {}
        btc_price = btc.get("latest")
        btc_stale = bool(btc.get("stale") or (pf or {}).get("stale"))
    except Exception:
        pass

    cur_s = _shape(current, with_strike=True)
    upcoming_s = [_shape(m) for m in upcoming]
    return JSONResponse({
        "current": cur_s,
        "next": upcoming_s[0] if upcoming_s else None,
        "upcoming_count": len(upcoming_s),
        "upcoming": upcoming_s,
        "btc_price": btc_price,
        "btc_stale": btc_stale,
    })


@app.get("/api/price/{condition_id}")
def get_price(condition_id: str):
    """Fresh UP/DOWN prices for one market (fast poll for the market cards)."""
    import polymarket_markets
    prices = polymarket_markets.current_prices(condition_id)
    if not prices:
        return JSONResponse({"yes": None, "no": None})
    # Fall back to complement if one side's book is momentarily empty.
    yes = prices.get("yes")
    no = prices.get("no")
    if yes is not None and no is None:
        no = round(1.0 - yes, 4)
    if no is not None and yes is None:
        yes = round(1.0 - no, 4)
    return JSONResponse({"yes": yes, "no": no})


@app.get("/api/maker-status")
def get_maker_status():
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
def get_overview():
    stats = db.get_dashboard_stats()
    active_bots = db.get_active_bots()
    return JSONResponse({
        "stats": stats,
        "active_bots": active_bots,
        "mode": config.get_current_mode(),
        "paper_bankroll": db.get_paper_bankroll(),
        "paper_available": db.get_paper_available(),
    })


@app.get("/api/entry-buckets")
def get_entry_buckets(mode: str = "paper", hours: int = None):
    """ROI by entry-price bucket — reveals whether a high WR is bought at bad
    prices. ``breakeven_gap`` = win_rate − avg_entry (cents of edge over the
    break-even line; <0 is losing, ≥0.05 is healthy)."""
    return JSONResponse(db.get_entry_price_buckets(mode=mode, hours=hours))


@app.get("/api/hybrid-meta")
def get_hybrid_meta():
    """Hybrid meta-learner state (arena_state 'hybrid_meta'): per-sub online
    multipliers with per-regime-bucket records, and the last effective
    sub-strategy weights the ensemble actually used. Empty until the hybrid
    has run at least once on this DB."""
    raw = db.get_arena_state("hybrid_meta")
    try:
        return JSONResponse(json.loads(raw) if raw else {})
    except (json.JSONDecodeError, TypeError):
        return JSONResponse({})


@app.get("/api/skips")
def get_skips():
    """Skip-reason tally the arena persists (why it sat flat, not just what it
    traded). Empty until the arena process has flushed at least once."""
    raw = db.get_arena_state("skip_counts")
    try:
        return JSONResponse(json.loads(raw) if raw else {})
    except (json.JSONDecodeError, TypeError):
        return JSONResponse({})


@app.get("/api/settings/bankroll")
def get_bankroll(_auth: str = Depends(verify_auth)):
    return JSONResponse({
        "bankroll": db.get_paper_bankroll(),
        "available": db.get_paper_available(),
    })


@app.post("/api/settings/bankroll")
async def set_bankroll(request: Request, _auth: str = Depends(verify_auth)):
    """Top the shared paper pool up to the entered balance.

    The number the user enters becomes the new *available* shared balance: it
    tops the pool up to that figure while preserving trade history and open
    positions (see ``db.topup_paper_bankroll``). Entering $200 when the pool is
    at $45 sets available to $200.
    """
    body = await request.json()
    try:
        amount = float(body.get("amount"))
    except (TypeError, ValueError):
        return JSONResponse({"error": "amount must be a number"}, status_code=400)
    if amount < 0:
        return JSONResponse({"error": "amount must be non-negative"}, status_code=400)
    db.topup_paper_bankroll(amount)
    return JSONResponse({
        "success": True,
        "bankroll": db.get_paper_bankroll(),
        "available": db.get_paper_available(),
    })


@app.get("/api/settings/kelly")
def get_kelly(_auth: str = Depends(verify_auth)):
    return JSONResponse({"kelly_fraction": db.get_kelly_fraction()})


@app.post("/api/settings/kelly")
async def set_kelly(request: Request, _auth: str = Depends(verify_auth)):
    """Set the Kelly fraction used for bet sizing (0 < f <= 1).

    The arena reads it from the DB on a short cache, so edits take effect
    within seconds without a restart. 0.25 = quarter-Kelly (conservative);
    1.0 = full Kelly (growth-optimal only if model probabilities are exact).
    """
    body = await request.json()
    try:
        fraction = float(body.get("fraction"))
    except (TypeError, ValueError):
        return JSONResponse({"error": "fraction must be a number"}, status_code=400)
    try:
        db.set_kelly_fraction(fraction)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    return JSONResponse({"success": True, "kelly_fraction": db.get_kelly_fraction()})


@app.get("/api/portfolio")
def get_portfolio(_auth: str = Depends(verify_auth)):
    """Portfolio capital allocation state (weights, metrics, method, toggles).

    Weights sum to 1 across active bots when allocation is enabled; each bot
    Kelly-sizes against bankroll × weight. See arena/portfolio.py.
    """
    from arena import portfolio
    try:
        snap = portfolio.dashboard_snapshot()
    except Exception as e:
        logger.exception("portfolio snapshot failed")
        return JSONResponse({"error": str(e)}, status_code=500)
    return JSONResponse(snap)


@app.get("/api/risk")
def get_risk(_auth: str = Depends(verify_auth)):
    """Risk engine snapshot: limits, per-bot status, VaR, kill switch, events."""
    from arena import risk_engine
    try:
        return JSONResponse(risk_engine.dashboard_snapshot())
    except Exception as e:
        logger.exception("risk snapshot failed")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/risk")
async def update_risk(request: Request, _auth: str = Depends(verify_auth)):
    """Update risk engine settings / evaluate / pause-resume bots.

    Body fields (all optional):
      enabled: bool
      limits: {bot_daily_loss, portfolio_daily_loss, bot_max_drawdown, ...}
      evaluate: bool  — force full recompute
      pause_bot: str  — bot name to manually pause
      resume_bot: str
      reason: str     — reason for pause
    """
    from arena import risk_engine
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        return JSONResponse({"error": "JSON object required"}, status_code=400)
    try:
        state = risk_engine.load_state()
        if "enabled" in body:
            state = risk_engine.set_enabled(bool(body["enabled"]))
        if isinstance(body.get("limits"), dict):
            state = risk_engine.update_limits(body["limits"])
        if body.get("pause_bot"):
            risk_engine.pause_bot(
                str(body["pause_bot"]),
                reason=str(body.get("reason") or "manual_pause"),
            )
        if body.get("resume_bot"):
            risk_engine.resume_bot(str(body["resume_bot"]))
        if body.get("evaluate"):
            state = risk_engine.evaluate()
        else:
            state = risk_engine.load_state()
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:
        logger.exception("risk update failed")
        return JSONResponse({"error": str(e)}, status_code=500)
    return JSONResponse({"success": True, **state})


@app.post("/api/risk/kill-switch")
async def risk_kill_switch(request: Request, _auth: str = Depends(verify_auth)):
    """Arm or clear the global kill switch (halts all trading).

    Body: {armed: bool, reason?: str}
    Also mirrored to logs/KILL_SWITCH file for operator file-based control.
    """
    from arena import risk_engine
    try:
        body = await request.json()
    except Exception:
        body = {}
    armed = bool(body.get("armed"))
    reason = str(body.get("reason") or ("dashboard_arm" if armed else "dashboard_clear"))
    try:
        state = risk_engine.set_kill_switch(armed, reason=reason, source="dashboard")
    except Exception as e:
        logger.exception("kill switch failed")
        return JSONResponse({"error": str(e)}, status_code=500)
    return JSONResponse({
        "success": True,
        "kill_switch": state.get("kill_switch"),
        "killed": bool(state.get("kill_switch")) or risk_engine._file_kill_armed(),
        "kill_reason": state.get("kill_reason"),
        "kill_file": str(risk_engine.kill_switch_file_path()),
    })


@app.post("/api/portfolio")
async def update_portfolio(request: Request, _auth: str = Depends(verify_auth)):
    """Update portfolio allocation settings and/or force a rebalance.

    Body fields (all optional):
      enabled: bool
      method: equal | sharpe | expectancy | kelly_portfolio
      window_hours: float
      manual_overrides: {bot_name: weight}  (empty dict clears pins)
      merge_overrides: bool (default false — replace overrides)
      rebalance: bool (force rebalance now)
    """
    from arena import portfolio
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        return JSONResponse({"error": "JSON object required"}, status_code=400)

    try:
        state = portfolio.load_state()
        if "enabled" in body:
            state = portfolio.set_enabled(bool(body["enabled"]))
        if "method" in body and body["method"] is not None:
            state = portfolio.set_method(str(body["method"]))
        if "window_hours" in body and body["window_hours"] is not None:
            try:
                wh = float(body["window_hours"])
            except (TypeError, ValueError):
                return JSONResponse(
                    {"error": "window_hours must be a number"}, status_code=400)
            if wh < 1 or wh > 168:
                return JSONResponse(
                    {"error": "window_hours must be in [1, 168]"}, status_code=400)
            state = portfolio.load_state()
            state["window_hours"] = wh
            portfolio.save_state(state)
            state = portfolio.rebalance(force=True, reason="window_change")
        if "manual_overrides" in body:
            ov = body["manual_overrides"]
            if ov is None:
                ov = {}
            if not isinstance(ov, dict):
                return JSONResponse(
                    {"error": "manual_overrides must be an object"}, status_code=400)
            state = portfolio.set_manual_overrides(
                ov, merge=bool(body.get("merge_overrides")))
        if body.get("rebalance"):
            state = portfolio.rebalance(force=True, reason="manual")
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:
        logger.exception("portfolio update failed")
        return JSONResponse({"error": str(e)}, status_code=500)

    return JSONResponse({"success": True, **state})


@app.get("/api/learned-rules")
def get_learned_rules(_auth: str = Depends(verify_auth)):
    """Signal Lab: data-driven skip/go/continuous rules + skip-reason bandit.

    Mined from decision_events (walk-forward OOS gated). Read-only snapshot
    for the dashboard; mining runs in the arena evolution loop.
    """
    try:
        from arena.learned_rules import snapshot
        return JSONResponse(snapshot())
    except Exception as e:
        return JSONResponse({"error": str(e), "rules": [], "enabled": False})


@app.post("/api/learned-rules/mine")
def post_learned_rules_mine(_auth: str = Depends(verify_auth)):
    """Force a mine cycle (dashboard button). Safe: read decision_events only."""
    try:
        from arena.learned_rules import mine_and_update, snapshot
        mine_and_update()
        return JSONResponse({"success": True, **snapshot()})
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.get("/api/lane-proposals")
def get_lane_proposals(_auth: str = Depends(verify_auth)):
    """Signal Lab: candidate-lane proposals + approved overrides + last run.

    Proposals are filed by the offline harness (validate_signals --propose)
    when a kill-switched lane clears the promotion thresholds; approving one
    here activates the lane live via the DB override (no restart).
    """
    # Live lane monitor report (arena/lane_monitor.py): per-lane live
    # direction-accuracy for enabled overrides — the demotion half of the
    # pipeline. Written by the arena every LANE_MONITOR_INTERVAL_SEC.
    try:
        monitor = json.loads(db.get_arena_state("lane_monitor") or "{}")
    except (json.JSONDecodeError, TypeError):
        monitor = {}
    # Core-lane tuner report (arena/core_lane_tuner.py): per-(strategy, lane)
    # live accuracy + current/suggested drift/mom/strat weights.
    try:
        core_tuner = json.loads(db.get_arena_state("core_lane_tuner") or "{}")
    except (json.JSONDecodeError, TypeError):
        core_tuner = {}
    return JSONResponse({
        "proposals": db.get_lane_proposals(),
        "overrides": db.get_lane_overrides(),
        "last_run": db.get_latest_lane_run(),
        "monitor": monitor,
        "core_tuner": core_tuner,
        "auto_approve": db.get_auto_approve_lanes(),
    })


@app.get("/api/backtests")
def get_backtests(limit: int = 20, _auth: str = Depends(verify_auth)):
    """Signal Lab: recent offline backtest runs (backtest/ package).

    Runs are recorded by ``python -m backtest --to-db`` (or the
    backtest.run_backtest API); summaries carry expectancy/WR/PF/Sharpe/
    drawdown, per-regime splits and per-signal contribution. Read-only —
    the backtester never writes trade tables.
    """
    return JSONResponse({"runs": db.get_backtest_runs(limit=limit)})


@app.post("/api/lane-auto-approve")
async def set_lane_auto_approve(request: Request,
                                _auth: str = Depends(verify_auth)):
    """Flip the closed-loop auto-approve toggle (body: {"enabled": true}).

    ON: the promoter auto-approves candidate lanes that clear the LIVE bar.
    OFF: it only annotates proposals with live evidence for a human decision.
    """
    body = await request.json()
    enabled = bool(body.get("enabled"))
    db.set_auto_approve_lanes(enabled)
    return JSONResponse({"success": True, "auto_approve": enabled})


@app.get("/api/regime-map")
def get_regime_map():
    """Regime-discovery map: discovered/OOS-validated regimes, per-bot shrunk
    edges, the current cell, and whether Layer-3 conditioning is enabled.

    Read-only. Behind the app-wide Basic-auth dependency like every route
    except /healthz.
    """
    payload = db.get_regime_map()
    payload["conditioning_enabled"] = db.get_regime_conditioning()
    return JSONResponse(payload)


@app.post("/api/regime-conditioning")
async def set_regime_conditioning_toggle(request: Request,
                                         _auth: str = Depends(verify_auth)):
    """Flip regime-conditioning (body: {"enabled": true}).

    ON: the portfolio allocator + core-lane tuner tilt toward what works in the
    current validated regime. OFF: the map is still built/shown, but no
    controller acts on it.
    """
    body = await request.json()
    enabled = bool(body.get("enabled"))
    db.set_regime_conditioning(enabled)
    return JSONResponse({"success": True, "conditioning_enabled": enabled})


@app.post("/api/lane-proposals/{proposal_id}/decide")
async def decide_lane_proposal(proposal_id: int, request: Request,
                               _auth: str = Depends(verify_auth)):
    """Approve or deny a pending lane proposal (body: {"action": "approve"})."""
    body = await request.json()
    action = (body.get("action") or "").strip().lower()
    try:
        status_out = db.decide_lane_proposal(proposal_id, action)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    return JSONResponse({
        "success": True,
        "status": status_out,
        "overrides": db.get_lane_overrides(),
    })


@app.post("/api/lane-overrides/{lane}/disable")
async def disable_lane(lane: str, _auth: str = Depends(verify_auth)):
    """Safety hatch: switch an approved lane back off without a restart."""
    if db.disable_lane_override(lane):
        return JSONResponse({"success": True, "overrides": db.get_lane_overrides()})
    return JSONResponse({"error": f"no override for lane '{lane}'"}, status_code=404)


# --- Signal Lab: run the validation harness from the dashboard -------------
# The harness is network-heavy (minutes for 300 markets), so it runs as a
# detached subprocess; the UI polls /status until it exits, then reloads the
# proposals (the run itself lands in the DB via --propose). One run at a
# time — a second click while running is a 409.
_validation_run = {"proc": None, "started_at": None, "markets": None}
_VALIDATION_LOG = config.LOG_DIR / "lane_validation.log"


def _validation_running() -> bool:
    proc = _validation_run["proc"]
    return proc is not None and proc.poll() is None


@app.post("/api/lane-validation/run")
async def run_lane_validation(request: Request, _auth: str = Depends(verify_auth)):
    """Launch `validate_signals.py --markets N --propose` in the background."""
    import subprocess
    import sys as _sys
    from datetime import datetime, timezone

    if _validation_running():
        return JSONResponse({"error": "a validation run is already in progress"},
                            status_code=409)
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        markets = int(body.get("markets") or 300)
    except (TypeError, ValueError):
        return JSONResponse({"error": "markets must be a number"}, status_code=400)
    markets = max(50, min(1000, markets))

    repo_root = Path(__file__).resolve().parent.parent
    script = repo_root / "tools" / "validate_signals.py"
    log = open(_VALIDATION_LOG, "w")          # truncate: one run per log
    proc = subprocess.Popen(
        [_sys.executable, str(script), "--markets", str(markets), "--propose"],
        cwd=str(repo_root), stdout=log, stderr=subprocess.STDOUT)
    _validation_run.update({
        "proc": proc,
        "started_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "markets": markets,
    })
    return JSONResponse({"success": True, "markets": markets,
                         "started_at": _validation_run["started_at"]})


@app.get("/api/lane-validation/status")
def lane_validation_status(_auth: str = Depends(verify_auth)):
    """Poll target for the Signal Lab: running state + log tail + exit code."""
    proc = _validation_run["proc"]
    tail = ""
    try:
        if _VALIDATION_LOG.exists():
            tail = _VALIDATION_LOG.read_text()[-2000:]
    except OSError:
        pass
    return JSONResponse({
        "running": _validation_running(),
        "started_at": _validation_run["started_at"],
        "markets": _validation_run["markets"],
        "returncode": (None if proc is None else proc.poll()),
        "log_tail": tail,
    })


@app.post("/api/settings/soak-report")
async def run_soak_report(request: Request, _auth: str = Depends(verify_auth)):
    """Build a soak report and push it to configured notification channels.

    Body optional: ``{"notify": true}`` (default true). Returns the text body
    and notify result so the Settings UI can show a status line.
    """
    try:
        body = await request.json()
    except Exception:
        body = {}
    notify = body.get("notify", True)
    try:
        from tools.soak_report import build_report, format_text, notify as soak_notify
        report = build_report()
        text = format_text(report)
        result = {"success": True, "report_text": text,
                  "overall": report.get("overall")}
        if notify:
            result["notify"] = soak_notify(report)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.get("/api/bots")
def get_bots():
    active = db.get_active_bots()
    result = []
    for bot_cfg in active:
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

        # Balance: real wallet USDC for live bots, shared virtual bankroll for paper.
        balance, balance_is_live = get_bot_balance(trading_mode)

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
def get_evolution():
    """Evolution event log (survivors / replaced / new bots) + GA spawn lineage.

    ``evolution_events`` only rows when a bot is actually replaced; skipped GA
    cycles (everyone survived) live only in ``ga_generations``. We merge both
    so the Bots page shows a complete cycle timeline.
    """
    history = db.get_evolution_history(limit=40)
    by_cycle: dict = {}
    for h in history:
        for key in ("survivors", "replaced", "new_bots", "rankings"):
            if isinstance(h.get(key), str):
                try:
                    h[key] = json.loads(h[key])
                except Exception:
                    pass
        h.setdefault("spawned", [])
        h.setdefault("elites", [])
        h["ga_skipped"] = False
        by_cycle[h.get("cycle_number")] = h

    try:
        for g in db.get_ga_history(limit=50):
            cyc = g.get("cycle_number")
            report = g.get("report") or {}
            if not isinstance(report, dict):
                report = {}
            inds = report.get("individuals") or []
            rankings = [
                {
                    "name": i.get("name"),
                    "strategy_type": i.get("strategy_type"),
                    "generation": i.get("generation"),
                    "pnl": i.get("pnl"),
                    "win_rate": i.get("win_rate"),
                    "trades": i.get("trades"),
                    "be_gap": i.get("be_gap"),
                    "fitness": i.get("fitness"),
                    "components": i.get("components"),
                    "ranks": i.get("ranks"),
                    "status": i.get("status"),
                    "elite": i.get("elite"),
                    "lineage": i.get("lineage"),
                }
                for i in inds
            ]
            if cyc in by_cycle:
                h = by_cycle[cyc]
                h["spawned"] = report.get("spawned") or h.get("spawned") or []
                h["elites"] = report.get("elites") or h.get("elites") or []
                h["ga_skipped"] = bool(g.get("skipped") or report.get("skipped"))
                h["ga_reason"] = report.get("reason")
                h["best_fitness"] = g.get("best_fitness")
                if not h.get("rankings") and rankings:
                    h["rankings"] = rankings
                if not h.get("survivors"):
                    h["survivors"] = [
                        i.get("name") for i in inds
                        if i.get("status") in ("survivor", "immune", "elite_protected")
                        or i.get("elite")
                    ]
            else:
                by_cycle[cyc] = {
                    "id": None,
                    "cycle_number": cyc,
                    "survivors": [
                        i.get("name") for i in inds
                        if i.get("status") in ("survivor", "immune", "elite_protected")
                        or i.get("elite")
                    ],
                    "replaced": report.get("replaced") or [],
                    "new_bots": [s.get("name") for s in (report.get("spawned") or [])],
                    "rankings": rankings,
                    "spawned": report.get("spawned") or [],
                    "elites": report.get("elites") or [],
                    "ga_skipped": bool(g.get("skipped") or report.get("skipped")),
                    "ga_reason": report.get("reason"),
                    "best_fitness": g.get("best_fitness"),
                    "created_at": g.get("created_at"),
                }
    except Exception:
        pass

    merged = sorted(
        by_cycle.values(),
        key=lambda r: (r.get("cycle_number") is None, -(r.get("cycle_number") or 0)),
    )
    return JSONResponse(merged[:30])


@app.get("/api/regime")
def get_regime():
    """Current market regime, transition log, and per-regime performance."""
    try:
        from signals.regime_detector import get_detector
        status = get_detector().status()
    except Exception as e:
        status = {"error": str(e), "current": {}, "performance": {},
                  "transitions": []}
    # Prefer durable DB history when available
    try:
        events = db.get_regime_events(limit=25)
    except Exception:
        events = []
    if events:
        status["db_events"] = events
    return JSONResponse(status)


@app.get("/api/ga")
def get_ga():
    """Genetic Algorithm status: last cycle snapshot, fitness curve, recent gens,
    and the shadow elite gene bank used as future parents.
    """
    status = db.get_ga_status()
    gens = db.get_ga_history(limit=15)
    # Compact generation rows for the UI (full report available on demand)
    compact = []
    for g in gens:
        report = g.get("report") or {}
        compact.append({
            "cycle_number": g.get("cycle_number"),
            "best_fitness": g.get("best_fitness"),
            "mean_fitness": g.get("mean_fitness"),
            "n_elites": g.get("n_elites"),
            "n_replaced": g.get("n_replaced"),
            "n_spawned": g.get("n_spawned"),
            "skipped": bool(g.get("skipped")),
            "created_at": g.get("created_at"),
            "elites": report.get("elites") or [],
            "replaced": report.get("replaced") or [],
            "spawned": [
                {
                    "name": s.get("name"),
                    "parents": s.get("parents"),
                    "operator": s.get("operator"),
                    "lineage": s.get("lineage"),
                    "strategy_type": s.get("strategy_type"),
                }
                for s in (report.get("spawned") or [])
            ],
            "individuals": [
                {
                    "name": i.get("name"),
                    "fitness": i.get("fitness"),
                    "components": i.get("components"),
                    "status": i.get("status"),
                    "elite": i.get("elite"),
                    "pnl": i.get("pnl"),
                    "win_rate": i.get("win_rate"),
                    "trades": i.get("trades"),
                    "strategy_type": i.get("strategy_type"),
                    "lineage": i.get("lineage"),
                }
                for i in (report.get("individuals") or [])
            ],
        })

    # Shadow gene bank — elites deposited each cycle (newest last in storage).
    gene_bank_entries = []
    gene_bank_max = int(getattr(config, "GA_GENE_BANK_SIZE", 20))
    try:
        from evolution.gene_bank import load_bank, _max_size
        gene_bank_entries = load_bank()
        gene_bank_max = _max_size()
    except Exception:
        gene_bank_entries = []

    return JSONResponse({
        "status": status,
        "generations": compact,
        "gene_bank": {
            "entries": list(reversed(gene_bank_entries)),  # newest first for UI
            "count": len(gene_bank_entries),
            "max_size": gene_bank_max,
        },
        "config": {
            "elite_count": getattr(config, "GA_ELITE_COUNT", 1),
            "mutation_rate": getattr(config, "GA_MUTATION_RATE", 0.2),
            "mutation_sigma": getattr(config, "GA_MUTATION_SIGMA", 0.12),
            "tournament_k": getattr(config, "GA_TOURNAMENT_K", 3),
            "interval_hours": getattr(config, "EVOLUTION_INTERVAL_HOURS", 2),
            "window_hours": getattr(config, "EVOLUTION_WINDOW_HOURS", 24),
            "perf_trigger_enabled": getattr(config, "GA_PERF_TRIGGER_ENABLED", True),
            "perf_trigger_pnl": getattr(config, "GA_PERF_TRIGGER_PNL", -25.0),
            "fitness_weights": getattr(config, "GA_FITNESS_WEIGHTS", {}),
            "gene_bank_size": gene_bank_max,
            "type_alloc_enabled": getattr(config, "GA_TYPE_ALLOC_ENABLED", True),
            "backtest_gate_enabled": getattr(config, "GA_BACKTEST_GATE_ENABLED", True),
        },
    })


@app.get("/api/trades")
def get_trades(bot: str = None, limit: int = 50):
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
def get_copytrading():
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
def get_earnings():
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
def get_learning():
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
def credentials_status_endpoint():
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

    Body: {"which": "polymarket" | "all"} (default "all").
    Returns key-by-key results without persisting anything.
    """
    try:
        body = await request.json()
    except Exception:
        body = {}
    which = (body or {}).get("which", "all")
    results = {}

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
