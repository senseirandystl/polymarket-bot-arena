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
    # per market. Prices current + the next card; both YES and NO are set.
    polymarket_markets.price_markets([current, upcoming[0] if upcoming else None])

    def _shape(m):
        if not m:
            return None
        tr = m.get("time_remaining_seconds")
        yes = m.get("current_price")
        no = m.get("no_price")
        if no is None and yes is not None:
            no = round(1.0 - yes, 4)
        shaped = {
            "id": m.get("id"),
            "question": m.get("question"),
            "current_price": yes,                     # YES/Up (0-1)
            "no_price": no,                           # NO/Down (0-1), real mid
            "resolves_at": m.get("resolves_at"),
            "time_remaining_seconds": tr,
            "is_current_window": tr is not None and 0 < tr <= 300,
            "url": None,
        }
        with db.get_conn() as conn:
            rows = conn.execute(
                "SELECT bot_name, side, amount, shares_bought, entry_price, "
                "outcome, pnl, created_at "
                "FROM trades WHERE market_id=? ORDER BY created_at ASC",
                (shaped["id"],),
            ).fetchall()
        shaped["trades"] = [dict(r) for r in rows]
        return shaped

    cur_s = _shape(current)
    upcoming_s = [_shape(m) for m in upcoming]
    return JSONResponse({
        "current": cur_s,
        "next": upcoming_s[0] if upcoming_s else None,
        "upcoming_count": len(upcoming_s),
        "upcoming": upcoming_s,
    })


@app.get("/api/price/{condition_id}")
def get_price(condition_id: str):
    """Fresh YES/NO prices for one market (fast poll for the market cards)."""
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
    history = db.get_evolution_history(limit=20)
    for h in history:
        for key in ("survivors", "replaced", "new_bots", "rankings"):
            if isinstance(h.get(key), str):
                h[key] = json.loads(h[key])
    return JSONResponse(history)


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
