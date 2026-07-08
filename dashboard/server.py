"""FastAPI dashboard backend for the Bot Arena."""

import json
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
import db
import learning

security = HTTPBasic()

DASHBOARD_USER = "admin"
DASHBOARD_PASS = "Thor"


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
        try:
            import hmac as _hmac
            import hashlib as _hashlib
            import base64 as _base64
            import json as _json
            import requests as _req
            from pathlib import Path as _Path
            with open(_Path.home() / ".config/polymarket/credentials.json") as f:
                creds = _json.load(f)
            api_key = creds["api_key"]
            api_secret = creds["api_secret"]
            api_passphrase = creds["api_passphrase"]
            signer_address = creds["signer_address"]
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
    return {
        "mode": config.get_current_mode(),
        "venue": config.get_venue(),
        "max_position": config.get_max_position(),
        "max_daily_loss_per_bot": config.get_max_daily_loss_per_bot(),
        "max_daily_loss_total": config.get_max_daily_loss_total(),
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


def _window_contains_now(question: str, now_utc) -> bool:
    """Return True if the question's ET time range contains the current ET time."""
    import re
    q = question.lower()
    match = re.search(r'(\d{1,2}):(\d{2})\s*(am|pm)\s*[-–]\s*(\d{1,2}):(\d{2})\s*(am|pm)', q)
    if not match:
        return False
    h1, m1, ap1 = int(match.group(1)), int(match.group(2)), match.group(3)
    h2, m2, ap2 = int(match.group(4)), int(match.group(5)), match.group(6)
    if ap1 == 'pm' and h1 != 12: h1 += 12
    if ap1 == 'am' and h1 == 12: h1 = 0
    if ap2 == 'pm' and h2 != 12: h2 += 12
    if ap2 == 'am' and h2 == 12: h2 = 0
    start_min, end_min = h1 * 60 + m1, h2 * 60 + m2
    now_et = _to_et(now_utc)
    now_min = now_et.hour * 60 + now_et.minute
    if end_min > start_min:
        return start_min <= now_min < end_min
    else:  # crosses midnight ET
        return now_min >= start_min or now_min < end_min


@app.get("/api/markets")
async def get_markets():
    """Get active BTC fast markets as {current, upcoming_count, upcoming}.

    Queries two sources: SDK (upcoming tagged markets) + public API (live markets).
    The SDK endpoint drops live markets from its results once they enter their window.
    """
    import requests as req
    from datetime import datetime, timezone
    try:
        api_key = json.load(open(config.SIMMER_API_KEY_PATH))["api_key"]
        headers = {"Authorization": f"Bearer {api_key}"}

        seen_ids: set = set()
        markets_list = []

        # Source 1: SDK upcoming markets
        r1 = req.get(
            f"{config.SIMMER_BASE_URL}/api/sdk/markets",
            headers=headers,
            params={"limit": 50, "tags": "fast-5m"},
            timeout=10,
        )
        if r1.status_code == 200:
            d = r1.json()
            for m in (d if isinstance(d, list) else d.get("markets", [])):
                mid = m.get("id")
                if mid and mid not in seen_ids:
                    seen_ids.add(mid)
                    markets_list.append(m)

        # Source 2: public endpoint for currently-live markets
        r2 = req.get(
            f"{config.SIMMER_BASE_URL}/api/markets",
            headers=headers,
            params={"limit": 20},
            timeout=10,
        )
        if r2.status_code == 200:
            d = r2.json()
            for m in (d if isinstance(d, list) else d.get("markets", [])):
                mid = m.get("id")
                if mid and mid not in seen_ids:
                    seen_ids.add(mid)
                    markets_list.append(m)

        now_utc = datetime.now(timezone.utc)
        btc_markets = []
        for m in markets_list:
            q = m.get("question", "").lower()
            tags = m.get("tags") or []
            is_btc_updown = (
                ("bitcoin" in q or "btc" in q)
                and ("up or down" in q or "up/down" in q)
            ) or ("fast-5m" in tags and ("bitcoin" in q or "btc" in q))
            if not is_btc_updown:
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
                except Exception:
                    pass

            if time_remaining is not None and time_remaining < 0:
                continue

            btc_markets.append({
                "id": m.get("id"),
                "question": m.get("question"),
                "current_price": m.get("current_price"),
                "resolves_at": m.get("resolves_at"),
                "time_remaining_seconds": time_remaining,
                "is_current_window": (
                    (time_remaining is not None and 0 < time_remaining <= 300)
                    or _window_contains_now(m.get("question", ""), now_utc)
                ),
                "url": m.get("url"),
            })

        btc_markets.sort(key=lambda x: x.get("time_remaining_seconds") or 999999)

        # Priority 1: market whose question window contains now
        current = next((m for m in btc_markets if m["is_current_window"]), None)
        # Priority 2: soonest market closing within 20 min
        if not current:
            soon = [m for m in btc_markets if (m.get("time_remaining_seconds") or 999999) <= 1200]
            current = soon[0] if soon else None

        upcoming = [m for m in btc_markets if m is not current]
        return JSONResponse({
            "current": current,
            "upcoming_count": len(upcoming),
            "upcoming": upcoming,
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


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
            try:
                with open(config.POLYMARKET_KEY_PATH) as f:
                    pk_creds = json.load(f)
                trading_key_address = pk_creds.get("signer_address")
            except Exception:
                pass

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
        # Show trades with real P&L first, then pending. Skip phantom pnl=0 resolved trades.
        rows = conn.execute(
            """SELECT * FROM trades
               WHERE NOT (outcome IS NOT NULL AND (pnl IS NULL OR pnl = 0))
               ORDER BY
                   CASE WHEN outcome IS NOT NULL THEN 0 ELSE 1 END,
                   resolved_at DESC, created_at DESC
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
        daily = conn.execute("""
            SELECT date(created_at) as day, COALESCE(SUM(pnl), 0) as pnl,
                   COUNT(*) as trades,
                   SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins
            FROM trades WHERE outcome IN ('win', 'loss')
            GROUP BY date(created_at) ORDER BY day DESC LIMIT 30
        """).fetchall()

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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.DASHBOARD_HOST, port=config.DASHBOARD_PORT)
