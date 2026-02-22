"""FastAPI dashboard backend for the Bot Arena."""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
import config
import db
import learning
from datetime import datetime

app = FastAPI(title="Polymarket Bot Arena Dashboard")

# ────────────────────────────────────────────────────────────────
# CSP Middleware to allow inline <script> for local development
# ────────────────────────────────────────────────────────────────
class CSPMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)
        # Allow inline scripts + unsafe-eval (needed for the current index.html)
        # For production you'd move JS to external file and tighten this
        csp = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; "
            "connect-src 'self';"
        )
        response.headers["Content-Security-Policy"] = csp
        return response

# Add the middleware
app.add_middleware(CSPMiddleware)

# Balance cache: slot_name -> {"balance": float, "fetched_at": float}
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


def get_bot_balance(slot_name, bot_keys):
    """Get cached or fresh balance for a bot slot."""
    now = time.time()
    cached = _balance_cache.get(slot_name)
    if cached and (now - cached["fetched_at"]) < BALANCE_CACHE_TTL:
        return cached["balance"]

    api_key = bot_keys.get(slot_name)
    if not api_key:
        return None

    balance = _fetch_slot_balance(api_key)
    _balance_cache[slot_name] = {"balance": balance, "fetched_at": now}
    return balance


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


@app.get("/api/markets")
async def get_markets(page: int = Query(1, ge=1), limit: int = Query(5, ge=1, le=20)):
    """Get active BTC 5-min markets with close times."""
    import requests as req
    try:
        api_key = json.load(open(config.SIMMER_API_KEY_PATH))["api_key"]
        headers = {"Authorization": f"Bearer {api_key}"}
        resp = req.get(
            f"{config.SIMMER_BASE_URL}/api/sdk/markets",
            headers=headers,
            params={"status": "active", "limit": 100},
            timeout=10,
        )
        data = resp.json()
        markets_list = data if isinstance(data, list) else data.get("markets", [])
        btc_markets = []
        for m in markets_list:
            q = m.get("question", "").lower()
            if "bitcoin" in q and "up or down" in q:
                btc_markets.append({
                    "id": m.get("id"),
                    "question": m.get("question"),
                    "current_price": m.get("current_price"),
                    "resolves_at": m.get("resolves_at"),
                    "url": m.get("url"),
                })
        
        # Sort by resolves_at (soonest first)
        def get_resolve_time(m):
            resolves_at = m.get("resolves_at")
            if resolves_at:
                resolves_at = resolves_at.replace("Z", "+00:00").replace(" ", "T")
                return datetime.fromisoformat(resolves_at)
            return datetime.max  # Push invalid to end
        
        btc_markets.sort(key=get_resolve_time)
        
        # Pagination
        total = len(btc_markets)
        start = (page - 1) * limit
        end = start + limit
        paginated = btc_markets[start:end]
        
        return {
            "markets": paginated,
            "total": total,
            "page": page,
            "pages": (total + limit - 1) // limit
        }
    except Exception as e:
        return {"error": str(e)}


# ... (the rest of the file remains exactly the same as before)
# Including get_overview, get_bots, get_evolution, get_trades, etc.