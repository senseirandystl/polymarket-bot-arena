"""Desk-cycle dashboard routes. Import from dashboard/server.py:

    from dashboard.desk_routes import register_desk_routes
    register_desk_routes(app, verify_auth=verify_auth)
"""

from __future__ import annotations

from fastapi import Depends, Request
from fastapi.responses import HTMLResponse, JSONResponse


def register_desk_routes(app, *, verify_auth):
    @app.get("/api/desk/floor")
    def get_desk_floor(_auth: str = Depends(verify_auth)):
        try:
            from desk.cycle import get_host
            snap = get_host().snapshot().as_dict()
        except Exception as e:
            snap = {"error": str(e), "roles": [], "hypotheses": [],
                    "pipeline_counts": {}}
        try:
            from desk.universe import phase_universe
            snap["universe"] = [s.as_dict() for s in phase_universe()]
        except Exception:
            snap["universe"] = []
        return JSONResponse(snap)

    @app.get("/api/desk/events")
    def get_desk_events(limit: int = 40, _auth: str = Depends(verify_auth)):
        from desk.store import HypothesisStore
        return JSONResponse(HypothesisStore().events(limit=limit))

    @app.post("/api/desk/tick")
    def post_desk_tick(_auth: str = Depends(verify_auth)):
        from desk.cycle import get_host
        return JSONResponse(get_host().tick())

    @app.post("/api/desk/settings")
    async def post_desk_settings(request: Request, _auth: str = Depends(verify_auth)):
        import config
        body = await request.json()
        if "factory_mode" in body:
            config.DESK_FACTORY_MODE = bool(body["factory_mode"])
        if "auto_live" in body:
            config.DESK_AUTO_LIVE = bool(body["auto_live"])
        if "universe_phase" in body:
            try:
                config.CRYPTO_UNIVERSE_PHASE = max(1, min(3, int(body["universe_phase"])))
            except (TypeError, ValueError):
                pass
        if "llm_provider" in body:
            val = str(body["llm_provider"] or "none").lower()
            if val in ("none", "ollama", "grok"):
                config.DESK_LLM_PROVIDER = val
        return JSONResponse({
            "factory_mode": bool(getattr(config, "DESK_FACTORY_MODE", False)),
            "auto_live": bool(getattr(config, "DESK_AUTO_LIVE", False)),
            "universe_phase": int(getattr(config, "CRYPTO_UNIVERSE_PHASE", 1)),
            "llm_provider": getattr(config, "DESK_LLM_PROVIDER", "none"),
        })

    @app.get("/floor", response_class=HTMLResponse)
    def get_floor_page(_auth: str = Depends(verify_auth)):
        from pathlib import Path
        html = Path(__file__).with_name("floor.html").read_text()
        return HTMLResponse(html)
