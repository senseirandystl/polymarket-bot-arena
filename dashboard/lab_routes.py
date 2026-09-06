"""Strategy Lab pipeline dashboard routes (Phase 1+2).

    from dashboard.lab_routes import register_lab_routes
    register_lab_routes(app, verify_auth=verify_auth)
"""

from __future__ import annotations

from fastapi import Depends, Request
from fastapi.responses import JSONResponse


def register_lab_routes(app, *, verify_auth):
    @app.get("/api/lab/pipeline/status")
    def get_lab_status(_auth: str = Depends(verify_auth)):
        try:
            from signals.strategy_pipeline.api import status
            return JSONResponse(status())
        except Exception as e:
            return JSONResponse(
                {"error": str(e), "hypotheses": [], "pipeline_counts": {}},
            )

    @app.get("/api/lab/pipeline/settings")
    def get_lab_settings(_auth: str = Depends(verify_auth)):
        from signals.strategy_pipeline.api import settings
        return JSONResponse(settings(None))

    @app.post("/api/lab/pipeline/settings")
    async def post_lab_settings(request: Request, _auth: str = Depends(verify_auth)):
        from signals.strategy_pipeline.api import settings

        body = await request.json()
        if not isinstance(body, dict):
            body = {}
        return JSONResponse(settings(body))

    @app.post("/api/lab/pipeline/tick")
    def post_lab_tick(_auth: str = Depends(verify_auth)):
        from signals.strategy_pipeline.api import tick

        report = tick()
        code = 200 if report.get("ok") else 409
        return JSONResponse(report, status_code=code)

    @app.post("/api/lab/pipeline/promote")
    async def post_lab_promote(request: Request, _auth: str = Depends(verify_auth)):
        from signals.strategy_pipeline.api import promote

        body = await request.json()
        spec_id = str((body or {}).get("spec_id") or "").strip()
        if not spec_id:
            return JSONResponse(
                {"ok": False, "reason": "missing_spec_id"}, status_code=400,
            )
        result = promote(spec_id)
        code = 200 if result.get("ok") else 409
        return JSONResponse(result, status_code=code)

    @app.post("/api/lab/pipeline/approve_paper")
    async def post_lab_approve_paper(request: Request, _auth: str = Depends(verify_auth)):
        from signals.strategy_pipeline.api import approve_paper

        body = await request.json()
        spec_id = str((body or {}).get("spec_id") or "").strip()
        if not spec_id:
            return JSONResponse(
                {"ok": False, "reason": "missing_spec_id"}, status_code=400,
            )
        result = approve_paper(spec_id)
        code = 200 if result.get("ok") else 409
        return JSONResponse(result, status_code=code)

    @app.post("/api/lab/pipeline/reject")
    async def post_lab_reject(request: Request, _auth: str = Depends(verify_auth)):
        from signals.strategy_pipeline.api import reject

        body = await request.json()
        spec_id = str((body or {}).get("spec_id") or "").strip()
        reason = str((body or {}).get("reason") or "operator_deny").strip()
        if not spec_id:
            return JSONResponse(
                {"ok": False, "reason": "missing_spec_id"}, status_code=400,
            )
        result = reject(spec_id, reason=reason)
        code = 200 if result.get("ok") else 409
        return JSONResponse(result, status_code=code)
