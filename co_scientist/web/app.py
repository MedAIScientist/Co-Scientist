"""FastAPI web UI for the co-scientist.

One process per host: launching `co-scientist serve` runs both the API + UI
and the worker pool — the queue is DB-backed so CLI `co-scientist run` in a
separate terminal feeds tasks to whatever Supervisor is currently active.

The UI is server-side Jinja2 + htmx for partial updates + SSE for live events.
No JS build step. Pico.css for default styling.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging as stdlib_logging
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sse_starlette.sse import EventSourceResponse

from .. import ids
from ..config import Config, load_config
from ..logging import get_logger
from ..models import SystemFeedback
from ..orchestrator.events import GLOBAL_BUS
from ..storage import db as db_mod
from ..storage.repos import events as events_repo
from ..storage.repos import feedback as fb_repo
from ..storage.repos import hypotheses as hyp_repo
from ..storage.repos import reviews as rev_repo
from ..storage.repos import sessions as sess_repo
from ..storage.repos import transcripts as tx_repo
from .sanitize import render_markdown

log = get_logger("web")
HERE = Path(__file__).parent
TEMPLATES = Jinja2Templates(directory=HERE / "templates")


def create_app(cfg: Config | None = None) -> FastAPI:
    cfg = cfg or load_config()
    app = FastAPI(title="AI Co-Scientist")
    app.state.cfg = cfg
    app.state.background_runs: dict[str, asyncio.Task] = {}

    # Static
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")

    # ----------------------------- pages ----------------------------- #

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        rows = await _list_sessions(cfg)
        return TEMPLATES.TemplateResponse(
            request, "index.html", {"sessions": rows}
        )

    @app.get("/sessions/new", response_class=HTMLResponse)
    async def new_session_form(request: Request) -> HTMLResponse:
        return TEMPLATES.TemplateResponse(
            request, "new_session.html", {"default_budget": cfg.run.budget_usd}
        )

    @app.post("/sessions/new")
    async def create_session(
        request: Request,
        background_tasks: BackgroundTasks,
        goal: str = Form(...),
        preferences: str = Form(""),
        budget_usd: float = Form(cfg.run.budget_usd),
        n_initial: int = Form(3),
        wall_clock_seconds: int = Form(cfg.run.wall_clock_seconds),
    ) -> RedirectResponse:
        from ..agents.supervisor import Supervisor

        # Hand the Supervisor a fresh Config copy so per-session knobs don't leak.
        sup_cfg = cfg.model_copy(deep=True)
        sup_cfg.run.budget_usd = budget_usd
        sup_cfg.run.wall_clock_seconds = wall_clock_seconds
        sup = Supervisor(sup_cfg)

        async def _run() -> None:
            try:
                await sup.run_session(
                    goal=goal, preferences_text=preferences or None,
                    n_initial=n_initial, wall_clock_seconds=wall_clock_seconds,
                )
            except Exception:
                log.exception("background_run_failed")

        task = asyncio.create_task(_run())
        # No durable session id at this point — give the run a chance to insert
        # the row, then redirect to /. The user can find it in the listing.
        _ = task
        return RedirectResponse(url="/", status_code=303)

    @app.get("/sessions/{session_id}", response_class=HTMLResponse)
    async def session_detail(request: Request, session_id: str) -> HTMLResponse:
        conn = await db_mod.connect(cfg)
        try:
            session = await sess_repo.fetch(conn, session_id)
            if session is None:
                raise HTTPException(status_code=404, detail="session not found")
            hyps = await hyp_repo.list_for_session(conn, session_id)
            recent_matches = await _recent_matches(conn, session_id, limit=20)
            usage = await tx_repo.usage_summary(conn, session_id)
            return TEMPLATES.TemplateResponse(
                request,
                "session_detail.html",
                {
                    "session": session,
                    "hypotheses": sorted(hyps, key=lambda h: -(h.elo or 0)),
                    "recent_matches": recent_matches,
                    "usage": usage,
                },
            )
        finally:
            await conn.close()

    @app.get("/sessions/{session_id}/hypotheses/{hid}", response_class=HTMLResponse)
    async def hypothesis_detail(request: Request, session_id: str, hid: str) -> HTMLResponse:
        conn = await db_mod.connect(cfg)
        try:
            h = await hyp_repo.fetch(conn, hid)
            session = await sess_repo.fetch(conn, session_id)
            if h is None or session is None:
                raise HTTPException(status_code=404, detail="not found")
            reviews = await rev_repo.list_for_hypothesis(conn, hid)
            return TEMPLATES.TemplateResponse(
                request,
                "hypothesis_detail.html",
                {
                    "session": session,
                    "h": h,
                    "reviews": reviews,
                    "full_text_html": render_markdown(h.full_text or ""),
                },
            )
        finally:
            await conn.close()

    @app.get("/sessions/{session_id}/overview", response_class=HTMLResponse)
    async def session_overview(request: Request, session_id: str) -> HTMLResponse:
        conn = await db_mod.connect(cfg)
        try:
            session = await sess_repo.fetch(conn, session_id)
            if session is None or not session.final_overview:
                raise HTTPException(
                    status_code=404, detail="no final overview yet for this session"
                )
            # `final_overview` is written by the supervisor under
            # `data_dir/artifacts/...` but is stored as a string in the DB.
            # Resolve and confirm the path is still inside `data_dir` so a
            # tampered row can't read arbitrary files.
            base = cfg.data_dir.resolve()
            try:
                path = (cfg.data_dir / session.final_overview).resolve()
                path.relative_to(base)
            except (ValueError, OSError) as e:
                log.error("overview_path_escape", session=session_id, err=str(e))
                raise HTTPException(status_code=404, detail="overview unavailable") from e
            if not path.is_file():
                raise HTTPException(status_code=404, detail="overview missing on disk")
            overview_md = path.read_text(encoding="utf-8")
            return TEMPLATES.TemplateResponse(
                request,
                "overview.html",
                {
                    "session": session,
                    "overview_html": render_markdown(overview_md),
                    "overview_md": overview_md,
                },
            )
        finally:
            await conn.close()

    # ----------------------------- API + SSE ----------------------------- #

    @app.get("/api/sessions/{session_id}/metrics")
    async def api_metrics(session_id: str) -> JSONResponse:
        from ..obs.metrics import session_metrics_cached, to_dict

        conn = await db_mod.connect(cfg)
        try:
            m = await session_metrics_cached(conn, session_id)
            return JSONResponse(to_dict(m))
        finally:
            await conn.close()

    @app.get("/api/sessions/{session_id}")
    async def api_session(session_id: str) -> JSONResponse:
        conn = await db_mod.connect(cfg)
        try:
            s = await sess_repo.fetch(conn, session_id)
            if s is None:
                raise HTTPException(status_code=404)
            return JSONResponse(s.model_dump(mode="json"))
        finally:
            await conn.close()

    @app.get("/api/sessions/{session_id}/events")
    async def api_events(session_id: str) -> EventSourceResponse:
        async def _stream() -> AsyncIterator[dict[str, Any]]:
            # Replay last 25 events from DB so refreshes don't go blank.
            conn = await db_mod.connect(cfg)
            try:
                history = await events_repo.recent(conn, session_id, limit=25)
            finally:
                await conn.close()
            for ev in reversed(history):
                yield {
                    "event": ev["event"],
                    "data": json.dumps({"payload": ev["payload"], "ts": ev["ts"]}),
                }
            async with contextlib.aclosing(GLOBAL_BUS.subscribe(session_id)) as gen:
                async for ev in gen:
                    yield {
                        "event": ev.name,
                        "data": ev.to_json(),
                    }

        return EventSourceResponse(_stream())

    @app.post("/api/sessions/{session_id}/pause")
    async def api_pause(session_id: str) -> JSONResponse:
        conn = await db_mod.connect(cfg)
        try:
            await sess_repo.set_status(conn, session_id, "paused")
            await GLOBAL_BUS.publish(session_id, "session_paused", {})
            return JSONResponse({"ok": True})
        finally:
            await conn.close()

    @app.post("/api/sessions/{session_id}/resume")
    async def api_resume(session_id: str) -> JSONResponse:
        conn = await db_mod.connect(cfg)
        try:
            await sess_repo.set_status(conn, session_id, "running")
            await GLOBAL_BUS.publish(session_id, "session_resumed", {})
            return JSONResponse({"ok": True})
        finally:
            await conn.close()

    @app.post("/api/sessions/{session_id}/abort")
    async def api_abort(session_id: str) -> JSONResponse:
        conn = await db_mod.connect(cfg)
        try:
            await sess_repo.set_status(conn, session_id, "aborted")
            await GLOBAL_BUS.publish(session_id, "session_aborted", {})
            return JSONResponse({"ok": True})
        finally:
            await conn.close()

    @app.post("/api/sessions/{session_id}/feedback")
    async def api_feedback(
        session_id: str,
        text: str = Form(...),
        kind: str = Form("directive"),
        target_id: str = Form(""),
    ) -> JSONResponse:
        conn = await db_mod.connect(cfg)
        try:
            fb = SystemFeedback(
                id=ids.feedback_id(), session_id=session_id,
                created_at=datetime.now(UTC),
                source="human", kind=kind,
                target_id=target_id or None, text=text, active=True,
            )
            await fb_repo.insert(conn, fb)
            if kind == "pin" and target_id:
                await hyp_repo.set_state(conn, target_id, "pinned")
            elif kind == "rejection" and target_id:
                await hyp_repo.set_state(conn, target_id, "rejected")
            await GLOBAL_BUS.publish(session_id, "human_feedback", {
                "kind": kind, "target_id": target_id or None, "text": text[:200],
            })
            return JSONResponse({"ok": True, "feedback_id": fb.id})
        finally:
            await conn.close()

    @app.get("/healthz")
    async def health() -> JSONResponse:
        return JSONResponse({"ok": True})

    # quiet uvicorn access spam during streaming
    stdlib_logging.getLogger("uvicorn.access").setLevel(stdlib_logging.WARNING)
    return app


# ----------------------------- helpers ----------------------------- #


async def _list_sessions(cfg: Config) -> list[dict[str, Any]]:
    conn = await db_mod.connect(cfg)
    try:
        async with conn.execute(
            """SELECT id, status, research_goal, created_at, updated_at,
                      budget_usd, budget_used_usd,
                      (SELECT COUNT(*) FROM hypotheses WHERE session_id = s.id) AS n_hyps,
                      (SELECT MAX(elo) FROM hypotheses WHERE session_id = s.id) AS top_elo
                 FROM sessions s
                 ORDER BY updated_at DESC LIMIT 50""",
        ) as cur:
            rows = await cur.fetchall()
    finally:
        await conn.close()
    return [dict(r) for r in rows]


async def _recent_matches(conn, session_id: str, *, limit: int) -> list[dict[str, Any]]:
    async with conn.execute(
        """SELECT id, hyp_a, hyp_b, mode, winner, elo_a_after, elo_b_after, created_at
              FROM tournament_matches
             WHERE session_id=?
             ORDER BY created_at DESC LIMIT ?""",
        (session_id, limit),
    ) as cur:
        rows = await cur.fetchall()
    return [dict(r) for r in rows]
