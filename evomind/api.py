"""FastAPI server for EvoMind — REST API + WebSocket for real-time dashboard.

Endpoints:
    POST /api/analyze           Submit a dataset + task, returns a job ID
    GET  /api/jobs/{id}         Poll job status/results
    GET  /api/memory            Browse strategy memory
    GET  /api/memory/tree/{sig} Get the evolution tree for a task signature
    GET  /api/strategies/{sig}  Best strategies for a task type
    WS   /ws/jobs/{id}          Stream iteration updates in real-time
    GET  /                      Serve the React dashboard

Production env vars:
    ALLOWED_ORIGINS  Comma-separated allowed CORS origins, e.g. https://myapp.up.railway.app
                     Defaults to * (allow all) when not set.
    APP_PASSWORD     If set, all API routes require HTTP Basic Auth with this password.
                     Username can be anything. Leave unset to disable auth.
    GROQ_API_KEY     Groq LLM key (primary provider)
    HF_TOKEN         HuggingFace token (fallback provider)
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import sqlite3
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles

from evomind.config import load_config
from evomind.llm import get_llm_client
from evomind.loader import load_dataset, validate_dataset
from evomind.memory import StrategyMemory
from evomind.nodes import AVAILABLE_STEPS, make_task_signature, summarize_dataset
from evomind.telemetry import RunTracker

# ---------------------------------------------------------------------------
# CORS — reads from ALLOWED_ORIGINS env var in production
# ---------------------------------------------------------------------------

_raw_origins = os.environ.get("ALLOWED_ORIGINS", "").strip()
_ALLOWED_ORIGINS: list[str] = (
    [o.strip() for o in _raw_origins.split(",") if o.strip()]
    if _raw_origins else ["*"]
)

app = FastAPI(title="EvoMind API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Optional password auth — set APP_PASSWORD env var to enable
# ---------------------------------------------------------------------------

_APP_PASSWORD = os.environ.get("APP_PASSWORD", "").strip()
_http_security = HTTPBasic(auto_error=False)


def _check_auth(credentials: HTTPBasicCredentials | None = Depends(_http_security)) -> None:
    """If APP_PASSWORD is set, require HTTP Basic Auth on every request."""
    if not _APP_PASSWORD:
        return  # auth disabled
    ok = (
        credentials is not None
        and secrets.compare_digest(
            credentials.password.encode("utf-8"),
            _APP_PASSWORD.encode("utf-8"),
        )
    )
    if not ok:
        raise HTTPException(
            status_code=401,
            detail="Incorrect password.",
            headers={"WWW-Authenticate": 'Basic realm="EvoMind"'},
        )


# ---------------------------------------------------------------------------
# Persistent job store (SQLite) — jobs survive server restarts
# ---------------------------------------------------------------------------

_JOBS_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "jobs.db"


class _JobDB:
    """Thin SQLite wrapper that persists job dicts across restarts."""

    def __init__(self, path: Path = _JOBS_DB_PATH) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id   TEXT PRIMARY KEY,
                    job_json TEXT NOT NULL,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def load_all(self) -> dict[str, dict[str, Any]]:
        with sqlite3.connect(self.path) as conn:
            rows = conn.execute("SELECT job_id, job_json FROM jobs ORDER BY rowid").fetchall()
        result = {}
        for job_id, job_json in rows:
            try:
                result[job_id] = json.loads(job_json)
            except Exception:
                pass
        return result

    def save(self, job_id: str, job: dict[str, Any]) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                INSERT INTO jobs (job_id, job_json, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(job_id) DO UPDATE SET
                    job_json   = excluded.job_json,
                    updated_at = excluded.updated_at
                """,
                (job_id, json.dumps(job, default=str)),
            )
            conn.commit()


_job_db = _JobDB()

# Load jobs from disk on startup (persists across restarts)
_jobs: dict[str, dict[str, Any]] = _job_db.load_all()
_job_subscribers: dict[str, list[WebSocket]] = {}

# Global memory instance
_memory = StrategyMemory()


# ---------------------------------------------------------------------------
# Static files (React dashboard)
# ---------------------------------------------------------------------------

DASHBOARD_DIR = Path(__file__).resolve().parent / "dashboard" / "dist"

if DASHBOARD_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(DASHBOARD_DIR)), name="static")


@app.get("/")
async def serve_dashboard():
    """Serve the React dashboard."""
    index = DASHBOARD_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return HTMLResponse(
        "<h1>EvoMind API</h1>"
        "<p>Dashboard not built yet. Run <code>cd evomind/dashboard && npm run build</code></p>"
        "<p>API docs: <a href='/docs'>/docs</a></p>"
    )


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health():
    return {"status": "ok", "available_steps": AVAILABLE_STEPS}


@app.post("/api/analyze", dependencies=[Depends(_check_auth)])
async def submit_analysis(
    file: UploadFile = File(...),
    task: str = Form(...),
    iterations: int = Form(5),
    threshold: float = Form(0.8),
):
    """Submit a dataset for analysis. Returns a job ID to poll/subscribe."""
    job_id = str(uuid.uuid4())[:8]

    # Save uploaded file
    suffix = Path(file.filename or "data.csv").suffix
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    content = await file.read()
    tmp.write(content)
    tmp.close()

    _jobs[job_id] = {
        "id": job_id,
        "status": "queued",
        "task": task,
        "filename": file.filename,
        "iterations": iterations,
        "threshold": threshold,
        "data_path": tmp.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "history": [],
        "result": None,
    }
    _job_db.save(job_id, _jobs[job_id])  # persist immediately

    # Launch in background
    asyncio.create_task(_run_job(job_id))

    return {"job_id": job_id, "status": "queued"}


@app.get("/api/jobs/{job_id}", dependencies=[Depends(_check_auth)])
async def get_job(job_id: str):
    """Get job status and results."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job


@app.get("/api/jobs", dependencies=[Depends(_check_auth)])
async def list_jobs():
    """List all jobs."""
    return list(_jobs.values())


@app.get("/api/memory", dependencies=[Depends(_check_auth)])
async def get_memory():
    """Browse all strategies in memory."""
    return {
        "global_best": _memory.get_global_best(top_k=20),
    }


@app.get("/api/memory/tree/{task_signature}", dependencies=[Depends(_check_auth)])
async def get_evolution_tree(task_signature: str):
    """Get the evolution tree for a task signature."""
    tree = _memory.get_evolution_tree(task_signature)
    return {"task_signature": task_signature, "tree": tree}


@app.get("/api/strategies/{task_signature}", dependencies=[Depends(_check_auth)])
async def get_strategies(task_signature: str, top_k: int = 5):
    """Get best strategies for a task type."""
    strategies = _memory.get_best_strategies(task_signature, top_k=top_k)
    return {"task_signature": task_signature, "strategies": strategies}


@app.get("/api/available-steps", dependencies=[Depends(_check_auth)])
async def available_steps():
    """List all available analysis steps."""
    return {"steps": AVAILABLE_STEPS}


# ---------------------------------------------------------------------------
# WebSocket for real-time updates
# ---------------------------------------------------------------------------

@app.websocket("/ws/jobs/{job_id}")
async def ws_job_updates(websocket: WebSocket, job_id: str):
    """Stream iteration updates in real-time."""
    await websocket.accept()
    if job_id not in _job_subscribers:
        _job_subscribers[job_id] = []
    _job_subscribers[job_id].append(websocket)

    try:
        # Send current state
        job = _jobs.get(job_id)
        if job:
            await websocket.send_json(job)

        # Keep connection open
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        if job_id in _job_subscribers:
            _job_subscribers[job_id].remove(websocket)


async def _notify_subscribers(job_id: str, data: dict[str, Any]) -> None:
    """Push update to all WebSocket subscribers for a job."""
    subscribers = _job_subscribers.get(job_id, [])
    dead: list[WebSocket] = []
    for ws in subscribers:
        try:
            await ws.send_json(data)
        except Exception:
            dead.append(ws)
    for ws in dead:
        subscribers.remove(ws)


# ---------------------------------------------------------------------------
# Background job runner
# ---------------------------------------------------------------------------

async def _run_job(job_id: str) -> None:
    """Run the EvoMind analysis in the background, streaming updates."""
    from evomind.graph import build_graph
    from evomind.nodes import evaluator_node, executor_node, planner_node, reflector_node

    job = _jobs[job_id]
    job["status"] = "running"
    _job_db.save(job_id, job)
    await _notify_subscribers(job_id, job)

    try:
        df = load_dataset(job["data_path"])
        validate_dataset(df)

        dataset_summary = summarize_dataset(df)
        task_signature = make_task_signature(job["task"], dataset_summary)
        llm = get_llm_client()

        state = {
            "task_description": job["task"],
            "dataset_path": job["data_path"],
            "dataset_summary": dataset_summary,
            "task_signature": task_signature,
            "history": [],
            "iteration": 0,
            "max_iterations": job["iterations"],
            "score_threshold": job["threshold"],
            "best_score": -1.0,
            "should_continue": True,
        }

        # Manual loop (instead of graph.invoke) so we can stream each iteration
        while state.get("should_continue", True):
            # Planner
            planner_out = planner_node(state, llm=llm, memory=_memory)
            state.update(planner_out)

            # Executor
            executor_out = executor_node(state, df=df)
            state.update(executor_out)

            # Evaluator
            evaluator_out = evaluator_node(state, llm=llm)
            state.update(evaluator_out)

            # Reflector
            reflector_out = reflector_node(state, memory=_memory)
            state.update(reflector_out)

            # Stream the update
            iteration_data = {
                "iteration": state["iteration"] - 1,
                "strategy": state["strategy"],
                "evaluation": state["evaluation"],
                "best_score": state["best_score"],
                "best_strategy": state.get("best_strategy"),
            }
            job["history"].append(iteration_data)
            job["status"] = "running"
            _job_db.save(job_id, job)
            await _notify_subscribers(job_id, {**job, "latest_iteration": iteration_data})

            # Small delay to not overwhelm
            await asyncio.sleep(0.1)

        job["status"] = "completed"
        job["result"] = {
            "best_strategy": state.get("best_strategy"),
            "best_score": state.get("best_score"),
            "stop_reason": state.get("stop_reason"),
            "total_iterations": len(state.get("history", [])),
            "task_signature": task_signature,
            "dataset_summary": dataset_summary,
        }

    except Exception as exc:
        job["status"] = "failed"
        job["error"] = str(exc)

    _job_db.save(job_id, job)
    await _notify_subscribers(job_id, job)

    # Cleanup temp file
    try:
        os.unlink(job["data_path"])
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_server(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
