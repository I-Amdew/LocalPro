import asyncio
import json
from pathlib import Path
from typing import Dict

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .config import AppSettings, CONFIG_PATH, load_settings, save_settings
from .db import Database
from .llm import LMStudioClient
from .orchestrator import EventBus, new_run_id, run_question
from .schemas import StartRunRequest
from .tavily import TavilyClient


settings = load_settings()
db = Database(settings.database_path)
lm_client = LMStudioClient(settings.lm_studio_base_url)
tavily_client = TavilyClient(settings.tavily_api_key)
bus = EventBus(db)

app = FastAPI(title="LocalPro Chat Orchestrator")
static_dir = Path(__file__).parent / "web" / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")

model_check: Dict[str, str] = {}


@app.on_event("startup")
async def startup_event():
    await db.init()
    await db.save_config(settings.model_dump())
    global model_check
    try:
        resp = await lm_client.list_models()
        ids = [m.get("id") for m in resp.get("data", [])]
        missing = []
        for key, val in {
            "orchestrator": settings.model_orch,
            "worker": settings.model_qwen8,
            "router": settings.model_qwen4,
        }.items():
            if val not in ids:
                missing.append(f"{key}:{val}")
        model_check = {"ok": not missing, "missing": missing, "available": ids}
    except Exception as exc:
        model_check = {"ok": False, "error": str(exc)}


@app.on_event("shutdown")
async def shutdown_event():
    await lm_client.close()
    await tavily_client.close()


def sse_format(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


@app.get("/")
async def index():
    return FileResponse(static_dir / "index.html")


@app.get("/settings")
async def get_settings():
    return {"settings": settings.model_dump(), "model_check": model_check}


@app.post("/settings")
async def update_settings(request: Request):
    body = await request.json()
    new_settings = AppSettings(**{**settings.model_dump(), **body})
    save_settings(new_settings)
    await db.save_config(new_settings.model_dump())
    global settings
    settings = new_settings
    lm_client.base_url = settings.lm_studio_base_url
    tavily_client.api_key = settings.tavily_api_key
    return {"ok": True}


@app.post("/api/run")
async def start_run(payload: StartRunRequest):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question is required.")
    run_id = new_run_id()
    models = {
        "orch": settings.model_orch,
        "worker": settings.model_qwen8,
        "router": settings.model_qwen4,
        "verifier": settings.model_qwen8,
    }
    asyncio.create_task(
        run_question(
            run_id=run_id,
            question=payload.question,
            decision_mode=payload.reasoning_mode,
            manual_level=payload.manual_level,
            search_depth_mode=payload.search_depth_mode,
            max_results_override=payload.max_results or 0,
            strict_mode=payload.strict_mode,
            db=db,
            bus=bus,
            lm_client=lm_client,
            tavily=tavily_client,
            settings_models=models,
        )
    )
    return {"run_id": run_id}


@app.get("/api/run/{run_id}")
async def get_run(run_id: str):
    run = await db.get_run_summary(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@app.get("/api/run/{run_id}/artifacts")
async def get_artifacts(run_id: str):
    run = await db.get_run_summary(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    sources = await db.get_sources(run_id)
    claims = await db.get_claims(run_id)
    draft = await db.get_latest_draft(run_id)
    verifier = await db.get_verifier_report(run_id)
    return {"run": run, "sources": sources, "claims": claims, "draft": draft, "verifier": verifier}


@app.get("/runs/{run_id}/events")
async def stream_events(run_id: str):
    # Preload past events then stream new ones
    async def event_generator():
        queue = await bus.subscribe(run_id)
        try:
            past = await db.list_events(run_id)
            for ev in past:
                yield sse_format(ev)
            while True:
                ev = await queue.get()
                yield sse_format(ev)
        finally:
            await bus.unsubscribe(run_id, queue)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=settings.port, reload=True)
